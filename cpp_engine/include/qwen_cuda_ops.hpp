#pragma once

#include <cstddef>
#include <cstdint>

namespace dsv4 {

// Qwen FP8 weights use E4M3 codes and 128x128 block scales. On Turing,
// checkpoint BF16 scales are converted to IEEE FP16 before device upload.
// These kernels decode each code at the point of use; no expanded copy of the
// weight matrix is allocated.
bool qwen_fp8_e4m3_fp16scale_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Decode-only fused gate/up projection. The two matvecs retain independent FP32
// accumulation, then write silu(gate) * up without materializing either input.
bool qwen_fp8_e4m3_fp16scale_swiglu_matvec_cuda(
    const float* d_x,
    const uint8_t* d_gate_weight,
    const uint16_t* d_gate_scale_fp16,
    const uint8_t* d_up_weight,
    const uint16_t* d_up_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Explicit experimental prefill path used by correctness/performance A/B.
// It decodes only a block-local K tile and preserves FP32 accumulation.
bool qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Compatibility/reference path for callers that still keep the scale in BF16
// bits. Qwen Turing loading should use the FP16-scale APIs above.
bool qwen_fp8_e4m3_bf16_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_bf16_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Qwen RMSNorm has a non-standard affine parameter convention: output is
// normalized * (1 + weight).  The gated variant deliberately has no +1.
bool qwen_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                           float* d_y, int rows, int cols, float eps,
                           void* stream = nullptr);
bool qwen_gated_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                                 const float* d_gate, float* d_y,
                                 int rows, int cols, float eps,
                                 void* stream = nullptr);
bool qwen_l2_norm_f32_cuda(const float* d_x, float* d_y, int rows, int cols,
                           float eps = 1.0e-6f, void* stream = nullptr);

// --- Dense FP16 weights (checkpoint BF16 materialized for Turing) -----------

// y[b, r] = sum_c x[b, c] * fp16(w[r, c]).
bool qwen_fp16_matmul_rows_cuda(const float* d_x, const uint16_t* d_w_fp16,
                                float* d_y, int batch, int rows, int cols,
                                int x_stride, int y_stride, int weight_stride,
                                void* stream = nullptr);

// Vocab-parallel embedding lookup. Tokens outside [row_start, row_start +
// row_count) produce a zero row so the caller can all-reduce the result.
bool qwen_embedding_fp16_gather_cuda(const uint16_t* d_table_fp16,
                                     const int* d_tokens, float* d_out,
                                     int count, int cols, int row_start,
                                     int row_count, void* stream = nullptr);

// RMSNorm variants whose affine parameter is stored as FP16 on device.
bool qwen_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                       float* d_y, int rows, int cols, float eps,
                                       void* stream = nullptr);
bool qwen_gated_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                             const float* d_gate, float* d_y,
                                             int rows, int cols, float eps,
                                             void* stream = nullptr);

// --- Gated DeltaNet linear attention ---------------------------------------

// Depthwise causal convolution over the packed QKV channels followed by SiLU.
// `d_tail` holds the previous kernel-1 inputs per channel and is updated in
// place when `update_tail` is set, which is what makes decode continue a prefill.
bool qwen_split_packed_qkv_cuda(const float* d_packed, float* d_q, float* d_k,
                                float* d_v, int rows, int key_dim, int value_dim,
                                void* stream = nullptr);

bool qwen_causal_depthwise_conv_silu_cuda(const float* d_x_rows,
                                          const uint16_t* d_weight_fp16,
                                          float* d_tail, float* d_y_rows,
                                          int seq_len, int channels, int kernel,
                                          bool update_tail, void* stream = nullptr);

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
bool qwen_linear_attn_gates_cuda(const float* d_a_rows, const float* d_b_rows,
                                 const uint16_t* d_a_log_fp16,
                                 const uint16_t* d_dt_bias_fp16,
                                 float* d_g_rows, float* d_beta_rows,
                                 int rows, int heads, void* stream = nullptr);

// One recurrent step of the gated delta rule over a persistent FP32 state
// [heads, key_dim, value_dim]. Query/key carry `key_heads` heads and are
// repeated `heads / key_heads` times to cover the value heads.
bool qwen_gated_delta_step_cuda(float* d_state, const float* d_q, const float* d_k,
                                const float* d_v, const float* d_g, const float* d_beta,
                                float* d_out, int heads, int key_heads, int key_dim,
                                int value_dim, float q_scale, void* stream = nullptr);

// Batched causal recurrent pass. Rows are processed strictly in order while
// the persistent state is updated in place; this is the prefill-only path.
bool qwen_gated_delta_sequence_cuda(float* d_state, const float* d_q, const float* d_k,
                                    const float* d_v, const float* d_g, const float* d_beta,
                                    float* d_out, int rows, int heads, int key_heads,
                                    int key_dim, int value_dim, float q_scale,
                                    void* stream = nullptr);

bool qwen_partial_rope_cuda(float* d_q, float* d_k, int position, int rotary_dim,
                            float theta, int q_heads, int kv_heads, int head_dim,
                            void* stream = nullptr);

// Prefill form of the above over `rows` consecutive positions in one launch.
bool qwen_partial_rope_rows_cuda(float* d_q, float* d_k, int start_position, int rows,
                                 int rotary_dim, float theta, int q_heads, int kv_heads,
                                 int head_dim, void* stream = nullptr);

// q_proj stores [q, gate] pairs per head; split them into contiguous matrices.
bool qwen_split_q_gate_cuda(const float* d_q_proj, float* d_q, float* d_gate,
                            int rows, int q_heads, int head_dim,
                            void* stream = nullptr);

// Select the rank-local KV head range from a replicated K/V projection.
bool qwen_select_kv_heads_cuda(const float* d_src, float* d_dst, int rows,
                               int total_kv_heads, int local_kv_heads,
                               int head_dim, int head_offset,
                               void* stream = nullptr);

// Append one or more tokens to a per-head contiguous KV cache.
bool qwen_append_kv_cache_cuda(const float* d_k_rows, const float* d_v_rows,
                               float* d_k_cache, float* d_v_cache, int seq_len,
                               int kv_heads, int head_dim, int start_pos,
                               int max_context, void* stream = nullptr);

// Single-token GQA attention against the cache. Query head h reads KV head
// h / (q_heads / kv_heads). d_score_scratch holds q_heads * context_len floats.
bool qwen_gqa_decode_attention_cuda(const float* d_q, const float* d_k_cache,
                                    const float* d_v_cache, float* d_out,
                                    float* d_score_scratch,
                                    int q_heads, int kv_heads, int head_dim,
                                    int context_len, int max_context,
                                    void* stream = nullptr);

// Multi-token causal GQA attention. Query row t is at absolute position
// `position_offset + t` and may attend to cache entries up to that position.
bool qwen_gqa_prefill_attention_cuda(const float* d_q_rows, const float* d_k_cache,
                                     const float* d_v_cache, float* d_out_rows,
                                     int seq_len, int q_heads, int kv_heads,
                                     int head_dim, int position_offset,
                                     int max_context, void* stream = nullptr);

// --- Small elementwise helpers ---------------------------------------------

bool qwen_sigmoid_mul_cuda(const float* d_x, const float* d_gate, float* d_y,
                           int count, void* stream = nullptr);
bool qwen_add_inplace_cuda(float* d_y, const float* d_x, int count,
                           void* stream = nullptr);
bool qwen_silu_mul_rows_cuda(const float* d_gate, const float* d_up, float* d_y,
                             int rows, int cols, void* stream = nullptr);

}  // namespace dsv4
