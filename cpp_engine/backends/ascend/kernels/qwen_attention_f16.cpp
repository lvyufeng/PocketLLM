// First-generation Ascend 910 full-attention kernels.
//
// Attention vectors stay in UB and vector instructions perform the QK dot,
// softmax, and weighted-V accumulation. This is especially important on the
// first-generation scalar unit: recomputing one QK dot per output dimension made
// the correctness baseline O(head_dim^2 * context) and dominated the whole model.
// Cache padding, causal limits, and split ownership remain explicit.

#include "qwen_ascend_kernel_common.hpp"

namespace {

using namespace pocket;

constexpr uint32_t kMaxRotaryDim = 128;
constexpr uint32_t kMaxHeadsPerBlock = 1;

__aicore__ inline float half_value(const AscendC::GlobalTensor<half>& tensor,
                                   uint32_t index) {
    return static_cast<float>(tensor.GetValue(index));
}

__aicore__ inline void set_half(AscendC::GlobalTensor<half>& tensor,
                                uint32_t index, float value) {
    tensor.SetValue(index, static_cast<half>(value));
}

__aicore__ inline float dot_query_key(
    const AscendC::GlobalTensor<half>& q,
    const AscendC::GlobalTensor<half>& k_cache,
    uint32_t q_offset, uint32_t k_offset, uint32_t head_dim) {
    float result = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        result += half_value(q, q_offset + d) * half_value(k_cache, k_offset + d);
    }
    return result;
}

// The softmax scale is a kernel argument rather than 1/sqrt(head_dim) computed
// here: aicore rejects a cast between a floating and an unsigned integer
// variable, and the integer sqrt overload that does compile truncates (head_dim
// 128 would give 11 instead of 11.3137). The host already knows the exact value.

// The scalar Exp helper needs an event pair around the scalar write and vector
// read. It is called repeatedly by the scalar attention baseline, so keep the
// eight-lane temporary in the worker's TBuf rather than allocating one per score.
__aicore__ inline float exp_score(const AscendC::LocalTensor<float>& exp_buf,
                                  float score) {
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::Duplicate(exp_buf, score, kAlignFloat);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(exp_buf, exp_buf, kAlignFloat);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return exp_buf.GetValue(0);
}

}  // namespace

// In-place partial RoPE. The host launcher supplies FP32 sin/cos tables with shape
// [rows, rotary_dim / 2], because this CANN release has no classic-aicore Sin/Cos
// or Pow implementation. Every q and KV head shares the row's angle table. The
// absolute start position is already folded into those per-call table values.
extern "C" __global__ __aicore__ void qwen_partial_rope_rows_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR cos_table, GM_ADDR sin_table,
    uint32_t rows, uint32_t rotary_dim,
    uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> q_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> k_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cos_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sin_buf;
    pipe.InitBuffer(q_buf, kMaxRotaryDim * sizeof(half));
    pipe.InitBuffer(k_buf, kMaxRotaryDim * sizeof(half));
    pipe.InitBuffer(cos_buf, (kMaxRotaryDim / 2) * sizeof(float));
    pipe.InitBuffer(sin_buf, (kMaxRotaryDim / 2) * sizeof(float));

    const uint32_t half_rotary = rotary_dim / 2;
    const uint32_t table_stride = half_rotary;
    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<float> cos_gm;
    AscendC::GlobalTensor<float> sin_gm;
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q),
                         rows * q_heads * head_dim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k),
                         rows * kv_heads * head_dim);
    cos_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos_table),
                           rows * table_stride);
    sin_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin_table),
                           rows * table_stride);

    AscendC::LocalTensor<half> q_local = q_buf.Get<half>();
    AscendC::LocalTensor<half> k_local = k_buf.Get<half>();
    AscendC::LocalTensor<float> cos_local = cos_buf.Get<float>();
    AscendC::LocalTensor<float> sin_local = sin_buf.Get<float>();
    for (uint32_t row = AscendC::GetBlockIdx(); row < rows;
         row += AscendC::GetBlockNum()) {
        AscendC::DataCopy(cos_local, cos_gm[row * table_stride], table_stride);
        AscendC::DataCopy(sin_local, sin_gm[row * table_stride], table_stride);
        wait_load_before_scalar();

        const uint32_t q_row = row * q_heads * head_dim;
        for (uint32_t head = 0; head < q_heads; ++head) {
            const uint32_t base = q_row + head * head_dim;
            load_half_exact(q_local, q_gm, base, rotary_dim);
            wait_load_before_scalar();
            for (uint32_t index = 0; index < half_rotary; ++index) {
                const float a = static_cast<float>(q_local.GetValue(index));
                const float b = static_cast<float>(q_local.GetValue(index + half_rotary));
                const float c = cos_local.GetValue(index);
                const float s = sin_local.GetValue(index);
                q_local.SetValue(index, static_cast<half>(a * c - b * s));
                q_local.SetValue(index + half_rotary,
                                 static_cast<half>(b * c + a * s));
            }
            wait_scalar_before_store();
            store_half_exact(q_gm, base, q_local, rotary_dim);
            wait_store_before_load();
        }

        const uint32_t k_row = row * kv_heads * head_dim;
        for (uint32_t head = 0; head < kv_heads; ++head) {
            const uint32_t base = k_row + head * head_dim;
            load_half_exact(k_local, k_gm, base, rotary_dim);
            wait_load_before_scalar();
            for (uint32_t index = 0; index < half_rotary; ++index) {
                const float a = static_cast<float>(k_local.GetValue(index));
                const float b = static_cast<float>(k_local.GetValue(index + half_rotary));
                const float c = cos_local.GetValue(index);
                const float s = sin_local.GetValue(index);
                k_local.SetValue(index, static_cast<half>(a * c - b * s));
                k_local.SetValue(index + half_rotary,
                                 static_cast<half>(b * c + a * s));
            }
            wait_scalar_before_store();
            store_half_exact(k_gm, base, k_local, rotary_dim);
            wait_store_before_load();
        }
        wait_scalar_before_load();
    }
}

// Copy [seq_len, kv_heads, head_dim] rows into their cache positions. The real
// Qwen row is 128 halfs; a 512-half tile covers four rows and keeps every GM
// transfer block-aligned. The launcher uses one block for the unaligned fallback
// so two scalar tails can never race on the same 32-byte cache line.
extern "C" __global__ __aicore__ void qwen_append_kv_cache_kernel(
    GM_ADDR k_rows, GM_ADDR v_rows, GM_ADDR k_cache, GM_ADDR v_cache,
    uint32_t seq_len, uint32_t kv_heads, uint32_t head_dim,
    uint32_t start_pos, uint32_t max_context) {
    constexpr uint32_t kCopyTile = 512;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> k_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> v_buf;
    pipe.InitBuffer(k_buf, kCopyTile * sizeof(half));
    pipe.InitBuffer(v_buf, kCopyTile * sizeof(half));

    const uint32_t rows = seq_len * kv_heads;
    const uint32_t total = rows * head_dim;
    const uint32_t destination_base = start_pos * kv_heads * head_dim;
    AscendC::GlobalTensor<half> k_src;
    AscendC::GlobalTensor<half> v_src;
    AscendC::GlobalTensor<half> k_dst;
    AscendC::GlobalTensor<half> v_dst;
    k_src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_rows), total);
    v_src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_rows), total);
    k_dst.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                          max_context * kv_heads * head_dim);
    v_dst.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_cache),
                          max_context * kv_heads * head_dim);

    AscendC::LocalTensor<half> k_local = k_buf.Get<half>();
    AscendC::LocalTensor<half> v_local = v_buf.Get<half>();
    const uint32_t tiles = (total + kCopyTile - 1) / kCopyTile;
    for (uint32_t tile = AscendC::GetBlockIdx(); tile < tiles;
         tile += AscendC::GetBlockNum()) {
        const uint32_t offset = tile * kCopyTile;
        const uint32_t count = min_u32(kCopyTile, total - offset);
        load_half_exact(k_local, k_src, offset, count);
        store_half_exact(k_dst, destination_base + offset, k_local, count);
        wait_store_before_load();
        load_half_exact(v_local, v_src, offset, count);
        store_half_exact(v_dst, destination_base + offset, v_local, count);
        wait_store_before_load();
    }
}

// Decode GQA baseline. A block owns one query head and writes its score line and
// output row. The score scratch remains normalized probabilities after this call,
// matching the useful behavior of the CUDA baseline.
extern "C" __global__ __aicore__ void qwen_gqa_decode_attention_kernel(
    GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR out,
    GM_ADDR score_scratch, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t context_len, uint32_t max_context,
    float scale) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buf;
    pipe.InitBuffer(exp_buf, kAlignFloat * sizeof(float));
    AscendC::LocalTensor<float> exp_tile = exp_buf.Get<float>();

    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> out_gm;
    AscendC::GlobalTensor<float> scores;
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), q_heads * head_dim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                         max_context * kv_heads * head_dim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_cache),
                         max_context * kv_heads * head_dim);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), q_heads * head_dim);
    scores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(score_scratch),
                           q_heads * context_len);

    const uint32_t repeat = q_heads / kv_heads;
    for (uint32_t head = AscendC::GetBlockIdx(); head < q_heads;
         head += AscendC::GetBlockNum()) {
        const uint32_t kv_head = head / repeat;
        const uint32_t q_offset = head * head_dim;
        float maximum = -3.402823466e+38F;
        for (uint32_t pos = 0; pos < context_len; ++pos) {
            const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
            const float score = dot_query_key(q_gm, k_gm, q_offset, cache_offset,
                                              head_dim) * scale;
            scores.SetValue(head * context_len + pos, score);
            if (score > maximum) maximum = score;
        }

        float denominator = 0.0f;
        for (uint32_t pos = 0; pos < context_len; ++pos) {
            const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
            const float score = dot_query_key(q_gm, k_gm, q_offset, cache_offset,
                                              head_dim) * scale;
            denominator += exp_score(exp_tile, score - maximum);
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (uint32_t pos = 0; pos < context_len; ++pos) {
            const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
            const float score = dot_query_key(q_gm, k_gm, q_offset, cache_offset,
                                              head_dim) * scale;
            scores.SetValue(head * context_len + pos,
                            exp_score(exp_tile, score - maximum) * inverse);
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            float value = 0.0f;
            for (uint32_t pos = 0; pos < context_len; ++pos) {
                const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
                const float score = dot_query_key(q_gm, k_gm, q_offset,
                                                  cache_offset, head_dim) * scale;
                value += exp_score(exp_tile, score - maximum) *
                         half_value(v_gm, cache_offset + d);
            }
            set_half(out_gm, q_offset + d, value * inverse);
        }
    }
}

// Causal prefill baseline. One block owns a (row, query-head) pair. Recompute the
// score in the normalization and value passes rather than allocating a scratch
// plane: this keeps the neutral prefill signature scratch-free.
extern "C" __global__ __aicore__ void qwen_gqa_prefill_attention_kernel(
    GM_ADDR q_rows, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR out_rows,
    uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
    uint32_t position_offset, uint32_t max_context, float scale) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buf;
    pipe.InitBuffer(exp_buf, kAlignFloat * sizeof(float));
    AscendC::LocalTensor<float> exp_tile = exp_buf.Get<float>();

    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> out_gm;
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_rows),
                         seq_len * q_heads * head_dim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                         max_context * kv_heads * head_dim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_cache),
                         max_context * kv_heads * head_dim);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out_rows),
                           seq_len * q_heads * head_dim);

    const uint32_t total = seq_len * q_heads;
    const uint32_t repeat = q_heads / kv_heads;
    for (uint32_t work = AscendC::GetBlockIdx(); work < total;
         work += AscendC::GetBlockNum()) {
        const uint32_t row = work / q_heads;
        const uint32_t head = work % q_heads;
        const uint32_t kv_head = head / repeat;
        const uint32_t q_offset = (row * q_heads + head) * head_dim;
        const uint32_t context_len = position_offset + row + 1;

        float maximum = -3.402823466e+38F;
        for (uint32_t pos = 0; pos < context_len; ++pos) {
            const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
            const float score = dot_query_key(q_gm, k_gm, q_offset, cache_offset,
                                              head_dim) * scale;
            if (score > maximum) maximum = score;
        }

        float denominator = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            float value = 0.0f;
            for (uint32_t pos = 0; pos < context_len; ++pos) {
                const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
                const float score = dot_query_key(q_gm, k_gm, q_offset,
                                                  cache_offset, head_dim) * scale;
                const float probability = exp_score(exp_tile, score - maximum);
                denominator += (d == 0 ? probability : 0.0f);
                value += probability * half_value(v_gm, cache_offset + d);
            }
            set_half(out_gm, q_offset + d,
                     denominator > 0.0f ? value / denominator : 0.0f);
        }
    }
}

// Verify split + merge. One block owns a (row, query-head) output and fills all
// split partials in the caller-provided [rows,q_heads,splits,head_dim+2] plane.
extern "C" __global__ __aicore__ void qwen_gqa_verify_attention_kernel(
    GM_ADDR q_rows, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR out_rows,
    GM_ADDR partial_scratch, uint32_t rows, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t position_offset, uint32_t max_context,
    uint32_t splits, float scale) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buf;
    pipe.InitBuffer(exp_buf, kAlignFloat * sizeof(float));
    AscendC::LocalTensor<float> exp_tile = exp_buf.Get<float>();

    AscendC::GlobalTensor<half> q_gm;
    AscendC::GlobalTensor<half> k_gm;
    AscendC::GlobalTensor<half> v_gm;
    AscendC::GlobalTensor<half> out_gm;
    AscendC::GlobalTensor<float> partial;
    q_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_rows),
                         rows * q_heads * head_dim);
    k_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                         max_context * kv_heads * head_dim);
    v_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_cache),
                         max_context * kv_heads * head_dim);
    out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out_rows),
                           rows * q_heads * head_dim);
    partial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial_scratch),
                            rows * q_heads * splits * (head_dim + 2));

    const uint32_t total = rows * q_heads;
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t context_len = position_offset + rows;
    const uint32_t positions_per_split =
        (context_len + splits - 1) / splits;

    for (uint32_t work = AscendC::GetBlockIdx(); work < total;
         work += AscendC::GetBlockNum()) {
        const uint32_t row = work / q_heads;
        const uint32_t head = work % q_heads;
        const uint32_t kv_head = head / repeat;
        const uint32_t q_offset = (row * q_heads + head) * head_dim;
        const uint32_t context_limit = position_offset + row + 1;
        const uint32_t output_base = (row * q_heads + head) *
                                     splits * (head_dim + 2);

        for (uint32_t split = 0; split < splits; ++split) {
            const uint32_t split_start = split * positions_per_split;
            const uint32_t split_end =
                min_u32(context_len, split_start + positions_per_split);
            const uint32_t begin = split_start;
            const uint32_t end = min_u32(split_end, context_limit);
            const uint32_t base = output_base + split * (head_dim + 2);
            float maximum = -3.402823466e+38F;
            if (begin < end) {
                for (uint32_t pos = begin; pos < end; ++pos) {
                    const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
                    const float score = dot_query_key(q_gm, k_gm, q_offset,
                                                      cache_offset, head_dim) * scale;
                    if (score > maximum) maximum = score;
                }
            }
            partial.SetValue(base, maximum);
            if (begin >= end) {
                partial.SetValue(base + 1, 0.0f);
                for (uint32_t d = 0; d < head_dim; ++d) partial.SetValue(base + 2 + d, 0.0f);
                continue;
            }

            float denominator = 0.0f;
            for (uint32_t pos = begin; pos < end; ++pos) {
                const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
                const float score = dot_query_key(q_gm, k_gm, q_offset,
                                                  cache_offset, head_dim) * scale;
                denominator += exp_score(exp_tile, score - maximum);
            }
            partial.SetValue(base + 1, denominator);
            for (uint32_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (uint32_t pos = begin; pos < end; ++pos) {
                    const uint32_t cache_offset = (pos * kv_heads + kv_head) * head_dim;
                    const float score = dot_query_key(q_gm, k_gm, q_offset,
                                                      cache_offset, head_dim) * scale;
                    value += exp_score(exp_tile, score - maximum) *
                             half_value(v_gm, cache_offset + d);
                }
                partial.SetValue(base + 2 + d, value);
            }
        }

        float global_max = -3.402823466e+38F;
        for (uint32_t split = 0; split < splits; ++split) {
            const uint32_t base = output_base + split * (head_dim + 2);
            const float maximum = partial.GetValue(base);
            if (maximum > global_max) global_max = maximum;
        }
        float denominator = 0.0f;
        for (uint32_t split = 0; split < splits; ++split) {
            const uint32_t base = output_base + split * (head_dim + 2);
            const float maximum = partial.GetValue(base);
            const float weight = maximum == -3.402823466e+38F
                ? 0.0f : exp_score(exp_tile, maximum - global_max);
            denominator += weight * partial.GetValue(base + 1);
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            float value = 0.0f;
            for (uint32_t split = 0; split < splits; ++split) {
                const uint32_t base = output_base + split * (head_dim + 2);
                const float maximum = partial.GetValue(base);
                const float weight = maximum == -3.402823466e+38F
                    ? 0.0f : exp_score(exp_tile, maximum - global_max);
                value += weight * partial.GetValue(base + 2 + d);
            }
            set_half(out_gm, q_offset + d,
                     denominator > 0.0f ? value / denominator : 0.0f);
        }
    }
}
