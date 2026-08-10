#include "dspark.hpp"
#include "cuda_ops.hpp"
#include "json_lite.hpp"
#include "safetensors_reader.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>
#include <cuda_runtime.h>

namespace dspark {

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// Optional scalar lookups: config.json for this checkpoint carries every field
// we need, but falling back to the struct default keeps older configs loadable.
int json_int_or(const dsv4::JsonObject& obj, const std::string& key, int fallback) {
    const dsv4::JsonValue* v = dsv4::object_get(obj, key);
    if (v == nullptr || !v->is_number()) return fallback;
    return static_cast<int>(v->number());
}

float json_float_or(const dsv4::JsonObject& obj, const std::string& key, float fallback) {
    const dsv4::JsonValue* v = dsv4::object_get(obj, key);
    if (v == nullptr || !v->is_number()) return fallback;
    return static_cast<float>(v->number());
}

std::vector<int> json_int_array_or(const dsv4::JsonObject& obj, const std::string& key,
                                   const std::vector<int>& fallback) {
    const dsv4::JsonValue* v = dsv4::object_get(obj, key);
    if (v == nullptr || !v->is_array()) return fallback;
    std::vector<int> out;
    for (const auto& item : v->array()) {
        if (!item.is_number()) return fallback;
        out.push_back(static_cast<int>(item.number()));
    }
    return out.empty() ? fallback : out;
}

// Row-major [rows, cols] slice of `rows_take` rows starting at `row_start`.
std::vector<uint8_t> slice_rows_u8(const uint8_t* src, int row_start, int rows_take, int cols) {
    std::vector<uint8_t> out(static_cast<size_t>(rows_take) * cols);
    std::memcpy(out.data(),
                src + static_cast<size_t>(row_start) * cols,
                out.size());
    return out;
}

// Row-major [rows, cols] slice of `cols_take` columns starting at `col_start`.
std::vector<uint8_t> slice_cols_u8(const uint8_t* src, int rows, int cols,
                                   int col_start, int cols_take) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * cols_take);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(out.data() + static_cast<size_t>(r) * cols_take,
                    src + static_cast<size_t>(r) * cols + col_start,
                    static_cast<size_t>(cols_take));
    }
    return out;
}

}  // namespace

// ============================================================================
// Config
// ============================================================================

Config Config::from_json(const char* config_path) {
    Config cfg;

    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error(std::string("DSpark: cannot open config ") + config_path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const dsv4::JsonValue root = dsv4::parse_json(ss.str());
    if (!root.is_object()) {
        throw std::runtime_error(std::string("DSpark: config is not a JSON object: ") + config_path);
    }
    const dsv4::JsonObject& o = root.object();

    cfg.block_size = json_int_or(o, "dspark_block_size", cfg.block_size);
    cfg.noise_token_id = json_int_or(o, "dspark_noise_token_id", cfg.noise_token_id);
    cfg.target_layer_ids = json_int_array_or(o, "dspark_target_layer_ids", cfg.target_layer_ids);
    cfg.markov_rank = json_int_or(o, "dspark_markov_rank", cfg.markov_rank);
    cfg.window_size = json_int_or(o, "sliding_window", cfg.window_size);
    cfg.dim = json_int_or(o, "hidden_size", cfg.dim);
    cfg.vocab_size = json_int_or(o, "vocab_size", cfg.vocab_size);
    cfg.hc_mult = json_int_or(o, "hc_mult", cfg.hc_mult);
    cfg.norm_eps = json_float_or(o, "rms_norm_eps", cfg.norm_eps);

    cfg.n_heads = json_int_or(o, "num_attention_heads", cfg.n_heads);
    cfg.head_dim = json_int_or(o, "head_dim", cfg.head_dim);
    cfg.q_lora_rank = json_int_or(o, "q_lora_rank", cfg.q_lora_rank);
    cfg.o_lora_rank = json_int_or(o, "o_lora_rank", cfg.o_lora_rank);
    cfg.o_groups = json_int_or(o, "o_groups", cfg.o_groups);
    cfg.rope_dim = json_int_or(o, "qk_rope_head_dim", cfg.rope_dim);
    cfg.rope_theta = json_float_or(o, "rope_theta", cfg.rope_theta);

    cfg.n_experts = json_int_or(o, "n_routed_experts", cfg.n_experts);
    cfg.topk = json_int_or(o, "num_experts_per_tok", cfg.topk);
    cfg.moe_inter = json_int_or(o, "moe_intermediate_size", cfg.moe_inter);
    cfg.route_scale = json_float_or(o, "routed_scaling_factor", cfg.route_scale);
    cfg.swiglu_limit = json_float_or(o, "swiglu_limit", cfg.swiglu_limit);

    // The checkpoint stores one DSpark stage per `mtp.N.` prefix. config.json
    // has no field for it (num_nextn_predict_layers counts predicted tokens,
    // not stages), so this default is refined by probing the index at load.
    cfg.n_stages = json_int_or(o, "dspark_n_stages", cfg.n_stages);

    return cfg;
}

// ============================================================================
// DSparkEngine::Impl
// ============================================================================

struct DSparkEngine::Impl {
    Config config;
    int tp_rank;
    int tp_world_size;
    std::string checkpoint_dir;

    // Stage 0: main_proj + main_norm
    struct Stage0 {
        // main_proj: [4096, 12288] FP8 linear (transposed storage)
        // weight: F8_E4M3, scale: F8_E8M0
        uint8_t* main_proj_weight = nullptr;  // [out_dim, in_dim]
        uint8_t* main_proj_scale = nullptr;   // [out_dim/128, in_dim/128]
        uint16_t* main_norm_weight = nullptr; // [dim] BF16

        // Embedding (reuse main model's embed.weight)
        uint16_t* embed_weight = nullptr;     // [vocab, dim] BF16
    } stage0;

    // Routed-expert sharding, derived from config + tp_world_size. Each rank
    // owns a contiguous slice [experts_start, experts_start + experts_per_rank);
    // the grouped MoE kernel indexes experts by their local id within it.
    int experts_per_rank = 0;
    int experts_start = 0;

    // Per-rank attention dimensions, derived from config + tp_world_size.
    struct AttnDims {
        int dim = 0;
        int q_a_dim = 0;      // q_lora_rank
        int heads = 0;        // local heads = 64 / tp
        int head_dim = 0;
        int q_dim = 0;        // heads * head_dim
        int kv_dim = 0;       // 1 KV head * head_dim
        int groups = 0;       // local o groups = 8 / tp
        int group_rank = 0;   // o_lora_rank
        int group_dim = 0;    // q_dim / groups
        int attn_mid = 0;     // groups * group_rank
        int rope_dim = 0;
        int window_size = 0;
    } adims;

    // One DSpark draft stage. Same tensor layout as a main-model layer minus
    // the compressor/indexer, so the checkpoint dtypes match 1:1:
    //   attn.w*: F8_E4M3 weight + F8_E8M0 scale, norms BF16, hc_* F32.
    struct StageBlock {
        // hc_pre parameters (F32)
        float* hc_attn_fn = nullptr;      // [3*hc, hc*dim]
        float* hc_attn_scale = nullptr;   // [3]
        float* hc_attn_base = nullptr;    // [3*hc]
        float* hc_ffn_fn = nullptr;
        float* hc_ffn_scale = nullptr;
        float* hc_ffn_base = nullptr;

        // Attention (FP8 weight + FP8 scale, BF16 norms)
        struct Attention {
            uint8_t* wq_a_weight = nullptr;   // [q_a_dim, dim]
            uint8_t* wq_a_scale = nullptr;
            uint8_t* wq_b_weight = nullptr;   // [q_dim, q_a_dim] (TP row-sliced)
            uint8_t* wq_b_scale = nullptr;
            uint16_t* q_norm_weight = nullptr;  // [q_a_dim] BF16
            uint8_t* wkv_weight = nullptr;    // [kv_dim, dim]
            uint8_t* wkv_scale = nullptr;
            uint16_t* kv_norm_weight = nullptr; // [kv_dim] BF16
            uint8_t* wo_a_weight = nullptr;   // [attn_mid, dim] (TP row-sliced)
            uint8_t* wo_a_scale = nullptr;
            uint8_t* wo_b_weight = nullptr;   // [dim, attn_mid] (TP col-sliced)
            uint8_t* wo_b_scale = nullptr;
            float* attn_sink = nullptr;       // [heads] (TP sliced)

            // KV ring cache: [window_size, head_dim] float
            float* kv_cache = nullptr;
        } attn;

        uint16_t* attn_norm_weight = nullptr;  // [dim] BF16

        // FFN (MoE) — identical layout to a main-model layer's ffn.
        struct FFN {
            // Gate
            uint16_t* gate_weight = nullptr;  // [n_experts, dim] BF16
            float* gate_bias = nullptr;       // [n_experts] F32

            // Shared experts: FP8 weight + FP8 scale
            uint8_t* shared_w1_weight = nullptr;  // [moe_inter, dim]
            uint8_t* shared_w1_scale = nullptr;
            uint8_t* shared_w3_weight = nullptr;  // [moe_inter, dim]
            uint8_t* shared_w3_scale = nullptr;
            uint8_t* shared_w2_weight = nullptr;  // [dim, moe_inter]
            uint8_t* shared_w2_scale = nullptr;

            // Routed experts. Unlike the main model these stay resident on the
            // device: a rank owns n_experts/tp_world of them, and at 13.4 MB per
            // expert that is 10.3 GB at TP=1 but only 2.6 GB at TP=4. Staging
            // them per draft instead would put a PCIe transfer on the critical
            // path of every draft round, which is the one thing speculative
            // decoding cannot afford -- the draft has to be cheap relative to
            // the verify it is trying to skip.
            //
            // Laid out as one contiguous arena per matrix so the grouped MoE
            // kernel can index expert `local` at a fixed stride, exactly like
            // the main model's DeviceFp4ActiveArena.
            uint8_t* routed_w1q = nullptr;
            uint8_t* routed_w1s = nullptr;
            uint8_t* routed_w2q = nullptr;
            uint8_t* routed_w2s = nullptr;
            uint8_t* routed_w3q = nullptr;
            uint8_t* routed_w3s = nullptr;
            size_t routed_w1q_stride = 0;
            size_t routed_w1s_stride = 0;
            size_t routed_w2q_stride = 0;
            size_t routed_w2s_stride = 0;
            size_t routed_w3q_stride = 0;
            size_t routed_w3s_stride = 0;
        } ffn;

        uint16_t* ffn_norm_weight = nullptr;  // [dim] BF16
    };

    // 3 stages: mtp.0 / mtp.1 / mtp.2. mtp.0 also owns main_proj/main_norm,
    // mtp.2 also owns norm/hc_head/markov_head/confidence_head.
    static constexpr int kNumStages = 3;
    StageBlock stages[kNumStages];

    // Stage 2 special heads
    struct Stage2Heads {
        uint16_t* norm_weight = nullptr;  // [dim] BF16

        // hc_head parameters (F32)
        float* hc_head_fn = nullptr;      // [hc, hc*dim]
        float* hc_head_scale = nullptr;   // [1]
        float* hc_head_base = nullptr;    // [hc]

        // Markov head. Both tensors are stored [vocab, markov_rank] BF16:
        // w1 is the bigram embedding table, w2 is the output projection whose
        // rows are vocab entries (so it is used transposed).
        uint16_t* markov_w1_weight = nullptr;  // [vocab, rank] BF16
        uint16_t* markov_w2_weight = nullptr;  // [vocab, rank] BF16

        // Confidence head: [1, dim + markov_rank] BF16, no bias in checkpoint.
        uint16_t* confidence_proj_weight = nullptr;
    } stage2_heads;

    // Shared embedding and head (tied to main model, not owned here).
    uint16_t* embed_weight = nullptr;
    uint16_t* head_weight = nullptr;

    // Device buffers for forward pass
    struct DeviceBuffers {
        // Stage 0 buffers
        int* d_draft_input_ids = nullptr;    // [block_size] token IDs
        float* d_main_concat = nullptr;      // [1, 1, dim*3]
        float* d_main_proj_out = nullptr;    // [1, 1, dim]
        float* d_main_normed = nullptr;      // [1, 1, dim]
        float* d_draft_input = nullptr;      // [1, block_size, dim]
        float* d_draft_x = nullptr;          // [1, block_size, hc_mult, dim]

        // Stage 1-2 buffers
        float* d_hidden = nullptr;           // [1, block_size, hc_mult, dim]
        float* d_logits = nullptr;           // [block_size, vocab_size]

        // Attention path buffers
        float* d_attn_x = nullptr;           // [block_size, dim]
        float* d_attn_post = nullptr;        // [block_size, hc_mult]
        float* d_attn_comb = nullptr;        // [block_size, hc_mult, hc_mult]
        float* d_attn_normed = nullptr;      // [block_size, dim]
        float* d_attn_out = nullptr;         // [block_size, dim]

        // DSparkAttention internals
        float* d_q_a = nullptr;              // [block_size, q_a_dim]
        float* d_q_normed = nullptr;         // [block_size, q_a_dim]
        float* d_q = nullptr;                // [block_size, q_dim]
        float* d_kv_a = nullptr;             // [block_size, kv_dim]
        float* d_draft_kv = nullptr;         // [block_size, head_dim]
        float* d_main_kv_a = nullptr;        // [1, kv_dim]
        float* d_main_kv = nullptr;          // [1, head_dim]
        // Concatenated keys the draft attends to: the whole ring window
        // followed by this block's own draft keys.
        float* d_kv_concat = nullptr;        // [window_size + block_size, head_dim]
        int32_t* d_topk_indices = nullptr;   // [block_size, window_size + block_size]
        float* d_attn_value = nullptr;       // [block_size, q_dim]
        float* d_attn_mid = nullptr;         // [block_size, attn_mid]

        // FFN path buffers
        float* d_ffn_x = nullptr;            // [block_size, dim]
        float* d_ffn_post = nullptr;         // [block_size, hc_mult]
        float* d_ffn_comb = nullptr;         // [block_size, hc_mult, hc_mult]
        float* d_ffn_normed = nullptr;       // [block_size, dim]
        float* d_ffn_out = nullptr;          // [block_size, dim]

        // MoE routing + grouped-expert scratch. Sized for block_size tokens, so
        // every allocation here is small and can live for the engine's lifetime.
        int64_t* d_route_indices = nullptr;  // [block_size, topk]
        float* d_route_weights = nullptr;    // [block_size, topk]
        int64_t* d_group_route_tokens = nullptr;  // [block_size * topk]
        float* d_group_route_weights = nullptr;   // [block_size * topk]
        int32_t* d_seg_starts = nullptr;     // [experts_per_rank + 1]
        int32_t* d_counts = nullptr;         // [experts_per_rank]
        int32_t* d_offsets = nullptr;        // [experts_per_rank]
        int32_t* d_total_routes = nullptr;   // [1]
        int32_t* d_token_slot_routes = nullptr;  // [block_size, topk]
        // Shared-expert intermediates.
        float* d_shared_gate = nullptr;      // [block_size, moe_inter]
        float* d_shared_up = nullptr;        // [block_size, moe_inter]
        float* d_shared_hidden = nullptr;    // [block_size, moe_inter]
        float* d_shared_out = nullptr;       // [block_size, dim]
        // Grouped-MoE workspace, kept across calls: a cudaMalloc/cudaFree pair
        // per call synchronizes the device and costs far more than the kernel.
        dsv4::MoePrefillFp4GroupedWorkspace moe_ws;

        void allocate(const Config& cfg, const AttnDims& ad, int experts_per_rank) {
            const int dim = cfg.dim;
            const int bsz = cfg.block_size;
            const int hc = cfg.hc_mult;
            const int vocab = cfg.vocab_size;
            const int n_target = static_cast<int>(cfg.target_layer_ids.size());
            const int kv_len = ad.window_size + bsz;

            cudaMalloc(&d_draft_input_ids, bsz * sizeof(int));
            cudaMalloc(&d_main_concat, static_cast<size_t>(dim) * n_target * sizeof(float));
            cudaMalloc(&d_main_proj_out, 1 * 1 * dim * sizeof(float));
            cudaMalloc(&d_main_normed, 1 * 1 * dim * sizeof(float));
            cudaMalloc(&d_draft_input, 1 * bsz * dim * sizeof(float));
            cudaMalloc(&d_draft_x, 1 * bsz * hc * dim * sizeof(float));
            cudaMalloc(&d_hidden, 1 * bsz * hc * dim * sizeof(float));
            cudaMalloc(&d_logits, static_cast<size_t>(bsz) * vocab * sizeof(float));

            // Attention path buffers
            cudaMalloc(&d_attn_x, bsz * dim * sizeof(float));
            cudaMalloc(&d_attn_post, bsz * hc * sizeof(float));
            cudaMalloc(&d_attn_comb, bsz * hc * hc * sizeof(float));
            cudaMalloc(&d_attn_normed, bsz * dim * sizeof(float));
            cudaMalloc(&d_attn_out, bsz * dim * sizeof(float));

            // DSparkAttention internals
            cudaMalloc(&d_q_a, static_cast<size_t>(bsz) * ad.q_a_dim * sizeof(float));
            cudaMalloc(&d_q_normed, static_cast<size_t>(bsz) * ad.q_a_dim * sizeof(float));
            cudaMalloc(&d_q, static_cast<size_t>(bsz) * ad.q_dim * sizeof(float));
            cudaMalloc(&d_kv_a, static_cast<size_t>(bsz) * ad.kv_dim * sizeof(float));
            cudaMalloc(&d_draft_kv, static_cast<size_t>(bsz) * ad.head_dim * sizeof(float));
            cudaMalloc(&d_main_kv_a, static_cast<size_t>(ad.kv_dim) * sizeof(float));
            cudaMalloc(&d_main_kv, static_cast<size_t>(ad.head_dim) * sizeof(float));
            cudaMalloc(&d_kv_concat, static_cast<size_t>(kv_len) * ad.head_dim * sizeof(float));
            cudaMalloc(&d_topk_indices, static_cast<size_t>(bsz) * kv_len * sizeof(int32_t));
            cudaMalloc(&d_attn_value, static_cast<size_t>(bsz) * ad.q_dim * sizeof(float));
            cudaMalloc(&d_attn_mid, static_cast<size_t>(bsz) * ad.attn_mid * sizeof(float));

            // FFN path buffers
            cudaMalloc(&d_ffn_x, bsz * dim * sizeof(float));
            cudaMalloc(&d_ffn_post, bsz * hc * sizeof(float));
            cudaMalloc(&d_ffn_comb, bsz * hc * hc * sizeof(float));
            cudaMalloc(&d_ffn_normed, bsz * dim * sizeof(float));
            cudaMalloc(&d_ffn_out, bsz * dim * sizeof(float));

            // MoE routing scratch
            const int topk = cfg.topk;
            const int moe_inter = cfg.moe_inter;
            const size_t routes_cap = static_cast<size_t>(bsz) * topk;
            cudaMalloc(&d_route_indices, routes_cap * sizeof(int64_t));
            cudaMalloc(&d_route_weights, routes_cap * sizeof(float));
            cudaMalloc(&d_group_route_tokens, routes_cap * sizeof(int64_t));
            cudaMalloc(&d_group_route_weights, routes_cap * sizeof(float));
            cudaMalloc(&d_seg_starts, (static_cast<size_t>(experts_per_rank) + 1) * sizeof(int32_t));
            cudaMalloc(&d_counts, static_cast<size_t>(experts_per_rank) * sizeof(int32_t));
            cudaMalloc(&d_offsets, static_cast<size_t>(experts_per_rank) * sizeof(int32_t));
            cudaMalloc(&d_total_routes, sizeof(int32_t));
            cudaMalloc(&d_token_slot_routes, routes_cap * sizeof(int32_t));
            cudaMalloc(&d_shared_gate, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            cudaMalloc(&d_shared_up, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            cudaMalloc(&d_shared_hidden, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            cudaMalloc(&d_shared_out, bsz * dim * sizeof(float));

            // Grouped-MoE workspace. Every route belongs to one of block_size
            // tokens, so routes <= block_size * topk and the padded path needs
            // at most experts_per_rank * block_size rows.
            moe_ws.dim = dim;
            moe_ws.inter_dim = moe_inter;
            moe_ws.routes_cap = static_cast<int>(routes_cap);
            moe_ws.padded_rows_cap = experts_per_rank * bsz;
            const size_t rd = routes_cap * dim;
            const size_t pd = static_cast<size_t>(moe_ws.padded_rows_cap) * dim;
            const size_t pi = static_cast<size_t>(moe_ws.padded_rows_cap) * moe_inter;
            cudaMalloc(&moe_ws.d_x_sorted, rd * sizeof(float));
            cudaMalloc(&moe_ws.d_partials, rd * sizeof(float));
            cudaMalloc(&moe_ws.d_x_q, rd);
            cudaMalloc(&moe_ws.d_x_scale, routes_cap * sizeof(float));
            cudaMalloc(&moe_ws.d_x_pad, pd);
            cudaMalloc(&moe_ws.d_x_scale_pad, static_cast<size_t>(moe_ws.padded_rows_cap) * sizeof(float));
            cudaMalloc(&moe_ws.d_gate, pi * sizeof(float));
            cudaMalloc(&moe_ws.d_up, pi * sizeof(float));
            cudaMalloc(&moe_ws.d_hidden_q, pi);
            cudaMalloc(&moe_ws.d_hidden_scale, static_cast<size_t>(moe_ws.padded_rows_cap) * sizeof(float));
        }

        void free_all() {
            if (d_draft_input_ids) cudaFree(d_draft_input_ids);
            if (d_main_concat) cudaFree(d_main_concat);
            if (d_main_proj_out) cudaFree(d_main_proj_out);
            if (d_main_normed) cudaFree(d_main_normed);
            if (d_draft_input) cudaFree(d_draft_input);
            if (d_draft_x) cudaFree(d_draft_x);
            if (d_hidden) cudaFree(d_hidden);
            if (d_logits) cudaFree(d_logits);

            if (d_attn_x) cudaFree(d_attn_x);
            if (d_attn_post) cudaFree(d_attn_post);
            if (d_attn_comb) cudaFree(d_attn_comb);
            if (d_attn_normed) cudaFree(d_attn_normed);
            if (d_attn_out) cudaFree(d_attn_out);

            if (d_q_a) cudaFree(d_q_a);
            if (d_q_normed) cudaFree(d_q_normed);
            if (d_q) cudaFree(d_q);
            if (d_kv_a) cudaFree(d_kv_a);
            if (d_draft_kv) cudaFree(d_draft_kv);
            if (d_main_kv_a) cudaFree(d_main_kv_a);
            if (d_main_kv) cudaFree(d_main_kv);
            if (d_kv_concat) cudaFree(d_kv_concat);
            if (d_topk_indices) cudaFree(d_topk_indices);
            if (d_attn_value) cudaFree(d_attn_value);
            if (d_attn_mid) cudaFree(d_attn_mid);

            if (d_ffn_x) cudaFree(d_ffn_x);
            if (d_ffn_post) cudaFree(d_ffn_post);
            if (d_ffn_comb) cudaFree(d_ffn_comb);
            if (d_ffn_normed) cudaFree(d_ffn_normed);
            if (d_ffn_out) cudaFree(d_ffn_out);

            for (void* p : {(void*)d_route_indices, (void*)d_route_weights,
                            (void*)d_group_route_tokens, (void*)d_group_route_weights,
                            (void*)d_seg_starts, (void*)d_counts, (void*)d_offsets,
                            (void*)d_total_routes, (void*)d_token_slot_routes,
                            (void*)d_shared_gate, (void*)d_shared_up,
                            (void*)d_shared_hidden, (void*)d_shared_out,
                            (void*)moe_ws.d_x_sorted, (void*)moe_ws.d_partials,
                            (void*)moe_ws.d_x_q, (void*)moe_ws.d_x_scale,
                            (void*)moe_ws.d_x_pad, (void*)moe_ws.d_x_scale_pad,
                            (void*)moe_ws.d_gate, (void*)moe_ws.d_up,
                            (void*)moe_ws.d_hidden_q, (void*)moe_ws.d_hidden_scale}) {
                if (p != nullptr) cudaFree(p);
            }
        }
    } buffers;

    Impl(const char* ckpt_dir, int rank, int world_size)
        : tp_rank(rank), tp_world_size(world_size), checkpoint_dir(ckpt_dir) {
        config = Config::from_json((std::string(ckpt_dir) + "/config.json").c_str());
        init_dims();
        buffers.allocate(config, adims, experts_per_rank);
    }

    ~Impl() {
        buffers.free_all();
        free_weights();
    }

    void init_dims() {
        if (config.n_heads % tp_world_size != 0) {
            throw std::runtime_error("DSpark: num_attention_heads not divisible by tp_world_size");
        }
        if (config.o_groups % tp_world_size != 0) {
            throw std::runtime_error("DSpark: o_groups not divisible by tp_world_size");
        }
        if (config.n_experts % tp_world_size != 0) {
            throw std::runtime_error("DSpark: n_routed_experts not divisible by tp_world_size");
        }
        experts_per_rank = config.n_experts / tp_world_size;
        experts_start = tp_rank * experts_per_rank;
        adims.dim = config.dim;
        adims.q_a_dim = config.q_lora_rank;
        adims.heads = config.n_heads / tp_world_size;
        adims.head_dim = config.head_dim;
        adims.q_dim = adims.heads * adims.head_dim;
        adims.kv_dim = adims.head_dim;  // single KV head
        adims.groups = config.o_groups / tp_world_size;
        adims.group_rank = config.o_lora_rank;
        adims.group_dim = adims.q_dim / adims.groups;
        adims.attn_mid = adims.groups * adims.group_rank;
        adims.rope_dim = config.rope_dim;
        adims.window_size = config.window_size;
    }

    // ------------------------------------------------------------------
    // Weight loading
    // ------------------------------------------------------------------

    // Every device allocation made by load_weights, so the destructor can free
    // them without each field needing its own cudaFree call.
    std::vector<void*> owned_device_buffers;

    void* device_alloc(size_t bytes, const char* what) {
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, bytes), what);
        owned_device_buffers.push_back(p);
        return p;
    }

    void free_weights() {
        for (void* p : owned_device_buffers) {
            if (p != nullptr) cudaFree(p);
        }
        owned_device_buffers.clear();
    }

    void* upload(const void* host, size_t bytes, const char* what) {
        void* d = device_alloc(bytes, what);
        check_cuda(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice), what);
        return d;
    }

    void load_weights() {
        using namespace dsv4;

        SafeTensorsIndex index(checkpoint_dir);

        // Determine the real stage count from the index rather than trusting a
        // config default: a checkpoint with a different number of `mtp.N.`
        // prefixes would otherwise be silently half-loaded.
        {
            int found = 0;
            for (int s = 0; s < kNumStages; ++s) {
                if (index.shard_for_tensor("mtp." + std::to_string(s) + ".attn_norm.weight") != nullptr) {
                    found = s + 1;
                } else {
                    break;
                }
            }
            if (found == 0) {
                throw std::runtime_error("DSpark: no mtp.* stages found in checkpoint " +
                                         checkpoint_dir);
            }
            if (index.shard_for_tensor("mtp." + std::to_string(kNumStages) + ".attn_norm.weight") != nullptr) {
                throw std::runtime_error("DSpark: checkpoint has more than " +
                                         std::to_string(kNumStages) + " mtp stages");
            }
            config.n_stages = found;
        }

        // Shards are mmap'd, so keeping every one we touch open for the whole
        // load is cheap and avoids re-opening per tensor.
        std::map<std::string, std::unique_ptr<SafeTensorsShard>> open_shards;

        auto shard_for = [&](const std::string& name) -> SafeTensorsShard& {
            const std::string* s = index.shard_for_tensor(name);
            if (s == nullptr) {
                throw std::runtime_error("DSpark: tensor not found in index: " + name);
            }
            auto it = open_shards.find(*s);
            if (it == open_shards.end()) {
                it = open_shards.emplace(*s, std::make_unique<SafeTensorsShard>(
                                                 index.shard_path(*s)))
                         .first;
            }
            return *it->second;
        };

        // Fetch a tensor and assert its dtype and shape, so a checkpoint whose
        // layout differs fails here rather than as silent numerical garbage.
        struct Loaded {
            const uint8_t* data;
            const SafeTensorInfo* info;
        };
        auto get = [&](const std::string& name, SafeDType want,
                       std::vector<uint64_t> want_shape) -> Loaded {
            SafeTensorsShard& sh = shard_for(name);
            const SafeTensorInfo* info = sh.find_tensor(name);
            if (info == nullptr) {
                throw std::runtime_error("DSpark: tensor missing from shard: " + name);
            }
            if (info->dtype != want) {
                throw std::runtime_error("DSpark: dtype mismatch for " + name + ": expected " +
                                         safe_dtype_name(want) + ", got " +
                                         safe_dtype_name(info->dtype));
            }
            if (!want_shape.empty() && info->shape != want_shape) {
                std::string got;
                for (auto d : info->shape) got += std::to_string(d) + ",";
                std::string exp;
                for (auto d : want_shape) exp += std::to_string(d) + ",";
                throw std::runtime_error("DSpark: shape mismatch for " + name +
                                         ": expected [" + exp + "], got [" + got + "]");
            }
            return Loaded{reinterpret_cast<const uint8_t*>(sh.tensor_data(*info)), info};
        };

        auto upload_whole = [&](const std::string& name, SafeDType want,
                                std::vector<uint64_t> want_shape) -> void* {
            Loaded t = get(name, want, std::move(want_shape));
            return upload(t.data, t.info->nbytes, name.c_str());
        };

        const int dim = config.dim;
        const int hc = config.hc_mult;
        const uint64_t udim = static_cast<uint64_t>(dim);

        for (int stage = 0; stage < config.n_stages; ++stage) {
            const std::string p = "mtp." + std::to_string(stage) + ".";
            StageBlock& b = stages[stage];

            // hc_pre parameters. fn is [2*hc + hc*hc, hc*dim]: the pre gates,
            // the post gates, and the hc x hc combine matrix stacked in that
            // order (24 rows for hc=4), matching hc_pre_float_rows_kernel.
            const uint64_t hc_fn_rows = static_cast<uint64_t>(2 * hc + hc * hc);
            const uint64_t hc_fn_cols = static_cast<uint64_t>(hc) * udim;
            b.hc_attn_fn = static_cast<float*>(
                upload_whole(p + "hc_attn_fn", SafeDType::F32, {hc_fn_rows, hc_fn_cols}));
            b.hc_attn_scale = static_cast<float*>(
                upload_whole(p + "hc_attn_scale", SafeDType::F32, {3}));
            b.hc_attn_base = static_cast<float*>(
                upload_whole(p + "hc_attn_base", SafeDType::F32, {hc_fn_rows}));
            b.hc_ffn_fn = static_cast<float*>(
                upload_whole(p + "hc_ffn_fn", SafeDType::F32, {hc_fn_rows, hc_fn_cols}));
            b.hc_ffn_scale = static_cast<float*>(
                upload_whole(p + "hc_ffn_scale", SafeDType::F32, {3}));
            b.hc_ffn_base = static_cast<float*>(
                upload_whole(p + "hc_ffn_base", SafeDType::F32, {hc_fn_rows}));

            // Norms
            b.attn_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn_norm.weight", SafeDType::BF16, {udim}));
            b.ffn_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "ffn_norm.weight", SafeDType::BF16, {udim}));

            // --- Attention ---
            // wq_a / wkv are replicated across ranks (input-dim sharding is not
            // used here); wq_b / wo_a / wo_b / attn_sink are TP-sharded exactly
            // like the main model's layer weights.
            const uint64_t q_a = static_cast<uint64_t>(config.q_lora_rank);
            b.attn.wq_a_weight = static_cast<uint8_t*>(
                upload_whole(p + "attn.wq_a.weight", SafeDType::F8_E4M3, {q_a, udim}));
            b.attn.wq_a_scale = static_cast<uint8_t*>(
                upload_whole(p + "attn.wq_a.scale", SafeDType::F8_E8M0, {q_a / 128, udim / 128}));
            b.attn.q_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn.q_norm.weight", SafeDType::BF16, {q_a}));

            const uint64_t kv_dim_u = static_cast<uint64_t>(adims.kv_dim);
            b.attn.wkv_weight = static_cast<uint8_t*>(
                upload_whole(p + "attn.wkv.weight", SafeDType::F8_E4M3, {kv_dim_u, udim}));
            b.attn.wkv_scale = static_cast<uint8_t*>(
                upload_whole(p + "attn.wkv.scale", SafeDType::F8_E8M0,
                             {kv_dim_u / 128, udim / 128}));
            b.attn.kv_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn.kv_norm.weight", SafeDType::BF16, {kv_dim_u}));

            // wq_b: [n_heads*head_dim, q_a] -> local rows [q_dim, q_a]
            {
                const int row_start = tp_rank * adims.q_dim;
                Loaded w = get(p + "attn.wq_b.weight", SafeDType::F8_E4M3,
                               {static_cast<uint64_t>(config.n_heads) * config.head_dim, q_a});
                Loaded s = get(p + "attn.wq_b.scale", SafeDType::F8_E8M0,
                               {static_cast<uint64_t>(config.n_heads) * config.head_dim / 128,
                                q_a / 128});
                auto wl = slice_rows_u8(w.data, row_start, adims.q_dim, adims.q_a_dim);
                auto sl = slice_rows_u8(s.data, row_start / 128, adims.q_dim / 128,
                                        adims.q_a_dim / 128);
                b.attn.wq_b_weight = static_cast<uint8_t*>(upload(wl.data(), wl.size(), "wq_b"));
                b.attn.wq_b_scale = static_cast<uint8_t*>(upload(sl.data(), sl.size(), "wq_b scale"));
            }

            // wo_a: [o_groups*o_lora_rank, dim] -> local rows [attn_mid, dim]
            // wo_b: [dim, o_groups*o_lora_rank] -> local cols [dim, attn_mid]
            {
                const uint64_t o_full = static_cast<uint64_t>(config.o_groups) * config.o_lora_rank;
                const int row_start = tp_rank * adims.attn_mid;
                Loaded wa = get(p + "attn.wo_a.weight", SafeDType::F8_E4M3, {o_full, udim});
                Loaded sa = get(p + "attn.wo_a.scale", SafeDType::F8_E8M0,
                                {o_full / 128, udim / 128});
                auto wal = slice_rows_u8(wa.data, row_start, adims.attn_mid, dim);
                auto sal = slice_rows_u8(sa.data, row_start / 128, adims.attn_mid / 128, dim / 128);
                b.attn.wo_a_weight = static_cast<uint8_t*>(upload(wal.data(), wal.size(), "wo_a"));
                b.attn.wo_a_scale = static_cast<uint8_t*>(upload(sal.data(), sal.size(), "wo_a scale"));

                Loaded wb = get(p + "attn.wo_b.weight", SafeDType::F8_E4M3, {udim, o_full});
                Loaded sb = get(p + "attn.wo_b.scale", SafeDType::F8_E8M0,
                                {udim / 128, o_full / 128});
                auto wbl = slice_cols_u8(wb.data, dim, static_cast<int>(o_full), row_start,
                                         adims.attn_mid);
                auto sbl = slice_cols_u8(sb.data, dim / 128, static_cast<int>(o_full / 128),
                                         row_start / 128, adims.attn_mid / 128);
                b.attn.wo_b_weight = static_cast<uint8_t*>(upload(wbl.data(), wbl.size(), "wo_b"));
                b.attn.wo_b_scale = static_cast<uint8_t*>(upload(sbl.data(), sbl.size(), "wo_b scale"));
            }

            // attn_sink: [n_heads] -> local [heads]
            {
                Loaded t = get(p + "attn.attn_sink", SafeDType::F32,
                               {static_cast<uint64_t>(config.n_heads)});
                const float* src = reinterpret_cast<const float*>(t.data) + tp_rank * adims.heads;
                b.attn.attn_sink = static_cast<float*>(
                    upload(src, static_cast<size_t>(adims.heads) * sizeof(float), "attn_sink"));
            }

            // KV ring cache, zeroed: [window_size, head_dim]
            {
                const size_t bytes = static_cast<size_t>(adims.window_size) * adims.head_dim *
                                     sizeof(float);
                b.attn.kv_cache = static_cast<float*>(device_alloc(bytes, "dspark kv_cache"));
                check_cuda(cudaMemset(b.attn.kv_cache, 0, bytes), "zero dspark kv_cache");
            }

            // --- FFN ---
            const uint64_t n_exp = static_cast<uint64_t>(config.n_experts);
            const uint64_t inter = static_cast<uint64_t>(config.moe_inter);
            b.ffn.gate_weight = static_cast<uint16_t*>(
                upload_whole(p + "ffn.gate.weight", SafeDType::BF16, {n_exp, udim}));
            b.ffn.gate_bias = static_cast<float*>(
                upload_whole(p + "ffn.gate.bias", SafeDType::F32, {n_exp}));

            b.ffn.shared_w1_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w1.weight", SafeDType::F8_E4M3, {inter, udim}));
            b.ffn.shared_w1_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w1.scale", SafeDType::F8_E8M0,
                             {inter / 128, udim / 128}));
            b.ffn.shared_w3_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w3.weight", SafeDType::F8_E4M3, {inter, udim}));
            b.ffn.shared_w3_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w3.scale", SafeDType::F8_E8M0,
                             {inter / 128, udim / 128}));
            b.ffn.shared_w2_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w2.weight", SafeDType::F8_E4M3, {udim, inter}));
            b.ffn.shared_w2_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w2.scale", SafeDType::F8_E8M0,
                             {udim / 128, inter / 128}));

            // Routed experts: this rank's slice, uploaded into one arena per
            // matrix. Experts are FP4 (int8-packed pairs) with e8m0 scales, the
            // same format the main model's grouped kernel consumes.
            {
                const std::string ep = p + "ffn.experts.";
                auto expert_name = [&](int e, const char* w, const char* suffix) {
                    return ep + std::to_string(e) + "." + w + "." + suffix;
                };

                // Size the arena from expert 0; every expert shares a shape, and
                // `get` below re-checks each one, so a ragged checkpoint fails
                // loudly rather than writing past a stride.
                Loaded w1q0 = get(expert_name(experts_start, "w1", "weight"), SafeDType::I8, {inter, udim / 2});
                Loaded w1s0 = get(expert_name(experts_start, "w1", "scale"), SafeDType::F8_E8M0, {inter, udim / 32});
                Loaded w2q0 = get(expert_name(experts_start, "w2", "weight"), SafeDType::I8, {udim, inter / 2});
                Loaded w2s0 = get(expert_name(experts_start, "w2", "scale"), SafeDType::F8_E8M0, {udim, inter / 32});
                Loaded w3q0 = get(expert_name(experts_start, "w3", "weight"), SafeDType::I8, {inter, udim / 2});
                Loaded w3s0 = get(expert_name(experts_start, "w3", "scale"), SafeDType::F8_E8M0, {inter, udim / 32});

                b.ffn.routed_w1q_stride = w1q0.info->nbytes;
                b.ffn.routed_w1s_stride = w1s0.info->nbytes;
                b.ffn.routed_w2q_stride = w2q0.info->nbytes;
                b.ffn.routed_w2s_stride = w2s0.info->nbytes;
                b.ffn.routed_w3q_stride = w3q0.info->nbytes;
                b.ffn.routed_w3s_stride = w3s0.info->nbytes;

                const size_t n = static_cast<size_t>(experts_per_rank);
                b.ffn.routed_w1q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w1q_stride, "dspark routed w1q"));
                b.ffn.routed_w1s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w1s_stride, "dspark routed w1s"));
                b.ffn.routed_w2q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w2q_stride, "dspark routed w2q"));
                b.ffn.routed_w2s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w2s_stride, "dspark routed w2s"));
                b.ffn.routed_w3q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w3q_stride, "dspark routed w3q"));
                b.ffn.routed_w3s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w3s_stride, "dspark routed w3s"));

                for (int local = 0; local < experts_per_rank; ++local) {
                    const int e = experts_start + local;
                    struct Upload {
                        const char* w;
                        const char* suffix;
                        SafeDType dtype;
                        std::vector<uint64_t> shape;
                        uint8_t* dst;
                        size_t stride;
                    };
                    const Upload uploads[] = {
                        {"w1", "weight", SafeDType::I8,      {inter, udim / 2},  b.ffn.routed_w1q, b.ffn.routed_w1q_stride},
                        {"w1", "scale",  SafeDType::F8_E8M0, {inter, udim / 32}, b.ffn.routed_w1s, b.ffn.routed_w1s_stride},
                        {"w2", "weight", SafeDType::I8,      {udim, inter / 2},  b.ffn.routed_w2q, b.ffn.routed_w2q_stride},
                        {"w2", "scale",  SafeDType::F8_E8M0, {udim, inter / 32}, b.ffn.routed_w2s, b.ffn.routed_w2s_stride},
                        {"w3", "weight", SafeDType::I8,      {inter, udim / 2},  b.ffn.routed_w3q, b.ffn.routed_w3q_stride},
                        {"w3", "scale",  SafeDType::F8_E8M0, {inter, udim / 32}, b.ffn.routed_w3s, b.ffn.routed_w3s_stride},
                    };
                    for (const Upload& u : uploads) {
                        Loaded t = get(expert_name(e, u.w, u.suffix), u.dtype, u.shape);
                        if (t.info->nbytes != u.stride) {
                            throw std::runtime_error("DSpark: routed expert size mismatch for " +
                                                     expert_name(e, u.w, u.suffix));
                        }
                        check_cuda(cudaMemcpy(u.dst + static_cast<size_t>(local) * u.stride,
                                              t.data, u.stride, cudaMemcpyHostToDevice),
                                   "upload dspark routed expert");
                    }
                }
            }

            // Stage 0 extras: main_proj + main_norm
            if (stage == 0) {
                const uint64_t n_target = static_cast<uint64_t>(config.target_layer_ids.size());
                stage0.main_proj_weight = static_cast<uint8_t*>(
                    upload_whole(p + "main_proj.weight", SafeDType::F8_E4M3,
                                 {udim, udim * n_target}));
                stage0.main_proj_scale = static_cast<uint8_t*>(
                    upload_whole(p + "main_proj.scale", SafeDType::F8_E8M0,
                                 {udim / 128, udim * n_target / 128}));
                stage0.main_norm_weight = static_cast<uint16_t*>(
                    upload_whole(p + "main_norm.weight", SafeDType::BF16, {udim}));
            }

            // Last stage extras: norm + hc_head + markov + confidence
            if (stage == config.n_stages - 1) {
                stage2_heads.norm_weight = static_cast<uint16_t*>(
                    upload_whole(p + "norm.weight", SafeDType::BF16, {udim}));
                stage2_heads.hc_head_fn = static_cast<float*>(
                    upload_whole(p + "hc_head_fn", SafeDType::F32,
                                 {static_cast<uint64_t>(hc), static_cast<uint64_t>(hc) * udim}));
                stage2_heads.hc_head_scale = static_cast<float*>(
                    upload_whole(p + "hc_head_scale", SafeDType::F32, {1}));
                stage2_heads.hc_head_base = static_cast<float*>(
                    upload_whole(p + "hc_head_base", SafeDType::F32,
                                 {static_cast<uint64_t>(hc)}));

                const uint64_t vocab = static_cast<uint64_t>(config.vocab_size);
                const uint64_t rank_u = static_cast<uint64_t>(config.markov_rank);
                // Both markov tensors are [vocab, rank]: w1 is the bigram
                // embedding table, w2 the output projection used transposed.
                // Vocab-major layout means TP would shard rows; keep them whole
                // for now and gather at the head, as the reference does.
                stage2_heads.markov_w1_weight = static_cast<uint16_t*>(
                    upload_whole(p + "markov_head.markov_w1.weight", SafeDType::BF16,
                                 {vocab, rank_u}));
                stage2_heads.markov_w2_weight = static_cast<uint16_t*>(
                    upload_whole(p + "markov_head.markov_w2.weight", SafeDType::BF16,
                                 {vocab, rank_u}));
                stage2_heads.confidence_proj_weight = static_cast<uint16_t*>(
                    upload_whole(p + "confidence_head.proj.weight", SafeDType::BF16,
                                 {1, udim + rank_u}));
            }
        }

        // Embedding is tied to the main model's table.
        embed_weight = static_cast<uint16_t*>(
            upload_whole("embed.weight", SafeDType::BF16,
                         {static_cast<uint64_t>(config.vocab_size), udim}));
        stage0.embed_weight = embed_weight;

        weights_loaded = true;
    }

    bool weights_loaded = false;

    // Stage 0: main_proj + main_norm + embed
    void forward_stage0(int input_token, const std::vector<float*>& main_hiddens,
                        const std::vector<int>& draft_input_ids) {
        const int dim = config.dim;
        const int bsz = config.block_size;

        if (static_cast<int>(draft_input_ids.size()) != bsz) {
            throw std::runtime_error("draft_input_ids size must match block_size");
        }

        // 1. Concat main_hidden_states from layers 40/41/42
        // main_hiddens[0/1/2] are [1, 1, dim] each
        // TODO: implement concat kernel or use cudaMemcpy
        for (int i = 0; i < 3; ++i) {
            cudaMemcpy(buffers.d_main_concat + i * dim,
                      main_hiddens[i],
                      dim * sizeof(float),
                      cudaMemcpyDeviceToDevice);
        }

        // 2. FP8 linear: [1, dim*3] @ [dim, dim*3]^T -> [1, dim]
        // Note: weight is stored as [out_dim=4096, in_dim=12288]
        // x: [1, 12288], weight: [4096, 12288], output: [1, 4096]
        if (stage0.main_proj_weight == nullptr || stage0.main_proj_scale == nullptr) {
            throw std::runtime_error("Stage 0 main_proj weights not loaded");
        }
        bool success = dsv4::fp8_e4m3_e8m0_matmul_cuda(
            buffers.d_main_concat,        // input: [1, 12288]
            stage0.main_proj_weight,      // weight: [4096, 12288] F8_E4M3
            stage0.main_proj_scale,       // scale: [32, 96] F8_E8M0
            buffers.d_main_proj_out,      // output: [1, 4096]
            1,                            // batch
            4096,                         // rows (output dim)
            12288,                        // cols (input dim)
            nullptr                       // stream
        );
        if (!success) {
            throw std::runtime_error("FP8 matmul failed in main_proj");
        }

        // 3. RMSNorm: [1, dim] -> [1, dim]
        // Apply RMSNorm to main_proj output
        if (stage0.main_norm_weight == nullptr) {
            throw std::runtime_error("Stage 0 main_norm weights not loaded");
        }
        success = dsv4::rmsnorm_bf16_gamma_cuda(
            buffers.d_main_proj_out,     // input: [1, 4096]
            stage0.main_norm_weight,     // gamma: [4096] BF16
            buffers.d_main_normed,       // output: [1, 4096]
            dim,                         // cols = 4096
            1e-6f,                       // eps
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("RMSNorm failed in main_norm");
        }

        // 4. Embed draft_input_ids: [block_size] -> [block_size, dim]
        // Upload draft_input_ids to device
        cudaMemcpy(buffers.d_draft_input_ids, draft_input_ids.data(),
                   bsz * sizeof(int), cudaMemcpyHostToDevice);

        // Lookup embeddings using bf16_rows_to_float_cuda
        if (stage0.embed_weight == nullptr) {
            throw std::runtime_error("Stage 0 embed_weight not loaded");
        }
        success = dsv4::bf16_rows_to_float_cuda(
            stage0.embed_weight,         // matrix: [vocab, dim] BF16
            buffers.d_draft_input_ids,   // row indices: [block_size]
            buffers.d_draft_input,       // output: [block_size, dim]
            bsz,                         // rows
            dim,                         // cols
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("Embedding lookup failed");
        }

        // 5. hc_mult expansion: [block_size, dim] -> [block_size, hc_mult, dim]
        // Repeat each embedding hc_mult=4 times along dim=1
        const int hc = config.hc_mult;
        success = dsv4::hc_repeat_rows_cuda(
            buffers.d_draft_input,       // input: [block_size, dim]
            buffers.d_draft_x,           // output: [block_size, hc_mult, dim]
            bsz,                         // rows
            dim,                         // dim
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("hc_repeat failed");
        }
    }

    // Keys the draft attends to: the live ring window, then this block's own
    // draft keys. Every draft position sees the same set (the reference builds
    // one row and expands it), so causality across draft positions is
    // deliberately not enforced. `start_pos` is the committed token's position.
    // Returns the number of key slots per row.
    int build_topk_indices(int start_pos) {
        const int bsz = config.block_size;
        const int win = adims.window_size;
        // The ring holds positions 0..start_pos, capped at the window.
        const int committed = std::min(win, start_pos + 1);
        const int topk = committed + bsz;

        std::vector<int32_t> row(topk);
        for (int i = 0; i < committed; ++i) row[i] = i;
        for (int i = 0; i < bsz; ++i) row[committed + i] = win + i;

        // Same row for every draft position.
        std::vector<int32_t> all(static_cast<size_t>(bsz) * topk);
        for (int t = 0; t < bsz; ++t) {
            std::memcpy(all.data() + static_cast<size_t>(t) * topk, row.data(),
                        row.size() * sizeof(int32_t));
        }
        check_cuda(cudaMemcpy(buffers.d_topk_indices, all.data(),
                              all.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "copy dspark topk indices");
        return topk;
    }

    // DSparkAttention: queries come from the draft tokens, keys/values from the
    // main model's hidden state. Mirrors DSparkAttention.forward in
    // src/models/deepseek_v4/dspark.py.
    //
    //   d_x       [block_size, dim]  attn_norm output for the draft tokens
    //   d_main_x  [dim]              main model's projected+normed hidden
    //   start_pos                    position of the committed token; the draft
    //                                tokens occupy start_pos+1 .. start_pos+bsz
    //   d_out     [block_size, dim]  attention output
    void forward_attention(StageBlock* block, const float* d_x, const float* d_main_x,
                           int start_pos, float* d_out) {
        using namespace dsv4;
        const int bsz = config.block_size;
        const int dim = config.dim;
        const int win = adims.window_size;
        const int hd = adims.head_dim;
        const int rd = adims.rope_dim;
        const int draft_pos = start_pos + 1;  // position of draft token 0
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        if (start_pos <= 0) {
            // The reference asserts start_pos > 0 when building topk indices:
            // with an empty window there is nothing to draft from.
            throw std::runtime_error("DSpark attention requires start_pos > 0");
        }

        // --- Q from the draft tokens ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->attn.wq_a_weight, block->attn.wq_a_scale,
                                       buffers.d_q_a, bsz, adims.q_a_dim, dim))
            throw std::runtime_error("dspark wq_a failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_q_a, block->attn.q_norm_weight,
                                          buffers.d_q_normed, bsz, adims.q_a_dim,
                                          config.norm_eps))
            throw std::runtime_error("dspark q_norm failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_q_normed, block->attn.wq_b_weight,
                                       block->attn.wq_b_scale, buffers.d_q, bsz,
                                       adims.q_dim, adims.q_a_dim))
            throw std::runtime_error("dspark wq_b failed");
        // Per-head RMSNorm (no gamma) then rope; draft token i sits at draft_pos+i.
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_q, bsz, adims.heads, hd, rd,
                                         draft_pos, config.rope_theta, false,
                                         config.norm_eps))
            throw std::runtime_error("dspark q rope failed");

        // --- KV for the committed position, from the main model's hidden ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_main_x, block->attn.wkv_weight, block->attn.wkv_scale,
                                       buffers.d_main_kv_a, 1, adims.kv_dim, dim))
            throw std::runtime_error("dspark main wkv failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_main_kv_a, block->attn.kv_norm_weight,
                                          buffers.d_main_kv, 1, adims.kv_dim, config.norm_eps))
            throw std::runtime_error("dspark main kv_norm failed");
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_main_kv, 1, 1, hd, rd,
                                         start_pos, config.rope_theta, false, 0.0f))
            throw std::runtime_error("dspark main kv rope failed");
        // act_quant covers only the non-rope prefix of each row.
        if (!fp8_act_quant_dequant_rows_strided_cuda(buffers.d_main_kv, 1, hd - rd, hd, 64))
            throw std::runtime_error("dspark main kv act_quant failed");
        // Ring write, before the concat below reads the window.
        if (!copy_rows_to_kv_cache_cuda(buffers.d_main_kv, block->attn.kv_cache, 1, hd,
                                        win, start_pos))
            throw std::runtime_error("dspark ring cache write failed");

        // --- KV from the draft tokens themselves (never cached) ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->attn.wkv_weight, block->attn.wkv_scale,
                                       buffers.d_kv_a, bsz, adims.kv_dim, dim))
            throw std::runtime_error("dspark draft wkv failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_kv_a, block->attn.kv_norm_weight,
                                          buffers.d_draft_kv, bsz, adims.kv_dim,
                                          config.norm_eps))
            throw std::runtime_error("dspark draft kv_norm failed");
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_draft_kv, bsz, 1, hd, rd,
                                         draft_pos, config.rope_theta, false, 0.0f))
            throw std::runtime_error("dspark draft kv rope failed");
        if (!fp8_act_quant_dequant_rows_strided_cuda(buffers.d_draft_kv, bsz, hd - rd, hd, 64))
            throw std::runtime_error("dspark draft kv act_quant failed");

        // --- Sparse attention over [ring window ++ draft keys] ---
        check_cuda(cudaMemcpy(buffers.d_kv_concat, block->attn.kv_cache,
                              static_cast<size_t>(win) * hd * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "dspark kv concat window");
        check_cuda(cudaMemcpy(buffers.d_kv_concat + static_cast<size_t>(win) * hd,
                              buffers.d_draft_kv,
                              static_cast<size_t>(bsz) * hd * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "dspark kv concat draft");
        const int topk = build_topk_indices(start_pos);
        if (!prefill_sparse_attention_indexed_cuda(
                buffers.d_q, buffers.d_kv_concat, block->attn.attn_sink,
                buffers.d_topk_indices, buffers.d_attn_value, bsz, adims.heads,
                win + bsz, topk, hd, scale))
            throw std::runtime_error("dspark sparse attention failed");
        // Inverse rope on the attention output, at the draft positions.
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_attn_value, bsz, adims.heads, hd, rd,
                                         draft_pos, config.rope_theta, true, 0.0f))
            throw std::runtime_error("dspark inverse rope failed");

        // --- Output projection: grouped wo_a, then wo_b ---
        for (int g = 0; g < adims.groups; ++g) {
            const float* group_x = buffers.d_attn_value + static_cast<size_t>(g) * adims.group_dim;
            const uint8_t* group_w = block->attn.wo_a_weight +
                static_cast<size_t>(g) * adims.group_rank * adims.group_dim;
            const uint8_t* group_s = block->attn.wo_a_scale +
                static_cast<size_t>(g) * (adims.group_rank / 128) * (adims.group_dim / 128);
            float* group_y = buffers.d_attn_mid + static_cast<size_t>(g) * adims.group_rank;
            // Each token's slice of d_attn_value / d_attn_mid is strided.
            if (!fp8_e4m3_e8m0_matmul_strided_cuda(group_x, group_w, group_s, group_y, bsz,
                                                   adims.group_rank, adims.group_dim,
                                                   adims.q_dim, adims.attn_mid))
                throw std::runtime_error("dspark wo_a failed");
        }
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_attn_mid, block->attn.wo_b_weight,
                                       block->attn.wo_b_scale, d_out, bsz, dim,
                                       adims.attn_mid))
            throw std::runtime_error("dspark wo_b failed");
        // NOTE: with tp_world_size > 1 the caller must all-reduce d_out across
        // ranks before the residual add, exactly as the main model does. That
        // hook is not wired yet, so TP>1 draft output is incomplete.
    }

    // Routed MoE + shared expert for `rows` tokens. d_x is the ffn_norm output
    // [rows, dim]; d_out receives shared + sum(route_weight * expert(x)).
    //
    // Mirrors the main model's prefill MoE: gate -> group routes by expert ->
    // one grouped FP4 kernel over all routes -> add the shared expert. The
    // routed weights are already resident (see StageBlock::FFN), so unlike the
    // main model there is no staging step on the critical path.
    void forward_moe(StageBlock* block, const float* d_x, float* d_out, int rows) {
        using namespace dsv4;
        const int dim = config.dim;
        const int inter = config.moe_inter;
        const int topk = config.topk;

        // Gate: scores = sqrt_softplus(W x), select topk by score + bias, then
        // renormalize the *unbiased* scores over the selection.
        if (!gate_topk_bf16_rows_cuda(d_x, block->ffn.gate_weight, block->ffn.gate_bias,
                                      buffers.d_route_indices, buffers.d_route_weights, rows,
                                      config.n_experts, dim, topk, config.route_scale)) {
            throw std::runtime_error("dspark gate topk failed");
        }

        // Bucket the routes this rank owns by expert. Routes whose expert lives
        // on another rank are dropped here and contributed by that rank's
        // all-reduce (which TP>1 still needs wiring for -- see forward_attention).
        if (!moe_group_routes_cuda(buffers.d_route_indices, buffers.d_route_weights,
                                   buffers.d_group_route_tokens, buffers.d_group_route_weights,
                                   buffers.d_seg_starts, buffers.d_counts, buffers.d_offsets,
                                   buffers.d_total_routes, rows, topk, experts_start,
                                   experts_per_rank, buffers.d_token_slot_routes)) {
            throw std::runtime_error("dspark group routes failed");
        }

        int32_t total_routes = 0;
        check_cuda(cudaMemcpy(&total_routes, buffers.d_total_routes, sizeof(int32_t),
                              cudaMemcpyDeviceToHost), "copy dspark total routes");

        if (total_routes > 0) {
            std::vector<int32_t> h_counts(static_cast<size_t>(experts_per_rank));
            check_cuda(cudaMemcpy(h_counts.data(), buffers.d_counts,
                                  h_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                       "copy dspark route counts");
            int max_count = 0;
            for (int32_t c : h_counts) max_count = std::max(max_count, static_cast<int>(c));

            // tile_count = 0 selects the padded path. With at most block_size
            // tokens the compact tiling has nothing to save, and the padded
            // path needs no host-built tile table.
            buffers.moe_ws.tile_count = 0;
            if (!moe_prefill_fp4_grouped_cuda_with_workspace(
                    d_x, buffers.d_group_route_tokens, buffers.d_group_route_weights,
                    buffers.d_seg_starts, block->ffn.routed_w1q, block->ffn.routed_w1s,
                    block->ffn.routed_w2q, block->ffn.routed_w2s, block->ffn.routed_w3q,
                    block->ffn.routed_w3s, d_out, rows, topk, total_routes,
                    experts_per_rank, max_count, dim, inter, config.swiglu_limit,
                    buffers.moe_ws, buffers.d_token_slot_routes)) {
                throw std::runtime_error("dspark grouped fp4 moe failed");
            }
        } else {
            check_cuda(cudaMemset(d_out, 0,
                                  static_cast<size_t>(rows) * dim * sizeof(float)),
                       "zero dspark moe out");
        }

        // NOTE: TP>1 needs an all-reduce of d_out here, before the shared
        // expert is added -- each rank only summed its own experts' routes.

        // Shared expert: SwiGLU FFN applied to every token, added on top.
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->ffn.shared_w1_weight,
                                       block->ffn.shared_w1_scale, buffers.d_shared_gate, rows,
                                       inter, dim))
            throw std::runtime_error("dspark shared w1 failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->ffn.shared_w3_weight,
                                       block->ffn.shared_w3_scale, buffers.d_shared_up, rows,
                                       inter, dim))
            throw std::runtime_error("dspark shared w3 failed");
        if (!silu_mul_rows_cuda(buffers.d_shared_gate, buffers.d_shared_up,
                                buffers.d_shared_hidden, rows, inter))
            throw std::runtime_error("dspark shared silu failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_shared_hidden, block->ffn.shared_w2_weight,
                                       block->ffn.shared_w2_scale, buffers.d_shared_out, rows,
                                       dim, inter))
            throw std::runtime_error("dspark shared w2 failed");
        if (!vector_accum_rows_cuda(buffers.d_shared_out, d_out, rows, dim, 1.0f))
            throw std::runtime_error("dspark shared accum failed");
    }

    void forward_block(float* x, const float* d_main_x, int start_pos,
                       const std::vector<int>& draft_input_ids, int block_id) {
        const int bsz = config.block_size;
        const int hc = config.hc_mult;
        const int dim = config.dim;
        const int rows = bsz;  // Process all block_size tokens together

        // Get block-specific weights (one StageBlock per mtp.N stage)
        if (block_id < 0 || block_id >= config.n_stages || block_id >= kNumStages) {
            throw std::runtime_error("DSpark: block_id out of range");
        }
        StageBlock* block = &stages[block_id];

        // 1. Attention path: hc_pre + attn_norm + attention + hc_post

        // 1.1 hc_pre for attention
        // Input: x [rows, hc, dim]
        // Output: x_attn [rows, dim], post_attn [rows, hc], comb_attn [rows, hc, hc]
        bool success = dsv4::hc_pre_float_rows_cuda(
            x,                          // d_h4_rows: [rows, hc, dim]
            block->hc_attn_fn,          // d_fn: [hc, hc*dim]
            block->hc_attn_scale,       // d_scale: [hc]
            block->hc_attn_base,        // d_base: [hc]
            buffers.d_attn_x,           // d_x_rows: [rows, dim]
            buffers.d_attn_post,        // d_post_rows: [rows, hc]
            buffers.d_attn_comb,        // d_comb_rows: [rows, hc, hc]
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_pre attention failed");
        }

        // 1.2 attn_norm: RMSNorm on [rows, dim]
        success = dsv4::rmsnorm_bf16_gamma_rows_cuda(
            buffers.d_attn_x,           // input: [rows, dim]
            block->attn_norm_weight,    // gamma: [dim] BF16
            buffers.d_attn_normed,      // output: [rows, dim]
            rows,
            dim,
            config.norm_eps,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("attn_norm failed");
        }

        // 1.3 DSparkAttention forward. main_x carries the main model's hidden;
        // start_pos is the committed token's position.
        forward_attention(block, buffers.d_attn_normed, d_main_x, start_pos,
                          buffers.d_attn_out);

        // 1.4 hc_post: merge attention output back
        // Output: x [rows, hc, dim]
        success = dsv4::hc_post_float_rows_cuda(
            buffers.d_attn_out,         // d_x_rows: [rows, dim]
            x,                          // d_residual_h4_rows: [rows, hc, dim]
            buffers.d_attn_post,        // d_post_rows: [rows, hc]
            buffers.d_attn_comb,        // d_comb_rows: [rows, hc, hc]
            x,                          // d_y_h4_rows: [rows, hc, dim] (in-place)
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_post attention failed");
        }

        // 2. FFN path: hc_pre + ffn_norm + ffn + hc_post

        // 2.1 hc_pre for FFN
        success = dsv4::hc_pre_float_rows_cuda(
            x,                          // d_h4_rows: [rows, hc, dim]
            block->hc_ffn_fn,           // d_fn: [hc, hc*dim]
            block->hc_ffn_scale,        // d_scale: [hc]
            block->hc_ffn_base,         // d_base: [hc]
            buffers.d_ffn_x,            // d_x_rows: [rows, dim]
            buffers.d_ffn_post,         // d_post_rows: [rows, hc]
            buffers.d_ffn_comb,         // d_comb_rows: [rows, hc, hc]
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_pre FFN failed");
        }

        // 2.2 ffn_norm: RMSNorm on [rows, dim]
        success = dsv4::rmsnorm_bf16_gamma_rows_cuda(
            buffers.d_ffn_x,            // input: [rows, dim]
            block->ffn_norm_weight,     // gamma: [dim] BF16
            buffers.d_ffn_normed,       // output: [rows, dim]
            rows,
            dim,
            config.norm_eps,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("ffn_norm failed");
        }

        // 2.3 FFN forward: shared expert + routed MoE, same structure as a
        // main-model layer. The reference inherits Block's MoE unchanged, so
        // this must match it: sqrt-softplus gate scores, topk by score+bias,
        // weights renormalized from the unbiased scores, times route_scale.
        forward_moe(block, buffers.d_ffn_normed, buffers.d_ffn_out, rows);

        // 2.4 hc_post: merge FFN output back
        success = dsv4::hc_post_float_rows_cuda(
            buffers.d_ffn_out,          // d_x_rows: [rows, dim]
            x,                          // d_residual_h4_rows: [rows, hc, dim]
            buffers.d_ffn_post,         // d_post_rows: [rows, hc]
            buffers.d_ffn_comb,         // d_comb_rows: [rows, hc, hc]
            x,                          // d_y_h4_rows: [rows, hc, dim] (in-place)
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_post FFN failed");
        }
    }

    DraftOutput forward(int input_token, int start_pos,
                       const std::vector<float*>& main_hidden_states) {
        const int bsz = config.block_size;

        // Generate draft_input_ids: [input_token, noise, noise, ...]
        // For now use noise_token_id from config (typically 128000)
        std::vector<int> draft_input_ids(bsz);
        draft_input_ids[0] = input_token;
        for (int i = 1; i < bsz; ++i) {
            draft_input_ids[i] = config.noise_token_id;
        }

        // Stage 0: main_proj + main_norm + embed
        forward_stage0(input_token, main_hidden_states, draft_input_ids);

        // Stage 1-2: DSparkBlock forward. The checkpoint carries one block per
        // `mtp.N.` prefix, so every stage runs — not just the first two.
        // main_x is computed once in stage 0 and shared by every stage, matching
        // DSpark.forward in the reference.
        float* x = buffers.d_draft_x;  // [block_size, hc_mult, dim]
        for (int block_id = 0; block_id < config.n_stages; ++block_id) {
            // forward_block modifies x in-place
            forward_block(x, buffers.d_main_normed, start_pos, draft_input_ids, block_id);
        }

        // Stage 2 heads: markov + confidence (not yet implemented)

        DraftOutput output;
        output.tokens.resize(config.block_size, 0);
        output.confidence.resize(config.block_size, 0.0f);
        return output;
    }
};

// ============================================================================
// DSparkEngine public interface
// ============================================================================

DSparkEngine::DSparkEngine(const char* checkpoint_dir, int tp_rank, int tp_world_size) {
    impl_ = new Impl(checkpoint_dir, tp_rank, tp_world_size);
    try {
        impl_->load_weights();
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

DSparkEngine::~DSparkEngine() {
    delete impl_;
}

DraftOutput DSparkEngine::draft(int input_token, int start_pos,
                                const std::vector<float*>& main_hidden_states) {
    if (main_hidden_states.size() != impl_->config.target_layer_ids.size()) {
        throw std::runtime_error(
            "main_hidden_states size mismatch: expected " +
            std::to_string(impl_->config.target_layer_ids.size()) +
            ", got " + std::to_string(main_hidden_states.size()));
    }

    return impl_->forward(input_token, start_pos, main_hidden_states);
}

const Config& DSparkEngine::config() const {
    return impl_->config;
}

void DSparkEngine::debug_set_kv_cache(int stage_id, const float* h_cache) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_set_kv_cache: stage_id out of range");
    }
    const size_t bytes = static_cast<size_t>(impl_->adims.window_size) *
                         impl_->adims.head_dim * sizeof(float);
    check_cuda(cudaMemcpy(impl_->stages[stage_id].attn.kv_cache, h_cache, bytes,
                          cudaMemcpyHostToDevice),
               "debug_set_kv_cache");
}

void DSparkEngine::debug_attention(int stage_id, const float* h_x, const float* h_main_x,
                                   int start_pos, float* h_out) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_attention: stage_id out of range");
    }
    Impl& impl = *impl_;
    const int bsz = impl.config.block_size;
    const int dim = impl.config.dim;

    // d_attn_normed / d_attn_out are the same scratch the real path uses.
    check_cuda(cudaMemcpy(impl.buffers.d_attn_normed, h_x,
                          static_cast<size_t>(bsz) * dim * sizeof(float),
                          cudaMemcpyHostToDevice),
               "debug_attention x");
    check_cuda(cudaMemcpy(impl.buffers.d_main_normed, h_main_x,
                          static_cast<size_t>(dim) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "debug_attention main_x");

    impl.forward_attention(&impl.stages[stage_id], impl.buffers.d_attn_normed,
                           impl.buffers.d_main_normed, start_pos,
                           impl.buffers.d_attn_out);
    check_cuda(cudaDeviceSynchronize(), "debug_attention sync");

    check_cuda(cudaMemcpy(h_out, impl.buffers.d_attn_out,
                          static_cast<size_t>(bsz) * dim * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "debug_attention out");
}

void DSparkEngine::debug_moe(int stage_id, const float* h_x, int rows, float* h_out) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_moe: stage_id out of range");
    }
    Impl& impl = *impl_;
    const int dim = impl.config.dim;
    if (rows <= 0 || rows > impl.config.block_size) {
        throw std::runtime_error("debug_moe: rows must be in [1, block_size]");
    }
    const size_t n = static_cast<size_t>(rows) * dim;

    check_cuda(cudaMemcpy(impl.buffers.d_ffn_normed, h_x, n * sizeof(float),
                          cudaMemcpyHostToDevice),
               "debug_moe x");
    impl.forward_moe(&impl.stages[stage_id], impl.buffers.d_ffn_normed,
                     impl.buffers.d_ffn_out, rows);
    check_cuda(cudaDeviceSynchronize(), "debug_moe sync");
    check_cuda(cudaMemcpy(h_out, impl.buffers.d_ffn_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "debug_moe out");
}

} // namespace dspark
