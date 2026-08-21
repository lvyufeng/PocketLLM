#pragma once

#include "qwen_config.hpp"
#include "qwen_weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

enum class QwenKvCacheDType {
    Fp16,
    Fp8,
};

const char* qwen_kv_cache_dtype_name(QwenKvCacheDType dtype);
QwenKvCacheDType parse_qwen_kv_cache_dtype(const std::string& value);

struct QwenEngineOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    int prefill_chunk_tokens = 512;
    QwenKvCacheDType kv_cache_dtype = QwenKvCacheDType::Fp16;
    // 0 preserves exact full attention. Nonzero values enable the explicit
    // sink-plus-sliding-window attention path in the optimized FP16 kernels.
    int attention_window = 0;
    int attention_sink_tokens = 0;
    // Exact prefix reuse across sequential requests. The full-attention KV
    // cache is already position-indexed, so only the recurrent DeltaNet state
    // has to be carried forward or rolled back.
    bool prefix_cache = true;
    // Recurrent-state snapshots let a diverging prompt roll back to an interior
    // position instead of recomputing from zero. 0 disables snapshots; live
    // continuation of an appended prompt still applies.
    int state_snapshot_interval_tokens = 4096;
    // 82 covers dense 256-token checkpoints through 4K, every 4K boundary
    // through the 262,144-token limit, and a few request-boundary snapshots.
    // This uses about 3.0 GiB/rank for 48-layer Qwen3.8 recurrent snapshots.
    int max_state_snapshots = 82;
    // Native Qwen MTP is opt-in until real TP4 parity/performance validation
    // proves that speculative verification is a win on the target GPU.
    bool mtp = false;
    int mtp_speculative_tokens = 1;
    // When enabled, start with one draft token, double K after full acceptance,
    // and back off after rejection. This protects low-acceptance prompts while
    // quickly reaching the configured maximum on predictable continuations.
    bool mtp_adaptive = false;
    std::string nccl_id_path;
};

struct QwenForwardResult {
    int token = 0;
    int layers = 0;
    int dim = 0;
    int logits = 0;
    int top_token = 0;
    // Filled by native MTP speculative steps; plain forwards leave these zero.
    int correct_drafts = 0;
    int bonus_token = 0;
    std::vector<int> accept_tokens;
    std::vector<float> accept_logits;
    std::vector<float> accept_checksums;
    float top_logit = 0.0f;
    float checksum = 0.0f;
    int position = 0;
};

// Accounting for one native-MTP generate call.
struct QwenMtpStats {
    uint64_t verify_count = 0;
    uint64_t proposed_drafts = 0;
    uint64_t correct_drafts = 0;
    uint64_t rollback_count = 0;
    uint64_t replay_tokens = 0;
    double prefill_seconds = 0.0;
    double draft_seconds = 0.0;
    double verify_seconds = 0.0;
    double replay_seconds = 0.0;

    double accept_rate() const {
        return proposed_drafts == 0
            ? 0.0
            : static_cast<double>(correct_drafts) /
                  static_cast<double>(proposed_drafts);
    }
};

struct QwenPrefixCacheStats {
    // Tokens whose KV cache and recurrent state were reused unchanged.
    int reused_tokens = 0;
    // Tokens actually pushed through the network by the last prefill.
    int computed_tokens = 0;
    // Total prompt length of the last prefill.
    int prompt_tokens = 0;
    // Recurrent state source: "empty", "live", or "snapshot".
    std::string resume_source = "empty";
    // Longest common prefix with the previous prompt, before snapshot rounding.
    int matched_tokens = 0;
    int snapshots = 0;
    uint64_t snapshot_bytes = 0;
    int hits = 0;
    int misses = 0;
};

// Independent Qwen3.5 hybrid dense runtime. Checkpoint BF16 tensors are
// materialized as FP16 on Turing; FP8 projections retain their compressed
// codes and BF16 block scales are uploaded as FP16 for online unpack.
class QwenEngine {
public:
    QwenEngine(const std::string& ckpt_dir, const QwenEngineOptions& options,
               int layer_count = 0, int max_context = 8192);
    ~QwenEngine();

    QwenEngine(const QwenEngine&) = delete;
    QwenEngine& operator=(const QwenEngine&) = delete;

    const QwenConfig& config() const { return config_; }
    const QwenWeightMap& weight_map() const { return weights_; }
    const QwenEngineOptions& options() const { return options_; }
    int max_context() const { return max_context_; }
    int position() const { return position_; }
    uint64_t resident_weight_bytes() const { return resident_weight_bytes_; }
    uint64_t resident_scale_bytes() const { return resident_scale_bytes_; }
    uint64_t activation_workspace_peak_bytes() const;
    uint64_t kv_cache_bytes() const;
    uint64_t kv_cache_scale_bytes() const;

    const QwenPrefixCacheStats& prefix_cache_stats() const {
        return prefix_stats_;
    }
    const QwenMtpStats& mtp_stats() const { return mtp_stats_; }

    void reset();
    // Drops every cached prefix so the next prefill recomputes from zero.
    void clear_prefix_cache();
    void warmup_tp();
    QwenForwardResult prefill(const std::vector<int>& token_ids);
    QwenForwardResult decode_step(int token_id);
    std::vector<QwenForwardResult> generate(const std::vector<int>& prompt_ids,
                                             int max_new_tokens);

private:
    struct Impl;

    std::string ckpt_dir_;
    QwenEngineOptions options_;
    QwenConfig config_;
    SafeTensorsIndex index_;
    QwenWeightMap weights_;
    int active_layers_ = 0;
    int max_context_ = 0;
    int position_ = 0;
    uint64_t resident_weight_bytes_ = 0;
    uint64_t resident_scale_bytes_ = 0;
    QwenPrefixCacheStats prefix_stats_;
    QwenMtpStats mtp_stats_;
    QwenForwardResult cached_result_;
    bool has_cached_result_ = false;
    Impl* impl_ = nullptr;
};

}  // namespace dsv4
