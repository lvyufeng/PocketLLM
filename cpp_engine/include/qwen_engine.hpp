#pragma once

#include "qwen_config.hpp"
#include "qwen_dflash2.hpp"
#include "qwen_weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

enum class QwenKvCacheDType {
    Fp16,
    Fp8,
    // TurboQuant K8V4: one combined byte slot per token and KV head holding an
    // FP8 E5M2 key, a 4-bit uniformly quantized value, and the value's FP16
    // scale and minimum. This is a lossy cache and stays opt-in; the FP16 and
    // separate-array FP8 paths remain the exact defaults.
    TurboQuantK8V4,
};

const char* qwen_kv_cache_dtype_name(QwenKvCacheDType dtype);
QwenKvCacheDType parse_qwen_kv_cache_dtype(const std::string& value);

struct QwenEngineOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    // The FP8 projections dequantize the weight into an FP16 scratch buffer once
    // per call, a fixed cost that amortises over the chunk: measured per-call
    // dequant overhead on the real projection shapes is 29-34% at 512 rows but
    // only 9-12% at 2048. Raising the chunk from 512 recovers that on the real
    // 64-layer TP4 checkpoint at a cost of 0.42 GiB/rank, well inside the 22 GB
    // budget, with generated tokens unchanged:
    //
    //   context   chunk 512   chunk 4096   prefill
    //     8192     1125.7      1229.9      1.09x
    //    32768     1059.1      1244.3      1.17x
    //    65536      886.8      1077.7      1.21x
    //
    // The DSV4 engine already defaults to 4096 for the same reason.
    int prefill_chunk_tokens = 4096;
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
    // External Qwen DSpark drafter. Empty keeps the feature disabled. DSpark and
    // native MTP are mutually exclusive because both own the speculative target
    // transaction and hidden-state side channel.
    std::string dspark_checkpoint;
    // External Qwen DFlash2 block-diffusion drafter. This remains opt-in and
    // is mutually exclusive with native MTP and DSpark.
    std::string dflash2_checkpoint;
    std::string nccl_id_path;
    // Stochastic sampling. temperature <= 1e-5 keeps the exact greedy argmax
    // path, so existing greedy results stay reproducible by default. Sampling
    // draws its uniforms on the host and shares them across TP ranks, which is
    // what keeps every rank committing the same token.
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 20;
    unsigned long long sampling_seed = 0;
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

// Accounting for one native-MTP or external-DSpark generate call.
struct QwenMtpStats {
    uint64_t verify_count = 0;
    uint64_t proposed_drafts = 0;
    uint64_t correct_drafts = 0;
    uint64_t rollback_count = 0;
    uint64_t replay_tokens = 0;
    uint64_t confidence_count = 0;
    double confidence_sum = 0.0;
    float confidence_min = 0.0f;
    float confidence_max = 0.0f;
    double prefill_seconds = 0.0;
    double draft_seconds = 0.0;
    double verify_seconds = 0.0;
    double replay_seconds = 0.0;
    // Mean committed tokens per speculative verification, including the target
    // bonus token. This matches DSpark's published spec_accept_length metric.
    double accept_length() const {
        return verify_count == 0
            ? 0.0
            : static_cast<double>(correct_drafts + verify_count) /
                  static_cast<double>(verify_count);
    }

    // Draft-token match ratio only. This excludes the target bonus and must not
    // be compared directly with bonus-inclusive spec_accept_length results.
    double accept_rate() const {
        return proposed_drafts == 0
            ? 0.0
            : static_cast<double>(correct_drafts) /
                  static_cast<double>(proposed_drafts);
    }

    double mean_confidence() const {
        return confidence_count == 0
            ? 0.0 : confidence_sum / static_cast<double>(confidence_count);
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
    uint64_t verify_weight_bytes() const;
    uint64_t activation_workspace_peak_bytes() const;
    uint64_t kv_cache_bytes() const;
    uint64_t kv_cache_scale_bytes() const;

    const QwenPrefixCacheStats& prefix_cache_stats() const {
        return prefix_stats_;
    }
    const QwenMtpStats& mtp_stats() const { return mtp_stats_; }

    void set_dflash2_debug_callback(QwenDFlash2DebugCallback callback);
    // Runs target prefill with the normal native kernels while exporting the
    // captured [row, tap, hidden] DFlash2 feature matrix through the callback.
    // This path does not load the drafter and is therefore usable in TP=1 within
    // one 22 GiB card.
    QwenForwardResult debug_prefill_dflash2(
        const std::vector<int>& token_ids,
        const std::vector<int>& target_layer_ids,
        QwenDFlash2DebugCallback callback);

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
