#pragma once

#include "qwen_config.hpp"
#include "qwen_weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

struct QwenEngineOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    std::string nccl_id_path;
};

struct QwenForwardResult {
    int token = 0;
    int layers = 0;
    int dim = 0;
    int logits = 0;
    int top_token = 0;
    float top_logit = 0.0f;
    float checksum = 0.0f;
    int position = 0;
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

    void reset();
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
    Impl* impl_ = nullptr;
};

}  // namespace dsv4
