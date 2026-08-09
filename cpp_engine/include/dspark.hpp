#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace dspark {

// DSpark draft output: drafted tokens + confidence scores
struct DraftOutput {
    std::vector<int> tokens;        // [block_size+1]: input token + drafted tokens
    std::vector<float> confidence;  // [block_size]: per-draft-position confidence

    DraftOutput() = default;
    DraftOutput(int block_size) : tokens(block_size + 1), confidence(block_size) {}
};

// DSpark configuration loaded from config.json
struct Config {
    int block_size = 5;                      // Number of tokens to draft per round
    int noise_token_id = 128799;             // Placeholder token for draft positions 1-4
    std::vector<int> target_layer_ids = {40, 41, 42};  // Main model layers to concat
    int markov_rank = 256;                   // Bigram bias embedding rank
    int window_size = 128;                   // Attention sliding window size
    int dim = 4096;                          // Hidden dimension
    int vocab_size = 129280;                 // Vocabulary size
    int hc_mult = 4;                         // HC expansion multiplier
    float norm_eps = 1e-6f;                  // RMSNorm epsilon

    // Attention geometry (global, before TP sharding)
    int n_heads = 64;                        // num_attention_heads
    int head_dim = 512;                      // head_dim
    int q_lora_rank = 1024;                  // wq_a output dim
    int o_lora_rank = 1024;                  // per-group rank for wo_a
    int o_groups = 8;                        // output groups
    int rope_dim = 64;                       // qk_rope_head_dim
    float rope_theta = 10000.0f;

    // MoE geometry
    int n_experts = 256;                     // n_routed_experts
    int topk = 6;                            // num_experts_per_tok
    int moe_inter = 2048;                    // moe_intermediate_size
    float route_scale = 1.5f;                // routed_scaling_factor
    float swiglu_limit = 10.0f;

    int n_stages = 3;                        // mtp.0 / mtp.1 / mtp.2

    // Load from config.json
    static Config from_json(const char* config_path);
};

// DSpark 3-stage draft engine
// Stage 0: main_proj + main_norm + Block (attn + ffn)
// Stage 1: Block (attn + ffn)
// Stage 2: Block (attn + ffn) + norm + hc_head + markov_head + confidence_head
class DSparkEngine {
public:
    // Initialize DSpark engine from checkpoint directory
    // checkpoint_dir: path to safetensors checkpoint (e.g., /path/to/model/)
    // tp_rank, tp_world_size: tensor parallelism config
    DSparkEngine(const char* checkpoint_dir, int tp_rank, int tp_world_size);
    ~DSparkEngine();

    // Draft block_size tokens from a committed token
    // input_token: the committed token at position start_pos
    // start_pos: position index of the input token
    // main_hidden_states: hidden states from main model's target layers [n_target][dim]
    //                     (layers 40, 41, 42 for DeepSeek-V4-Flash-0731)
    // Returns: DraftOutput with drafted tokens and confidence scores
    DraftOutput draft(int input_token, int start_pos,
                     const std::vector<float*>& main_hidden_states);

    const Config& config() const;

    // ---- Test hooks -------------------------------------------------------
    // These exist so individual sub-paths can be compared against the PyTorch
    // reference before the full draft loop works. Nothing in the normal draft
    // path calls them.

    // Overwrite one stage's attention KV ring cache. h_cache is
    // [window_size, head_dim] in row-major order.
    void debug_set_kv_cache(int stage_id, const float* h_cache);

    // Run one stage's DSparkAttention in isolation.
    //   h_x      [block_size, dim]  attn_norm output for the draft tokens
    //   h_main_x [dim]              main model's projected + normed hidden
    //   start_pos                   position of the committed token (> 0)
    //   h_out    [block_size, dim]  caller-allocated output
    // Mutates the stage's ring cache, exactly as the real forward does.
    void debug_attention(int stage_id, const float* h_x, const float* h_main_x,
                         int start_pos, float* h_out);

private:
    struct Impl;
    Impl* impl_;

    // Non-copyable
    DSparkEngine(const DSparkEngine&) = delete;
    DSparkEngine& operator=(const DSparkEngine&) = delete;
};

} // namespace dspark
