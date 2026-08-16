#pragma once

#include "json_lite.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dsv4 {

enum class QwenLayerType {
    LinearAttention,
    FullAttention,
};

struct QwenLinearAttentionConfig {
    uint64_t key_heads = 0;
    uint64_t value_heads = 0;
    uint64_t key_head_dim = 0;
    uint64_t value_head_dim = 0;
    uint64_t conv_kernel_dim = 0;

    uint64_t qkv_dim() const { return 2 * key_heads * key_head_dim + value_heads * value_head_dim; }
    uint64_t value_state_dim() const { return value_heads * value_head_dim; }
};

struct QwenFullAttentionConfig {
    uint64_t num_heads = 0;
    uint64_t num_key_value_heads = 0;
    uint64_t head_dim = 0;
    bool output_gate = false;

    uint64_t attention_dim() const { return num_heads * head_dim; }
    uint64_t kv_dim() const { return num_key_value_heads * head_dim; }
    uint64_t q_dim() const { return attention_dim() * (output_gate ? 2 : 1); }
};

struct QwenDenseMlpConfig {
    uint64_t intermediate_size = 0;
};

struct QwenConfig {
    std::string architecture;
    std::string model_type;
    uint64_t vocab_size = 0;
    uint64_t hidden_size = 0;
    uint64_t num_hidden_layers = 0;
    uint64_t max_position_embeddings = 0;
    double rms_norm_eps = 1.0e-6;
    double rope_theta = 10000000.0;
    double partial_rotary_factor = 1.0;
    uint64_t fp8_block_size = 128;
    QwenLinearAttentionConfig linear_attention;
    QwenFullAttentionConfig full_attention;
    QwenDenseMlpConfig mlp;
    std::vector<QwenLayerType> layer_types;

    bool is_qwen3_5() const;
    uint64_t linear_attention_layers() const;
    uint64_t full_attention_layers() const;
    uint64_t partial_rotary_dim() const;
    std::string layer_type_name(uint64_t layer) const;
    std::string to_string() const;

    static QwenConfig from_hf_config(const std::string& ckpt_dir);
};

// Read only config.json. This deliberately does not inspect tensor files, so it
// can be used by CLI dispatch before constructing either the DSV4 or Qwen engine.
bool is_qwen3_5_checkpoint(const std::string& ckpt_dir);

}  // namespace dsv4
