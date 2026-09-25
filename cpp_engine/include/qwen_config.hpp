#pragma once

#include "json_lite.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pocket {

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
    // Native Qwen next-token predictor layers. Zero means the checkpoint has
    // no native MTP branch and the plain runtime remains the only path.
    uint64_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;
    double rms_norm_eps = 1.0e-6;
    double rope_theta = 10000000.0;
    double partial_rotary_factor = 1.0;
    uint64_t fp8_block_size = 128;
    QwenLinearAttentionConfig linear_attention;
    QwenFullAttentionConfig full_attention;
    QwenDenseMlpConfig mlp;
    std::vector<QwenLayerType> layer_types;
    // Tokens that end generation. Read from generation_config.json, whose
    // eos_token_id is a list on the published Qwen3.5 checkpoints
    // ([248046, 248044] on 27B), so this is a set rather than one id. Empty when
    // the checkpoint ships no generation_config.json, which leaves generation
    // bounded only by max_new_tokens as it was before.
    std::vector<int> eos_token_ids;

    bool is_qwen3_5() const;
    uint64_t linear_attention_layers() const;
    uint64_t full_attention_layers() const;
    uint64_t partial_rotary_dim() const;
    std::string layer_type_name(uint64_t layer) const;
    std::string to_string() const;

    // Every invariant a QwenConfig has to satisfy, whichever way it was built.
    // Shared rather than duplicated so the GGUF path cannot be the lax one: the
    // two readers describe the same model, and a field that is required from
    // config.json is required from a GGUF header too.
    void validate() const;

    static QwenConfig from_hf_config(const std::string& ckpt_dir);

    // The same model, read out of a single-file GGUF instead of an HF directory.
    //
    // The GGUF spelling of this architecture is `general.architecture = qwen35`,
    // and its `qwen35.*` keys carry every field config.json does -- with three
    // that have to be derived rather than read: the vocabulary size (from the
    // token table's length), the rotary fraction (rope dimension over head
    // dimension, since the file stores the rotated width in absolute terms), and
    // the layer types (from `full_attention_interval`, the file having no
    // layer_types list).
    static QwenConfig from_gguf(const std::string& path);
};

// Read only the checkpoint's own metadata -- config.json for an HF directory, the
// header for a GGUF. This deliberately does not inspect tensor payloads, so it can
// be used by CLI dispatch before constructing any engine.
bool is_qwen3_5_checkpoint(const std::string& ckpt_path);

// The architecture a GGUF declares, lowercased: `general.architecture`.
std::string gguf_declared_architecture(const std::string& path);

// Fold the spellings of one runtime onto one key. A checkpoint may be a GGUF
// (`qwen35`) or an HF directory (`qwen3_5`, and `qwen3_5_text` inside the
// multimodal wrapper's text_config), and all three name this engine.
std::string canonical_qwen_architecture(const std::string& raw);

}  // namespace pocket
