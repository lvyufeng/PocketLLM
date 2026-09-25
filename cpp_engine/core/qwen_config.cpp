#include "qwen_config.hpp"

#include "gguf_reader.hpp"
#include "json_lite.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pocket {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// generation_config.json is optional: a checkpoint without one simply has no
// stop tokens, so this returns false rather than throwing.
bool try_read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    *out = buffer.str();
    return true;
}

int checked_token_id(const JsonValue& value, uint64_t vocab_size) {
    if (!value.is_number()) {
        throw std::runtime_error("Qwen eos_token_id entry is not a number");
    }
    const double number = value.number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen eos_token_id is not a non-negative integer");
    }
    const uint64_t id = static_cast<uint64_t>(rounded);
    // An out-of-range stop id can never be produced by sampling, so it would
    // silently never fire. Fail loudly instead of shipping a dead stop token.
    if (id >= vocab_size) {
        throw std::runtime_error("Qwen eos_token_id is outside the vocabulary");
    }
    return static_cast<int>(id);
}

// Accepts both shapes HF uses: a bare id, or a list of them.
std::vector<int> parse_eos_token_ids(const std::string& ckpt_dir,
                                    uint64_t vocab_size) {
    std::string text;
    if (!try_read_file(ckpt_dir + "/generation_config.json", &text)) return {};
    const JsonValue root_value = parse_json(text);
    if (!root_value.is_object()) return {};
    const JsonValue* eos = object_get(root_value.object(), "eos_token_id");
    if (eos == nullptr || eos->is_null()) return {};

    std::vector<int> ids;
    if (eos->is_array()) {
        for (const JsonValue& entry : eos->array()) {
            const int id = checked_token_id(entry, vocab_size);
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
        }
    } else {
        ids.push_back(checked_token_id(*eos, vocab_size));
    }
    return ids;
}

const JsonObject& text_config(const JsonObject& root) {
    const JsonValue* nested = object_get(root, "text_config");
    if (nested == nullptr) return root;
    if (!nested->is_object()) throw std::runtime_error("Qwen config text_config must be an object");
    return nested->object();
}

const JsonObject* optional_object(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return nullptr;
    if (!value->is_object()) throw std::runtime_error("Qwen config key must be an object: " + key);
    return &value->object();
}

uint64_t required_u64(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || !value->is_number()) throw std::runtime_error("missing Qwen integer: " + key);
    const double number = value->number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen integer is not integral: " + key);
    }
    return static_cast<uint64_t>(rounded);
}

uint64_t optional_u64(const JsonObject& obj, const std::string& key, uint64_t fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_number()) throw std::runtime_error("Qwen config integer has wrong type: " + key);
    const double number = value->number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen integer is not integral: " + key);
    }
    return static_cast<uint64_t>(rounded);
}

double optional_f64(const JsonObject& obj, const std::string& key, double fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_number()) throw std::runtime_error("Qwen config number has wrong type: " + key);
    return value->number();
}

bool required_bool(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || !value->is_bool()) throw std::runtime_error("missing Qwen boolean: " + key);
    return value->boolean();
}

bool optional_bool(const JsonObject& obj, const std::string& key, bool fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_bool()) throw std::runtime_error("Qwen config boolean has wrong type: " + key);
    return value->boolean();
}

std::string optional_string(const JsonObject& obj, const std::string& key, const std::string& fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_string()) throw std::runtime_error("Qwen config string has wrong type: " + key);
    return value->string();
}

std::string root_model_type(const JsonObject& root) {
    return optional_string(root, "model_type", "");
}

bool has_gguf_suffix(const std::string& path) {
    static const std::string suffix = ".gguf";
    if (path.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), path.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}

std::string lowered(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

// A required integer out of the GGUF header, named in the error the way the key
// is so a missing key is a one-line fix rather than a hunt.
uint64_t gguf_required_u64(const GGUFFile& file, const std::string& key) {
    const std::optional<uint64_t> value = file.metadata_u64(key);
    if (!value) throw std::runtime_error("GGUF config is missing key: " + key);
    return *value;
}

double gguf_required_f64(const GGUFFile& file, const std::string& key) {
    const std::optional<double> value = file.metadata_f64(key);
    if (!value) throw std::runtime_error("GGUF config is missing key: " + key);
    return *value;
}

void validate_positive(double value, const std::string& name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error("invalid Qwen positive number: " + name);
    }
}

}  // namespace

QwenConfig QwenConfig::from_hf_config(const std::string& ckpt_dir) {
    const JsonValue root_value = parse_json(read_file(ckpt_dir + "/config.json"));
    if (!root_value.is_object()) throw std::runtime_error("Qwen config root must be an object");
    const JsonObject& root = root_value.object();
    const JsonObject& text = text_config(root);

    QwenConfig cfg;
    cfg.architecture = root_model_type(root);
    cfg.model_type = optional_string(text, "model_type", cfg.architecture);
    cfg.vocab_size = required_u64(text, "vocab_size");
    cfg.hidden_size = required_u64(text, "hidden_size");
    cfg.num_hidden_layers = required_u64(text, "num_hidden_layers");
    cfg.max_position_embeddings = required_u64(text, "max_position_embeddings");
    cfg.mtp_num_hidden_layers = optional_u64(text, "mtp_num_hidden_layers", 0);
    cfg.mtp_use_dedicated_embeddings = optional_bool(
        text, "mtp_use_dedicated_embeddings", false);
    cfg.rms_norm_eps = optional_f64(text, "rms_norm_eps", 1.0e-6);
    cfg.partial_rotary_factor = optional_f64(text, "partial_rotary_factor", 1.0);
    cfg.fp8_block_size = 128;

    const JsonObject* rope_parameters = optional_object(text, "rope_parameters");
    if (rope_parameters != nullptr) {
        cfg.rope_theta = optional_f64(*rope_parameters, "rope_theta", cfg.rope_theta);
        cfg.partial_rotary_factor = optional_f64(*rope_parameters, "partial_rotary_factor", cfg.partial_rotary_factor);
    } else {
        cfg.rope_theta = optional_f64(text, "rope_theta", cfg.rope_theta);
    }

    cfg.linear_attention.key_heads = required_u64(text, "linear_num_key_heads");
    cfg.linear_attention.value_heads = required_u64(text, "linear_num_value_heads");
    cfg.linear_attention.key_head_dim = required_u64(text, "linear_key_head_dim");
    cfg.linear_attention.value_head_dim = required_u64(text, "linear_value_head_dim");
    cfg.linear_attention.conv_kernel_dim = required_u64(text, "linear_conv_kernel_dim");

    cfg.full_attention.num_heads = required_u64(text, "num_attention_heads");
    cfg.full_attention.num_key_value_heads = required_u64(text, "num_key_value_heads");
    cfg.full_attention.head_dim = optional_u64(text, "head_dim", cfg.hidden_size / cfg.full_attention.num_heads);
    cfg.full_attention.output_gate = required_bool(text, "attn_output_gate");

    cfg.mlp.intermediate_size = optional_u64(text, "intermediate_size", 0);
    if (cfg.mlp.intermediate_size == 0) {
        throw std::runtime_error("missing Qwen dense MLP intermediate_size");
    }

    const JsonValue* raw_layers = object_get(text, "layer_types");
    if (raw_layers == nullptr || !raw_layers->is_array() || raw_layers->array().size() != cfg.num_hidden_layers) {
        throw std::runtime_error("Qwen layer_types must contain one entry per layer");
    }
    cfg.layer_types.reserve(cfg.num_hidden_layers);
    for (size_t i = 0; i < raw_layers->array().size(); ++i) {
        const JsonValue& value = raw_layers->array()[i];
        if (!value.is_string()) throw std::runtime_error("Qwen layer_types entry is not a string");
        if (value.string() == "linear_attention") {
            cfg.layer_types.push_back(QwenLayerType::LinearAttention);
        } else if (value.string() == "full_attention") {
            cfg.layer_types.push_back(QwenLayerType::FullAttention);
        } else {
            throw std::runtime_error("unsupported Qwen layer type at " + std::to_string(i) + ": " + value.string());
        }
    }

    cfg.eos_token_ids = parse_eos_token_ids(ckpt_dir, cfg.vocab_size);

    if (!cfg.is_qwen3_5()) throw std::runtime_error("config is not a Qwen3.5 text checkpoint");
    cfg.validate();
    return cfg;
}

QwenConfig QwenConfig::from_gguf(const std::string& path) {
    const GGUFFile file(path);
    std::string declared = lowered(file.metadata_string("general.architecture").value_or(std::string()));
    if (declared.empty()) {
        throw std::runtime_error("GGUF config declares no general.architecture: " + path);
    }
    // The metadata keys are prefixed by the *declared* name, not by the
    // canonical one: this file says `qwen35.`, and folding that to `qwen3_5`
    // before building the keys would look for `qwen3_5.block_count`.
    const auto key = [&declared](const char* suffix) { return declared + "." + suffix; };

    QwenConfig cfg;
    cfg.architecture = canonical_qwen_architecture(declared);
    cfg.model_type = cfg.architecture;
    // The vocabulary is not a metadata key -- it is the length of the token
    // table, which the metadata array reader keeps as a value rather than as a
    // count.
    const auto tokens = file.metadata_array_length("tokenizer.ggml.tokens");
    if (!tokens || *tokens == 0) {
        throw std::runtime_error("GGUF config has no tokenizer.ggml.tokens table");
    }
    cfg.vocab_size = *tokens;

    cfg.hidden_size = gguf_required_u64(file, key("embedding_length"));
    cfg.num_hidden_layers = gguf_required_u64(file, key("block_count"));
    cfg.max_position_embeddings = gguf_required_u64(file, key("context_length"));
    // The released artifact carries no next-token-predictor branch, while the
    // FP8 checkpoint this shares config with declares one; a GGUF header has no
    // key for it either way, so the plain runtime is the only path here.
    cfg.mtp_num_hidden_layers = 0;
    cfg.mtp_use_dedicated_embeddings = false;
    cfg.rms_norm_eps = gguf_required_f64(file, key("attention.layer_norm_rms_epsilon"));
    cfg.rope_theta = gguf_required_f64(file, key("rope.freq_base"));

    const uint64_t head_length = gguf_required_u64(file, key("attention.key_length"));
    const uint64_t value_length = gguf_required_u64(file, key("attention.value_length"));
    if (head_length != value_length) {
        throw std::runtime_error(
            "GGUF config declares different key and value head lengths; this runtime has one head_dim");
    }
    // config.json states the rotary fraction directly; the GGUF states the
    // rotated width in absolute terms, so the fraction is the ratio. Deriving it
    // rather than assuming 0.25 keeps a checkpoint that changes the ratio from
    // silently rotating the wrong number of dimensions.
    const uint64_t rotary_dims = gguf_required_u64(file, key("rope.dimension_count"));
    if (rotary_dims == 0 || head_length == 0) {
        throw std::runtime_error("GGUF config declares a zero rotary or head dimension");
    }
    cfg.partial_rotary_factor =
        static_cast<double>(rotary_dims) / static_cast<double>(head_length);

    cfg.linear_attention.key_heads = gguf_required_u64(file, key("ssm.group_count"));
    cfg.linear_attention.key_head_dim = gguf_required_u64(file, key("ssm.state_size"));
    cfg.linear_attention.value_head_dim = cfg.linear_attention.key_head_dim;
    cfg.linear_attention.conv_kernel_dim = gguf_required_u64(file, key("ssm.conv_kernel"));
    // The value head count is the time-step rank, and the file also states the
    // product of the two as `ssm.inner_size`. Requiring them to agree is the one
    // cheap cross-check that catches this mapping being off by a key.
    const uint64_t value_heads = gguf_required_u64(file, key("ssm.time_step_rank"));
    const uint64_t inner_size = gguf_required_u64(file, key("ssm.inner_size"));
    if (value_heads * cfg.linear_attention.value_head_dim != inner_size) {
        throw std::runtime_error(
            "GGUF config's ssm.time_step_rank and ssm.inner_size disagree about the value width");
    }
    cfg.linear_attention.value_heads = value_heads;

    cfg.full_attention.num_heads = gguf_required_u64(file, key("attention.head_count"));
    cfg.full_attention.num_key_value_heads =
        gguf_required_u64(file, key("attention.head_count_kv"));
    cfg.full_attention.head_dim = head_length;

    // config.json lists one type per layer; the GGUF states only the period at
    // which a full-attention layer replaces a linear one. The published
    // checkpoint's list is exactly that pattern -- its sixteen `full_attention`
    // entries are layers 3, 7, ... 63 -- so the period reproduces it, and the
    // period is what the file can be held to.
    const uint64_t interval = gguf_required_u64(file, key("full_attention_interval"));
    if (interval == 0 || cfg.num_hidden_layers == 0) {
        throw std::runtime_error("GGUF config declares a zero full-attention interval");
    }
    cfg.layer_types.reserve(cfg.num_hidden_layers);
    for (uint64_t layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        cfg.layer_types.push_back((layer % interval) == (interval - 1)
                                      ? QwenLayerType::FullAttention
                                      : QwenLayerType::LinearAttention);
    }

    // No key states whether the query projection carries a gate, and the file
    // does not need one: the projection's own output width answers it, since a
    // gated query is exactly twice the attention width and an ungated one is
    // exactly once. Reading it off the first full-attention layer's tensor costs
    // a header lookup and cannot drift from what the weights actually hold.
    const uint64_t attention_dim = cfg.full_attention.num_heads * head_length;
    {
        const GGUFTensorInfo* q = nullptr;
        for (uint64_t layer = 0; layer < cfg.num_hidden_layers && q == nullptr; ++layer) {
            if (cfg.layer_types[layer] != QwenLayerType::FullAttention) continue;
            q = file.find_tensor("blk." + std::to_string(layer) + ".attn_q.weight");
        }
        if (q == nullptr) {
            throw std::runtime_error(
                "GGUF config has no blk.*.attn_q.weight to read the query width from");
        }
        if (q->shape.size() != 2 || q->shape[1] % attention_dim != 0) {
            throw std::runtime_error(
                "GGUF config's attn_q.weight does not tile the declared head count");
        }
        const uint64_t factor = q->shape[1] / attention_dim;
        if (factor != 1 && factor != 2) {
            throw std::runtime_error(
                "GGUF config's attn_q.weight is neither the attention width nor twice it");
        }
        cfg.full_attention.output_gate = factor == 2;
    }

    cfg.mlp.intermediate_size = gguf_required_u64(file, key("feed_forward_length"));

    // Both stop tokens the checkpoint declares, in the order its own
    // generation_config.json lists them: the end-of-turn token first, then the
    // end-of-text one it also stops on.
    const std::optional<uint64_t> eos = file.metadata_u64("tokenizer.ggml.eos_token_id");
    const std::optional<uint64_t> bos = file.metadata_u64("tokenizer.ggml.bos_token_id");
    if (eos) cfg.eos_token_ids.push_back(static_cast<int>(*eos));
    if (bos && (!eos || *bos != *eos)) cfg.eos_token_ids.push_back(static_cast<int>(*bos));

    if (!cfg.is_qwen3_5()) {
        throw std::runtime_error("GGUF config is not a Qwen3.5 text checkpoint: " + path);
    }
    cfg.validate();
    return cfg;
}

void QwenConfig::validate() const {
    if (hidden_size == 0 || num_hidden_layers == 0 || vocab_size == 0 ||
        max_position_embeddings == 0) {
        throw std::runtime_error("invalid zero-sized Qwen config");
    }
    if (mlp.intermediate_size == 0) {
        throw std::runtime_error("missing Qwen dense MLP intermediate_size");
    }
    validate_positive(rms_norm_eps, "rms_norm_eps");
    validate_positive(rope_theta, "rope_theta");
    if (!(partial_rotary_factor > 0.0 && partial_rotary_factor <= 1.0)) {
        throw std::runtime_error("Qwen partial_rotary_factor must be in (0, 1]");
    }
    if (linear_attention.key_heads == 0 || linear_attention.value_heads == 0 ||
        linear_attention.key_head_dim == 0 || linear_attention.value_head_dim == 0 ||
        linear_attention.conv_kernel_dim == 0) {
        throw std::runtime_error("invalid Qwen linear-attention dimensions");
    }
    if (full_attention.num_heads == 0 || full_attention.num_key_value_heads == 0 || full_attention.head_dim == 0) {
        throw std::runtime_error("invalid Qwen full-attention dimensions");
    }
    if (full_attention.num_heads % full_attention.num_key_value_heads != 0) {
        throw std::runtime_error("Qwen attention heads must be divisible by key/value heads");
    }
    if (partial_rotary_dim() == 0 || partial_rotary_dim() > full_attention.head_dim ||
        (partial_rotary_dim() % 2) != 0) {
        throw std::runtime_error("Qwen partial rotary dimension must be a positive even head dimension");
    }
    if (linear_attention.value_heads % linear_attention.key_heads != 0) {
        throw std::runtime_error("Qwen linear value heads must be divisible by key heads");
    }
    if (layer_types.size() != num_hidden_layers) {
        throw std::runtime_error("Qwen layer_types must contain one entry per layer");
    }
}

bool QwenConfig::is_qwen3_5() const {
    return architecture == "qwen3_5" || architecture == "qwen3_5_text" ||
           model_type == "qwen3_5" || model_type == "qwen3_5_text";
}

uint64_t QwenConfig::linear_attention_layers() const {
    uint64_t count = 0;
    for (QwenLayerType type : layer_types) {
        if (type == QwenLayerType::LinearAttention) ++count;
    }
    return count;
}

uint64_t QwenConfig::full_attention_layers() const {
    uint64_t count = 0;
    for (QwenLayerType type : layer_types) {
        if (type == QwenLayerType::FullAttention) ++count;
    }
    return count;
}

uint64_t QwenConfig::partial_rotary_dim() const {
    return static_cast<uint64_t>(std::llround(static_cast<double>(full_attention.head_dim) * partial_rotary_factor));
}

std::string QwenConfig::layer_type_name(uint64_t layer) const {
    if (layer >= layer_types.size()) throw std::out_of_range("Qwen layer index out of range");
    return layer_types[layer] == QwenLayerType::LinearAttention ? "linear_attention" : "full_attention";
}

std::string QwenConfig::to_string() const {
    std::ostringstream out;
    out << "architecture=" << architecture << '\n'
        << "model_type=" << model_type << '\n'
        << "vocab_size=" << vocab_size << '\n'
        << "hidden_size=" << hidden_size << '\n'
        << "num_hidden_layers=" << num_hidden_layers << '\n'
        << "max_position_embeddings=" << max_position_embeddings << '\n'
        << "mtp_num_hidden_layers=" << mtp_num_hidden_layers << '\n'
        << "mtp_use_dedicated_embeddings=" << (mtp_use_dedicated_embeddings ? 1 : 0) << '\n'
        << "rms_norm_eps=" << rms_norm_eps << '\n'
        << "rope_theta=" << rope_theta << '\n'
        << "partial_rotary_factor=" << partial_rotary_factor << '\n'
        << "partial_rotary_dim=" << partial_rotary_dim() << '\n'
        << "linear_key_heads=" << linear_attention.key_heads << '\n'
        << "linear_value_heads=" << linear_attention.value_heads << '\n'
        << "linear_key_head_dim=" << linear_attention.key_head_dim << '\n'
        << "linear_value_head_dim=" << linear_attention.value_head_dim << '\n'
        << "linear_conv_kernel_dim=" << linear_attention.conv_kernel_dim << '\n'
        << "full_num_heads=" << full_attention.num_heads << '\n'
        << "full_num_key_value_heads=" << full_attention.num_key_value_heads << '\n'
        << "full_head_dim=" << full_attention.head_dim << '\n'
        << "attn_output_gate=" << (full_attention.output_gate ? 1 : 0) << '\n'
        << "intermediate_size=" << mlp.intermediate_size << '\n'
        << "linear_attention_layers=" << linear_attention_layers() << '\n'
        << "full_attention_layers=" << full_attention_layers() << '\n';
    return out.str();
}

std::string gguf_declared_architecture(const std::string& path) {
    const GGUFFile file(path);
    return lowered(file.metadata_string("general.architecture").value_or(std::string()));
}

std::string canonical_qwen_architecture(const std::string& raw) {
    if (raw == "qwen3_5_text") return "qwen3_5";
    // llama.cpp's name for this family, which is what the ternary checkpoint's
    // header declares. It is the same 64-block hybrid model field for field, so
    // it has to reach the same engine key -- and its own file is the only place
    // that spelling appears.
    if (raw == "qwen35") return "qwen3_5";
    return raw;
}

bool is_qwen3_5_checkpoint(const std::string& ckpt_path) {
    if (has_gguf_suffix(ckpt_path)) {
        return canonical_qwen_architecture(gguf_declared_architecture(ckpt_path)) == "qwen3_5";
    }
    const JsonValue root_value = parse_json(read_file(ckpt_path + "/config.json"));
    if (!root_value.is_object()) return false;
    const JsonObject& root = root_value.object();
    const std::string arch = root_model_type(root);
    if (arch == "qwen3_5" || arch == "qwen3_5_text") return true;
    const JsonValue* nested = object_get(root, "text_config");
    if (nested == nullptr || !nested->is_object()) return false;
    const std::string type = optional_string(nested->object(), "model_type", "");
    return type == "qwen3_5" || type == "qwen3_5_text";
}

}  // namespace pocket
