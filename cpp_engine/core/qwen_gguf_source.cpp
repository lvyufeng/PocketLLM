// The GGUF container, as a Qwen checkpoint source. This translation unit is
// device-agnostic: it reads a header and addresses bytes, and nothing here needs
// an accelerator toolkit, so an artifact can be audited on a machine that has
// none.

#include "qwen_gguf.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace pocket {
namespace {

// One canonical-to-GGUF name pair, and whether the tensor is the depthwise
// convolution -- the one tensor whose GGUF spelling is a *different rank* rather
// than a different word, because a GGML tensor is two-dimensional and the
// checkpoint's `[channels, 1, kernel]` is a GGML `{kernel, channels}`.
struct NameRule {
    const char* canonical;
    const char* gguf;
    bool conv_kernel;
};

// The whole of the naming difference between the two spellings. Every entry is a
// suffix: the layer prefix is rewritten once by the caller, so a mapping that
// grows a layer does not grow this table.
const NameRule kNameRules[] = {
    {"input_layernorm.weight", "attn_norm.weight", false},
    {"post_attention_layernorm.weight", "post_attention_norm.weight", false},
    {"self_attn.q_proj.weight", "attn_q.weight", false},
    {"self_attn.k_proj.weight", "attn_k.weight", false},
    {"self_attn.v_proj.weight", "attn_v.weight", false},
    {"self_attn.o_proj.weight", "attn_output.weight", false},
    {"self_attn.q_norm.weight", "attn_q_norm.weight", false},
    {"self_attn.k_norm.weight", "attn_k_norm.weight", false},
    {"linear_attn.in_proj_qkv.weight", "attn_qkv.weight", false},
    {"linear_attn.in_proj_z.weight", "attn_gate.weight", false},
    {"linear_attn.out_proj.weight", "ssm_out.weight", false},
    {"linear_attn.in_proj_a.weight", "ssm_alpha.weight", false},
    {"linear_attn.in_proj_b.weight", "ssm_beta.weight", false},
    {"linear_attn.conv1d.weight", "ssm_conv1d.weight", true},
    {"linear_attn.A_log", "ssm_a", false},
    {"linear_attn.dt_bias", "ssm_dt.bias", false},
    {"linear_attn.norm.weight", "ssm_norm.weight", false},
    {"mlp.gate_proj.weight", "ffn_gate.weight", false},
    {"mlp.up_proj.weight", "ffn_up.weight", false},
    {"mlp.down_proj.weight", "ffn_down.weight", false},
};

// The three tensors with no layer prefix. They are not suffixes of anything, so
// they are matched whole.
struct TrunkRule {
    const char* canonical;
    const char* gguf;
};

const TrunkRule kTrunkRules[] = {
    {"model.language_model.embed_tokens.weight", "token_embd.weight"},
    {"model.language_model.norm.weight", "output_norm.weight"},
    {"lm_head.weight", "output.weight"},
};

const NameRule* find_rule_by_canonical(const std::string& canonical,
                                       size_t* match_length) {
    const NameRule* best = nullptr;
    size_t best_length = 0;
    for (const NameRule& rule : kNameRules) {
        const std::string suffix(rule.canonical);
        if (canonical.size() < suffix.size()) continue;
        if (canonical.compare(canonical.size() - suffix.size(), suffix.size(),
                              suffix) != 0) {
            continue;
        }
        // The longest match wins, so that a suffix which is the tail of another
        // cannot shadow it. Nothing in the table nests today; the rule is here
        // because a silent shadow would be a wrong tensor rather than an error.
        if (suffix.size() > best_length) {
            best = &rule;
            best_length = suffix.size();
        }
    }
    *match_length = best_length;
    return best;
}

bool is_layer_prefix(const std::string& canonical, const char* prefix,
                     size_t prefix_length, uint64_t* layer) {
    if (canonical.size() < prefix_length) return false;
    if (canonical.compare(0, prefix_length, prefix) != 0) return false;
    const size_t digits_begin = prefix_length;
    size_t cursor = digits_begin;
    uint64_t value = 0;
    while (cursor < canonical.size() && canonical[cursor] >= '0' &&
           canonical[cursor] <= '9') {
        value = value * 10 + static_cast<uint64_t>(canonical[cursor] - '0');
        ++cursor;
    }
    if (cursor == digits_begin) return false;
    if (cursor >= canonical.size() || canonical[cursor] != '.') return false;
    *layer = value;
    return true;
}

// SafeDType is what the weight map speaks; DType is what the reader speaks. The
// two are not the same enumeration because one of them carries the GGML block
// formats and the other does not.
SafeDType safe_dtype_of(const GGUFTensorInfo& info, bool* ternary) {
    *ternary = false;
    switch (info.dtype) {
        case DType::F32: return SafeDType::F32;
        case DType::F16: return SafeDType::F16;
        case DType::BF16: return SafeDType::BF16;
        case DType::PTQ1_0:
            // The three-valued pack. Its storage has no element size, so the map
            // is told U8 and told separately that a row is blocks rather than
            // elements; nothing downstream will try to read one as a number.
            *ternary = true;
            return SafeDType::U8;
        default:
            throw std::runtime_error(
                "the ternary checkpoint holds a tensor type this runtime has no "
                "policy for: " + info.name + " is " + dtype_name(info.dtype));
    }
}

// The canonical shape of a tensor the reader has just found. A GGUF states a
// matrix input-first where every other file this runtime reads states it
// output-first, and the convolution loses its singleton middle axis on the way
// through a format that has only two dimensions.
std::vector<uint64_t> canonical_shape(const GGUFTensorInfo& info, bool conv) {
    if (conv) {
        if (info.shape.size() != 2) {
            throw std::runtime_error("the GGUF convolution is not two-dimensional: " +
                                     info.name);
        }
        return {info.shape[1], 1, info.shape[0]};
    }
    if (info.shape.size() <= 1) return info.shape;
    std::vector<uint64_t> reversed(info.shape.rbegin(), info.shape.rend());
    return reversed;
}

bool has_suffix(const std::string& value, const char* suffix) {
    const std::string tail(suffix);
    return value.size() >= tail.size() &&
           value.compare(value.size() - tail.size(), tail.size(), tail) == 0;
}

}  // namespace

std::string qwen_gguf_tensor_name(const std::string& canonical_name) {
    for (const TrunkRule& rule : kTrunkRules) {
        if (canonical_name == rule.canonical) return rule.gguf;
    }
    const std::string layer_prefix = "model.language_model.layers.";
    uint64_t layer = 0;
    if (!is_layer_prefix(canonical_name, layer_prefix.c_str(),
                         layer_prefix.size(), &layer)) {
        return std::string();
    }
    size_t match_length = 0;
    const NameRule* rule = find_rule_by_canonical(canonical_name, &match_length);
    if (rule == nullptr) return std::string();
    // The layer prefix is a fixed number of characters before the suffix, and
    // the digits in between are the layer index.
    const size_t digits_end = canonical_name.size() - match_length - 1;
    if (digits_end <= layer_prefix.size()) return std::string();
    return "blk." + canonical_name.substr(layer_prefix.size(),
                                          digits_end - layer_prefix.size()) +
           "." + rule->gguf;
}

std::string qwen_canonical_tensor_name(const std::string& gguf_name) {
    for (const TrunkRule& rule : kTrunkRules) {
        if (gguf_name == rule.gguf) return rule.canonical;
    }
    const std::string layer_prefix = "blk.";
    uint64_t layer = 0;
    if (!is_layer_prefix(gguf_name, layer_prefix.c_str(), layer_prefix.size(),
                         &layer)) {
        return gguf_name;
    }
    const size_t digits_end = gguf_name.find('.', layer_prefix.size());
    if (digits_end == std::string::npos) return gguf_name;
    const std::string suffix = gguf_name.substr(digits_end + 1);
    for (const NameRule& rule : kNameRules) {
        if (suffix == rule.gguf) {
            return "model.language_model.layers." +
                   gguf_name.substr(layer_prefix.size(),
                                    digits_end - layer_prefix.size()) +
                   "." + rule.canonical;
        }
    }
    return gguf_name;
}

QwenGgufSource::QwenGgufSource(const std::string& path) : file_(path) {
    if (file_.tensor_count() == 0) {
        throw std::runtime_error("the GGUF has no tensors: " + path);
    }
}

QwenSourceTensor QwenGgufSource::lookup(const std::string& canonical_name) const {
    QwenSourceTensor out;
    const std::string name = qwen_gguf_tensor_name(canonical_name);
    if (name.empty()) return out;
    const GGUFTensorInfo* info = file_.find_tensor(name);
    if (info == nullptr) return out;
    out.name = canonical_name;
    out.dtype = safe_dtype_of(*info, &out.ternary_blocks);
    out.shape = canonical_shape(*info, has_suffix(canonical_name, "conv1d.weight"));
    out.shard_name = file_.path();
    out.present = true;
    return out;
}

const uint8_t* QwenGgufSource::data_of(const QwenSourceTensor& tensor) const {
    const std::string name = qwen_gguf_tensor_name(tensor.name);
    const GGUFTensorInfo* info = file_.find_tensor(name);
    if (info == nullptr) {
        throw std::runtime_error("the GGUF tensor vanished: " + name);
    }
    return file_.bytes() + info->absolute_offset;
}

std::vector<std::string> QwenGgufSource::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(file_.tensors().size());
    for (const GGUFTensorInfo& info : file_.tensors()) {
        names.push_back(qwen_canonical_tensor_name(info.name));
    }
    return names;
}

size_t QwenGgufSource::tensor_count() const {
    return static_cast<size_t>(file_.tensor_count());
}

bool QwenGgufSource::accepts_storage_dtype(SafeDType declared, SafeDType actual,
                                           bool ternary) const {
    const bool declared_element = declared == SafeDType::BF16 ||
                                  declared == SafeDType::F16;
    if (!declared_element) return declared == actual;
    // The three widths this conversion used, in the places it used them: fp32
    // for the norms and the gated-DeltaNet scalars, bf16 for the two per-head
    // projections, and the ternary pack for every weight that is not a table.
    // The list is short on purpose -- a tensor of some other type landing in one
    // of these slots is a conversion bug, not a container difference.
    if (actual == SafeDType::BF16) return true;
    if (actual == SafeDType::F32) return true;
    return ternary && actual == SafeDType::U8;
}

SafeDType QwenGgufSource::device_dtype(SafeDType storage_dtype) const {
    // The kernels take fp16 gamma and fp16 activations, and this checkpoint has
    // no element format that is usable as stored: fp32 norms are twice the bytes
    // and fp16 is what the norm kernel reads anyway. A ternary linear never
    // reaches here -- its storage dtype is U8 and a U8 device tensor is exactly
    // what the GEMM wants, so it stays as it is.
    switch (storage_dtype) {
        case SafeDType::F32:
        case SafeDType::BF16:
        case SafeDType::F16:
            return SafeDType::F16;
        default:
            return storage_dtype;
    }
}

const QwenGgufSource::ValueHeads& QwenGgufSource::value_heads() const {
    if (!value_heads_.has_value()) {
        ValueHeads heads;
        const auto group_count = file_.metadata_u64("qwen35.ssm.group_count");
        const auto state_size = file_.metadata_u64("qwen35.ssm.state_size");
        const auto time_step_rank = file_.metadata_u64("qwen35.ssm.time_step_rank");
        const auto inner_size = file_.metadata_u64("qwen35.ssm.inner_size");
        if (!group_count || !state_size || !time_step_rank || !inner_size ||
            *group_count == 0 || *state_size == 0 || *time_step_rank == 0 ||
            *time_step_rank % *group_count != 0 ||
            *inner_size % *time_step_rank != 0) {
            throw std::runtime_error(
                "the ternary checkpoint does not state a gated-DeltaNet head "
                "geometry this mapping can be read against: " + file_.path());
        }
        heads.key_heads = *group_count;
        heads.key_head_dim = *state_size;
        heads.value_heads = *time_step_rank;
        // `inner_size` is the value width the scan produces, so one value head is
        // that over the number of value heads.
        heads.value_head_dim = *inner_size / *time_step_rank;
        value_heads_ = heads;
    }
    return *value_heads_;
}

const QwenHadamardSpec* QwenGgufSource::hadamard() const {
    if (!hadamard_parsed_) {
        // The file's own name list, so that a declared tensor the file does not
        // hold is an error here rather than a rotation applied to a tensor that
        // was never folded.
        hadamard_ = QwenHadamardSpec::find(file_, tensor_names());
        hadamard_parsed_ = true;
    }
    return hadamard_.has_value() ? &*hadamard_ : nullptr;
}

std::vector<uint64_t> QwenGgufSource::row_order(
    const QwenSourceTensor& tensor) const {
    // The gated-DeltaNet value axis does not arrive in the order the model uses.
    //
    // The fork's converter reorders V heads from the training order -- grouped,
    // head `k * rep + r`, which is what an HF checkpoint holds -- into the tiled
    // order its graph broadcasts in, head `r * key_heads + k`, because
    // `ggml_repeat` broadcasts tiled. Seven tensors carry a value axis and six of
    // them take the swap: the two decay projections and the two small per-head
    // projections, the packed QKV's value block, the depthwise convolution's
    // value channels, and `A_log`. The seventh, `out_proj`, is the one the
    // Hadamard fold also touches, and there the conversion deliberately keeps the
    // training order -- a column permutation on a rotation axis cannot be
    // refolded. That asymmetry is what `prism.hadamard.gdn_v_grouped` records: it
    // is set exactly when `out_proj` kept the training order, and it is the flag
    // this remap is conditional on.
    //
    // So the file's rows are mapped back to the model's here, once, at load. The
    // table is `storage_row[canonical_row]`, and the swap is its own inverse as a
    // permutation of the head axis but not as an index map, so the direction
    // matters and is pinned by test_qwen_gguf_weights.
    //
    // Reading the flag rather than the tensor name is what keeps a file that was
    // never permuted -- an upstream export, or any container with no Hadamard
    // block -- out of this path: with nothing permuted there is nothing to map
    // back, and the model's order is already the file's.
    const QwenHadamardSpec* spec = hadamard();
    if (spec == nullptr || !spec->gdn_v_grouped()) return {};

    const ValueHeads& heads = value_heads();
    const uint64_t rep = heads.value_heads / heads.key_heads;
    const uint64_t key_width = heads.key_heads * heads.key_head_dim;

    // Where the value block starts, whether the tensor is only the value block,
    // and how many rows one value head owns -- which is 1 for the per-head
    // vectors, where the row *is* the head rather than its features.
    uint64_t first_row = 0;
    uint64_t rows = 0;
    uint64_t head_dim = 0;
    const std::string& name = tensor.name;
    if (has_suffix(name, "linear_attn.in_proj_qkv.weight") ||
        has_suffix(name, "linear_attn.conv1d.weight")) {
        // Packed `[q, k, v]`, and only the value part is reordered.
        first_row = 2 * key_width;
        rows = heads.value_heads * heads.value_head_dim;
        head_dim = heads.value_head_dim;
    } else if (has_suffix(name, "linear_attn.in_proj_z.weight")) {
        rows = heads.value_heads * heads.value_head_dim;
        head_dim = heads.value_head_dim;
    } else if (has_suffix(name, "linear_attn.in_proj_a.weight") ||
               has_suffix(name, "linear_attn.in_proj_b.weight") ||
               has_suffix(name, "linear_attn.A_log") ||
               has_suffix(name, "linear_attn.dt_bias")) {
        // One row per value head, so the head is the row.
        rows = heads.value_heads;
        head_dim = 1;
    } else {
        return {};
    }

    if (tensor.shape.empty() || rows == 0 || first_row + rows > tensor.shape[0]) {
        throw std::runtime_error(
            "a tensor with a gated-DeltaNet value axis does not have one: " + name);
    }

    std::vector<uint64_t> order(tensor.shape[0]);
    for (uint64_t row = 0; row < order.size(); ++row) order[row] = row;
    for (uint64_t head = 0; head < heads.value_heads; ++head) {
        // Training order groups by key head; the file tiles it.
        const uint64_t tiled = (head % rep) * heads.key_heads + head / rep;
        for (uint64_t feature = 0; feature < head_dim; ++feature) {
            order[first_row + head * head_dim + feature] =
                first_row + tiled * head_dim + feature;
        }
    }
    return order;
}

}  // namespace pocket
