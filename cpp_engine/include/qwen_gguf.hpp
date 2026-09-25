#pragma once

#include "gguf_reader.hpp"
#include "qwen_weights.hpp"

#include <optional>
#include <string>

namespace pocket {

// A single-file GGUF, as a Qwen checkpoint source.
//
// The map, the sharding arithmetic and the coverage audit are container-neutral
// and live in qwen_weight_map.cpp. What is specific to this container is here,
// and it is three things: the tensor *names*, which a GGUF spells the llama.cpp
// way and this runtime spells the HF way; the *dimension order*, which a GGUF
// states input-first; and the tensor *types*, which are the GGML block formats
// rather than element arrays.
//
// The names are derived rather than enumerated. Every tensor this runtime reads
// has one GGUF spelling and it is the same word every time -- `input_layernorm`
// is always `attn_norm`, `A_log` is always `ssm_a` -- so the table below is
// twenty-one suffix pairs rather than eight hundred names, and a layer added to
// the architecture later needs no new entry here.
class QwenGgufSource : public QwenCheckpointSource {
public:
    explicit QwenGgufSource(const std::string& path);

    QwenSourceTensor lookup(const std::string& canonical_name) const override;
    const uint8_t* data_of(const QwenSourceTensor& tensor) const override;
    std::vector<std::string> tensor_names() const override;
    size_t tensor_count() const override;
    const char* format_name() const override { return "gguf"; }

    // A GGUF states every tensor in one of the GGML block formats, and the
    // runtime's device policy does not know about them: it knows Safetensors
    // dtypes. The three this checkpoint uses map onto fp16 for the device,
    // because the kernels want fp16 gamma and there is no element format here
    // that Turing can use directly.
    SafeDType device_dtype(SafeDType storage_dtype) const override;

    // The released artifact bakes the `1 + gamma` convention into its norm
    // weights, because a GGML graph applies the weight directly. This runtime
    // adds the one, so the loader takes it back out.
    bool folds_one_plus_norm_gamma() const override { return true; }

    // A GGUF states a tensor's GGML type rather than the model's element type.
    // The map declares what the model holds -- a 16-bit element for every dense
    // tensor, fp8 or fp32 for the scale metadata that only an FP8 checkpoint
    // has -- and this file stores whichever of the GGML widths the conversion
    // chose, plus the ternary pack for the weights themselves.
    bool accepts_storage_dtype(SafeDType declared, SafeDType actual,
                               bool ternary) const override;

    // The gated-DeltaNet value axis, in the tiled order the conversion gave it.
    // See the definition; this is the one place where the file's row order is not
    // the model's.
    std::vector<uint64_t> row_order(const QwenSourceTensor& tensor) const override;

    const GGUFFile& file() const { return file_; }

private:
    // The value-head geometry the conversion reordered against, read out of the
    // file's own metadata rather than assumed: it is what ties the swap to this
    // checkpoint instead of to the architecture in general.
    struct ValueHeads {
        uint64_t key_heads = 0;
        uint64_t key_head_dim = 0;
        uint64_t value_heads = 0;
        uint64_t value_head_dim = 0;
    };
    const ValueHeads& value_heads() const;

    GGUFFile file_;
    mutable std::optional<ValueHeads> value_heads_;
};

// The GGUF spelling of a canonical Qwen tensor name, or an empty string when the
// name is not one this mapping knows -- which is also the answer for the scale
// tensors an FP8 linear would be probed for, so a probe finds nothing rather
// than failing.
std::string qwen_gguf_tensor_name(const std::string& canonical_name);

// The canonical name of a GGUF tensor, or the GGUF name unchanged when nothing
// in the table claims it. The caller reports the unchanged ones as unmapped,
// which is what makes coverage a check on the file rather than on itself.
std::string qwen_canonical_tensor_name(const std::string& gguf_name);

}  // namespace pocket
