// The `prism.hadamard.*` block, decoded and validated. This translation unit is
// device-agnostic: it reads metadata and refuses what it has not been shown the
// semantics of, and nothing here needs an accelerator toolkit -- so a checkpoint's
// own description of its transform can be checked on a machine that has none.

#include "qwen_hadamard.hpp"

#include "qwen_gguf.hpp"

#include <stdexcept>
#include <string>

namespace pocket {
namespace {

constexpr const char* kPrefix = "prism.hadamard.";

// The one version, transform, axis and sign mode whose semantics have been read
// out of the fork at 842b188. Anything else is refused rather than mapped onto
// the same code: a derived or random sign vector is not the same transform, and
// the difference is invisible in the output.
constexpr int64_t kSupportedVersion = 1;
constexpr const char* kSupportedTransform = "normalized-sylvester-walsh-hadamard";
constexpr const char* kSupportedAxis = "input-last-dimension";
constexpr const char* kSupportedSignMode = "explicit";

// The width of `ssm_out`, `value_heads * value_head_dim` -- 48 * 128 here. The
// flag below says the converter kept that matrix in the training head order
// while permuting its six neighbours, so the block has to declare the sign
// vector for this width and not for the tiled one; the width is what makes the
// flag checkable against the file rather than taken on faith.
constexpr uint64_t kGdnGroupedWidth = 6144;

std::string key(const char* suffix) { return std::string(kPrefix) + suffix; }

[[noreturn]] void fail(const GGUFFile& file, const std::string& what) {
    throw std::runtime_error("the ternary checkpoint's " + key("") +
                             " block " + what + ": " + file.path());
}

// The block's own spelling of a value it declares. Every integer in the block is
// read through this: the sign values are declared as an array of int8 and both
// readings of "-1" are wrong in a way nothing downstream would notice.
std::vector<int64_t> require_ints(const GGUFFile& file, const char* suffix) {
    const auto found = file.metadata().find(key(suffix));
    if (found == file.metadata().end()) {
        fail(file, std::string("does not declare ") + key(suffix));
    }
    const MetadataValue& value = found->second;
    if (const auto* v = std::get_if<std::vector<int64_t>>(&value)) return *v;
    if (const auto* v = std::get_if<std::vector<uint64_t>>(&value)) {
        return std::vector<int64_t>(v->begin(), v->end());
    }
    if (const auto* v = std::get_if<std::vector<double>>(&value)) {
        std::vector<int64_t> out;
        out.reserve(v->size());
        for (double item : *v) out.push_back(static_cast<int64_t>(item));
        return out;
    }
    fail(file, std::string("declares ") + key(suffix) + " as a " +
                   metadata_value_to_string(value, 1) +
                   " rather than an array of integers");
}

std::vector<std::string> require_strings(const GGUFFile& file, const char* suffix) {
    const auto found = file.metadata().find(key(suffix));
    if (found == file.metadata().end()) {
        fail(file, std::string("does not declare ") + key(suffix));
    }
    if (const auto* v = std::get_if<std::vector<std::string>>(&found->second)) {
        return *v;
    }
    fail(file, std::string("declares ") + key(suffix) +
                   " as something other than an array of names");
}

int64_t require_int(const GGUFFile& file, const char* suffix) {
    const auto found = file.metadata().find(key(suffix));
    if (found == file.metadata().end()) {
        fail(file, std::string("does not declare ") + key(suffix));
    }
    const MetadataValue& value = found->second;
    if (const auto* v = std::get_if<int64_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint64_t>(&value)) {
        return static_cast<int64_t>(*v);
    }
    fail(file, std::string("declares ") + key(suffix) + " as a " +
                   metadata_value_to_string(value, 1) + " rather than an integer");
}

std::string require_string(const GGUFFile& file, const char* suffix) {
    const auto found = file.metadata().find(key(suffix));
    if (found == file.metadata().end()) {
        fail(file, std::string("does not declare ") + key(suffix));
    }
    if (const auto* v = std::get_if<std::string>(&found->second)) return *v;
    fail(file, std::string("declares ") + key(suffix) +
                   " as something other than a string");
}

}  // namespace

std::optional<QwenHadamardSpec> QwenHadamardSpec::find(
    const GGUFFile& file, const std::vector<std::string>& known_tensors) {
    bool declared = false;
    for (const auto& [name, value] : file.metadata()) {
        (void)value;
        if (name.compare(0, std::string(kPrefix).size(), kPrefix) == 0) {
            declared = true;
            break;
        }
    }
    if (!declared) return std::nullopt;

    QwenHadamardSpec spec;
    const int64_t version = require_int(file, "version");
    if (version != kSupportedVersion) {
        fail(file, "is version " + std::to_string(version) + ", and only version " +
                       std::to_string(kSupportedVersion) +
                       " has been read out of the reference implementation");
    }
    const std::string transform = require_string(file, "transform");
    if (transform != kSupportedTransform) {
        fail(file, "declares the transform " + transform + " rather than " +
                       kSupportedTransform);
    }
    const std::string axis = require_string(file, "axis");
    if (axis != kSupportedAxis) {
        fail(file, "declares the axis " + axis + " rather than " + kSupportedAxis);
    }
    const std::string sign_mode = require_string(file, "sign_mode");
    if (sign_mode != kSupportedSignMode) {
        fail(file, "declares the sign mode " + sign_mode + " rather than " +
                       kSupportedSignMode);
    }
    spec.block_size_ = require_int(file, "block_size");
    if (spec.block_size_ <= 0 || (spec.block_size_ & (spec.block_size_ - 1)) != 0) {
        fail(file, "declares a block size of " + std::to_string(spec.block_size_) +
                       ", which is not a power of two");
    }

    const std::vector<int64_t> widths = require_ints(file, "sign_widths");
    const std::vector<int64_t> values = require_ints(file, "sign_values");
    int64_t total = 0;
    for (int64_t width : widths) {
        if (width <= 0 || width % spec.block_size_ != 0) {
            fail(file, "declares a sign width of " + std::to_string(width) +
                           ", which is not a whole number of blocks");
        }
        for (int64_t other : spec.sign_widths_) {
            if (other == static_cast<uint64_t>(width)) {
                // The lookup would have to guess which vector a repeated width
                // means, and both readings are plausible.
                fail(file, "declares the width " + std::to_string(width) + " twice");
            }
        }
        spec.sign_widths_.push_back(static_cast<uint64_t>(width));
        total += width;
    }
    if (total != static_cast<int64_t>(values.size())) {
        fail(file, "declares " + std::to_string(values.size()) +
                       " sign values for widths summing to " + std::to_string(total));
    }
    int64_t cursor = 0;
    for (uint64_t width : spec.sign_widths_) {
        std::vector<int64_t> signs(values.begin() + cursor, values.begin() + cursor + width);
        for (int64_t sign : signs) {
            if (sign != 1 && sign != -1) {
                fail(file, "declares a sign value of " + std::to_string(sign) +
                               ", which is neither +1 nor -1");
            }
        }
        spec.signs_.push_back(std::move(signs));
        cursor += static_cast<int64_t>(width);
    }

    // The two name lists are the block's own spelling of the tensors it folded,
    // which is the GGUF's. Everything above this line works in the canonical
    // spelling, so the translation happens once, here.
    const std::set<std::string> known(known_tensors.begin(), known_tensors.end());
    for (const auto& [suffix, into] :
         {std::pair<const char*, std::set<std::string>*>{"weight_names", &spec.folded_},
          {"inverse_weight_names", &spec.inverse_}}) {
        for (const std::string& gguf_name : require_strings(file, suffix)) {
            const std::string canonical = qwen_canonical_tensor_name(gguf_name);
            if (!known.empty() && known.count(canonical) == 0) {
                fail(file, "declares the tensor " + gguf_name +
                               ", which the file does not hold");
            }
            into->insert(canonical);
        }
    }
    for (const std::string& name : spec.folded_) {
        if (spec.inverse_.count(name) != 0) {
            fail(file, "declares " + name +
                           " as both forward and inverse, which are different "
                           "transforms");
        }
    }

    const auto grouped = file.metadata().find(key("gdn_v_grouped"));
    if (grouped != file.metadata().end()) {
        if (const auto* v = std::get_if<bool>(&grouped->second)) {
            spec.gdn_v_grouped_ = *v;
        } else {
            fail(file, "declares gdn_v_grouped as something other than a flag");
        }
    }
    if (spec.gdn_v_grouped_) {
        bool declares_grouped = false;
        for (uint64_t width : spec.sign_widths_) {
            declares_grouped = declares_grouped || width == kGdnGroupedWidth;
        }
        if (!declares_grouped) {
            fail(file, "keeps the gated-DeltaNet output projection in the "
                       "training head order without declaring a sign vector for "
                       "the grouped width " +
                           std::to_string(kGdnGroupedWidth));
        }
    }
    return spec;
}

const std::vector<int64_t>& QwenHadamardSpec::signs_for_width(uint64_t width) const {
    for (size_t i = 0; i < sign_widths_.size(); ++i) {
        if (sign_widths_[i] == width) return signs_[i];
    }
    std::string declared;
    for (uint64_t item : sign_widths_) {
        if (!declared.empty()) declared += ", ";
        declared += std::to_string(item);
    }
    throw std::runtime_error(
        "the checkpoint's hadamard block declares no sign vector for the width " +
        std::to_string(width) + "; it declares " + declared);
}

}  // namespace pocket
