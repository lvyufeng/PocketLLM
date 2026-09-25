#pragma once

#include "gguf_reader.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace pocket {

// The incoherence transform a ternary checkpoint declares in its own header, as
// the `prism.hadamard.*` metadata block.
//
// A 1.75-bit weight is not representable without it. The ternary pack holds three
// values per weight, and a weight that is *quantized* to three values is only
// recoverable if the error it carries is spread rather than concentrated -- so the
// release rotated every weight matrix by a fixed orthogonal transform before
// quantizing it, `W' = W . R^-1` for `R = (1/sqrt(N)) . H_N . diag(s)`, which makes
// the quantization error incoherent. The weights in the file are therefore in a
// rotated frame, and the activation that meets them has to be rotated into the
// same frame at run time:
//
//   * `x |-> (1/sqrt(N)) . H . (s * x)`, signs first and then the Walsh-Hadamard
//     transform, blockwise along the last axis -- every folded *matrix* weight,
//     including `output.weight`;
//   * `z |-> (1/sqrt(N)) . s * (H . z)`, the other order, for `token_embd`: an
//     embedding row is indexed rather than multiplied, so its rotation is undone
//     after the lookup rather than applied before it.
//
// Which tensors are folded is a fact about the *file* and is read from it rather
// than inferred from a name, and a block whose semantics have not been read out of
// the fork is refused instead of defaulted. Getting this wrong does not produce an
// error: it produces a model that runs and generates fluent nonsense, which is why
// the validation here is as long as it is.
class QwenHadamardSpec {
public:
    // The block `file` declares, or nothing when it declares none -- which is
    // every other checkpoint this repository reads. An absence is not an error; a
    // *partial* block is.
    //
    // `known_tensors` is the file's own canonical name list. Passing it turns a
    // name list carried over from another conversion into an error here, rather
    // than into a tensor that is transformed and was not supposed to be.
    static std::optional<QwenHadamardSpec> find(
        const GGUFFile& file, const std::vector<std::string>& known_tensors);

    // The width of one Hadamard block. Every folded width is a multiple of it.
    int64_t block_size() const { return block_size_; }

    // The declared sign vectors' widths, in declaration order. A width here is
    // the *last* dimension of the pre-rotation tensor, which is what the block's
    // `axis = input-last-dimension` means.
    const std::vector<uint64_t>& sign_widths() const { return sign_widths_; }

    // The sign vector declared for a folded width, or an error for a width the
    // block does not declare: the signs are per-width and there is no sensible
    // default, and a wrong length or a shifted sign vector is a model that still
    // runs.
    const std::vector<int64_t>& signs_for_width(uint64_t width) const;

    // Whether this tensor's input is the rotated activation. `canonical_name` is
    // the HF spelling, because that is what the weight map and the engine speak;
    // the block itself spells its names the GGUF way and they are translated on
    // the way in.
    bool folds(const std::string& canonical_name) const {
        return folded_.count(canonical_name) != 0;
    }

    // Whether the transform is undone after the tensor rather than applied before
    // it, which is `token_embd` and nothing else.
    bool takes_inverse(const std::string& canonical_name) const {
        return inverse_.count(canonical_name) != 0;
    }

    // Whether the converter left the gated-DeltaNet value axis split between two
    // orders: it permutes six of the seven tensors that carry that axis from the
    // training head order into the tiled order its graph broadcasts in, and
    // `ssm_out` -- the one folded tensor of the seven -- deliberately keeps the
    // training order, because a column permutation on a rotation axis cannot be
    // refolded. So the file's `ssm_out` and the file's six are in different
    // orders, and this flag is what says so; the loader reads it to know that its
    // row map applies (`QwenGgufSource::row_order`).
    //
    // The consequence for the fold is that `ssm_out`'s width is the grouped one,
    // which is why the flag implies a sign vector for it. Nothing permutes an
    // activation: by the time one reaches a folded matrix, the loader has already
    // put the axis in the model's order.
    bool gdn_v_grouped() const { return gdn_v_grouped_; }

private:
    int64_t block_size_ = 0;
    std::vector<uint64_t> sign_widths_;
    std::vector<std::vector<int64_t>> signs_;
    std::set<std::string> folded_;
    std::set<std::string> inverse_;
    bool gdn_v_grouped_ = false;
};

}  // namespace pocket
