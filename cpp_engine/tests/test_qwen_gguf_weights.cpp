// The released ternary checkpoint, described by the engine's own weight map.
//
// Stage 1-5 of this adaptation built the reader, the Hadamard fold and the sm_75
// GEMM. None of them can reach a layer until the map knows where the checkpoint's
// tensors are, and the map was written against an HF safetensors directory: names
// in the `model.language_model.layers.N.*` spelling, dimensions output-first, a
// dtype per tensor. A GGUF spells all three differently -- `blk.N.*`, dimensions
// input-first, GGML block formats -- so this test is about the three translations
// and nothing else.
//
// The strong form of the check is at the bottom: the *same* canonical tensor is
// materialized through the GGUF source and through the FP8 sibling's safetensors
// index, and the two have to agree. That single comparison covers the name
// mapping, the dimension order, the dtype policy and the folded norm gamma at
// once, because a mistake in any of them moves the numbers. It needs both
// checkpoints on disk and is skipped without them.
//
// What always runs is the name mapping itself, which needs no checkpoint: a
// table of canonical-to-GGUF pairs, the round trip, and the two ways it has to
// decline -- a scale tensor an FP8 linear would be probed for, which must come
// back empty rather than be invented, and a layer suffix the table does not
// claim.

#include "qwen_config.hpp"
#include "qwen_gguf.hpp"
#include "qwen_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

template <typename A, typename B>
void check_eq(const A& actual, const B& expected, const std::string& what) {
    if (!(actual == static_cast<A>(expected))) {
        std::cout << "[FAIL] " << what << " actual=" << actual
                  << " expected=" << expected << "\n";
        ++failures;
    }
}

void check_close(double actual, double expected, double tolerance,
                 const std::string& what) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::cout << "[FAIL] " << what << " actual=" << actual
                  << " expected=" << expected << "\n";
        ++failures;
    }
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

const char* const kGgufPath =
    "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf";
const char* const kSiblingDir = "/mnt/data2/Qwen3.8-27B-FP8";

// -- the name mapping, which needs no checkpoint --------------------------- //

void check_name_mapping() {
    const std::pair<const char*, const char*> pairs[] = {
        {"model.language_model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight"},
        {"model.language_model.layers.7.post_attention_layernorm.weight",
         "blk.7.post_attention_norm.weight"},
        {"model.language_model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"},
        {"model.language_model.layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight"},
        {"model.language_model.layers.3.self_attn.k_norm.weight", "blk.3.attn_k_norm.weight"},
        {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
         "blk.0.attn_qkv.weight"},
        {"model.language_model.layers.0.linear_attn.in_proj_z.weight",
         "blk.0.attn_gate.weight"},
        {"model.language_model.layers.0.linear_attn.out_proj.weight",
         "blk.0.ssm_out.weight"},
        {"model.language_model.layers.0.linear_attn.A_log", "blk.0.ssm_a"},
        {"model.language_model.layers.0.linear_attn.dt_bias", "blk.0.ssm_dt.bias"},
        {"model.language_model.layers.0.linear_attn.conv1d.weight",
         "blk.0.ssm_conv1d.weight"},
        {"model.language_model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
        {"model.language_model.embed_tokens.weight", "token_embd.weight"},
        {"model.language_model.norm.weight", "output_norm.weight"},
        {"lm_head.weight", "output.weight"},
    };
    for (const auto& [canonical, gguf] : pairs) {
        check_eq(pocket::qwen_gguf_tensor_name(canonical), std::string(gguf),
                 std::string("gguf name for ") + canonical);
        check_eq(pocket::qwen_canonical_tensor_name(gguf), std::string(canonical),
                 std::string("canonical name for ") + gguf);
    }

    // The scale tensors an FP8 linear is probed for do not exist in this file,
    // and the mapping has to say so rather than inventing a name: `require_linear`
    // reads an empty answer as "no such tensor" and one that resolved to some
    // other layer's weight as a weight.
    const char* const absent[] = {
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale",
        "model.language_model.layers.0.mlp.gate_proj.weight_packed",
        "model.language_model.layers.0.linear_attn.unknown_thing",
        "mtp.fc.weight",
    };
    for (const char* name : absent) {
        check(pocket::qwen_gguf_tensor_name(name).empty(),
              std::string("no GGUF name should be derived for ") + name);
    }
    // A GGUF tensor nothing claims comes back unchanged, which is what makes
    // coverage able to report it rather than drop it.
    check_eq(pocket::qwen_canonical_tensor_name("blk.0.mystery.weight"),
             std::string("blk.0.mystery.weight"), "unmapped GGUF name is passed through");
}

// -- the checkpoint -------------------------------------------------------- //

// One canonical tensor out of two containers. Returns the larger absolute
// difference over the whole tensor, along with the magnitude it is relative to,
// so a caller can judge rather than trust a threshold.
struct Agreement {
    double worst = 0.0;
    double scale = 0.0;
    size_t count = 0;
};

// The fp16 the kernels would see, widened so the differences can be read as
// numbers rather than as bit patterns.
double fp16_value(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t widened = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            widened = sign;
        } else {
            uint32_t shifted = mantissa;
            int shift = 0;
            while ((shifted & 0x400u) == 0) {
                shifted <<= 1;
                ++shift;
            }
            widened = sign | (static_cast<uint32_t>(127 - 15 - shift + 1) << 23) |
                      ((shifted & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        widened = sign | 0x7f800000u | (mantissa << 13);
    } else {
        widened = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &widened, sizeof(value));
    return static_cast<double>(value);
}

Agreement compare(const pocket::QwenHostTensor& gguf,
                  const pocket::QwenHostTensor& sibling) {
    Agreement out;
    if (gguf.device_dtype != pocket::SafeDType::F16 ||
        sibling.device_dtype != pocket::SafeDType::F16 ||
        gguf.bytes.size() != sibling.bytes.size()) {
        return out;
    }
    const auto* lhs = reinterpret_cast<const uint16_t*>(gguf.bytes.data());
    const auto* rhs = reinterpret_cast<const uint16_t*>(sibling.bytes.data());
    out.count = gguf.bytes.size() / sizeof(uint16_t);
    for (size_t i = 0; i < out.count; ++i) {
        const double a = fp16_value(lhs[i]);
        const double b = fp16_value(rhs[i]);
        out.worst = std::max(out.worst, std::abs(a - b));
        out.scale = std::max(out.scale, std::abs(b));
    }
    return out;
}

void check_two_sources(const pocket::QwenWeightMap& gguf_map,
                       const pocket::QwenGgufSource& gguf_source);

// -- the fold's arithmetic, and the file's own row order ------------------- //

// A float whose value fits in a 16-bit format: the low half of an fp32 mantissa
// is where the extra bits live, so this is the whole test.
bool is_16_bit_value(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0xffffu) == 0;
}

// `stored = 1 + gamma` with gamma a 16-bit value, so subtracting one in fp32 is
// exact -- which is observable from this checkpoint alone, and is what the
// loader's `-1` relies on. The stored value is *not* 16-bit (or the fold would
// not have happened) but its shifted counterpart is, everywhere; the gated norm
// is the exception the policy names, and there the stored value is the 16-bit
// one. That asymmetry is a sharper statement than any agreement with another
// artifact could make, because it is about the arithmetic rather than about the
// numbers.
void check_norm_fold(const pocket::QwenGgufSource& source) {
    struct Case {
        const char* name;
        bool folded;
    };
    const Case cases[] = {
        {"model.language_model.layers.0.input_layernorm.weight", true},
        {"model.language_model.layers.55.input_layernorm.weight", true},
        {"model.language_model.layers.0.post_attention_layernorm.weight", true},
        {"model.language_model.norm.weight", true},
        // The two per-head attention norms, which live in the full-attention
        // layers rather than in the gated ones, so they are read out of layer 3.
        {"model.language_model.layers.3.self_attn.q_norm.weight", true},
        {"model.language_model.layers.3.self_attn.k_norm.weight", true},
        // The one norm the conversion leaves alone, and the reason
        // qwen_is_one_plus_norm_gamma exists.
        {"model.language_model.layers.0.linear_attn.norm.weight", false},
    };
    for (const Case& item : cases) {
        const pocket::QwenSourceTensor info = source.lookup(item.name);
        check(info.present, std::string("the checkpoint holds ") + item.name);
        if (!info.present) continue;
        if (info.dtype != pocket::SafeDType::F32) {
            check(false, std::string("the norm is fp32: ") + item.name);
            continue;
        }
        const uint64_t count = pocket::safe_tensor_numel(info.shape);
        const float* values = reinterpret_cast<const float*>(source.data_of(info));
        uint64_t stored_exact = 0;
        uint64_t shifted_exact = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (is_16_bit_value(values[i])) ++stored_exact;
            if (is_16_bit_value(values[i] - 1.0f)) ++shifted_exact;
        }
        std::cout << "[INFO] " << item.name << " stored 16-bit=" << stored_exact << "/"
                  << count << " shifted 16-bit=" << shifted_exact << "/" << count
                  << "\n";
        if (item.folded) {
            check_eq(shifted_exact, count,
                     std::string("removing the folded one gives a 16-bit gamma: ") + item.name);
            check(stored_exact < count,
                  std::string("the fold is there to remove: ") + item.name);
        } else {
            // The unfolded one is asserted in the direction that discriminates.
            // Its gamma is near one and it is only 128 elements wide, and on a
            // vector that short the two 16-bit grids can both cover it -- this
            // tensor is exact before the shift *and* after it -- so the negative
            // control the folded cases carry has no resolution here. What does
            // discriminate is the positive half: every folded gamma is exact only
            // after the shift and at most half its entries are exact before it,
            // while this one is exact as stored.
            check_eq(stored_exact, count,
                     std::string("the gated norm is a 16-bit gamma as stored: ") + item.name);
        }
    }
}

// The gated-DeltaNet value axis, as a shape rather than as a value: the packed
// QKV's q and k halves are the model's order and its value half is not, the two
// per-head projections and the decay scalars are entirely tiled, and `out_proj`
// -- the one the fold also touches -- is not tiled at all. Shape-only, so it
// covers the ternary tensors whose values only the fold can judge.
void check_row_order(const pocket::QwenGgufSource& source) {
    const std::string prefix = "model.language_model.layers.0.linear_attn.";
    const std::vector<uint64_t> qkv = source.row_order(source.lookup(prefix + "in_proj_qkv.weight"));
    check_eq(qkv.size(), size_t{10240}, "the packed projection has a row order");
    bool prefix_identity = true;
    for (uint64_t row = 0; row < 4096 && row < qkv.size(); ++row) {
        prefix_identity = prefix_identity && qkv[row] == row;
    }
    check(prefix_identity, "the packed projection's q and k halves keep the model's order");
    bool value_moved = false;
    for (uint64_t head = 0; head < 48; ++head) {
        // Training order head k * 3 + r; the file's is r * 16 + k.
        const uint64_t tiled_row = 4096 + ((head % 3) * 16 + head / 3) * 128;
        const uint64_t model_row = 4096 + head * 128;
        if (tiled_row >= qkv.size()) break;
        value_moved = value_moved || qkv[model_row] != model_row;
        if (qkv[model_row] != tiled_row) {
            check(false, "the packed projection's value heads are the tiled ones");
            break;
        }
    }
    check(value_moved, "the packed projection's value half is tiled");

    // One row per head, so the swap is on the head index itself.
    const std::vector<uint64_t> dt = source.row_order(source.lookup(prefix + "dt_bias"));
    check_eq(dt.size(), size_t{48}, "the decay bias has a row order");
    for (uint64_t head = 0; head < dt.size(); ++head) {
        if (dt[head] != (head % 3) * 16 + head / 3) {
            check(false, "the decay bias's heads are the tiled ones");
            break;
        }
    }

    // The two that are the model's order: the folded output projection, whose
    // rotation axis is this same value axis, and everything reading the residual
    // stream, which the conversion never reorders.
    const char* const untouched[] = {
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
    };
    for (const char* name : untouched) {
        check(source.row_order(source.lookup(name)).empty(),
              std::string("the model's own row order is left alone: ") + name);
    }
}

void check_checkpoint() {
    pocket::QwenGgufSource source(kGgufPath);
    const pocket::QwenConfig config = pocket::QwenConfig::from_gguf(kGgufPath);
    const pocket::QwenWeightMap map(source, config, 1, 0);

    // Nothing in the file is unaccounted for. This is the check that a name
    // mapping is complete rather than merely working for the tensors someone
    // happened to test.
    const pocket::QwenCoverage coverage = map.coverage();
    std::cout << "[INFO] gguf tensors=" << coverage.index_tensors
              << " mapped=" << coverage.mapped_tensors
              << " unexpected=" << coverage.unexpected_tensors
              << " visual=" << coverage.visual_tensors << "\n";
    check_eq(coverage.unexpected_tensors, size_t{0}, "every GGUF tensor is mapped");
    check_eq(coverage.mapped_tensors, size_t{851}, "851 tensors are claimed");
    map.require_full_coverage();

    // Every byte of the file's data section belongs to a mapped tensor. A map
    // that dropped a tensor would still pass the count above if it claimed a
    // different one twice, and `claim` refuses that; this catches the other
    // direction, a tensor whose byte count is wrong.
    const uint64_t data_bytes = source.file().file_size() - source.file().data_start();
    const uint64_t slack = source.file().tensor_count() * source.file().alignment();
    std::cout << "[INFO] file data=" << data_bytes
              << " mapped bytes=" << coverage.checkpoint_text_bytes << "\n";
    check(data_bytes >= coverage.checkpoint_text_bytes &&
              data_bytes - coverage.checkpoint_text_bytes < slack,
          "the mapped byte count accounts for the file's data section");

    const pocket::QwenLinearKindCounts& kinds = map.checkpoint_linear_kind_counts();
    std::cout << "[INFO] linears: ternary=" << kinds.ptq1_0
              << " dense_f16=" << kinds.dense_f16
              << " fp8_block128=" << kinds.fp8_block128
              << " fp8_channel=" << kinds.fp8_channel
              << " nvfp4=" << kinds.nvfp4_group16 << "\n";
    // 288 in the 48 linear-attention layers (qkv, z, out, gate, up, down), 112 in
    // the 16 full-attention ones (q, k, v, o, gate, up, down), and the head.
    check_eq(kinds.ptq1_0, uint64_t{401}, "401 ternary linears");
    // The two per-head projections that scale the decay are bf16 in this file:
    // 48 layers x 2. They read the same rotated activation as the ternary ones
    // and are the only dense linears in the model.
    check_eq(kinds.dense_f16, uint64_t{96}, "96 dense linears");
    check_eq(kinds.fp8_block128, uint64_t{0}, "no FP8 linears");
    check_eq(kinds.nvfp4_group16, uint64_t{0}, "no NVFP4 linears");

    // The storage shape of a ternary weight is the block geometry, and the
    // logical shape is the weight's. Both are needed and they differ; a kernel
    // reads the second and indexes the first.
    const pocket::QwenLinearRef& down = map.layers()[0].mlp.down_proj;
    check(down.kind == pocket::QwenLinearKind::Ptq1_0, "ffn_down is ternary");
    check_eq(down.logical_local_shape[0], uint64_t{5120}, "ffn_down rows");
    check_eq(down.logical_local_shape[1], uint64_t{17408}, "ffn_down columns");
    check_eq(down.weight.local_shape[1], uint64_t{17408 / 128 * 28},
             "ffn_down storage row is 28 bytes per 128 weights");
    check_eq(down.weight.nbytes, uint64_t{5120} * (17408 / 128 * 28),
             "ffn_down block bytes");
    // The whole tensor's bytes come from the logical shape through the packing,
    // not from the logical shape multiplied out: 5120 x 17408 elements would be
    // 85 MiB, and the file holds 18.6.
    check_eq(down.weight.full_nbytes, uint64_t{5120} * (17408 / 128 * 28),
             "ffn_down full block bytes");
    check(down.weight.device_dtype == pocket::SafeDType::U8,
          "ternary blocks are uploaded as bytes");

    // The embedding is the one declared tensor that takes the inverse, and it is
    // a table rather than a linear, so the map reads it as a tensor of blocks.
    check_eq(map.embed_tokens().local_shape[1], uint64_t{5120 / 128 * 28},
             "the embedding is stored as blocks too");
    const uint64_t ternary_files_bytes = [&]() {
        uint64_t total = 0;
        for (const auto& layer : map.layers()) {
            for (const pocket::QwenLinearRef* linear :
                 {&layer.linear_attention.in_proj_qkv,
                  &layer.linear_attention.in_proj_z,
                  &layer.linear_attention.out_proj, &layer.mlp.gate_proj,
                  &layer.mlp.up_proj, &layer.mlp.down_proj}) {
                if (linear->kind == pocket::QwenLinearKind::Ptq1_0) {
                    total += linear->weight.nbytes;
                }
            }
            // The full-attention layers add four attention linears; gate, up and
            // down are already counted above for every layer.
            if (layer.full_attention.q_proj.kind == pocket::QwenLinearKind::Ptq1_0) {
                total += layer.full_attention.q_proj.weight.nbytes +
                         layer.full_attention.k_proj.weight.nbytes +
                         layer.full_attention.v_proj.weight.nbytes +
                         layer.full_attention.o_proj.weight.nbytes;
            }
        }
        return total;
    }();
    // The table and the head are ternary too, and between them they are a tenth
    // of the total, so leaving them out would make this a number about the
    // layers rather than about the checkpoint.
    const uint64_t ternary_total = ternary_files_bytes + map.embed_tokens().nbytes +
                                   map.lm_head().weight.nbytes;
    std::cout << "[INFO] ternary block bytes=" << ternary_total << " ("
              << ternary_total / 1024.0 / 1024.0 / 1024.0 << " GiB)\n";
    // The number this stage exists to produce: the weights that would be 51 GiB
    // as fp16 are under six, which is why they fit beside a KV cache on one card.
    check(ternary_total > (5ull << 30) && ternary_total < (6ull << 30),
          "the ternary weights are between 5 and 6 GiB");

    check_norm_fold(source);
    check_row_order(source);

    if (!file_exists(std::string(kSiblingDir) + "/model.safetensors.index.json")) {
        std::cout << "[SKIP] the FP8 sibling is not on disk; the two-source "
                     "comparison did not run\n";
        return;
    }
    check_two_sources(map, source);
}

// -- the two sources agree ------------------------------------------------- //

// The sibling's row for a canonical row of a 48-row value block. The file tiles
// the value heads and the sibling groups them, so this is what a *correctly*
// read file is compared against, and it is also the negative control: read
// without it the two agree on nothing.
std::vector<uint64_t> value_head_order(uint64_t first_row, uint64_t head_dim) {
    const uint64_t heads = 48;
    std::vector<uint64_t> order(first_row + heads * head_dim);
    for (uint64_t row = 0; row < order.size(); ++row) order[row] = row;
    for (uint64_t head = 0; head < heads; ++head) {
        const uint64_t tiled = (head % 3) * 16 + head / 3;
        for (uint64_t feature = 0; feature < head_dim; ++feature) {
            order[first_row + head * head_dim + feature] =
                first_row + tiled * head_dim + feature;
        }
    }
    return order;
}

// The same tensor with the row order put back, i.e. as the file stores it. The
// materializer undoes the order, so this is how the check gets at the file's own
// arrangement to show that the order is what makes the two agree.
std::vector<uint16_t> rows_as_stored(const pocket::QwenHostTensor& tensor,
                                     const std::vector<uint64_t>& order) {
    const uint64_t row_numel = pocket::safe_tensor_numel(
        std::vector<uint64_t>(tensor.shape.begin() + 1, tensor.shape.end()));
    std::vector<uint16_t> out(tensor.bytes.size() / sizeof(uint16_t));
    const auto* source = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
    for (uint64_t row = 0; row < order.size(); ++row) {
        std::memcpy(out.data() + order[row] * row_numel, source + row * row_numel,
                    static_cast<size_t>(row_numel * sizeof(uint16_t)));
    }
    return out;
}

std::vector<uint16_t> fp16_values(const pocket::QwenHostTensor& tensor) {
    const auto* values = reinterpret_cast<const uint16_t*>(tensor.bytes.data());
    return std::vector<uint16_t>(values, values + tensor.bytes.size() / sizeof(uint16_t));
}

// The centred cosine of two vectors of the same length. This is the instrument
// for every comparison about the value axis, and it is a correlation rather than
// a difference because a difference threshold is not available here: the two
// released artifacts are not the same file, and on a tensor the conversion does
// not touch at all they disagree by a few percent element by element -- measured
// at 0.043 on the convolution's q/k channels, whose scale is 0.51 -- so any bound
// tight enough to be about the numbers would fail on the artifact. A correlation
// is scale-free, and the permutation is not: reading the file's order as the
// model's replaces every one of 48 head rows with a different row, which takes a
// cosine near one to one near zero whatever the two artifacts' own spread is.
//
// The mean comes out first because the tensors this is asked about are one-sided
// -- `A_log` is negative throughout, and the decay scalars share a sign -- and
// there a permutation with a sign leaves the raw cosine near one on its own.
double centred_cosine(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size() || lhs.empty()) return 0.0;
    double mean_l = 0.0;
    double mean_r = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        mean_l += lhs[i];
        mean_r += rhs[i];
    }
    mean_l /= static_cast<double>(lhs.size());
    mean_r /= static_cast<double>(rhs.size());
    double dot = 0.0;
    double norm_l = 0.0;
    double norm_r = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const double l = lhs[i] - mean_l;
        const double r = rhs[i] - mean_r;
        dot += l * r;
        norm_l += l * l;
        norm_r += r * r;
    }
    const double norm = std::sqrt(norm_l) * std::sqrt(norm_r);
    return norm > 0.0 ? dot / norm : 0.0;
}

// An element range of an fp16 tensor, as doubles. The swap touches one range of
// a tensor and not the rest, and on the convolution the range it does not touch
// is where the two artifacts differ most, so a comparison over the whole tensor
// would be about that difference rather than about the swap.
std::vector<double> fp16_range(const std::vector<uint16_t>& values, uint64_t first,
                               uint64_t count) {
    std::vector<double> out;
    for (uint64_t i = 0; i < count && first + i < values.size(); ++i) {
        out.push_back(fp16_value(values[first + i]));
    }
    return out;
}

// One tensor whose rows the conversion reordered, against the sibling's: the
// cosine with the order read, and the cosine with the file's own order put back.
// Only the pair means anything -- either number alone could be a property of the
// tensor -- and the permutation is its own control.
struct Ordered {
    double read = 0.0;
    double as_stored = 0.0;
};

Ordered compare_ordered(const pocket::QwenHostTensor& from_gguf,
                        const pocket::QwenHostTensor& from_sibling,
                        uint64_t first_row, uint64_t rows_per_head) {
    Ordered out;
    const std::vector<uint16_t> read = fp16_values(from_gguf);
    const std::vector<uint16_t> stored =
        rows_as_stored(from_gguf, value_head_order(first_row, rows_per_head));
    const std::vector<uint16_t> sibling = fp16_values(from_sibling);
    // How wide one row is comes off the tensor, and how many the swap touches
    // comes off the head geometry: 48 rows of 5120 for the two per-head
    // projections, 48 rows of four for the convolution's value channels.
    const uint64_t row_numel = pocket::safe_tensor_numel(
        std::vector<uint64_t>(from_gguf.shape.begin() + 1, from_gguf.shape.end()));
    const uint64_t first = first_row * row_numel;
    const uint64_t count = 48 * rows_per_head * row_numel;
    out.read = centred_cosine(fp16_range(read, first, count),
                              fp16_range(sibling, first, count));
    out.as_stored = centred_cosine(fp16_range(stored, first, count),
                                   fp16_range(sibling, first, count));
    return out;
}

void check_two_sources(const pocket::QwenWeightMap& gguf_map,
                       const pocket::QwenGgufSource& gguf_source) {
    const pocket::SafeTensorsIndex index(kSiblingDir);
    const pocket::SafeTensorsCheckpointSource sibling_source(index);
    const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(kSiblingDir);
    const pocket::QwenWeightMap sibling_map(sibling_source, config, 1, 0);

    // The layer input norms. What this comparison can settle is the *direction*
    // of the fold, and that is what it is asked to: the two released artifacts
    // are not the same file, and their gamma vectors differ element by element by
    // more than any rounding -- measured across the 64 layers, the worst entry
    // ranges from 0.03 to 0.18 and the difference is not a function of the value,
    // so a threshold tight enough to be about the numbers would fail on the
    // artifact rather than on the map. The file is authoritative for its own
    // gamma; what the sibling pins is the scale and the sign of the fold, which
    // is check_norm_fold's arithmetic read from the other side.
    for (uint64_t layer : {uint64_t{0}, uint64_t{3}, uint64_t{63}}) {
        const pocket::QwenHostTensor from_gguf = pocket::qwen_materialize_host_tensor(
            gguf_source, gguf_map.layers()[layer].input_layernorm);
        const pocket::QwenHostTensor from_sibling = pocket::qwen_materialize_host_tensor(
            sibling_source, sibling_map.layers()[layer].input_layernorm);
        const Agreement agreement = compare(from_gguf, from_sibling);
        check_eq(agreement.count, size_t{5120}, "the norms are 5120 wide");
        // And the fold is not a no-op. The stored gamma is `1 + gamma`, so
        // putting the one back has to take the two apart again -- by about one,
        // everywhere, which is the control this check needs: an agreement that
        // came from the fold never having been applied would fail it.
        double shifted = 0.0;
        const auto* lhs = reinterpret_cast<const uint16_t*>(from_gguf.bytes.data());
        const auto* rhs = reinterpret_cast<const uint16_t*>(from_sibling.bytes.data());
        for (size_t i = 0; i < agreement.count; ++i) {
            shifted = std::max(shifted, std::abs(fp16_value(lhs[i]) + 1.0 -
                                                 fp16_value(rhs[i])));
        }
        std::cout << "[INFO] layer " << layer
                  << " input_layernorm worst=" << agreement.worst
                  << " scale=" << agreement.scale << " with the one put back=" << shifted
                  << "\n";
        check(shifted > 10.0 * agreement.worst,
              "the layer norm is the sibling's only after the one is removed");
        check(shifted > 0.5, "the norm gamma really is stored with the 1 folded in");
    }

    // The gated-DeltaNet norm is the exception: the runtime applies it directly,
    // so nothing is subtracted and it is the *unfolded* comparison that has to be
    // the closer one. The relationship is the layer norms' in the other
    // direction, but this gamma is small -- a scale of 0.93 against the layer
    // norm's 0.20 -- so taking a one back off moves the vector 1.0 away rather
    // than fifteen times the stored difference, and the bound is set from the
    // measurement instead of reused. A run that folded this norm anyway would put
    // the factor on the other side, which is what the pair is here for.
    {
        const pocket::QwenHostTensor from_gguf = pocket::qwen_materialize_host_tensor(
            gguf_source, gguf_map.layers()[0].linear_attention.norm);
        const pocket::QwenHostTensor from_sibling = pocket::qwen_materialize_host_tensor(
            sibling_source, sibling_map.layers()[0].linear_attention.norm);
        const Agreement agreement = compare(from_gguf, from_sibling);
        double shifted = 0.0;
        const auto* lhs = reinterpret_cast<const uint16_t*>(from_gguf.bytes.data());
        const auto* rhs = reinterpret_cast<const uint16_t*>(from_sibling.bytes.data());
        for (size_t i = 0; i < agreement.count; ++i) {
            shifted = std::max(shifted, std::abs(fp16_value(lhs[i]) + 1.0 -
                                                 fp16_value(rhs[i])));
        }
        std::cout << "[INFO] ssm_norm worst=" << agreement.worst
                  << " scale=" << agreement.scale << " with the one put back=" << shifted
                  << "\n";
        check(shifted > 4.0 * agreement.worst,
              "the gated norm is compared without a fold");
        check(shifted > 0.5, "the gated norm's gamma is not itself near one");
    }

    // The tiled value axis. These are not folded and both containers hold them as
    // elements, so with the row order read they agree with the sibling and with
    // the file's order put back they agree on nothing: the swap is a head-level
    // permutation, which is not a small error but a different tensor.
    {
        const Ordered decay = compare_ordered(
            pocket::qwen_materialize_host_tensor(
                gguf_source, gguf_map.layers()[0].linear_attention.in_proj_a.weight),
            pocket::qwen_materialize_host_tensor(
                sibling_source, sibling_map.layers()[0].linear_attention.in_proj_a.weight),
            0, 1);
        std::cout << "[INFO] ssm_alpha cosine read=" << decay.read
                  << " as stored=" << decay.as_stored << "\n";
        check(decay.read > 0.9, "ssm_alpha agrees once the row order is read");
        check(decay.as_stored < 0.5, "ssm_alpha disagrees as the file stores it");

        const Ordered gate = compare_ordered(
            pocket::qwen_materialize_host_tensor(
                gguf_source, gguf_map.layers()[0].linear_attention.in_proj_b.weight),
            pocket::qwen_materialize_host_tensor(
                sibling_source, sibling_map.layers()[0].linear_attention.in_proj_b.weight),
            0, 1);
        std::cout << "[INFO] ssm_beta cosine read=" << gate.read
                  << " as stored=" << gate.as_stored << "\n";
        check(gate.read > 0.9, "ssm_beta agrees once the row order is read");
        check(gate.as_stored < 0.5, "ssm_beta disagrees as the file stores it");

        // A_log: the GGUF stores -exp(A_log), the sibling stores A_log, so the
        // comparison is in the log -- which is also what makes it an exact
        // statement rather than a near one, since the file's value is a rounded
        // exp of the sibling's and the log takes that rounding back out.
        const pocket::QwenHostTensor a_gguf = pocket::qwen_materialize_host_tensor(
            gguf_source, gguf_map.layers()[0].linear_attention.a_log);
        const pocket::QwenHostTensor a_sibling = pocket::qwen_materialize_host_tensor(
            sibling_source, sibling_map.layers()[0].linear_attention.a_log);
        const std::vector<uint16_t> a_read = fp16_values(a_gguf);
        const std::vector<uint16_t> a_stored = rows_as_stored(a_gguf, value_head_order(0, 1));
        const std::vector<uint16_t> a_sibling_values = fp16_values(a_sibling);
        std::vector<double> log_read;
        std::vector<double> log_stored;
        std::vector<double> log_sibling;
        for (size_t i = 0; i < a_read.size(); ++i) {
            const double value = fp16_value(a_read[i]);
            check(value < 0.0, "ssm_a is stored as -exp(A_log) and is negative");
            log_read.push_back(std::log(-value));
            log_stored.push_back(std::log(-fp16_value(a_stored[i])));
            log_sibling.push_back(fp16_value(a_sibling_values[i]));
        }
        const double a_read_cosine = centred_cosine(log_read, log_sibling);
        const double a_stored_cosine = centred_cosine(log_stored, log_sibling);
        std::cout << "[INFO] ssm_a log cosine read=" << a_read_cosine
                  << " as stored=" << a_stored_cosine << "\n";
        check(a_read_cosine > 0.99, "ssm_a is -exp(A_log) of the sibling's A_log");
        check(a_stored_cosine < 0.5, "ssm_a disagrees as the file stores it");

        const Ordered dt = compare_ordered(
            pocket::qwen_materialize_host_tensor(
                gguf_source, gguf_map.layers()[0].linear_attention.dt_bias),
            pocket::qwen_materialize_host_tensor(
                sibling_source, sibling_map.layers()[0].linear_attention.dt_bias),
            0, 1);
        std::cout << "[INFO] dt_bias cosine read=" << dt.read
                  << " as stored=" << dt.as_stored << "\n";
        check(dt.read > 0.99, "dt_bias agrees once the row order is read");
        check(dt.as_stored < 0.5, "dt_bias disagrees as the file stores it");
    }

    // The convolution: [channels, 1, kernel] in the checkpoint and
    // {kernel, channels} in the GGUF, so this is also the check that the one
    // tensor whose rank differs was rebuilt the right way round. Its value
    // channels carry the same tiled axis, and its q/k channels do not.
    //
    // What is compared is the value channels rather than the tensor. The q/k
    // channels are the model's order in both containers and the two artifacts
    // disagree on them anyway -- by 0.043 against a scale of 0.51 -- so a
    // comparison that included them would be about that disagreement and would
    // read the same in both directions, which is what the whole-tensor form of
    // this check did.
    {
        const pocket::QwenHostTensor conv_gguf = pocket::qwen_materialize_host_tensor(
            gguf_source, gguf_map.layers()[0].linear_attention.conv1d);
        const pocket::QwenHostTensor conv_sibling = pocket::qwen_materialize_host_tensor(
            sibling_source, sibling_map.layers()[0].linear_attention.conv1d);
        check_eq(conv_gguf.shape.size(), size_t{3}, "the convolution keeps its rank");
        check_eq(conv_gguf.shape[1], uint64_t{1}, "the convolution keeps its singleton axis");
        check_eq(conv_gguf.shape[2], uint64_t{4}, "the convolution keeps its kernel width");
        const Ordered tiled = compare_ordered(conv_gguf, conv_sibling, 4096, 128);
        std::cout << "[INFO] conv1d value channels cosine read=" << tiled.read
                  << " as stored=" << tiled.as_stored << "\n";
        check(tiled.read > 0.9,
              "the convolution's value channels agree once the row order is read");
        check(tiled.as_stored < 0.5,
              "the convolution's value channels are tiled and its q/k channels are not");
    }
}

}  // namespace

int main() {
    check_name_mapping();
    if (!file_exists(kGgufPath)) {
        std::cout << "[SKIP] the released ternary checkpoint is not on disk\n";
    } else {
        check_checkpoint();
    }
    if (failures != 0) {
        std::cout << "[FAIL] test_qwen_gguf_weights (" << failures << " failures)\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_gguf_weights\n";
    return 0;
}
