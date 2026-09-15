// Numeric validation for the Cube (Mmad) prefill attention path.
//
// The Cube kernel is a reimplementation of the same operator the vector kernel
// already computes, so the cheapest honest reference is the vector kernel itself:
// it is what the engine has been running, and it is exercised here on the same
// binary and the same inputs. A float host reference is kept alongside it because
// "two kernels agree" is not the same statement as "both are right" -- an error
// shared by the staging code would survive the first comparison and not the
// second.
//
// The two paths are selected by QWEN_ASCEND_GQA_CUBE, which the launcher reads per
// call rather than caching, so one process can obtain both results.
//
//   ./tests/test_qwen_ascend_cube_attention [--device N] [--lengths 64,256,1024]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;

// Above this the double-precision reference is skipped and the vector kernel
// becomes the reference instead; see the note in run_case.
constexpr int kMaxReferenceRows = 1024;

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count == 0) return;
        if (!pocket::device_malloc_into(ptr_, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, count * sizeof(uint16_t))) {
            throw std::runtime_error("device_memset failed");
        }
    }
    explicit DeviceBuffer(const std::vector<uint16_t>& host) : DeviceBuffer(host.size()) {
        if (!pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(uint16_t))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    uint16_t* get() const { return ptr_; }

private:
    uint16_t* ptr_ = nullptr;
    size_t count_ = 0;
};

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

// Causal GQA attention in double, row by row, straight from the definition. Slow
// and obvious on purpose: everything the kernels do to be fast is what could be
// wrong, so the reference does none of it.
std::vector<float> reference_attention(const std::vector<uint16_t>& q,
                                       const std::vector<uint16_t>& k,
                                       const std::vector<uint16_t>& v, int rows) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    std::vector<float> out(static_cast<size_t>(rows) * kQHeads * kHeadDim, 0.0f);
    std::vector<double> scores(rows);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < kQHeads; ++head) {
            const size_t q_base = (static_cast<size_t>(row) * kQHeads + head) * kHeadDim;
            double maximum = -1e30;
            for (int pos = 0; pos <= row; ++pos) {
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(half_to_float(q[q_base + d])) *
                           static_cast<double>(half_to_float(k[static_cast<size_t>(pos) * kHeadDim + d]));
                }
                scores[pos] = dot * scale;
                if (scores[pos] > maximum) maximum = scores[pos];
            }
            double denominator = 0.0;
            for (int pos = 0; pos <= row; ++pos) {
                scores[pos] = std::exp(scores[pos] - maximum);
                denominator += scores[pos];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double accum = 0.0;
                for (int pos = 0; pos <= row; ++pos) {
                    accum += scores[pos] *
                             static_cast<double>(half_to_float(v[static_cast<size_t>(pos) * kHeadDim + d]));
                }
                out[q_base + d] = static_cast<float>(accum / denominator);
            }
        }
    }
    return out;
}

// The decode shape: one query row per head, attending every position in the
// context. Same definition as above with the causal limit pinned at the end of the
// context instead of at the row index.
std::vector<float> reference_decode(const std::vector<uint16_t>& q,
                                    const std::vector<uint16_t>& k,
                                    const std::vector<uint16_t>& v, int context_len) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    std::vector<float> out(static_cast<size_t>(kQHeads) * kHeadDim, 0.0f);
    std::vector<double> scores(context_len);
    for (int head = 0; head < kQHeads; ++head) {
        const size_t q_base = static_cast<size_t>(head) * kHeadDim;
        double maximum = -1e30;
        for (int pos = 0; pos < context_len; ++pos) {
            double dot = 0.0;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += static_cast<double>(half_to_float(q[q_base + d])) *
                       static_cast<double>(
                           half_to_float(k[static_cast<size_t>(pos) * kHeadDim + d]));
            }
            scores[pos] = dot * scale;
            if (scores[pos] > maximum) maximum = scores[pos];
        }
        double denominator = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            scores[pos] = std::exp(scores[pos] - maximum);
            denominator += scores[pos];
        }
        for (int d = 0; d < kHeadDim; ++d) {
            double accum = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                accum += scores[pos] *
                         static_cast<double>(
                             half_to_float(v[static_cast<size_t>(pos) * kHeadDim + d]));
            }
            out[q_base + d] = static_cast<float>(accum / denominator);
        }
    }
    return out;
}

struct Error {
    double max_abs = 0.0;
    size_t bad = 0;
    size_t total = 0;
};

Error compare(const std::vector<uint16_t>& actual, const std::vector<float>& reference,
              double tolerance) {
    Error err;
    err.total = reference.size();
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = std::fabs(static_cast<double>(half_to_float(actual[i])) -
                                            static_cast<double>(reference[i]));
        if (difference > err.max_abs) err.max_abs = difference;
        if (difference > tolerance) ++err.bad;
    }
    return err;
}

int run_case(int device, int rows) {
    (void)device;
    std::mt19937 rng(4242u + static_cast<unsigned>(rows));
    const size_t q_elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
    const size_t kv_elements = static_cast<size_t>(rows) * kKvHeads * kHeadDim;
    const std::vector<uint16_t> host_q = random_halves(q_elements, rng, 0.5f);
    const std::vector<uint16_t> host_k = random_halves(kv_elements, rng, 0.5f);
    const std::vector<uint16_t> host_v = random_halves(kv_elements, rng, 0.5f);

    DeviceBuffer q(host_q), k(host_k), v(host_v);
    DeviceBuffer vector_out(q_elements);
    DeviceBuffer cube_out(q_elements);

    setenv("QWEN_ASCEND_GQA_CUBE", "0", 1);
    if (!pocket::qwen_gqa_prefill_attention_f16(q.get(), k.get(), v.get(), vector_out.get(),
                                                rows, kQHeads, kKvHeads, kHeadDim, 0, rows)) {
        std::printf("[FAIL] vector prefill launch failed rows=%d\n", rows);
        return 1;
    }
    unsetenv("QWEN_ASCEND_GQA_CUBE");
    if (!pocket::qwen_gqa_prefill_attention_f16(q.get(), k.get(), v.get(), cube_out.get(),
                                                rows, kQHeads, kKvHeads, kHeadDim, 0, rows)) {
        std::printf("[FAIL] cube prefill launch failed rows=%d\n", rows);
        return 1;
    }
    if (!pocket::device_synchronize()) {
        std::printf("[FAIL] synchronize failed rows=%d\n", rows);
        return 1;
    }

    std::vector<uint16_t> host_vector(q_elements), host_cube(q_elements);
    if (!pocket::memcpy_d2h(host_vector.data(), vector_out.get(), q_elements * sizeof(uint16_t)) ||
        !pocket::memcpy_d2h(host_cube.data(), cube_out.get(), q_elements * sizeof(uint16_t))) {
        std::printf("[FAIL] memcpy_d2h failed rows=%d\n", rows);
        return 1;
    }

    // The host reference costs one inner product per (row, position, dim) triple,
    // so it is quadratic in rows and takes minutes past a few thousand. It is not
    // what the long lengths are for -- those check that the chunk grid, the ragged
    // transpose tail and the row-tile split all still line up, and for that the
    // vector kernel is the reference.
    const double tolerance = 4e-3;
    Error vector_error;
    Error cube_error;
    const std::vector<float> reference =
        rows <= kMaxReferenceRows ? reference_attention(host_q, host_k, host_v, rows)
                                  : std::vector<float>();
    if (!reference.empty()) {
        // fp16 operands and an fp16 output put the floor near 1e-3 of the output
        // scale; the tolerance is per element and the reported bad count is what
        // distinguishes a few rounding outliers from a real layout error.
        vector_error = compare(host_vector, reference, tolerance);
        cube_error = compare(host_cube, reference, tolerance);
    } else {
        std::vector<float> reference_half(host_vector.size());
        for (size_t i = 0; i < host_vector.size(); ++i) {
            reference_half[i] = half_to_float(host_vector[i]);
        }
        vector_error = compare(host_cube, reference_half, tolerance);
        cube_error = vector_error;
    }

    size_t diverged_rows = 0;
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < kQHeads; ++head) {
            const size_t base = (static_cast<size_t>(row) * kQHeads + head) * kHeadDim;
            bool diverge = false;
            for (int d = 0; d < kHeadDim; ++d) {
                if (std::fabs(static_cast<double>(half_to_float(host_cube[base + d])) -
                              static_cast<double>(half_to_float(host_vector[base + d]))) > tolerance) {
                    diverge = true;
                    break;
                }
            }
            if (diverge) ++diverged_rows;
        }
    }

    const bool ok = cube_error.bad == 0 && vector_error.bad == 0;
    std::printf(
        "rows=%-5d ref=%-6s vector max_abs=%.3e bad=%zu/%-7zu  cube max_abs=%.3e bad=%zu/%-7zu  "
        "head-rows diverging=%zu/%d %s\n",
        rows, reference.empty() ? "vector" : "double", vector_error.max_abs, vector_error.bad,
        vector_error.total, cube_error.max_abs, cube_error.bad, cube_error.total, diverged_rows,
        rows * kQHeads, ok ? "ok" : "MISMATCH");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

// The decode entry point, checked the same way: Cube against the vector kernel it
// replaces and against the definition. The two use different work decomposition --
// the vector kernel gives each head its own block, the Cube one stacks the whole
// head group into a single tile -- so agreement between them is not implied by the
// prefill result.
//
// The context-split Cube kernel is checked in the same run, against the same
// reference. It is the third decomposition of the same sum: one block per
// (partition, KV head) with a separate cross-partition merge, so its agreement is
// evidence about the online-softmax rescaling, not about the Cube entry above.
int run_decode_case(int context) {
    std::mt19937 rng(7311u + static_cast<unsigned>(context));
    const size_t q_elements = static_cast<size_t>(kQHeads) * kHeadDim;
    const size_t kv_elements = static_cast<size_t>(context) * kKvHeads * kHeadDim;
    const std::vector<uint16_t> host_q = random_halves(q_elements, rng, 0.5f);
    const std::vector<uint16_t> host_k = random_halves(kv_elements, rng, 0.5f);
    const std::vector<uint16_t> host_v = random_halves(kv_elements, rng, 0.5f);

    DeviceBuffer q(host_q), k(host_k), v(host_v);
    DeviceBuffer vector_out(q_elements);
    DeviceBuffer cube_out(q_elements);
    DeviceBuffer split_out(q_elements);
    std::vector<float> scratch(std::max<size_t>(
        static_cast<size_t>(kQHeads) * context, static_cast<size_t>(kQHeads) * kHeadDim));
    float* scores = nullptr;
    if (!pocket::device_malloc_into(scores, scratch.size() * sizeof(float))) {
        std::printf("[FAIL] scratch malloc failed context=%d\n", context);
        return 1;
    }

    setenv("QWEN_ASCEND_GQA_CUBE", "0", 1);
    const bool vector_ok = pocket::qwen_gqa_decode_attention_f16(
        q.get(), k.get(), v.get(), vector_out.get(), scores, kQHeads, kKvHeads, kHeadDim,
        context, context);
    unsetenv("QWEN_ASCEND_GQA_CUBE");
    const bool cube_ok = pocket::qwen_gqa_decode_attention_f16(
        q.get(), k.get(), v.get(), cube_out.get(), scores, kQHeads, kKvHeads, kHeadDim,
        context, context);
    // `partitions = 0` picks the widest split the context allows. A context that
    // fits in a single 512-column chunk has nothing to split, and the kernel refuses
    // rather than quietly running one partition -- so the arm is required to succeed
    // past that point and required to refuse before it.
    const bool split_applies = context > 512;
    const bool split_ok = pocket::qwen_gqa_decode_attention_cube_split_f16(
        q.get(), k.get(), v.get(), split_out.get(), kQHeads, kKvHeads, kHeadDim,
        context, context, 0);
    const bool synced = pocket::device_synchronize();
    pocket::device_free(scores);
    if (!vector_ok || !cube_ok || !synced || split_ok != split_applies) {
        std::printf("[FAIL] decode launch failed context=%d vector=%d cube=%d split=%d\n",
                    context, vector_ok ? 1 : 0, cube_ok ? 1 : 0, split_ok ? 1 : 0);
        return 1;
    }

    std::vector<uint16_t> host_vector(q_elements), host_cube(q_elements),
        host_split(q_elements);
    if (!pocket::memcpy_d2h(host_vector.data(), vector_out.get(), q_elements * sizeof(uint16_t)) ||
        !pocket::memcpy_d2h(host_cube.data(), cube_out.get(), q_elements * sizeof(uint16_t)) ||
        (split_applies &&
         !pocket::memcpy_d2h(host_split.data(), split_out.get(),
                             q_elements * sizeof(uint16_t)))) {
        std::printf("[FAIL] decode memcpy_d2h failed context=%d\n", context);
        return 1;
    }

    const double tolerance = 4e-3;
    const std::vector<float> reference = reference_decode(host_q, host_k, host_v, context);
    const Error vector_error = compare(host_vector, reference, tolerance);
    const Error cube_error = compare(host_cube, reference, tolerance);
    const Error split_error =
        split_applies ? compare(host_split, reference, tolerance) : Error{0.0, 0, 0};
    const bool ok = cube_error.bad == 0 && vector_error.bad == 0 && split_error.bad == 0;
    char split_note[48];
    if (split_applies) {
        std::snprintf(split_note, sizeof(split_note), "max_abs=%.3e bad=%zu/%-7zu",
                      split_error.max_abs, split_error.bad, split_error.total);
    } else {
        std::snprintf(split_note, sizeof(split_note), "n/a (refused: one chunk)    ");
    }
    std::printf(
        "decode context=%-5d vector max_abs=%.3e bad=%zu/%-7zu  cube max_abs=%.3e "
        "bad=%zu/%-7zu  cube-split %s %s\n",
        context, vector_error.max_abs, vector_error.bad, vector_error.total,
        cube_error.max_abs, cube_error.bad, cube_error.total, split_note,
        ok ? "ok" : "MISMATCH");
    std::fflush(stdout);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    std::vector<int> lengths = {64, 256, 1024};
    std::vector<int> contexts = {512, 1024, 4096};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--lengths" && i + 1 < argc) {
            lengths.clear();
            const std::string rest = argv[++i];
            size_t at = 0;
            while (at <= rest.size()) {
                const size_t comma = rest.find(',', at);
                const std::string piece =
                    rest.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (!piece.empty()) lengths.push_back(std::stoi(piece));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        } else if (arg == "--contexts" && i + 1 < argc) {
            contexts.clear();
            const std::string rest = argv[++i];
            size_t at = 0;
            while (at <= rest.size()) {
                const size_t comma = rest.find(',', at);
                const std::string piece =
                    rest.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (!piece.empty()) contexts.push_back(std::stoi(piece));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    int failures = 0;
    for (const int rows : lengths) failures += run_case(device, rows);
    for (const int context : contexts) failures += run_decode_case(context);
    return failures == 0 ? 0 : 1;
}
