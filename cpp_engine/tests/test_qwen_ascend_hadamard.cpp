// Does the Ascend incoherence rotation compute the transform it claims?
//
// The rotation is `R = (1/sqrt(N)) H_N diag(s)` with `H_N[i][j] = (-1)^popcount(i
// AND j)` the natural-order Sylvester Hadamard matrix, applied independently to each
// `block` run of the last axis, in one of two orders:
//
//   forward  y = (1/sqrt(N)) H (s * x)
//   inverse  y = (1/sqrt(N)) s * (H x)
//
// The reference here is the definition, not the CUDA kernel. Two kernels agreeing
// on the same misunderstanding is not evidence, and this one has a specific
// misunderstanding available to it: three of the ten butterfly levels pair elements
// 4, 8 and 16 bytes apart, which the vector unit cannot address, so those levels run
// on the scalar unit and a wrong pairing there would still produce a
// plausible-looking orthogonal transform.
//
// Three kinds of case:
//
//   *Small integers.* The input is an integer in [-16, 16] and the scale is
//   1/sqrt(1024) = 1/32, so every value the transform touches -- and every
//   partial sum, whose magnitude is bounded by 16 * 1024 / 32 = 512 -- is a
//   multiple of 1/32 and exact in both FP32 and FP64. The arithmetic before the
//   final narrowing is therefore exact on both sides, and the case is decided by
//   how each narrows to FP16. That is what makes the comparison equality rather
//   than a bound: a wrong pairing cannot land on the reference's own FP16 value
//   by accident.
//
//   *Random FP16.* The inputs carry full 10-bit mantissas and the reference is
//   FP64, so the two separate by the kernel's fp32 accumulation order alone. This
//   is the case that would catch a rounding mode or a dropped pass; the integer
//   case would not. It is judged by a bound rather than by equality, because on
//   full mantissas a one-ulp disagreement is expected rather than a symptom: the
//   fp32 error is ~1e-6 relative against a half-precision rounding boundary
//   spacing of 5e-4, so a few elements per run land on the far side of a
//   boundary. The bound has two arms, per-element and per-call, for the reason
//   `compare` gives.
//
//   *Zeros.* The one input whose answer is known by inspection, and the one that
//   pins the sign of a zero rather than a value.
//
// The launcher's refusals are checked too, because the kernel sizes its UB tiles
// statically and a `block` above that ceiling is a copy past the end of a tile
// rather than an error.
//
//   ./tests/test_qwen_ascend_hadamard [--device N]

#include "device_runtime.hpp"
#include "qwen_ascend_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("  FAIL %s\n", what.c_str());
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t shifted = mantissa;
            int shift = 0;
            while ((shifted & 0x400u) == 0) {
                shifted <<= 1;
                ++shift;
            }
            bits = sign | ((127u - 15u - static_cast<uint32_t>(shift) + 1u) << 23) |
                   ((shifted & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

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
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign |
                                         (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half);
}

// The definition, in double, over `rows` x `width` with the transform applied to
// each `block` run of the row.
//
// `H x` is the standard in-place butterfly network: level `step` pairs `i` with
// `i + step` wherever `i`'s `step` bit is clear, which visits the levels in the
// order 1, 2, 4, ... and produces `H_N[i][j] = (-1)^popcount(i AND j)` on the
// natural (Hadamard) ordering.
std::vector<double> reference(const std::vector<uint16_t>& x,
                              const std::vector<float>& signs, int rows, int width,
                              int block, bool forward) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(block));
    std::vector<double> out(x.size());
    for (int row = 0; row < rows; ++row) {
        for (int run = 0; run * block < width; ++run) {
            const size_t base = static_cast<size_t>(row) * width + run * block;
            std::vector<double> tile(static_cast<size_t>(block));
            for (int i = 0; i < block; ++i) {
                const double value = static_cast<double>(half_to_float(x[base + i]));
                const double sign = static_cast<double>(signs[run * block + i]);
                tile[i] = (forward ? value * sign : value) * scale;
            }
            for (int step = 1; step < block; step <<= 1) {
                for (int i = 0; i < block; i += 2 * step) {
                    for (int j = 0; j < step; ++j) {
                        const double low = tile[i + j];
                        const double high = tile[i + step + j];
                        tile[i + j] = low + high;
                        tile[i + step + j] = low - high;
                    }
                }
            }
            for (int i = 0; i < block; ++i) {
                double value = tile[i];
                if (!forward) value *= static_cast<double>(signs[run * block + i]);
                out[base + i] = value;
            }
        }
    }
    return out;
}

uint32_t rng_state = 0x9e3779b9u;

uint32_t next_random() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

// Full-mantissa FP16 inputs: sign and a random exponent in a range that neither
// overflows nor denormalizes through ten butterfly levels.
uint16_t random_half() {
    const uint32_t bits = (next_random() & 0x8000u) | (15u << 10) | (next_random() & 0x3ffu);
    return static_cast<uint16_t>(bits);
}

struct Geometry {
    int rows;
    int width;
    int block;
};

enum class Input {
    kIntegers,
    kRandom,
    kZeros,
};

struct Case {
    const char* name;
    Input input;
    bool alias;
    Geometry geometry;
};

// Is `got` the reference's own value?
//
// A zero counts whichever sign it carries. The vector unit's multiply and its
// narrowing both hand back +0 for a zero product (`+0 * -1` measured as +0000,
// not 8000), while the reference keeps the sign the last multiply gave it. A
// zero is the one value the rotation does not have to reproduce a sign for --
// every weight it meets is multiplied by it -- so this is a comparison on
// values, and the two zeros are the same value.
bool same_value(uint16_t got, double want) {
    if (want == 0.0) return (got & 0x7fffu) == 0;
    return float_to_half(static_cast<float>(want)) == got;
}

struct Comparison {
    int exact = 0;             // the reference half itself, up to the sign of zero
    int within_tolerance = 0;  // within one of the two bounds below
    double peak = 0.0;         // largest |reference| in the call
    double worst = 0.0;        // largest |got - want| seen
};

// The reference is FP64 and the kernel FP32, so the two separate by the
// accumulation order alone. That is a part in a million of the *terms*, but a
// Hadamard output cancels: an element that lands near zero can carry a
// relative error of order one while being as close to the reference as the
// run's own arithmetic allows, so an element-relative bound is meaningless
// there. Two bounds, and an element has to clear one of them:
//
//   *An element's own 2 ulp*, which is what a broken final narrowing would
//   break -- a truncating cast is off by up to one ulp of the value itself,
//   and that is invisible against any bound sized to the whole call.
//
//  *Two ulp of the call's peak*, for the elements the transform cancelled.
//   fp32 accumulation error is absolute -- it scales with the largest partial
//   sum, not with the result -- so the peak is the honest ruler for those.
double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 10); }

Comparison compare(const std::vector<uint16_t>& got, const std::vector<double>& want) {
    Comparison out;
    for (size_t i = 0; i < got.size(); ++i) {
        out.peak = std::max(out.peak, std::fabs(want[i]));
    }
    const double peak_tolerance = 2.0 * ulp_of(out.peak);
    for (size_t i = 0; i < got.size(); ++i) {
        const double got_value = static_cast<double>(half_to_float(got[i]));
        const double error = std::fabs(got_value - want[i]);
        if (error > out.worst) out.worst = error;
        if (same_value(got[i], want[i])) ++out.exact;
        if (error <= 2.0 * ulp_of(std::fabs(want[i])) || error <= peak_tolerance) {
            ++out.within_tolerance;
        }
    }
    return out;
}

int run(int device) {
    if (!pocket::device_runtime_available()) {
        std::printf("[SKIP] no device runtime available\n");
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::printf("[SKIP] device_set failed for device %d\n", device);
        return 0;
    }

    // The checkpoint's own geometry first -- 1024-element runs over a 5120-wide
    // folded projection -- then a one-run row, a block below the checkpoint's, and
    // the largest block the kernel's static tiles admit.
    const Case cases[] = {
        {"integers, block 1024 over 5120", Input::kIntegers, false, {3, 5120, 1024}},
        {"random, block 1024 over 5120", Input::kRandom, false, {3, 5120, 1024}},
        {"random, block 1024 over 1024", Input::kRandom, false, {1, 1024, 1024}},
        {"random, block 512 over 2048", Input::kRandom, false, {2, 2048, 512}},
        {"random, block 16, smallest legal", Input::kRandom, false, {1, 16, 16}},
        {"random, block 4096, largest legal", Input::kRandom, false, {1, 4096, 4096}},
        {"random, in place (x is y)", Input::kRandom, true, {2, 1024, 1024}},
        {"zeros, block 1024 over 5120", Input::kZeros, false, {2, 5120, 1024}},
    };

    for (const Case& test : cases) {
        const int rows = test.geometry.rows;
        const int width = test.geometry.width;
        const int block = test.geometry.block;
        const size_t count = static_cast<size_t>(rows) * width;

        std::vector<uint16_t> host_x(count);
        std::vector<float> signs(static_cast<size_t>(width));
        for (size_t i = 0; i < count; ++i) {
            if (test.input == Input::kIntegers) {
                // An integer in [-16, 16], exactly representable in FP16. It has
                // to go through the float conversion rather than have its
                // magnitude written into the mantissa field: a half with a zero
                // exponent is subnormal, so `sign | 16` is 16 * 2^-24 and not 16,
                // which would leave the case testing subnormals whose sums are
                // nowhere near the bound the header's argument rests on.
                const int value = static_cast<int>(next_random() % 33) - 16;
                host_x[i] = float_to_half(static_cast<float>(value));
            } else if (test.input == Input::kZeros) {
                // Both zeros, so the input itself carries the sign the vector
                // unit will not reproduce.
                host_x[i] = static_cast<uint16_t>(next_random() & 1u ? 0x0000u : 0x8000u);
            } else {
                host_x[i] = random_half();
            }
        }
        // The signs are the checkpoint's shape: one over the whole feature axis,
        // +-1 exactly, read once per run rather than once per row.
        for (int i = 0; i < width; ++i) {
            signs[static_cast<size_t>(i)] = (next_random() & 1u) != 0 ? 1.0f : -1.0f;
        }

        uint16_t* d_x = nullptr;
        uint16_t* d_y = nullptr;
        float* d_signs = nullptr;
        if (!pocket::device_malloc_into(d_x, count * sizeof(uint16_t)) ||
            !pocket::device_malloc_into(d_y, count * sizeof(uint16_t)) ||
            !pocket::device_malloc_into(d_signs, static_cast<size_t>(width) * sizeof(float))) {
            std::printf("[FAIL] device_malloc failed\n");
            return 1;
        }
        struct Free {
            uint16_t* x;
            uint16_t* y;
            float* s;
            ~Free() {
                pocket::device_free(x);
                pocket::device_free(y);
                pocket::device_free(s);
            }
        } free_guard{d_x, d_y, d_signs};

        if (!pocket::memcpy_h2d(d_x, host_x.data(), count * sizeof(uint16_t)) ||
            !pocket::memcpy_h2d(d_signs, signs.data(),
                                static_cast<size_t>(width) * sizeof(float)) ||
            !pocket::device_memset(d_y, 0, count * sizeof(uint16_t))) {
            std::printf("[FAIL] upload failed\n");
            return 1;
        }

        for (int direction = 0; direction < 2; ++direction) {
            const bool forward = direction == 0;
            const char* label = forward ? "forward" : "inverse";

            // The alias case writes over its own input, so the source is restored
            // from the host copy before the second direction runs.
            if (!pocket::memcpy_h2d(d_x, host_x.data(), count * sizeof(uint16_t))) {
                std::printf("[FAIL] re-upload failed\n");
                return 1;
            }

            const bool launched =
                forward ? pocket::qwen_hadamard_forward_f16_ascend(
                              d_x, d_signs, test.alias ? d_x : d_y, rows, width, block, nullptr)
                        : pocket::qwen_hadamard_inverse_f16_ascend(
                              d_x, d_signs, test.alias ? d_x : d_y, rows, width, block,
                              nullptr);
            const std::string name =
                std::string(label) + ", " + test.name;
            if (!launched) {
                expect(false, name + ": launch refused");
                continue;
            }
            pocket::device_synchronize();
            const std::string sync_error = pocket::device_last_error();
            if (!sync_error.empty()) {
                expect(false, name + ": device error: " + sync_error);
                continue;
            }

            const uint16_t* source = test.alias ? d_x : d_y;
            std::vector<uint16_t> got(count);
            if (!pocket::memcpy_d2h(got.data(), source, count * sizeof(uint16_t))) {
                expect(false, name + ": download failed");
                continue;
            }

            const std::vector<double> want =
                reference(host_x, signs, rows, width, block, forward);
            const Comparison result = compare(got, want);
            std::printf("  %-40s exact %zu/%zu  worst absolute %.3e of peak %.3g\n",
                        name.c_str(), static_cast<size_t>(result.exact), count,
                        result.worst, result.peak);

            // Equality is the verdict only where the arithmetic before the final
            // narrowing is exact on both sides. On full FP16 mantissas the FP32
            // result and the FP64 reference land on opposite sides of a
            // half-precision rounding boundary often enough to matter -- the
            // fp32 error is ~1e-6 relative against a boundary spacing of 5e-4,
            // so a few elements in a run of thousands will differ by one ulp of
            // the result -- and that says nothing about the kernel. The bound
            // below is what judges those.
            const bool exact_expected = test.input != Input::kRandom;
            if (exact_expected) {
                expect(result.exact == static_cast<int>(count),
                       name + ": element is not the reference's own value");
            }
            expect(result.within_tolerance == static_cast<int>(count),
                   name + ": element past both the per-element and the per-call bound");
        }
    }

    // The launcher's refusals. The kernel's UB tiles are sized statically, so a
    // `block` over the ceiling is a copy past the end of a tile -- and the refusals
    // have to match the CUDA launcher's, or a caller could tell the backends apart
    // by which arguments they reject.
    {
        std::vector<uint16_t> host_x(4096);
        std::vector<float> signs(1024, 1.0f);
        uint16_t* d_x = nullptr;
        uint16_t* d_y = nullptr;
        float* d_signs = nullptr;
        if (!pocket::device_malloc_into(d_x, host_x.size() * sizeof(uint16_t)) ||
            !pocket::device_malloc_into(d_y, host_x.size() * sizeof(uint16_t)) ||
            !pocket::device_malloc_into(d_signs, signs.size() * sizeof(float))) {
            std::printf("[FAIL] device_malloc failed\n");
            return 1;
        }
        struct Free {
            uint16_t* x;
            uint16_t* y;
            float* s;
            ~Free() {
                pocket::device_free(x);
                pocket::device_free(y);
                pocket::device_free(s);
            }
        } free_guard{d_x, d_y, d_signs};
        pocket::memcpy_h2d(d_x, host_x.data(), host_x.size() * sizeof(uint16_t));
        pocket::memcpy_h2d(d_signs, signs.data(), signs.size() * sizeof(float));

        // Each arm is written with the width that makes its bound the reason for
        // the refusal, rather than a neighbour that happens to be illegal for some
        // other reason.
        expect(!pocket::qwen_hadamard_forward_f16_ascend(nullptr, d_signs, d_y, 1, 1024, 1024,
                                                         nullptr),
               "null x is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, nullptr, d_y, 1, 1024, 1024,
                                                         nullptr),
               "null signs is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, nullptr, 1, 1024, 1024,
                                                         nullptr),
               "null y is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 0, 1024, 1024,
                                                         nullptr),
               "zero rows is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 0, 1024,
                                                         nullptr),
               "zero width is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 1024, 8,
                                                         nullptr),
               "a block under the copy granularity is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 8192, 8192,
                                                         nullptr),
               "a block over the tile size is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 1024, 768,
                                                         nullptr),
               "a non-power-of-two block is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 1000, 512,
                                                         nullptr),
               "a block that does not divide the width is refused");
        expect(!pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 0, 0, nullptr),
               "a zero block is refused");
        expect(pocket::qwen_hadamard_forward_f16_ascend(d_x, d_signs, d_y, 1, 1024, 1024,
                                                        nullptr),
               "the smallest legal geometry is accepted");
    }

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }
    return run(device);
}
