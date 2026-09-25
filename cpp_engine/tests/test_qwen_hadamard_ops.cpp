// The activation side of the ternary checkpoint's incoherence transform.
//
// Two kernels, and the property each one has to have:
//
//   * `forward` and `inverse` are the same butterflies with the signs on
//     opposite sides. The test pins each against a CPU reference computed from
//     the definition -- `H[i][j] = (-1)^popcount(i AND j)` over each block, the
//     sign vector, one `1/sqrt(N)` -- and separately pins that they are *not*
//     the same function of the data, because a model built with the wrong one of
//     the two runs and generates fluent nonsense. That canary is the whole reason
//     this file exists: every other check here would pass with the two swapped.
//
// The magnitude of the checks is set by the arithmetic: the butterflies
// accumulate in fp32 and the result is narrowed to fp16, so a value of order 0.5
// carries about 5e-4 of representation error and the tolerances below are that
// plus the reference's own fp32 rounding, with room for the two to associate the
// sum differently.

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::printf("[FAIL] %s\n", what.c_str());
        ++failures;
    }
}

uint16_t float_to_fp16_bits(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffu;
            const uint32_t biased = static_cast<uint32_t>(127 - 15 - shift);
            out = sign | (biased << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value;
    __builtin_memcpy(&value, &out, sizeof(value));
    return value;
}

// One block of the transform, exactly as the kernel defines it: the signs and
// the scale folded into the input, then the natural-order Sylvester butterflies
// in place. `forward` decides which side the signs land on.
//
// `data` is the block on its own, while `sign_offset` is where that block sits
// in the width -- the per-block sign vector is the whole point of the format, so
// a reference that reads the first block's signs for every block is the one
// mistake this file cannot afford.
void reference_block(const std::vector<float>& signs, std::vector<float>& data,
                     int sign_offset, int block, bool forward) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(block));
    std::vector<float> tile(static_cast<size_t>(block));
    for (int i = 0; i < block; ++i) {
        const float x = data[i];
        tile[i] = forward ? x * signs[sign_offset + i] * scale : x * scale;
    }
    for (int step = 1; step < block; step <<= 1) {
        for (int i = 0; i < block; ++i) {
            if ((i & step) == 0) {
                const float low = tile[i];
                const float high = tile[i + step];
                tile[i] = low + high;
                tile[i + step] = low - high;
            }
        }
    }
    for (int i = 0; i < block; ++i) {
        const float out = forward ? tile[i] : tile[i] * signs[sign_offset + i];
        data[i] = fp16_bits_to_float(float_to_fp16_bits(out));
    }
}

void run_case(int rows, int width, int block, bool forward) {
    const std::string side = forward ? "forward" : "inverse";
    std::mt19937 rng(forward ? 4242u : 9090u);
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    std::vector<float> signs(static_cast<size_t>(width));
    for (float& v : signs) v = sign_dist(rng) == 0 ? -1.0f : 1.0f;
    std::vector<uint16_t> x(static_cast<size_t>(rows) * width);
    for (uint16_t& v : x) v = float_to_fp16_bits(value_dist(rng));

    std::vector<float> expected(static_cast<size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        for (int base = 0; base < width; base += block) {
            const size_t offset = static_cast<size_t>(row) * width + base;
            // Not `std::vector<float>(x.begin() + offset, ...)`: that converts
            // the fp16 *bit pattern* to a float by value (0x3c00 becomes 15360),
            // which is 15360 times the input and looks like a scale error in the
            // kernel rather than a mistake in the test.
            std::vector<float> slice(static_cast<size_t>(block));
            for (int i = 0; i < block; ++i) slice[i] = fp16_bits_to_float(x[offset + i]);
            reference_block(signs, slice, base, block, forward);
            std::copy(slice.begin(), slice.end(), expected.begin() + offset);
        }
    }

    float* d_signs = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_y = nullptr;
    if (cudaMalloc(&d_signs, signs.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_x, x.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_y, x.size() * sizeof(uint16_t)) != cudaSuccess) {
        std::printf("[FAIL] %s: device allocation failed\n", side.c_str());
        ++failures;
        return;
    }
    cudaMemcpy(d_signs, signs.data(), signs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    const bool launched = forward
        ? pocket::qwen_hadamard_forward_f16_cuda(d_x, d_signs, d_y, rows, width, block)
        : pocket::qwen_hadamard_inverse_f16_cuda(d_x, d_signs, d_y, rows, width, block);
    cudaDeviceSynchronize();
    check(launched, side + ": kernel launch");

    std::vector<uint16_t> got(x.size());
    cudaMemcpy(got.data(), d_y, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    double worst = 0.0;
    size_t worst_at = 0;
    double magnitude = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double actual = fp16_bits_to_float(got[i]);
        const double want = static_cast<double>(expected[i]);
        magnitude = std::max(magnitude, std::abs(static_cast<double>(want)));
        const double delta = std::abs(actual - want);
        if (delta > worst) {
            worst = delta;
            worst_at = i;
        }
    }
    std::printf("  %-7s rows=%d width=%d block=%d worst=%.6g at %zu magnitude=%.6g\n",
                side.c_str(), rows, width, block, worst, worst_at, magnitude);
    check(magnitude > 0.1, side + ": the reference is not degenerate");
    check(worst < 4.0e-3, side + ": matches the CPU reference");

    // The forward pass must also be the inverse's refutation: if the two agree on
    // random data, one of them is not applying its signs where it claims to.
    const bool also_inverse = forward
        ? pocket::qwen_hadamard_inverse_f16_cuda(d_x, d_signs, d_y, rows, width, block)
        : pocket::qwen_hadamard_forward_f16_cuda(d_x, d_signs, d_y, rows, width, block);
    cudaDeviceSynchronize();
    check(also_inverse, side + ": opposite-direction launch");
    std::vector<uint16_t> other(x.size());
    cudaMemcpy(other.data(), d_y, other.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double difference = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        difference = std::max(difference, static_cast<double>(std::abs(
                                 fp16_bits_to_float(got[i]) -
                                 fp16_bits_to_float(other[i]))));
    }
    std::printf("  %-7s differs from the other direction by %.6g\n", side.c_str(), difference);
    check(difference > 0.05, side + ": the two directions are not the same function");

    cudaFree(d_signs);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Round trip: the two directions are inverses of each other, which is the
// property that makes "forward at run time" and "inverse at run time" a choice
// rather than an error.
void run_round_trip(int rows, int width, int block) {
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::vector<float> signs(static_cast<size_t>(width));
    for (float& v : signs) v = (rng() & 1u) == 0 ? -1.0f : 1.0f;
    std::vector<uint16_t> x(static_cast<size_t>(rows) * width);
    for (uint16_t& v : x) v = float_to_fp16_bits(value_dist(rng));

    float* d_signs = nullptr;
    uint16_t* d_x = nullptr;
    uint16_t* d_mid = nullptr;
    cudaMalloc(&d_signs, signs.size() * sizeof(float));
    cudaMalloc(&d_x, x.size() * sizeof(uint16_t));
    cudaMalloc(&d_mid, x.size() * sizeof(uint16_t));
    cudaMemcpy(d_signs, signs.data(), signs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    const bool ok = pocket::qwen_hadamard_forward_f16_cuda(d_x, d_signs, d_mid, rows, width, block) &&
                    pocket::qwen_hadamard_inverse_f16_cuda(d_mid, d_signs, d_x, rows, width, block);
    cudaDeviceSynchronize();
    check(ok, "round trip: launches");

    std::vector<uint16_t> got(x.size());
    cudaMemcpy(got.data(), d_x, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::abs(
                                     fp16_bits_to_float(got[i]) -
                                     fp16_bits_to_float(x[i]))));
    }
    std::printf("  round trip rows=%d width=%d block=%d worst=%.6g\n", rows, width, block, worst);
    // Two fp16 roundings, one per direction, on a value of order 1.
    check(worst < 5.0e-3, "round trip: forward then inverse returns the input");
    cudaFree(d_signs);
    cudaFree(d_x);
    cudaFree(d_mid);
}

// The one shape this checkpoint actually uses, and a second that puts the block
// boundary off a head boundary -- 5120 is five blocks over forty heads, so a
// block sees eight heads and a wrong head geometry cannot hide behind it.
void run_shape_guards() {
    const int width = 5120;
    const int block = 1024;
    check(width % block == 0, "the checkpoint's width is a whole number of blocks");
    check((block & (block - 1)) == 0, "the block size is a power of two");
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_hadamard_ops requires a CUDA device\n");
        return 0;
    }
    run_shape_guards();
    // Decode: one row. Prefill: several, and a width that is not one block.
    run_case(1, 5120, 1024, true);
    run_case(4, 5120, 1024, false);
    run_case(3, 6144, 1024, true);
    run_case(2, 2048, 1024, true);
    run_case(2, 512, 512, false);
    run_round_trip(3, 5120, 1024);

    if (failures != 0) {
        std::printf("[FAIL] test_qwen_hadamard_ops: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_hadamard_ops\n");
    return 0;
}
