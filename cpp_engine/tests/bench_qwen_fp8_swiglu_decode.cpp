// Decode-shape A/B for the fused FP8 gate/up SwiGLU matvec: the scalar column
// loop against the uchar4 vectorized variant. Reports per-call time, the
// implied weight-stream bandwidth, and each variant's error against a
// double-precision reference so a reduction-order change can be judged on
// accuracy rather than only on speed.
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qwen_cuda_ops.hpp"

namespace {

constexpr int kBlock = 128;

float fp8_e4m3_to_float_host(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code >> 7) << 31;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    if (exponent == 0) {
        if (mantissa == 0) {
            float zero;
            const uint32_t bits = sign;
            std::memcpy(&zero, &bits, sizeof(zero));
            return zero;
        }
        const float value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
        return (sign != 0u) ? -value : value;
    }
    if (exponent == 0xf && mantissa == 0x7) {
        return (sign != 0u) ? -std::numeric_limits<float>::quiet_NaN()
                            : std::numeric_limits<float>::quiet_NaN();
    }
    const float value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                                   exponent - 7);
    return (sign != 0u) ? -value : value;
}

uint16_t float_to_half_host(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                (mantissa >> 13));
}

float half_to_float_host(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                ++shift;
            }
            m &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

double silu_double(double v) { return v / (1.0 + std::exp(-v)); }

struct Case {
    int rows;
    int cols;
    const char* label;
};

}  // namespace

int main() {
    // Qwen3.8-27B-FP8 TP4: hidden 5120, intermediate 17408 sharded to 4352.
    const std::vector<Case> cases = {
        {4352, 5120, "tp4 mlp gate/up"},
        {17408, 5120, "tp1 mlp gate/up"},
    };
    const char* iters_env = std::getenv("BENCH_ITERS");
    const int iters = iters_env != nullptr && std::atoi(iters_env) > 0
                          ? std::atoi(iters_env) : 200;

    std::mt19937 rng(1234);
    // 0x7f and 0xff are E4M3 NaN. A single NaN weight makes every comparison
    // below false, and std::max(0.0, NaN) returns 0.0, so the error column
    // would silently read zero.
    // Codes are drawn from the middle of the E4M3 exponent range, which avoids
    // both NaN (0x7f/0xff) and magnitudes that would overflow the FP16 output
    // once 5120 columns are accumulated.
    auto sample_code = [&](std::mt19937& gen) {
        std::uniform_int_distribution<int> sign(0, 1);
        std::uniform_int_distribution<int> exponent(1, 4);
        std::uniform_int_distribution<int> mantissa(0, 7);
        return static_cast<uint8_t>((sign(gen) << 7) | (exponent(gen) << 3) |
                                    mantissa(gen));
    };

    std::printf("%-18s %6s %6s %10s %10s %10s %10s %8s %8s %8s %8s %9s\n",
                "case", "rows", "cols", "scalar_ms", "vec4_ms", "vec8_ms",
                "vec16_ms", "sc_GB/s", "v4_GB/s", "v8_GB/s", "v16_GB/s",
                "worst_err");

    for (const Case& c : cases) {
        const int scale_rows = (c.rows + kBlock - 1) / kBlock;
        const int scale_cols = (c.cols + kBlock - 1) / kBlock;
        std::vector<uint16_t> x(c.cols);
        std::vector<uint8_t> gw(static_cast<size_t>(c.rows) * c.cols);
        std::vector<uint8_t> uw(gw.size());
        std::vector<uint16_t> gs(static_cast<size_t>(scale_rows) * scale_cols);
        std::vector<uint16_t> us(gs.size());
        for (uint16_t& v : x) v = float_to_half_host(
            std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng));
        for (uint8_t& v : gw) v = sample_code(rng);
        for (uint8_t& v : uw) v = sample_code(rng);
        for (uint16_t& v : gs) v = float_to_half_host(
            std::uniform_real_distribution<float>(0.01f, 0.05f)(rng));
        for (uint16_t& v : us) v = float_to_half_host(
            std::uniform_real_distribution<float>(0.01f, 0.05f)(rng));

        uint16_t* d_x = nullptr;
        uint8_t* d_gw = nullptr;
        uint8_t* d_uw = nullptr;
        uint16_t* d_gs = nullptr;
        uint16_t* d_us = nullptr;
        uint16_t* d_y = nullptr;
        if (cudaMalloc(&d_x, x.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_gw, gw.size()) != cudaSuccess ||
            cudaMalloc(&d_uw, uw.size()) != cudaSuccess ||
            cudaMalloc(&d_gs, gs.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_us, us.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_y, static_cast<size_t>(c.rows) * 2) != cudaSuccess) {
            std::printf("[FAIL] alloc\n");
            return 1;
        }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gw, gw.data(), gw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uw, uw.data(), uw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gs, gs.data(), gs.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_us, us.data(), us.size() * 2, cudaMemcpyHostToDevice);

        // Double-precision reference in plain column order.
        std::vector<double> ref(c.rows);
        for (int row = 0; row < c.rows; ++row) {
            double g = 0.0;
            double u = 0.0;
            for (int col = 0; col < c.cols; ++col) {
                const double xv = half_to_float_host(x[col]);
                const size_t wi = static_cast<size_t>(row) * c.cols + col;
                const size_t si = static_cast<size_t>(row / kBlock) * scale_cols +
                                  col / kBlock;
                g += xv * fp8_e4m3_to_float_host(gw[wi]) * half_to_float_host(gs[si]);
                u += xv * fp8_e4m3_to_float_host(uw[wi]) * half_to_float_host(us[si]);
            }
            ref[row] = silu_double(g) * u;
        }

        auto run = [&](bool vectorized, int cols_per_lane, double* ms,
                       double* err) {
            setenv("QWEN_FP8_F16_SWIGLU_VECTORIZE", vectorized ? "1" : "0", 1);
            char width[8];
            std::snprintf(width, sizeof(width), "%d", cols_per_lane);
            setenv("QWEN_FP8_F16_SWIGLU_COLS_PER_LANE", width, 1);
            // The dispatch caches nothing, so the env var takes effect per call.
            auto launch = [&] {
                return dsv4::qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                    d_x, d_gw, d_gs, d_uw, d_us, d_y, c.rows, c.cols, c.cols,
                    scale_cols, nullptr);
            };
            if (!launch() || cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("[FAIL] launch vectorized=%d\n",
                            vectorized ? 1 : 0);
                std::exit(1);
            }
            std::vector<uint16_t> y(c.rows);
            cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
            double worst = 0.0;
            for (int row = 0; row < c.rows; ++row) {
                const double got = half_to_float_host(y[row]);
                const double denom = std::max(1e-3, std::fabs(ref[row]));
                worst = std::max(worst, std::fabs(got - ref[row]) / denom);
            }
            *err = worst;

            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *ms = static_cast<double>(elapsed) / iters;
        };

        double scalar_ms = 0.0;
        double vec4_ms = 0.0;
        double vec8_ms = 0.0;
        double vec16_ms = 0.0;
        double scalar_err = 0.0;
        double vec4_err = 0.0;
        double vec8_err = 0.0;
        double vec16_err = 0.0;
        run(false, 4, &scalar_ms, &scalar_err);
        run(true, 4, &vec4_ms, &vec4_err);
        run(true, 8, &vec8_ms, &vec8_err);
        run(true, 16, &vec16_ms, &vec16_err);

        // Two FP8 weight planes dominate the stream.
        const double bytes = 2.0 * static_cast<double>(c.rows) * c.cols;
        std::printf("%-18s %6d %6d %10.4f %10.4f %10.4f %10.4f "
                    "%8.1f %8.1f %8.1f %8.1f %9.2e\n",
                    c.label, c.rows, c.cols, scalar_ms, vec4_ms, vec8_ms,
                    vec16_ms, bytes / (scalar_ms * 1e-3) / 1e9,
                    bytes / (vec4_ms * 1e-3) / 1e9,
                    bytes / (vec8_ms * 1e-3) / 1e9,
                    bytes / (vec16_ms * 1e-3) / 1e9,
                    std::max(std::max(scalar_err, vec4_err),
                             std::max(vec8_err, vec16_err)));

        cudaFree(d_x);
        cudaFree(d_gw);
        cudaFree(d_uw);
        cudaFree(d_gs);
        cudaFree(d_us);
        cudaFree(d_y);
    }

    // The plain matvec is the other large decode kernel: the down projection
    // (5120 x 4352 per rank) plus every attention projection.
    struct MatvecCase {
        int rows;
        int cols;
        const char* label;
    };
    const std::vector<MatvecCase> matvec_cases = {
        {5120, 4352, "tp4 mlp down"},
        {1536, 5120, "tp4 attn qkv-ish"},
    };
    std::printf("\n%-18s %6s %6s %10s %10s %8s %8s %9s\n", "matvec case",
                "rows", "cols", "vec4_ms", "vec8_ms", "v4_GB/s", "v8_GB/s",
                "worst_err");
    for (const MatvecCase& c : matvec_cases) {
        const int scale_rows = (c.rows + kBlock - 1) / kBlock;
        const int scale_cols = (c.cols + kBlock - 1) / kBlock;
        std::vector<uint16_t> x(c.cols);
        std::vector<uint8_t> w(static_cast<size_t>(c.rows) * c.cols);
        std::vector<uint16_t> s(static_cast<size_t>(scale_rows) * scale_cols);
        std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> s_dist(0.01f, 0.05f);
        for (uint16_t& value : x) value = float_to_half_host(x_dist(rng));
        for (uint8_t& value : w) value = sample_code(rng);
        for (uint16_t& value : s) value = float_to_half_host(s_dist(rng));

        uint16_t* d_x = nullptr;
        uint8_t* d_w = nullptr;
        uint16_t* d_s = nullptr;
        uint16_t* d_y = nullptr;
        cudaMalloc(&d_x, x.size() * 2);
        cudaMalloc(&d_w, w.size());
        cudaMalloc(&d_s, s.size() * 2);
        cudaMalloc(&d_y, static_cast<size_t>(c.rows) * 2);
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, s.data(), s.size() * 2, cudaMemcpyHostToDevice);

        std::vector<double> ref(c.rows);
        for (int row = 0; row < c.rows; ++row) {
            double sum = 0.0;
            for (int col = 0; col < c.cols; ++col) {
                const size_t wi = static_cast<size_t>(row) * c.cols + col;
                const size_t si =
                    static_cast<size_t>(row / kBlock) * scale_cols + col / kBlock;
                sum += half_to_float_host(x[col]) *
                       fp8_e4m3_to_float_host(w[wi]) * half_to_float_host(s[si]);
            }
            ref[row] = sum;
        }

        auto run = [&](int cols_per_lane, double* ms, double* err) {
            setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
            setenv("QWEN_FP8_F16_COLS_PER_LANE",
                   cols_per_lane == 4 ? "4" : "8", 1);
            auto launch = [&] {
                return dsv4::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    d_x, d_w, d_s, d_y, c.rows, c.cols, c.cols, scale_cols,
                    nullptr);
            };
            if (!launch() || cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("[FAIL] matvec launch cols_per_lane=%d\n",
                            cols_per_lane);
                std::exit(1);
            }
            std::vector<uint16_t> y(c.rows);
            cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
            double worst = 0.0;
            for (int row = 0; row < c.rows; ++row) {
                const double denom = std::max(1e-3, std::fabs(ref[row]));
                worst = std::max(
                    worst, std::fabs(half_to_float_host(y[row]) - ref[row]) / denom);
            }
            *err = worst;

            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *ms = static_cast<double>(elapsed) / iters;
        };

        double vec4_ms = 0.0;
        double vec8_ms = 0.0;
        double vec4_err = 0.0;
        double vec8_err = 0.0;
        run(4, &vec4_ms, &vec4_err);
        run(8, &vec8_ms, &vec8_err);
        unsetenv("QWEN_FP8_F16_COLS_PER_LANE");
        unsetenv("QWEN_FP8_F16_VECTORIZE");

        const double bytes = static_cast<double>(c.rows) * c.cols;
        std::printf("%-18s %6d %6d %10.4f %10.4f %8.1f %8.1f %9.2e\n", c.label,
                    c.rows, c.cols, vec4_ms, vec8_ms,
                    bytes / (vec4_ms * 1e-3) / 1e9,
                    bytes / (vec8_ms * 1e-3) / 1e9,
                    std::max(vec4_err, vec8_err));

        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_s);
        cudaFree(d_y);
    }
    std::printf("[PASS] bench_qwen_fp8_swiglu_decode\n");
    return 0;
}
