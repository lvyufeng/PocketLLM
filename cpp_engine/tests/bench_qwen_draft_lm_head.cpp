// Microbenchmark for the drafter LM head shape (batch 7, vocab 62080, hidden
// 5120). The generic FP32 matmul launches a rows x batch grid and therefore
// re-reads the whole weight matrix once per draft row. This measures that
// against the weight-reuse small-batch kernel and cuBLAS on the same shape.
#include "qwen_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace dsv4;

namespace {

uint16_t to_half(float value) {
    uint16_t out = 0;
    const __half h = __float2half(value);
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Timing {
    double ms;
    double gbps;
};

template <typename Fn>
Timing time_kernel(Fn fn, int rows, int cols, int iters = 30) {
    for (int i = 0; i < 5; ++i) {
        if (!fn()) {
            std::printf("    launch failed\n");
            return {0.0, 0.0};
        }
    }
    cudaDeviceSynchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto started = std::chrono::steady_clock::now();
        fn();
        cudaDeviceSynchronize();
        samples.push_back(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count() * 1000.0);
    }
    const double ms = median(samples);
    const double bytes = static_cast<double>(rows) * cols * sizeof(uint16_t);
    return {ms, bytes / (ms / 1000.0) / 1e9};
}

}  // namespace

int main() {
    const int batch = 7;
    const int rows = 62080;   // local vocab shard
    const int cols = 5120;    // hidden
    const int x_stride = cols;
    const int y_stride = rows;
    const int weight_stride = cols;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<uint16_t> host_x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint16_t> host_w(static_cast<size_t>(rows) * weight_stride);
    for (uint16_t& v : host_x) v = to_half(dist(rng));
    for (uint16_t& v : host_w) v = to_half(dist(rng));

    uint16_t* d_x = nullptr;
    uint16_t* d_w = nullptr;
    float* d_generic = nullptr;
    float* d_cublas = nullptr;
    if (cudaMalloc(&d_x, host_x.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_w, host_w.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_generic, static_cast<size_t>(batch) * y_stride * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_cublas, static_cast<size_t>(batch) * y_stride * sizeof(float)) != cudaSuccess) {
        std::printf("allocation failed\n");
        return 1;
    }
    cudaMemcpy(d_x, host_x.data(), host_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, host_w.data(), host_w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    const double weight_mb = static_cast<double>(rows) * cols * sizeof(uint16_t) / 1e6;
    std::printf("draft LM head batch=%d rows=%d cols=%d weight=%.1f MB\n",
                batch, rows, cols, weight_mb);

    // Current production path: FP32 output, small-batch reuse disabled.
    const Timing generic = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32(
            d_x, d_w, d_generic, batch, rows, cols, x_stride, y_stride,
            weight_stride);
    }, rows, cols);
    std::printf("  generic (rows x batch grid)  %7.2f ms  %6.1f GB/s effective\n",
                generic.ms, generic.gbps);

    const Timing cublas = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
            d_x, d_w, d_cublas, batch, rows, cols, x_stride, y_stride,
            weight_stride);
    }, rows, cols);
    std::printf("  cublasGemmEx                 %7.2f ms  %6.1f GB/s effective\n",
                cublas.ms, cublas.gbps);

    if (generic.ms > 0.0 && cublas.ms > 0.0) {
        std::printf("  cublas speedup %.2fx\n", generic.ms / cublas.ms);
    }

    // Numerical agreement between the two paths on this shape.
    std::vector<float> a(static_cast<size_t>(batch) * y_stride);
    std::vector<float> b(a.size());
    cudaMemcpy(a.data(), d_generic, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_cublas, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    size_t argmax_disagreements = 0;
    for (int sample = 0; sample < batch; ++sample) {
        int best_a = 0;
        int best_b = 0;
        for (int row = 0; row < rows; ++row) {
            const size_t at = static_cast<size_t>(sample) * y_stride + row;
            worst = std::max(worst, std::fabs(static_cast<double>(a[at]) - b[at]));
            if (a[at] > a[static_cast<size_t>(sample) * y_stride + best_a]) best_a = row;
            if (b[at] > b[static_cast<size_t>(sample) * y_stride + best_b]) best_b = row;
        }
        if (best_a != best_b) ++argmax_disagreements;
    }
    std::printf("  max |generic - cublas| = %.3e, argmax disagreements = %zu/%d\n",
                worst, argmax_disagreements, batch);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_generic);
    cudaFree(d_cublas);
    return 0;
}
