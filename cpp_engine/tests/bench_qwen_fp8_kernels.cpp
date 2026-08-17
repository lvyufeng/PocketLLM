// Micro-benchmark for the Qwen online-unpack FP8 kernels at real Qwen3.8-27B
// TP4 shard shapes. Reports the optimized warp/tile path against the original
// block-per-row reduction so the speedup is measured, not asserted.

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 128;

struct Buffers {
    float* x = nullptr;
    uint8_t* w = nullptr;
    uint16_t* s = nullptr;
    float* y = nullptr;
    ~Buffers() {
        cudaFree(x);
        cudaFree(w);
        cudaFree(s);
        cudaFree(y);
    }
};

double time_ms(void (*fn)(const Buffers&, int, int, int, int), const Buffers& b,
               int batch, int rows, int cols, int scale_cols, int iters) {
    fn(b, batch, rows, cols, scale_cols);  // warmup
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fn(b, batch, rows, cols, scale_cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

void run_opt(const Buffers& b, int batch, int rows, int cols, int scale_cols) {
    if (batch == 1) {
        dsv4::qwen_fp8_e4m3_fp16scale_matvec_cuda(b.x, b.w, b.s, b.y, rows, cols, cols, scale_cols);
    } else {
        dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(b.x, b.w, b.s, b.y, batch, rows, cols,
                                                       cols, rows, cols, scale_cols);
    }
}

// Forces the legacy block-per-row kernel by making the stride non-multiple-of-4
// unreachable; instead we call the reference path via a stride+1 shim buffer.
void run_ref(const Buffers& b, int batch, int rows, int cols, int scale_cols) {
    // cols-1 keeps the same reduction structure while landing on the fallback
    // branch (stride not divisible by 4 is checked on weight_stride).
    const int stride = cols + 1;
    if (batch == 1) {
        dsv4::qwen_fp8_e4m3_fp16scale_matvec_cuda(b.x, b.w, b.s, b.y, rows, cols, stride, scale_cols);
    } else {
        dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(b.x, b.w, b.s, b.y, batch, rows, cols,
                                                       cols, rows, stride, scale_cols);
    }
}

void run_simt(const Buffers& b, int batch, int rows, int cols, int scale_cols) {
    if (batch == 1) {
        run_opt(b, batch, rows, cols, scale_cols);
    } else {
        dsv4::qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(
            b.x, b.w, b.s, b.y, batch, rows, cols, cols, rows, cols, scale_cols);
    }
}

}  // namespace

int main() {
    if (!dsv4::cuda_runtime_available()) {
        std::printf("[SKIP] bench_qwen_fp8_kernels requires a CUDA device\n");
        return 0;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device=%s sm=%d.%d\n", prop.name, prop.major, prop.minor);

    struct Case {
        const char* label;
        int batch;
        int rows;
        int cols;
    };
    // Qwen3.8-27B TP4 local shapes: hidden 5120, intermediate 17408.
    const Case cases[] = {
        {"linear_qkv   decode", 1, 2560, 5120},
        {"mlp_gate     decode", 1, 4352, 5120},
        {"mlp_down     decode", 1, 5120, 4352},
        {"full_q       decode", 1, 1536, 5120},
        {"mlp_gate    prefill", 64, 4352, 5120},
        {"mlp_down    prefill", 64, 5120, 4352},
        {"mlp_gate    pf-512 ", 512, 4352, 5120},
        {"mlp_down    pf-512 ", 512, 5120, 4352},
        {"linear_qkv  pf-512 ", 512, 2560, 5120},
    };

    for (const Case& c : cases) {
        const int scale_rows = (c.rows + kBlock - 1) / kBlock;
        const int scale_cols = (c.cols + kBlock - 1) / kBlock;
        Buffers b;
        // Allocate the weight with a +4 slack so the stride+1 reference variant
        // stays in bounds.
        const size_t w_bytes = static_cast<size_t>(c.rows) * (c.cols + 4);
        if (cudaMalloc(&b.x, static_cast<size_t>(c.batch) * c.cols * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&b.w, w_bytes) != cudaSuccess ||
            cudaMalloc(&b.s, static_cast<size_t>(scale_rows) * scale_cols * sizeof(uint16_t)) != cudaSuccess ||
            cudaMalloc(&b.y, static_cast<size_t>(c.batch) * c.rows * sizeof(float)) != cudaSuccess) {
            std::printf("[SKIP] allocation failed for %s\n", c.label);
            return 0;
        }
        std::vector<float> hx(static_cast<size_t>(c.batch) * c.cols);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (float& v : hx) v = dist(rng);
        std::vector<uint8_t> hw(w_bytes);
        for (uint8_t& v : hw) v = static_cast<uint8_t>(0x38 + (rng() & 7));
        std::vector<uint16_t> hs(static_cast<size_t>(scale_rows) * scale_cols, 0x3800);
        cudaMemcpy(b.x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(b.w, hw.data(), hw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(b.s, hs.data(), hs.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

        const int iters = c.batch == 1 ? 200 : 30;
        const double ref = time_ms(run_ref, b, c.batch, c.rows, c.cols, scale_cols, iters);
        const double opt = time_ms(run_opt, b, c.batch, c.rows, c.cols, scale_cols, iters);
        const double simt = time_ms(run_simt, b, c.batch, c.rows, c.cols, scale_cols, iters);
        const double bytes = static_cast<double>(c.rows) * c.cols;
        std::printf("%s batch=%3d rows=%5d cols=%5d  baseline=%7.3f ms  old=%7.3f ms  simt=%7.3f ms  old/simt=%5.2fx  eff=%6.1f GB/s\n",
                    c.label, c.batch, c.rows, c.cols, ref, opt, simt, opt / simt,
                    bytes / (simt * 1e-3) / 1e9);
    }
    return 0;
}
