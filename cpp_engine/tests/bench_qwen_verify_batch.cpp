// Measures the per-row cost of the Qwen3.8 TP4 projection kernels at the shard
// shapes that a DFlash2 verification actually issues. Speculative decoding can
// only win when an 8-row verify costs far less than 8 sequential 1-row decodes;
// this benchmark reports that ratio per projection so the speculative ceiling is
// measured instead of assumed.

#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda;
using dsv4::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda;

constexpr int kFp8Block = 128;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t to_half(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

struct Shape {
    const char* name;
    int out_rows;
    int cols;
};

// Real TP4 shard shapes for Qwen3.8-27B: hidden 5120, linear key/value shards
// 512/1536, full attention 3072 fused q/gate with 256-wide kv, MLP shard 4352.
const Shape kShapes[] = {
    {"linear.qkv", 2560, 5120},
    {"linear.z", 1536, 5120},
    {"linear.out", 5120, 1536},
    {"full.q_gate", 3072, 5120},
    {"full.k", 256, 5120},
    {"full.v", 256, 5120},
    {"full.out", 5120, 1536},
    {"mlp.gate", 4352, 5120},
    {"mlp.up", 4352, 5120},
    {"mlp.down", 5120, 4352},
};

struct DeviceBuffers {
    uint16_t* x = nullptr;
    uint8_t* weight = nullptr;
    uint16_t* scale = nullptr;
    uint16_t* y = nullptr;
    ~DeviceBuffers() {
        cudaFree(x);
        cudaFree(weight);
        cudaFree(scale);
        cudaFree(y);
    }
};

double time_rows(const DeviceBuffers& buffers, int batch, int out_rows,
                 int cols, int scale_stride, int iters) {
    auto launch = [&]() {
        if (batch == 1) {
            return qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                buffers.x, buffers.weight, buffers.scale, buffers.y, out_rows,
                cols, cols, scale_stride);
        }
        return qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            buffers.x, buffers.weight, buffers.scale, buffers.y, batch, out_rows,
            cols, cols, out_rows, cols, scale_stride);
    };
    if (!launch()) {
        std::fprintf(stderr, "launch failed batch=%d rows=%d cols=%d\n", batch,
                     out_rows, cols);
        std::exit(1);
    }
    check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) (void)launch();
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

// The DFlash2 drafter runs FP16 weights with FP32 output. Its LM head is a tall
// skinny GEMM (local vocab x hidden) at 7 rows, where cuBLAS wastes most of its
// tile. Compare it against the warp-per-output-channel kernel.
struct F16Shape {
    const char* name;
    int out_rows;
    int cols;
};

const F16Shape kF16Shapes[] = {
    {"dflash2.lm_head", 62080, 5120},
    {"dflash2.qkv", 4096, 5120},
    {"dflash2.gate", 4352, 5120},
    {"dflash2.down", 5120, 4352},
    {"dflash2.selector", 256, 5120},
};

struct F16Buffers {
    uint16_t* x = nullptr;
    uint16_t* weight = nullptr;
    float* y = nullptr;
    ~F16Buffers() {
        cudaFree(x);
        cudaFree(weight);
        cudaFree(y);
    }
};

double time_f16(const F16Buffers& buffers, int batch, int out_rows, int cols,
                bool cublas, int iters) {
    auto launch = [&]() {
        if (cublas) {
            return dsv4::qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                buffers.x, buffers.weight, buffers.y, batch, out_rows, cols,
                cols, out_rows, cols);
        }
        // Cost probe for the warp-per-output-channel path. The FP16-output entry
        // point is the only one that currently allows the small-batch kernel; the
        // arithmetic and memory traffic match the FP32-output variant.
        return dsv4::qwen_fp16_matmul_rows_f16_cuda(
            buffers.x, buffers.weight,
            reinterpret_cast<uint16_t*>(buffers.y), batch, out_rows, cols, cols,
            out_rows, cols);
    };
    if (!launch()) {
        std::fprintf(stderr, "f16 launch failed batch=%d rows=%d cols=%d cublas=%d\n",
                     batch, out_rows, cols, cublas ? 1 : 0);
        std::exit(1);
    }
    check(cudaDeviceSynchronize(), "f16 warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) (void)launch();
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

// The DFlash2 context projector is the widest reduction in the drafter:
// taps [rows, 5 * 5120] times fc.weight [5120, 25600]. Real TP4 profiles show it
// dominating prepare_context, so compare the two available GEMM entry points at
// the exact shape.
void bench_context_projection(int iters, std::mt19937& rng) {
    constexpr int kWidth = 25600;
    constexpr int kHidden = 5120;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int row_counts[] = {1, 2, 3, 8};
    uint16_t* x = nullptr;
    uint16_t* weight = nullptr;
    uint16_t* y = nullptr;
    float* y32 = nullptr;
    const size_t weight_elements =
        static_cast<size_t>(kHidden) * kWidth;
    check(cudaMalloc(&x, static_cast<size_t>(8) * kWidth * sizeof(uint16_t)),
          "malloc taps");
    check(cudaMalloc(&weight, weight_elements * sizeof(uint16_t)),
          "malloc projector");
    check(cudaMalloc(&y, static_cast<size_t>(8) * kHidden * sizeof(uint16_t)),
          "malloc projected");
    check(cudaMalloc(&y32, static_cast<size_t>(8) * kHidden * sizeof(float)),
          "malloc projected f32");
    std::vector<uint16_t> host(static_cast<size_t>(8) * kWidth);
    for (uint16_t& value : host) value = to_half(dist(rng) * 0.1f);
    check(cudaMemcpy(x, host.data(), host.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy taps");
    std::vector<uint16_t> host_weight(weight_elements);
    for (uint16_t& value : host_weight) value = to_half(dist(rng) * 0.05f);
    check(cudaMemcpy(weight, host_weight.data(),
                     host_weight.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy projector");
    const double bytes =
        static_cast<double>(weight_elements) * sizeof(uint16_t);
    std::printf("dflash2_context_projection hidden=%d width=%d weight=%.1f MiB "
                "iters=%d\n", kHidden, kWidth, bytes / (1024.0 * 1024.0), iters);
    for (int rows : row_counts) {
        auto time_one = [&](int which) {
            auto launch = [&]() {
                if (which == 0) {
                    return dsv4::qwen_dspark_fp16_gemm_rows_f16_cuda(
                        x, weight, y, rows, kHidden, kWidth);
                }
                return dsv4::qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                    x, weight, y32, rows, kHidden, kWidth, kWidth, kHidden,
                    kWidth);
            };
            if (!launch()) {
                std::fprintf(stderr, "context projection launch failed rows=%d "
                             "which=%d\n", rows, which);
                std::exit(1);
            }
            check(cudaDeviceSynchronize(), "context warmup sync");
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            check(cudaEventCreate(&start), "event create");
            check(cudaEventCreate(&stop), "event create");
            check(cudaEventRecord(start), "event record");
            for (int i = 0; i < iters; ++i) (void)launch();
            check(cudaEventRecord(stop), "event record");
            check(cudaEventSynchronize(stop), "event sync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return static_cast<double>(ms) / iters;
        };
        const double direct = time_one(0);
        const double staged = time_one(1);
        std::printf("  rows=%d direct_TN=%.4f ms (%.0f GB/s) "
                    "staged_f32=%.4f ms (%.0f GB/s) speedup=%.3f\n",
                    rows, direct, bytes / (direct * 1.0e6), staged,
                    bytes / (staged * 1.0e6),
                    staged > 0.0 ? direct / staged : 0.0);
    }
    cudaFree(x);
    cudaFree(weight);
    cudaFree(y);
    cudaFree(y32);
}

void bench_f16_draft(int draft_rows, int iters, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::printf("dflash2_draft_projections rows=%d iters=%d\n", draft_rows, iters);
    double cublas_total = 0.0;
    double warp_total = 0.0;
    for (const F16Shape& shape : kF16Shapes) {
        F16Buffers buffers;
        const size_t x_elements = static_cast<size_t>(draft_rows) * shape.cols;
        const size_t weight_elements =
            static_cast<size_t>(shape.out_rows) * shape.cols;
        const size_t y_elements =
            static_cast<size_t>(draft_rows) * shape.out_rows;
        check(cudaMalloc(&buffers.x, x_elements * sizeof(uint16_t)), "malloc x");
        check(cudaMalloc(&buffers.weight, weight_elements * sizeof(uint16_t)),
              "malloc weight");
        check(cudaMalloc(&buffers.y, y_elements * sizeof(float)), "malloc y");
        std::vector<uint16_t> host(x_elements);
        for (uint16_t& value : host) value = to_half(dist(rng) * 0.1f);
        check(cudaMemcpy(buffers.x, host.data(), host.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy x");
        std::vector<uint16_t> host_weight(weight_elements);
        for (uint16_t& value : host_weight) value = to_half(dist(rng) * 0.05f);
        check(cudaMemcpy(buffers.weight, host_weight.data(),
                         host_weight.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy weight");
        const double cublas_ms =
            time_f16(buffers, draft_rows, shape.out_rows, shape.cols, true, iters);
        const double warp_ms =
            time_f16(buffers, draft_rows, shape.out_rows, shape.cols, false, iters);
        cublas_total += cublas_ms;
        warp_total += warp_ms;
        const double bytes =
            static_cast<double>(weight_elements) * sizeof(uint16_t);
        std::printf(
            "  %-16s out=%6d cols=%5d cublas=%.4f ms (%.0f GB/s) "
            "warp=%.4f ms (%.0f GB/s) speedup=%.3f\n",
            shape.name, shape.out_rows, shape.cols, cublas_ms,
            bytes / (cublas_ms * 1.0e6), warp_ms, bytes / (warp_ms * 1.0e6),
            warp_ms > 0.0 ? cublas_ms / warp_ms : 0.0);
    }
    std::printf("  total cublas=%.4f ms warp=%.4f ms draft_speedup=%.3f\n",
                cublas_total, warp_total,
                warp_total > 0.0 ? cublas_total / warp_total : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace dsv4;
    int iters = 50;
    int verify_rows = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--rows" && i + 1 < argc) verify_rows = std::atoi(argv[++i]);
    }
    if (iters <= 0 || verify_rows <= 1) {
        std::fprintf(stderr, "usage: bench_qwen_verify_batch [--iters N] [--rows R>=2]\n");
        return 1;
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    double serial_total = 0.0;
    double verify_total = 0.0;
    std::printf("qwen_verify_batch rows=%d iters=%d\n", verify_rows, iters);
    for (const Shape& shape : kShapes) {
        const int scale_stride = (shape.cols + kFp8Block - 1) / kFp8Block;
        DeviceBuffers buffers;
        const size_t x_elements =
            static_cast<size_t>(verify_rows) * shape.cols;
        const size_t weight_elements =
            static_cast<size_t>(shape.out_rows) * shape.cols;
        const size_t scale_elements =
            static_cast<size_t>(shape.out_rows) * scale_stride;
        const size_t y_elements =
            static_cast<size_t>(verify_rows) * shape.out_rows;
        check(cudaMalloc(&buffers.x, x_elements * sizeof(uint16_t)), "malloc x");
        check(cudaMalloc(&buffers.weight, weight_elements), "malloc weight");
        check(cudaMalloc(&buffers.scale, scale_elements * sizeof(uint16_t)),
              "malloc scale");
        check(cudaMalloc(&buffers.y, y_elements * sizeof(uint16_t)), "malloc y");

        std::vector<uint16_t> host_x(x_elements);
        for (uint16_t& value : host_x) value = to_half(dist(rng) * 0.1f);
        std::vector<uint8_t> host_weight(weight_elements);
        // 32..223 stays inside finite E4M3 codes and avoids NaN encodings.
        for (uint8_t& code : host_weight) {
            code = static_cast<uint8_t>(32 + (rng() % 192));
        }
        std::vector<uint16_t> host_scale(scale_elements);
        for (uint16_t& value : host_scale) value = to_half(0.02f);
        check(cudaMemcpy(buffers.x, host_x.data(),
                         host_x.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy x");
        check(cudaMemcpy(buffers.weight, host_weight.data(), host_weight.size(),
                         cudaMemcpyHostToDevice), "copy weight");
        check(cudaMemcpy(buffers.scale, host_scale.data(),
                         host_scale.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy scale");

        const double one_row = time_rows(buffers, 1, shape.out_rows, shape.cols,
                                         scale_stride, iters);
        const double batched = time_rows(buffers, verify_rows, shape.out_rows,
                                         shape.cols, scale_stride, iters);
        const double serial = one_row * verify_rows;
        serial_total += serial;
        verify_total += batched;
        std::printf(
            "  %-12s out=%5d cols=%5d row1=%.4f ms serial%d=%.4f ms "
            "batch%d=%.4f ms ratio=%.3f\n",
            shape.name, shape.out_rows, shape.cols, one_row, verify_rows,
            serial, verify_rows, batched,
            batched > 0.0 ? serial / batched : 0.0);
    }
    std::printf(
        "  total serial%d=%.4f ms batch%d=%.4f ms projection_speedup=%.3f\n",
        verify_rows, serial_total, verify_rows, verify_total,
        verify_total > 0.0 ? serial_total / verify_total : 0.0);
    // DFlash2 proposes verify_rows - 1 draft rows per block.
    bench_f16_draft(verify_rows - 1, iters, rng);
    bench_context_projection(iters, rng);
    return 0;
}
