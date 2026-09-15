// All-reduce bandwidth at prefill chunk sizes.
//
// bench_qwen_tp_allreduce covers the decode/draft shapes (1 and 8 rows). Long
// prefill reduces whole chunks: 4096 rows x 5120 hidden is 41.9 MB per call and
// two calls per layer, so the collective moves 85.9 GB over a 64-layer 65K
// prefill. That is a different bandwidth regime from an 80 KB decode reduce, and
// it is what decides whether the exposed comm cost is a hardware floor.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tp_comm.hpp"

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int world = 4;
    int rank = 0;
    int device = 0;
    int iters = 30;
    int hidden = 5120;
    std::string id_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tp-world" && i + 1 < argc) world = std::atoi(argv[++i]);
        else if (arg == "--tp-rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--hidden" && i + 1 < argc) hidden = std::atoi(argv[++i]);
        else if (arg == "--nccl-id-path" && i + 1 < argc) id_path = argv[++i];
    }
    if (id_path.empty() || world <= 1 || iters <= 0 || hidden <= 0) {
        std::fprintf(stderr,
                     "usage: bench_qwen_prefill_allreduce --nccl-id-path P "
                     "--tp-world N --tp-rank R [--device D] [--iters N] "
                     "[--hidden H]\n");
        return 1;
    }
    if (!dsv4::nccl_available()) {
        std::fprintf(stderr, "this build has no NCCL support\n");
        return 1;
    }
    check(cudaSetDevice(device), "cudaSetDevice");

    const int row_counts[] = {512, 1024, 2048, 4096, 8192};
    for (int rows : row_counts) {
        const size_t count = static_cast<size_t>(rows) * hidden;
        uint16_t* buffer = nullptr;
        check(cudaMalloc(&buffer, count * sizeof(uint16_t)), "cudaMalloc");
        check(cudaMemset(buffer, 0, count * sizeof(uint16_t)), "cudaMemset");

        for (int i = 0; i < 5; ++i) {
            dsv4::nccl_all_reduce_sum_f16_inplace(
                world, rank, device, id_path.c_str(), buffer, count);
        }
        check(cudaDeviceSynchronize(), "warmup sync");

        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            dsv4::nccl_all_reduce_sum_f16_inplace(
                world, rank, device, id_path.c_str(), buffer, count);
        }
        check(cudaDeviceSynchronize(), "timed sync");
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();

        const double per_call_ms = 1e3 * seconds / iters;
        const double payload_mb = count * sizeof(uint16_t) / 1e6;
        // Ring all-reduce moves 2(p-1)/p of the payload in each direction per
        // rank, so bus bandwidth is the comparable figure against a link rating.
        const double bus_gbs =
            payload_mb * 1e6 * 2.0 * (world - 1) / world / (per_call_ms * 1e-3) / 1e9;
        if (rank == 0) {
            std::printf("rows=%5d payload=%7.1f MB  %8.3f ms/call  "
                        "algbw=%6.2f GB/s  busbw=%6.2f GB/s\n",
                        rows, payload_mb, per_call_ms,
                        payload_mb * 1e6 / (per_call_ms * 1e-3) / 1e9, bus_gbs);
            std::fflush(stdout);
        }
        check(cudaFree(buffer), "cudaFree");
    }
    return 0;
}
