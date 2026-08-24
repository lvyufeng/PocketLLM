// Microbenchmark for the GQA prefill attention kernel at the real TP4 shape.
//
// full_attention is the largest single phase of long-prompt prefill (4.24 s of
// 9.33 s at 8192), and every CTA streams the whole KV history for its KV head,
// so the kernel is dominated by history traffic. Running the real chunk shape
// here is far cheaper than a full four-rank prefill for each candidate.

#include "cuda_ops.hpp"
#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// Qwen3.8 TP4 shard of full attention. The model has 24 q heads over 4 kv heads
// at head_dim 256, and TP4 shards the heads, so each rank runs 6 q heads over a
// single kv head. head_dim is not sharded.
constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;

double time_case(int rows, int position_offset, int max_context, int iters) {
    std::mt19937 rng(97531);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int context_len = position_offset + rows;
    const size_t q_elements = static_cast<size_t>(rows) * kQHeads * kHeadDim;
    const size_t kv_elements =
        static_cast<size_t>(max_context) * kKvHeads * kHeadDim;
    std::vector<uint16_t> host_q(q_elements);
    std::vector<uint16_t> host_kv(kv_elements, 0);
    for (uint16_t& value : host_q) {
        value = __half_as_ushort(__float2half(dist(rng)));
    }
    for (size_t i = 0;
         i < static_cast<size_t>(context_len) * kKvHeads * kHeadDim; ++i) {
        host_kv[i] = __half_as_ushort(__float2half(dist(rng)));
    }

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_out = nullptr;
    if (cudaMalloc(&d_q, q_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_out, q_elements * sizeof(uint16_t)) != cudaSuccess) {
        std::printf("[SKIP] allocation failed\n");
        return -1.0;
    }
    cudaMemcpy(d_q, host_q.data(), q_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, host_kv.data(), kv_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, host_kv.data(), kv_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);

    for (int warm = 0; warm < 2; ++warm) {
        dsv4::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_out, rows, kQHeads, kKvHeads, kHeadDim,
            position_offset, max_context, 0, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        dsv4::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_out, rows, kQHeads, kKvHeads, kHeadDim,
            position_offset, max_context, 0, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    return static_cast<double>(ms) / iters;
}

}  // namespace

int main() {
    if (!dsv4::cuda_runtime_available()) {
        std::printf("[SKIP] bench_qwen_gqa_prefill requires a CUDA device\n");
        return 0;
    }
    // The real run prefills 8192 tokens in two 4096-token chunks, so the second
    // chunk starts at offset 4096. Each chunk is dispatched per 512-row slice.
    struct Case {
        int rows;
        int offset;
    };
    // The real 8192 run prefills in two 4096-token chunks, so these are the two
    // shapes full_attention is actually dispatched with, 16 times each.
    const Case cases[] = {
        {4096, 0}, {4096, 4096},
    };
    for (const Case& item : cases) {
        const double ms = time_case(item.rows, item.offset, 8192, 10);
        if (ms < 0.0) return 0;
        // Every CTA reads the whole history for its KV head; report the K+V bytes
        // a single pass over the causal window would need as a traffic floor.
        const double window = item.offset + item.rows / 2.0;
        const double bytes = window * kKvHeads * kHeadDim * 2.0 * 2.0;
        std::printf("rows=%4d offset=%4d %8.4f ms  min-traffic=%6.1f GB/s\n",
                    item.rows, item.offset, ms,
                    bytes / (ms * 1.0e-3) / 1.0e9);
    }
    return 0;
}
