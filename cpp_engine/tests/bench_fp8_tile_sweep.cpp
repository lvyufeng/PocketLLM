#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../include/qwen_cuda_ops.hpp"
#include <chrono>

int main() {
    const int batch = 8, rows = 5120, cols = 4352;
    const int scale_cols = (cols + 127) / 128;
    uint16_t *x, *s, *y;
    uint8_t *w;
    cudaMalloc(&x, batch * cols * sizeof(uint16_t));
    cudaMalloc(&w, rows * cols);
    cudaMalloc(&s, rows * scale_cols * sizeof(uint16_t));
    cudaMalloc(&y, batch * rows * sizeof(uint16_t));
    
    for (int tile : {256, 512, 1024, 2048}) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", tile);
        setenv("QWEN_FP8_F16_SMALL_BATCH_TILE", buf, 1);
        
        cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) {
            dsv4::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                x, w, s, y, batch, rows, cols, cols, rows, cols, scale_cols, nullptr);
        }
        cudaDeviceSynchronize();
        auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        
        double bytes = (batch * cols + rows * cols + rows * scale_cols + batch * rows) * 2.0;
        double ms = elapsed / 100.0 * 1000.0;
        double bw = bytes / (elapsed / 100.0) / 1e9;
        std::printf("tile=%4d: %6.2f ms  %5.0f GB/s\n", tile, ms, bw);
    }
    
    cudaFree(x); cudaFree(w); cudaFree(s); cudaFree(y);
    return 0;
}
