#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace dsv4 {
namespace {

constexpr int kBlock = 128;

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    float value;
    if (exponent == 0) {
        value = ldexpf(static_cast<float>(mantissa) * (1.0f / 8.0f), -6);
    } else {
        value = ldexpf(1.0f + static_cast<float>(mantissa) * (1.0f / 8.0f), exponent - 7);
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x03ffu;
    uint32_t value;
    if (exponent == 0) {
        if (mantissa == 0) value = sign;
        else {
            uint32_t normalized = mantissa;
            int exp = -14;
            while ((normalized & 0x400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            value = sign | static_cast<uint32_t>(exp + 127) << 23 |
                    ((normalized & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return __uint_as_float(value);
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <bool kFp16Scale>
__device__ __forceinline__ float scale_bits_to_float(uint16_t bits) {
    return kFp16Scale ? fp16_bits_to_float(bits) : bf16_bits_to_float(bits);
}

template <bool kFp16Scale>
__device__ __forceinline__ float block_scale(const uint16_t* scale,
                                              int scale_stride,
                                              int row,
                                              int col) {
    return scale_bits_to_float<kFp16Scale>(
        scale[static_cast<size_t>(row / kBlock) * scale_stride + col / kBlock]);
}

__device__ __forceinline__ float reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

// Prefill path: one warp per output row, but each warp carries kTileBatch token
// accumulators so a decoded FP8 weight is reused across tokens instead of being
// re-read once per (row, token) pair. This turns the projection from
// bandwidth-bound-per-token into a single weight sweep per row tile.
template <bool kFp16Scale, int kRowsPerBlock, int kTileBatch>
__global__ void fp8_matmul_tiled_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const int batch_base = static_cast<int>(blockIdx.y) * kTileBatch;
    // The LUT fill is a block-wide barrier, so no thread may return before it.
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    if (row >= rows || batch_base >= batch) return;

    const int active = min(kTileBatch, batch - batch_base);
    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* scale_row = scale + static_cast<size_t>(row / kBlock) * scale_stride;
    float acc[kTileBatch];
    for (int b = 0; b < kTileBatch; ++b) acc[b] = 0.0f;

    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 codes = *reinterpret_cast<const uchar4*>(w_row + col);
        const float s = scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        const float w0 = lut[codes.x] * s;
        const float w1 = lut[codes.y] * s;
        const float w2 = lut[codes.z] * s;
        const float w3 = lut[codes.w] * s;
        for (int b = 0; b < active; ++b) {
            const float* x_row = x + static_cast<size_t>(batch_base + b) * x_stride;
            acc[b] += x_row[col + 0] * w0 + x_row[col + 1] * w1 +
                      x_row[col + 2] * w2 + x_row[col + 3] * w3;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float w = lut[w_row[col]] *
                        scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        for (int b = 0; b < active; ++b) {
            acc[b] += x[static_cast<size_t>(batch_base + b) * x_stride + col] * w;
        }
    }
    for (int b = 0; b < active; ++b) {
        float sum = acc[b];
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) y[static_cast<size_t>(batch_base + b) * y_stride + row] = sum;
    }
}

template <bool kFp16Scale>
__global__ void fp8_matmul_rows_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;

    const float* x_row = x + static_cast<size_t>(sample) * x_stride;
    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float w = fp8_e4m3_to_float(w_row[col]) * block_scale<kFp16Scale>(scale, scale_stride, row, col);
        sum += x_row[col] * w;
    }

    extern __shared__ float scratch[];
    const float total = reduce_sum(sum, scratch);
    if (threadIdx.x == 0) y[static_cast<size_t>(sample) * y_stride + row] = total;
}

template <bool kFp16Scale>
__global__ void fp8_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float w = fp8_e4m3_to_float(w_row[col]) * block_scale<kFp16Scale>(scale, scale_stride, row, col);
        sum += x[col] * w;
    }

    extern __shared__ float scratch[];
    const float total = reduce_sum(sum, scratch);
    if (threadIdx.x == 0) y[row] = total;
}

// Decode-latency path: one warp per output row, so there is no block-wide
// __syncthreads in the reduction and several rows retire per block. FP8 codes
// are pulled 4 bytes at a time; the 128-wide scale block means the scale lookup
// is loop-invariant across each aligned run of 128 columns.
template <bool kFp16Scale, int kRowsPerBlock>
__global__ void fp8_matvec_warp_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    // Decode E4M3 through a 256-entry shared-memory LUT (1 KB) instead of the
    // per-byte ldexpf/branch sequence. x itself is left in global memory: every
    // warp sweeps the same vector so L1/L2 already serves that reuse, and
    // staging x measured slower because its 20 KB footprint caps occupancy.
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    if (row >= rows) return;
    const float* x_shared = x;

    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* scale_row = scale + static_cast<size_t>(row / kBlock) * scale_stride;
    const int vec_cols = cols & ~3;
    float sum = 0.0f;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 codes = *reinterpret_cast<const uchar4*>(w_row + col);
        const float s = scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        const float4 xv = *reinterpret_cast<const float4*>(x_shared + col);
        sum += (xv.x * lut[codes.x] + xv.y * lut[codes.y] +
                xv.z * lut[codes.z] + xv.w * lut[codes.w]) * s;
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        sum += x_shared[col] * lut[w_row[col]] *
               scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) y[row] = sum;
}

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ weight,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float variance_sum = reduce_sum(sum, scratch);
    const float inv = rsqrtf(variance_sum / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y_row[col] = x_row[col] * inv * (1.0f + weight[col]);
    }
}

__global__ void gated_rmsnorm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ gate,
                                     float* __restrict__ y,
                                     int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    const float* gate_row = gate + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float variance_sum = reduce_sum(sum, scratch);
    const float inv = rsqrtf(variance_sum / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y_row[col] = weight[col] * x_row[col] * inv * silu(gate_row[col]);
    }
}

__global__ void l2_norm_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float norm_sq = reduce_sum(sum, scratch);
    const float inv = rsqrtf(norm_sq + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) y_row[col] = x_row[col] * inv;
}

bool valid_common(const void* x, const void* weight, const void* scale, void* y,
                  int rows, int cols, int scale_stride) {
    return x != nullptr && weight != nullptr && scale != nullptr && y != nullptr &&
           rows > 0 && cols > 0 && scale_stride >= (cols + kBlock - 1) / kBlock;
}

}  // namespace

bool qwen_fp8_e4m3_fp16scale_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_fp16, d_y, rows, cols, scale_stride) ||
        weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // uchar4 loads need a 4-byte aligned row base; every Qwen3.8 projection has
    // a stride that is a multiple of 128, so this is the common path.
    // Staging x needs cols*4 bytes of shared memory; 2080 Ti allows 48 KB per
    // block by default, which covers every Qwen3.8 projection (max 5120 cols).
    const size_t shmem = static_cast<size_t>(cols) * sizeof(float);
    if (weight_stride % 4 == 0 && shmem <= 48u * 1024u) {
        constexpr int kRowsPerBlock = 8;  // 8 warps = 256 threads
        const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
        fp8_matvec_warp_kernel<true, kRowsPerBlock><<<blocks, kRowsPerBlock * 32, shmem, s>>>(
            d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
        return cudaGetLastError() == cudaSuccess;
    }
    fp8_matvec_kernel<true><<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_fp16, d_y, rows, cols, scale_stride) ||
        batch <= 0 || x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (weight_stride % 4 == 0) {
        constexpr int kRowsPerBlock = 4;   // 4 warps = 128 threads
        constexpr int kTileBatch = 8;
        dim3 grid(static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock),
                  static_cast<unsigned>((batch + kTileBatch - 1) / kTileBatch), 1);
        fp8_matmul_tiled_kernel<true, kRowsPerBlock, kTileBatch>
            <<<grid, kRowsPerBlock * 32, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
                x_stride, y_stride, weight_stride, scale_stride);
        return cudaGetLastError() == cudaSuccess;
    }
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
    fp8_matmul_rows_kernel<true><<<grid, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_bf16_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_bf16, d_y, rows, cols, scale_stride) ||
        weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    fp8_matvec_kernel<false><<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_bf16, d_y, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_bf16_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_bf16, d_y, rows, cols, scale_stride) ||
        batch <= 0 || x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
    fp8_matmul_rows_kernel<false><<<grid, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_bf16, d_y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                           float* d_y, int rows, int cols, float eps,
                           void* stream) {
    if (!d_x || !d_weight || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(d_x, d_weight, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                                 const float* d_gate, float* d_y,
                                 int rows, int cols, float eps,
                                 void* stream) {
    if (!d_x || !d_weight || !d_gate || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    gated_rmsnorm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_gate, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_l2_norm_f32_cuda(const float* d_x, float* d_y, int rows, int cols,
                           float eps, void* stream) {
    if (!d_x || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    l2_norm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(d_x, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
