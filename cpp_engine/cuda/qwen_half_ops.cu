#include "qwen_cuda_ops.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace dsv4 {
namespace {

constexpr int kFp8WeightBlock = 128;
constexpr int kThreads = 256;

__device__ __forceinline__ float half_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float silu(float value) {
    return value * sigmoid(value);
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
    const uint32_t exponent = (code >> 3) & 0xfu;
    const uint32_t mantissa = code & 0x7u;
    uint32_t bits;
    if (exponent != 0) {
        bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    } else if (mantissa >= 4) {
        bits = sign | (120u << 23) | ((mantissa - 4u) << 21);
    } else if (mantissa >= 2) {
        bits = sign | (119u << 23) | ((mantissa - 2u) << 22);
    } else {
        bits = sign | (mantissa == 0 ? 0u : (118u << 23));
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float value) {
    const bool negative = signbit(value);
    float magnitude = fminf(fabsf(value), 448.0f);
    uint8_t code = 0;
    if (magnitude >= 0.0009765625f) {
        if (magnitude < 0.015625f) {
            int mantissa = __float2int_rn(magnitude * 512.0f);
            mantissa = max(1, mantissa);
            code = static_cast<uint8_t>(min(8, mantissa));
        } else {
            int exponent = static_cast<int>(floorf(log2f(magnitude)));
            exponent = max(-6, min(8, exponent));
            const float base = exp2f(static_cast<float>(exponent));
            int mantissa = __float2int_rn((magnitude / base - 1.0f) * 8.0f);
            if (mantissa == 8) {
                mantissa = 0;
                ++exponent;
            }
            int biased = exponent + 7;
            if (biased >= 15 && mantissa > 6) mantissa = 6;
            biased = min(15, biased);
            code = static_cast<uint8_t>((biased << 3) | mantissa);
        }
    }
    return static_cast<uint8_t>(code | (negative ? 0x80u : 0u));
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

// One warp owns one output row. This is the decode path; all products and the
// warp reduction remain FP32 while the activation vector and result are FP16.
template <int kWarps>
__global__ void fp8_matvec_f16_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale, uint16_t* __restrict__ y,
    int rows, int cols, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const uint8_t* w = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* s = scale + static_cast<size_t>(row / kFp8WeightBlock) * scale_stride;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        sum += half_to_float(x[col]) * lut[w[col]] * half_to_float(s[col / kFp8WeightBlock]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[row] = float_to_half(sum);
}

// Same decode arithmetic as fp8_matvec_f16_kernel, but each warp owns
// kRowsPerWarp output rows and reads every FP16 activation element once for
// all of them. Column order and per-row accumulation order are unchanged, so
// only the number of activation loads differs.
template <int kWarps, int kRowsPerWarp>
__global__ void fp8_matvec_f16_multirow_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale, uint16_t* __restrict__ y,
    int rows, int cols, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_base = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRowsPerWarp;
    if (row_base >= rows) return;
    const int active = min(kRowsPerWarp, rows - row_base);

    const uint8_t* weight_rows[kRowsPerWarp];
    const uint16_t* scale_rows[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = min(row_base + r, rows - 1);
        weight_rows[r] = weight + static_cast<size_t>(row) * weight_stride;
        scale_rows[r] = scale + static_cast<size_t>(row / kFp8WeightBlock) * scale_stride;
    }

    float sums[kRowsPerWarp] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uint2 packed = *reinterpret_cast<const uint2*>(x + col);
        const float x0 = half_to_float(static_cast<uint16_t>(packed.x));
        const float x1 = half_to_float(static_cast<uint16_t>(packed.x >> 16));
        const float x2 = half_to_float(static_cast<uint16_t>(packed.y));
        const float x3 = half_to_float(static_cast<uint16_t>(packed.y >> 16));
        const int block_col = col / kFp8WeightBlock;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            const uchar4 codes = *reinterpret_cast<const uchar4*>(weight_rows[r] + col);
            const float s = half_to_float(scale_rows[r][block_col]);
            sums[r] += (x0 * lut[codes.x] + x1 * lut[codes.y] +
                        x2 * lut[codes.z] + x3 * lut[codes.w]) * s;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float xv = half_to_float(x[col]);
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            sums[r] += xv * lut[weight_rows[r][col]] *
                       half_to_float(scale_rows[r][col / kFp8WeightBlock]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        if (r >= active) break;
        float sum = sums[r];
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) y[row_base + r] = float_to_half(sum);
    }
}

// Small speculative batches are weight-bandwidth bound. One warp owns one
// output channel and evaluates all token rows while each FP8 weight code is in
// registers, so a K+1 verify block reads every projection weight only once.
template <int kWarps, int kBatchCapacity>
__global__ void fp8_matmul_f16_small_batch_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale, uint16_t* __restrict__ y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const uint8_t* w = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* s = scale + static_cast<size_t>(row / kFp8WeightBlock) * scale_stride;
    float sums[kBatchCapacity] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 codes = *reinterpret_cast<const uchar4*>(w + col);
        const float weight_scale = half_to_float(s[col / kFp8WeightBlock]);
        const float w0 = lut[codes.x] * weight_scale;
        const float w1 = lut[codes.y] * weight_scale;
        const float w2 = lut[codes.z] * weight_scale;
        const float w3 = lut[codes.w] * weight_scale;
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const uint2 packed = *reinterpret_cast<const uint2*>(
                x + static_cast<size_t>(sample) * x_stride + col);
            sums[sample] +=
                half_to_float(static_cast<uint16_t>(packed.x)) * w0 +
                half_to_float(static_cast<uint16_t>(packed.x >> 16)) * w1 +
                half_to_float(static_cast<uint16_t>(packed.y)) * w2 +
                half_to_float(static_cast<uint16_t>(packed.y >> 16)) * w3;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float weight_value = lut[w[col]] *
            half_to_float(s[col / kFp8WeightBlock]);
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            sums[sample] += half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]) * weight_value;
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float sum = sums[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            y[static_cast<size_t>(sample) * y_stride + row] = float_to_half(sum);
        }
    }
}

// Shared-activation verify kernel. Pairing two 128-column lane strides reduces
// loop and shared-load instructions while preserving each lane's accumulation
// order, so the result stays bit-identical to plain decode.
template <int kWarps, int kBatchCapacity, int kTile>
__global__ void fp8_matmul_f16_small_batch_paired_shared_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale, uint16_t* __restrict__ y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    __shared__ uint16_t xs[kBatchCapacity * kTile];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    const bool active = row < rows;
    const uint8_t* w = weight + static_cast<size_t>(active ? row : 0) *
        weight_stride;
    const uint16_t* s = scale +
        static_cast<size_t>((active ? row : 0) / kFp8WeightBlock) * scale_stride;
    float sums[kBatchCapacity] = {};
    const int vec_cols = cols & ~3;
    const int threads = static_cast<int>(blockDim.x);
    for (int tile = 0; tile < vec_cols; tile += kTile) {
        const int tile_cols = min(kTile, vec_cols - tile);
        const int vec_count = tile_cols >> 2;
        __syncthreads();
        if constexpr (kWarps >= 16) {
            const int load_count = batch * vec_count;
            for (int index = static_cast<int>(threadIdx.x); index < load_count;
                 index += threads) {
                const int sample = index / vec_count;
                const int i = index - sample * vec_count;
                const uint2* source = reinterpret_cast<const uint2*>(
                    x + static_cast<size_t>(sample) * x_stride + tile);
                uint2* target = reinterpret_cast<uint2*>(xs + sample * kTile);
                target[i] = source[i];
            }
        } else {
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint2* source = reinterpret_cast<const uint2*>(
                    x + static_cast<size_t>(sample) * x_stride + tile);
                uint2* target = reinterpret_cast<uint2*>(xs + sample * kTile);
                for (int i = static_cast<int>(threadIdx.x); i < vec_count;
                     i += threads) {
                    target[i] = source[i];
                }
            }
        }
        __syncthreads();
        if (!active) continue;
        const int pair_limit = tile_cols - 128;
        int local = lane * 4;
        for (; local < pair_limit; local += 256) {
            const int col = tile + local;
            const uchar4 c0 = *reinterpret_cast<const uchar4*>(w + col);
            const uchar4 c1 = *reinterpret_cast<const uchar4*>(w + col + 128);
            const float scale0 = half_to_float(s[col / kFp8WeightBlock]);
            const float scale1 = half_to_float(
                s[(col + 128) / kFp8WeightBlock]);
            const float a0 = lut[c0.x] * scale0;
            const float a1 = lut[c0.y] * scale0;
            const float a2 = lut[c0.z] * scale0;
            const float a3 = lut[c0.w] * scale0;
            const float b0 = lut[c1.x] * scale1;
            const float b1 = lut[c1.y] * scale1;
            const float b2 = lut[c1.z] * scale1;
            const float b3 = lut[c1.w] * scale1;
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint16_t* base = xs + sample * kTile + local;
                const uint2 p0 = *reinterpret_cast<const uint2*>(base);
                const uint2 p1 = *reinterpret_cast<const uint2*>(base + 128);
                const float2 lo0 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p0.x));
                const float2 hi0 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p0.y));
                const float2 lo1 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p1.x));
                const float2 hi1 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p1.y));
                float acc = sums[sample];
                acc += lo0.x * a0 + lo0.y * a1 + hi0.x * a2 + hi0.y * a3;
                acc += lo1.x * b0 + lo1.y * b1 + hi1.x * b2 + hi1.y * b3;
                sums[sample] = acc;
            }
        }
        for (; local < tile_cols; local += 128) {
            const int col = tile + local;
            const uchar4 codes = *reinterpret_cast<const uchar4*>(w + col);
            const float weight_scale = half_to_float(
                s[col / kFp8WeightBlock]);
            const float w0 = lut[codes.x] * weight_scale;
            const float w1 = lut[codes.y] * weight_scale;
            const float w2 = lut[codes.z] * weight_scale;
            const float w3 = lut[codes.w] * weight_scale;
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint2 packed = *reinterpret_cast<const uint2*>(
                    xs + sample * kTile + local);
                const float2 lo = __half22float2(
                    *reinterpret_cast<const __half2*>(&packed.x));
                const float2 hi = __half22float2(
                    *reinterpret_cast<const __half2*>(&packed.y));
                sums[sample] += lo.x * w0 + lo.y * w1 + hi.x * w2 + hi.y * w3;
            }
        }
    }
    if (!active) return;
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float weight_value = lut[w[col]] *
            half_to_float(s[col / kFp8WeightBlock]);
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            sums[sample] += half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]) * weight_value;
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float sum = sums[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            y[static_cast<size_t>(sample) * y_stride + row] = float_to_half(sum);
        }
    }
}

// Shared-activation variant of fp8_swiglu_small_batch_f16_kernel. Gate and up
// weights are already read once per block, so the un-staged kernel spends most of
// its bandwidth re-reading the activation tile from every warp. Column tiles are
// multiples of the 128-element per-lane stride, keeping each lane's accumulation
// order and therefore the result bit-identical.
template <int kWarps, int kBatchCapacity, int kTile>
__global__ void fp8_swiglu_small_batch_shared_f16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ gate_scale,
    const uint8_t* __restrict__ up_weight,
    const uint16_t* __restrict__ up_scale,
    uint16_t* __restrict__ y, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    __shared__ uint16_t xs[kBatchCapacity * kTile];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    const bool active = row < rows;
    const int safe_row = active ? row : 0;
    const uint8_t* gate_row = gate_weight +
        static_cast<size_t>(safe_row) * weight_stride;
    const uint8_t* up_row = up_weight +
        static_cast<size_t>(safe_row) * weight_stride;
    const uint16_t* gate_scale_row = gate_scale +
        static_cast<size_t>(safe_row / kFp8WeightBlock) * scale_stride;
    const uint16_t* up_scale_row = up_scale +
        static_cast<size_t>(safe_row / kFp8WeightBlock) * scale_stride;
    float gate_sums[kBatchCapacity] = {};
    float up_sums[kBatchCapacity] = {};
    const int vec_cols = cols & ~3;
    const int threads = static_cast<int>(blockDim.x);
    for (int tile = 0; tile < vec_cols; tile += kTile) {
        const int tile_cols = min(kTile, vec_cols - tile);
        const int vec_count = tile_cols >> 2;
        __syncthreads();
        if constexpr (kWarps >= 16) {
            const int load_count = batch * vec_count;
            for (int index = static_cast<int>(threadIdx.x); index < load_count;
                 index += threads) {
                const int sample = index / vec_count;
                const int i = index - sample * vec_count;
                const uint2* source = reinterpret_cast<const uint2*>(
                    x + static_cast<size_t>(sample) * x_stride + tile);
                uint2* target = reinterpret_cast<uint2*>(xs + sample * kTile);
                target[i] = source[i];
            }
        } else {
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint2* source = reinterpret_cast<const uint2*>(
                    x + static_cast<size_t>(sample) * x_stride + tile);
                uint2* target = reinterpret_cast<uint2*>(xs + sample * kTile);
                for (int i = static_cast<int>(threadIdx.x); i < vec_count;
                     i += threads) {
                    target[i] = source[i];
                }
            }
        }
        __syncthreads();
        if (!active) continue;
        const int pair_limit = tile_cols - 128;
        int local = lane * 4;
        for (; local < pair_limit; local += 256) {
            const int col = tile + local;
            const uchar4 gate0 =
                *reinterpret_cast<const uchar4*>(gate_row + col);
            const uchar4 gate1 =
                *reinterpret_cast<const uchar4*>(gate_row + col + 128);
            const uchar4 up0 = *reinterpret_cast<const uchar4*>(up_row + col);
            const uchar4 up1 =
                *reinterpret_cast<const uchar4*>(up_row + col + 128);
            const float gate_s0 =
                half_to_float(gate_scale_row[col / kFp8WeightBlock]);
            const float gate_s1 = half_to_float(
                gate_scale_row[(col + 128) / kFp8WeightBlock]);
            const float up_s0 =
                half_to_float(up_scale_row[col / kFp8WeightBlock]);
            const float up_s1 =
                half_to_float(up_scale_row[(col + 128) / kFp8WeightBlock]);
            const float gw00 = lut[gate0.x] * gate_s0;
            const float gw01 = lut[gate0.y] * gate_s0;
            const float gw02 = lut[gate0.z] * gate_s0;
            const float gw03 = lut[gate0.w] * gate_s0;
            const float gw10 = lut[gate1.x] * gate_s1;
            const float gw11 = lut[gate1.y] * gate_s1;
            const float gw12 = lut[gate1.z] * gate_s1;
            const float gw13 = lut[gate1.w] * gate_s1;
            const float uw00 = lut[up0.x] * up_s0;
            const float uw01 = lut[up0.y] * up_s0;
            const float uw02 = lut[up0.z] * up_s0;
            const float uw03 = lut[up0.w] * up_s0;
            const float uw10 = lut[up1.x] * up_s1;
            const float uw11 = lut[up1.y] * up_s1;
            const float uw12 = lut[up1.z] * up_s1;
            const float uw13 = lut[up1.w] * up_s1;
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint16_t* base = xs + sample * kTile + local;
                const uint2 p0 = *reinterpret_cast<const uint2*>(base);
                const uint2 p1 = *reinterpret_cast<const uint2*>(base + 128);
                const float2 lo0 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p0.x));
                const float2 hi0 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p0.y));
                const float2 lo1 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p1.x));
                const float2 hi1 = __half22float2(
                    *reinterpret_cast<const __half2*>(&p1.y));
                float gate_acc = gate_sums[sample];
                float up_acc = up_sums[sample];
                gate_acc += lo0.x * gw00 + lo0.y * gw01 +
                    hi0.x * gw02 + hi0.y * gw03;
                up_acc += lo0.x * uw00 + lo0.y * uw01 +
                    hi0.x * uw02 + hi0.y * uw03;
                gate_acc += lo1.x * gw10 + lo1.y * gw11 +
                    hi1.x * gw12 + hi1.y * gw13;
                up_acc += lo1.x * uw10 + lo1.y * uw11 +
                    hi1.x * uw12 + hi1.y * uw13;
                gate_sums[sample] = gate_acc;
                up_sums[sample] = up_acc;
            }
        }
        for (; local < tile_cols; local += 128) {
            const int col = tile + local;
            const uchar4 gate_codes =
                *reinterpret_cast<const uchar4*>(gate_row + col);
            const uchar4 up_codes =
                *reinterpret_cast<const uchar4*>(up_row + col);
            const float gate_s =
                half_to_float(gate_scale_row[col / kFp8WeightBlock]);
            const float up_s =
                half_to_float(up_scale_row[col / kFp8WeightBlock]);
            const float gw0 = lut[gate_codes.x] * gate_s;
            const float gw1 = lut[gate_codes.y] * gate_s;
            const float gw2 = lut[gate_codes.z] * gate_s;
            const float gw3 = lut[gate_codes.w] * gate_s;
            const float uw0 = lut[up_codes.x] * up_s;
            const float uw1 = lut[up_codes.y] * up_s;
            const float uw2 = lut[up_codes.z] * up_s;
            const float uw3 = lut[up_codes.w] * up_s;
#pragma unroll
            for (int sample = 0; sample < kBatchCapacity; ++sample) {
                if (sample >= batch) break;
                const uint2 packed = *reinterpret_cast<const uint2*>(
                    xs + sample * kTile + local);
                const float2 lo = __half22float2(
                    *reinterpret_cast<const __half2*>(&packed.x));
                const float2 hi = __half22float2(
                    *reinterpret_cast<const __half2*>(&packed.y));
                gate_sums[sample] +=
                    lo.x * gw0 + lo.y * gw1 + hi.x * gw2 + hi.y * gw3;
                up_sums[sample] +=
                    lo.x * uw0 + lo.y * uw1 + hi.x * uw2 + hi.y * uw3;
            }
        }
    }
    if (!active) return;
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float gate_value = lut[gate_row[col]] *
            half_to_float(gate_scale_row[col / kFp8WeightBlock]);
        const float up_value = lut[up_row[col]] *
            half_to_float(up_scale_row[col / kFp8WeightBlock]);
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const float xv = half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]);
            gate_sums[sample] += xv * gate_value;
            up_sums[sample] += xv * up_value;
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float gate_sum = gate_sums[sample];
        float up_sum = up_sums[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, offset);
            up_sum += __shfl_xor_sync(0xffffffffu, up_sum, offset);
        }
        if (lane == 0) {
            y[static_cast<size_t>(sample) * y_stride + row] =
                float_to_half(silu(gate_sum) * up_sum);
        }
    }
}

template <int kWarps, int kBatchCapacity>
__global__ void fp8_swiglu_small_batch_f16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ gate_scale,
    const uint8_t* __restrict__ up_weight,
    const uint16_t* __restrict__ up_scale,
    uint16_t* __restrict__ y, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const uint8_t* gate_row = gate_weight +
        static_cast<size_t>(row) * weight_stride;
    const uint8_t* up_row = up_weight +
        static_cast<size_t>(row) * weight_stride;
    const uint16_t* gate_scale_row = gate_scale +
        static_cast<size_t>(row / kFp8WeightBlock) * scale_stride;
    const uint16_t* up_scale_row = up_scale +
        static_cast<size_t>(row / kFp8WeightBlock) * scale_stride;
    float gate_sums[kBatchCapacity] = {};
    float up_sums[kBatchCapacity] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 gate_codes = *reinterpret_cast<const uchar4*>(gate_row + col);
        const uchar4 up_codes = *reinterpret_cast<const uchar4*>(up_row + col);
        const float gate_s = half_to_float(gate_scale_row[col / kFp8WeightBlock]);
        const float up_s = half_to_float(up_scale_row[col / kFp8WeightBlock]);
        const float gw0 = lut[gate_codes.x] * gate_s;
        const float gw1 = lut[gate_codes.y] * gate_s;
        const float gw2 = lut[gate_codes.z] * gate_s;
        const float gw3 = lut[gate_codes.w] * gate_s;
        const float uw0 = lut[up_codes.x] * up_s;
        const float uw1 = lut[up_codes.y] * up_s;
        const float uw2 = lut[up_codes.z] * up_s;
        const float uw3 = lut[up_codes.w] * up_s;
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const uint2 packed = *reinterpret_cast<const uint2*>(
                x + static_cast<size_t>(sample) * x_stride + col);
            const float x0 = half_to_float(static_cast<uint16_t>(packed.x));
            const float x1 = half_to_float(static_cast<uint16_t>(packed.x >> 16));
            const float x2 = half_to_float(static_cast<uint16_t>(packed.y));
            const float x3 = half_to_float(static_cast<uint16_t>(packed.y >> 16));
            gate_sums[sample] += x0 * gw0 + x1 * gw1 + x2 * gw2 + x3 * gw3;
            up_sums[sample] += x0 * uw0 + x1 * uw1 + x2 * uw2 + x3 * uw3;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float gate_value = lut[gate_row[col]] *
            half_to_float(gate_scale_row[col / kFp8WeightBlock]);
        const float up_value = lut[up_row[col]] *
            half_to_float(up_scale_row[col / kFp8WeightBlock]);
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const float xv = half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]);
            gate_sums[sample] += xv * gate_value;
            up_sums[sample] += xv * up_value;
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float gate_sum = gate_sums[sample];
        float up_sum = up_sums[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, offset);
            up_sum += __shfl_xor_sync(0xffffffffu, up_sum, offset);
        }
        if (lane == 0) {
            y[static_cast<size_t>(sample) * y_stride + row] =
                float_to_half(silu(gate_sum) * up_sum);
        }
    }
}

template <int kWarps, int kRowsPerWarp>
__global__ void fp8_swiglu_matvec_f16_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ gate_scale,
    const uint8_t* __restrict__ up_weight,
    const uint16_t* __restrict__ up_scale,
    uint16_t* __restrict__ y, int rows, int cols, int weight_stride,
    int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_base = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRowsPerWarp;
    if (row_base >= rows) return;
    const int active = min(kRowsPerWarp, rows - row_base);
    float gate_sum[kRowsPerWarp] = {};
    float up_sum[kRowsPerWarp] = {};
    for (int col = lane; col < cols; col += 32) {
        const float xv = half_to_float(x[col]);
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            const int row = row_base + r;
            const size_t wi = static_cast<size_t>(row) * weight_stride + col;
            const size_t si = static_cast<size_t>(row / kFp8WeightBlock) * scale_stride +
                              col / kFp8WeightBlock;
            gate_sum[r] += xv * lut[gate_weight[wi]] * half_to_float(gate_scale[si]);
            up_sum[r] += xv * lut[up_weight[wi]] * half_to_float(up_scale[si]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        if (r >= active) break;
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum[r] += __shfl_xor_sync(0xffffffffu, gate_sum[r], offset);
            up_sum[r] += __shfl_xor_sync(0xffffffffu, up_sum[r], offset);
        }
        if (lane == 0) y[row_base + r] = float_to_half(silu(gate_sum[r]) * up_sum[r]);
    }
}

// A 64-token by 64-output tile reuses each online-decoded weight across all
// tokens in the tile. Shared storage is FP32 so arithmetic association and
// accumulation are independent of activation storage precision.
template <int kK>
__global__ void fp8_matmul_f16_tiled_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale, uint16_t* __restrict__ y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, int scale_stride) {
    constexpr int kM = 64;
    constexpr int kN = 64;
    __shared__ float xs[kM][kK + 1];
    __shared__ float ws[kN][kK + 1];
    __shared__ float lut[256];
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += blockDim.x) lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    __syncthreads();
    const int batch_base = static_cast<int>(blockIdx.y) * kM;
    const int row_base = static_cast<int>(blockIdx.x) * kN;
    const int tm = tid >> 4;
    const int tn = tid & 15;
    float acc[4][4] = {};
    for (int col_base = 0; col_base < cols; col_base += kK) {
        for (int index = tid; index < kM * kK; index += blockDim.x) {
            const int m = index / kK;
            const int k = index % kK;
            const int gb = batch_base + m;
            const int gc = col_base + k;
            xs[m][k] = gb < batch && gc < cols
                ? half_to_float(x[static_cast<size_t>(gb) * x_stride + gc]) : 0.0f;
        }
        for (int index = tid; index < kN * kK; index += blockDim.x) {
            const int n = index / kK;
            const int k = index % kK;
            const int gr = row_base + n;
            const int gc = col_base + k;
            float value = 0.0f;
            if (gr < rows && gc < cols) {
                value = lut[weight[static_cast<size_t>(gr) * weight_stride + gc]] *
                        half_to_float(scale[static_cast<size_t>(gr / kFp8WeightBlock) *
                                                  scale_stride + gc / kFp8WeightBlock]);
            }
            ws[n][k] = value;
        }
        __syncthreads();
        float tile[4][4] = {};
#pragma unroll
        for (int k = 0; k < kK; ++k) {
            float a[4];
            float b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[tm + i * 16][k];
                b[i] = ws[tn + i * 16][k];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) tile[i][j] += a[i] * b[j];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += tile[i][j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gb = batch_base + tm + i * 16;
        if (gb >= batch) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gr = row_base + tn + j * 16;
            if (gr < rows) y[static_cast<size_t>(gb) * y_stride + gr] = float_to_half(acc[i][j]);
        }
    }
}

// Lower-register FP16-activation variant of the wide N64 prefill tile. The
// layout matches the validated FP32 kernel: 128 input tokens by 64 output
// rows, with FP8 weights decoded once per K tile and reused across tokens.
__global__ void fp8_matmul_f16_wide_n64_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    uint16_t* __restrict__ y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, int scale_stride) {
    constexpr int kM = 128;
    constexpr int kN = 64;
    constexpr int kK = 16;
    constexpr int kThreads = 256;
    __shared__ float as[kK][kM];
    __shared__ float bs[kK][kN];
    __shared__ float lut[256];

    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += kThreads) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int batch_base = static_cast<int>(blockIdx.y) * kM;
    const int row_base = static_cast<int>(blockIdx.x) * kN;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = ty * 4;
    const int m1 = 64 + ty * 4;
    const int n0 = tx * 4;
    const int load_lane = (tid & 3) * 4;
    const int load_line = tid >> 2;
    float acc[8][4] = {};

    for (int col_base = 0; col_base < cols; col_base += kK) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int local_batch = load_line + it * 64;
            const int global_batch = batch_base + local_batch;
            const int global_col = col_base + load_lane;
            float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_batch < batch) {
                const uint16_t* src = x + static_cast<size_t>(global_batch) * x_stride;
                if (global_col + 3 < cols) {
                    const uint2 packed = *reinterpret_cast<const uint2*>(src + global_col);
                    values[0] = half_to_float(static_cast<uint16_t>(packed.x));
                    values[1] = half_to_float(static_cast<uint16_t>(packed.x >> 16));
                    values[2] = half_to_float(static_cast<uint16_t>(packed.y));
                    values[3] = half_to_float(static_cast<uint16_t>(packed.y >> 16));
                } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (global_col + t < cols) values[t] = half_to_float(src[global_col + t]);
                    }
                }
            }
            as[load_lane + 0][local_batch] = values[0];
            as[load_lane + 1][local_batch] = values[1];
            as[load_lane + 2][local_batch] = values[2];
            as[load_lane + 3][local_batch] = values[3];
        }
#pragma unroll
        for (int it = 0; it < 1; ++it) {
            const int local_row = load_line;
            const int global_row = row_base + local_row;
            const int global_col = col_base + load_lane;
            float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_row < rows && global_col < cols) {
                const uint8_t* src = weight + static_cast<size_t>(global_row) * weight_stride;
                const float s = half_to_float(
                    scale[static_cast<size_t>(global_row / kFp8WeightBlock) * scale_stride +
                          global_col / kFp8WeightBlock]);
                if (global_col + 3 < cols) {
                    const uchar4 codes = *reinterpret_cast<const uchar4*>(src + global_col);
                    values[0] = lut[codes.x] * s;
                    values[1] = lut[codes.y] * s;
                    values[2] = lut[codes.z] * s;
                    values[3] = lut[codes.w] * s;
                } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (global_col + t < cols) values[t] = lut[src[global_col + t]] * s;
                    }
                }
            }
            bs[load_lane + 0][local_row] = values[0];
            bs[load_lane + 1][local_row] = values[1];
            bs[load_lane + 2][local_row] = values[2];
            bs[load_lane + 3][local_row] = values[3];
        }
        __syncthreads();

        float tile[8][4] = {};
#pragma unroll
        for (int k = 0; k < kK; ++k) {
            float a[8];
            float b[4];
            *reinterpret_cast<float4*>(&a[0]) = *reinterpret_cast<const float4*>(&as[k][m0]);
            *reinterpret_cast<float4*>(&a[4]) = *reinterpret_cast<const float4*>(&as[k][m1]);
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&bs[k][n0]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) tile[i][j] += a[i] * b[j];
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += tile[i][j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int local_batch = i < 4 ? m0 + i : m1 + (i - 4);
        const int global_batch = batch_base + local_batch;
        if (global_batch >= batch) continue;
        uint16_t* dst = y + static_cast<size_t>(global_batch) * y_stride;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int global_row = row_base + n0 + j;
            if (global_row < rows) dst[global_row] = float_to_half(acc[i][j]);
        }
    }
}

// Opt-in prefill candidate: decode raw FP8 weights into a reusable FP16
// workspace, then use cuBLAS tensor-op GEMM. Quantized checkpoint storage is
// unchanged; decode continues to use the online-unpack path.
// Vectorized raw-FP8 expansion used by the opt-in cuBLAS prefill path. Four
// adjacent codes share one block scale, and the lookup table avoids repeating
// the E4M3 bit decode for every conversion thread.
__global__ void fp8_weight_fp16scale_to_half_kernel(
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    uint16_t* __restrict__ output,
    int rows, int cols, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int vec_cols = cols / 4;
    const size_t total = static_cast<size_t>(rows) * vec_cols;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / vec_cols);
        const int vec = static_cast<int>(index - static_cast<size_t>(row) * vec_cols);
        const int col = vec * 4;
        const uint8_t* src = weight + static_cast<size_t>(row) * weight_stride + col;
        const uchar4 codes = *reinterpret_cast<const uchar4*>(src);
        const float block_scale = half_to_float(
            scale[static_cast<size_t>(row / kFp8WeightBlock) * scale_stride +
                  col / kFp8WeightBlock]);
        const uint16_t h0 = float_to_half(lut[codes.x] * block_scale);
        const uint16_t h1 = float_to_half(lut[codes.y] * block_scale);
        const uint16_t h2 = float_to_half(lut[codes.z] * block_scale);
        const uint16_t h3 = float_to_half(lut[codes.w] * block_scale);
        uint16_t* dst = output + static_cast<size_t>(row) * cols + col;
        // For a non-four-aligned logical row width, odd rows may start at a
        // two-byte offset. Avoid a misaligned uint2 store while retaining the
        // vectorized path for aligned rows.
        if ((static_cast<size_t>(row) * cols + col) & 1u) {
            dst[0] = h0;
            dst[1] = h1;
            dst[2] = h2;
            dst[3] = h3;
        } else {
            const uint32_t p0 = static_cast<uint32_t>(h0) |
                (static_cast<uint32_t>(h1) << 16);
            const uint32_t p1 = static_cast<uint32_t>(h2) |
                (static_cast<uint32_t>(h3) << 16);
            *reinterpret_cast<uint2*>(dst) = make_uint2(p0, p1);
        }
    }
}

// The vectorized expansion intentionally stops at the last complete uchar4.
// Keep a scalar tail so the cuBLAS candidate remains correct for generic
// projection shapes whose input width is not a multiple of four.
__global__ void fp8_weight_fp16scale_to_half_tail_kernel(
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    uint16_t* __restrict__ output,
    int rows, int cols, int weight_stride, int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    const int vec_cols = cols / 4;
    const int tail = cols - vec_cols * 4;
    const size_t total = static_cast<size_t>(rows) * tail;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / tail);
        const int tail_col = static_cast<int>(index - static_cast<size_t>(row) * tail);
        const int col = vec_cols * 4 + tail_col;
        const float block_scale = half_to_float(
            scale[static_cast<size_t>(row / kFp8WeightBlock) * scale_stride +
                  col / kFp8WeightBlock]);
        output[static_cast<size_t>(row) * cols + col] = float_to_half(
            lut[weight[static_cast<size_t>(row) * weight_stride + col]] * block_scale);
    }
}

struct Fp8F16CublasWorkspace {
    int device = -1;
    uint16_t* weight = nullptr;
    size_t weight_capacity = 0;
    cublasHandle_t handle = nullptr;
};

Fp8F16CublasWorkspace& fp8_f16_cublas_workspace() {
    static thread_local Fp8F16CublasWorkspace workspace;
    return workspace;
}

bool ensure_fp8_f16_cublas_workspace(
    Fp8F16CublasWorkspace& workspace, size_t elements,
    bool require_weight = true) {
    int current_device = 0;
    if (cudaGetDevice(&current_device) != cudaSuccess) return false;
    if (workspace.device != -1 && workspace.device != current_device) {
        cudaFree(workspace.weight);
        if (workspace.handle != nullptr) cublasDestroy(workspace.handle);
        workspace = {};
    }
    workspace.device = current_device;
    if (workspace.handle == nullptr) {
        if (cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS) return false;
        (void)cublasSetMathMode(workspace.handle, CUBLAS_TENSOR_OP_MATH);
    }
    if (require_weight && workspace.weight_capacity < elements) {
        cudaFree(workspace.weight);
        workspace.weight = nullptr;
        workspace.weight_capacity = 0;
        if (cudaMalloc(&workspace.weight, elements * sizeof(uint16_t)) !=
            cudaSuccess) return false;
        workspace.weight_capacity = elements;
    }
    return true;
}

bool dequantize_fp8_weight_f16(
    const uint8_t* weight, const uint16_t* scale, uint16_t* output,
    int rows, int cols, int weight_stride, int scale_stride,
    cudaStream_t stream) {
    const size_t weight_elements = static_cast<size_t>(rows) * cols;
    constexpr int kThreads = 256;
    const size_t block_count = (weight_elements + kThreads - 1) / kThreads;
    const int blocks = static_cast<int>(block_count > 65535 ? 65535 : block_count);
    fp8_weight_fp16scale_to_half_kernel<<<blocks, kThreads, 0, stream>>>(
        weight, scale, output, rows, cols, weight_stride, scale_stride);
    if (cudaGetLastError() != cudaSuccess) return false;
    const int tail = cols & 3;
    if (tail == 0) return true;
    const size_t tail_elements = static_cast<size_t>(rows) * tail;
    const size_t tail_block_count = (tail_elements + kThreads - 1) / kThreads;
    const int tail_blocks = static_cast<int>(std::min<size_t>(
        tail_block_count, 65535));
    fp8_weight_fp16scale_to_half_tail_kernel<<<tail_blocks, kThreads, 0, stream>>>(
        weight, scale, output, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool fp16_matmul_f16_cublas(
    const uint16_t* x, const uint16_t* weight, uint16_t* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, cudaStream_t stream) {
    Fp8F16CublasWorkspace& workspace = fp8_f16_cublas_workspace();
    if (!ensure_fp8_f16_cublas_workspace(workspace, 0, false)) return false;
    if (cublasSetStream(workspace.handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasGemmAlgo_t algo =
        (rows % 8 == 0 && batch % 8 == 0 && cols % 8 == 0)
            ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    return cublasGemmEx(
        workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, batch, cols, &alpha,
        weight, CUDA_R_16F, weight_stride,
        x, CUDA_R_16F, x_stride,
        &beta, y, CUDA_R_16F, y_stride,
        CUBLAS_COMPUTE_32F, algo) == CUBLAS_STATUS_SUCCESS;
}

bool fp8_matmul_f16_cublas(
    const uint16_t* x, const uint8_t* weight, const uint16_t* scale,
    uint16_t* y, int batch, int rows, int cols, int x_stride,
    int y_stride, int weight_stride, int scale_stride, cudaStream_t stream) {
    Fp8F16CublasWorkspace& workspace = fp8_f16_cublas_workspace();
    const size_t weight_elements = static_cast<size_t>(rows) * cols;
    if (!ensure_fp8_f16_cublas_workspace(workspace, weight_elements)) return false;
    if (!dequantize_fp8_weight_f16(weight, scale, workspace.weight, rows, cols,
                                   weight_stride, scale_stride, stream)) {
        return false;
    }
    return fp16_matmul_f16_cublas(
        x, workspace.weight, y, batch, rows, cols, x_stride, y_stride, cols,
        stream);
}

template <bool kFloatOutput, int kWarps, int kBatchCapacity>
__global__ void fp16_matmul_f16_small_batch_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ weight,
    void* __restrict__ output, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const uint16_t* wr = weight + static_cast<size_t>(row) * weight_stride;
    float sums[kBatchCapacity] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uint2 packed_w = *reinterpret_cast<const uint2*>(wr + col);
        const float w0 = half_to_float(static_cast<uint16_t>(packed_w.x));
        const float w1 = half_to_float(static_cast<uint16_t>(packed_w.x >> 16));
        const float w2 = half_to_float(static_cast<uint16_t>(packed_w.y));
        const float w3 = half_to_float(static_cast<uint16_t>(packed_w.y >> 16));
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const uint2 packed_x = *reinterpret_cast<const uint2*>(
                x + static_cast<size_t>(sample) * x_stride + col);
            sums[sample] +=
                half_to_float(static_cast<uint16_t>(packed_x.x)) * w0 +
                half_to_float(static_cast<uint16_t>(packed_x.x >> 16)) * w1 +
                half_to_float(static_cast<uint16_t>(packed_x.y)) * w2 +
                half_to_float(static_cast<uint16_t>(packed_x.y >> 16)) * w3;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float weight_value = half_to_float(wr[col]);
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            sums[sample] += half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]) * weight_value;
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float sum = sums[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            const size_t index = static_cast<size_t>(sample) * y_stride + row;
            if constexpr (kFloatOutput) {
                static_cast<float*>(output)[index] = sum;
            } else {
                static_cast<uint16_t*>(output)[index] = float_to_half(sum);
            }
        }
    }
}

template <int kBatchCapacity>
__global__ void fp16_swiglu_matmul_f16_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ up_weight,
    uint16_t* __restrict__ output,
    int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * 8 + warp;
    if (row >= rows || warp >= 8) return;
    const uint16_t* gate_row = gate_weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* up_row = up_weight + static_cast<size_t>(row) * weight_stride;
    float gate[kBatchCapacity] = {};
    float up[kBatchCapacity] = {};
    const int vectorized_cols = cols & ~3;
    for (int col = lane * 4; col < vectorized_cols; col += 128) {
        const uint2 gate_values = *reinterpret_cast<const uint2*>(gate_row + col);
        const uint2 up_values = *reinterpret_cast<const uint2*>(up_row + col);
        const float gate0 = half_to_float(static_cast<uint16_t>(gate_values.x));
        const float gate1 = half_to_float(static_cast<uint16_t>(gate_values.x >> 16));
        const float gate2 = half_to_float(static_cast<uint16_t>(gate_values.y));
        const float gate3 = half_to_float(static_cast<uint16_t>(gate_values.y >> 16));
        const float up0 = half_to_float(static_cast<uint16_t>(up_values.x));
        const float up1 = half_to_float(static_cast<uint16_t>(up_values.x >> 16));
        const float up2 = half_to_float(static_cast<uint16_t>(up_values.y));
        const float up3 = half_to_float(static_cast<uint16_t>(up_values.y >> 16));
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const uint2 x_values = *reinterpret_cast<const uint2*>(
                x + static_cast<size_t>(sample) * x_stride + col);
            const float x0 = half_to_float(static_cast<uint16_t>(x_values.x));
            const float x1 = half_to_float(static_cast<uint16_t>(x_values.x >> 16));
            const float x2 = half_to_float(static_cast<uint16_t>(x_values.y));
            const float x3 = half_to_float(static_cast<uint16_t>(x_values.y >> 16));
            gate[sample] += x0 * gate0 + x1 * gate1 + x2 * gate2 + x3 * gate3;
            up[sample] += x0 * up0 + x1 * up1 + x2 * up2 + x3 * up3;
        }
    }
    for (int col = vectorized_cols + lane; col < cols; col += 32) {
#pragma unroll
        for (int sample = 0; sample < kBatchCapacity; ++sample) {
            if (sample >= batch) break;
            const float input = half_to_float(
                x[static_cast<size_t>(sample) * x_stride + col]);
            gate[sample] += input * half_to_float(gate_row[col]);
            up[sample] += input * half_to_float(up_row[col]);
        }
    }
#pragma unroll
    for (int sample = 0; sample < kBatchCapacity; ++sample) {
        if (sample >= batch) break;
        float gate_sum = gate[sample];
        float up_sum = up[sample];
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate_sum += __shfl_xor_sync(0xffffffffu, gate_sum, offset);
            up_sum += __shfl_xor_sync(0xffffffffu, up_sum, offset);
        }
        if (lane == 0) {
            output[static_cast<size_t>(sample) * y_stride + row] =
                float_to_half(silu(gate_sum) * up_sum);
        }
    }
}

template <bool kFloatOutput>
__global__ void fp16_matmul_f16_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ weight,
    void* __restrict__ output, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;
    const uint16_t* xr = x + static_cast<size_t>(sample) * x_stride;
    const uint16_t* wr = weight + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += half_to_float(xr[col]) * half_to_float(wr[col]);
    }
    extern __shared__ float scratch[];
    const float total = block_reduce_sum(sum, scratch);
    if (threadIdx.x == 0) {
        const size_t index = static_cast<size_t>(sample) * y_stride + row;
        if constexpr (kFloatOutput) static_cast<float*>(output)[index] = total;
        else static_cast<uint16_t*>(output)[index] = float_to_half(total);
    }
}

__global__ void embedding_f16_kernel(const uint16_t* table, const int* tokens,
                                     uint16_t* output, int count, int cols,
                                     int row_start, int row_count) {
    const int token_index = static_cast<int>(blockIdx.x);
    if (token_index >= count) return;
    const int token = tokens[token_index];
    uint16_t* dst = output + static_cast<size_t>(token_index) * cols;
    if (token < row_start || token >= row_start + row_count) {
        for (int col = threadIdx.x; col < cols; col += blockDim.x) dst[col] = 0;
        return;
    }
    const uint16_t* src = table + static_cast<size_t>(token - row_start) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) dst[col] = src[col];
}

__global__ void concat_rows_f16_kernel(const uint16_t* left, const uint16_t* right,
                                       uint16_t* output, int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const uint16_t* left_row = left + static_cast<size_t>(row) * cols;
    const uint16_t* right_row = right + static_cast<size_t>(row) * cols;
    uint16_t* output_row = output + static_cast<size_t>(row) * 2 * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        output_row[col] = left_row[col];
        output_row[cols + col] = right_row[col];
    }
}

__global__ void rmsnorm_f16_kernel(const uint16_t* x, const uint16_t* gamma,
                                   uint16_t* y, int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const uint16_t* src = x + static_cast<size_t>(row) * cols;
    uint16_t* dst = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = half_to_float(src[col]);
        sum += value * value;
    }
    extern __shared__ float scratch[];
    const float inv = rsqrtf(block_reduce_sum(sum, scratch) / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        dst[col] = float_to_half(half_to_float(src[col]) * inv * (1.0f + half_to_float(gamma[col])));
    }
}

__global__ void residual_add_rmsnorm_f16_kernel(
    const uint16_t* hidden, const uint16_t* delta, const uint16_t* gamma,
    uint16_t* residual, uint16_t* normalized, int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const size_t row_offset = static_cast<size_t>(row) * cols;
    const uint16_t* hidden_row = hidden + row_offset;
    const uint16_t* delta_row = delta + row_offset;
    uint16_t* residual_row = residual + row_offset;
    uint16_t* normalized_row = normalized + row_offset;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint16_t value = float_to_half(
            half_to_float(hidden_row[col]) + half_to_float(delta_row[col]));
        residual_row[col] = value;
        const float rounded = half_to_float(value);
        sum += rounded * rounded;
    }
    extern __shared__ float scratch[];
    const float inv = rsqrtf(
        block_reduce_sum(sum, scratch) / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        normalized_row[col] = float_to_half(
            half_to_float(residual_row[col]) * inv *
            (1.0f + half_to_float(gamma[col])));
    }
}

__global__ void gated_rmsnorm_f16_kernel(
    const uint16_t* x, const uint16_t* gamma, const uint16_t* gate,
    uint16_t* y, int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const uint16_t* src = x + static_cast<size_t>(row) * cols;
    const uint16_t* gr = gate + static_cast<size_t>(row) * cols;
    uint16_t* dst = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = half_to_float(src[col]);
        sum += value * value;
    }
    extern __shared__ float scratch[];
    const float inv = rsqrtf(block_reduce_sum(sum, scratch) / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = half_to_float(src[col]) * inv * half_to_float(gamma[col]);
        dst[col] = float_to_half(value * silu(half_to_float(gr[col])));
    }
}

__global__ void split_packed_qkv_f16_kernel(
    const uint16_t* packed, uint16_t* q, uint16_t* k, uint16_t* v,
    int rows, int key_dim, int value_dim) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int packed_dim = 2 * key_dim + value_dim;
    const int total = rows * packed_dim;
    if (index >= total) return;
    const int row = index / packed_dim;
    const int col = index % packed_dim;
    if (col < key_dim) q[static_cast<size_t>(row) * key_dim + col] = packed[index];
    else if (col < 2 * key_dim) k[static_cast<size_t>(row) * key_dim + col - key_dim] = packed[index];
    else v[static_cast<size_t>(row) * value_dim + col - 2 * key_dim] = packed[index];
}

__global__ void conv_silu_f16_kernel(
    const uint16_t* x, const uint16_t* weight, uint16_t* tail,
    uint16_t* y, int seq_len, int channels, int kernel, bool update_tail) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const uint16_t* w = weight + static_cast<size_t>(channel) * kernel;
    const int tail_len = kernel - 1;
    for (int t = 0; t < seq_len; ++t) {
        float value = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            const int source_t = t - tail_len + j;
            float input = 0.0f;
            if (source_t >= 0) input = half_to_float(x[static_cast<size_t>(source_t) * channels + channel]);
            else if (tail != nullptr) input = half_to_float(tail[static_cast<size_t>(source_t + tail_len) * channels + channel]);
            value += input * half_to_float(w[j]);
        }
        y[static_cast<size_t>(t) * channels + channel] = float_to_half(silu(value));

    }
    if (update_tail && tail != nullptr && tail_len > 0) {
        constexpr int kMaxTail = 8;
        uint16_t next[kMaxTail];
        for (int j = 0; j < tail_len; ++j) {
            const int source_t = seq_len - tail_len + j;
            next[j] = source_t >= 0
                ? x[static_cast<size_t>(source_t) * channels + channel]
                : tail[static_cast<size_t>(source_t + tail_len) * channels + channel];
        }
        for (int j = 0; j < tail_len; ++j) tail[static_cast<size_t>(j) * channels + channel] = next[j];
    }
}

__global__ void linear_gates_f16_kernel(
    const uint16_t* a, const uint16_t* b, const uint16_t* a_log,
    const uint16_t* dt_bias, uint16_t* g, uint16_t* beta,
    int rows, int heads) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= heads) return;
    const float al = half_to_float(a_log[head]);
    const float dt = half_to_float(dt_bias[head]);
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * heads + head;
        const float av = half_to_float(a[index]);
        const float bv = half_to_float(b[index]);
        g[index] = float_to_half(-expf(al) * log1pf(expf(av + dt)));
        beta[index] = float_to_half(sigmoid(bv));
    }
}

__global__ void gated_delta_step_f16_kernel(
    float* state, const uint16_t* q, const uint16_t* k,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int heads, int key_heads, int key_dim,
    int value_dim, float q_scale) {
    const int head = static_cast<int>(blockIdx.x);
    const int value = static_cast<int>(threadIdx.x);
    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    extern __shared__ float smem[];
    float* k_shared = smem;
    float* q_shared = smem + key_dim;
    float* reduce = smem + 2 * key_dim;
    const int tid = static_cast<int>(threadIdx.x);
    float q_partial = 0.0f;
    float k_partial = 0.0f;
    for (int i = tid; i < key_dim; i += blockDim.x) {
        const float qv = half_to_float(q[static_cast<size_t>(key_head) * key_dim + i]);
        const float kv = half_to_float(k[static_cast<size_t>(key_head) * key_dim + i]);
        q_partial += qv * qv;
        k_partial += kv * kv;
    }
    reduce[tid] = q_partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float q_inv = rsqrtf(reduce[0] + 1.0e-6f);
    __syncthreads();
    reduce[tid] = k_partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float k_inv = rsqrtf(reduce[0] + 1.0e-6f);
    for (int i = tid; i < key_dim; i += blockDim.x) {
        q_shared[i] = half_to_float(q[static_cast<size_t>(key_head) * key_dim + i]) * q_inv;
        k_shared[i] = half_to_float(k[static_cast<size_t>(key_head) * key_dim + i]) * k_inv;
    }
    __syncthreads();
    if (head >= heads || value >= value_dim) return;
    float* sh = state + static_cast<size_t>(head) * key_dim * value_dim;
    const float decay = expf(half_to_float(g[head]));
    const float b = half_to_float(beta[head]);
    float kv_mem = 0.0f;
    for (int i = 0; i < key_dim; ++i) {
        const size_t index = static_cast<size_t>(i) * value_dim + value;
        const float cell = sh[index] * decay;
        sh[index] = cell;
        kv_mem += cell * k_shared[i];
    }
    const float delta = (half_to_float(v[static_cast<size_t>(head) * value_dim + value]) - kv_mem) * b;
    float result = 0.0f;
    for (int i = 0; i < key_dim; ++i) {
        const size_t index = static_cast<size_t>(i) * value_dim + value;
        const float cell = sh[index] + k_shared[i] * delta;
        sh[index] = cell;
        result += cell * q_shared[i];
    }
    output[static_cast<size_t>(head) * value_dim + value] = float_to_half(result * q_scale);
}

__global__ void normalize_gated_delta_qk_f16_kernel(
    const uint16_t* q, const uint16_t* k, float* q_normalized,
    float* k_normalized, int rows, int key_heads, int key_dim) {
    const int key_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const size_t offset =
        (static_cast<size_t>(token) * key_heads + key_head) * key_dim;
    float q_partial = 0.0f;
    float k_partial = 0.0f;
    for (int i = tid; i < key_dim; i += blockDim.x) {
        const float qv = half_to_float(q[offset + i]);
        const float kv = half_to_float(k[offset + i]);
        q_partial += qv * qv;
        k_partial += kv * kv;
    }
    __shared__ float q_reduce[128];
    __shared__ float k_reduce[128];
    q_reduce[tid] = q_partial;
    k_reduce[tid] = k_partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_reduce[tid] += q_reduce[tid + stride];
            k_reduce[tid] += k_reduce[tid + stride];
        }
        __syncthreads();
    }
    const float q_inv = rsqrtf(q_reduce[0] + 1.0e-6f);
    const float k_inv = rsqrtf(k_reduce[0] + 1.0e-6f);
    for (int i = tid; i < key_dim; i += blockDim.x) {
        q_normalized[offset + i] = half_to_float(q[offset + i]) * q_inv;
        k_normalized[offset + i] = half_to_float(k[offset + i]) * k_inv;
    }
}

template <int kKeyDim>
__global__ void gated_delta_sequence_normalized_f16_kernel(
    float* state, const float* q_normalized, const float* k_normalized,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads,
    int value_dim, float q_scale) {
    const int head = static_cast<int>(blockIdx.x);
    const int value = static_cast<int>(threadIdx.x);
    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    const int key_stride = key_heads * kKeyDim;
    float* sh = state + static_cast<size_t>(head) * kKeyDim * value_dim;
    float st[kKeyDim];
#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) {
        st[i] = sh[static_cast<size_t>(i) * value_dim + value];
    }
    for (int token = 0; token < rows; ++token) {
        const float* qr = q_normalized +
            static_cast<size_t>(token) * key_stride +
            static_cast<size_t>(key_head) * kKeyDim;
        const float* kr = k_normalized +
            static_cast<size_t>(token) * key_stride +
            static_cast<size_t>(key_head) * kKeyDim;
        const size_t gate_index = static_cast<size_t>(token) * heads + head;
        const float decay = expf(half_to_float(g[gate_index]));
        const float b = half_to_float(beta[gate_index]);
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] *= decay;
            kv_mem += st[i] * kr[i];
        }
        const size_t value_index = static_cast<size_t>(token) * heads * value_dim +
                                   static_cast<size_t>(head) * value_dim + value;
        const float delta = (half_to_float(v[value_index]) - kv_mem) * b;
        float result = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] += kr[i] * delta;
            result += st[i] * qr[i];
        }
        output[value_index] = float_to_half(result * q_scale);
    }
#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) {
        sh[static_cast<size_t>(i) * value_dim + value] = st[i];
    }
}

template <int kKeyDim, int kValueTile, int kChunk>
__global__ void gated_delta_sequence_normalized_shared_f16_kernel(
    float* state, const float* q_normalized, const float* k_normalized,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads,
    int value_dim, float q_scale) {
    const int tile = static_cast<int>(blockIdx.x) %
                     ((value_dim + kValueTile - 1) / kValueTile);
    const int head = static_cast<int>(blockIdx.x) /
                     ((value_dim + kValueTile - 1) / kValueTile);
    const int value = tile * kValueTile + static_cast<int>(threadIdx.x);
    if (head >= heads || value >= value_dim) return;

    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    const int key_stride = key_heads * kKeyDim;
    const int lane = static_cast<int>(threadIdx.x);
    extern __shared__ float state_tile[];
    float* sh = state + static_cast<size_t>(head) * kKeyDim * value_dim;

    // Keep the recurrent state in shared memory and use a compile-time key
    // chunk only to control loop unrolling. This preserves the original scalar
    // FP32 order without keeping the full state vector in registers.
    for (int i = 0; i < kKeyDim; ++i) {
        state_tile[i * kValueTile + lane] =
            sh[static_cast<size_t>(i) * value_dim + value];
    }
    for (int token = 0; token < rows; ++token) {
        const float* qr = q_normalized +
            static_cast<size_t>(token) * key_stride +
            static_cast<size_t>(key_head) * kKeyDim;
        const float* kr = k_normalized +
            static_cast<size_t>(token) * key_stride +
            static_cast<size_t>(key_head) * kKeyDim;
        const size_t gate_index = static_cast<size_t>(token) * heads + head;
        const float decay = expf(half_to_float(g[gate_index]));
        const float b = half_to_float(beta[gate_index]);
        float kv_mem = 0.0f;
        for (int base = 0; base < kKeyDim; base += kChunk) {
#pragma unroll
            for (int j = 0; j < kChunk; ++j) {
                const int i = base + j;
                const int index = i * kValueTile + lane;
                const float cell = state_tile[index] * decay;
                state_tile[index] = cell;
                kv_mem += cell * kr[i];
            }
        }
        const size_t value_index = static_cast<size_t>(token) * heads * value_dim +
                                   static_cast<size_t>(head) * value_dim + value;
        const float delta = (half_to_float(v[value_index]) - kv_mem) * b;
        float result = 0.0f;
        for (int base = 0; base < kKeyDim; base += kChunk) {
#pragma unroll
            for (int j = 0; j < kChunk; ++j) {
                const int i = base + j;
                const int index = i * kValueTile + lane;
                const float cell = state_tile[index] + kr[i] * delta;
                state_tile[index] = cell;
                result += cell * qr[i];
            }
        }
        output[value_index] = float_to_half(result * q_scale);
    }

    for (int i = 0; i < kKeyDim; ++i) {
        sh[static_cast<size_t>(i) * value_dim + value] =
            state_tile[i * kValueTile + lane];
    }
}

template <int kKeyDim>
__global__ void gated_delta_sequence_f16_kernel(
    float* state, const uint16_t* q, const uint16_t* k,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads,
    int value_dim, float q_scale) {
    const int head = static_cast<int>(blockIdx.x);
    const int value = static_cast<int>(threadIdx.x);
    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    const int key_stride = key_heads * kKeyDim;
    __shared__ float q_shared[kKeyDim];
    __shared__ float k_shared[kKeyDim];
    extern __shared__ float reduce[];
    float* sh = state + static_cast<size_t>(head) * kKeyDim * value_dim;
    float st[kKeyDim];
#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) st[i] = sh[static_cast<size_t>(i) * value_dim + value];
    for (int token = 0; token < rows; ++token) {
        const uint16_t* qr = q + static_cast<size_t>(token) * key_stride + static_cast<size_t>(key_head) * kKeyDim;
        const uint16_t* kr = k + static_cast<size_t>(token) * key_stride + static_cast<size_t>(key_head) * kKeyDim;
        float q_partial = 0.0f;
        float k_partial = 0.0f;
        for (int i = value; i < kKeyDim; i += blockDim.x) {
            const float qv = half_to_float(qr[i]);
            const float kv = half_to_float(kr[i]);
            q_partial += qv * qv;
            k_partial += kv * kv;
        }
        reduce[value] = q_partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (value < stride) reduce[value] += reduce[value + stride];
            __syncthreads();
        }
        const float q_inv = rsqrtf(reduce[0] + 1.0e-6f);
        __syncthreads();
        reduce[value] = k_partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (value < stride) reduce[value] += reduce[value + stride];
            __syncthreads();
        }
        const float k_inv = rsqrtf(reduce[0] + 1.0e-6f);
        for (int i = value; i < kKeyDim; i += blockDim.x) {
            q_shared[i] = half_to_float(qr[i]) * q_inv;
            k_shared[i] = half_to_float(kr[i]) * k_inv;
        }
        __syncthreads();
        const size_t gate_index = static_cast<size_t>(token) * heads + head;
        const float decay = expf(half_to_float(g[gate_index]));
        const float b = half_to_float(beta[gate_index]);
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] *= decay;
            kv_mem += st[i] * k_shared[i];
        }
        const size_t value_index = static_cast<size_t>(token) * heads * value_dim +
                                   static_cast<size_t>(head) * value_dim + value;
        const float delta = (half_to_float(v[value_index]) - kv_mem) * b;
        float result = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] += k_shared[i] * delta;
            result += st[i] * q_shared[i];
        }
        output[value_index] = float_to_half(result * q_scale);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) sh[static_cast<size_t>(i) * value_dim + value] = st[i];
}

__device__ __forceinline__ float rope_inv_freq(int index, int rotary_dim, float theta) {
    return powf(theta, -2.0f * static_cast<float>(index) / static_cast<float>(rotary_dim));
}

__global__ void partial_rope_f16_kernel(
    uint16_t* q, uint16_t* k, int start_position, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim) {
    const int half = rotary_dim / 2;
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y);
    if (index >= half || row >= rows) return;
    const float angle = static_cast<float>(start_position + row) * rope_inv_freq(index, rotary_dim, theta);
    const float c = cosf(angle);
    const float s = sinf(angle);
    uint16_t* qr = q + static_cast<size_t>(row) * q_heads * head_dim;
    uint16_t* kr = k + static_cast<size_t>(row) * kv_heads * head_dim;
    for (int head = 0; head < q_heads; ++head) {
        uint16_t* line = qr + static_cast<size_t>(head) * head_dim;
        const float a = half_to_float(line[index]);
        const float b = half_to_float(line[index + half]);
        line[index] = float_to_half(a * c - b * s);
        line[index + half] = float_to_half(b * c + a * s);
    }
    for (int head = 0; head < kv_heads; ++head) {
        uint16_t* line = kr + static_cast<size_t>(head) * head_dim;
        const float a = half_to_float(line[index]);
        const float b = half_to_float(line[index + half]);
        line[index] = float_to_half(a * c - b * s);
        line[index + half] = float_to_half(b * c + a * s);
    }
}

__global__ void split_q_gate_f16_kernel(
    const uint16_t* source, uint16_t* q, uint16_t* gate,
    int rows, int q_heads, int head_dim) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * q_heads * head_dim;
    if (index >= total) return;
    const int d = index % head_dim;
    const int head = (index / head_dim) % q_heads;
    const int row = index / (q_heads * head_dim);
    const size_t base = (static_cast<size_t>(row) * q_heads + head) * head_dim * 2;
    q[index] = source[base + d];
    gate[index] = source[base + head_dim + d];
}

__global__ void sigmoid_mul_f16_kernel(const uint16_t* x, const uint16_t* gate,
                                       uint16_t* y, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) y[index] = float_to_half(half_to_float(x[index]) * sigmoid(half_to_float(gate[index])));
}

__global__ void add_f16_kernel(uint16_t* y, const uint16_t* x, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) y[index] = float_to_half(half_to_float(y[index]) + half_to_float(x[index]));
}

__global__ void silu_mul_f16_kernel(const uint16_t* gate, const uint16_t* up,
                                    uint16_t* y, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) y[index] = float_to_half(silu(half_to_float(gate[index])) * half_to_float(up[index]));
}

__global__ void append_kv_f16_kernel(
    const uint16_t* k_rows, const uint16_t* v_rows,
    uint16_t* k_cache, uint16_t* v_cache, int total,
    int kv_heads, int head_dim, int start_pos, int max_context) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int token = index / (kv_heads * head_dim);
    const int within = index % (kv_heads * head_dim);
    const int destination_token = start_pos + token;
    if (destination_token >= max_context) return;
    const size_t destination = static_cast<size_t>(destination_token) * kv_heads * head_dim + within;
    k_cache[destination] = k_rows[index];
    v_cache[destination] = v_rows[index];
}

__global__ void append_kv_fp8_kernel(
    const uint16_t* k_rows, const uint16_t* v_rows,
    uint8_t* k_cache, uint8_t* v_cache,
    uint16_t* k_scale, uint16_t* v_scale,
    int seq_len, int kv_heads, int head_dim, int scale_block,
    int start_pos, int max_context) {
    const int block_index = static_cast<int>(blockIdx.x);
    const int blocks_per_head = head_dim / scale_block;
    const int token = block_index / (kv_heads * blocks_per_head);
    const int within = block_index % (kv_heads * blocks_per_head);
    const int head = within / blocks_per_head;
    const int channel_block = within % blocks_per_head;
    if (token >= seq_len || start_pos + token >= max_context) return;
    const int source_base = (token * kv_heads + head) * head_dim + channel_block * scale_block;
    float k_max = 0.0f;
    float v_max = 0.0f;
    for (int d = threadIdx.x; d < scale_block; d += blockDim.x) {
        k_max = fmaxf(k_max, fabsf(half_to_float(k_rows[source_base + d])));
        v_max = fmaxf(v_max, fabsf(half_to_float(v_rows[source_base + d])));
    }
    __shared__ float reduce_k[128];
    __shared__ float reduce_v[128];
    reduce_k[threadIdx.x] = k_max;
    reduce_v[threadIdx.x] = v_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_k[threadIdx.x] = fmaxf(reduce_k[threadIdx.x], reduce_k[threadIdx.x + stride]);
            reduce_v[threadIdx.x] = fmaxf(reduce_v[threadIdx.x], reduce_v[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float ks = reduce_k[0] > 0.0f ? reduce_k[0] / 448.0f : 1.0f;
    const float vs = reduce_v[0] > 0.0f ? reduce_v[0] / 448.0f : 1.0f;
    const int destination_token = start_pos + token;
    const size_t scale_index = (static_cast<size_t>(destination_token) * kv_heads + head) * blocks_per_head + channel_block;
    if (threadIdx.x == 0) {
        k_scale[scale_index] = float_to_half(ks);
        v_scale[scale_index] = float_to_half(vs);
    }
    const size_t destination_base = (static_cast<size_t>(destination_token) * kv_heads + head) * head_dim + channel_block * scale_block;
    for (int d = threadIdx.x; d < scale_block; d += blockDim.x) {
        k_cache[destination_base + d] = float_to_fp8_e4m3(half_to_float(k_rows[source_base + d]) / ks);
        v_cache[destination_base + d] = float_to_fp8_e4m3(half_to_float(v_rows[source_base + d]) / vs);
    }
}

template <bool kFp8Cache>
__device__ __forceinline__ float cache_value(
    const void* cache, const uint16_t* scale, size_t value_index,
    size_t scale_index) {
    if constexpr (kFp8Cache) {
        return fp8_e4m3_to_float(static_cast<const uint8_t*>(cache)[value_index]) *
               half_to_float(scale[scale_index]);
    }
    return half_to_float(static_cast<const uint16_t*>(cache)[value_index]);
}

template <bool kFp8Cache, int kThreadsPerBlock>
__global__ void gqa_scores_f16_kernel(
    const uint16_t* q, const void* k_cache, const uint16_t* k_scale,
    float* scores, int q_heads, int kv_heads, int head_dim,
    int scale_block, int context_len) {
    const uint64_t work = static_cast<uint64_t>(blockIdx.x);
    const int head = static_cast<int>(work / static_cast<uint64_t>(context_len));
    const int position = static_cast<int>(work % static_cast<uint64_t>(context_len));
    if (head >= q_heads) return;
    const int kv_head = head / (q_heads / kv_heads);
    const uint16_t* qr = q + static_cast<size_t>(head) * head_dim;
    const int blocks_per_head = head_dim / scale_block;
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += kThreadsPerBlock) {
        const size_t value_index = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim + d;
        const size_t scale_index = (static_cast<size_t>(position) * kv_heads + kv_head) * blocks_per_head + d / scale_block;
        partial += half_to_float(qr[d]) * cache_value<kFp8Cache>(k_cache, k_scale, value_index, scale_index);
    }
    extern __shared__ float reduce[];
    reduce[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[static_cast<size_t>(head) * context_len + position] =
            reduce[0] * rsqrtf(static_cast<float>(head_dim));
    }
}

template <int kThreadsPerBlock>
__global__ void softmax_scores_kernel(float* scores, int q_heads, int context_len) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= q_heads) return;
    float* line = scores + static_cast<size_t>(head) * context_len;
    __shared__ float reduce[kThreadsPerBlock];
    float local_max = -INFINITY;
    for (int pos = threadIdx.x; pos < context_len; pos += kThreadsPerBlock) local_max = fmaxf(local_max, line[pos]);
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduce[0];
    float sum = 0.0f;
    for (int pos = threadIdx.x; pos < context_len; pos += kThreadsPerBlock) {
        line[pos] = expf(line[pos] - maximum);
        sum += line[pos];
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int pos = threadIdx.x; pos < context_len; pos += kThreadsPerBlock) line[pos] *= inverse;
}

template <bool kFp8Cache, int kThreadsPerBlock, int kHeadsPerGroup>
__global__ void gqa_values_f16_kernel(
    const float* probabilities, const void* v_cache, const uint16_t* v_scale,
    uint16_t* output, int q_heads, int kv_heads, int head_dim,
    int scale_block, int context_len) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv = (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int d = static_cast<int>(blockIdx.y) * kThreadsPerBlock + threadIdx.x;
    if (kv_head >= kv_heads || first_head >= (kv_head + 1) * q_per_kv || d >= head_dim) return;
    const int blocks_per_head = head_dim / scale_block;
    float acc[kHeadsPerGroup] = {};
    for (int pos = 0; pos < context_len; ++pos) {
        const size_t value_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim + d;
        const size_t scale_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * blocks_per_head + d / scale_block;
        const float value = cache_value<kFp8Cache>(v_cache, v_scale, value_index, scale_index);
#pragma unroll
        for (int i = 0; i < kHeadsPerGroup; ++i) {
            const int head = first_head + i;
            if (head < (kv_head + 1) * q_per_kv) acc[i] += probabilities[static_cast<size_t>(head) * context_len + pos] * value;
        }
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerGroup; ++i) {
        const int head = first_head + i;
        if (head < (kv_head + 1) * q_per_kv) output[static_cast<size_t>(head) * head_dim + d] = float_to_half(acc[i]);
    }
}

template <bool kFp8Cache, int kThreadsPerBlock>
__global__ void gqa_prefill_f16_kernel(
    const uint16_t* q_rows, const void* k_cache, const void* v_cache,
    const uint16_t* k_scale, const uint16_t* v_scale,
    uint16_t* output, int seq_len, int q_heads, int kv_heads,
    int head_dim, int scale_block, int position_offset) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (head >= q_heads || token >= seq_len) return;
    const int kv_head = head / (q_heads / kv_heads);
    const int context_len = position_offset + token + 1;
    const size_t query_offset = (static_cast<size_t>(token) * q_heads + head) * head_dim;
    const uint16_t* query = q_rows + query_offset;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* reduce = smem + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += kThreadsPerBlock) q_shared[d] = half_to_float(query[d]);
    __syncthreads();
    constexpr int kMaxSlice = 8;
    float acc[kMaxSlice] = {};
    const int slice = (head_dim + kThreadsPerBlock - 1) / kThreadsPerBlock;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const int blocks_per_head = head_dim / scale_block;
    for (int pos = 0; pos < context_len; ++pos) {
        float partial = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += kThreadsPerBlock) {
            const size_t value_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim + d;
            const size_t scale_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * blocks_per_head + d / scale_block;
            partial += q_shared[d] * cache_value<kFp8Cache>(k_cache, k_scale, value_index, scale_index);
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
            __syncthreads();
        }
        const float score = reduce[0] * rsqrtf(static_cast<float>(head_dim));
        __syncthreads();
        const float new_max = fmaxf(running_max, score);
        const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - new_max);
        const float probability = expf(score - new_max);
        running_sum = running_sum * rescale + probability;
        running_max = new_max;
        for (int i = 0, d = threadIdx.x; i < slice && d < head_dim; ++i, d += kThreadsPerBlock) {
            const size_t value_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim + d;
            const size_t scale_index = (static_cast<size_t>(pos) * kv_heads + kv_head) * blocks_per_head + d / scale_block;
            acc[i] = acc[i] * rescale + probability * cache_value<kFp8Cache>(v_cache, v_scale, value_index, scale_index);
        }
    }
    const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (int i = 0, d = threadIdx.x; i < slice && d < head_dim; ++i, d += kThreadsPerBlock) {
        output[query_offset + d] = float_to_half(acc[i] * inverse);
    }
}

bool valid_attention(int q_heads, int kv_heads, int head_dim,
                     int context_len, int max_context, int scale_block) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
           head_dim > 0 && context_len > 0 && context_len <= max_context &&
           scale_block > 0 && head_dim % scale_block == 0;
}

template <bool kFp8Cache>
bool launch_decode_attention(
    const uint16_t* q, const void* k_cache, const void* v_cache,
    const uint16_t* k_scale, const uint16_t* v_scale,
    uint16_t* output, float* scores, int q_heads, int kv_heads,
    int head_dim, int scale_block, int context_len, int max_context,
    cudaStream_t stream) {
    if (!q || !k_cache || !v_cache || !output || !scores ||
        !valid_attention(q_heads, kv_heads, head_dim, context_len, max_context, scale_block)) return false;
    if constexpr (kFp8Cache) {
        if (!k_scale || !v_scale) return false;
    }
    constexpr int kAttentionThreads = 128;
    const uint64_t score_blocks = static_cast<uint64_t>(q_heads) * context_len;
    if (score_blocks > static_cast<uint64_t>(UINT32_MAX)) return false;
    dim3 score_grid(static_cast<unsigned>(score_blocks), 1, 1);
    gqa_scores_f16_kernel<kFp8Cache, kAttentionThreads><<<score_grid, kAttentionThreads,
        kAttentionThreads * sizeof(float), stream>>>(q, k_cache, k_scale, scores,
        q_heads, kv_heads, head_dim, scale_block, context_len);
    if (cudaGetLastError() != cudaSuccess) return false;
    softmax_scores_kernel<kAttentionThreads><<<q_heads, kAttentionThreads, 0, stream>>>(scores, q_heads, context_len);
    if (cudaGetLastError() != cudaSuccess) return false;
    constexpr int kHeadsPerGroup = 3;
    const int groups_per_kv = ((q_heads / kv_heads) + kHeadsPerGroup - 1) / kHeadsPerGroup;
    dim3 value_grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                    static_cast<unsigned>((head_dim + kAttentionThreads - 1) / kAttentionThreads), 1);
    gqa_values_f16_kernel<kFp8Cache, kAttentionThreads, kHeadsPerGroup>
        <<<value_grid, kAttentionThreads, 0, stream>>>(scores, v_cache, v_scale,
        output, q_heads, kv_heads, head_dim, scale_block, context_len);
    return cudaGetLastError() == cudaSuccess;
}

template <bool kFp8Cache>
bool launch_prefill_attention(
    const uint16_t* q, const void* k_cache, const void* v_cache,
    const uint16_t* k_scale, const uint16_t* v_scale,
    uint16_t* output, int seq_len, int q_heads, int kv_heads,
    int head_dim, int scale_block, int position_offset, int max_context,
    cudaStream_t stream) {
    if (!q || !k_cache || !v_cache || !output || seq_len <= 0 || position_offset < 0 ||
        position_offset + seq_len > max_context ||
        !valid_attention(q_heads, kv_heads, head_dim, position_offset + seq_len,
                         max_context, scale_block)) return false;
    if constexpr (kFp8Cache) {
        if (!k_scale || !v_scale) return false;
    }
    constexpr int kAttentionThreads = 128;
    if ((head_dim + kAttentionThreads - 1) / kAttentionThreads > 8) return false;
    dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(seq_len), 1);
    const size_t shared = (static_cast<size_t>(head_dim) + kAttentionThreads) * sizeof(float);
    gqa_prefill_f16_kernel<kFp8Cache, kAttentionThreads><<<grid, kAttentionThreads, shared, stream>>>(
        q, k_cache, v_cache, k_scale, v_scale, output, seq_len, q_heads, kv_heads,
        head_dim, scale_block, position_offset);
    return cudaGetLastError() == cudaSuccess;
}

template <int kWarps>
bool launch_fp8_swiglu_small_batch_f16(
    const uint16_t* x, const uint8_t* gate_weight,
    const uint16_t* gate_scale, const uint8_t* up_weight,
    const uint16_t* up_scale, uint16_t* y, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride, int scale_stride,
    bool use_shared, int tile, cudaStream_t stream) {
    const int blocks = (rows + kWarps - 1) / kWarps;
#define LAUNCH_FP8_SWIGLU_TILED(CAPACITY, TILE) \
    fp8_swiglu_small_batch_shared_f16_kernel<kWarps, CAPACITY, TILE> \
        <<<blocks, kWarps * 32, 0, stream>>>( \
            x, gate_weight, gate_scale, up_weight, up_scale, y, batch, rows, \
            cols, x_stride, y_stride, weight_stride, scale_stride)
#define LAUNCH_FP8_SWIGLU_CAP(TILE) \
    do { \
        if (batch <= 2) { LAUNCH_FP8_SWIGLU_TILED(2, TILE); } \
        else if (batch == 3) { LAUNCH_FP8_SWIGLU_TILED(3, TILE); } \
        else if (batch == 4) { LAUNCH_FP8_SWIGLU_TILED(4, TILE); } \
        else if (batch == 5) { LAUNCH_FP8_SWIGLU_TILED(5, TILE); } \
        else { LAUNCH_FP8_SWIGLU_TILED(8, TILE); } \
    } while (0)
    if (use_shared) {
        if (tile == 512) { LAUNCH_FP8_SWIGLU_CAP(512); }
        else if (tile == 2048) { LAUNCH_FP8_SWIGLU_CAP(2048); }
        else { LAUNCH_FP8_SWIGLU_CAP(1024); }
    } else if (batch <= 2) {
        fp8_swiglu_small_batch_f16_kernel<kWarps, 2>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, gate_weight, gate_scale, up_weight, up_scale, y, batch,
                rows, cols, x_stride, y_stride, weight_stride, scale_stride);
    } else if (batch == 3) {
        fp8_swiglu_small_batch_f16_kernel<kWarps, 3>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, gate_weight, gate_scale, up_weight, up_scale, y, batch,
                rows, cols, x_stride, y_stride, weight_stride, scale_stride);
    } else if (batch == 4) {
        fp8_swiglu_small_batch_f16_kernel<kWarps, 4>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, gate_weight, gate_scale, up_weight, up_scale, y, batch,
                rows, cols, x_stride, y_stride, weight_stride, scale_stride);
    } else if (batch == 5) {
        fp8_swiglu_small_batch_f16_kernel<kWarps, 5>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, gate_weight, gate_scale, up_weight, up_scale, y, batch,
                rows, cols, x_stride, y_stride, weight_stride, scale_stride);
    } else {
        fp8_swiglu_small_batch_f16_kernel<kWarps, 8>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, gate_weight, gate_scale, up_weight, up_scale, y, batch,
                rows, cols, x_stride, y_stride, weight_stride, scale_stride);
    }
#undef LAUNCH_FP8_SWIGLU_CAP
#undef LAUNCH_FP8_SWIGLU_TILED
    return cudaGetLastError() == cudaSuccess;
}

template <int kWarps>
bool launch_fp8_matmul_f16_small_batch(
    const uint16_t* x, const uint8_t* weight, const uint16_t* scale,
    uint16_t* y, int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, int scale_stride, bool use_shared,
    cudaStream_t stream) {
    const int blocks = (rows + kWarps - 1) / kWarps;
    if (use_shared && batch > 1) {
#define LAUNCH_FP8_SMALL_PAIRED(CAPACITY) \
        fp8_matmul_f16_small_batch_paired_shared_kernel< \
            kWarps, CAPACITY, 1024> \
            <<<blocks, kWarps * 32, 0, stream>>>( \
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride, \
                weight_stride, scale_stride)
        if (batch == 2) { LAUNCH_FP8_SMALL_PAIRED(2); }
        else if (batch == 3) { LAUNCH_FP8_SMALL_PAIRED(3); }
        else if (batch == 4) { LAUNCH_FP8_SMALL_PAIRED(4); }
        else if (batch == 5) { LAUNCH_FP8_SMALL_PAIRED(5); }
        else { LAUNCH_FP8_SMALL_PAIRED(8); }
#undef LAUNCH_FP8_SMALL_PAIRED
        return cudaGetLastError() == cudaSuccess;
    }
    if (batch <= 2) {
        fp8_matmul_f16_small_batch_kernel<kWarps, 2>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
    } else if (batch == 3) {
        fp8_matmul_f16_small_batch_kernel<kWarps, 3>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
    } else if (batch == 4) {
        fp8_matmul_f16_small_batch_kernel<kWarps, 4>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
    } else if (batch == 5) {
        fp8_matmul_f16_small_batch_kernel<kWarps, 5>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
    } else {
        fp8_matmul_f16_small_batch_kernel<kWarps, 8>
            <<<blocks, kWarps * 32, 0, stream>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
    const uint16_t* x, const uint8_t* weight, const uint16_t* scale,
    uint16_t* y, int rows, int cols, int weight_stride, int scale_stride,
    void* stream) {
    if (!x || !weight || !scale || !y || rows <= 0 || cols <= 0 ||
        weight_stride < cols || scale_stride < (cols + kFp8WeightBlock - 1) / kFp8WeightBlock) return false;
    constexpr int kWarps = 8;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const char* multirow = std::getenv("QWEN_FP8_F16_MULTIROW");
    const char* rows_env = std::getenv("QWEN_FP8_F16_MULTIROW_ROWS");
    const char* vectorized = std::getenv("QWEN_FP8_F16_VECTORIZE");
    const bool multirow_disabled =
        multirow != nullptr && std::strcmp(multirow, "0") == 0;
    const bool use_multirow = multirow != nullptr && !multirow_disabled;
    const bool use_vectorized =
        !multirow_disabled &&
        (vectorized == nullptr || std::strcmp(vectorized, "0") != 0);
    const int requested_rows = rows_env != nullptr ? std::atoi(rows_env) : 0;
    constexpr int kMinBlocks = 136;
    auto blocks_for = [&](int rows_per_warp) {
        return (rows + kWarps * rows_per_warp - 1) / (kWarps * rows_per_warp);
    };
    if (use_multirow && weight_stride % 4 == 0 && requested_rows == 4 &&
        blocks_for(4) >= kMinBlocks) {
        fp8_matvec_f16_multirow_kernel<kWarps, 4><<<blocks_for(4), kWarps * 32, 0, s>>>(
            x, weight, scale, y, rows, cols, weight_stride, scale_stride);
    } else if (use_multirow && weight_stride % 4 == 0 && requested_rows == 2 &&
               blocks_for(2) >= kMinBlocks) {
        fp8_matvec_f16_multirow_kernel<kWarps, 2><<<blocks_for(2), kWarps * 32, 0, s>>>(
            x, weight, scale, y, rows, cols, weight_stride, scale_stride);
    } else if (use_vectorized && weight_stride % 4 == 0) {
        fp8_matvec_f16_multirow_kernel<kWarps, 1><<<blocks_for(1), kWarps * 32, 0, s>>>(
            x, weight, scale, y, rows, cols, weight_stride, scale_stride);
    } else {
        fp8_matvec_f16_kernel<kWarps><<<(rows + kWarps - 1) / kWarps,
            kWarps * 32, 0, s>>>(x, weight, scale, y, rows, cols,
            weight_stride, scale_stride);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
    const uint16_t* x, const uint8_t* gate_weight,
    const uint16_t* gate_scale, const uint8_t* up_weight,
    const uint16_t* up_scale, uint16_t* y, int rows, int cols,
    int weight_stride, int scale_stride, void* stream) {
    if (!x || !gate_weight || !gate_scale || !up_weight || !up_scale || !y ||
        rows <= 0 || cols <= 0 || weight_stride < cols ||
        scale_stride < (cols + kFp8WeightBlock - 1) / kFp8WeightBlock) return false;
    constexpr int kWarps = 8;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const char* multirow = std::getenv("QWEN_FP8_F16_MULTIROW");
    const char* rows_env = std::getenv("QWEN_FP8_F16_MULTIROW_ROWS");
    const bool use_multirow =
        multirow != nullptr && std::strcmp(multirow, "0") != 0;
    const int requested_rows = rows_env != nullptr ? std::atoi(rows_env) : 0;
    constexpr int kMinBlocks = 136;
    auto blocks_for = [&](int rows_per_warp) {
        return (rows + kWarps * rows_per_warp - 1) / (kWarps * rows_per_warp);
    };
    if (use_multirow && weight_stride % 4 == 0 && requested_rows == 4 &&
        blocks_for(4) >= kMinBlocks) {
        fp8_swiglu_matvec_f16_kernel<kWarps, 4><<<blocks_for(4), kWarps * 32, 0, s>>>(
            x, gate_weight, gate_scale, up_weight, up_scale, y, rows, cols,
            weight_stride, scale_stride);
    } else if (use_multirow && weight_stride % 4 == 0 && requested_rows == 2 &&
               blocks_for(2) >= kMinBlocks) {
        fp8_swiglu_matvec_f16_kernel<kWarps, 2><<<blocks_for(2), kWarps * 32, 0, s>>>(
            x, gate_weight, gate_scale, up_weight, up_scale, y, rows, cols,
            weight_stride, scale_stride);
    } else {
        // The fused gate/up kernel has no aligned vectorized variant yet; keep
        // its one-row path independent of the weight stride alignment.
        fp8_swiglu_matvec_f16_kernel<kWarps, 1><<<blocks_for(1), kWarps * 32, 0, s>>>(
            x, gate_weight, gate_scale, up_weight, up_scale, y, rows, cols,
            weight_stride, scale_stride);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
    const uint16_t* x, const uint8_t* gate_weight,
    const uint16_t* gate_scale, const uint8_t* up_weight,
    const uint16_t* up_scale, uint16_t* y, int batch, int rows, int cols,
    int x_stride, int y_stride, int weight_stride, int scale_stride,
    void* stream) {
    if (!x || !gate_weight || !gate_scale || !up_weight || !up_scale || !y ||
        batch <= 1 || batch > 8 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride < cols ||
        x_stride % 4 != 0 || weight_stride % 4 != 0 ||
        scale_stride < (cols + kFp8WeightBlock - 1) / kFp8WeightBlock) {
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const char* shared_env = std::getenv("QWEN_FP8_F16_SMALL_BATCH_SHARED");
    const bool use_shared =
        shared_env == nullptr || std::strcmp(shared_env, "0") != 0;
    int tile = 1024;
    if (const char* tile_env = std::getenv("QWEN_FP8_SWIGLU_TILE")) {
        const int parsed = std::atoi(tile_env);
        if (parsed == 512 || parsed == 1024 || parsed == 2048) tile = parsed;
    }
    int warps = batch >= 3 ? 16 : 8;
    if (const char* warps_env = std::getenv("QWEN_FP8_SWIGLU_WARPS")) {
        const int parsed = std::atoi(warps_env);
        if (parsed == 8 || parsed == 16) warps = parsed;
    }
    if (warps == 16) {
        return launch_fp8_swiglu_small_batch_f16<16>(
            x, gate_weight, gate_scale, up_weight, up_scale, y, batch, rows,
            cols, x_stride, y_stride, weight_stride, scale_stride, use_shared,
            tile, s);
    }
    return launch_fp8_swiglu_small_batch_f16<8>(
        x, gate_weight, gate_scale, up_weight, up_scale, y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride, use_shared, tile, s);
}

bool qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
    const uint8_t* weight, const uint16_t* scale, uint16_t* output,
    int rows, int cols, int weight_stride, int scale_stride, void* stream) {
    if (!weight || !scale || !output || rows <= 0 || cols <= 0 ||
        weight_stride < cols ||
        scale_stride < (cols + kFp8WeightBlock - 1) / kFp8WeightBlock) {
        return false;
    }
    return dequantize_fp8_weight_f16(
        weight, scale, output, rows, cols, weight_stride, scale_stride,
        static_cast<cudaStream_t>(stream));
}

bool qwen_fp16_matmul_rows_f16_cublas_cuda(
    const uint16_t* x, const uint16_t* weight, uint16_t* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, void* stream) {
    if (!x || !weight || !y || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride < cols) {
        return false;
    }
    return fp16_matmul_f16_cublas(
        x, weight, y, batch, rows, cols, x_stride, y_stride, weight_stride,
        static_cast<cudaStream_t>(stream));
}

bool qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
    const uint16_t* x, const uint8_t* weight, const uint16_t* scale,
    uint16_t* y, int batch, int rows, int cols, int x_stride,
    int y_stride, int weight_stride, int scale_stride, void* stream) {
    if (!x || !weight || !scale || !y || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride < cols ||
        scale_stride < (cols + kFp8WeightBlock - 1) / kFp8WeightBlock) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const char* small_batch_env = std::getenv("QWEN_FP8_F16_SMALL_BATCH");
    const bool use_small_batch = small_batch_env == nullptr ||
        std::strcmp(small_batch_env, "0") != 0;
    // Prefill-sized batches go to the tensor-core cuBLAS path by default. The
    // hand-written tiles scale linearly with batch because each output tile
    // re-reads the FP8 weight block, while cuBLAS dequantises the block once into
    // FP16 scratch and then runs a tensor-core GEMM whose cost is nearly flat in
    // batch. Measured on SM75 at the real TP4 projection shapes and batch 512:
    // mlp.gate 2.61 -> 0.56 ms, mlp.down 2.73 -> 0.65, linear.qkv 1.95 -> 0.39,
    // full.q_gate 1.97 -> 0.52, linear.out 0.97 -> 0.26 (3.8x-5.0x). Set
    // QWEN_FP8_F16_PREFILL_CUBLAS=0 to fall back to the previous kernels.
    const char* cublas_env = std::getenv("QWEN_FP8_F16_PREFILL_CUBLAS");
    const bool use_cublas =
        cublas_env == nullptr || std::strcmp(cublas_env, "0") != 0;
    if (use_small_batch && batch <= 8 && x_stride % 4 == 0 &&
        weight_stride % 4 == 0) {
        const char* shared_env =
            std::getenv("QWEN_FP8_F16_SMALL_BATCH_SHARED");
        const bool use_shared =
            shared_env == nullptr || std::strcmp(shared_env, "0") != 0;
        int warps = batch >= 3 ? 16 : 8;
        if (const char* warps_env =
                std::getenv("QWEN_FP8_F16_SMALL_BATCH_WARPS")) {
            const int parsed = std::atoi(warps_env);
            if (parsed == 8 || parsed == 16) warps = parsed;
        }
        if (warps == 16) {
            return launch_fp8_matmul_f16_small_batch<16>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride, use_shared, s);
        }
        return launch_fp8_matmul_f16_small_batch<8>(
            x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride, use_shared, s);
    } else {
        // cuBLAS needs the dequantised weight to be a dense `rows x cols` matrix
        // with leading dimension `cols`. The expansion kernels write that layout
        // only when `cols` is a multiple of four; for other widths the tail kernel
        // and the vector kernel disagree on the row stride, so those shapes stay
        // on the hand-written tiles.
        if (use_cublas && batch >= 96 && cols % 4 == 0) {
            return fp8_matmul_f16_cublas(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride, s);
        }
        const char* wide_env =
            std::getenv("QWEN_FP8_F16_PREFILL_WIDE_N64");
        const bool use_wide =
            wide_env == nullptr || std::strcmp(wide_env, "0") != 0;
        if (use_wide && batch >= 96 && x_stride % 4 == 0 &&
            weight_stride % 4 == 0) {
            dim3 grid(static_cast<unsigned>((rows + 63) / 64),
                      static_cast<unsigned>((batch + 127) / 128), 1);
            fp8_matmul_f16_wide_n64_kernel<<<grid, 256, 0, s>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
        } else {
            dim3 grid(static_cast<unsigned>((rows + 63) / 64),
                      static_cast<unsigned>((batch + 63) / 64), 1);
            fp8_matmul_f16_tiled_kernel<32><<<grid, 256, 0, s>>>(
                x, weight, scale, y, batch, rows, cols, x_stride, y_stride,
                weight_stride, scale_stride);
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

template <bool kFloatOutput>
bool launch_fp16_matmul_rows_f16(
    const uint16_t* x, const uint16_t* weight, void* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, bool allow_small_batch, void* stream) {
    if (!x || !weight || !y || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const char* small_batch_env = std::getenv("QWEN_FP16_SMALL_BATCH");
    const bool use_small_batch = small_batch_env == nullptr ||
        std::strcmp(small_batch_env, "0") != 0;
    constexpr int kWarps = 8;
    if (allow_small_batch && use_small_batch && batch > 1 && batch <= 8 &&
        x_stride % 4 == 0 &&
        weight_stride % 4 == 0) {
        const int blocks = (rows + kWarps - 1) / kWarps;
#define LAUNCH_FP16_SMALL(CAPACITY) \
        fp16_matmul_f16_small_batch_kernel<kFloatOutput, kWarps, CAPACITY> \
            <<<blocks, kWarps * 32, 0, s>>>( \
                x, weight, y, batch, rows, cols, x_stride, y_stride, weight_stride)
        if (batch <= 2) { LAUNCH_FP16_SMALL(2); }
        else if (batch == 3) { LAUNCH_FP16_SMALL(3); }
        else if (batch == 4) { LAUNCH_FP16_SMALL(4); }
        else if (batch == 5) { LAUNCH_FP16_SMALL(5); }
        else { LAUNCH_FP16_SMALL(8); }
#undef LAUNCH_FP16_SMALL
    } else {
        dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
        fp16_matmul_f16_kernel<kFloatOutput>
            <<<grid, kThreads, kThreads * sizeof(float), s>>>(
                x, weight, y, batch, rows, cols, x_stride, y_stride,
                weight_stride);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp16_matmul_rows_f16_cuda(
    const uint16_t* x, const uint16_t* weight, uint16_t* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, void* stream) {
    return launch_fp16_matmul_rows_f16<false>(
        x, weight, y, batch, rows, cols, x_stride, y_stride, weight_stride,
        true, stream);
}

bool qwen_fp16_matmul_rows_f16_f32_cuda(
    const uint16_t* x, const uint16_t* weight, float* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, void* stream) {
    // Keep FP32 logits on the established reduction order. Greedy argmax can be
    // sensitive to near ties even when the small-batch dot product is numerically
    // close; FP16 activation outputs may still use the weight-reuse kernel. The
    // DFlash2 down projection also uses this path so its FP32 residual boundary
    // is preserved and cannot overflow in an intermediate FP16 result.
    return launch_fp16_matmul_rows_f16<true>(
        x, weight, y, batch, rows, cols, x_stride, y_stride, weight_stride,
        false, stream);
}

bool qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
    const uint16_t* x, const uint16_t* weight, float* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int weight_stride, void* stream) {
    if (!x || !weight || !y || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride != cols) {
        return false;
    }
    Fp8F16CublasWorkspace& workspace = fp8_f16_cublas_workspace();
    if (!ensure_fp8_f16_cublas_workspace(workspace, 0, false)) return false;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (cublasSetStream(workspace.handle, cuda_stream) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmEx(
        workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, batch, cols, &alpha,
        weight, CUDA_R_16F, weight_stride,
        x, CUDA_R_16F, x_stride,
        &beta, y, CUDA_R_32F, y_stride,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) ==
        CUBLAS_STATUS_SUCCESS;
}

bool qwen_fp16_swiglu_matmul_rows_f16_cuda(
    const uint16_t* x, const uint16_t* gate_weight,
    const uint16_t* up_weight, uint16_t* y, int batch, int rows,
    int cols, int x_stride, int y_stride, int weight_stride,
    void* stream) {
    if (!x || !gate_weight || !up_weight || !y || batch <= 0 || batch > 8 ||
        rows <= 0 || cols <= 0 || x_stride < cols || y_stride < rows ||
        weight_stride < cols || x_stride % 4 != 0 || weight_stride % 4 != 0) {
        return false;
    }
    const int blocks = (rows + 7) / 8;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (batch <= 2) {
        fp16_swiglu_matmul_f16_kernel<2><<<blocks, 256, 0, s>>>(
            x, gate_weight, up_weight, y, batch, rows, cols, x_stride,
            y_stride, weight_stride);
    } else if (batch == 3) {
        fp16_swiglu_matmul_f16_kernel<3><<<blocks, 256, 0, s>>>(
            x, gate_weight, up_weight, y, batch, rows, cols, x_stride,
            y_stride, weight_stride);
    } else if (batch == 4) {
        fp16_swiglu_matmul_f16_kernel<4><<<blocks, 256, 0, s>>>(
            x, gate_weight, up_weight, y, batch, rows, cols, x_stride,
            y_stride, weight_stride);
    } else if (batch == 5) {
        fp16_swiglu_matmul_f16_kernel<5><<<blocks, 256, 0, s>>>(
            x, gate_weight, up_weight, y, batch, rows, cols, x_stride,
            y_stride, weight_stride);
    } else {
        fp16_swiglu_matmul_f16_kernel<8><<<blocks, 256, 0, s>>>(
            x, gate_weight, up_weight, y, batch, rows, cols, x_stride,
            y_stride, weight_stride);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_embedding_fp16_gather_f16_cuda(
    const uint16_t* table, const int* tokens, uint16_t* output,
    int count, int cols, int row_start, int row_count, void* stream) {
    if (!table || !tokens || !output || count <= 0 || cols <= 0 ||
        row_start < 0 || row_count <= 0) return false;
    embedding_f16_kernel<<<count, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        table, tokens, output, count, cols, row_start, row_count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_concat_rows_f16_cuda(const uint16_t* left, const uint16_t* right,
                               uint16_t* output, int rows, int cols,
                               void* stream) {
    if (!left || !right || !output || rows <= 0 || cols <= 0) return false;
    concat_rows_f16_kernel<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        left, right, output, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_copy_rows_strided_f16_cuda(
    const uint16_t* source, int source_row_stride,
    uint16_t* destination, int destination_row_stride,
    int rows, int columns, void* stream) {
    if (!source || !destination || rows <= 0 || columns <= 0 ||
        source_row_stride < columns || destination_row_stride < columns) {
        return false;
    }
    const cudaError_t status = cudaMemcpy2DAsync(
        destination,
        static_cast<size_t>(destination_row_stride) * sizeof(uint16_t),
        source,
        static_cast<size_t>(source_row_stride) * sizeof(uint16_t),
        static_cast<size_t>(columns) * sizeof(uint16_t),
        static_cast<size_t>(rows),
        cudaMemcpyDeviceToDevice,
        static_cast<cudaStream_t>(stream));
    return status == cudaSuccess;
}

namespace {

template <bool kGather>
__global__ void copy_regions_kernel(
    const QwenCopyRegion* __restrict__ regions, int region_count,
    uint8_t* __restrict__ packed) {
    const uint64_t global_block = static_cast<uint64_t>(blockIdx.x);
    int lo = 0;
    int hi = region_count;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (regions[mid].first_block <= global_block) lo = mid;
        else hi = mid;
    }
    const QwenCopyRegion region = regions[lo];
    const uint64_t local_block = global_block - region.first_block;
    const uint64_t local_offset = local_block * kQwenCopyRegionBlockBytes;
    if (local_offset >= region.bytes) return;
    const uint64_t remaining = region.bytes - local_offset;
    const uint64_t bytes = remaining < kQwenCopyRegionBlockBytes
        ? remaining : kQwenCopyRegionBlockBytes;
    uint8_t* region_data = reinterpret_cast<uint8_t*>(region.device_address) +
        local_offset;
    uint8_t* packed_data = packed + region.packed_offset + local_offset;
    const uint8_t* source = kGather ? region_data : packed_data;
    uint8_t* destination = kGather ? packed_data : region_data;

    const uintptr_t alignment = reinterpret_cast<uintptr_t>(source) |
        reinterpret_cast<uintptr_t>(destination);
    if ((alignment & 0xfu) == 0 && (bytes & 0xfu) == 0) {
        const uint4* source4 = reinterpret_cast<const uint4*>(source);
        uint4* destination4 = reinterpret_cast<uint4*>(destination);
        for (uint64_t index = threadIdx.x; index < bytes / sizeof(uint4);
             index += blockDim.x) {
            destination4[index] = source4[index];
        }
    } else {
        for (uint64_t index = threadIdx.x; index < bytes;
             index += blockDim.x) {
            destination[index] = source[index];
        }
    }
}

template <bool kGather>
bool launch_copy_regions(
    const QwenCopyRegion* regions, int region_count, uint8_t* packed,
    uint64_t total_blocks, void* stream) {
    if (!regions || !packed || region_count <= 0 || total_blocks == 0 ||
        total_blocks > static_cast<uint64_t>(UINT32_MAX)) {
        return false;
    }
    copy_regions_kernel<kGather><<<
        static_cast<unsigned>(total_blocks), kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(regions, region_count, packed);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool qwen_gather_copy_regions_cuda(
    const QwenCopyRegion* regions, int region_count, uint8_t* packed,
    uint64_t total_blocks, void* stream) {
    return launch_copy_regions<true>(
        regions, region_count, packed, total_blocks, stream);
}

bool qwen_scatter_copy_regions_cuda(
    const QwenCopyRegion* regions, int region_count, const uint8_t* packed,
    uint64_t total_blocks, void* stream) {
    return launch_copy_regions<false>(
        regions, region_count, const_cast<uint8_t*>(packed), total_blocks,
        stream);
}

bool qwen_rmsnorm_fp16_gamma_rows_f16_cuda(
    const uint16_t* x, const uint16_t* gamma, uint16_t* y,
    int rows, int cols, float eps, void* stream) {
    if (!x || !gamma || !y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    rmsnorm_f16_kernel<<<rows, kThreads, kThreads * sizeof(float),
        static_cast<cudaStream_t>(stream)>>>(x, gamma, y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_cuda(
    const uint16_t* hidden, const uint16_t* delta, const uint16_t* gamma,
    uint16_t* residual, uint16_t* normalized, int rows, int cols, float eps,
    void* stream) {
    if (!hidden || !delta || !gamma || !residual || !normalized || rows <= 0 ||
        cols <= 0 || eps < 0.0f || residual == hidden || residual == delta ||
        normalized == hidden || normalized == delta || normalized == residual) {
        return false;
    }
    residual_add_rmsnorm_f16_kernel<<<
        rows, kThreads, kThreads * sizeof(float),
        static_cast<cudaStream_t>(stream)>>>(
        hidden, delta, gamma, residual, normalized, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_rmsnorm_fp16_gamma_rows_f16_cuda(
    const uint16_t* x, const uint16_t* gamma, const uint16_t* gate,
    uint16_t* y, int rows, int cols, float eps, void* stream) {
    if (!x || !gamma || !gate || !y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    gated_rmsnorm_f16_kernel<<<rows, kThreads, kThreads * sizeof(float),
        static_cast<cudaStream_t>(stream)>>>(x, gamma, gate, y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_split_packed_qkv_f16_cuda(
    const uint16_t* packed, uint16_t* q, uint16_t* k, uint16_t* v,
    int rows, int key_dim, int value_dim, void* stream) {
    if (!packed || !q || !k || !v || rows <= 0 || key_dim <= 0 || value_dim <= 0) return false;
    const int total = rows * (2 * key_dim + value_dim);
    split_packed_qkv_f16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(packed, q, k, v, rows, key_dim, value_dim);
    return cudaGetLastError() == cudaSuccess;
}

__global__ void split_rows_pair_f16_kernel(const uint16_t* __restrict__ packed,
                                           uint16_t* __restrict__ first,
                                           uint16_t* __restrict__ second,
                                           int rows, int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const int row = index / width;
    const int column = index - row * width;
    const int packed_row = row * 2 * width;
    first[index] = packed[packed_row + column];
    second[index] = packed[packed_row + width + column];
}

bool qwen_split_rows_pair_f16_cuda(const uint16_t* packed, uint16_t* first,
                                   uint16_t* second, int rows, int width,
                                   void* stream) {
    if (!packed || !first || !second || rows <= 0 || width <= 0) return false;
    const int total = rows * width;
    split_rows_pair_f16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(packed, first, second, rows, width);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_causal_depthwise_conv_silu_f16_cuda(
    const uint16_t* x, const uint16_t* weight, uint16_t* tail,
    uint16_t* y, int seq_len, int channels, int kernel,
    bool update_tail, void* stream) {
    if (!x || !weight || !y || seq_len <= 0 || channels <= 0 || kernel <= 0 || kernel - 1 > 8) return false;
    conv_silu_f16_kernel<<<(channels + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(x, weight, tail, y, seq_len,
        channels, kernel, update_tail);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_linear_attn_gates_f16_cuda(
    const uint16_t* a, const uint16_t* b, const uint16_t* a_log,
    const uint16_t* dt_bias, uint16_t* g, uint16_t* beta,
    int rows, int heads, void* stream) {
    if (!a || !b || !a_log || !dt_bias || !g || !beta || rows <= 0 || heads <= 0) return false;
    linear_gates_f16_kernel<<<heads, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        a, b, a_log, dt_bias, g, beta, rows, heads);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_step_f16_cuda(
    float* state, const uint16_t* q, const uint16_t* k,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int heads, int key_heads, int key_dim,
    int value_dim, float q_scale, void* stream) {
    if (!state || !q || !k || !v || !g || !beta || !output || heads <= 0 ||
        key_heads <= 0 || heads % key_heads != 0 || key_dim <= 0 || value_dim <= 0 || q_scale <= 0.0f) return false;
    int threads = 32;
    while (threads < value_dim) threads <<= 1;
    if (threads > 1024) return false;
    const size_t shared = (2u * static_cast<size_t>(key_dim) + threads) * sizeof(float);
    gated_delta_step_f16_kernel<<<heads, threads, shared, static_cast<cudaStream_t>(stream)>>>(
        state, q, k, v, g, beta, output, heads, key_heads, key_dim, value_dim, q_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_sequence_f16_cuda(
    float* state, const uint16_t* q, const uint16_t* k,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads, int key_dim,
    int value_dim, float q_scale, void* stream) {
    if (!state || !q || !k || !v || !g || !beta || !output || rows <= 0 ||
        heads <= 0 || key_heads <= 0 || heads % key_heads != 0 || key_dim != 128 ||
        value_dim != 128 || q_scale <= 0.0f) return false;
    gated_delta_sequence_f16_kernel<128><<<heads, value_dim,
        static_cast<size_t>(value_dim) * sizeof(float),
        static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, g, beta, output, rows, heads, key_heads,
            value_dim, q_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_normalize_gated_delta_qk_f16_cuda(
    const uint16_t* q, const uint16_t* k, float* q_normalized,
    float* k_normalized, int rows, int key_heads, int key_dim, void* stream) {
    if (!q || !k || !q_normalized || !k_normalized || rows <= 0 ||
        key_heads <= 0 || key_dim != 128) {
        return false;
    }
    dim3 grid(static_cast<unsigned>(key_heads),
              static_cast<unsigned>(rows), 1);
    normalize_gated_delta_qk_f16_kernel<<<
        grid, key_dim, 0, static_cast<cudaStream_t>(stream)>>>(
            q, k, q_normalized, k_normalized, rows, key_heads, key_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_sequence_normalized_f16_cuda(
    float* state, const float* q_normalized, const float* k_normalized,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads, int key_dim,
    int value_dim, float q_scale, void* stream) {
    if (!state || !q_normalized || !k_normalized || !v || !g || !beta ||
        !output || rows <= 0 || heads <= 0 || key_heads <= 0 ||
        heads % key_heads != 0 || key_dim != 128 || value_dim != 128 ||
        q_scale <= 0.0f) {
        return false;
    }
    gated_delta_sequence_normalized_f16_kernel<128><<<
        heads, value_dim, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q_normalized, k_normalized, v, g, beta, output, rows,
            heads, key_heads, value_dim, q_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_sequence_normalized_shared_f16_cuda(
    float* state, const float* q_normalized, const float* k_normalized,
    const uint16_t* v, const uint16_t* g, const uint16_t* beta,
    uint16_t* output, int rows, int heads, int key_heads, int key_dim,
    int value_dim, float q_scale, void* stream) {
    if (!state || !q_normalized || !k_normalized || !v || !g || !beta ||
        !output || rows <= 0 || heads <= 0 || key_heads <= 0 ||
        heads % key_heads != 0 || key_dim != 128 || value_dim != 128 ||
        q_scale <= 0.0f) {
        return false;
    }
    // 32 lanes keeps one warp per block with a 16 KiB state tile. 16 halves the
    // tile but doubles the block count, 64 needs 32 KiB and drops to one block
    // per SM pair; the sweep below picks between them at build-probe time.
    auto launch = [&](auto tile_tag, auto chunk_tag) -> bool {
        constexpr int kValueTile = decltype(tile_tag)::value;
        constexpr int kChunk = decltype(chunk_tag)::value;
        const int tiles = (value_dim + kValueTile - 1) / kValueTile;
        const size_t shared_bytes = static_cast<size_t>(key_dim) * kValueTile *
                                    sizeof(float);
        if (shared_bytes > 48u * 1024u) {
            const cudaError_t attribute = cudaFuncSetAttribute(
                gated_delta_sequence_normalized_shared_f16_kernel<
                    128, kValueTile, kChunk>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes));
            if (attribute != cudaSuccess) return false;
        }
        gated_delta_sequence_normalized_shared_f16_kernel<
            128, kValueTile, kChunk><<<
                heads * tiles, kValueTile, shared_bytes,
                static_cast<cudaStream_t>(stream)>>>(
            state, q_normalized, k_normalized, v, g, beta, output, rows,
            heads, key_heads, value_dim, q_scale);
        return cudaGetLastError() == cudaSuccess;
    };
    auto launch_chunk = [&](auto tile_tag, int chunk) -> bool {
        if (chunk == 4) {
            return launch(tile_tag, std::integral_constant<int, 4>{});
        }
        if (chunk == 16) {
            return launch(tile_tag, std::integral_constant<int, 16>{});
        }
        if (chunk == 32) {
            return launch(tile_tag, std::integral_constant<int, 32>{});
        }
        return launch(tile_tag, std::integral_constant<int, 8>{});
    };
    static const int tile_choice = [] {
        const char* raw = std::getenv("QWEN_GATED_DELTA_VALUE_TILE");
        const int parsed = raw != nullptr ? std::atoi(raw) : 32;
        return (parsed == 16 || parsed == 32 || parsed == 64) ? parsed : 32;
    }();
    static const int chunk_choice = [] {
        const char* raw = std::getenv("QWEN_GATED_DELTA_CHUNK");
        const int parsed = raw != nullptr ? std::atoi(raw) : 8;
        return (parsed == 4 || parsed == 8 || parsed == 16 || parsed == 32)
            ? parsed : 8;
    }();
    if (tile_choice == 16) {
        return launch_chunk(std::integral_constant<int, 16>{}, chunk_choice);
    }
    if (tile_choice == 64) {
        return launch_chunk(std::integral_constant<int, 64>{}, chunk_choice);
    }
    return launch_chunk(std::integral_constant<int, 32>{}, chunk_choice);
}

bool qwen_partial_rope_f16_cuda(
    uint16_t* q, uint16_t* k, int position, int rotary_dim,
    float theta, int q_heads, int kv_heads, int head_dim, void* stream) {
    return qwen_partial_rope_rows_f16_cuda(q, k, position, 1, rotary_dim,
        theta, q_heads, kv_heads, head_dim, stream);
}

bool qwen_partial_rope_rows_f16_cuda(
    uint16_t* q, uint16_t* k, int start_position, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads,
    int head_dim, void* stream) {
    if (!q || !k || start_position < 0 || rows <= 0 || rotary_dim <= 0 ||
        (rotary_dim & 1) != 0 || rotary_dim > head_dim || theta <= 0.0f ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) return false;
    dim3 grid(static_cast<unsigned>((rotary_dim / 2 + kThreads - 1) / kThreads),
              static_cast<unsigned>(rows), 1);
    partial_rope_f16_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        q, k, start_position, rows, rotary_dim, theta, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_split_q_gate_f16_cuda(
    const uint16_t* source, uint16_t* q, uint16_t* gate,
    int rows, int q_heads, int head_dim, void* stream) {
    if (!source || !q || !gate || rows <= 0 || q_heads <= 0 || head_dim <= 0) return false;
    const int total = rows * q_heads * head_dim;
    split_q_gate_f16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(source, q, gate, rows, q_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_sigmoid_mul_f16_cuda(
    const uint16_t* x, const uint16_t* gate, uint16_t* y,
    int count, void* stream) {
    if (!x || !gate || !y || count <= 0) return false;
    sigmoid_mul_f16_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(x, gate, y, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_add_inplace_f16_cuda(
    uint16_t* y, const uint16_t* x, int count, void* stream) {
    if (!x || !y || count <= 0) return false;
    add_f16_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(y, x, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_silu_mul_rows_f16_cuda(
    const uint16_t* gate, const uint16_t* up, uint16_t* y,
    int rows, int cols, void* stream) {
    if (!gate || !up || !y || rows <= 0 || cols <= 0) return false;
    const int count = rows * cols;
    silu_mul_f16_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(gate, up, y, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_append_kv_cache_f16_cuda(
    const uint16_t* k_rows, const uint16_t* v_rows,
    uint16_t* k_cache, uint16_t* v_cache, int seq_len,
    int kv_heads, int head_dim, int start_pos, int max_context,
    void* stream) {
    if (!k_rows || !v_rows || !k_cache || !v_cache || seq_len <= 0 ||
        kv_heads <= 0 || head_dim <= 0 || start_pos < 0 || max_context <= 0 ||
        start_pos + seq_len > max_context) return false;
    const int total = seq_len * kv_heads * head_dim;
    append_kv_f16_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(k_rows, v_rows, k_cache, v_cache,
        total, kv_heads, head_dim, start_pos, max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_append_kv_cache_fp8_cuda(
    const uint16_t* k_rows, const uint16_t* v_rows,
    uint8_t* k_cache, uint8_t* v_cache, uint16_t* k_scale,
    uint16_t* v_scale, int seq_len, int kv_heads, int head_dim,
    int scale_block, int start_pos, int max_context, void* stream) {
    if (!k_rows || !v_rows || !k_cache || !v_cache || !k_scale || !v_scale ||
        seq_len <= 0 || kv_heads <= 0 || head_dim <= 0 || scale_block <= 0 ||
        scale_block > 128 || head_dim % scale_block != 0 || start_pos < 0 ||
        max_context <= 0 || start_pos + seq_len > max_context) return false;
    const int blocks = seq_len * kv_heads * (head_dim / scale_block);
    append_kv_fp8_kernel<<<blocks, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        k_rows, v_rows, k_cache, v_cache, k_scale, v_scale, seq_len,
        kv_heads, head_dim, scale_block, start_pos, max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_f16_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* scores,
    int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, void* stream) {
    return launch_decode_attention<false>(q, k_cache, v_cache, nullptr,
        nullptr, output, scores, q_heads, kv_heads, head_dim, head_dim,
        context_len, max_context, static_cast<cudaStream_t>(stream));
}

bool qwen_gqa_prefill_attention_f16_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, int seq_len,
    int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, void* stream) {
    return launch_prefill_attention<false>(q, k_cache, v_cache, nullptr,
        nullptr, output, seq_len, q_heads, kv_heads, head_dim, head_dim,
        position_offset, max_context, static_cast<cudaStream_t>(stream));
}

bool qwen_gqa_decode_attention_fp8_cuda(
    const uint16_t* q, const uint8_t* k_cache,
    const uint8_t* v_cache, const uint16_t* k_scale,
    const uint16_t* v_scale, uint16_t* output, float* scores,
    int q_heads, int kv_heads, int head_dim, int scale_block,
    int context_len, int max_context, void* stream) {
    return launch_decode_attention<true>(q, k_cache, v_cache, k_scale,
        v_scale, output, scores, q_heads, kv_heads, head_dim, scale_block,
        context_len, max_context, static_cast<cudaStream_t>(stream));
}

bool qwen_gqa_prefill_attention_fp8_cuda(
    const uint16_t* q, const uint8_t* k_cache,
    const uint8_t* v_cache, const uint16_t* k_scale,
    const uint16_t* v_scale, uint16_t* output, int seq_len,
    int q_heads, int kv_heads, int head_dim, int scale_block,
    int position_offset, int max_context, void* stream) {
    return launch_prefill_attention<true>(q, k_cache, v_cache, k_scale,
        v_scale, output, seq_len, q_heads, kv_heads, head_dim, scale_block,
        position_offset, max_context, static_cast<cudaStream_t>(stream));
}

}  // namespace dsv4
