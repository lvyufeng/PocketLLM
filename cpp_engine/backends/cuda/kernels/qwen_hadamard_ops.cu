// The activation side of the ternary checkpoint's incoherence transform.
//
// The weights in the file are in a rotated frame -- each matrix was multiplied by
// `R^-1` before it was quantized to three values, which is what spreads the
// quantization error instead of concentrating it -- so the activation that meets
// them has to be rotated into the same frame. `R = (1/sqrt(N)) H_N diag(s)` with
// `H_N[i][j] = (-1)^popcount(i AND j)` the natural-order Sylvester Hadamard
// matrix, applied independently to each `block` run of the last axis.
//
// Two directions, and the file says which tensor takes which:
//
//   forward  `x |-> (1/sqrt(N)) H (s * x)`  signs, then the butterflies
//   inverse  `z |-> (1/sqrt(N)) s * (H z)`  the butterflies, then the signs
//
// The scale multiplies the *input* rather than the output because that is the
// fork's own order (`dst = src * scale`, then the passes) and it is the reason the
// two agree bit for bit rather than nearly: scaling after a ten-deep butterfly
// tree rounds ten times where this rounds once.
//
// The accumulation is fp32 and only the result is narrowed, which is the one place
// the arithmetic can afford to be wide: the rounding error of a ten-pass butterfly
// in fp16 would land in exactly the part of the activation that the 1.75-bit
// weights cannot afford.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// One block per (row, block-run) pair, `block` elements in shared memory, one
// butterfly pass per step. `block` is a power of two and at most a warp's worth of
// steps short of the whole run, so the passes are `log2(block)` synchronizations
// and no more.
template <bool kForward>
__global__ void hadamard_kernel(const uint16_t* __restrict__ x, const float* __restrict__ signs,
                                uint16_t* __restrict__ y, int block, float scale) {
    extern __shared__ float tile[];
    // blockIdx.x walks the block runs within a row and blockIdx.y is the row, so
    // the row pitch is the *number of runs* times the block, which is how many
    // blocks the launch put on x -- not how many rows there are.
    const long row_pitch = (long) gridDim.x * block;
    const long row = blockIdx.y;
    const int base = blockIdx.x * block;
    const uint16_t* source = x + row * row_pitch + base;
    uint16_t* destination = y + row * row_pitch + base;

    // The signs and the scale are elementwise and commute with each other -- a
    // sign is +-1 and so exact -- so folding them into one pass reproduces the
    // reference's two multiplies exactly.
    if (kForward) {
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            tile[i] = __half2float(__ushort_as_half(source[i])) * signs[base + i] * scale;
        }
    } else {
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            tile[i] = __half2float(__ushort_as_half(source[i])) * scale;
        }
    }
    __syncthreads();

    for (int step = 1; step < block; step <<= 1) {
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            // Only the lower half of each 2*step run computes: its partner is the
            // element `step` away, and the pair writes both of them.
            if ((i & step) == 0) {
                const float low = tile[i];
                const float high = tile[i + step];
                tile[i] = low + high;
                tile[i + step] = low - high;
            }
        }
        __syncthreads();
    }

    if (kForward) {
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            destination[i] = __half_as_ushort(__float2half(tile[i]));
        }
    } else {
        for (int i = threadIdx.x; i < block; i += blockDim.x) {
            destination[i] = __half_as_ushort(__float2half(tile[i] * signs[base + i]));
        }
    }
}

template <bool kForward>
bool launch(const uint16_t* x, const float* signs, uint16_t* y, int rows, int width, int block,
            void* stream_ptr) {
    if (x == nullptr || signs == nullptr || y == nullptr) return false;
    if (rows <= 0 || width <= 0 || block <= 0) return false;
    if (width % block != 0 || (block & (block - 1)) != 0) return false;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int threads = block < 1024 ? block : 1024;
    const dim3 grid(width / block, rows, 1);
    const float scale = 1.0f / sqrtf(static_cast<float>(block));
    hadamard_kernel<kForward><<<grid, threads, block * sizeof(float), stream>>>(x, signs, y, block,
                                                                               scale);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

namespace pocket {

bool qwen_hadamard_forward_f16_cuda(const uint16_t* d_x_fp16, const float* d_signs_fp32,
                                    uint16_t* d_y_fp16, int rows, int width, int block,
                                    void* stream) {
    return launch<true>(d_x_fp16, d_signs_fp32, d_y_fp16, rows, width, block, stream);
}

bool qwen_hadamard_inverse_f16_cuda(const uint16_t* d_z_fp16, const float* d_signs_fp32,
                                    uint16_t* d_y_fp16, int rows, int width, int block,
                                    void* stream) {
    return launch<false>(d_z_fp16, d_signs_fp32, d_y_fp16, rows, width, block, stream);
}

}  // namespace pocket
