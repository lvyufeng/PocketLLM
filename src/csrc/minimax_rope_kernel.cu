// MiniMax-M2 half-split RoPE fused kernel (fp16)
// Replaces PyTorch ops in MiniMaxAttention._apply_rope with single CUDA kernel
//
// RoPE style: GPT-Neox half-split (not llama.cpp interleaved)
//   x1 = x[..., :rope_dim/2]
//   x2 = x[..., rope_dim/2:rope_dim]
//   y1 = x1 * cos - x2 * sin
//   y2 = x1 * sin + x2 * cos
//
// Input: [B, S, H, D] fp16 or bf16
// RoPE applied to last rope_dim elements (typically rope_dim=128, D=128)
// freqs_real/freqs_imag: [S, rope_dim/2] fp32 (pre-computed cos/sin)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ float fp16_to_float(__half v) {
    return __half2float(v);
}

__device__ __forceinline__ __half float_to_fp16(float v) {
    return __float2half(v);
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Fused half-split RoPE kernel (fp16)
// One block per (batch, seq, head) row, threads cooperate over D dimension
template <int kThreads>
__global__ void fused_minimax_rope_halfsplit_fp16_kernel(
    __half* __restrict__ x,           // [B, S, H, D] input/output (in-place)
    const float* __restrict__ freqs_cos,  // [S, rope_dim/2] pre-computed cos
    const float* __restrict__ freqs_sin,  // [S, rope_dim/2] pre-computed sin
    int total_rows,                   // B * S * H
    int S,                            // sequence length
    int H,                            // num heads
    int D,                            // head_dim
    int rope_dim) {                   // rope dimension (e.g. 128)

    const int row = blockIdx.x;
    if (row >= total_rows) return;

    // Decode position index from row: row = b*S*H + s*H + h
    const int s_idx = (row / H) % S;
    const int tid = threadIdx.x;

    __half* x_row = x + static_cast<int64_t>(row) * D;
    const int half_rd = rope_dim >> 1;
    const float* cos_row = freqs_cos + s_idx * half_rd;
    const float* sin_row = freqs_sin + s_idx * half_rd;

    // RoPE: x1 = x[:half_rd], x2 = x[half_rd:rope_dim]
    // y1 = x1*cos - x2*sin, y2 = x1*sin + x2*cos
    for (int p = tid; p < half_rd; p += kThreads) {
        float x1 = fp16_to_float(x_row[p]);
        float x2 = fp16_to_float(x_row[half_rd + p]);
        float c = cos_row[p];
        float s = sin_row[p];
        x_row[p] = float_to_fp16(x1 * c - x2 * s);
        x_row[half_rd + p] = float_to_fp16(x1 * s + x2 * c);
    }
    // Tail [rope_dim:D] unchanged
}

// Fused half-split RoPE kernel (bf16)
template <int kThreads>
__global__ void fused_minimax_rope_halfsplit_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ freqs_cos,
    const float* __restrict__ freqs_sin,
    int total_rows,
    int S,
    int H,
    int D,
    int rope_dim) {

    const int row = blockIdx.x;
    if (row >= total_rows) return;

    const int s_idx = (row / H) % S;
    const int tid = threadIdx.x;

    __nv_bfloat16* x_row = x + static_cast<int64_t>(row) * D;
    const int half_rd = rope_dim >> 1;
    const float* cos_row = freqs_cos + s_idx * half_rd;
    const float* sin_row = freqs_sin + s_idx * half_rd;

    for (int p = tid; p < half_rd; p += kThreads) {
        float x1 = bf16_to_float(x_row[p]);
        float x2 = bf16_to_float(x_row[half_rd + p]);
        float c = cos_row[p];
        float s = sin_row[p];
        x_row[p] = float_to_bf16(x1 * c - x2 * s);
        x_row[half_rd + p] = float_to_bf16(x1 * s + x2 * c);
    }
}

}  // namespace

void fused_minimax_rope_halfsplit_inplace_cuda(
    torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin) {

    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be [B, S, H, D]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(freqs_cos.is_cuda() && freqs_sin.is_cuda(), "freqs must be CUDA");
    TORCH_CHECK(freqs_cos.scalar_type() == at::kFloat && freqs_sin.scalar_type() == at::kFloat, "freqs must be fp32");
    TORCH_CHECK(freqs_cos.is_contiguous() && freqs_sin.is_contiguous(), "freqs must be contiguous");
    TORCH_CHECK(freqs_cos.dim() == 2 && freqs_sin.dim() == 2, "freqs must be [S, rope_dim/2]");

    const int B = static_cast<int>(x.size(0));
    const int S = static_cast<int>(x.size(1));
    const int H = static_cast<int>(x.size(2));
    const int D = static_cast<int>(x.size(3));
    const int total_rows = B * S * H;

    TORCH_CHECK(freqs_cos.size(0) == S && freqs_sin.size(0) == S, "freqs S mismatch");
    const int half_rd = static_cast<int>(freqs_cos.size(1));
    const int rope_dim = half_rd * 2;
    TORCH_CHECK(rope_dim <= D && rope_dim > 0, "rope_dim must be > 0 and <= D");

    constexpr int kThreads = 128;
    const dim3 grid(total_rows);
    const dim3 block(kThreads);

    if (x.scalar_type() == at::kHalf) {
        fused_minimax_rope_halfsplit_fp16_kernel<kThreads><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<__half*>(x.data_ptr<at::Half>()),
            freqs_cos.data_ptr<float>(),
            freqs_sin.data_ptr<float>(),
            total_rows, S, H, D, rope_dim
        );
    } else if (x.scalar_type() == at::kBFloat16) {
        fused_minimax_rope_halfsplit_bf16_kernel<kThreads><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            freqs_cos.data_ptr<float>(),
            freqs_sin.data_ptr<float>(),
            total_rows, S, H, D, rope_dim
        );
    } else {
        TORCH_CHECK(false, "x must be fp16 or bf16");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
