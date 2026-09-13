// MiniMax GQA-aware decode attention kernels
// Eliminates repeat_interleave overhead for n_heads != n_kv_heads
//
// Background: MiniMax uses GQA with n_heads=48, n_kv_heads=8 (6× repeat factor)
// Current PyTorch path expands KV 6× via repeat_interleave before SDPA
// This kernel directly maps each Q head to its KV head group without expansion
//
// Decode scenario: seqlen_q=1, seqlen_kv=T (up to 64K)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ float fp16_to_float(__half v) {
    return __half2float(v);
}

__device__ __forceinline__ __half float_to_fp16(float v) {
    return __float2half(v);
}

// Kernel 1: GQA-aware Q·K^T for decode (seqlen_q=1)
// One block per (batch, query_head), threads cooperate over KV sequence
// Q: [B, n_heads, 1, D]
// K: [B, n_kv_heads, T, D]
// scores: [B, n_heads, 1, T]
template <int kThreads>
__global__ void gqa_decode_qk_gemv_fp16_kernel(
    const __half* __restrict__ Q,      // [B, n_heads, 1, D]
    const __half* __restrict__ K,      // [B, n_kv_heads, T, D]
    __half* __restrict__ scores,       // [B, n_heads, 1, T]
    int B, int n_heads, int n_kv_heads,
    int T, int head_dim, float scale) {

    const int total_q_heads = B * n_heads;
    const int q_head_idx = blockIdx.x;
    if (q_head_idx >= total_q_heads) return;

    const int b = q_head_idx / n_heads;
    const int h = q_head_idx % n_heads;
    const int repeat_factor = n_heads / n_kv_heads;
    const int h_kv = h / repeat_factor;  // GQA mapping: query head → KV head

    const int tid = threadIdx.x;

    // Pointers to this query head's Q and corresponding KV head's K
    const __half* q_ptr = Q + static_cast<int64_t>(q_head_idx) * head_dim;
    const __half* k_base = K + (static_cast<int64_t>(b) * n_kv_heads + h_kv) * T * head_dim;
    __half* score_out = scores + static_cast<int64_t>(q_head_idx) * T;

    // Load Q into shared memory (broadcast to all threads)
    extern __shared__ float smem[];
    float* q_shared = smem;

    for (int d = tid; d < head_dim; d += kThreads) {
        q_shared[d] = fp16_to_float(q_ptr[d]);
    }
    __syncthreads();

    // Compute Q·K^T for all T positions (each thread handles subset)
    for (int t = tid; t < T; t += kThreads) {
        const __half* k_ptr = k_base + static_cast<int64_t>(t) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_shared[d] * fp16_to_float(k_ptr[d]);
        }
        score_out[t] = float_to_fp16(dot * scale);
    }
}

// Kernel 2: GQA-aware attn·V for decode
// attn_weights: [B, n_heads, 1, T] (after softmax)
// V: [B, n_kv_heads, T, D]
// out: [B, n_heads, 1, D]
template <int kThreads>
__global__ void gqa_decode_attn_v_gemv_fp16_kernel(
    const __half* __restrict__ attn_weights,  // [B, n_heads, 1, T]
    const __half* __restrict__ V,             // [B, n_kv_heads, T, D]
    __half* __restrict__ out,                 // [B, n_heads, 1, D]
    int B, int n_heads, int n_kv_heads,
    int T, int head_dim) {

    const int total_q_heads = B * n_heads;
    const int q_head_idx = blockIdx.x;
    if (q_head_idx >= total_q_heads) return;

    const int b = q_head_idx / n_heads;
    const int h = q_head_idx % n_heads;
    const int repeat_factor = n_heads / n_kv_heads;
    const int h_kv = h / repeat_factor;

    const int tid = threadIdx.x;

    const __half* attn_ptr = attn_weights + static_cast<int64_t>(q_head_idx) * T;
    const __half* v_base = V + (static_cast<int64_t>(b) * n_kv_heads + h_kv) * T * head_dim;
    __half* out_ptr = out + static_cast<int64_t>(q_head_idx) * head_dim;

    // Each thread computes a subset of output dimensions
    for (int d = tid; d < head_dim; d += kThreads) {
        float sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            float weight = fp16_to_float(attn_ptr[t]);
            float v_val = fp16_to_float(v_base[static_cast<int64_t>(t) * head_dim + d]);
            sum += weight * v_val;
        }
        out_ptr[d] = float_to_fp16(sum);
    }
}

}  // namespace

void gqa_decode_qk_gemv_cuda(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    torch::Tensor& scores,
    int n_heads, int n_kv_heads,
    int T, int head_dim, float scale) {

    c10::cuda::CUDAGuard device_guard(Q.device());
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && scores.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(Q.scalar_type() == at::kHalf && K.scalar_type() == at::kHalf, "Q/K must be fp16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && scores.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must be 4D [B, H, S, D]");

    const int B = static_cast<int>(Q.size(0));
    const int total_q_heads = B * n_heads;

    constexpr int kThreads = 256;
    const int smem = head_dim * sizeof(float);  // Q shared memory
    const dim3 grid(total_q_heads);
    const dim3 block(kThreads);

    gqa_decode_qk_gemv_fp16_kernel<kThreads><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
        B, n_heads, n_kv_heads, T, head_dim, scale
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gqa_decode_attn_v_gemv_cuda(
    const torch::Tensor& attn_weights,
    const torch::Tensor& V,
    torch::Tensor& out,
    int n_heads, int n_kv_heads,
    int T, int head_dim) {

    c10::cuda::CUDAGuard device_guard(attn_weights.device());
    TORCH_CHECK(attn_weights.is_cuda() && V.is_cuda() && out.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(attn_weights.scalar_type() == at::kHalf && V.scalar_type() == at::kHalf, "Must be fp16");
    TORCH_CHECK(attn_weights.is_contiguous() && V.is_contiguous() && out.is_contiguous(), "Must be contiguous");

    const int B = static_cast<int>(attn_weights.size(0));
    const int total_q_heads = B * n_heads;

    constexpr int kThreads = 256;
    const dim3 grid(total_q_heads);
    const dim3 block(kThreads);

    gqa_decode_attn_v_gemv_fp16_kernel<kThreads><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(attn_weights.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        B, n_heads, n_kv_heads, T, head_dim
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
