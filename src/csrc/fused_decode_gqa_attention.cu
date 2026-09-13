// Fused decode GQA attention kernel for MiniMax-M2
// Combines cache_copy + transpose + GQA-repeat + attention in a single kernel
//
// Replaces the PyTorch sequence:
//   cache[start_pos:end_pos] = k_new/v_new
//   k/v = cache[:end_pos].transpose(1,2).repeat_interleave(repeat, dim=1)
//   out = F.scaled_dot_product_attention(q, k, v)
//
// With a single fused kernel that:
//   1. Updates cache with k_new/v_new
//   2. Computes attention directly from cache without repeat (GQA-aware)
//
// Parameters: n_heads=48, n_kv_heads=8, head_dim=128, repeat=6

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace {

__device__ __forceinline__ float safe_exp(float x) {
    // Clamp to avoid overflow/underflow
    return expf(fmaxf(fminf(x, 50.0f), -50.0f));
}

// Fused decode GQA attention kernel
// One block per Q head, threads cooperate over KV sequence
// q: [B, n_heads, 1, D] (already in [B, H, S, D] layout)
// k_new: [B, 1, n_kv_heads, D] (new K for current token, in [B, S, Hkv, D] layout)
// v_new: [B, 1, n_kv_heads, D]
// cache_k: [B, max_seq, n_kv_heads, D] (will be updated at start_pos)
// cache_v: [B, max_seq, n_kv_heads, D]
// out: [B, n_heads, 1, D]
template <int kThreads>
__global__ void fused_decode_gqa_attention_kernel(
        const __half* __restrict__ q,            // [B, n_heads, 1, D]
        const __half* __restrict__ k_new,        // [B, 1, n_kv_heads, D]
        const __half* __restrict__ v_new,        // [B, 1, n_kv_heads, D]
        __half* __restrict__ cache_k,            // [B, max_seq, n_kv_heads, D]
        __half* __restrict__ cache_v,            // [B, max_seq, n_kv_heads, D]
        __half* __restrict__ out,                // [B, n_heads, 1, D]
        const int B,
        const int n_heads,
        const int n_kv_heads,
        const int start_pos,
        const int max_seq,
        const int head_dim,
        const float sm_scale) {

    const int total_q_heads = B * n_heads;
    const int q_head_idx = blockIdx.x;
    if (q_head_idx >= total_q_heads) return;

    const int b = q_head_idx / n_heads;
    const int h = q_head_idx % n_heads;
    const int repeat_factor = n_heads / n_kv_heads;
    const int h_kv = h / repeat_factor;  // GQA mapping

    const int tid = threadIdx.x;
    const int kv_len = start_pos + 1;  // Total KV length after appending k_new/v_new

    // Pointers
    const __half* q_ptr = q + static_cast<int64_t>(q_head_idx) * head_dim;
    const __half* k_new_ptr = k_new + (static_cast<int64_t>(b) * n_kv_heads + h_kv) * head_dim;
    const __half* v_new_ptr = v_new + (static_cast<int64_t>(b) * n_kv_heads + h_kv) * head_dim;
    // cache layout: [B, max_seq, n_kv_heads, head_dim], stride=(max_seq*n_kv_heads*head_dim, n_kv_heads*head_dim, head_dim, 1)
    __half* cache_k_base = cache_k + (static_cast<int64_t>(b) * max_seq * n_kv_heads + h_kv) * head_dim;
    __half* cache_v_base = cache_v + (static_cast<int64_t>(b) * max_seq * n_kv_heads + h_kv) * head_dim;
    __half* out_ptr = out + static_cast<int64_t>(q_head_idx) * head_dim;

    // Shared memory for Q and attention scores
    extern __shared__ float smem[];
    float* q_shared = smem;                          // [head_dim]
    float* scores_shared = smem + head_dim;          // [kv_len], dynamically sized

    // Step 1: Update cache with k_new/v_new (only the first Q head of each KV head group)
    const bool should_update_cache = (h % repeat_factor == 0);
    if (should_update_cache) {
        for (int d = tid; d < head_dim; d += kThreads) {
            __half k_val = k_new_ptr[d];
            __half v_val = v_new_ptr[d];
            // cache layout: [B, max_seq, Hkv, D], so position t at offset t * n_kv_heads * head_dim + d
            cache_k_base[start_pos * n_kv_heads * head_dim + d] = k_val;
            cache_v_base[start_pos * n_kv_heads * head_dim + d] = v_val;
        }
    }
    // All threads load Q into shared mem
    for (int d = tid; d < head_dim; d += kThreads) {
        q_shared[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    // Step 2: Compute Q·K^T scores for all positions [0, kv_len)
    // All threads cooperate to fill scores_shared[0..kv_len-1] contiguously
    for (int t = tid; t < kv_len; t += kThreads) {
        const __half* k_ptr = cache_k_base + static_cast<int64_t>(t) * n_kv_heads * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_shared[d] * __half2float(k_ptr[d]);
        }
        scores_shared[t] = dot * sm_scale;
    }
    __syncthreads();

    // Step 3: Softmax over scores (numerically stable with max subtraction)
    // Find max
    float local_max = -INFINITY;
    for (int t = tid; t < kv_len; t += kThreads) {
        local_max = fmaxf(local_max, scores_shared[t]);
    }
    // Block-wide max reduction (assuming kThreads=256, 8 warps)
    __shared__ float warp_max[8];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    // Final reduce across warps
    float global_max = (tid < 8) ? warp_max[tid] : -INFINITY;
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            global_max = fmaxf(global_max, __shfl_xor_sync(0xFFFFFFFF, global_max, offset));
        }
    }
    // Broadcast
    if (tid == 0) warp_max[0] = global_max;
    __syncthreads();
    global_max = warp_max[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < kv_len; t += kThreads) {
        float e = safe_exp(scores_shared[t] - global_max);
        scores_shared[t] = e;
        local_sum += e;
    }
    // Block-wide sum reduction
    __shared__ float warp_sum[8];
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    float global_sum = (tid < 8) ? warp_sum[tid] : 0.0f;
    if (tid < 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            global_sum += __shfl_xor_sync(0xFFFFFFFF, global_sum, offset);
        }
    }
    if (tid == 0) warp_sum[0] = global_sum;
    __syncthreads();
    global_sum = warp_sum[0];

    // Normalize
    float inv_sum = (global_sum > 1e-20f) ? (1.0f / global_sum) : 0.0f;
    for (int t = tid; t < kv_len; t += kThreads) {
        scores_shared[t] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Compute attention·V
    for (int d = tid; d < head_dim; d += kThreads) {
        float sum = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            float weight = scores_shared[t];
            float v_val = __half2float(cache_v_base[static_cast<int64_t>(t) * n_kv_heads * head_dim + d]);
            sum += weight * v_val;
        }
        out_ptr[d] = __float2half(sum);
    }
}

}  // namespace

torch::Tensor fused_decode_gqa_attention_cuda(
        const torch::Tensor& q,           // [B, n_heads, 1, D]
        const torch::Tensor& k_new,       // [B, 1, n_kv_heads, D]
        const torch::Tensor& v_new,       // [B, 1, n_kv_heads, D]
        torch::Tensor& cache_k,           // [B, max_seq, n_kv_heads, D]
        torch::Tensor& cache_v,           // [B, max_seq, n_kv_heads, D]
        int start_pos,
        float sm_scale) {

    c10::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(q.is_cuda() && k_new.is_cuda() && v_new.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kHalf, "Only fp16 supported");
    TORCH_CHECK(q.dim() == 4 && k_new.dim() == 4, "Expected 4D tensors");
    TORCH_CHECK(q.size(2) == 1 && k_new.size(1) == 1, "Decode: seqlen must be 1");

    const int B = static_cast<int>(q.size(0));
    const int n_heads = static_cast<int>(q.size(1));
    const int head_dim = static_cast<int>(q.size(3));
    const int n_kv_heads = static_cast<int>(k_new.size(2));
    const int max_seq = static_cast<int>(cache_k.size(1));

    auto out = torch::empty_like(q);

    constexpr int kThreads = 256;
    const int kv_len = start_pos + 1;
    const size_t smem_bytes = sizeof(float) * (head_dim + kv_len);  // q_shared + scores_shared
    const dim3 grid(B * n_heads);
    const dim3 block(kThreads);

    fused_decode_gqa_attention_kernel<kThreads><<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(cache_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(cache_v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        B, n_heads, n_kv_heads, start_pos, max_seq, head_dim, sm_scale
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
