#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace dsv4 {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kHeadsPerGroup = 3;
constexpr int kQueryRows = 2;
constexpr int kMaxHeadDim = 256;
constexpr int kValuesPerThread = kMaxHeadDim / kThreads;
constexpr int kPrefillCombos = kHeadsPerGroup * kQueryRows;
constexpr int kDecodeMaxSplits = 64;
constexpr int kDecodeTargetPositions = 2048;
constexpr int kDecodeMinContext = 4096;

// A window of 0 keeps exact full attention. Otherwise every query attends to
// the leading sink prefix plus the most recent window positions, expressed as
// two logical ranges over the unmodified KV cache.
struct SparseRanges {
    int sink_count;
    int window_start;
    int window_count;

    __host__ __device__ int total() const { return sink_count + window_count; }

    __host__ __device__ int position(int logical) const {
        return logical < sink_count
            ? logical
            : window_start + (logical - sink_count);
    }
};

__host__ __device__ SparseRanges sparse_ranges(
    int context_len, int window, int sink) {
    SparseRanges ranges;
    if (window <= 0) {
        ranges.sink_count = 0;
        ranges.window_start = 0;
        ranges.window_count = context_len;
        return ranges;
    }
    ranges.sink_count = sink < context_len ? sink : context_len;
    const int tail = context_len - window;
    ranges.window_start = tail > ranges.sink_count ? tail : ranges.sink_count;
    ranges.window_count = context_len - ranges.window_start;
    return ranges;
}

__device__ __forceinline__ float half_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <int kCombos>
__device__ __forceinline__ void reduce_dot_products(
    const float (&partial)[kCombos], float (&warp_sums)[kCombos][kWarps],
    float (&scores)[kCombos], float scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const float sum = warp_sum(partial[combo]);
        if (lane == 0) warp_sums[combo][warp] = sum;
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < kCombos) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            sum += warp_sums[threadIdx.x][w];
        }
        scores[threadIdx.x] = sum * scale;
    }
    __syncthreads();
}

__host__ __device__ bool attends(
    int position, int context_limit, int window, int sink) {
    if (position >= context_limit) return false;
    if (window <= 0 || position < sink) return true;
    const int start = context_limit - window;
    return position >= (start > sink ? start : sink);
}

// Two adjacent query rows and three Q heads sharing a KV head are processed by
// one CTA. Each K/V element is loaded once for six attention outputs.
__global__ void gqa_prefill_tiled_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int window,
    int sink) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int first_token = static_cast<int>(blockIdx.y) * kQueryRows;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ float warp_sums[kPrefillCombos][kWarps];
    __shared__ float scores[kPrefillCombos];
    __shared__ float running_max[kPrefillCombos];
    __shared__ float running_sum[kPrefillCombos];
    __shared__ float rescale[kPrefillCombos];
    __shared__ float probability[kPrefillCombos];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kPrefillCombos][kValuesPerThread];
    float accumulator[kPrefillCombos][kValuesPerThread];
    bool active[kPrefillCombos];
    int context_limit[kPrefillCombos];

#pragma unroll
    for (int combo = 0; combo < kPrefillCombos; ++combo) {
        const int query_row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[combo] = token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + token + 1;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[combo][i] = active[combo] && d < head_dim
                ? half_to_float(q_rows[
                      (static_cast<size_t>(token) * q_heads + head) * head_dim + d])
                : 0.0f;
            accumulator[combo][i] = 0.0f;
        }
        if (tid == combo) {
            running_max[combo] = -INFINITY;
            running_sum[combo] = 0.0f;
        }
    }
    __syncthreads();

    const int last_token = min(first_token + kQueryRows - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    // The earliest query in this tile bounds the shared window; positions
    // between the sink prefix and that bound are attended by no query row.
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int position = 0; position < tile_context; ++position) {
        if (window > 0 && position >= sink && position < tile_window_start) {
            position = tile_window_start - 1;
            continue;
        }
        const size_t kv_base = static_cast<size_t>(position) * kv_stride +
            static_cast<size_t>(kv_head) * head_dim;
        float key[kValuesPerThread];
        float value[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
            value[i] = d < head_dim ? half_to_float(v_cache[kv_base + d]) : 0.0f;
        }

        float partial[kPrefillCombos] = {};
#pragma unroll
        for (int combo = 0; combo < kPrefillCombos; ++combo) {
            if (active[combo] &&
                attends(position, context_limit[combo], window, sink)) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    partial[combo] += query[combo][i] * key[i];
                }
            }
        }
        reduce_dot_products(partial, warp_sums, scores, attention_scale);

        if (tid < kPrefillCombos) {
            if (active[tid] &&
                attends(position, context_limit[tid], window, sink)) {
                const float old_max = running_max[tid];
                const float new_max = fmaxf(old_max, scores[tid]);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(scores[tid] - new_max);
                running_max[tid] = new_max;
                running_sum[tid] = running_sum[tid] * old_scale + weight;
                rescale[tid] = old_scale;
                probability[tid] = weight;
            } else {
                rescale[tid] = 1.0f;
                probability[tid] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int combo = 0; combo < kPrefillCombos; ++combo) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[combo][i] = accumulator[combo][i] * rescale[combo] +
                    probability[combo] * value[i];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int combo = 0; combo < kPrefillCombos; ++combo) {
        if (!active[combo]) continue;
        const int query_row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        const float inverse = running_sum[combo] > 0.0f
            ? 1.0f / running_sum[combo] : 0.0f;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) {
                output[(static_cast<size_t>(token) * q_heads + head) * head_dim + d] =
                    float_to_half(accumulator[combo][i] * inverse);
            }
        }
    }
}

// Each CTA computes exact online-softmax partials for three Q heads over one
// context split while loading their shared K/V row only once.
__global__ void gqa_decode_split_f16_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partial_output,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len,
    int splits,
    int positions_per_split,
    int window,
    int sink) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int grouped = static_cast<int>(blockIdx.x) / splits;
    const int split = static_cast<int>(blockIdx.x) % splits;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const SparseRanges ranges = sparse_ranges(context_len, window, sink);
    const int start = split * positions_per_split;
    const int end = min(ranges.total(), start + positions_per_split);
    if (kv_head >= kv_heads || start >= end) return;

    __shared__ float warp_sums[kHeadsPerGroup][kWarps];
    __shared__ float scores[kHeadsPerGroup];
    __shared__ float running_max[kHeadsPerGroup];
    __shared__ float running_sum[kHeadsPerGroup];
    __shared__ float rescale[kHeadsPerGroup];
    __shared__ float probability[kHeadsPerGroup];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kHeadsPerGroup][kValuesPerThread];
    float accumulator[kHeadsPerGroup][kValuesPerThread] = {};
    bool active[kHeadsPerGroup];
#pragma unroll
    for (int h = 0; h < kHeadsPerGroup; ++h) {
        const int head = first_head + h;
        active[h] = head < (kv_head + 1) * q_per_kv && head < q_heads;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[h][i] = active[h] && d < head_dim
                ? half_to_float(q[static_cast<size_t>(head) * head_dim + d])
                : 0.0f;
        }
        if (tid == h) {
            running_max[h] = -INFINITY;
            running_sum[h] = 0.0f;
        }
    }
    __syncthreads();

    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int logical = start; logical < end; ++logical) {
        const int position = ranges.position(logical);
        const size_t kv_base = static_cast<size_t>(position) * kv_stride +
            static_cast<size_t>(kv_head) * head_dim;
        float key[kValuesPerThread];
        float value[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
            value[i] = d < head_dim ? half_to_float(v_cache[kv_base + d]) : 0.0f;
        }
        float dot[kHeadsPerGroup] = {};
#pragma unroll
        for (int h = 0; h < kHeadsPerGroup; ++h) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                dot[h] += query[h][i] * key[i];
            }
        }
        reduce_dot_products(dot, warp_sums, scores, attention_scale);
        if (tid < kHeadsPerGroup) {
            if (active[tid]) {
                const float old_max = running_max[tid];
                const float new_max = fmaxf(old_max, scores[tid]);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(scores[tid] - new_max);
                running_max[tid] = new_max;
                running_sum[tid] = running_sum[tid] * old_scale + weight;
                rescale[tid] = old_scale;
                probability[tid] = weight;
            } else {
                rescale[tid] = 1.0f;
                probability[tid] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kHeadsPerGroup; ++h) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[h][i] = accumulator[h][i] * rescale[h] +
                    probability[h] * value[i];
            }
        }
        __syncthreads();
    }

    const int partial_stride = head_dim + 2;
#pragma unroll
    for (int h = 0; h < kHeadsPerGroup; ++h) {
        if (!active[h]) continue;
        const int head = first_head + h;
        float* destination = partial_output +
            (static_cast<size_t>(head) * splits + split) * partial_stride;
        if (tid == 0) {
            destination[0] = running_max[h];
            destination[1] = running_sum[h];
        }
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) destination[2 + d] = accumulator[h][i];
        }
    }
}

__global__ void gqa_decode_merge_f16_kernel(
    const float* __restrict__ partial_output,
    uint16_t* __restrict__ output,
    int q_heads,
    int head_dim,
    int splits) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= q_heads) return;
    __shared__ float weights[kDecodeMaxSplits];
    if (threadIdx.x == 0) {
        const int stride = head_dim + 2;
        float maximum = -INFINITY;
        for (int split = 0; split < splits; ++split) {
            maximum = fmaxf(maximum,
                partial_output[(static_cast<size_t>(head) * splits + split) * stride]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output +
                (static_cast<size_t>(head) * splits + split) * stride;
            weights[split] = expf(source[0] - maximum);
            denominator += weights[split] * source[1];
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int split = 0; split < splits; ++split) weights[split] *= inverse;
    }
    __syncthreads();
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += kThreads) {
        float value = 0.0f;
        const int stride = head_dim + 2;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output +
                (static_cast<size_t>(head) * splits + split) * stride;
            value += weights[split] * source[2 + d];
        }
        output[static_cast<size_t>(head) * head_dim + d] = float_to_half(value);
    }
}

bool valid_shape(int q_heads, int kv_heads, int head_dim) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
        head_dim > 0 && head_dim <= kMaxHeadDim;
}

}  // namespace

bool qwen_gqa_prefill_attention_f16_tiled_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, int seq_len,
    int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, int attention_window, int sink_tokens, void* stream) {
    if (!q || !k_cache || !v_cache || !output || seq_len <= 0 ||
        position_offset < 0 || position_offset + seq_len > max_context ||
        attention_window < 0 || sink_tokens < 0 ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const dim3 grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                    static_cast<unsigned>((seq_len + kQueryRows - 1) / kQueryRows), 1);
    gqa_prefill_tiled_f16_kernel<<<grid, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(q, k_cache, v_cache, output,
        seq_len, q_heads, kv_heads, head_dim, position_offset,
        attention_window, sink_tokens);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_f16_fused_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int attention_window, int sink_tokens, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !partial_scratch ||
        context_len > max_context ||
        attention_window < 0 || sink_tokens < 0 ||
        (attention_window <= 0 && context_len < kDecodeMinContext) ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const SparseRanges ranges =
        sparse_ranges(context_len, attention_window, sink_tokens);
    const int attended = ranges.total();
    int splits = std::min(kDecodeMaxSplits,
        (attended + kDecodeTargetPositions - 1) / kDecodeTargetPositions);
    splits = std::max(splits, 1);
    const int positions_per_split = (attended + splits - 1) / splits;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int blocks = kv_heads * groups_per_kv * splits;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    gqa_decode_split_f16_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
        q, k_cache, v_cache, partial_scratch, q_heads, kv_heads, head_dim,
        context_len, splits, positions_per_split, attention_window,
        sink_tokens);
    if (cudaGetLastError() != cudaSuccess) return false;
    gqa_decode_merge_f16_kernel<<<q_heads, kThreads, 0, cuda_stream>>>(
        partial_scratch, output, q_heads, head_dim, splits);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dsv4
