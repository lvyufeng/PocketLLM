#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
constexpr int kVerifyMaxRows = 8;

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

// Butterfly reduction: every lane ends with the full warp total, so a warp that
// owns one dot product needs no lane-0 broadcast afterwards.
__device__ __forceinline__ float warp_sum_all(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
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

// Batched-position variant of the tiled prefill kernel below. The per-position
// kernel spends four block-wide __syncthreads() on every history element, so an
// 8192-token prefill pays roughly half a million barriers per layer and becomes
// synchronisation bound rather than bandwidth bound. This scores kPosTile
// positions per barrier round: the dot products for the whole sub-tile are
// reduced together, then the online-softmax update walks the sub-tile in
// position order exactly as before. The running max/sum recurrence and the
// accumulator update order are therefore identical to the per-position kernel,
// so the result is bit-identical.
constexpr int kPosTile = 8;

template <int kHPG, int kVPT, int kQR>
__global__ void gqa_prefill_tiled_batched_f16_kernel(
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
    constexpr int kCombosT = kHPG * kQR;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHPG - 1) / kHPG;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ float warp_sums[kPosTile][kCombosT][kWarps];
    __shared__ float scores[kPosTile][kCombosT];
    __shared__ float rescale[kPosTile][kCombosT];
    __shared__ float probability[kPosTile][kCombosT];
    __shared__ float running_max[kCombosT];
    __shared__ float running_sum[kCombosT];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float query[kCombosT][kVPT];
    float accumulator[kCombosT][kVPT];
    bool active[kCombosT];
    int context_limit[kCombosT];

#pragma unroll
    for (int combo = 0; combo < kCombosT; ++combo) {
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[combo] = token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + token + 1;
#pragma unroll
        for (int i = 0; i < kVPT; ++i) {
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

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;

    for (int base = 0; base < tile_context; base += kPosTile) {
        // Skip whole sub-tiles that no query row in this CTA attends. The
        // per-position kernel jumps the same gap one element at a time.
        if (window > 0) {
            const int last = min(base + kPosTile, tile_context) - 1;
            if (last >= sink && last < tile_window_start) continue;
        }
        const int count = min(kPosTile, tile_context - base);
        float value[kPosTile][kVPT];
#pragma unroll
        for (int slot = 0; slot < kPosTile; ++slot) {
            if (slot >= count) break;
            const int position = base + slot;
            const bool skipped = window > 0 && position >= sink &&
                position < tile_window_start;
            const size_t kv_base = static_cast<size_t>(position) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim;
            float key[kVPT];
#pragma unroll
            for (int i = 0; i < kVPT; ++i) {
                const int d = tid + i * kThreads;
                const bool load = !skipped && d < head_dim;
                key[i] = load ? half_to_float(k_cache[kv_base + d]) : 0.0f;
                value[slot][i] = load ? half_to_float(v_cache[kv_base + d]) : 0.0f;
            }
#pragma unroll
            for (int combo = 0; combo < kCombosT; ++combo) {
                float partial = 0.0f;
                if (!skipped && active[combo] &&
                    attends(position, context_limit[combo], window, sink)) {
#pragma unroll
                    for (int i = 0; i < kVPT; ++i) {
                        partial += query[combo][i] * key[i];
                    }
                }
                const float sum = warp_sum(partial);
                if (lane == 0) warp_sums[slot][combo][warp] = sum;
            }
        }
        __syncthreads();
        // One thread per (slot, combo) folds the warp partials, then the online
        // softmax recurrence is replayed in position order by thread `combo`.
        if (tid < kCombosT) {
            const int combo = tid;
            for (int slot = 0; slot < count; ++slot) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < kWarps; ++w) sum += warp_sums[slot][combo][w];
                scores[slot][combo] = sum * attention_scale;
            }
            for (int slot = 0; slot < count; ++slot) {
                const int position = base + slot;
                const bool skipped = window > 0 && position >= sink &&
                    position < tile_window_start;
                if (!skipped && active[combo] &&
                    attends(position, context_limit[combo], window, sink)) {
                    const float old_max = running_max[combo];
                    const float new_max = fmaxf(old_max, scores[slot][combo]);
                    const float old_scale = old_max == -INFINITY
                        ? 0.0f : expf(old_max - new_max);
                    const float weight = expf(scores[slot][combo] - new_max);
                    running_max[combo] = new_max;
                    running_sum[combo] = running_sum[combo] * old_scale + weight;
                    rescale[slot][combo] = old_scale;
                    probability[slot][combo] = weight;
                } else {
                    rescale[slot][combo] = 1.0f;
                    probability[slot][combo] = 0.0f;
                }
            }
        }
        __syncthreads();
        // Replay the accumulator update in the same position order.
        for (int slot = 0; slot < count; ++slot) {
#pragma unroll
            for (int combo = 0; combo < kCombosT; ++combo) {
#pragma unroll
                for (int i = 0; i < kVPT; ++i) {
                    accumulator[combo][i] =
                        accumulator[combo][i] * rescale[slot][combo] +
                        probability[slot][combo] * value[slot][i];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int combo = 0; combo < kCombosT; ++combo) {
        if (!active[combo]) continue;
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        const float inverse = running_sum[combo] > 0.0f
            ? 1.0f / running_sum[combo] : 0.0f;
#pragma unroll
        for (int i = 0; i < kVPT; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) {
                output[(static_cast<size_t>(token) * q_heads + head) * head_dim + d] =
                    float_to_half(accumulator[combo][i] * inverse);
            }
        }
    }
}

// Warp-per-combo variant. The kernels above spread each dot product across all
// 128 threads, so with head_dim 128 every thread contributes a single FMA and
// then pays a five-shuffle reduction plus block barriers: measured 27.6 GB/s and
// 441 GFLOP/s on SM75, roughly 3% of both roofs, so the cost is reduction
// instructions rather than data. Here each warp owns a slice of the (query row,
// head) combos and each lane holds kDPL contiguous head dimensions, so a dot
// product is one warp butterfly with no block barrier and the online-softmax
// state stays in registers. K/V for the position tile is staged once in shared
// memory so the four warps do not re-read it. The reduction tree changes, so
// results differ from the per-position kernel in the last FP32 bits.
template <int kHPG, int kQR, int kDPL>
__global__ void gqa_prefill_warp_combo_f16_kernel(
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
    constexpr int kCombosT = kHPG * kQR;
    constexpr int kCPW = (kCombosT + kWarps - 1) / kWarps;
    constexpr int kDim = kDPL * 32;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv = (q_per_kv + kHPG - 1) / kHPG;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ uint16_t ks[kPosTile][kDim];
    __shared__ uint16_t vs[kPosTile][kDim];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane_base = lane * kDPL;

    float query[kCPW][kDPL];
    float accumulator[kCPW][kDPL];
    float running_max[kCPW];
    float running_sum[kCPW];
    bool active[kCPW];
    int context_limit[kCPW];

#pragma unroll
    for (int c = 0; c < kCPW; ++c) {
        const int combo = warp * kCPW + c;
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[c] = combo < kCombosT && token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[c] = position_offset + token + 1;
        running_max[c] = -INFINITY;
        running_sum[c] = 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            query[c][i] = active[c]
                ? half_to_float(q_rows[
                      (static_cast<size_t>(token) * q_heads + head) * head_dim +
                      lane_base + i])
                : 0.0f;
            accumulator[c][i] = 0.0f;
        }
    }

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;

    for (int base = 0; base < tile_context; base += kPosTile) {
        if (window > 0) {
            const int last = min(base + kPosTile, tile_context) - 1;
            if (last >= sink && last < tile_window_start) continue;
        }
        const int count = min(kPosTile, tile_context - base);
        __syncthreads();
        for (int idx = tid; idx < count * kDim; idx += kThreads) {
            const int slot = idx / kDim;
            const int d = idx - slot * kDim;
            const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim + d;
            ks[slot][d] = k_cache[at];
            vs[slot][d] = v_cache[at];
        }
        __syncthreads();
        for (int slot = 0; slot < count; ++slot) {
            const int position = base + slot;
            const bool skipped = window > 0 && position >= sink &&
                position < tile_window_start;
            if (skipped) continue;
            float key[kDPL];
            float value[kDPL];
#pragma unroll
            for (int i = 0; i < kDPL; ++i) {
                key[i] = half_to_float(ks[slot][lane_base + i]);
                value[i] = half_to_float(vs[slot][lane_base + i]);
            }
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const bool contributes = active[c] &&
                    attends(position, context_limit[c], window, sink);
                float partial = 0.0f;
                if (contributes) {
#pragma unroll
                    for (int i = 0; i < kDPL; ++i) {
                        partial += query[c][i] * key[i];
                    }
                }
                // Butterfly reduction leaves the full sum in every lane, so no
                // broadcast is needed before the softmax update.
                const float score = warp_sum_all(partial) * attention_scale;
                if (!contributes) continue;
                const float old_max = running_max[c];
                const float new_max = fmaxf(old_max, score);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(score - new_max);
                running_max[c] = new_max;
                running_sum[c] = running_sum[c] * old_scale + weight;
#pragma unroll
                for (int i = 0; i < kDPL; ++i) {
                    accumulator[c][i] =
                        accumulator[c][i] * old_scale + weight * value[i];
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < kCPW; ++c) {
        if (!active[c]) continue;
        const int combo = warp * kCPW + c;
        const int token = first_token + combo / kHPG;
        const int head = first_head + combo % kHPG;
        const float inverse = running_sum[c] > 0.0f
            ? 1.0f / running_sum[c] : 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            output[(static_cast<size_t>(token) * q_heads + head) * head_dim +
                   lane_base + i] = float_to_half(accumulator[c][i] * inverse);
        }
    }
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

// Speculative verification scans one context split per CTA. A CTA owns one
// KV head, one group of three Q heads, and every query row, so each K/V element
// is loaded once for up to rows * 3 outputs. Splitting the history restores
// enough parallelism for the 4-32K context regime where a single CTA per group
// leaves SM75 mostly idle.
template <int kRows>
__global__ void gqa_verify_split_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partial_output,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int splits,
    int positions_per_split) {
    constexpr int kCombos = kRows * kHeadsPerGroup;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int grouped = static_cast<int>(blockIdx.x) / splits;
    const int split = static_cast<int>(blockIdx.x) % splits;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int split_start = split * positions_per_split;
    const int split_end = min(
        position_offset + rows, split_start + positions_per_split);
    if (kv_head >= kv_heads || split_start >= split_end) return;

    __shared__ float warp_sums[kCombos][kWarps];
    __shared__ float scores[kCombos];
    __shared__ float running_max[kCombos];
    __shared__ float running_sum[kCombos];
    __shared__ float rescale[kCombos];
    __shared__ float probability[kCombos];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kCombos][kValuesPerThread];
    float accumulator[kCombos][kValuesPerThread];
    bool active[kCombos];
    int context_limit[kCombos];
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const int row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int head = first_head + head_in_group;
        active[combo] = row < rows &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + row + 1;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[combo][i] = active[combo] && d < head_dim
                ? half_to_float(q_rows[
                      (static_cast<size_t>(row) * q_heads + head) * head_dim + d])
                : 0.0f;
            accumulator[combo][i] = 0.0f;
        }
        if (tid == combo) {
            running_max[combo] = -INFINITY;
            running_sum[combo] = 0.0f;
        }
    }
    __syncthreads();

    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int position = split_start; position < split_end; ++position) {
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
        float dot[kCombos] = {};
#pragma unroll
        for (int combo = 0; combo < kCombos; ++combo) {
            if (active[combo] && position < context_limit[combo]) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    dot[combo] += query[combo][i] * key[i];
                }
            }
        }
        reduce_dot_products(dot, warp_sums, scores, attention_scale);
        if (tid < kCombos) {
            if (active[tid] && position < context_limit[tid]) {
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
        for (int combo = 0; combo < kCombos; ++combo) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[combo][i] = accumulator[combo][i] * rescale[combo] +
                    probability[combo] * value[i];
            }
        }
        __syncthreads();
    }

    const int partial_stride = head_dim + 2;
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        if (!active[combo]) continue;
        const int row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int head = first_head + head_in_group;
        float* destination = partial_output +
            ((static_cast<size_t>(row) * q_heads + head) * splits + split) *
                partial_stride;
        if (tid == 0) {
            destination[0] = running_max[combo];
            destination[1] = running_sum[combo];
        }
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) destination[2 + d] = accumulator[combo][i];
        }
    }
}

__global__ void gqa_verify_merge_f16_kernel(
    const float* __restrict__ partial_output,
    uint16_t* __restrict__ output,
    int rows,
    int q_heads,
    int head_dim,
    int splits) {
    const int row = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.x);
    if (row >= rows || head >= q_heads) return;
    __shared__ float weights[kDecodeMaxSplits];
    if (threadIdx.x == 0) {
        const int stride = head_dim + 2;
        const size_t base =
            (static_cast<size_t>(row) * q_heads + head) * splits * stride;
        float maximum = -INFINITY;
        for (int split = 0; split < splits; ++split) {
            maximum = fmaxf(maximum,
                partial_output[base + static_cast<size_t>(split) * stride]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output + base +
                static_cast<size_t>(split) * stride;
            weights[split] = expf(source[0] - maximum);
            denominator += weights[split] * source[1];
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int split = 0; split < splits; ++split) weights[split] *= inverse;
    }
    __syncthreads();
    const int stride = head_dim + 2;
    const size_t base =
        (static_cast<size_t>(row) * q_heads + head) * splits * stride;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += kThreads) {
        float value = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output + base +
                static_cast<size_t>(split) * stride;
            value += weights[split] * source[2 + d];
        }
        output[(static_cast<size_t>(row) * q_heads + head) * head_dim + d] =
            float_to_half(value);
    }
}

template <int kRows>
__global__ void gqa_verify_scores_exact_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    float* __restrict__ scores,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int context_len) {
    constexpr int kCombos = kRows * kHeadsPerGroup;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int grouped = static_cast<int>(blockIdx.x) / context_len;
    const int position = static_cast<int>(blockIdx.x) % context_len;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    if (kv_head >= kv_heads) return;

    __shared__ float warp_sums[kCombos][kWarps];
    __shared__ float reduced[kCombos];
    const int tid = static_cast<int>(threadIdx.x);
    const size_t kv_base =
        (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
    float key[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int d = tid + i * kThreads;
        key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
    }
    float partial[kCombos] = {};
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const int row = combo / kHeadsPerGroup;
        const int head = first_head + combo % kHeadsPerGroup;
        if (row >= rows || head >= (kv_head + 1) * q_per_kv ||
            head >= q_heads || position >= position_offset + row + 1) continue;
        const uint16_t* query = q_rows +
            (static_cast<size_t>(row) * q_heads + head) * head_dim;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) partial[combo] += half_to_float(query[d]) * key[i];
        }
    }
    reduce_dot_products(partial, warp_sums, reduced,
                        rsqrtf(static_cast<float>(head_dim)));
    if (tid < kCombos) {
        const int row = tid / kHeadsPerGroup;
        const int head = first_head + tid % kHeadsPerGroup;
        if (row < rows && head < (kv_head + 1) * q_per_kv && head < q_heads) {
            scores[(static_cast<size_t>(row) * q_heads + head) * context_len +
                   position] = position < position_offset + row + 1
                ? reduced[tid] : -INFINITY;
        }
    }
}

__global__ void gqa_verify_softmax_exact_f16_kernel(
    float* __restrict__ scores, int rows, int q_heads, int context_len) {
    const int row = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.x);
    if (row >= rows || head >= q_heads) return;
    float* line = scores +
        (static_cast<size_t>(row) * q_heads + head) * context_len;
    __shared__ float reduce[kThreads];
    float local_max = -INFINITY;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        local_max = fmaxf(local_max, line[position]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduce[0];
    float sum = 0.0f;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        line[position] = expf(line[position] - maximum);
        sum += line[position];
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        line[position] *= inverse;
    }
}

// A warp owns one candidate row, three Q heads sharing a KV head, and a
// contiguous 32-channel value tile. Splitting rows and channels into separate
// CTAs preserves each output element's left-to-right FP32 accumulation order,
// while exposing enough independent work to occupy all SMs. The previous kernel
// kept every candidate row in one CTA; for Qwen TP4 that launched only four
// long-running CTAs per layer at head_dim=256.
__global__ void gqa_verify_values_exact_f16_kernel(
    const float* __restrict__ probabilities,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int row = static_cast<int>(blockIdx.y);
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int d = static_cast<int>(blockIdx.z) * 32 + threadIdx.x;
    if (kv_head >= kv_heads || row >= rows || d >= head_dim) return;

    float accumulators[kHeadsPerGroup] = {};
    for (int position = 0; position < context_len; ++position) {
        const float value = half_to_float(v_cache[
            (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim + d]);
#pragma unroll
        for (int i = 0; i < kHeadsPerGroup; ++i) {
            const int head = first_head + i;
            if (head < (kv_head + 1) * q_per_kv && head < q_heads) {
                accumulators[i] += probabilities[
                    (static_cast<size_t>(row) * q_heads + head) * context_len +
                    position] * value;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerGroup; ++i) {
        const int head = first_head + i;
        if (head < (kv_head + 1) * q_per_kv && head < q_heads) {
            output[(static_cast<size_t>(row) * q_heads + head) * head_dim + d] =
                float_to_half(accumulators[i]);
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
    // The batched-position kernel amortises the per-history-element barriers
    // over kPosTile positions while preserving the online-softmax order, so it
    // is bit-identical. Set DSV4_QWEN_GQA_POS_TILE=0 to fall back.
    const char* pos_tile = std::getenv("DSV4_QWEN_GQA_POS_TILE");
    const bool use_pos_tile = pos_tile == nullptr ||
        std::strcmp(pos_tile, "0") != 0;
    if (use_pos_tile) {
        // Choose the head grouping that divides q_per_kv exactly. With the fixed
        // group of three and the real q_per_kv of four, the second group carried a
        // single head yet still streamed the whole KV history for its KV head, so
        // each history element was read twice per layer. A group of four covers
        // all four heads in one CTA and halves that traffic. Shapes that do not
        // divide evenly keep the original grouping.
        const int q_per_kv = q_heads / kv_heads;
        int hpg = kHeadsPerGroup;
        // Wider groups cut history traffic but cost registers per CTA. Measured at
        // the real TP4 shape (6 q heads over 1 kv head, head_dim 256, 4096 rows),
        // groups of six spill and lose: 67.7/215.4 ms against 54.9/186.8 for two.
        if (q_per_kv % 4 == 0) hpg = 4;
        else if (q_per_kv % 2 == 0) hpg = 2;
        else if (q_per_kv == 1) hpg = 1;
        if (const char* hpg_env = std::getenv("DSV4_QWEN_GQA_HEADS_PER_GROUP")) {
            const int requested = std::atoi(hpg_env);
            if (requested >= 1 && requested <= 6 && q_per_kv % requested == 0) {
                hpg = requested;
            }
        }
        const int batched_groups = (q_per_kv + hpg - 1) / hpg;
        // Every CTA streams the whole KV history for its KV head, so the number of
        // query-row tiles is a direct multiplier on history traffic. At 8192 with
        // two rows per CTA the 16 GQA layers re-read 544 GiB per rank and run at
        // ~118 GB/s of the ~600 GB/s the card can sustain. Widening the tile
        // divides that traffic; the accumulator lives in registers, so the width
        // is capped to keep occupancy.
        int qr = 2;
        const char* qr_env = std::getenv("DSV4_QWEN_GQA_QUERY_ROWS");
        if (qr_env != nullptr) {
            const int requested = std::atoi(qr_env);
            if (requested == 2 || requested == 4 || requested == 8) qr = requested;
        } else if (hpg <= 2) {
            // Measured on the real 8192 TP4 prefill: full_attention 4.96 s at two
            // rows, 4.51 s at four, 6.60 s at eight. Past four rows the register
            // accumulator spills and the traffic saving is more than undone. At
            // head_dim 256 the accumulator is twice as wide, so four rows only pay
            // off for the narrow groups: 53.7/178.5 ms against 54.9/186.8 at two
            // rows for hpg=2, while hpg=3 regresses to 69.7/214.3.
            qr = 4;
        }
        const dim3 batched_grid(
            static_cast<unsigned>(kv_heads * batched_groups),
            static_cast<unsigned>((seq_len + qr - 1) / qr), 1);
        // Warp-per-combo variant. Opt-in: it reassociates the dot-product
        // reduction, so the last FP32 bits differ from the per-position kernel and
        // a near-tie greedy argmax could flip. Requires head_dim to split evenly
        // across a warp and the combo count to cover the warps.
        const char* warp_combo = std::getenv("DSV4_QWEN_GQA_WARP_COMBO");
        if (warp_combo != nullptr && std::strcmp(warp_combo, "0") != 0 &&
            (head_dim == 128 || head_dim == 256) && attention_window <= 0) {
            const int combos = hpg * qr;
            const int dpl = head_dim / 32;
            if (combos % kWarps == 0) {
#define DSV4_LAUNCH_WARP_COMBO_D(HPG, QR, DPL) \
                gqa_prefill_warp_combo_f16_kernel<HPG, QR, DPL> \
                    <<<batched_grid, kThreads, 0, \
                       static_cast<cudaStream_t>(stream)>>>( \
                        q, k_cache, v_cache, output, seq_len, q_heads, kv_heads, \
                        head_dim, position_offset, attention_window, sink_tokens)
#define DSV4_LAUNCH_WARP_COMBO(HPG, QR) \
                do { \
                    if (dpl == 8) { DSV4_LAUNCH_WARP_COMBO_D(HPG, QR, 8); } \
                    else { DSV4_LAUNCH_WARP_COMBO_D(HPG, QR, 4); } \
                } while (0)
                bool launched = true;
                if (hpg == 4 && qr == 4) { DSV4_LAUNCH_WARP_COMBO(4, 4); }
                else if (hpg == 4 && qr == 2) { DSV4_LAUNCH_WARP_COMBO(4, 2); }
                else if (hpg == 4 && qr == 8) { DSV4_LAUNCH_WARP_COMBO(4, 8); }
                else if (hpg == 2 && qr == 2) { DSV4_LAUNCH_WARP_COMBO(2, 2); }
                else if (hpg == 2 && qr == 4) { DSV4_LAUNCH_WARP_COMBO(2, 4); }
                else if (hpg == 2 && qr == 8) { DSV4_LAUNCH_WARP_COMBO(2, 8); }
                else if (hpg == 1 && qr == 4) { DSV4_LAUNCH_WARP_COMBO(1, 4); }
                else if (hpg == 1 && qr == 8) { DSV4_LAUNCH_WARP_COMBO(1, 8); }
                else { launched = false; }
#undef DSV4_LAUNCH_WARP_COMBO
#undef DSV4_LAUNCH_WARP_COMBO_D
                if (launched) return cudaGetLastError() == cudaSuccess;
            }
        }
        // kValuesPerThread is sized for the largest supported head_dim. The real
        // GQA head_dim is 128, which leaves half of every register array and half
        // of each inner loop doing nothing, so specialise on the exact count.
        const int vpt = (head_dim + kThreads - 1) / kThreads;
#define DSV4_LAUNCH_POS_TILE(HPG, VPT, QR) \
        gqa_prefill_tiled_batched_f16_kernel<HPG, VPT, QR> \
            <<<batched_grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>( \
                q, k_cache, v_cache, output, seq_len, q_heads, kv_heads, \
                head_dim, position_offset, attention_window, sink_tokens)
#define DSV4_LAUNCH_POS_TILE_QR(HPG, VPT) \
        do { \
            if (qr == 8) { DSV4_LAUNCH_POS_TILE(HPG, VPT, 8); } \
            else if (qr == 4) { DSV4_LAUNCH_POS_TILE(HPG, VPT, 4); } \
            else { DSV4_LAUNCH_POS_TILE(HPG, VPT, 2); } \
        } while (0)
#define DSV4_LAUNCH_POS_TILE_HPG(HPG) \
        do { \
            if (vpt == 1) { DSV4_LAUNCH_POS_TILE_QR(HPG, 1); } \
            else { DSV4_LAUNCH_POS_TILE_QR(HPG, kValuesPerThread); } \
        } while (0)
        if (hpg == 6) { DSV4_LAUNCH_POS_TILE_HPG(6); }
        else if (hpg == 4) { DSV4_LAUNCH_POS_TILE_HPG(4); }
        else if (hpg == 2) { DSV4_LAUNCH_POS_TILE_HPG(2); }
        else if (hpg == 1) { DSV4_LAUNCH_POS_TILE_HPG(1); }
        else { DSV4_LAUNCH_POS_TILE_HPG(3); }
#undef DSV4_LAUNCH_POS_TILE_HPG
#undef DSV4_LAUNCH_POS_TILE_QR
#undef DSV4_LAUNCH_POS_TILE
        return cudaGetLastError() == cudaSuccess;
    }
    gqa_prefill_tiled_f16_kernel<<<grid, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(q, k_cache, v_cache, output,
        seq_len, q_heads, kv_heads, head_dim, position_offset,
        attention_window, sink_tokens);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_verify_attention_f16_exact_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* score_scratch,
    int rows, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !score_scratch ||
        rows < 2 || rows > kVerifyMaxRows || position_offset < 0 ||
        position_offset + rows > max_context ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int context_len = position_offset + rows;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const uint64_t score_blocks =
        static_cast<uint64_t>(kv_heads) * groups_per_kv * context_len;
    if (score_blocks > static_cast<uint64_t>(UINT32_MAX)) return false;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#define DSV4_LAUNCH_EXACT_SCORES(ROWS) \
    gqa_verify_scores_exact_f16_kernel<ROWS> \
        <<<static_cast<unsigned>(score_blocks), kThreads, 0, cuda_stream>>>( \
            q, k_cache, score_scratch, rows, q_heads, kv_heads, head_dim, \
            position_offset, context_len)
    switch (rows) {
        case 2: DSV4_LAUNCH_EXACT_SCORES(2); break;
        case 3: DSV4_LAUNCH_EXACT_SCORES(3); break;
        case 4: DSV4_LAUNCH_EXACT_SCORES(4); break;
        case 5: DSV4_LAUNCH_EXACT_SCORES(5); break;
        case 6: DSV4_LAUNCH_EXACT_SCORES(6); break;
        case 7: DSV4_LAUNCH_EXACT_SCORES(7); break;
        case 8: DSV4_LAUNCH_EXACT_SCORES(8); break;
        default: return false;
    }
#undef DSV4_LAUNCH_EXACT_SCORES
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 softmax_grid(static_cast<unsigned>(q_heads),
                            static_cast<unsigned>(rows), 1);
    gqa_verify_softmax_exact_f16_kernel<<<softmax_grid, kThreads, 0, cuda_stream>>>(
        score_scratch, rows, q_heads, context_len);
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 value_grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                          static_cast<unsigned>(rows),
                          static_cast<unsigned>((head_dim + 31) / 32));
    gqa_verify_values_exact_f16_kernel<<<value_grid, 32, 0, cuda_stream>>>(
        score_scratch, v_cache, output, rows, q_heads, kv_heads, head_dim,
        context_len);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_verify_attention_f16_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int rows, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, int splits, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !partial_scratch ||
        rows < 2 || rows > kVerifyMaxRows || position_offset < 0 ||
        position_offset + rows > max_context || splits <= 0 ||
        splits > kDecodeMaxSplits ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int context_len = position_offset + rows;
    const int positions_per_split = (context_len + splits - 1) / splits;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int blocks = kv_heads * groups_per_kv * splits;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#define DSV4_LAUNCH_VERIFY(ROWS) \
    gqa_verify_split_f16_kernel<ROWS><<<blocks, kThreads, 0, cuda_stream>>>( \
        q, k_cache, v_cache, partial_scratch, rows, q_heads, kv_heads, \
        head_dim, position_offset, splits, positions_per_split)
    switch (rows) {
        case 2: DSV4_LAUNCH_VERIFY(2); break;
        case 3: DSV4_LAUNCH_VERIFY(3); break;
        case 4: DSV4_LAUNCH_VERIFY(4); break;
        case 5: DSV4_LAUNCH_VERIFY(5); break;
        case 6: DSV4_LAUNCH_VERIFY(6); break;
        case 7: DSV4_LAUNCH_VERIFY(7); break;
        case 8: DSV4_LAUNCH_VERIFY(8); break;
        default: return false;
    }
#undef DSV4_LAUNCH_VERIFY
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 merge_grid(static_cast<unsigned>(q_heads),
                          static_cast<unsigned>(rows), 1);
    gqa_verify_merge_f16_kernel<<<merge_grid, kThreads, 0, cuda_stream>>>(
        partial_scratch, output, rows, q_heads, head_dim, splits);
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
