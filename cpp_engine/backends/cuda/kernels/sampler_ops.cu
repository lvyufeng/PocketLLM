#include "sampler_ops.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace pocket {

namespace {

// One block per row. A full-vocab probability array does not fit in shared
// memory (62,080 local vocab = 242 KB against a 48 KB limit), so the kernel
// keeps only the top-k candidates per row. This is not an approximation of the
// requested distribution: top-k runs before top-p in the standard sampling
// order, so any token top-p could keep is already inside the top-k set.
constexpr int kMaxTopK = 64;
constexpr int kBlockThreads = 256;

struct Candidate {
    float logit;
    int index;
};

// Descending by logit, with the lower vocab index winning ties so every rank
// and every run breaks ties identically.
__device__ inline bool better(const Candidate& a, const Candidate& b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.index < b.index;
}

// Insert into a descending-ordered list of length `len`, keeping at most `cap`.
__device__ inline void insert_sorted(Candidate* list, int& len, int cap,
                                     const Candidate& item) {
    if (len == cap && !better(item, list[len - 1])) return;
    int pos = (len < cap) ? len : cap - 1;
    if (len < cap) ++len;
    while (pos > 0 && better(item, list[pos - 1])) {
        list[pos] = list[pos - 1];
        --pos;
    }
    list[pos] = item;
}

// Writes the head of a row's candidate list for a caller that wants log
// probabilities. Both sampling paths below reach this with the full sorted list
// in hand -- the greedy one has run the same top-k reduction the sampled one
// draws from -- so a request that asks for a ranking gets the real distribution
// whichever path its temperature sends it down.
//
// `len` is the number of live entries; the tail up to `top_n` is padded with
// token -1 and logit -inf, the same convention `local_topk_candidates_kernel`
// uses, so a consumer can filter on the token rather than carry a count.
__device__ inline void write_ranking(const Candidate* merged, int len,
                                     int vocab_start, float* out_max_logit,
                                     int* out_top_tokens, float* out_top_logits,
                                     int top_n, int row) {
    // An empty list has no maximum. That only happens when the caller handed
    // this kernel nothing to rank, which the TP path reports as -inf rather
    // than as a zero that would read back as a perfectly likely token.
    if (out_max_logit != nullptr) {
        out_max_logit[row] = (len > 0) ? merged[0].logit : -INFINITY;
    }
    if (out_top_tokens == nullptr || out_top_logits == nullptr) return;
    for (int i = 0; i < top_n; ++i) {
        const size_t slot = static_cast<size_t>(row) * top_n + i;
        if (i < len) {
            out_top_tokens[slot] = merged[i].index + vocab_start;
            out_top_logits[slot] = merged[i].logit;
        } else {
            out_top_tokens[slot] = -1;
            out_top_logits[slot] = -INFINITY;
        }
    }
}

__global__ void sample_top_k_top_p_kernel(
    const float* __restrict__ logits, int* __restrict__ out_tokens,
    float* __restrict__ out_logits, int rows, int vocab, int vocab_start,
    float temperature, float top_p, int top_k, curandState* rng_states,
    const float* __restrict__ uniforms, float* __restrict__ out_max_logit,
    int* __restrict__ out_top_tokens, float* __restrict__ out_top_logits,
    int top_n) {

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int k = (top_k > 0 && top_k < kMaxTopK) ? top_k : kMaxTopK;
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;

    // Per-thread top-k over a strided slice of the vocabulary.
    Candidate local[kMaxTopK];
    int local_len = 0;
    for (int i = threadIdx.x; i < vocab; i += kBlockThreads) {
        insert_sorted(local, local_len, k, Candidate{row_logits[i], i});
    }

    // One shared list receives every thread's candidates.
    __shared__ Candidate merged[kMaxTopK];
    __shared__ int merged_len;
    if (threadIdx.x == 0) merged_len = 0;
    __syncthreads();

    // Serialize the merge over threads to keep the comparator identical to a
    // single-threaded top-k. Correctness first; this runs once per decoded row.
    for (int t = 0; t < kBlockThreads; ++t) {
        if (threadIdx.x == t) {
            for (int i = 0; i < local_len; ++i) {
                insert_sorted(merged, merged_len, k, local[i]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) return;

    // Greedy path: temperature 0 means argmax, which the top-k list already has.
    if (temperature <= 1e-5f || merged_len <= 1) {
        write_ranking(merged, merged_len, vocab_start, out_max_logit,
                      out_top_tokens, out_top_logits, top_n, row);
        out_tokens[row] = merged[0].index + vocab_start;
        out_logits[row] = merged[0].logit;
        return;
    }

    // Softmax over the top-k candidates. Subtracting the max keeps this stable
    // and leaves the relative probabilities of the kept set unchanged.
    const float inv_t = 1.0f / temperature;
    const float max_scaled = merged[0].logit * inv_t;
    float probs[kMaxTopK];
    float sum = 0.0f;
    for (int i = 0; i < merged_len; ++i) {
        probs[i] = __expf(merged[i].logit * inv_t - max_scaled);
        sum += probs[i];
    }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < merged_len; ++i) probs[i] *= inv_sum;

    // Nucleus truncation. The list is already sorted descending.
    int keep = merged_len;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < merged_len; ++i) {
            cum += probs[i];
            if (cum >= top_p) { keep = i + 1; break; }
        }
    }

    float kept_sum = 0.0f;
    for (int i = 0; i < keep; ++i) kept_sum += probs[i];

    // A host-provided uniform keeps the draw identical across TP ranks; the
    // device generator is the single-rank fallback.
    float u;
    if (uniforms != nullptr) {
        u = uniforms[row];
    } else {
        u = curand_uniform(&rng_states[row]);
    }
    if (u >= 1.0f) u = 0.999999f;

    const float target = u * kept_sum;
    float accum = 0.0f;
    int chosen = keep - 1;
    for (int i = 0; i < keep; ++i) {
        accum += probs[i];
        if (accum >= target) { chosen = i; break; }
    }

    out_tokens[row] = merged[chosen].index + vocab_start;
    out_logits[row] = merged[chosen].logit;
    write_ranking(merged, merged_len, vocab_start, out_max_logit,
                  out_top_tokens, out_top_logits, top_n, row);
}

// One block per row, striding the whole vocabulary. This reads a row's logits a
// second time, which is a few megabytes against the gigabytes a decode step
// already moves, and it only runs for a request that asked for log
// probabilities. Doing it here rather than inside the sampler is what makes the
// reported value a true log-softmax: the sampler's candidate list is top-k
// truncated, so summing over it would inflate every probability by however much
// mass the truncation dropped.
//
// A masked-out logit is -inf and contributes exp(-inf) = 0, so a constrained
// row normalizes over exactly the tokens the sampler was allowed to draw.
constexpr int kSumExpThreads = 256;

__global__ void vocab_logsumexp_kernel(
    const float* __restrict__ logits, const float* __restrict__ max_logits,
    float* __restrict__ out_sumexp, int rows, int vocab) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float max_logit = max_logits[row];
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;

    // Accurate exp, not __expf: this sum is a number the client reads back, and
    // it is computed once per generated token, so there is nothing to buy with
    // the approximate one.
    float local = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += kSumExpThreads) {
        local += expf(row_logits[i] - max_logit);
    }

    __shared__ float scratch[kSumExpThreads];
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kSumExpThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_sumexp[row] = scratch[0];
}

// Stage 1 of the TP path: reduce a sharded row to its local top-k and write the
// candidates out as global ids so NCCL can merge them across ranks.
__global__ void local_topk_candidates_kernel(
    const float* __restrict__ logits, int* __restrict__ out_tokens,
    float* __restrict__ out_logits, int rows, int vocab, int vocab_start,
    int top_k) {

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int k = (top_k > 0 && top_k < kMaxTopK) ? top_k : kMaxTopK;
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;

    Candidate local[kMaxTopK];
    int local_len = 0;
    for (int i = threadIdx.x; i < vocab; i += kBlockThreads) {
        insert_sorted(local, local_len, k, Candidate{row_logits[i], i});
    }

    __shared__ Candidate merged[kMaxTopK];
    __shared__ int merged_len;
    if (threadIdx.x == 0) merged_len = 0;
    __syncthreads();
    for (int t = 0; t < kBlockThreads; ++t) {
        if (threadIdx.x == t) {
            for (int i = 0; i < local_len; ++i) {
                insert_sorted(merged, merged_len, k, local[i]);
            }
        }
        __syncthreads();
    }

    // Pad short rows with -inf so every rank contributes a fixed-width block.
    for (int i = threadIdx.x; i < k; i += kBlockThreads) {
        const size_t slot = static_cast<size_t>(row) * k + i;
        if (i < merged_len) {
            out_tokens[slot] = merged[i].index + vocab_start;
            out_logits[slot] = merged[i].logit;
        } else {
            out_tokens[slot] = -1;
            out_logits[slot] = -INFINITY;
        }
    }
}

// Stage 2 of the TP path: sample from candidates already merged across ranks.
// `cand_tokens` holds global ids, so no vocab_start offset applies here.
__global__ void merge_topk_candidates_kernel(
    const int* __restrict__ gathered_tokens,
    const float* __restrict__ gathered_logits,
    int* __restrict__ out_tokens,
    float* __restrict__ out_logits,
    int world,
    int rows,
    int top_k) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0) return;

    Candidate list[kMaxTopK];
    int len = 0;
    const size_t local_count = static_cast<size_t>(rows) * top_k;
    for (int rank = 0; rank < world; ++rank) {
        const size_t row_base = static_cast<size_t>(rank) * local_count +
                                static_cast<size_t>(row) * top_k;
        for (int candidate = 0; candidate < top_k; ++candidate) {
            const int token = gathered_tokens[row_base + candidate];
            const float logit = gathered_logits[row_base + candidate];
            if (token < 0 || isnan(logit)) continue;
            insert_sorted(list, len, top_k, Candidate{logit, token});
        }
    }
    for (int candidate = 0; candidate < top_k; ++candidate) {
        const size_t output = static_cast<size_t>(row) * top_k + candidate;
        if (candidate < len) {
            out_tokens[output] = list[candidate].index;
            out_logits[output] = list[candidate].logit;
        } else {
            out_tokens[output] = -1;
            out_logits[output] = -INFINITY;
        }
    }
}

__global__ void sample_from_candidates_kernel(
    const int* __restrict__ cand_tokens, const float* __restrict__ cand_logits,
    int* __restrict__ out_tokens, float* __restrict__ out_logits, int rows,
    int cand_stride, float temperature, float top_p, int top_k,
    curandState* rng_states, const float* __restrict__ uniforms,
    float* __restrict__ out_max_logit, int* __restrict__ out_top_tokens,
    float* __restrict__ out_top_logits, int top_n) {

    const int row = blockIdx.x;
    if (row != blockIdx.x || threadIdx.x != 0) return;

    // Clamp against the retained-list capacity, not the gathered stride: k
    // indexes `list`, which holds at most kMaxTopK entries.
    int k = (top_k > 0) ? top_k : kMaxTopK;
    if (k > kMaxTopK) k = kMaxTopK;
    if (k > cand_stride) k = cand_stride;
    const int* toks = cand_tokens + static_cast<size_t>(row) * cand_stride;
    const float* lgs = cand_logits + static_cast<size_t>(row) * cand_stride;

    // The merged input is already descending, but re-sort defensively so this
    // kernel does not depend on the reduction's ordering guarantees.
    // The retained list is bounded by k, but the scan must cover every gathered
    // candidate: cand_stride is top_k * tp_world and can exceed kMaxTopK, so
    // stopping at kMaxTopK would silently discard the last ranks' candidates.
    Candidate list[kMaxTopK];
    int len = 0;
    for (int i = 0; i < cand_stride; ++i) {
        if (toks[i] < 0) continue;
        insert_sorted(list, len, k, Candidate{lgs[i], toks[i]});
    }
    if (len == 0) {
        write_ranking(list, 0, 0, out_max_logit, out_top_tokens, out_top_logits,
                      top_n, row);
        out_tokens[row] = 0;
        out_logits[row] = 0.0f;
        return;
    }

    if (temperature <= 1e-5f || len == 1) {
        // Same greedy ranking as the unsharded kernel: the merged list is the
        // list a draw would have come from, so its head is the distribution's
        // most likely token whichever branch commits it.
        write_ranking(list, len, 0, out_max_logit, out_top_tokens, out_top_logits,
                      top_n, row);
        out_tokens[row] = list[0].index;
        out_logits[row] = list[0].logit;
        return;
    }

    const float inv_t = 1.0f / temperature;
    const float max_scaled = list[0].logit * inv_t;
    float probs[kMaxTopK];
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        probs[i] = __expf(list[i].logit * inv_t - max_scaled);
        sum += probs[i];
    }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; ++i) probs[i] *= inv_sum;

    int keep = len;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < len; ++i) {
            cum += probs[i];
            if (cum >= top_p) { keep = i + 1; break; }
        }
    }
    float kept_sum = 0.0f;
    for (int i = 0; i < keep; ++i) kept_sum += probs[i];

    float u = (uniforms != nullptr) ? uniforms[row]
                                    : curand_uniform(&rng_states[row]);
    if (u >= 1.0f) u = 0.999999f;

    const float target = u * kept_sum;
    float accum = 0.0f;
    int chosen = keep - 1;
    for (int i = 0; i < keep; ++i) {
        accum += probs[i];
        if (accum >= target) { chosen = i; break; }
    }
    out_tokens[row] = list[chosen].index;
    out_logits[row] = list[chosen].logit;
    write_ranking(list, len, 0, out_max_logit, out_top_tokens, out_top_logits,
                  top_n, row);
}

__global__ void init_curand_kernel(curandState* states, int count,
                                   unsigned long long seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) curand_init(seed, idx, 0, &states[idx]);
}

}  // namespace

size_t sampler_rng_state_size() { return sizeof(curandState); }

bool init_rng_states(DeviceRngState* states, int count,
                          unsigned long long seed, void* stream) {
    if (states == nullptr || count <= 0) return false;
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    init_curand_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<curandState*>(states), count, seed);
    return cudaGetLastError() == cudaSuccess;
}

bool sample_top_k_top_p_rows(
    const float* logits, int* out_tokens, float* out_logits, int rows,
    int vocab, int vocab_start, float temperature, float top_p, int top_k,
    DeviceRngState* rng_states, const float* uniforms, void* stream,
    float* out_max_logit, int* out_top_tokens, float* out_top_logits,
    int top_n) {
    if (logits == nullptr || out_tokens == nullptr || out_logits == nullptr) {
        return false;
    }
    if (rows <= 0 || vocab <= 0) return false;
    if (uniforms == nullptr && rng_states == nullptr) return false;
    if (top_n < 0 || top_n > kMaxTopK) return false;
    sample_top_k_top_p_kernel<<<rows, kBlockThreads, 0,
                               static_cast<cudaStream_t>(stream)>>>(
        logits, out_tokens, out_logits, rows, vocab, vocab_start, temperature,
        top_p, top_k, reinterpret_cast<curandState*>(rng_states), uniforms,
        out_max_logit, out_top_tokens, out_top_logits, top_n);
    return cudaGetLastError() == cudaSuccess;
}

bool vocab_logsumexp_rows(
    const float* logits, const float* max_logits, float* out_sumexp, int rows,
    int vocab, void* stream) {
    if (logits == nullptr || max_logits == nullptr || out_sumexp == nullptr) {
        return false;
    }
    if (rows <= 0 || vocab <= 0) return false;
    vocab_logsumexp_kernel<<<rows, kSumExpThreads, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        logits, max_logits, out_sumexp, rows, vocab);
    return cudaGetLastError() == cudaSuccess;
}

bool local_topk_candidates(
    const float* logits, int* out_tokens, float* out_logits, int rows,
    int vocab, int vocab_start, int top_k, void* stream) {
    if (logits == nullptr || out_tokens == nullptr || out_logits == nullptr) {
        return false;
    }
    if (rows <= 0 || vocab <= 0 || top_k <= 0 || top_k > kMaxTopK) {
        return false;
    }
    local_topk_candidates_kernel<<<rows, kBlockThreads, 0,
                                   static_cast<cudaStream_t>(stream)>>>(
        logits, out_tokens, out_logits, rows, vocab, vocab_start, top_k);
    return cudaGetLastError() == cudaSuccess;
}

bool merge_topk_candidates(
    const int* gathered_tokens, const float* gathered_logits,
    int* out_tokens, float* out_logits, int world, int rows, int top_k,
    void* stream) {
    if (gathered_tokens == nullptr || gathered_logits == nullptr ||
        out_tokens == nullptr || out_logits == nullptr) {
        return false;
    }
    if (world <= 0 || rows <= 0 || top_k <= 0 || top_k > kMaxTopK) {
        return false;
    }
    merge_topk_candidates_kernel<<<rows, 1, 0,
                                   static_cast<cudaStream_t>(stream)>>>(
        gathered_tokens, gathered_logits, out_tokens, out_logits, world, rows,
        top_k);
    return cudaGetLastError() == cudaSuccess;
}

bool sample_from_candidates(
    const int* cand_tokens, const float* cand_logits, int* out_tokens,
    float* out_logits, int rows, int cand_stride, float temperature,
    float top_p, int top_k, DeviceRngState* rng_states, const float* uniforms,
    void* stream, float* out_max_logit, int* out_top_tokens,
    float* out_top_logits, int top_n) {
    if (cand_tokens == nullptr || cand_logits == nullptr ||
        out_tokens == nullptr || out_logits == nullptr) {
        return false;
    }
    if (rows <= 0 || cand_stride <= 0) return false;
    if (uniforms == nullptr && rng_states == nullptr) return false;
    if (top_n < 0 || top_n > kMaxTopK) return false;
    sample_from_candidates_kernel<<<rows, 1, 0,
                                    static_cast<cudaStream_t>(stream)>>>(
        cand_tokens, cand_logits, out_tokens, out_logits, rows, cand_stride,
        temperature, top_p, top_k, reinterpret_cast<curandState*>(rng_states),
        uniforms, out_max_logit, out_top_tokens, out_top_logits, top_n);
    return cudaGetLastError() == cudaSuccess;
}

int sampler_max_top_k() { return kMaxTopK; }

}  // namespace pocket
