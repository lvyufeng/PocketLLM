#pragma once

#include <cstddef>

namespace pocket {

// Opaque per-row RNG state. The concrete type is backend-private (curandState on
// CUDA), so callers only ever handle it as a byte buffer sized by
// `sampler_rng_state_size()`. Keeping it opaque is what lets this header
// stay free of vendor SDK includes.
struct DeviceRngState;

// Byte size of one `DeviceRngState`, for allocating an [rows] array.
size_t sampler_rng_state_size();

// Device top-k -> temperature -> top-p sampling over FP32 logit rows.
//
// A full-vocab probability array does not fit in shared memory on SM75
// (62,080 local vocab is 242 KB against a 48 KB limit), so each block reduces
// its row to the top-k candidates first. Because top-k precedes top-p in the
// standard sampling order, restricting to top-k does not change the sampled
// distribution for top_k <= 64.
//
// logits:       [rows, vocab] FP32, device
// out_tokens:   [rows] INT32, device; global ids (vocab_start already added)
// out_logits:   [rows] FP32, device; the selected token's raw logit
// vocab_start:  first global id owned by this rank, for TP vocab sharding
// temperature:  <= 1e-5 falls back to argmax
// top_p:        nucleus threshold; <= 0 or >= 1 disables truncation
// top_k:        candidates retained; clamped to 64
// rng_states:   [rows] DeviceRngState, used only when `uniforms` is null
// uniforms:     optional [rows] host-drawn uniforms in [0,1). Supplying these
//               is what keeps a TP group's draw identical across ranks; the
//               device generator would otherwise diverge per rank.
//
// Ranking outputs, all optional and all off when null. `out_max_logit` receives
// [rows] and is the row's largest raw logit -- the same list's head, which is
// the max a log-softmax needs. `out_top_tokens`/`out_top_logits` receive
// [rows, top_n] row-major and hold the first `top_n` entries of the very list
// the draw was taken from, most likely first, with token -1 and logit -inf where
// a row kept fewer than `top_n` candidates. They are written on the greedy path
// too: an argmax still has a distribution behind it. The values are raw logits,
// so they carry no trace of `temperature` or `top_p`.
//
// TP note: with vocab sharding each rank sees only its own shard, so the
// caller must still reduce across ranks to pick one global token. Passing a
// shared `uniforms` row makes that reduction agree on every rank.
bool sample_top_k_top_p_rows(
    const float* logits,
    int* out_tokens,
    float* out_logits,
    int rows,
    int vocab,
    int vocab_start,
    float temperature,
    float top_p,
    int top_k,
    DeviceRngState* rng_states,
    const float* uniforms,
    void* stream = nullptr,
    float* out_max_logit = nullptr,
    int* out_top_tokens = nullptr,
    float* out_top_logits = nullptr,
    int top_n = 0);

// Per-row sum of exp(logit - max_logit) over the whole vocabulary, for turning
// the ranking above into log probabilities.
//
// logits:      [rows, vocab] FP32, device
// max_logits:  [rows] FP32, device; the row's largest logit, as reported by
//              `sample_top_k_top_p_rows`. Subtracting it keeps every term in
//              [0, 1] and the sum finite; it must be the true maximum or the
//              result overflows.
// out_sumexp:  [rows] FP32, device; the denominator, before `log`.
//
// A masked-out logit is -inf and contributes exp(-inf) = 0, so a constrained
// row normalizes over exactly the tokens the sampler was allowed to draw.
bool vocab_logsumexp_rows(
    const float* logits,
    const float* max_logits,
    float* out_sumexp,
    int rows,
    int vocab,
    void* stream = nullptr);

// TP stage 1: reduce each sharded row to its local top-k, emitted as global ids
// into [rows, top_k] buffers so NCCL can merge candidates across ranks. Short
// rows are padded with token -1 and logit -inf.
bool local_topk_candidates(
    const float* logits,
    int* out_tokens,
    float* out_logits,
    int rows,
    int vocab,
    int vocab_start,
    int top_k,
    void* stream = nullptr);

// Merge NCCL all-gathered local top-k lists into one exact top-k list per row.
// Input layout is world-major [world, rows, top_k], matching ncclAllGather;
// output layout is row-major [rows, top_k]. Ordering is descending logit with
// ascending global token id as the deterministic tie-break.
bool merge_topk_candidates(
    const int* gathered_tokens,
    const float* gathered_logits,
    int* out_tokens,
    float* out_logits,
    int world,
    int rows,
    int top_k,
    void* stream = nullptr);

// TP stage 2: sample from candidates already merged across ranks. `cand_tokens`
// are global ids, so no vocab_start offset is applied. Pass the same `uniforms`
// on every rank to make all ranks commit the same token.
//
// The ranking outputs behave as on `sample_top_k_top_p_rows`, with the token ids
// already global. Note what they rank: the merged candidate set, which is the
// best top_k the group found, not the full vocabulary. Turning those into log
// probabilities therefore needs `vocab_logsumexp_rows` over the *unsharded*
// vocabulary, or a reduction of its per-rank partial sums; normalizing over the
// candidate set alone would inflate every value.
bool sample_from_candidates(
    const int* cand_tokens,
    const float* cand_logits,
    int* out_tokens,
    float* out_logits,
    int rows,
    int cand_stride,
    float temperature,
    float top_p,
    int top_k,
    DeviceRngState* rng_states,
    const float* uniforms,
    void* stream = nullptr,
    float* out_max_logit = nullptr,
    int* out_top_tokens = nullptr,
    float* out_top_logits = nullptr,
    int top_n = 0);

// Largest supported top_k; callers must not request more.
int sampler_max_top_k();

// Initialize one generator per row. Seed identically across TP ranks.
bool init_rng_states(
    DeviceRngState* states,
    int count,
    unsigned long long seed,
    void* stream = nullptr);

}  // namespace pocket
