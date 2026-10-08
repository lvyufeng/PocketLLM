/* The operations the Qwen3 graph is built from, on the host.
 *
 * These are the operations `python/pocketllm/kernels/ops/` declares, restricted
 * to the ones this graph calls and named to match: the point of the ABI is that
 * a second backend implements the same operations, so an op that arrives under
 * a different name is a rename to do before the second backend exists rather
 * than after.
 *
 * The declarations here are the *CPU* implementations. A device backend does
 * not implement these -- it implements `KernelBackend` in `backend.h`, whose
 * methods carry the same signatures over pointers that may be device addresses.
 * The split is deliberate: these functions are the reference the device kernels
 * are checked against, and a reference that shares an interface with the thing
 * it verifies is harder to keep honest.
 *
 * Two conventions run through all of them and are worth stating once:
 *
 *   - A weight is row-major `(rows, cols)` and a product is
 *     ``y[r, j] = sum_k x[r, k] * w[j, k]`` -- the weight holds the *output*
 *     axis. That is the GGUF layout read straight off the mapping, so nothing
 *     is transposed on load, and it is why the GEMM is a "dot each input row
 *     with each weight row" rather than the BLAS `A @ B`.
 *   - Everything is float32. The weights may be f16 and the activations are
 *     whatever the last op wrote, but every kernel here computes in f32: the
 *     reference backend casts to f32 for the same reason, and the two are
 *     compared at rtol 2e-3.
 */

#ifndef POCKETLLM_KERNEL_KERNELS_H
#define POCKETLLM_KERNEL_KERNELS_H

#include <cstdint>

#include "kernel/backend.h"

namespace pocketllm {
namespace kernel {

/* ``out[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * weight[:]``.
 *
 * The mean is over the row, which for this graph is always the head or the
 * embedding dimension -- `d` is the row length, not the token count. */
void rms_norm(const float *x, const float *weight, float *out, int64_t n_tokens, int64_t d,
              float eps);

/* ``out[r, j] = sum_k x[r, k] * w[j, k]``, with `w` row-major `(n, k)`.
 *
 * `bias` may be null. `accumulate` selects accumulation: when it is true the
 * result is *added* to `out`, which is how the graph expresses a residual
 * connection without a separate add over the whole activation. */
void gemm(const float *x, const float *w, const float *bias, float *out, int64_t m, int64_t n,
          int64_t k, bool accumulate = false);

/* ``out[r, j] = sum_k x[r, k] * w[j, k]`` with `w` stored as packed blocks.
 *
 * The same product as `gemm` with the right operand read through its quantizer
 * on the fly, instead of from a float matrix a loader expanded. This is the op
 * that makes the width ladder real: a 0.6B checkpoint in f16 is 1.4 GB and the
 * same file at `q4_k_m` is 456 MB, and the difference is entirely in whether
 * this function exists.
 *
 * `blocks` is `(n, k / block_weights, block_bytes)` as GGUF stores it, one
 * `type_id` for the whole tensor, which is what GGUF requires -- a row's blocks
 * are contiguous and a tensor has one type. `k` must be a multiple of the
 * block width; a shape that is not is refused by the caller rather than padded
 * here, because a silent tail would be a weight read from the wrong place.
 *
 * ## Two ways to compute it, and the argument for the fast one
 *
 * The default path quantizes the *activations* to int8 once per row (see
 * `quant/q8k.h`) and takes the integer route through both operands: unpack the
 * weight to 4 or 6 unsigned bits, multiply against the signed activation byte
 * with `_mm256_maddubs_epi16`, fold the block scale in with `_mm256_madd_epi16`,
 * and apply one float multiply per 256 weights. Every product inside the dot is
 * then exact 8-bit integer arithmetic, and the per-weight decode and float
 * multiply of the exact path are gone. That is the algorithm llama.cpp runs,
 * and it is the reason its one-thread decode was three times ours.
 *
 * Its cost is a real precision change and not a last-bit one: an activation
 * element picks up up to half a step of its block's scale, measured at 0.5-0.6%
 * of the output's magnitude on the conformance shapes -- an order inside the
 * `QUANTIZED_RTOL` of 0.05 the tests hold it to, and *the same* error
 * llama.cpp's logits carry, which is what makes the two engines comparable
 * elementwise where `test_quantized_forward.py` records they were not.
 *
 * The exact path is not dead code. A build without AVX2 uses it, and
 * `$POCKETLLM_CPU_EXACT_GEMM` selects it on a build with one -- which is how
 * the fast path's error is measurable on a real checkpoint rather than only
 * asserted, and what `tests/native/test_cpu_parallel.py` turns on to compare
 * the two paths on identical input. */
void gemm_quant(const float *x, const uint8_t *blocks, const float *bias, float *out, int64_t m,
                int64_t n, int64_t k, int type_id, bool accumulate = false);

/* ``out[i, :] = table[token[i], :]``. A gather, not a product: the embedding is
 * the one op whose cost is the bytes it reads and not the arithmetic. */
void embedding(const int32_t *tokens, int64_t n_tokens, const float *table, int64_t vocab,
               int64_t d, float *out);

/* The same gather from a quantized table, `blocks` being
 * `(vocab, d / block_weights, block_bytes)`.
 *
 * A separate entry point rather than a flag on the one above because the two
 * read different memory: the dense table is `float`, this is packed bytes, and
 * a pointer that could be either is a cast that loses the type that says which.
 *
 * This exists because Qwen3-0.6B ties its output projection to the embedding,
 * so `token_embd.weight` is *both* the first op of the graph and a weight the
 * final matmul contracts against. A `q4_k_m` checkpoint leaves it in `q4_k`,
 * and a build that could only gather from a float table would have to expand
 * 155 MB to 622 MB to read one row of it per token -- which is most of the
 * saving the quantization was for. The decode is per row, so a token costs one
 * block walk and not a full-table dequantization. */
void embedding_quant(const int32_t *tokens, int64_t n_tokens, const uint8_t *blocks, int64_t vocab,
                     int64_t d, int type_id, float *out);

/* ``out = silu(gate) * up``, elementwise over the whole tensor. */
void silu_mul(const float *gate, const float *up, float *out, int64_t n);

/* RoPE, the split-half (NEOX) layout: ``x[i]`` is rotated against
 * ``x[i + d/2]``.
 *
 * Qwen3 selects NEOX (`LLAMA_ROPE_TYPE_NEOX`); the interleaved layout the ABI
 * also admits is a permutation of the same computation and is deliberately not
 * implemented here, because a second layout with no checkpoint to test it
 * against is a branch that would only be exercised by a bug.
 *
 * The tables arrive ready-made as ``(capacity, d/2)`` -- one row of ``d/2``
 * angles per absolute position -- rather than as a theta to compute from, which
 * keeps the trig out of the token loop where it would be the only transcendental
 * in it.
 *
 * `x` is `(n_tokens, n_heads, d)`, its tokens occupying positions
 * `start_pos .. start_pos + n_tokens - 1`, and is modified in place: the
 * rotation is linear and the caller has already put the pre-rotation value
 * wherever it needed it -- the KV cache keeps the *rotated* key, as llama.cpp
 * does. */
void rope_neox(float *x, int64_t n_tokens, int64_t n_heads, int64_t d, int64_t start_pos,
               const float *cos_table, const float *sin_table);

/* ``out[t, h, :] = softmax_s(q[t, h, :] . k[s, h/group, :] * scale) v[s, h/group, :]``
 * over ``s`` in ``[first_key, q_offset + t]``.
 *
 * `q` is the chunk being written and the caches are the whole history, laid out
 * ``[position][kv_head][d]`` with a row stride of ``n_head_kv * d``. Attention
 * is grouped: query head ``h`` reads KV head ``h / group``.
 *
 * `scores` is caller-owned scratch holding ``kAttentionScoreRowsPerTask *
 * (q_offset + q_len - first_key)`` floats per *concurrent* task, which the
 * backend sizes through its own ``attention_scratch``.  This kernel runs
 * ``(q_len / kAttentionRows, n_heads / kAttentionHeadBatch)`` units in parallel
 * and each unit writes ``kAttentionHeadBatch * kAttentionRows`` rows, so the
 * buffer is several rows per task, not one; a caller that sized it from the
 * one-row-per-task contract would under-allocate the moment more than one
 * thread is in play.
 *
 * **The size depends on the head batch and the task count, not on `n_heads`.**
 * A previous version of this comment said the buffer was ``n_heads`` rows wide,
 * which happened to be true when the batch was one head and over-allocated by
 * the group factor once batched heads share a task.  The one thing that must
 * not drift is `kAttentionScoreRowsPerTask` matching what `attention` indexes:
 * the backend's `attention_scratch` and any caller that sizes the buffer are
 * derived from the same two constants the kernel is. */
/* One cache row's worth of f16 elements, widened to `d` floats or narrowed from
 * them.  `row` is the cache's own layout -- `n_head_kv * d` f16 elements with
 * this head's `d` starting at `kv_head * d` -- so `dst`/`src` is a single head's
 * slice of it.
 *
 * These are the f16 cache's whole interface, and it is one function pair rather
 * than a second family of dot kernels on purpose: see `kv_row_to_float` in
 * `kernels.cpp` for why, and for why widening a `d`-element row costs nothing
 * the kernels were not already paying for an f32 one.  `kv_row_to_float` is
 * exact -- every f16 is an f32 -- and `float_to_kv_row` rounds to nearest, which
 * is the loss the cache accepts. */
void kv_row_to_float(const void *row, int64_t d, float *dst);
void float_to_kv_row(const float *src, int64_t d, void *row);

/* `k_cache`/`v_cache` are `KVDtype` elements wide -- f32 or f16 -- while `q`,
 * `out` and `scores` are always f32. See `Backend::attention` for why the cache
 * is the one operand whose width is worth trading. */
void attention(const float *q, int64_t q_len, int64_t n_heads, const void *k_cache,
               const void *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores,
               KVDtype kv_dtype = KVDtype::kF32);

/* How many *query rows of one head* one attention task scores in a single walk
 * over the key vector.
 *
 * The score pass is ``q_len * span`` dots per head over the same ``span`` key
 * vectors -- at prefill that is a triangular half-matrix, and the key row is
 * read once per query row. Tiling `kAttentionRows` query rows of the same head
 * against one key row reads it once for all of them. Both the kernel and the
 * CPU backend's `attention_scratch` use this constant, so the score-row
 * allocation and the partition the kernel takes cannot disagree.
 *
 * **It is 8, and the number is measured, not chosen.** The whole attention
 * call at ``q_len = 512``, d = 128, 16 heads, 44 threads, interleaved
 * best-of-five against R = 4 and R = 16 on the same host:
 *
 *     R=4    146.6 / 143.5 / 143.0 / 146.6 us   <- what "four rows" shipped as
 *     R=8    130.2 / 122.5 / 129.9 / 133.8 us   <- 1.12x, the shipped value
 *     R=16   146.3 us                            <- the score pass regressed
 *
 * The attention call is bandwidth-bound, so the quantity that decides R is how
 * many times the K and V slabs are streamed, and that is
 * ``q_len / R * n_heads / kAttentionHeadBatch`` -- 1024 slab-lengths at R = 4
 * against 512 at R = 8. The measured split of the call at R = 4 is score 66 ms,
 * softmax 16 ms, weighted sum 79 ms, and R = 8 moves both streamed passes in
 * the same direction: score 53 ms and weighted sum 58 ms, for 131 ms against
 * 161 ms. **R = 16 is where the score pass turns around and gives the win back**
 * -- 85 ms, worse than R = 4's 66 -- and the cause is register pressure rather
 * than traffic: `dot_tile_r` holds one `__m256` accumulator per two query rows,
 * so 16 rows is 8 live accumulator registers plus the key vector and the query
 * loads, which overflows what AVX2 has and spills. The weighted sum keeps
 * improving at 16 rows (58 -> 46 ms), which is why the constant is a compromise
 * and why 8 -- where both passes are better than at 4 -- is the one that ships.
 * The full sweep is in ``docs/architecture/c_engine.md`` under "Eight query rows
 * per key walk".
 *
 * **`kAttentionGrain = 1` is now measured in blocks.** The unit changed from
 * ``(token, head)`` to ``(kAttentionRows tokens, head)`` when the tiling
 * arrived, and a grain of 1 is still right: every block is one output row per
 * token it covers, so the smallest useful grain on the new unit is one.
 *
 * The tiling is *bit-exact* to the one-row kernel, and the row count is not
 * part of that argument: `dot_tile_r<R>` pairs rows two at a time and gives each
 * pair its own accumulator, so a row's four lanes and their reduce are the ones
 * it has at every R. Widening the tile changes how many rows share a walk, not
 * what any row computes. `tests/native/test_cpu_parallel.py` holds that to the bytes.
 * The tail rows of a causal block (``s`` past the last key every row can see)
 * take `dot4` one row at a time rather than a second tiled kernel, so there is
 * exactly one tiled code path to be right. */
constexpr int64_t kAttentionRows = 8;

/* How many attention tasks one thread owns before it is worth waking another.
 * Shared with the CPU backend, which sizes the score-row scratch from the same
 * partition the kernel takes; if the two used different grains the buffer
 * would be sized for a partition that does not happen.
 *
 * **This is 1, and the number it replaced was 32.**  A grain of 32 was read off
 * the wrong axis: the job is ``q_len * n_heads`` units wide, and 32 of them is a
 * whole decode step -- `q_len` is 1 and there are 16 heads, so the largest
 * partition `partition_size` can return is one task, and attention ran on one
 * core out of twenty-two.  At ``q_len = 1`` the unit count is 16 whatever the
 * grain is, so any grain below 16 splits into all 16 units and hands one to each
 * thread; only at 32 or above does it collapse to a single task.
 *
 * The unit is self-contained -- one output row, one private score row, no
 * coupling to any other unit -- so the split is safe at any grain, and it is
 * *bit-exact*: the per-unit arithmetic is untouched, only which core does it.
 * `tests/native/test_cpu_parallel.py` holds that to the token sequence.
 *
 * The kernel reads its own comment for why the unit is the right task and not a
 * finer slice of it. */
constexpr int64_t kAttentionGrain = 1;

/* How many query heads `attention` scores against one key row in a single walk.
 *
 * Grouped attention has ``group = n_heads / n_head_kv`` query heads reading each
 * KV head, and the score pass walks every key row once *per query head*.  A
 * decode step pays that in full: `span` key rows, `group` times each, and the K
 * cache is not in L2 at any useful context.  Scoring `kAttentionHeadBatch` heads
 * together halves the traffic, and the K row is what the pass is bound by -- `d`
 * floats are used out of a `n_head_kv * d` stride, so the walk streams 4096
 * bytes to read 512 per head.
 *
 * **Pairing is safe where tiling rows is not, and the difference is whether the
 * rows share a producer.**  Two heads of one KV group score *different* query
 * vectors against the *same* key vector, so both of the batching rewrites are
 * exact: the key load is reused, and no two lanes share an accumulator.  Two
 * query rows of one head have overlapping but unequal spans, and hoisting either
 * the key load or the weight lookup past the row loop changes which products are
 * summed in what order -- the last-bit change the softmax turns into a different
 * token.  See `kAttentionRows`, which is why the row tiling is confined to the
 * region a whole tile shares.
 *
 * 2 is the shipped value: it is a property of the model (``group`` is 2 for
 * Qwen3-0.6B), not a tuning knob, and any `group` is handled by the pairing
 * loop.  `$POCKETLLM_CPU_SCALAR_DOT` routes every lane to the scalar `dot`,
 * which is the equivalence check `tests/native/test_cpu_parallel.py` runs. */
constexpr int64_t kAttentionHeadBatch = 2;

/* The largest GQA group (query heads per KV head) the decode attention path can
 * score in one walk.
 *
 * `kAttentionHeadBatch` is the *prefill* unit and is independent of `group`: it
 * batches two heads that share a KV head however many do.  The flash *decode*
 * path is different -- it walks one KV head at a time and scores all `group`
 * query heads that read it in a single stack array -- so that array has to be
 * sized by the model's group, which the unit does not choose.  It used to be
 * sized `kAttentionHeadBatch` (= 2), which is why a `group == 4` model (Qwen3-4B
 * and 8B, 32 query heads over 8 KV heads) read uninitialized stack for heads
 * 1..3 and decoded to garbage while its first (prefill) token stayed correct.
 *
 * 8 covers every Qwen3 width in the ladder with room to spare; a model outside
 * it is refused by name rather than silently truncated. */
constexpr int64_t kAttentionMaxGroup = 8;

/* How many score rows of *one concurrent task* the caller's scratch has to hold.
 *
 * The kernel's attention unit is ``(a block of kAttentionRows query tokens, a
 * batch of kAttentionHeadBatch query heads sharing one KV head)``, and each unit
 * writes one score row per (batched head, token) pair into its own private
 * region.  A caller that sizes the buffer from this constant and the same
 * partition the kernel takes cannot under-allocate; one that reasons from
 * ``n_heads`` alone can, and the failure is a silent cross-task overwrite rather
 * than a crash.
 *
 * It lives here rather than in `kernels.cpp` because the CPU backend's
 * `attention_scratch` needs it and the kernel needs it -- the same reason
 * `kAttentionRows` is here. */
constexpr int64_t kAttentionScoreRowsPerTask = kAttentionRows * kAttentionHeadBatch;

/* The index of the largest of `n` values, ties going to the lowest index.
 *
 * A backend needs this because the logits may live on a device that cannot be
 * read from the host: a caller that does `argmax(download(logits))` transfers a
 * megabyte per token to compare it, and one that asks the backend transfers an
 * integer. The graph does not call it -- the session does -- but it belongs
 * beside the other ops because it is the same kind of thing. */
void argmax(const float *values, int64_t n, int64_t *out);

/* ``out[r, :] = exp(x[r, :] - max) / sum(exp(x[r, :] - max))``, the softmax
 * over each row of an ``(rows, cols)`` tensor.
 *
 * The shift by the row maximum is not an optimization: a vocabulary of 151936
 * accumulated logits has an exponent large enough to overflow the moment the
 * distribution is not already near-uniform, and `exp` of a large positive float
 * is infinity, whose ratio to infinity is a NaN distribution. The reference
 * subtracts the row max for the same reason and is the definition this matches.
 *
 * A *row* and not the whole tensor because the reference's `axis=-1` is a row
 * reduction over the last dimension, and a multi-row input is what the
 * conformance case shapes its input as.
 *
 * `topk_sample` calls the same helper this does -- one file-local function,
 * templated on the accumulator -- rather than carrying its own copy of the
 * reduction: the conformance test certifies the code the sampler runs only if
 * there is one of it. This op accumulates in float, matching the reference's
 * numpy sum; the sampler accumulates the same expression in double, and the
 * helper's own comment says why. */
void softmax(const float *x, float *out, int64_t rows, int64_t cols);

/* ``out[i] = logits[i] / temperature``, the one logit transform a decode
 * applies before sampling.
 *
 * Refuses a `temperature <= 0` by throwing, which is the reference's rule: a
 * division by zero is an infinity or a NaN in every entry, and a caller that
 * meant greedy is a caller that should not have called this at all -- the
 * engine's own greedy path never does. */
void logits_temperature(const float *logits, float *out, int64_t n, float temperature);

/* Draw one token from the top-k/top-p/min-p truncated softmax, given a uniform
 * variate in ``[0, 1)``.
 *
 * This is a faithful port of `backends/reference/kernels.py:topk_sample`, and
 * the places where a naive port picks a different token are each worth naming:
 *
 *   - The ranking is by *stable* descending probability, so a tie goes to the
 *     lower token id. A comparator that is not a strict weak order (`>=` on the
 *     value alone) reorders ties and is undefined behaviour besides.
 *   - The `min_p` cutoff keeps ``p >= min_p * top`` -- inclusive, so `min_p=1`
 *     keeps the argmax rather than discarding everything.
 *   - `top_p` keeps the smallest set whose cumulative mass reaches `p`, which
 *     is ``(cumulative - p) < top_p`` OR the first index where
 *     ``cumulative >= top_p``. When nothing crosses, the kept index is **0**,
 *     matching numpy's `argmax` of an all-false mask -- a loop that keeps the
 *     last index instead picks a different token.
 *   - The inverse CDF is a search for the first index with
 *     ``cumulative >= uniform`` (`side="left"` in numpy), not `>`.
 *   - A total mass that rounds to zero returns the top-ranked token rather than
 *     dividing by it; the top token is always kept, so this is nearly
 *     unreachable, but it is the reference's answer.
 *
 * `order` is caller-owned scratch of at least `vocab` int64s -- the ranked
 * index list -- passed in rather than allocated so a decode does not allocate
 * per token, the same reason `attention` takes its `scores` row from the
 * caller. `uniform` is a value and not a pointer because the schema types it as
 * a scalar: a real sampler passes the variate in, and the engine holds no RNG.
 *
 * NaN is not part of the contract -- the reference's answer on a NaN logit is
 * whatever `argsort` and `searchsorted` happen to do with it, and this does not
 * reproduce that. Do not feed it one.
 */
void topk_sample(const float *logits, int64_t vocab, float uniform, int64_t top_k, float top_p,
                 float min_p, int64_t *order, int64_t *out);

}  // namespace kernel
}  // namespace pocketllm

#endif /* POCKETLLM_KERNEL_KERNELS_H */