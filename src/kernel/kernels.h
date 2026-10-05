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
 * The decode is per *weight* and the value is consumed by the dot product
 * immediately, so no row is ever materialized: the reference backend reaches
 * the same answer by decoding the whole matrix with numpy first, and this does
 * it without the f32 copy. The tolerance they are compared at is
 * :data:`QUANTIZED_RTOL` on the test side. */
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
 * `scores` is caller-owned scratch holding ``q_offset + q_len - first_key``
 * floats per *concurrent* task, which the backend sizes through its own
 * ``attention_scratch``.  This kernel runs ``(token, head)`` units in parallel
 * and each task writes its own row, so the buffer is several rows, not one;
 * a caller that sized it from the old one-row contract would under-allocate
 * the moment more than one thread is in play. */
void attention(const float *q, int64_t q_len, int64_t n_heads, const float *k_cache,
               const float *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores);

/* How many ``(token, head)`` units one attention task owns before it is worth
 * waking a thread.  Shared with the CPU backend, which sizes the score-row
 * scratch from the same partition the kernel takes; if the two used different
 * grains the buffer would be sized for a partition that does not happen. */
constexpr int64_t kAttentionGrain = 32;

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