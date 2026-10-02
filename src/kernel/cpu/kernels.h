/* The CPU kernels the Qwen3 graph is built from.
 *
 * These are the operations `python/pocketllm/kernels/ops/` declares, restricted
 * to the ones this graph actually calls and to the f32 case of each. The names
 * match the ABI's on purpose: the point of the ABI is that a second backend
 * implements the same operations, so an op that arrives in the C engine under a
 * different name is a rename to do before CUDA lands rather than after.
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

#ifndef POCKETLLM_KERNEL_CPU_KERNELS_H
#define POCKETLLM_KERNEL_CPU_KERNELS_H

#include <cstdint>

namespace pocketllm {
namespace cpu {

/* ``out[r, :] = x[r, :] / sqrt(mean(x[r, :]^2) + eps) * weight[:]``.
 *
 * The mean is over the row, which for this graph is always the head or the
 * embedding dimension -- `d` is the row length, not the token count. */
void rms_norm(const float *x, const float *weight, float *out, int64_t n_tokens, int64_t d,
              float eps);

/* ``out[r, j] = sum_k x[r, k] * w[j, k]``, with `w` row-major `(n, k)`.
 *
 * `bias` may be null. `into` selects accumulation: when it is true the result
 * is *added* to `out`, which is how the graph expresses a residual connection
 * without a separate add over the whole activation. */
void gemm(const float *x, const float *w, const float *bias, float *out, int64_t m, int64_t n,
          int64_t k, bool accumulate = false);

/* ``out[i, :] = table[token[i], :]``. A gather, not a product: the embedding is
 * the one op whose cost is the bytes it reads and not the arithmetic. */
void embedding(const int32_t *tokens, int64_t n_tokens, const float *table, int64_t vocab,
               int64_t d, float *out);

/* ``out = silu(gate) * up``, elementwise over the whole tensor.
 *
 * ``silu(x) = x / (1 + exp(-x))``, and the exp is the branch-free
 * ``expf`` rather than a clamped approximation -- an f16 weight's rounding is
 * far larger than the difference, and the reference uses the same function. */
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
 * in it. The graph builds the table once per session, for the context length,
 * because it costs a few hundred kilobytes and recomputing it per forward would
 * be the same numbers again.
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
 * is grouped: query head ``h`` reads KV head ``h / group``, and a cache shorter
 * than the query needs is the ordinary decode case, not an error.
 *
 * `scores` is caller-owned scratch of at least ``q_offset + q_len - first_key``
 * floats. Passing it in rather than allocating is what makes the score row
 * reusable across head and token: the graph allocates it once per forward. */
void attention(const float *q, int64_t q_len, int64_t n_heads, const float *k_cache,
               const float *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores);

/* ``out[r, :] = table[token[r], :]`` for a quantized-capable table is step 5's
 * problem; the f16 path goes through `embedding` above after `Weight::data`. */

}  // namespace cpu
}  // namespace pocketllm

#endif /* POCKETLLM_KERNEL_CPU_KERNELS_H */