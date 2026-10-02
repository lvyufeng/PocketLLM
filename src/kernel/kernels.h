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
 * `scores` is caller-owned scratch of at least ``q_offset + q_len - first_key``
 * floats, passed in rather than allocated so the score row is reused across
 * heads and tokens. */
void attention(const float *q, int64_t q_len, int64_t n_heads, const float *k_cache,
               const float *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores);

/* The index of the largest of `n` values, ties going to the lowest index.
 *
 * A backend needs this because the logits may live on a device that cannot be
 * read from the host: a caller that does `argmax(download(logits))` transfers a
 * megabyte per token to compare it, and one that asks the backend transfers an
 * integer. The graph does not call it -- the session does -- but it belongs
 * beside the other ops because it is the same kind of thing. */
void argmax(const float *values, int64_t n, int64_t *out);

}  // namespace kernel
}  // namespace pocketllm

#endif /* POCKETLLM_KERNEL_KERNELS_H */