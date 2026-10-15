/* The device the graph runs on.
 *
 * `Qwen3Model` builds one sequence of operations; this is what those operations
 * are carried out *by*. There are two implementations -- `cpu` and `cuda` -- and
 * the graph is written once against this interface, so a change to the layer is
 * a change in one place rather than in one place per device.
 *
 * The interface is pointers and lengths and nothing else. A backend is free to
 * interpret an address as a host pointer (the CPU one does) or as a device
 * offset, and the graph is written so that it never dereferences one itself:
 * every value it touches between ops is a buffer this backend allocated and
 * only this backend reads. That is the property that makes the same source
 * correct on both -- and it is why `forward` returns a *device* buffer whose
 * conversion to host logits happens in `to_host` at the very end rather than by
 * the graph reading its own output.
 *
 * One process owns one device. There is no rank, no device list and no
 * collective here, and adding one would resurrect a feature this project
 * deleted on purpose.
 */

#ifndef POCKETLLM_KERNEL_BACKEND_H
#define POCKETLLM_KERNEL_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pocketllm {
namespace kernel {

/* The element type of the K/V cache. Only the two widths the attention kernel
 * reads are named -- a cache is never f32 *and* f16 within one call, and a
 * third value would be a type the kernels do not implement. */
enum class KVDtype : int32_t {
  kF32 = 0,
  kF16 = 1,
};

inline int64_t kv_dtype_size(KVDtype dtype) { return dtype == KVDtype::kF16 ? 2 : 4; }

/* A number that identifies a device buffer. An opaque handle rather than a
 * pointer because on the CPU it is an address and on a device it may not be one
 * -- and because a caller that treats it as a pointer is a caller that has
 * stopped being portable. */
struct DeviceBuffer {
  uintptr_t handle = 0;
  int64_t bytes = 0;
};

/* Which of a packed weight's two consumers a materialize call is for.
 *
 * The two are separate because they are separate *shapes* of the same bytes: a
 * weight's own `(rows, cols)` is what `gemm_quant` reads, while the embedding
 * gather is read as `(vocab, d)`, and a backend that files a decoded copy under
 * the numbers the forward path passes needs to be told which pair to build.
 * A tied checkpoint, where one buffer backs both, needs both forms -- and the
 * model says so by calling `materialize_packed` once with each. */
enum class PackedUse : int32_t {
  kGemm = 0,      /* `gemm_quant(x, blocks, ..., rows, cols, type_id, ...)` */
  kEmbedding = 1, /* `embedding_quant(tokens, n, blocks, vocab, d, type_id, ...)` */
};

class Backend {
 public:
  virtual ~Backend() = default;

  /* The name this backend answers to, matching `pocketllm_open`'s argument and
   * the Python registry's entry. */
  virtual const char *name() const = 0;

  /* Allocate `bytes` on the device, or throw. Each call is independent: the
   * graph allocates its activation buffers once per session and reuses them, so
   * this is not on the per-token path. */
  virtual DeviceBuffer allocate(int64_t bytes) = 0;
  virtual void release(DeviceBuffer buffer) = 0;

  /* Host <-> device. `copy_to_device` is used to upload weights and token ids;
   * `copy_to_host` is used to read back the logits, and nothing else -- the
   * graph never brings an intermediate value back, because a round trip per
   * layer would make the device pointless. */
  virtual void copy_to_device(DeviceBuffer dst, const void *src, int64_t bytes) = 0;
  virtual void copy_to_host(void *dst, DeviceBuffer src, int64_t bytes) = 0;

  /* Device to device, for a value that never needs to be seen by the host.
   *
   * The KV cache append is the case this exists for. Expressing it as
   * `copy_to_device` would be wrong in a way the CPU backend cannot detect --
   * there the two are the same memcpy -- and would fail on a card by
   * dereferencing a device address as if it were host memory. */
  virtual void copy_device_to_device(DeviceBuffer dst, DeviceBuffer src, int64_t bytes) = 0;

  /* Zero a device region. Filling rather than allocating zeroed memory because
   * a reused activation buffer has the previous token in it. */
  virtual void fill(DeviceBuffer dst, float value) = 0;

  virtual void rms_norm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer out, int64_t n_tokens,
                        int64_t d, float eps) = 0;

  /* Layer normalization, as `kernels.h` documents it.  `bias` may be the null
   * buffer (`handle == 0`), which the ABI declares as optional. */
  virtual void layer_norm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer bias, DeviceBuffer out,
                          int64_t rows, int64_t d, float eps) = 0;

  /* ``out[r, j] = sum_k x[r, k] * w[j, k]``, with `w` row-major `(n, k)`.
   * `bias` may be null; `accumulate` adds into `out` instead of overwriting
   * it, which is how the residual connections are expressed. */
  virtual void gemm(DeviceBuffer x, DeviceBuffer w, DeviceBuffer bias, DeviceBuffer out,
                    int64_t m, int64_t n, int64_t k, bool accumulate) = 0;

  /* ``out[r, j] = sum_k x[r, k] * w[j, k]`` with `w` packed as k-quant blocks.
   *
   * `blocks` is `(n, k / 256, block_bytes)` for the single `type_id` the tensor
   * carries, and the decode is per weight -- no f32 copy of the weight exists
   * at any point, which is the whole reason the packing is worth reading. A
   * backend whose device cannot decode a format refuses the call by name rather
   * than falling back to a dequantized path, because the fallback is the
   * memory the quantization was supposed to save.
   *
   * `type_id` is a GGML storage id, the same numbering `abi/spec.h` and the
   * checkpoint use, so a caller never translates between two vocabularies. */
  virtual void gemm_quant(DeviceBuffer x, DeviceBuffer blocks, DeviceBuffer bias, DeviceBuffer out,
                          int64_t m, int64_t n, int64_t k, int type_id, bool accumulate,
                          bool q6k_repacked = false) = 0;

  /* ``out[i, :] = table[token[i], :]``. `tokens` holds `n_tokens` int32 ids. */
  virtual void embedding(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer table, int64_t vocab,
                         int64_t d, DeviceBuffer out) = 0;

  /* The same gather from a packed table -- see `kernels.h`'s `embedding_quant`
   * for why a tied-embedding checkpoint needs it. */
  virtual void embedding_quant(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer blocks,
                               int64_t vocab, int64_t d, int type_id, DeviceBuffer out) = 0;

  virtual void silu_mul(DeviceBuffer gate, DeviceBuffer up, DeviceBuffer out, int64_t n) = 0;

  /* The tanh-approximated GELU, as `kernels.h` documents it. */
  virtual void gelu(DeviceBuffer x, DeviceBuffer out, int64_t n) = 0;

  /* The vision tower's patch embedding, as `kernels.h` documents it: a
   * square-kernel 2-D convolution with `weight` in GGUF's
   * `(out_ch, in_ch * kernel^2)` order and `out` patch-major.
   *
   * A backend whose arithmetic goes through a vendor op with no convolution in
   * it refuses this by name -- the same shape of refusal `gelu` gets -- rather
   * than answering with an image the tower never described. */
  virtual void conv2d(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer bias, DeviceBuffer out,
                      int64_t in_ch, int64_t height, int64_t width, int64_t out_ch, int64_t kernel,
                      int64_t stride, int64_t pad) = 0;

  /* RoPE, split-half layout, as `kernels.h` documents it. */
  virtual void rope_neox(DeviceBuffer x, int64_t n_tokens, int64_t n_heads, int64_t d,
                         int64_t start_pos, DeviceBuffer cos_table, DeviceBuffer sin_table) = 0;

  /* RoPE, interleaved layout -- the same rotation with the other pairing, which
   * `kernels.h` explains.  A separate entry point rather than a flag on
   * `rope_neox` because a backend may implement one convention and not the
   * other: a device that reaches its arithmetic through a vendor op (the
   * Ascend backend's `aclnnRopeCustom`) has whatever mode that op has, and
   * saying so is better than passing a flag it silently ignores. */
  virtual void rope_norm(DeviceBuffer x, int64_t n_tokens, int64_t n_heads, int64_t d,
                         int64_t start_pos, DeviceBuffer cos_table, DeviceBuffer sin_table) = 0;

  /* How many bytes the scratch for an `attention` call must hold, given that the
   * longest score row in it is `max_span` entries.
   *
   * This is a query rather than a constant because the answer is a property of
   * *how the backend is parallelized*, which the graph has no way to know. The
   * CPU computes the `q_len * n_heads` (query, head) pairs one at a time and
   * reuses a single row; a GPU runs them as concurrent blocks and each needs its
   * own, or two blocks writing the same row would interleave into a softmax over
   * a mixture of two different heads. Sizing this from the graph would be a
   * silent cross-backend assumption that is correct on exactly one of them.
   *
   * `max_span` must be at least `q_offset + q_len - first_key` for the call it
   * is sizing. */
  virtual int64_t attention_scratch(int64_t q_len, int64_t n_heads, int64_t max_span) const = 0;

  /* Grouped-query attention over a cache laid out ``[position][head][d]``, with
   * the query chunk at `q_offset`. `scores` holds what `attention_scratch` asked
   * for.
   *
   * **`causal` is whether the rows advance.**  With it, row `t` of the chunk
   * ends at `q_offset + t`: a prefill, or a decode step, where the query is at
   * its own position.  Without it, every row ends at `q_offset` -- the whole
   * span `[first_key, q_offset]` for all of them, which is a vision tower
   * attending over a patch sequence.  A backend that implements one and not the
   * other refuses the other *by name* rather than answering with the wrong mask,
   * because the wrong mask is fluent, finite and wrong.
   *
   * `kv_dtype` is the element type of `k_cache`/`v_cache`, and only those two:
   * the query, the output and the score scratch stay f32 on every backend. The
   * cache is the one operand whose *size* grows with the sequence, so it is the
   * one whose width is worth trading precision for -- at 512 rows a f32 cache is
   * twice the bytes of an f16 one on the decode path's only streaming read, and
   * the probe in `docs/architecture/c_engine.md` measures the attention call
   * halving when it does. A backend that cannot consume an f16 cache is expected
   * to widen it rather than refuse: the graph picks the cache width from
   * `preferred_kv_dtype`, so a backend that cannot must override that too. */
  virtual void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                         DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                         int64_t q_offset, bool causal, float scale, DeviceBuffer out,
                         DeviceBuffer scores, KVDtype kv_dtype) = 0;

  /* Append `n` rows of the chunk just computed to a K/V cache slab that is
   * `elem` bytes per element wide.
   *
   * `dst` and `src` are both device memory: `src` holds `n * n_head_kv * d`
   * floats laid out ``[token][head][d]``, `dst` begins at the chunk's own slot
   * in a slab whose row stride is `n_head_kv * d` elements and whose rows are
   * `capacity` wide, and this writes the same ``[token][head][d]`` shape into
   * it.  `elem` is 4 for an f32 slab -- where this is exactly the copy
   * `copy_device_to_device` would do -- and 2 for an f16 one, where the rows are
   * not contiguous in the destination and the conversion is per head.
   *
   * It exists as its own method rather than as a loop in the graph for the same
   * reason `copy_device_to_device` does: a caller that walked the pointers
   * itself would be dereferencing device addresses as host ones, which happens
   * to work on the CPU and is a crash on a card.  `src` rows are `d` apart and
   * `dst` rows are `n_head_kv * d` apart, so a flat elementwise conversion over
   * the chunk would write every head into the next head's slot -- finite,
   * plausible, wrong. */
  virtual void kv_append(DeviceBuffer dst, DeviceBuffer src, int64_t n, int64_t n_head_kv,
                         int64_t d, int64_t elem) = 0;

  /* What width this backend wants its K/V cache in.
   *
   * f16 by default: it is llama.cpp's *library* default (`-ctk f16 -ctv f16`),
   * so the two engines read the same number of bytes off the cache and a
   * comparison between them is a comparison of the walk rather than of the
   * storage -- and at 512 rows it is half the bytes on the decode path's only
   * streaming read (`docs/architecture/c_engine.md` measures the attention call
   * halving when it does).  A backend with no f16 path -- or one whose native
   * precision makes the cast pointless, like the card -- overrides this to
   * `kF32`.  The oracle is pinned to llama.cpp's f32 cache for a separate
   * reason that is llama.cpp's rather than this tree's; see
   * `tests/native/llama_oracle.py`. */
  virtual KVDtype preferred_kv_dtype() const { return KVDtype::kF16; }

  /* Whether this backend wants the K/V *projections* (`k_`, `v_`) stored in the
   * cache's own width on the device, so a later `kv_append` -> `attention` never
   * has to convert them on the host.
   *
   * OFF by default, and the default is the whole point: a backend for which the
   * f32 activation -> f16 cache cast is a free device operation (the card) or a
   * non-issue (the CPU, whose cache is host memory) keeps the existing path and
   * the existing numerics untouched.  A backend that overrides this to `true`
   * must produce a cache that `attention` reads back to *exactly* the values the
   * f32 path's `to_f16` would have written -- it is a cost removal, not a
   * precision change -- which is checked by the identity gate, not by this flag.
   *
   * The graph removes a cast in `kv_append`, never changes the cache's numerical
   * contents, so `preferred_kv_dtype` is unaffected: the storage width and the
   * projection width are separate questions and this answers only the second. */
  virtual bool prefers_kv_projection_in_cache_dtype() const { return false; }

  /* Whether this backend can be handed the checkpoint's packed weights at load,
   * asked to build its own decoded copy of each, and then left holding only
   * those copies -- the packed bytes released.
   *
   * OFF by default, and the default is the whole contract. On the CPU and on
   * the card the packed bytes *are* the weight: `gemm_quant` decodes them where
   * they lie and there is no second copy to fall back on, so releasing them
   * would be a use-after-free and the answer to this question is false for
   * them.  A backend that answers true is stating that it keeps a resident
   * decode of every packed weight it is asked for and that it can serve the
   * forward path from that decode alone -- the Ascend backend's device planes
   * are the case this exists for, where the packed bytes and the fp16 plane
   * they decode to are both resident for the model's whole life and only one of
   * them is ever read again.
   *
   * The numerics of a backend that answers false are untouched by this hook:
   * `Qwen3Model::load` calls nothing on it and no buffer of its moves.  That is
   * what keeps the CPU and CUDA greedy decodes byte-identical to what they were
   * before the hook existed. */
  virtual bool materializes_packed_weights() const { return false; }

  /* Build this backend's own decoded form of a packed weight, so that
   * `release_packed` below may let the checkpoint's bytes go.
   *
   * `rows`/`cols` and `use` are the same numbers and the same consumer the
   * forward path will pass -- a backend that keys its copy on the buffer's
   * handle plus the shape would otherwise miss on the very call the copy was
   * built for, and a miss after the release has no source left to decode.
   * Called only on a backend whose `materializes_packed_weights()` is true, and
   * only before that weight's `release_packed`.  A weight that two consumers
   * read -- a tied embedding, whose bytes are both the gather table and the
   * head's operand -- is materialized once per consumer and released once.
   *
   * May throw; a throw leaves the packed bytes owned by the caller, which is
   * what makes a failed load release exactly the buffers it still holds. */
  virtual void materialize_packed(DeviceBuffer blocks, int64_t rows, int64_t cols, int type_id,
                                  PackedUse use) {
    (void)blocks;
    (void)rows;
    (void)cols;
    (void)type_id;
    (void)use;
  }

  /* Release `blocks`, whose every decoded form the calls above have built.
   *
   * Called after the last `materialize_packed` for a buffer and at most once
   * per buffer, so a backend may record the handle here -- as spent, as this
   * one's Ascend implementation does -- rather than keep the bytes.  The caller
   * keeps the handle value: it is still what the forward path passes and still
   * what a backend keys its copy by, and the model marks the weight released so
   * its destructor does not free it a second time.  Only called on a backend
   * whose `materializes_packed_weights()` is true. */
  virtual void release_packed(DeviceBuffer blocks) { (void)blocks; }

  /* The index of the largest of `n` values, written to device memory. Returns a
   * device address rather than an integer so the caller transfers four bytes
   * instead of the whole logit vector when it only wants the token. */
  virtual void argmax(DeviceBuffer values, int64_t n, DeviceBuffer out) = 0;

  /* The three sampling-stage ops. Unlike everything above, these are a
   * deliberate exception to the rule that the two backends are implemented
   * independently: a sampler is a decision over the whole distribution, it is
   * made on the host by construction (see `kernels.h`), and a second device
   * transcription of the truncation arithmetic would be a second place for the
   * token to differ rather than a check. So the CUDA methods round-trip the
   * logits to the host and call the same `kernel::` functions the CPU backend
   * calls, and the two agree bit for bit.
   *
   * What that costs is stated plainly: `test_the_backends_agree_with_each_other`
   * is vacuous for these three -- it tests the transfer, not the arithmetic.
   * Their correctness is established by the reference comparison instead.
   *
   * `order` is caller-owned scratch of at least `vocab` int64s. */
  virtual void softmax(DeviceBuffer x, DeviceBuffer out, int64_t rows, int64_t cols) = 0;
  virtual void logits_temperature(DeviceBuffer logits, DeviceBuffer out, int64_t n,
                                  float temperature) = 0;
  virtual void topk_sample(DeviceBuffer logits, int64_t vocab, float uniform, int64_t top_k,
                           float top_p, float min_p, DeviceBuffer order, DeviceBuffer out) = 0;

  /* Run everything queued and report any device error. A backend may execute
   * eagerly, in which case this is a no-op; a backend that batches into a graph
   * needs it before a result is read. */
  virtual void synchronize() = 0;

  /* A short description for `pocketllm_open`'s diagnostics -- which device this
   * actually bound to, not which one was asked for. */
  virtual std::string describe() const = 0;
};

/* The backend named `name`. Throws `Error` naming the build's backends if there
 * is none, so a caller that reaches here with a bad name gets a message rather
 * than a null dereference. */
std::unique_ptr<Backend> make_backend(const std::string &name);

/* Throw the same refusal as `make_backend` without creating anything, and do
 * nothing at all when the name is good. The case this exists for is
 * `pocketllm_open` on a checkpoint this build cannot *run*: it still has to
 * reject a device the caller asked for that does not exist, and it wants to do
 * that before mapping a two-gigabyte file rather than after.
 *
 * Whether a backend can be created is a separate question from whether there is
 * a GPU: it depends on how the *library* was built, which is why the answer is a
 * compiled-in table and not a probe. */
void require_backend(const std::string &name);

/* `require_backend` as a predicate, for a caller that wants to branch rather
 * than throw. The two must agree for every name. */
bool backend_available(const std::string &name);

}  // namespace kernel
}  // namespace pocketllm

#endif /* POCKETLLM_KERNEL_BACKEND_H */
