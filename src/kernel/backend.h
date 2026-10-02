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

/* A number that identifies a device buffer. An opaque handle rather than a
 * pointer because on the CPU it is an address and on a device it may not be one
 * -- and because a caller that treats it as a pointer is a caller that has
 * stopped being portable. */
struct DeviceBuffer {
  uintptr_t handle = 0;
  int64_t bytes = 0;
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
                          int64_t m, int64_t n, int64_t k, int type_id, bool accumulate) = 0;

  /* ``out[i, :] = table[token[i], :]``. `tokens` holds `n_tokens` int32 ids. */
  virtual void embedding(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer table, int64_t vocab,
                         int64_t d, DeviceBuffer out) = 0;

  /* The same gather from a packed table -- see `kernels.h`'s `embedding_quant`
   * for why a tied-embedding checkpoint needs it. */
  virtual void embedding_quant(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer blocks,
                               int64_t vocab, int64_t d, int type_id, DeviceBuffer out) = 0;

  virtual void silu_mul(DeviceBuffer gate, DeviceBuffer up, DeviceBuffer out, int64_t n) = 0;

  /* RoPE, split-half layout, as `kernels.h` documents it. */
  virtual void rope_neox(DeviceBuffer x, int64_t n_tokens, int64_t n_heads, int64_t d,
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

  /* Causal grouped-query attention over a cache laid out ``[position][head][d]``,
   * with the query chunk at `q_offset`. `scores` holds what `attention_scratch`
   * asked for. */
  virtual void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                         DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                         int64_t q_offset, float scale, DeviceBuffer out,
                         DeviceBuffer scores) = 0;

  /* The index of the largest of `n` values, written to device memory. Returns a
   * device address rather than an integer so the caller transfers four bytes
   * instead of the whole logit vector when it only wants the token. */
  virtual void argmax(DeviceBuffer values, int64_t n, DeviceBuffer out) = 0;

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