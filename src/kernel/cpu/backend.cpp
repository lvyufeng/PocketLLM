/* The CPU backend: the kernel functions in `kernels.h`, wearing the interface.
 *
 * Nothing here does arithmetic. Every method is a call to the corresponding
 * function in `kernel/kernels.cpp` with the opaque device handles cast back to
 * the host pointers they are. That is the point of the split: the CPU kernels
 * stay readable as the specification of what each operation computes, and this
 * file is only the plumbing that lets the graph be written once.
 *
 * An "allocation" is a `std::vector<float>`, or rather the pointer it owns. A
 * handle is an address, which is why `DeviceBuffer` is a `uintptr_t` rather than
 * a pointer -- the CUDA backend puts an offset in the same field, and the graph
 * treats both as opaque.
 */

#include <cstring>
#include <utility>

#include "kernel/backend.h"
#include "kernel/kernels.h"
#include "kernel/parallel.h"
#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

namespace {

class CpuBackend final : public Backend {
 public:
  const char *name() const override { return "cpu"; }

  DeviceBuffer allocate(int64_t bytes) override {
    if (bytes <= 0) {
      throw Error("backend 'cpu': cannot allocate " + std::to_string(bytes) + " bytes");
    }
    /* `new` rather than a vector so the address is stable and ownership is
     * explicit: `release` is the only place it is freed, and a vector would
     * want to own the lifetime through an object this interface does not have
     * a place for. */
    auto *memory = new float[static_cast<std::size_t>((bytes + 3) / 4)];
    return DeviceBuffer{reinterpret_cast<uintptr_t>(memory), bytes};
  }

  void release(DeviceBuffer buffer) override {
    delete[] reinterpret_cast<float *>(buffer.handle);
  }

  void copy_to_device(DeviceBuffer dst, const void *src, int64_t bytes) override {
    /* On the CPU "to device" and "to host" are the same memcpy in opposite
     * directions, and both are real copies. That is not an inefficiency to
     * optimize away with a shared-pointer scheme: the graph is written as if
     * the two address spaces were distinct, and a CPU backend that aliased them
     * would be a CPU backend that hides a bug the CUDA one would hit.
     *
     * A striped copy was tried here and measured slower, on the theory that
     * spreading a weight's first touch would spread its pages across the NUMA
     * nodes. It does spread them -- over whatever node each pool thread happened
     * to be scheduled on -- and the result is worse than one node: the pages are
     * scattered, the copy pays a wake-up, and the compute threads on the node the
     * pages *were* concentrated on lose more than the rest gain. The measurement
     * is in the PR; the code is the plain copy because that is what it said to
     * keep. */
    std::memcpy(reinterpret_cast<void *>(dst.handle), src, static_cast<std::size_t>(bytes));
  }

  void copy_to_host(void *dst, DeviceBuffer src, int64_t bytes) override {
    std::memcpy(dst, reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes));
  }

  void copy_device_to_device(DeviceBuffer dst, DeviceBuffer src, int64_t bytes) override {
    /* The same memcpy, and worth saying so: on this backend the two address
     * spaces are one, so a backend that conflated the two copy methods would
     * still be correct here. The distinction earns its place on the CUDA side,
     * which is the only place it can be observed. */
    std::memcpy(reinterpret_cast<void *>(dst.handle),
                reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes));
  }

  void fill(DeviceBuffer dst, float value) override {
    float *ptr = reinterpret_cast<float *>(dst.handle);
    const std::size_t n = static_cast<std::size_t>(dst.bytes / 4);
    for (std::size_t i = 0; i < n; ++i) {
      ptr[i] = value;
    }
  }

  void rms_norm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer out, int64_t n_tokens,
                int64_t d, float eps) override {
    kernel::rms_norm(f(x), f(weight), w(out), n_tokens, d, eps);
  }

  void gemm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer bias, DeviceBuffer out, int64_t m,
            int64_t n, int64_t k, bool accumulate) override {
    kernel::gemm(f(x), f(weight), bias.handle ? f(bias) : nullptr, w(out), m, n, k, accumulate);
  }

  void gemm_quant(DeviceBuffer x, DeviceBuffer blocks, DeviceBuffer bias, DeviceBuffer out,
                  int64_t m, int64_t n, int64_t k, int type_id, bool accumulate) override {
    kernel::gemm_quant(f(x), bytes(blocks), bias.handle ? f(bias) : nullptr, w(out), m, n, k, type_id,
                       accumulate);
  }

  void embedding(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer table, int64_t vocab,
                 int64_t d, DeviceBuffer out) override {
    kernel::embedding(reinterpret_cast<const int32_t *>(tokens.handle), n_tokens, f(table), vocab, d,
                      w(out));
  }

  void embedding_quant(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer blocks, int64_t vocab,
                       int64_t d, int type_id, DeviceBuffer out) override {
    kernel::embedding_quant(reinterpret_cast<const int32_t *>(tokens.handle), n_tokens, bytes(blocks),
                            vocab, d, type_id, w(out));
  }

  void silu_mul(DeviceBuffer gate, DeviceBuffer up, DeviceBuffer out, int64_t n) override {
    kernel::silu_mul(f(gate), f(up), w(out), n);
  }

  void rope_neox(DeviceBuffer x, int64_t n_tokens, int64_t n_heads, int64_t d, int64_t start_pos,
                 DeviceBuffer cos_table, DeviceBuffer sin_table) override {
    kernel::rope_neox(w(x), n_tokens, n_heads, d, start_pos, f(cos_table), f(sin_table));
  }

  int64_t attention_scratch(int64_t q_len, int64_t n_heads, int64_t max_span) const override {
    /* One row per concurrent task, not one row for the whole call.  This backend
     * used to run the `(query, head)` pairs in a serial loop and reuse a single
     * row; it now runs them in parallel, so two tasks would otherwise interleave
     * their scores into one softmax over a mixture of two heads -- wrong, finite
     * and fluent, which is exactly what the CUDA backend's version of this
     * comment guards against.  The task count comes from `parallel_tasks` with
     * the grain `attention` uses, so the allocation and the kernel agree by
     * construction rather than by two copies of the same formula. */
    const int64_t tasks = parallel_tasks(q_len * n_heads, kAttentionGrain);
    return (tasks < 1 ? 1 : tasks) * max_span * 4;
  }

  void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                 DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                 int64_t q_offset, float scale, DeviceBuffer out, DeviceBuffer scores) override {
    kernel::attention(w(q), q_len, n_heads, f(k_cache), f(v_cache), n_head_kv, d, first_key,
                      q_offset, scale, w(out), w(scores));
  }

  void argmax(DeviceBuffer values, int64_t n, DeviceBuffer out) override {
    int64_t best = 0;
    kernel::argmax(f(values), n, &best);
    std::memcpy(reinterpret_cast<void *>(out.handle), &best, sizeof(best));
  }

  void softmax(DeviceBuffer x, DeviceBuffer out, int64_t rows, int64_t cols) override {
    kernel::softmax(f(x), w(out), rows, cols);
  }

  void logits_temperature(DeviceBuffer logits, DeviceBuffer out, int64_t n,
                          float temperature) override {
    kernel::logits_temperature(f(logits), w(out), n, temperature);
  }

  void topk_sample(DeviceBuffer logits, int64_t vocab, float uniform, int64_t top_k, float top_p,
                   float min_p, DeviceBuffer order, DeviceBuffer out) override {
    int64_t token = 0;
    kernel::topk_sample(f(logits), vocab, uniform, top_k, top_p, min_p,
                        reinterpret_cast<int64_t *>(w(order)), &token);
    /* The output is an int64 index, not a float: the same width `argmax`
     * writes, so the caller can read either with one `copy_to_host`. */
    std::memcpy(reinterpret_cast<void *>(out.handle), &token, sizeof(token));
  }

  /* The CPU runs as it goes, so there is nothing queued to wait for. The method
   * exists because the graph calls it, and a no-op is the honest answer. */
  void synchronize() override {}

  std::string describe() const override { return "cpu (host memory)"; }

 private:
  /* Two spellings of the same cast, because the compiler cannot pick a function
   * by return type alone: `f` yields the read-only view the inputs want and `w`
   * the writable one the outputs need. Handing a writable pointer to a kernel
   * that only reads is harmless here and would not be on a device, so the split
   * is also what keeps an `out` position from being passed where an `in` is
   * meant. */
  static const float *f(DeviceBuffer buffer) {
    return reinterpret_cast<const float *>(buffer.handle);
  }
  static float *w(DeviceBuffer buffer) { return reinterpret_cast<float *>(buffer.handle); }
  /* A packed weight is bytes, not floats, and the cast is the type: a tensor
   * whose handle was allocated for `nbytes` of blocks is read as `uint8_t*` and
   * never as an array of anything wider. The `f`/`w` split above is about
   * constness; this one is about what the memory *is*. */
  static const uint8_t *bytes(DeviceBuffer buffer) {
    return reinterpret_cast<const uint8_t *>(buffer.handle);
  }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace kernel
}  // namespace pocketllm