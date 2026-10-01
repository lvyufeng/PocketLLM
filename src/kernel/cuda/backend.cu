/* The CUDA backend: the same operations as `kernel/cpu/backend.cpp`, carried out
 * on one device.
 *
 * One process owns one device. There is no device index here, no stream per
 * rank, no collective and no peer access: this class binds device 0 and the
 * whole graph runs on it. That is the project's rule and not a limitation of
 * this file -- a checkpoint that does not fit is quantized further, never split.
 *
 * The kernels are the *device* implementations and are not shared with the CPU
 * ones in `kernel/kernels.cpp`. That is deliberate. Those functions are the
 * reference this backend is checked against, so their value is that they were
 * written independently and are simpler to read; a shared implementation would
 * agree with itself no matter which of the two were wrong.
 *
 * What the two do share is the arithmetic order, and where they do not, the
 * comment says so. Floating point addition is not associative, so a device
 * kernel that accumulated in a different order would differ from the CPU in the
 * last bits of every dot product, and comparing the two would then be measuring
 * the association order rather than the algorithm.
 *
 * Memory: `DeviceBuffer.handle` is a `cudaMalloc`'d device address. The graph
 * does arithmetic on those handles to address rows and layers -- handle + offset
 * -- which is exactly what a device address supports, and is the reason the
 * handle is an integer rather than a pointer.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kernel/backend.h"
#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

namespace {

/* Every launch is checked. A kernel that failed to launch leaves its output
 * buffer holding the previous token's value, which is a *plausible* production
 * rather than a visible failure -- the logits stay finite and the text stays
 * fluent. Reading `cudaGetLastError` immediately after the launch, before
 * anything could return to the host, is what attributes the error to the kernel
 * that caused it. */
void check(cudaError_t status, const char *what) {
  if (status == cudaSuccess) {
    return;
  }
  const std::string message = std::string("cuda: ") + what + " failed: " + cudaGetErrorString(status);
  /* Cleared as it is read: a sticky error would otherwise be reported again by
   * the next check, blaming a kernel that did nothing wrong. */
  cudaGetLastError();
  throw Error(message);
}

/* ``sum_k a[k] * b[k]``, in the same four-way association the CPU's `dot` uses.
 *
 * The strided read is not the fastest shape a device dot product can take -- a
 * shared-memory reduction over a block would be -- and that is accepted here for
 * the reason the file's header gives: this is the first version, the graph runs
 * through it, and its job is to be correct and comparable. The parallelism that
 * matters is between output elements, of which a projection has tens of
 * thousands. */
__device__ float dot4(const float *a, const float *b, int64_t k) {
  float s0 = 0.0F, s1 = 0.0F, s2 = 0.0F, s3 = 0.0F;
  int64_t i = 0;
  for (; i + 4 <= k; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < k; ++i) {
    s0 += a[i] * b[i];
  }
  return (s0 + s1) + (s2 + s3);
}

/* A block-wide sum, accumulated in index order by thread 0.
 *
 * A tree reduction would be more accurate and would disagree with the CPU's
 * single running sum by more than the association order accounts for -- which
 * would make the CPU-vs-GPU logit comparison report a difference that is not a
 * bug. `scratch` holds `blockDim.x` floats. */
__device__ float block_sum(float value, float *scratch) {
  const int lane = threadIdx.x;
  scratch[lane] = value;
  __syncthreads();
  if (lane == 0) {
    float total = 0.0F;
    for (int i = 0; i < blockDim.x; ++i) {
      total += scratch[i];
    }
    scratch[0] = total;
  }
  __syncthreads();
  return scratch[0];
}

/* A block-wide maximum, for the softmax shift. Order does not matter here --
 * the maximum is the same however it is found -- so this one is a tree. */
__device__ float block_max(float value, float *scratch) {
  const int lane = threadIdx.x;
  scratch[lane] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void fill_kernel(float *dst, float value, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = value;
  }
}

/* One block per row, 256 threads, the row visited in strided passes. `d` is 128
 * or 1024 in this graph, so a row is one to four passes. */
__global__ void rms_norm_kernel(const float *x, const float *weight, float *out, int64_t rows,
                                int64_t d, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float scratch[256];

  const float *src = x + row * d;
  float sum = 0.0F;
  for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
    sum += src[i] * src[i];
  }
  const float total = block_sum(sum, scratch);
  const float scale = 1.0F / sqrtf(total / static_cast<float>(d) + eps);
  float *dst = out + row * d;
  /* The weight is read again per row rather than staged in shared memory: it is
   * one row that is already in L2 after the first block touches it, and staging
   * it would cost a second `__syncthreads` to save a cache hit. */
  for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
    dst[i] = src[i] * scale * weight[i];
  }
}

/* One thread per output element; block `y` is the output row, `x` the columns. */
__global__ void gemm_kernel(const float *x, const float *w, const float *bias, float *out, int64_t m,
                            int64_t n, int64_t k, int accumulate) {
  const int64_t r = blockIdx.y;
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= m || j >= n) {
    return;
  }
  float value = dot4(x + r * k, w + j * k, k);
  if (bias != nullptr) {
    value += bias[j];
  }
  /* `+=` on the residual path, `=` otherwise -- the same two operations the CPU
   * gemm distinguishes. No atomic is needed: each thread owns one output
   * element, so its read-modify-write is private to it. */
  float *dst = out + r * n + j;
  *dst = accumulate ? *dst + value : value;
}

__global__ void embedding_kernel(const int32_t *tokens, int64_t n_tokens, const float *table,
                                 int64_t vocab, int64_t d, float *out) {
  const int64_t t = blockIdx.y;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_tokens || i >= d) {
    return;
  }
  const int32_t id = tokens[t];
  /* An id outside the table zeroes the row rather than skipping it, matching the
   * CPU kernel: a silently different row answers a different question, and a
   * zeroed one shows up in the logits. */
  out[t * d + i] = (id < 0 || id >= vocab) ? 0.0F : table[static_cast<int64_t>(id) * d + i];
}

__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float g = gate[i];
  /* `expf` is called with the value itself, not a clamped one: for a large
   * negative `g` the exponential goes to zero and the result to zero, which is
   * the limit and not an overflow. */
  out[i] = (g / (1.0F + expf(-g))) * up[i];
}

/* One thread per (token, head, pair). The two stores are independent of each
 * other's result -- both read `a` and `b` before either writes -- which is what
 * makes the in-place rotation safe. */
__global__ void rope_neox_kernel(float *x, int64_t n_tokens, int64_t n_heads, int64_t d,
                                 int64_t start_pos, const float *cos_table,
                                 const float *sin_table) {
  const int64_t half = d / 2;
  const int64_t total = n_tokens * n_heads * half;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int64_t i = idx % half;
  const int64_t h = (idx / half) % n_heads;
  const int64_t t = idx / (half * n_heads);
  const int64_t position = start_pos + t;

  float *vec = x + (t * n_heads + h) * d;
  const float c = cos_table[position * half + i];
  const float s = sin_table[position * half + i];
  const float a = vec[i];
  const float b = vec[i + half];
  vec[i] = a * c - b * s;
  vec[i + half] = a * s + b * c;
}

/* One block per (query token, head). The block owns a row of `scores`, and the
 * rows are all `score_stride` wide even though the block's own span may be
 * shorter -- a uniform stride is what lets a block find its row with a multiply
 * instead of a prefix sum.
 *
 * The per-block row is the reason `attention_scratch` is a per-backend query.
 * The CPU reuses one row across every (query, head) pair because it runs them in
 * a loop; here the pairs are concurrent blocks, so a shared row would have two
 * blocks interleaving their scores into one softmax over a mixture of two heads
 * -- wrong, finite, and fluent. */
__global__ void attention_kernel(const float *q, int64_t n_heads, const float *k_cache,
                                 const float *v_cache, int64_t n_head_kv, int64_t d,
                                 int64_t first_key, int64_t q_offset, int64_t score_stride,
                                 float scale, float *out, float *scores) {
  const int64_t t = blockIdx.y;
  const int64_t h = blockIdx.x;
  const int64_t group = n_heads / n_head_kv;
  const int64_t cache_row = n_head_kv * d;

  /* Causal: the query at absolute position `q_offset + t` sees cache rows
   * `first_key .. q_offset + t`. `first_key` is where the cache's live span
   * begins, which is 0 today -- a sliding-window variant would move it and
   * nothing else here would change. */
  const int64_t end = q_offset + t;
  const int64_t span = end - first_key + 1;
  float *row = scores + (t * n_heads + h) * score_stride;

  const float *qvec = q + (t * n_heads + h) * d;
  const int64_t kv_head = h / group;

  __shared__ float scratch[256];

  /* Pass one: the scores. */
  for (int64_t s = threadIdx.x; s < span; s += blockDim.x) {
    row[s] = dot4(qvec, k_cache + (first_key + s) * cache_row + kv_head * d, d) * scale;
  }
  __syncthreads();

  /* Both reductions return the same value to every thread -- `block_max` and
   * `block_sum` leave it in `scratch[0]` and end on a `__syncthreads` -- so the
   * results are locals and not shared variables. A thread with no element in
   * `span` contributes `-INFINITY` to the maximum, which is its identity. */
  float local_max = -INFINITY;
  for (int64_t s = threadIdx.x; s < span; s += blockDim.x) {
    local_max = fmaxf(local_max, row[s]);
  }
  const float max_score = block_max(local_max, scratch);

  /* Pass two: the shifted exponentials, in place, and their sum. Shifted by the
   * max for the same reason the CPU shifts: a score row at a scale of 1000 would
   * otherwise exponentiate to zero everywhere and normalize to 0/0. */
  float local_total = 0.0F;
  for (int64_t s = threadIdx.x; s < span; s += blockDim.x) {
    const float value = expf(row[s] - max_score);
    row[s] = value;
    local_total += value;
  }
  const float inv_total = 1.0F / block_sum(local_total, scratch);

  /* Pass three: the weighted sum of the values, each thread owning its own slice
   * of the head dimension, so no atomics and no second reduction. */
  float *dst = out + (t * n_heads + h) * d;
  for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
    float acc = 0.0F;
    for (int64_t s = 0; s < span; ++s) {
      acc += (row[s] * inv_total) * v_cache[(first_key + s) * cache_row + kv_head * d + i];
    }
    dst[i] = acc;
  }
}

/* One block for the whole vector. The logits are 151936 floats -- 600 KB, a
 * single pass over what is already in L2 -- so a grid-wide reduction would cost
 * more in a second launch than it saves.
 *
 * The tie rule is the ABI's: strictly greater, so the lowest index wins. Every
 * stage of this scan therefore prefers the *earlier* index on an equal value,
 * which is why the comparisons are `>` and the index is compared explicitly
 * rather than the value alone. */
__global__ void argmax_kernel(const float *values, int64_t n, int64_t *out) {
  __shared__ float best_value[256];
  __shared__ int64_t best_index[256];

  float local_value = -INFINITY;
  int64_t local_index = 0;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (values[i] > local_value) {
      local_value = values[i];
      local_index = i;
    }
  }
  best_value[threadIdx.x] = local_value;
  best_index[threadIdx.x] = local_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const bool take_right = best_value[threadIdx.x + stride] > best_value[threadIdx.x] ||
                              (best_value[threadIdx.x + stride] == best_value[threadIdx.x] &&
                               best_index[threadIdx.x + stride] < best_index[threadIdx.x]);
      if (take_right) {
        best_value[threadIdx.x] = best_value[threadIdx.x + stride];
        best_index[threadIdx.x] = best_index[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *out = best_index[0];
  }
}

}  // namespace

/* What follows is the interface. Like `cpu/backend.cpp`, this file is the
 * plumbing that lets the graph be written once; the kernels above are what it
 * drives. */
class CudaBackend final : public Backend {
 public:
  CudaBackend() {
    /* Device 0, and only device 0. A second device in this process would be a
     * second rank, which this project does not have. */
    check(cudaSetDevice(0), "cudaSetDevice(0)");
    check(cudaFree(nullptr), "initializing the CUDA context");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    name_ = prop.name;
    memory_ = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0);
  }

  const char *name() const override { return "cuda"; }

  DeviceBuffer allocate(int64_t bytes) override {
    if (bytes <= 0) {
      throw Error("backend 'cuda': cannot allocate " + std::to_string(bytes) + " bytes");
    }
    void *memory = nullptr;
    check(cudaMalloc(&memory, static_cast<std::size_t>(bytes)), "cudaMalloc");
    return DeviceBuffer{reinterpret_cast<uintptr_t>(memory), bytes};
  }

  void release(DeviceBuffer buffer) override {
    check(cudaFree(reinterpret_cast<void *>(buffer.handle)), "cudaFree");
  }

  void copy_to_device(DeviceBuffer dst, const void *src, int64_t bytes) override {
    check(cudaMemcpy(reinterpret_cast<void *>(dst.handle), src, static_cast<std::size_t>(bytes),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy host->device");
  }

  void copy_to_host(void *dst, DeviceBuffer src, int64_t bytes) override {
    check(cudaMemcpy(dst, reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy device->host");
  }

  void copy_device_to_device(DeviceBuffer dst, DeviceBuffer src, int64_t bytes) override {
    /* A device-to-device `cudaMemcpy`, not a host `memcpy`: both addresses are
     * device addresses, and a `memcpy` on them would either fault or, worse,
     * copy from a host address that happened to be mapped. This is the method
     * whose absence the CPU backend cannot detect. */
    check(cudaMemcpy(reinterpret_cast<void *>(dst.handle),
                     reinterpret_cast<const void *>(src.handle), static_cast<std::size_t>(bytes),
                     cudaMemcpyDeviceToDevice),
          "cudaMemcpy device->device");
  }

  void fill(DeviceBuffer dst, float value) override {
    const int64_t n = dst.bytes / 4;
    if (n <= 0) {
      return;
    }
    fill_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
        reinterpret_cast<float *>(dst.handle), value, n);
    check(cudaGetLastError(), "fill");
  }

  void rms_norm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer out, int64_t n_tokens, int64_t d,
                float eps) override {
    if (n_tokens <= 0) {
      return;
    }
    rms_norm_kernel<<<static_cast<unsigned>(n_tokens), 256>>>(f(x), f(weight), w(out), n_tokens, d,
                                                              eps);
    check(cudaGetLastError(), "rms_norm");
  }

  void gemm(DeviceBuffer x, DeviceBuffer weight, DeviceBuffer bias, DeviceBuffer out, int64_t m,
            int64_t n, int64_t k, bool accumulate) override {
    if (m <= 0 || n <= 0) {
      return;
    }
    const unsigned threads = 256;
    const dim3 grid(static_cast<unsigned>((n + threads - 1) / threads), static_cast<unsigned>(m));
    gemm_kernel<<<grid, threads>>>(f(x), f(weight), bias.handle ? f(bias) : nullptr, w(out), m, n, k,
                                   accumulate ? 1 : 0);
    check(cudaGetLastError(), "gemm");
  }

  void embedding(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer table, int64_t vocab, int64_t d,
                 DeviceBuffer out) override {
    if (n_tokens <= 0 || d <= 0) {
      return;
    }
    const unsigned threads = 256;
    const dim3 grid(static_cast<unsigned>((d + threads - 1) / threads),
                    static_cast<unsigned>(n_tokens));
    embedding_kernel<<<grid, threads>>>(reinterpret_cast<const int32_t *>(tokens.handle), n_tokens,
                                        f(table), vocab, d, w(out));
    check(cudaGetLastError(), "embedding");
  }

  void silu_mul(DeviceBuffer gate, DeviceBuffer up, DeviceBuffer out, int64_t n) override {
    if (n <= 0) {
      return;
    }
    const unsigned threads = 256;
    silu_mul_kernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads>>>(f(gate), f(up),
                                                                                    w(out), n);
    check(cudaGetLastError(), "silu_mul");
  }

  void rope_neox(DeviceBuffer x, int64_t n_tokens, int64_t n_heads, int64_t d, int64_t start_pos,
                 DeviceBuffer cos_table, DeviceBuffer sin_table) override {
    const int64_t total = n_tokens * n_heads * (d / 2);
    if (total <= 0) {
      return;
    }
    const unsigned threads = 256;
    rope_neox_kernel<<<static_cast<unsigned>((total + threads - 1) / threads), threads>>>(
        w(x), n_tokens, n_heads, d, start_pos, f(cos_table), f(sin_table));
    check(cudaGetLastError(), "rope_neox");
  }

  int64_t attention_scratch(int64_t q_len, int64_t n_heads, int64_t max_span) const override {
    /* One row per (query, head) pair, because the pairs are concurrent blocks
     * here. The rows are uniform width -- see `attention` -- so this is the
     * product and not a sum of the pairs' own spans. */
    return q_len * n_heads * max_span * 4;
  }

  void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                 DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                 int64_t q_offset, float scale, DeviceBuffer out, DeviceBuffer scores) override {
    if (q_len <= 0) {
      return;
    }
    /* The stride every block indexes its row with, and it has to be the number
     * `attention_scratch` was asked about. Both sides compute it the same way
     * from the same three arguments -- the last query in the chunk is at
     * `q_offset + q_len - 1`, so the longest visible span is
     * `q_offset + q_len - first_key` -- which is the whole contract. They are
     * not passed to each other because the graph sizes the buffer once per
     * batch, before it knows how many calls it will make; the formula is what
     * keeps the two in step. */
    const int64_t score_stride = q_offset + q_len - first_key;
    const dim3 grid(static_cast<unsigned>(n_heads), static_cast<unsigned>(q_len));
    attention_kernel<<<grid, 256>>>(f(q), n_heads, f(k_cache), f(v_cache), n_head_kv, d, first_key,
                                    q_offset, score_stride, scale, w(out), w(scores));
    check(cudaGetLastError(), "attention");
  }

  void argmax(DeviceBuffer values, int64_t n, DeviceBuffer out) override {
    if (n <= 0) {
      throw Error("backend 'cuda': argmax over an empty vector");
    }
    argmax_kernel<<<1, 256>>>(f(values), n, reinterpret_cast<int64_t *>(w(out)));
    check(cudaGetLastError(), "argmax");
  }

  void synchronize() override { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

  std::string describe() const override {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "cuda:0 (%s, %.0f MiB)", name_.c_str(), memory_);
    return buffer;
  }

 private:
  static const float *f(DeviceBuffer buffer) {
    return reinterpret_cast<const float *>(buffer.handle);
  }
  static float *w(DeviceBuffer buffer) { return reinterpret_cast<float *>(buffer.handle); }

  std::string name_;
  double memory_ = 0.0;
};

std::unique_ptr<Backend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }

}  // namespace kernel
}  // namespace pocketllm