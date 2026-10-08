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
#include "kernel/kernels.h"
#include "quant/blocks.h"
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

/* One 256-weight super-block added into a running sum, with the block's own
 * scale/min decode hoisted out of the weight loop.
 *
 * This is the packed GEMM's inner kernel and the reason the decode lives here
 * rather than in a per-weight call: `dequant_block` recomputes a Q4_K block's
 * `d`, `dmin` and all eight packed `(scale, min)` pairs -- a branchy
 * `get_scale_min_k4` each -- for *every one* of the 256 weights, so the obvious
 * "one weight at a time" loop does 256 decodes per block where 8 will do. The
 * CPU kernel hoists exactly this (`dot_q4_k_block_scalar`, `kernels.cpp`), and
 * so does this.
 *
 * What is deliberately *not* changed is the arithmetic or its order. The
 * expression per weight is `xs[col] * (sd * q - md)` for Q4_K, where
 * `sd = d * scale` and `md = dmin * minimum` are formed once per group, and
 * `xs[col] * (ds * (q - 32))` for Q6_K. `quant::dequant_q4_k` writes
 * `d * scale * q - dmin * minimum`, which associates left to
 * `(d * scale) * q - (dmin * minimum)` -- the same two products in the same
 * order -- and `dequant_q6_k` writes `(d * scale) * (q - 32)` with the same
 * hoist, so each weight is the float the per-weight decoder produced, bit for
 * bit. The sum is still a single serial `total += ...` in column order, so the
 * accumulation is not reassociated either: this kernel's output is *identical*
 * to the per-weight loop it replaces, which is what lets a performance change
 * land without moving a token.
 *
 * The scale index for Q6_K is `i / 16 + 2 * sub` -- indexed by run as well as by
 * position -- which is the piece a decoder indexing by position alone gets right
 * for one run in four; it is written the way `dequant_q6_k` writes it. */
__device__ void quant_block_accumulate(int type_id, const uint8_t *block, const float *xs,
                                       float &total, bool q6k_repacked) {
  if (q6k_repacked) {
    /* The repacked q6_K block: one `int8` per weight (`q - 32`), then sixteen
     * `int8` scales, then `d`. The per-weight assembly is gone -- a byte load
     * and a scale lookup replace the `ql`/`qh` shift-and-or -- but the value is
     * the one `dequant_q6_k` produced, in the same association, so the sum is
     * unchanged. The scale index `col / 16` is the file decoder's
     * `half * 8 + i / 16 + 2 * sub`, which for a run-major walk is just the
     * 16-weight group's ordinal. */
    const int8_t *qs = reinterpret_cast<const int8_t *>(block);
    const float d = quant::as_half(block, 272);
    for (int col = 0; col < quant::kBlockWeights; ++col) {
      /* The arithmetic is inlined, not routed through
       * `quant::dequant_q6_k_repacked`, for the reason the Q4_K branch inlines
       * its own: a device-function call hides the loop body from nvcc's
       * unroller and the decode runs ~5x slower. The expression is the one that
       * helper writes, so the value is the same bit for bit. */
      const int scale = quant::as_int8(block, 256 + col / 16);
      const float ds = d * static_cast<float>(scale);
      total += xs[col] * (ds * static_cast<float>(static_cast<int>(qs[col])));
    }
    return;
  }
  if (type_id == quant::kGgmlQ4K) {
    const float d = quant::as_half(block, 0);
    const float dmin = quant::as_half(block, 2);
    const uint8_t *scales = block + 4;
    int col = 0;
    for (int g = 0; g < 8; ++g) {
      int scale = 0;
      int minimum = 0;
      quant::get_scale_min_k4(scales, g, &scale, &minimum);
      const float sd = d * static_cast<float>(scale);
      const float md = dmin * static_cast<float>(minimum);
      /* The eight 32-weight groups map onto four 32-byte runs of packed
       * nibbles, low nibble for the earlier group and high for the later one --
       * the same `run`/`high` split `dequant_q4_k` makes, hoisted to the group. */
      const uint8_t *packed = block + 16 + (g / 2) * 32;
      const bool high = (g % 2) != 0;
      /* The run is 32 packed nibble-bytes, and it sits at `block + 16 + n*32`:
       * a Q4_K block is 144 bytes (`= 9 * 16`), so every run starts on a 16-byte
       * boundary and the run is two aligned `uint4` loads. Reading the bytes
       * back out of their eight little-endian words reproduces `as_byte`
       * exactly, so the nibbles and the sum are unchanged -- this trades 32
       * uncoalesced scalar byte loads, whose warp footprint at decode is one
       * used byte per 32-byte sector, for two full-sector vector loads. */
      const uint4 lo = *reinterpret_cast<const uint4 *>(packed);
      const uint4 hi = *reinterpret_cast<const uint4 *>(packed + 16);
      const unsigned word[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      for (int i = 0; i < 32; ++i, ++col) {
        const int q = static_cast<int>((word[i >> 2] >> ((i & 3) * 8)) & 0xFFu);
        const int nibble = high ? (q >> 4) : (q & 0x0F);
        total += xs[col] * (sd * static_cast<float>(nibble) - md);
      }
    }
    return;
  }
  /* Q6_K: two 128-weight halves, four 32-weight runs each, sixteen signed byte
   * scales. The per-weight work is the `ql`/`qh` bit assembly, which stays in
   * the loop; `d * scale` is formed once per 16-weight span. */
  const float d = quant::as_half(block, 208);
  int col = 0;
  for (int half = 0; half < 2; ++half) {
    const uint8_t *ql = block + half * 64;
    const uint8_t *qh = block + 128 + half * 32;
    for (int sub = 0; sub < 4; ++sub) {
      const uint8_t *ql_run = ql + (sub % 2) * 32;
      for (int i = 0; i < 32; ++i, ++col) {
        const int ql_byte = quant::as_byte(ql_run, i);
        const int qh_byte = quant::as_byte(qh, i);
        const int high = ((qh_byte >> (2 * sub)) & 3) << 4;
        const int low = sub < 2 ? (ql_byte & 0x0F) : (ql_byte >> 4);
        const int q = low | high;
        const int scale = quant::as_int8(block, 192 + half * 8 + i / 16 + 2 * sub);
        const float ds = d * static_cast<float>(scale);
        total += xs[col] * (ds * static_cast<float>(q - 32));
      }
    }
  }
}

/* The packed product over a tile of output columns, walking the weight row one
 * super-block at a time.
 *
 * `blockIdx.x` tiles the output columns -- `blockDim.x` of them per block, one
 * thread each -- and `blockIdx.y` is the output row.
 *
 * What a column tile can share is the *activation*, and only the activation:
 * every column of the tile multiplies the same `x` row, so the 256 activation
 * values of a super-block are staged in shared memory once and read by all
 * `blockDim.x` threads, instead of each thread pulling the same 256 floats from
 * L2. The weights are *not* staged, and cannot be: each column owns a different
 * weight row (`blocks + j * row_bytes`), so there is no single weight block to
 * share across the tile. The weight reuse in this kernel is within a thread
 * across `k`, which is what `quant_block_accumulate`'s hoist provides.
 *
 * Staging uses all `blockDim.x` lanes because `blockDim.x == kBlockWeights`; the
 * loop is written to tolerate any `blockDim.x` anyway, for the reason below.
 *
 * The two `__syncthreads` are unconditional. A thread whose column is past `n`
 * (the last, ragged tile) still reaches every barrier and simply skips the
 * accumulate -- a barrier some threads skip is a hang, not a wrong number, and an
 * earlier draft of this kernel that let an inactive lane *write zeros into the
 * staged block* corrupted the real columns sharing that buffer. Only lanes that
 * are loading a valid address write to shared here. */
__global__ void gemm_quant_kernel(const float *x, const uint8_t *blocks, const float *bias,
                                  float *out, int64_t m, int64_t n, int64_t k, int type_id,
                                  int block_bytes, int accumulate, int q6k_repacked) {
  const int64_t r = blockIdx.y;
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x;
  const bool active = (r < m) && (j < n);

  const int64_t n_blk = k / quant::kBlockWeights;
  const int64_t row_bytes = n_blk * block_bytes;
  const float *row = x + r * k;
  /* Guarded so the pointer arithmetic is not formed for an out-of-range column:
   * an inactive thread must not read `blocks` past the end even though it never
   * dereferences the result. */
  const uint8_t *col_blocks = active ? blocks + j * row_bytes : blocks;

  __shared__ float xs[quant::kBlockWeights];

  float total = 0.0F;
  for (int64_t b = 0; b < n_blk; ++b) {
    /* Cooperative stage of this row's activation super-block. Only the lanes
     * that own a real element write, so a `blockDim.x` smaller than the block
     * width cannot have a non-loading lane clobber another's value with a
     * default. */
    for (int i = lane; i < quant::kBlockWeights; i += blockDim.x) {
      xs[i] = row[b * quant::kBlockWeights + i];
    }
    __syncthreads();
    if (active) {
      /* The running sum is passed *into* the block walk rather than returned
       * from it: the sum stays one serial chain across the whole row of `k`
       * weights, exactly as the loop it replaces kept it, so no block boundary
       * reassociates the accumulation. This thread's own weight block -- the
       * column `j` it owns -- is read from global (L2-resident after the first
       * block touches it) and decoded with the hoisted scale/min unpack. */
      quant_block_accumulate(type_id, col_blocks + b * block_bytes, xs, total,
                             q6k_repacked != 0);
    }
    __syncthreads();
  }

  if (!active) {
    return;
  }
  if (bias != nullptr) {
    total += bias[j];
  }
  float *dst = out + r * n + j;
  *dst = accumulate ? *dst + total : total;
}

__global__ void embedding_quant_kernel(const int32_t *tokens, int64_t n_tokens,
                                       const uint8_t *blocks, int64_t vocab, int64_t d, int type_id,
                                       int block_bytes, float *out) {
  const int64_t t = blockIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_tokens || col >= d) {
    return;
  }
  const int32_t id = tokens[t];
  float *dst = out + t * d;
  if (id < 0 || id >= vocab) {
    dst[col] = 0.0F;
    return;
  }
  const int64_t row_bytes = (d / quant::kBlockWeights) * block_bytes;
  const uint8_t *row_blocks = blocks + static_cast<int64_t>(id) * row_bytes;
  const int64_t block = col / quant::kBlockWeights;
  const int within = static_cast<int>(col % quant::kBlockWeights);
  dst[col] = quant::dequant_block(type_id, row_blocks + block * block_bytes, within);
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

  void gemm_quant(DeviceBuffer x, DeviceBuffer blocks, DeviceBuffer bias, DeviceBuffer out,
                  int64_t m, int64_t n, int64_t k, int type_id, bool accumulate,
                  bool q6k_repacked = false) override {
    if (m <= 0 || n <= 0 || k <= 0) {
      return;
    }
    /* A repacked q6_K tensor carries `Q6KRepacked` blocks, not the file's 210
     * bytes, and `block_bytes_of` would answer for the file format. The flag is
     * the only thing that says which, so it also picks the stride. */
    const int block_bytes =
        q6k_repacked ? quant::kQ6KRepackedBytes : quant::block_bytes_of(type_id);
    if (block_bytes == 0) {
      /* Refused by name, and refused *here* rather than at the caller: a build
       * whose device half does not carry a decoder has to say so on the device
       * path, and a caller that caught it earlier would have had to know which
       * formats this particular build compiles. */
      throw Error("backend 'cuda': no packed kernel for GGML type id " + std::to_string(type_id));
    }
    /* A block covers a *tile* of output columns, one thread per column, so that
     * the activation super-block every column multiplies is staged into shared
     * memory once and read by the whole tile rather than re-fetched per column.
     * The weights cannot be shared this way -- each column owns a different
     * weight row -- so the weight reuse is the per-group decode hoist inside
     * `quant_block_accumulate`, not the staging.
     *
     * **32 columns to a block, not 256.** The block size is the decode
     * bottleneck and it is not a tuning detail. `grid.x` is
     * `ceil(n / threads)` and `grid.y` is `m`, so at decode (`m == 1`) a
     * 256-wide block gives a projection only `n / 256` blocks: `o_proj` and
     * `down_proj` (n=2048) become **8 blocks**, and `q/k/v` (n=1024/2048) fewer,
     * on a card with 68 SMs. The GPU was running eight blocks at a time on a
     * 68-SM machine -- 12% occupancy, the rest of the card idle -- which is why
     * the kernel moved weights at ~2% of HBM and why decode was ~100 ms/token.
     * At 32 columns the same projection is 64 blocks and the card is filled.
     *
     * Measured on Qwen3-1.7B-Q4_K_M (3 reps, interleaved, same host):
     *
     *     columns/block   pp512 t/s   tg128 t/s
     *          256          40.83        9.92     <- the old value
     *           32          41.61       24.20
     *
     * Decode is 2.44x faster, and prefill is unchanged, because prefill's `m`
     * is already large enough to fill the grid and never depended on `grid.x`.
     * The kernel still moves 1.10 GB of weights per token (1.276 GB of model
     * minus the gathered embedding table), so decode bandwidth went from
     * 10.9 GB/s to 26.6 GB/s -- 1.8% to 4.3% of the card's ~616 GB/s HBM peak.
     * The win therefore comes entirely from occupying more of the card, not
     * from touching memory more efficiently; the kernel remains far from the
     * HBM roofline. It is a pure launch-geometry change: every thread still
     * computes exactly one output column with the same single serial
     * accumulation chain, so the output is bit-identical and no token moves.
     *
     * 32 is the value that measured best. The win comes from having enough
     * blocks, and it falls off as the block widens again -- an earlier run gave
     * tg128 16.5 t/s at 64 and 13.8 at 128, against 24.2 at 32 -- so 256 is not
     * the only bad value, just the worst one measured. */
    const unsigned threads = 32;
    const dim3 grid(static_cast<unsigned>((n + threads - 1) / threads), static_cast<unsigned>(m));
    gemm_quant_kernel<<<grid, threads>>>(f(x), reinterpret_cast<const uint8_t *>(blocks.handle),
                                         bias.handle ? f(bias) : nullptr, w(out), m, n, k, type_id,
                                         block_bytes, accumulate ? 1 : 0, q6k_repacked ? 1 : 0);
    check(cudaGetLastError(), "gemm_quant");
  }

  void embedding_quant(DeviceBuffer tokens, int64_t n_tokens, DeviceBuffer blocks, int64_t vocab,
                       int64_t d, int type_id, DeviceBuffer out) override {
    if (n_tokens <= 0 || d <= 0) {
      return;
    }
    const int block_bytes = quant::block_bytes_of(type_id);
    if (block_bytes == 0) {
      throw Error("backend 'cuda': no packed kernel for GGML type id " + std::to_string(type_id));
    }
    const unsigned threads = 256;
    const dim3 grid(static_cast<unsigned>((d + threads - 1) / threads),
                    static_cast<unsigned>(n_tokens));
    embedding_quant_kernel<<<grid, threads>>>(reinterpret_cast<const int32_t *>(tokens.handle),
                                              n_tokens,
                                              reinterpret_cast<const uint8_t *>(blocks.handle),
                                              vocab, d, type_id, block_bytes, w(out));
    check(cudaGetLastError(), "embedding_quant");
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

  /* The card keeps an f32 cache.  `attention_kernel` reads `const float *` and
   * widening an f16 cache on the device would be a third kernel to keep in step
   * for a byte count that is not this backend's bottleneck -- the card is
   * nowhere near bandwidth-bound at this model size, so the trade the CPU
   * backend makes (half the bytes for a rounded element) buys nothing here and
   * costs precision.  Declaring it is what keeps the graph honest: it asks,
   * rather than assuming, and nothing in `qwen3.cpp` names a width. */
  KVDtype preferred_kv_dtype() const override { return KVDtype::kF32; }

  /* The card never takes the f16 arm -- `preferred_kv_dtype` is f32 -- so this
   * is the device-to-device copy the graph asks for and nothing else.  The
   * conversion is refused rather than implemented so that a future change that
   * flips the preferred dtype fails here, at the first token, with a name
   * instead of quietly storing the wrong layout. */
  void kv_append(DeviceBuffer dst, DeviceBuffer src, int64_t n, int64_t n_head_kv, int64_t d,
                 int64_t elem) override {
    if (elem != 4) {
      throw Error("cuda kv_append: the cache is f32 here -- see preferred_kv_dtype");
    }
    (void)n_head_kv;
    (void)d;
    check(cudaMemcpy(reinterpret_cast<void *>(dst.handle),
                     reinterpret_cast<const void *>(src.handle),
                     static_cast<std::size_t>(n) * static_cast<std::size_t>(n_head_kv * d) * 4,
                     cudaMemcpyDeviceToDevice),
          "kv_append");
  }

  void attention(DeviceBuffer q, int64_t q_len, int64_t n_heads, DeviceBuffer k_cache,
                 DeviceBuffer v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
                 int64_t q_offset, float scale, DeviceBuffer out, DeviceBuffer scores,
                 KVDtype kv_dtype) override {
    if (q_len <= 0) {
      return;
    }
    if (kv_dtype != KVDtype::kF32) {
      throw Error("cuda attention: the cache is f32 here -- see preferred_kv_dtype");
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

  /* The sampling stage, on the host by design -- see the note in `backend.h`.
   *
   * The logits are read back, the host kernel runs, and only the four-byte
   * result returns to the device. That is a full-vocabulary transfer per token,
   * which `Session::forward` already pays to hand the logits to its caller, so
   * this adds no round trip the decode was not already making. */
  void softmax(DeviceBuffer x, DeviceBuffer out, int64_t rows, int64_t cols) override {
    const int64_t n = rows * cols;
    std::vector<float> host(static_cast<std::size_t>(n));
    std::vector<float> result(static_cast<std::size_t>(n));
    copy_to_host(host.data(), x, n * 4);
    kernel::softmax(host.data(), result.data(), rows, cols);
    copy_to_device(out, result.data(), n * 4);
  }

  void logits_temperature(DeviceBuffer logits, DeviceBuffer out, int64_t n,
                          float temperature) override {
    std::vector<float> host(static_cast<std::size_t>(n));
    std::vector<float> result(static_cast<std::size_t>(n));
    copy_to_host(host.data(), logits, n * 4);
    kernel::logits_temperature(host.data(), result.data(), n, temperature);
    copy_to_device(out, result.data(), n * 4);
  }

  void topk_sample(DeviceBuffer logits, int64_t vocab, float uniform, int64_t top_k, float top_p,
                   float min_p, DeviceBuffer order, DeviceBuffer out) override {
    std::vector<float> host(static_cast<std::size_t>(vocab));
    std::vector<int64_t> ranked(static_cast<std::size_t>(vocab));
    copy_to_host(host.data(), logits, vocab * 4);
    int64_t token = 0;
    kernel::topk_sample(host.data(), vocab, uniform, top_k, top_p, min_p, ranked.data(), &token);
    /* `order` is scratch the caller allocated and nothing reads it back, so it
     * is not copied to the device; the token is what the caller wanted. */
    copy_to_device(out, &token, sizeof(token));
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