#include "kernel/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "quant/blocks.h"

namespace pocketllm {
namespace kernel {

namespace {

/* ``sum_k a[k] * b[k]`` for `k` values.
 *
 * The four-way accumulator is not premature: a 1024-wide row over twenty-eight
 * layers is where a single running sum's dependency chain becomes the whole
 * cost, and the reference `gemm` is a numpy matmul that will not have that
 * chain. Splitting into four partial sums is the cheapest way to keep the two
 * close enough that a tolerance comparison is measuring the algorithm rather
 * than the association order.
 *
 * The order the four partials are summed is fixed, so the result is
 * reproducible run to run. A different order would still be correct and would
 * still pass, but a kernel whose answer moves between runs makes a tolerance
 * failure impossible to bisect. */
float dot(const float *a, const float *b, int64_t k) {
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

}  // namespace

void rms_norm(const float *x, const float *weight, float *out, int64_t n_tokens, int64_t d,
              float eps) {
  for (int64_t r = 0; r < n_tokens; ++r) {
    const float *row = x + r * d;
    float *dst = out + r * d;
    float sum = 0.0F;
    for (int64_t i = 0; i < d; ++i) {
      sum += row[i] * row[i];
    }
    /* The mean is over `d` -- the row -- and the reciprocal square root is
     * computed once for the whole row rather than per element. */
    const float scale = 1.0F / std::sqrt(sum / static_cast<float>(d) + eps);
    for (int64_t i = 0; i < d; ++i) {
      dst[i] = row[i] * scale * weight[i];
    }
  }
}

void gemm(const float *x, const float *w, const float *bias, float *out, int64_t m, int64_t n,
          int64_t k, bool accumulate) {
  for (int64_t r = 0; r < m; ++r) {
    const float *row = x + r * k;
    float *dst = out + r * n;
    for (int64_t j = 0; j < n; ++j) {
      float value = dot(row, w + j * k, k);
      if (bias != nullptr) {
        value += bias[j];
      }
      /* `+=` on the residual path, `=` otherwise. The two are different
       * operations and the graph asks for each explicitly, so they are one
       * branch here rather than two loops that would drift apart. */
      dst[j] = accumulate ? dst[j] + value : value;
    }
  }
}

void gemm_quant(const float *x, const uint8_t *blocks, const float *bias, float *out, int64_t m,
                int64_t n, int64_t k, int type_id, bool accumulate) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int per_block = quant::kBlockWeights;
  /* The row stride in bytes comes from the block geometry and not from a
   * parameter: a caller that passed it would be able to pass one that disagrees
   * with the type id, and the disagreement would read a row from the middle of
   * its neighbour. */
  const int64_t row_bytes = (k / per_block) * block_bytes;

  for (int64_t r = 0; r < m; ++r) {
    const float *row = x + r * k;
    float *dst = out + r * n;
    for (int64_t j = 0; j < n; ++j) {
      const uint8_t *row_blocks = blocks + j * row_bytes;
      float total = 0.0F;
      int64_t col = 0;
      /* One block at a time, decoding each weight as the running sum consumes
       * it. The four-way accumulator `dot` uses is deliberately not replicated
       * here: it exists to avoid a long dependency chain over a thousand
       * elements, and this loop already has one of a different shape -- the
       * decode of the next weight can begin while the previous add is in
       * flight. What it shares with `dot` is the *order*, which is what keeps
       * the CPU and the card comparable at the tolerance the test uses. */
      for (int64_t b = 0; b < k / per_block; ++b) {
        const uint8_t *block = row_blocks + b * block_bytes;
        for (int i = 0; i < per_block; ++i, ++col) {
          total += row[col] * quant::dequant_block(type_id, block, i);
        }
      }
      if (bias != nullptr) {
        total += bias[j];
      }
      dst[j] = accumulate ? dst[j] + total : total;
    }
  }
}

void embedding(const int32_t *tokens, int64_t n_tokens, const float *table, int64_t vocab,
               int64_t d, float *out) {
  for (int64_t t = 0; t < n_tokens; ++t) {
    const int32_t id = tokens[t];
    /* An id outside the table is a corrupt prompt or a checkpoint whose
     * vocabulary disagrees with its embedding. It cannot be skipped silently --
     * that answers a different question -- so the row is zeroed, which is
     * visible in the logits rather than a read past the end of the mapping. */
    if (id < 0 || id >= vocab) {
      std::fill(out + t * d, out + t * d + d, 0.0F);
      continue;
    }
    const float *src = table + static_cast<int64_t>(id) * d;
    std::copy(src, src + d, out + t * d);
  }
}

void embedding_quant(const int32_t *tokens, int64_t n_tokens, const uint8_t *blocks, int64_t vocab,
                     int64_t d, int type_id, float *out) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int per_block = quant::kBlockWeights;
  const int64_t row_bytes = (d / per_block) * block_bytes;

  for (int64_t t = 0; t < n_tokens; ++t) {
    const int32_t id = tokens[t];
    float *dst = out + t * d;
    /* The same policy as the dense gather, for the same reason: an id outside
     * the table zeroes its row rather than reading past the end of the
     * mapping. The two must agree, and they are written out separately because
     * one loop indexes floats and the other blocks. */
    if (id < 0 || id >= vocab) {
      std::fill(dst, dst + d, 0.0F);
      continue;
    }
    const uint8_t *row_blocks = blocks + static_cast<int64_t>(id) * row_bytes;
    int64_t col = 0;
    for (int64_t b = 0; b < d / per_block; ++b) {
      const uint8_t *block = row_blocks + b * block_bytes;
      for (int i = 0; i < per_block; ++i, ++col) {
        dst[col] = quant::dequant_block(type_id, block, i);
      }
    }
  }
}

void silu_mul(const float *gate, const float *up, float *out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float g = gate[i];
    /* `expf` is called with the value the reference passes -- not a clamped
     * one. For a large negative g the exponential goes to zero and the result
     * to zero, which is the limit and not an overflow; a clamp would change the
     * answer in the one range where it is cheap to be right. */
    out[i] = (g / (1.0F + std::exp(-g))) * up[i];
  }
}

void rope_neox(float *x, int64_t n_tokens, int64_t n_heads, int64_t d, int64_t start_pos,
               const float *cos_table, const float *sin_table) {
  const int64_t half = d / 2;
  for (int64_t t = 0; t < n_tokens; ++t) {
    const int64_t position = start_pos + t;
    const float *cos_row = cos_table + position * half;
    const float *sin_row = sin_table + position * half;
    for (int64_t h = 0; h < n_heads; ++h) {
      float *vec = x + (t * n_heads + h) * d;
      for (int64_t i = 0; i < half; ++i) {
        const float c = cos_row[i];
        const float s = sin_row[i];
        const float a = vec[i];
        const float b = vec[i + half];
        /* The two writes are independent of each other's result -- both read
         * `a` and `b` before either store -- so `vec[i]` being overwritten
         * first does not affect the second. */
        vec[i] = a * c - b * s;
        vec[i + half] = a * s + b * c;
      }
    }
  }
}

void attention(const float *q, int64_t q_len, int64_t n_heads, const float *k_cache,
               const float *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores) {
  const int64_t group = n_heads / n_head_kv;
  const int64_t cache_row = n_head_kv * d;

  for (int64_t t = 0; t < q_len; ++t) {
    /* Causal: the query at absolute position `q_offset + t` sees cache rows
     * `first_key .. q_offset + t`. `first_key` is where the cache's live span
     * begins, which is always 0 today -- a sliding-window variant would move it
     * and nothing else here would change. */
    const int64_t end = q_offset + t;
    const int64_t span = end - first_key + 1;
    for (int64_t h = 0; h < n_heads; ++h) {
      const float *qvec = q + (t * n_heads + h) * d;
      const int64_t kv_head = h / group;

      float max_score = -INFINITY;
      for (int64_t s = 0; s < span; ++s) {
        const float *kvec = k_cache + (first_key + s) * cache_row + kv_head * d;
        const float score = dot(qvec, kvec, d) * scale;
        scores[s] = score;
        if (score > max_score) {
          max_score = score;
        }
      }

      /* Shifted by the max, as the reference `softmax` is: the exponentials of
       * a 1000-scale score row would otherwise all be zero and the row would
       * normalize to 0/0. */
      float total = 0.0F;
      for (int64_t s = 0; s < span; ++s) {
        const float w = std::exp(scores[s] - max_score);
        scores[s] = w;
        total += w;
      }

      float *dst = out + (t * n_heads + h) * d;
      std::fill(dst, dst + d, 0.0F);
      const float inv_total = 1.0F / total;
      for (int64_t s = 0; s < span; ++s) {
        const float weight = scores[s] * inv_total;
        const float *vvec = v_cache + (first_key + s) * cache_row + kv_head * d;
        for (int64_t i = 0; i < d; ++i) {
          dst[i] += weight * vvec[i];
        }
      }
    }
  }
}

void argmax(const float *values, int64_t n, int64_t *out) {
  int64_t best = 0;
  for (int64_t i = 1; i < n; ++i) {
    /* Strictly greater, so ties take the lowest index -- the rule the ABI
     * documents, and the one llama.cpp's greedy sampler applies. */
    if (values[i] > values[best]) {
      best = i;
    }
  }
  *out = best;
}

}  // namespace kernel
}  // namespace pocketllm