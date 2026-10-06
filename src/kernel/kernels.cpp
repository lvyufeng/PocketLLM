#include "kernel/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#if defined(__AVX2__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define POCKETLLM_HAVE_AVX2 1
#else
#define POCKETLLM_HAVE_AVX2 0
#endif

/* The f16 conversion the cache needs.  AVX2 implies F16C on every part that has
 * both, but the two are separate CPUID bits and the guard is written for what
 * is actually used rather than what usually comes with it -- an AVX2 build
 * without F16C would otherwise compile `_mm_cvtph_ps` into an illegal
 * instruction, which is a SIGILL at the first token rather than a build error. */
#if defined(__F16C__) && (defined(__x86_64__) || defined(__i386__))
#define POCKETLLM_HAVE_F16C 1
#else
#define POCKETLLM_HAVE_F16C 0
#endif

#include "kernel/parallel.h"
#include "quant/blocks.h"
#include "quant/q8k.h"
#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

namespace {

/* Must `attention`'s weighted sum walk one row at a time?
 *
 * The same kind of switch `$POCKETLLM_CPU_SCALAR_DOT` and
 * `$POCKETLLM_CPU_EXACT_GEMM` are, and for the same reason: "the tiling gives
 * the same number" is a claim that gets *checked* rather than asserted.  Unlike
 * the score dot's four lanes this one was not *required* to be bit-exact -- the
 * weighted sum is the last accumulation of the attention output, and the
 * tolerance tests plus the llama.cpp token match are what it has to hold -- and
 * it turned out to be bit-exact anyway, because the tiling moves *which row* is
 * being accumulated between two loads of the same V vector and not the order
 * within any row's own sum.  This switch is what keeps that measured rather
 * than incidental.
 *
 * `tests/native/test_cpu_parallel.py` runs one attention call each way and
 * compares the output bytes.  A function-local static so the running process
 * cannot change it and the hot loop pays a predicted branch rather than a
 * `getenv` per call. */
bool scalar_weighted_sum_forced() {
  static const bool forced = [] {
    const char *from_env = std::getenv("POCKETLLM_CPU_SCALAR_VSUM");
    return from_env != nullptr && from_env[0] != '\0' && from_env[0] != '0';
  }();
  return forced;
}

/* How many activation rows `gemm_quant` walks a weight row with, 4 or 8.
 *
 * **8 is the shipped answer and this is the escape hatch back to 4.**  An
 * environment variable rather than a constant because the two row counts are
 * bit-identical to each other (see `dot_Rrows_q8k`, and
 * `tests/native/test_cpu_parallel.py` holds it to the bytes) and differ only in
 * speed -- so the switch is how that claim stays checkable and how the row count
 * is re-measured on a different host without an edit.  Read once, on first use:
 * the answer cannot change under a running process, and the per-call `getenv`
 * would be a lock on the token path. */
int64_t gemm_rows_per_walk() {
  static const int64_t rpw = [] {
    const char *from_env = std::getenv("POCKETLLM_CPU_GEMM_RPW");
    if (from_env == nullptr) {
      return int64_t{8};
    }
    const long parsed = std::strtol(from_env, nullptr, 10);
    return parsed == 4 ? int64_t{4} : int64_t{8};
  }();
  return rpw;
}

/* The widest head dimension the tiled weighted sum holds on the stack.
 *
 * `attention`'s last pass keeps one accumulator row per tile row while it walks
 * the V rows, so the buffer is ``kAttentionRows * d`` floats.  It is a fixed
 * bound rather than a `std::vector` because this is the token path and the
 * allocation would be per call; a `d` past it takes the untiled loop, which is
 * the shipped code and gives the same answer more slowly.  Qwen3's head
 * dimension is 128, so this is two of them.
 *
 * It is deliberately *outside* the `POCKETLLM_HAVE_AVX2` block the integer
 * kernels live in: `attention` has a scalar path too, and a constant only an
 * AVX2 build can see would make a non-AVX2 build reference an undeclared name.
 * The rule is the same one `kAttentionRows` follows in the header -- shared with
 * the device-agnostic kernel, not with one instruction set. */
constexpr int64_t kAttentionSumMaxDim = 256;

/* An f16 bit pattern as a float.
 *
 * Exact for every input, including subnormals, infinities and NaN, which is why
 * it is bit arithmetic rather than a lookup: an f16 is an f32 whose exponent is
 * biased by 112 instead of 127, so widening is a shift and, for the subnormal
 * case, a rescale. `_mm_cvtph_ps` does this in hardware where F16C is available,
 * and this is the scalar path beside it and the reference for the tail. */
inline float half_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
  const uint32_t exp = (bits >> 10) & 0x1FU;
  const uint32_t man = bits & 0x3FFU;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign; /* +-0 */
    } else {
      /* Subnormal: normalize by shifting the mantissa up until its top bit is
       * set, the exponent following it. */
      uint32_t m = man;
      uint32_t e = 113U;
      while ((m & 0x400U) == 0) {
        m <<= 1;
        --e;
      }
      out = sign | (e << 23) | ((m & 0x3FFU) << 13);
    }
  } else if (exp == 0x1FU) {
    out = sign | 0x7F800000U | (man << 13); /* inf / NaN */
  } else {
    out = sign | ((exp + 112U) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

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

/* The same sum, four lanes at a time, with the *same* four accumulator chains.
 *
 * This exists for `attention` and it is worth being precise about why, because
 * writing a second dot is otherwise the kind of duplication that drifts.
 *
 * `attention` calls a dot once per `(query, head)` unit per cache row, and at a
 * 512-token context that is 16 units x 512 rows x 28 layers of 128-wide dots per
 * token -- 448 MiB of streamed K per decode step, against a machine whose one
 * core reads at a few GB/s.  The scalar loop above is correct and is the wrong
 * shape for it: at `k = 128` the loop runs 32 times and the four partials plus
 * the horizontal sum are the whole function.
 *
 * `s0..s3` are four independent accumulator chains and `i` advances by four, so
 * lanes 0..3 of a `__m128` *are* `s0..s3` and one `_mm_loadu_ps` plus one vector
 * `mul`/`add` pair does the four scalar statements.  The horizontal reduce below
 * is `(s0 + s1) + (s2 + s3)` in that order, over the same lane values.
 *
 * **The exactness is real, and finding it out took three tries because the
 * obvious reading of the disassembly is wrong.**  The vector form below is
 * `_mm_mul_ps` then `_mm_add_ps`, deliberately *not* `_mm_fmadd_ps`: GCC
 * contracts the scalar loop's `s += a[i] * b[i]` only when the multiply has a
 * single use, which the four separate tails deny it, so the shipped `dot`
 * rounds each product before adding it -- and an unconditional FMA does not.
 * An earlier version of this function used `_mm_fmadd_ps` and moved the
 * answer: over 200000 random 128-wide pairs the fused form disagrees with the
 * shipped `dot` on 135170 of them and this one on **zero**.  A bit-level
 * comparison is what settled it; the instruction mix in `objdump` is not
 * evidence about rounding, because the same source emits different contraction
 * decisions in different contexts.
 *
 * So `attention` gets the four lanes it could always have had without touching
 * a single result: the score rows, the softmax and the output are identical to
 * the pre-change engine's, which is why the llama.cpp token-for-token match
 * (32/32 and 64/64 on the recorded prompts) survives unchanged.
 *
 * `k` is 128 in every attention call the graph makes, but the tail loops are
 * kept so the function is not a trap for a shape that is not a multiple of four.
 *
 * `$POCKETLLM_CPU_SCALAR_DOT` forces `attention` back onto the scalar `dot`,
 * which is how the claim above is *checked* rather than asserted: one attention
 * call each way, compared byte for byte.  See the switch below.
 */
#if POCKETLLM_HAVE_AVX2

/* Must `attention` take the scalar `dot` instead of the four-lane one below?
 *
 * The same kind of switch `$POCKETLLM_CPU_EXACT_GEMM` is, and for the same
 * reason: "the two forms compute the same number" is a claim that gets checked
 * rather than asserted, and the only way to check it through the public API is
 * to be able to ask for the other form.  `tests/native/test_cpu_parallel.py`
 * runs one attention call each way and compares the output bytes.
 *
 * It is not a preference knob.  A four-lane dot whose rounding differs from
 * `dot`'s is not a slightly different answer -- the softmax turns a last-bit
 * score difference into a different token sequence, measured at 32/32 to 1/32
 * on the recorded prompt when this function was written with `_mm_fmadd_ps`.
 * A function-local static so the running process cannot change it and the hot
 * loop pays a predicted branch rather than a `getenv` per cache row. */
bool scalar_dot_forced() {
  static const bool forced = [] {
    const char *from_env = std::getenv("POCKETLLM_CPU_SCALAR_DOT");
    return from_env != nullptr && from_env[0] != '\0' && from_env[0] != '0';
  }();
  return forced;
}

__attribute__((optimize("fp-contract=off"))) inline float dot4(const float *a, const float *b,
                                                               int64_t k) {
  if (scalar_dot_forced()) {
    return dot(a, b, k);
  }
  __m128 acc = _mm_setzero_ps();
  int64_t i = 0;
  for (; i + 4 <= k; i += 4) {
    acc = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)), acc);
  }
  float lanes[4];
  _mm_storeu_ps(lanes, acc);
  float tail = 0.0F;
  for (; i < k; ++i) {
    tail += a[i] * b[i];
  }
  return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + tail;
}
#else
inline float dot4(const float *a, const float *b, int64_t k) { return dot(a, b, k); }
#endif

/* Two query heads that share one KV head, scored against one key vector in a
 * single walk over that vector.
 *
 * Attention is grouped: `n_heads / n_head_kv` query heads read each KV head's
 * K and V rows.  The score pass walks the key row once *per query head*, so a
 * KV head's K row is loaded as many times as it has query heads -- twice for
 * Qwen3's 16/8 split, four times for a 4-group model.  Scoring two of them
 * together halves that traffic, and the K row is what the pass is bound by:
 * `d` floats are used out of a `n_head_kv * d` stride, so the walk streams
 * 4096 bytes to read 512 per head and the second read of the same row is not
 * free the way it would be if the row were contiguous.
 *
 * **The two chains are `dot4`'s, not a wider one.**  One `__m256` holds query
 * `h0`'s four lanes in the low half and query `h1`'s in the high half, each
 * advanced by four at exactly the offsets `dot4` loads and combined with the
 * same `_mm_add_ps(_mm_mul_ps(...))` -- never a fused multiply-add.  Every lane
 * is a `dot4` lane and each result is reduced `((l0 + l1) + (l2 + l3)) + tail`,
 * so the two outputs are bit-identical to two `dot4` calls.  That is the
 * property the score dot is held to and the reason the pairing is expressed as
 * one register's two halves rather than as two registers: an independent chain
 * per query would be the same arithmetic but a different register assignment,
 * and it is *this* function's job to be indistinguishable from `dot4`.
 *
 * `$POCKETLLM_CPU_SCALAR_DOT` routes it to two scalar `dot` calls, like `dot4`
 * and `dot_tile_r`, so the equivalence stays checkable through the public API. */
#if POCKETLLM_HAVE_AVX2

__attribute__((optimize("fp-contract=off"))) inline void flash_fold(float *P, float x,
                                                                   const float *vvec, int64_t d) {
#if POCKETLLM_HAVE_AVX2
  if (x > P[0]) {
    const float corr = std::exp(P[0] - x);
    P[1] = P[1] * corr + 1.0F;
    const __m256 c = _mm256_set1_ps(corr);
    int64_t z = 0;
    for (; z + 8 <= d; z += 8) {
      _mm256_storeu_ps(P + 2 + z, _mm256_add_ps(
          _mm256_mul_ps(_mm256_loadu_ps(P + 2 + z), c), _mm256_loadu_ps(vvec + z)));
    }
    for (; z < d; ++z) {
      P[2 + z] = P[2 + z] * corr + vvec[z];
    }
    P[0] = x;
  } else {
    const float w = std::exp(x - P[0]);
    P[1] += w;
    const __m256 wv = _mm256_set1_ps(w);
    int64_t z = 0;
    for (; z + 8 <= d; z += 8) {
      _mm256_storeu_ps(P + 2 + z, _mm256_add_ps(
          _mm256_mul_ps(wv, _mm256_loadu_ps(vvec + z)), _mm256_loadu_ps(P + 2 + z)));
    }
    for (; z < d; ++z) {
      P[2 + z] += w * vvec[z];
    }
  }
#else
  if (x > P[0]) {
    const float corr = std::exp(P[0] - x);
    P[1] = P[1] * corr + 1.0F;
    for (int64_t z = 0; z < d; ++z) {
      P[2 + z] = P[2 + z] * corr + vvec[z];
    }
    P[0] = x;
  } else {
    const float w = std::exp(x - P[0]);
    P[1] += w;
    for (int64_t z = 0; z < d; ++z) {
      P[2 + z] += w * vvec[z];
    }
  }
#endif
}

__attribute__((optimize("fp-contract=off"))) inline void dot_pair(const float *qa,
                                                                  const float *qb,
                                                                  const float *key, int64_t k,
                                                                  float out[2]) {
  if (scalar_dot_forced()) {
    out[0] = dot(qa, key, k);
    out[1] = dot(qb, key, k);
    return;
  }
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 4 <= k; i += 4) {
    const __m128 kk = _mm_loadu_ps(key + i);
    /* The key's four values in both halves: each query's low half sees exactly
     * the vector `dot4` would load at this offset. */
    const __m256 kv = _mm256_insertf128_ps(_mm256_castps128_ps256(kk), kk, 1);
    const __m256 a = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(qa + i)),
                                          _mm_loadu_ps(qb + i), 1);
    acc = _mm256_add_ps(_mm256_mul_ps(a, kv), acc);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  const float *src[2] = {lanes, lanes + 4};
  const float *q[2] = {qa, qb};
  for (int r = 0; r < 2; ++r) {
    float tail = 0.0F;
    for (int64_t j = i; j < k; ++j) {
      tail += q[r][j] * key[j];
    }
    out[r] = ((src[r][0] + src[r][1]) + (src[r][2] + src[r][3])) + tail;
  }
}

#else
inline void dot_pair(const float *qa, const float *qb, const float *key, int64_t k, float out[2]) {
  out[0] = dot(qa, key, k);
  out[1] = dot(qb, key, k);
}
#endif

/* ``R`` query rows against one key vector, one `dot4` per row.
 *
 * The score pass is `q_len * span` dots over `span` key vectors, so the key row
 * is loaded `q_len / R` times per head; the call is bandwidth-bound, so the
 * load count *is* the cost.  `R` is `kAttentionRows` at the one call site; it is
 * a template parameter rather than a constant so a caller that wanted a
 * different block for a different reason -- the weighted sum below, say -- could
 * ask for one without a second copy of this code.
 *
 * **The lane structure is the whole design, and reading the code as "just R
 * accumulators" is how a last-bit change gets in.**  `dot4` is one `__m128`
 * chain advancing by four, reduced `((l0 + l1) + (l2 + l3)) + tail`.  A
 * query-row kernel that gave each row its own `__m256` would have eight lanes
 * per row and a different reduce -- a different last bit, and the score-dot
 * section on the C engine page records what a last bit costs here: the FMA
 * variant of `dot4` moved the model's greedy completion from 32/32 tokens
 * matching llama.cpp to 1/32, because the softmax turns a score's last bit into
 * a different argmax.
 *
 * So each `__m256` accumulator holds *two* queries' four-lane chains: rows
 * `2p` and `2p+1` in the low and high halves of `acc[p]`, loaded at the offsets
 * `dot4` would load them, combined with the same `_mm_add_ps(_mm_mul_ps(...))`
 * and never a fused multiply-add.  Every lane is `dot4`'s lane; only which
 * queries share a register has changed.  **And the row count is not part of
 * that argument**: rows are paired, and a pair's arithmetic does not know how
 * many other pairs exist, so `R = 8` produces the same bytes a row at a time
 * would and so would any other even `R`.  What an even larger `R` does cost is
 * registers -- one live `__m256` per two rows -- and `kAttentionRows` is 8
 * because 16 is where that shows up.
 *
 * 128 is the head width the graph calls this with, but the tail loops are kept
 * so the function is not a trap for a width that is not a multiple of four.
 * `$POCKETLLM_CPU_SCALAR_DOT` routes it to scalar `dot` calls, like `dot4`, so
 * the equivalence stays checkable through the public API. */
#if POCKETLLM_HAVE_AVX2

template <int64_t R>
__attribute__((optimize("fp-contract=off"))) inline void dot_tile_r(const float *const *q,
                                                                    const float *key, int64_t k,
                                                                    float out[R]) {
  static_assert(R % 2 == 0, "dot_tile_r holds two query rows per __m256");
  if (scalar_dot_forced()) {
    for (int64_t p = 0; p < R; ++p) {
      out[p] = dot(q[p], key, k);
    }
    return;
  }
  /* One 256-bit register per *pair* of queries, each carrying the pair's two
   * `__m128` chains. */
  __m256 acc[R / 2];
  for (int64_t p = 0; p < R / 2; ++p) {
    acc[p] = _mm256_setzero_ps();
  }
  int64_t i = 0;
  for (; i + 4 <= k; i += 4) {
    const __m128 kk = _mm_loadu_ps(key + i);
    /* The same four key values in both halves, so each query's low half sees
     * exactly the vector `dot4` would load at this offset. */
    const __m256 kv = _mm256_insertf128_ps(_mm256_castps128_ps256(kk), kk, 1);
    for (int64_t p = 0; p < R / 2; ++p) {
      const __m256 a = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(q[2 * p] + i)),
                                            _mm_loadu_ps(q[2 * p + 1] + i), 1);
      acc[p] = _mm256_add_ps(_mm256_mul_ps(a, kv), acc[p]);
    }
  }
  alignas(32) float lanes[R / 2][8];
  for (int64_t p = 0; p < R / 2; ++p) {
    _mm256_store_ps(lanes[p], acc[p]);
  }
  for (int64_t p = 0; p < R; ++p) {
    /* Row `p`'s chain is `p % 2`'s half of pair `p / 2`. */
    const float *l = lanes[p / 2] + (p % 2) * 4;
    float tail = 0.0F;
    for (int64_t j = i; j < k; ++j) {
      tail += q[p][j] * key[j];
    }
    out[p] = ((l[0] + l[1]) + (l[2] + l[3])) + tail;
  }
}

#else

template <int64_t R>
inline void dot_tile_r(const float *const *q, const float *key, int64_t k, float out[R]) {
  for (int64_t p = 0; p < R; ++p) {
    out[p] = dot(q[p], key, k);
  }
}

#endif

/* ``sum_i row[i] * dequant_q4_k(block, i)`` for one Q4_K super-block.
 *
 * The arithmetic is `dequant_q4_k`'s, term for term and in the same order, so
 * the sum is bit-identical to walking `quant::dequant_block` 256 times and
 * adding. What it saves is the *decode*: a Q4_K block carries one `d`, one
 * `dmin` and eight packed (scale, min) pairs, and the obvious loop recomputes
 * all of them for every one of the 256 weights -- a branchy `get_scale_min_k4`
 * call 256 times where eight will do. Here they are hoisted to the group.
 *
 * `d * sc` is formed once per group rather than per weight. That is the same
 * float as the `d * sc` inside `d * sc * q`, because the multiplication
 * associates left and both spell `(d * sc) * q` -- which is what makes the
 * hoist exact rather than merely close. */
/* Kept even where an AVX2 build never calls it: it is the portable fallback on
 * any other host, and it is the readable statement of what the vector path must
 * compute.  `[[maybe_unused]]` rather than a preprocessor guard because both
 * versions should keep compiling and keep being type-checked on every build. */
[[maybe_unused]] float dot_q4_k_block_scalar(const float *row, const uint8_t *block) {
  const float d = quant::as_half(block, 0);
  const float dmin = quant::as_half(block, 2);
  const uint8_t *scales = block + 4;

  float total = 0.0F;
  int col = 0;
  for (int g = 0; g < 8; ++g) {
    int scale = 0;
    int minimum = 0;
    quant::get_scale_min_k4(scales, g, &scale, &minimum);
    const float sd = d * static_cast<float>(scale);
    const float md = dmin * static_cast<float>(minimum);
    /* The eight 32-weight groups map onto four 32-byte runs of packed nibbles,
     * low nibble for the earlier group and high for the later one -- the same
     * `run`/`high` split `dequant_q4_k` makes, hoisted to the group. */
    const uint8_t *packed = block + 16 + (g / 2) * 32;
    const bool high = (g % 2) != 0;
    for (int i = 0; i < 32; ++i, ++col) {
      const int q = quant::as_byte(packed, i);
      const int nibble = high ? (q >> 4) : (q & 0x0F);
      total += row[col] * (sd * static_cast<float>(nibble) - md);
    }
  }
  return total;
}

/* The Q6_K counterpart: 256 weights, two 128-weight halves of four 32-weight
 * runs, sixteen signed byte scales. As above the decode is hoisted -- here the
 * per-weight work is the `ql`/`qh` bit assembly, which stays in the loop, while
 * `d * scale` is formed once per 16-weight span. `(d * scale) * (q - 32)` is
 * the expression `dequant_q6_k` writes, associated the same way. */
[[maybe_unused]] float dot_q6_k_block_scalar(const float *row, const uint8_t *block) {
  const float d = quant::as_half(block, 208);
  float total = 0.0F;
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
        /* `i / 16 + 2 * sub` indexes the sixteen scales of this half by run as
         * well as by position -- the piece a decoder indexing by position alone
         * gets right for one run in four. */
        const int scale = quant::as_int8(block, 192 + half * 8 + i / 16 + 2 * sub);
        const float ds = d * static_cast<float>(scale);
        total += row[col] * (ds * static_cast<float>(q - 32));
      }
    }
  }
  return total;
}

#if POCKETLLM_HAVE_AVX2

/* The AVX2 GEMM's packed dots, written per *row* rather than per block.
 *
 * The scalar `dot_q4_k_block_scalar` returns one super-block's contribution and
 * the GEMM sums those; that shape is what lets the vertical code stay simple,
 * and it is exactly wrong for the vector one.  A per-block dot ends in a
 * horizontal reduce, and a `k / 256`-long loop of reduces spends longer
 * assembling eight lanes than it does multiplying, while a single 256-weight
 * accumulator is a chain of dependent FMAs that never fills the pipe.  Folding
 * the blocks into lane-wise accumulators over the whole row removes the first,
 * and splitting them into two independent chains removes the second: the reduce
 * happens once per output element instead of once per 256 weights, and while one
 * chain's result is in flight the other issues.
 *
 * This is why the row functions take `k` and walk the blocks themselves.  The
 * *decode* is unchanged and stays exact -- every weight is the weight the scalar
 * decoder produces -- and only the association of the sum moves, which is the
 * difference the plan accepted when the build gained `-march=native`. */

namespace {

/* Eight packed weight bytes widened to eight float lanes.
 *
 * `_mm256_cvtepu8_epi32` takes the eight low bytes of a 128-bit register to
 * eight int32 lanes *in order*, so lane `i` is byte `i` and the row beside it
 * can be read in natural order -- which is what makes the row reads in the two
 * kernels below plain `_mm256_loadu_ps`. */
inline __m256 unpack_nibbles8(const uint8_t *packed, bool high) {
  const __m256i bytes =
      _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed)));
  const __m256i nibbles =
      high ? _mm256_and_si256(_mm256_srli_epi32(bytes, 4), _mm256_set1_epi32(0x0F))
           : _mm256_and_si256(bytes, _mm256_set1_epi32(0x0F));
  return _mm256_cvtepi32_ps(nibbles);
}

/* ``sum_i r[i] * (sd * q_i - md)`` for one 32-weight group, accumulated into
 * `*acc`.  `sd` and `md` are the group's pre-scaled scale and offset, which the
 * caller has already multiplied by the block's `d` and `dmin`; broadcasting them
 * once and folding ``(r * sd) * q - (r * md)`` leaves four eight-lane passes for
 * the whole 32-weight group. */
inline void accumulate_q4_group(__m256 *acc, const float *r, const uint8_t *packed, bool high,
                                float sd, float md) {
  const __m256 vsd = _mm256_set1_ps(sd);
  const __m256 vmd = _mm256_set1_ps(md);
  const __m256 zero = _mm256_setzero_ps();
  for (int i = 0; i < 32; i += 8) {
    const __m256 q = unpack_nibbles8(packed + i, high);
    const __m256 rv = _mm256_loadu_ps(r + i);
    /* ``sum r * (sd * q - md)``, which is ``(r * sd) * q`` *minus* ``r * md``.
     * The two terms are separate and the second carries no `q`: folding them
     * into one multiply would scale the minimum by the weight, which is the
     * bug this comment exists to keep out -- it makes the whole block wrong on
     * the high groups and is invisible in a spot check. */
    const __m256 offset = _mm256_fnmadd_ps(vmd, rv, zero); /* - r * md */
    *acc = _mm256_fmadd_ps(_mm256_mul_ps(rv, vsd), q, _mm256_add_ps(*acc, offset));
  }
}

/* Fold one 128-weight Q6_K half, four 32-weight runs of `ql`/`qh` bit assembly,
 * into the run's accumulator.  The shifts are the variable-count forms because
 * `sub` is a runtime loop index; the `ql` nibble is low or high by run and the
 * `qh` field sits at bit `2 * sub`, per `dequant_q6_k`. */
inline void accumulate_q6_half(__m256 *acc, const float *r, const uint8_t *block, int half) {
  const float d = quant::as_half(block, 208);
  const uint8_t *ql = block + half * 64;
  const uint8_t *qh = block + 128 + half * 32;
  for (int sub = 0; sub < 4; ++sub) {
    const uint8_t *ql_run = ql + (sub % 2) * 32;
    /* Two scales per run: weights 0..15 take scale index `2 * sub`, 16..31 take
     * `1 + 2 * sub`, both within this half's eight signed bytes. */
    const float ds0 = d * static_cast<float>(quant::as_int8(block, 192 + half * 8 + 2 * sub));
    const float ds1 = d * static_cast<float>(quant::as_int8(block, 192 + half * 8 + 1 + 2 * sub));
    const __m256 vds0 = _mm256_set1_ps(ds0);
    const __m256 vds1 = _mm256_set1_ps(ds1);
    for (int i = 0; i < 32; i += 8) {
      const __m256i ql_bytes =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(ql_run + i)));
      const __m256i ql_bits = sub < 2 ? _mm256_and_si256(ql_bytes, _mm256_set1_epi32(0x0F))
                                      : _mm256_srli_epi32(ql_bytes, 4);
      const __m256i qh_bytes =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(qh + i)));
      const __m256i qh_bits = _mm256_slli_epi32(
          _mm256_and_si256(_mm256_srlv_epi32(qh_bytes, _mm256_set1_epi32(2 * sub)),
                           _mm256_set1_epi32(3)),
          4);
      const __m256 q = _mm256_cvtepi32_ps(_mm256_or_si256(ql_bits, qh_bits));
      const __m256 qm32 = _mm256_sub_ps(q, _mm256_set1_ps(32.0F));
      const __m256 rv = _mm256_loadu_ps(r + half * 128 + sub * 32 + i);
      const __m256 ds = i < 16 ? vds0 : vds1;
      *acc = _mm256_fmadd_ps(_mm256_mul_ps(rv, ds), qm32, *acc);
    }
  }
}

}  // namespace

/* ``sum_k row[k] * dequant(type_id, blocks, k)`` for a row of `k` weights.
 *
 * `blocks` points at the row's first super-block and `k / 256` blocks follow.
 * The row is read once, in order, and the per-block scale hoist the scalar path
 * does still happens -- the block's `d`/`dmin` and its eight packed pairs are
 * decoded once per block, not once per weight. */
float dot_row_avx2(int type_id, const float *row, const uint8_t *blocks, int64_t k) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int64_t n_blocks = k / quant::kBlockWeights;
  /* Two accumulators, not one.  A single `__m256` makes every FMA depend on the
   * previous one, and on a Broadwell an FMA's latency is four or five times its
   * reciprocal throughput -- so one chain runs at 20-25% of the pipe.  Two
   * chains, alternated per 32-weight run, hide that: while one's result is in
   * flight the other issues.  Going wider than two buys little at this width,
   * because the decode and the row load already share the issue ports. */
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  if (type_id == quant::kGgmlQ4K) {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const float d = quant::as_half(block, 0);
      const float dmin = quant::as_half(block, 2);
      const uint8_t *scales = block + 4;
      const float *r = row + b * quant::kBlockWeights;
      for (int g = 0; g < 8; ++g) {
        int scale = 0;
        int minimum = 0;
        quant::get_scale_min_k4(scales, g, &scale, &minimum);
        __m256 *acc = (g & 1) != 0 ? &acc1 : &acc0;
        accumulate_q4_group(acc, r + g * 32, block + 16 + (g / 2) * 32, (g % 2) != 0,
                            d * static_cast<float>(scale), dmin * static_cast<float>(minimum));
      }
    }
  } else {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const float *r = row + b * quant::kBlockWeights;
      accumulate_q6_half(&acc0, r, block, 0);
      accumulate_q6_half(&acc1, r, block, 1);
    }
  }
  /* One horizontal reduce for the whole row, with a fixed pairing so the result
   * is reproducible run to run even though it is not the scalar order. */
  const __m256 acc = _mm256_add_ps(acc0, acc1);
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  return ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) +
         ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
}

/* The exact-float row dot, when it must be reachable from the same build the
 * integer one is in -- see `gemm_quant` for what selects each. */
#if !POCKETLLM_HAVE_AVX2
#define POCKETLLM_Q8K_HAVE_INT 0
#else
#define POCKETLLM_Q8K_HAVE_INT 1
#endif

#if POCKETLLM_Q8K_HAVE_INT

/* The broadcast tables the integer dot needs, one byte-selector per scale.  A
 * `_mm256_shuffle_epi8` with these replicates one byte of a packed (scale, min)
 * word across the lanes its 16 or 32 weights occupy, which is what lets the
 * per-group scale be applied with an exact integer `madd` instead of a float
 * multiply per weight.  Both are llama.cpp's tables, byte for byte -- an entry
 * off by one here applies the *neighbouring* group's scale, which is a small,
 * plausible, wrong answer rather than a crash. */
inline __m256i get_scale_shuffle_k4(int i) {
  static const uint8_t kShuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(kShuffle) + i);
}

/* The sixteen-scale variant: one 16-byte entry per 32-weight run, for q6_k's
 * per-16-weight scales.  Same table as llama.cpp's `get_scale_shuffle`. */
inline __m128i get_scale_shuffle_k6(int i) {
  static const uint8_t kShuffle[128] = {
      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,
      2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,
      4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,
      6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,
      8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,
      10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
      12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
      14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128(reinterpret_cast<const __m128i *>(kShuffle) + i);
}

/* Interpret a byte of Q4_K's six-bit (scale, min) pairs as llama.cpp's kernel
 * does before it can be shuffled: the eight pairs arrive as twelve packed
 * bytes, and the unpack rewrites them into four 32-bit words holding the eight
 * scales and eight minima as bytes.  That layout is what the `madd` against
 * `bsums` and the shuffle tables index, so a decoder that produced the same
 * numbers in a different arrangement would be correct and still wrong here. */
inline void unpack_scale_min_k4(const uint8_t *scales, uint32_t *out) {
  constexpr uint32_t kMask6 = 0x3f3f3f3f;
  constexpr uint32_t kMask4 = 0x0f0f0f0f;
  constexpr uint32_t kMask2 = 0x03030303;
  std::memcpy(out, scales, 12);
  out[3] = ((out[2] >> 4) & kMask4) | (((out[1] >> 6) & kMask2) << 4);
  const uint32_t aux = out[1] & kMask6;
  out[1] = (out[2] & kMask4) | (((out[0] >> 6) & kMask2) << 4);
  out[2] = aux;
  out[0] &= kMask6;
}

/* ``sum_k row[k] * dequant(row_blocks, k)`` with the activation in int8, and
 * the activation blocks already quantized by the caller.
 *
 * This is llama.cpp's `ggml_vec_dot_q4_K_q8_K`/`_q6_K_q8_K`, ported onto this
 * tree's block metadata.  The shape of the computation is what makes it fast:
 *
 *   - The activation is already quantized (one `Q8KBlock` per 256 weights, and
 *     the caller quantized the whole row once), so nothing on the right-hand
 *     side of the dot costs a decode per weight.
 *   - The weight, unpacked to 4 or 6 unsigned bits, is multiplied against the
 *     *signed* activation byte by `_mm256_maddubs_epi16`, which is a single
 *     instruction for 32 exact 8x8 -> 16-bit products.  `_mm256_madd_epi16`
 *     then folds the group scale in (also exact) and pairs the products into
 *     int32.  Every product inside the dot is an exact integer; the only float
 *     arithmetic is one multiply per block for the weight's `d` and its
 *     activation's `d`.
 *   - The Q4_K group *minimum* is `-(dmin * min) * sum(q8)`, and `sum(q8)` over
 *     the group is exactly what `bsums` holds.  So the whole minimum correction
 *     is one `madd` per block over the sixteen bsums rather than a per-weight
 *     subtraction; that is the reason the activation block carries `bsums` at
 *     all.
 *   - Q6_K is offset by 32 per weight, so the same correction is applied once
 *     per block: `sum_i w_i` over a 256-weight block is `d * sum(scale_group *
 *     sum(q8)_group)`, and the subtraction is `<< 5` of that, because 32 is a
 *     power of two.
 *
 * `acc_m` (the q4_k minimum term) accumulates in float and is folded in at the
 * end; `acc` accumulates the block scales.  Both are per-*row* accumulators --
 * one horizontal reduce for the whole row, as the float path does.
 *
 * The integer products are exact, so the result is bit-identical to llama.cpp's
 * AVX2 path modulo the one float multiply per block plus the horizontal sums --
 * and since both engines quantize the activation and the weights identically,
 * the two agree far more closely than the float path agreed with either. */
float dot_row_q8k(int type_id, const uint8_t *q8_blocks, const uint8_t *blocks, int64_t k) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int64_t n_blocks = k / quant::kBlockWeights;
  const quant::Q8KBlock *q8 = reinterpret_cast<const quant::Q8KBlock *>(q8_blocks);

  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i m3 = _mm256_set1_epi8(0x03);
  __m256 acc = _mm256_setzero_ps();
  __m128 acc_m = _mm_setzero_ps();

  if (type_id == quant::kGgmlQ4K) {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const quant::Q8KBlock &y = q8[b];
      const float d = y.d * quant::as_half(block, 0);
      const float dmin = -y.d * quant::as_half(block, 2);

      /* The eight (scale, min) pairs are six bits each packed into twelve
       * bytes; this is `get_scale_min_k4` written as the word-at-a-time
       * unpacking llama.cpp uses, so the result lands in one 256-bit register
       * that the shuffle tables below can index per 32-weight group.  The
       * splat into both 128-bit halves is what the scale shuffle reads. */
      uint32_t utmp[4];
      unpack_scale_min_k4(block + 4, utmp);

      const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
          _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                        static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));

      /* The minimum term: `dmin * sum_g min_g * sum(q8 over g)`.  The bsums
       * pair up into eight groups of two so that one `madd` covers all eight
       * groups at once. */
      const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.bsums));
      const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                         _mm256_extracti128_si256(q8sums, 1));
      const __m128i prod =
          _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
      acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

      const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
      const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

      const uint8_t *q4 = block + 16;
      const int8_t *q8s_ptr = y.qs;
      __m256i sumi = _mm256_setzero_si256();
      for (int j = 0; j < 4; ++j) {
        /* Each 32-byte packed run carries two 32-weight groups: the low nibbles
         * are group `2j` and the high nibbles group `2j + 1`, each with its own
         * scale.  `get_scale_shuffle_k4` broadcasts the right six-bit scale
         * across the sixteen positions its group's products occupy. */
        const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
        const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

        const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
        q4 += 32;
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr));
        q8s_ptr += 32;
        __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
        p16l = _mm256_madd_epi16(scale_l, p16l);

        const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr));
        q8s_ptr += 32;
        __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
        p16h = _mm256_madd_epi16(scale_h, p16h);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
      }
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
  } else {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const quant::Q8KBlock &y = q8[b];
      const float d = y.d * quant::as_half(block, 208);

      /* The (q - 32) offset, applied once per block rather than per weight.
       * `scales_16` widens the sixteen signed byte scales; `madd` pairs each
       * with the two 16-element bsum groups it multiplies, and the `<< 5`
       * multiplies the whole thing by 32. */
      const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.bsums));
      const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block + 192));
      const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
      const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

      const uint8_t *ql = block;
      const uint8_t *qh = block + 128;
      const int8_t *q8s_ptr = y.qs;
      __m256i sumi = _mm256_setzero_si256();
      for (int j = 0; j < 2; ++j) {
        /* 128 weights per iteration: both 64-byte `ql` spans and the whole
         * 32-byte `qh` span of this half, assembled into four 32-byte runs of
         * six-bit weights.  `qh` holds two bits per weight at four different
         * bit offsets, one per run -- the shifts below are those offsets. */
        const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
        const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
        const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
        const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
        const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
        const __m256i q4_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
        const __m256i q4_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

        const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr));
        const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr + 32));
        const __m256i q8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr + 64));
        const __m256i q8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8s_ptr + 96));
        q8s_ptr += 128;

        __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
        __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
        __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
        __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

        /* The sixteen scales of the block are one per 16 weights in position
         * order, so run `is` reads scale `is` -- the shuffle broadcasts it. */
        const int is = 4 * j;
        p16_0 = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 0))), p16_0);
        p16_1 = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 1))), p16_1);
        p16_2 = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 2))), p16_2);
        p16_3 = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 3))), p16_3);

        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        ql += 64;
        qh += 32;
      }
      sumi = _mm256_sub_epi32(sumi, q8sclsub);
      acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }
  }

  /* The minimum term lives in four int32 lanes of `acc_m`; fold them into one
   * and add it to the horizontal sum of `acc`. */
  acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
  acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  const float total = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) +
                      ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
  return total + _mm_cvtss_f32(acc_m);
}

/* Is the integer path selected at all?  A build without AVX2 cannot run it, and
 * `$POCKETLLM_CPU_EXACT_GEMM` turns it off on one that can, so the exact and
 * quantized paths can be compared on the same input.  Read once, on first use:
 * the answer cannot change under a running process and the per-call `getenv`
 * would be a lock on the token path. */
inline bool exact_gemm_forced() {
  const char *from_env = std::getenv("POCKETLLM_CPU_EXACT_GEMM");
  return from_env != nullptr && from_env[0] != '\0' && from_env[0] != '0';
}

#endif /* POCKETLLM_Q8K_HAVE_INT */

/* One weight row against `R` activation rows, R = 4 or 8.
 *
 * The four-row kernel was a hand-written copy of this; this template is that
 * bound and the body made a template on R, so the dispatcher can choose the row
 * count without a second hand-written kernel.  The row count is a template
 * parameter and not a runtime loop bound for the reason the four copies exist at
 * all: each row needs its own `__m256i` accumulator in a register, and an
 * indexed array of them would be spilled to the stack, which costs more than the
 * wider walk saves.
 *
 * **It is bit-identical to R calls to `dot_row_q8k`, and that is a requirement
 * and not a happy accident** -- the same property the four-row kernel carried, for
 * same reason: the prefill activation is an operand of the token-for-token
 * llama.cpp match, so a reassociated row would be a different model.  The row
 * count changes how many rows share one weight decode; it does not move a
 * reduce, change a per-block scale's order or touch the offset term.
 * `tests/native/test_cpu_parallel.py` holds R=4 and R=8 to the one-row kernel's
 * bytes.
 *
 * ## R = 8 is the shipped row count, and where the number comes from
 *
 * The tempting next step after the four-row tile is an eight-row one: it halves
 * the number of times the `n x k` weight panel is streamed, and llama.cpp's
 * repacked kernel is an eight-row form.  A block-templated probe on one core
 * (`/tmp/prof/gemmrow.cpp`) measured R=8 *ahead* of R=4 by 5-8% at every shape.
 *
 * **A single-core ratio does not settle it, and the first attempt to transfer it
 * went the wrong way.**  A second probe put the same two kernels behind the
 * engine's own `parallel_for` and reported R=8 *behind* R=4 at 22 threads -- but
 * that probe's traversal ran 6x slower than the engine's own GEMM, so it was
 * measuring its own memory behaviour and not the kernel's.  The number that
 * decides it is the engine's, on a quiet host, at the thread count that matters:
 *
 *     pp512, q4_k_m, all 88 hardware threads, interleaved, median of 6
 *       R=4   905 t/s
 *       R=8   978 t/s     (+8%)
 *
 * R=8 is ahead by a clear margin with the machine full, which is the opposite of
 * what register pressure predicts (the tile wants ~29 of the 32 AVX2 registers
 * at R=8 against 17 at R=4) and the reason it has to be measured rather than
 * reasoned about.  The row count is a compile-time bound in the dispatcher, so
 * changing it is a one-line experiment: see `gemm_rows_per_walk`.
 *
 * **Check the host load first.**  This machine is shared, and a loaded host turns
 * every one of these numbers into noise -- an early run of this very comparison
 * read 26 t/s at 44 threads against 500 at 22, which was contention and not
 * arithmetic. */
template <int R>
void dot_Rrows_q8k(int type_id, const quant::Q8KBlock *q8, int64_t q8_stride,
                   const uint8_t *blocks, int64_t k, float out[R]) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int64_t n_blocks = k / quant::kBlockWeights;
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i m03 = _mm256_set1_epi8(0x03);

  __m256 acc[R];
  __m128 mn[R];
  for (int i = 0; i < R; ++i) {
    acc[i] = _mm256_setzero_ps();
    mn[i] = _mm_setzero_ps();
  }
  __m256i a[R];
  __m256i off[R];

  if (type_id == quant::kGgmlQ4K) {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const float wd = quant::as_half(block, 0);
      const float wdmin = quant::as_half(block, 2);

      uint32_t utmp[4];
      unpack_scale_min_k4(block + 4, utmp);
      const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
          _mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                        static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
      const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
      const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
      const __m256i scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

      /* The minimum term, written once per row as `dot_row_q8k` writes it: the
       * bsums pair up into eight groups of two for one `madd`, four lanes. */
      for (int i = 0; i < R; ++i) {
        const quant::Q8KBlock &y = q8[i * q8_stride + b];
        const __m128i bs = _mm_hadd_epi16(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(y.bsums)),
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(y.bsums + 8)));
        mn[i] = _mm_fmadd_ps(_mm_set1_ps(-y.d * wdmin),
                             _mm_cvtepi32_ps(_mm_madd_epi16(mins128, bs)), mn[i]);
      }

      const uint8_t *q4 = block + 16;
      const int8_t *p[R];
      for (int i = 0; i < R; ++i) {
        p[i] = q8[i * q8_stride + b].qs;
        a[i] = _mm256_setzero_si256();
      }
      for (int j = 0; j < 4; ++j) {
        const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
        const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
        const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
        q4 += 32;
        const __m256i q4l = _mm256_and_si256(q4bits, m4);
        const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

        /* `R` rows against the same decoded nibbles, in the one-row kernel's
         * exact expression.  The macro is here because writing it eight times by
         * hand is eight chances to type the wrong pointer, not because the rows
         * are interchangeable. */
#define POCKETLLM_TILE_ACC(SUMI, PTR)                                                    \
  {                                                                                      \
    const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(PTR));       \
    __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);                                        \
    p16l = _mm256_madd_epi16(scale_l, p16l);                                              \
    const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(PTR + 32));  \
    __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);                                        \
    p16h = _mm256_madd_epi16(scale_h, p16h);                                              \
    SUMI = _mm256_add_epi32(SUMI, _mm256_add_epi32(p16l, p16h));                          \
  }
        for (int i = 0; i < R; ++i) {
          POCKETLLM_TILE_ACC(a[i], p[i])
          p[i] += 64;
        }
#undef POCKETLLM_TILE_ACC
      }
      for (int i = 0; i < R; ++i) {
        const quant::Q8KBlock &y = q8[i * q8_stride + b];
        acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(y.d * wd), _mm256_cvtepi32_ps(a[i]), acc[i]);
      }
    }
  } else {
    for (int64_t b = 0; b < n_blocks; ++b) {
      const uint8_t *block = blocks + b * block_bytes;
      const float wd = quant::as_half(block, 208);

      const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block + 192));
      const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);

      /* The (q - 32) offset, once per block per row -- an int32 lane vector and
       * not a float accumulator, the type `dot_row_q8k` keeps it in. */
      for (int i = 0; i < R; ++i) {
        const quant::Q8KBlock &y = q8[i * q8_stride + b];
        const __m256i ssums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.bsums));
        off[i] = _mm256_slli_epi32(_mm256_madd_epi16(ssums, scales_16), 5);
      }

      const uint8_t *ql = block;
      const uint8_t *qh = block + 128;
      const int8_t *p[R];
      for (int i = 0; i < R; ++i) {
        p[i] = q8[i * q8_stride + b].qs;
        a[i] = _mm256_setzero_si256();
      }
      for (int j = 0; j < 2; ++j) {
        const __m256i q4bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql));
        const __m256i q4bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ql + 32));
        const __m256i q4bitsH = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qh));

        const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m03), 4);
        const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
        const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
        const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

        const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
        const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
        const __m256i q4_2 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
        const __m256i q4_3 = _mm256_or_si256(
            _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

        const int is = 4 * j;
        const __m256i sc_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is)));
        const __m256i sc_1 =
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 1)));
        const __m256i sc_2 =
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 2)));
        const __m256i sc_3 =
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_k6(is + 3)));

#define POCKETLLM_TILE_ACC6(SUMI, PTR)                                                  \
  {                                                                                     \
    __m256i p0v = _mm256_maddubs_epi16(q4_0, _mm256_loadu_si256(                        \
        reinterpret_cast<const __m256i *>((PTR) + 0)));                                 \
    __m256i p1v = _mm256_maddubs_epi16(q4_1, _mm256_loadu_si256(                        \
        reinterpret_cast<const __m256i *>((PTR) + 32)));                                \
    __m256i p2v = _mm256_maddubs_epi16(q4_2, _mm256_loadu_si256(                        \
        reinterpret_cast<const __m256i *>((PTR) + 64)));                                \
    __m256i p3v = _mm256_maddubs_epi16(q4_3, _mm256_loadu_si256(                        \
        reinterpret_cast<const __m256i *>((PTR) + 96)));                                \
    p0v = _mm256_madd_epi16(sc_0, p0v);                                                 \
    p1v = _mm256_madd_epi16(sc_1, p1v);                                                 \
    p2v = _mm256_madd_epi16(sc_2, p2v);                                                 \
    p3v = _mm256_madd_epi16(sc_3, p3v);                                                 \
    SUMI = _mm256_add_epi32(SUMI, _mm256_add_epi32(p0v, p1v));                          \
    SUMI = _mm256_add_epi32(SUMI, _mm256_add_epi32(p2v, p3v));                          \
  }
        for (int i = 0; i < R; ++i) {
          POCKETLLM_TILE_ACC6(a[i], p[i])
          p[i] += 128;
        }
#undef POCKETLLM_TILE_ACC6
        ql += 64;
        qh += 32;
      }
      for (int i = 0; i < R; ++i) {
        const quant::Q8KBlock &y = q8[i * q8_stride + b];
        acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(y.d * wd),
                                 _mm256_cvtepi32_ps(_mm256_sub_epi32(a[i], off[i])), acc[i]);
      }
    }
  }

  /* One horizontal reduce per row, the expression `dot_row_q8k` ends with.
   *
   * The association is exactly `((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 +
   * l7))`, the tree the row kernel has always built, and it is what makes this
   * bit-identical to R calls to `dot_row_q8k`.  The pairing is *across* the two
   * 128-bit halves, so the halves are added first (`lo + hi` gives
   * `[l0+l4, l1+l5, l2+l6, l3+l7]`) and only then reduced horizontally --
   * `_mm_hadd_ps` on the raw 256-bit value would pair lanes *within* each half
   * and build `((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))` instead, which is a
   * different sum.  Writing the scalar form out lane by lane costs the compiler
   * a stack round trip plus eight `vaddss`; the three instructions below are the
   * same tree.  `tests/native/test_cpu_parallel.py` holds the two forms
   * together, and the point of stating the association here is that a future
   * change to either one has to keep it. */
  for (int i = 0; i < R; ++i) {
    const __m128 halves =
        _mm_add_ps(_mm256_castps256_ps128(acc[i]), _mm256_extractf128_ps(acc[i], 1));
    const __m128 s = _mm_hadd_ps(halves, halves);
    const float total = _mm_cvtss_f32(_mm_hadd_ps(s, s));
    if (type_id == quant::kGgmlQ4K) {
      __m128 am = mn[i];
      am = _mm_add_ps(am, _mm_movehl_ps(am, am));
      am = _mm_add_ss(am, _mm_movehdup_ps(am));
      out[i] = total + _mm_cvtss_f32(am);
    } else {
      out[i] = total;
    }
  }
}

#endif /* POCKETLLM_HAVE_AVX2 */

/* One row of `type_id` weights dotted against `row`.  The caller has checked
 * `type_id` against the type table, so the unsupported branch is unreachable;
 * it returns the scalar sum anyway rather than leaving an uninitialized value a
 * future caller could read. */
float dot_row(int type_id, const float *row, const uint8_t *blocks, int64_t k) {
#if POCKETLLM_HAVE_AVX2
  return dot_row_avx2(type_id, row, blocks, k);
#else
  const int block_bytes = quant::block_bytes_of(type_id);
  float total = 0.0F;
  for (int64_t b = 0; b < k / quant::kBlockWeights; ++b) {
    total += type_id == quant::kGgmlQ4K ? dot_q4_k_block_scalar(row + b * quant::kBlockWeights,
                                                                blocks + b * block_bytes)
                                        : dot_q6_k_block_scalar(row + b * quant::kBlockWeights,
                                                                blocks + b * block_bytes);
  }
  return total;
#endif
}

/* ``exp(x - max) / sum`` over one row -- the reduction `softmax` and
 * `topk_sample` both need, written once so the conformance test that certifies
 * `softmax` is certifying the arithmetic the sampler runs.
 *
 * `Real` is the accumulator, and it is not the same for the two callers.
 * `softmax` is the op the ABI declares and its reference is numpy's pairwise
 * float32 sum, so it accumulates in `float`. The sampler's cumulative sum runs
 * over a 151936-long tail and inverts it at a uniform draw, where the spacing
 * of a float32 at 1.0 is wider than the steps it is walking, so it accumulates
 * in `double`. The two share the structure and not the precision.
 *
 * The shift is not optional in either. The reference shifts by the row max, and
 * an unshifted `exp` of a logit at -800 underflows to zero -- which turns the
 * whole row into 0/0 and the sampler's answer into whatever NaN compares as.
 * The shift is taken in `double` for both, so the exponent is exact in the
 * wider type and the only rounding left is the one the caller asked for.
 *
 * The sum is a single running one, where numpy's `ex.sum()` is a pairwise
 * reduction: the two disagree in the last few ulps of a wide row, which is why
 * the conformance comparison for `softmax` is an absolute bound and not a
 * relative one -- a relative bound there is measuring the association order and
 * not the op. */
template <typename Real>
void softmax_row(const float *row, Real *dst, int64_t cols) {
  float max = row[0];
  for (int64_t i = 1; i < cols; ++i) {
    if (row[i] > max) {
      max = row[i];
    }
  }
  Real total = static_cast<Real>(0);
  for (int64_t i = 0; i < cols; ++i) {
    dst[i] = static_cast<Real>(std::exp(static_cast<double>(row[i]) - static_cast<double>(max)));
    total += dst[i];
  }
  if (total > static_cast<Real>(0)) {
    for (int64_t i = 0; i < cols; ++i) {
      dst[i] /= total;
    }
  }
}

/* Rank the indices of `values` by descending value, ties to the lower index --
 * numpy's `argsort(-probs, kind="stable")`.
 *
 * `std::stable_sort` on the value alone would give the same answer: a stable
 * sort preserves the input order among equals, and the input order here is
 * ascending index. The explicit index comparison is written out anyway so the
 * rule is on the line, because the property it depends on (the range is in
 * ascending index order) is not local to it.
 *
 * The caller ranks *probabilities* and not logits, because that is what the
 * reference ranks. The two orders are the same in exact arithmetic; they differ
 * only where two probabilities round to the same float32, and there numpy's
 * stable sort takes the lower index -- which is what this does too, provided
 * the values it is handed are the same rounded ones. */
template <typename Real>
void rank_descending(const Real *values, int64_t n, int64_t *order) {
  for (int64_t i = 0; i < n; ++i) {
    order[i] = i;
  }
  std::stable_sort(order, order + n, [values](int64_t a, int64_t b) {
    /* Written as two one-sided comparisons rather than `!=` then `>`, so the
     * order stays a strict weak order even if a NaN is present: a NaN compares
     * false both ways and falls through to the index, which is a valid total
     * order instead of undefined behaviour. The contract says not to feed it
     * one; this is what happens if somebody does. */
    if (values[a] > values[b]) {
      return true;
    }
    if (values[a] < values[b]) {
      return false;
    }
    return a < b;
  });
}

}  // namespace

void rms_norm(const float *x, const float *weight, float *out, int64_t n_tokens, int64_t d,
              float eps) {
  /* Over tokens: each row reads and writes its own `d` floats. */
  parallel_for(n_tokens, /*min_per_task=*/8, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t r = lo; r < hi; ++r) {
      const float *row = x + r * d;
      float *dst = out + r * d;
      float sum = 0.0F;
      int64_t i = 0;
#if POCKETLLM_HAVE_AVX2
      /* Independent lane sums remove the scalar reduction's dependency chain.
       * The tree is fixed per row, not per batch or pool size, so prefill and
       * decode normalize the same row identically. The scalar tail also covers
       * head widths that are not multiples of eight. */
      __m256 s0 = _mm256_setzero_ps();
      __m256 s1 = _mm256_setzero_ps();
      __m256 s2 = _mm256_setzero_ps();
      __m256 s3 = _mm256_setzero_ps();
      for (; i + 32 <= d; i += 32) {
        const __m256 x0 = _mm256_loadu_ps(row + i);
        const __m256 x1 = _mm256_loadu_ps(row + i + 8);
        const __m256 x2 = _mm256_loadu_ps(row + i + 16);
        const __m256 x3 = _mm256_loadu_ps(row + i + 24);
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, x0));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x1, x1));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(x2, x2));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(x3, x3));
      }
      __m256 sums = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
      for (; i + 8 <= d; i += 8) {
        const __m256 v = _mm256_loadu_ps(row + i);
        sums = _mm256_add_ps(sums, _mm256_mul_ps(v, v));
      }
      __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sums),
                               _mm256_extractf128_ps(sums, 1));
      lanes = _mm_hadd_ps(lanes, lanes);
      lanes = _mm_hadd_ps(lanes, lanes);
      sum = _mm_cvtss_f32(lanes);
#endif
      for (; i < d; ++i) {
        sum += row[i] * row[i];
      }
      /* The mean is over `d` -- the row -- and the reciprocal square root is
       * computed once for the whole row rather than per element. */
      const float scale = 1.0F / std::sqrt(sum / static_cast<float>(d) + eps);
      for (int64_t i = 0; i < d; ++i) {
        dst[i] = row[i] * scale * weight[i];
      }
    }
  });
}

void gemm(const float *x, const float *w, const float *bias, float *out, int64_t m, int64_t n,
          int64_t k, bool accumulate) {
  /* Over `(r, j)`, flattened: each output element is an independent `dot` over
   * `k`.  The `k` axis is never the split, because summing partials from
   * several threads would change the association order.  The row is computed
   * from the flattened index so there is one loop rather than a per-row
   * dispatch for a matrix that at decode is one row wide. */
  parallel_for(m * n, /*min_per_task=*/64, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t index = lo; index < hi; ++index) {
      const int64_t r = index / n;
      const int64_t j = index - r * n;
      const float value0 = dot(x + r * k, w + j * k, k);
      const float value = bias != nullptr ? value0 + bias[j] : value0;
      /* `+=` on the residual path, `=` otherwise. The two are different
       * operations and the graph asks for each explicitly, so they are one
       * branch here rather than two loops that would drift apart. */
      out[r * n + j] = accumulate ? out[r * n + j] + value : value;
    }
  });
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
  const int64_t row_blocks = k / per_block;

#if POCKETLLM_Q8K_HAVE_INT
  /* Quantize every row of the activation once, before the parallel walk: the
   * result is read by every one of the `n` outputs on that row, so doing it
   * inside the loop would repeat it per output column, which is the whole cost
   * the integer path exists to avoid.
   *
   * **The quantization runs on the pool, and that is the second largest term in
   * the prefill gap.**  It used to be a serial loop on the caller's thread,
   * which made it the one part of a prefill GEMM that did not scale: at `m=512,
   * k=1024` it is 2048 independent blocks of 256 weights and measured 1087 us
   * serial against a 4244 us call -- a fifth of the GEMM on one core while
   * twenty-one sat at the barrier.  The blocks are independent by construction
   * (each writes its own `d`, `qs` and `bsums`, and reads only its own 256
   * floats), so running them on the pool is a change of scheduling and not of
   * arithmetic: the same body, byte for byte, at 55 us.
   *
   * That byte-for-byte part is load-bearing rather than incidental.  The
   * quantized activation is an operand of the token-for-token llama.cpp match,
   * so a "parallel" version that split a block's reduction or reassociated its
   * scale would change the model.  `quantize_q8_block` is one whole block's
   * arithmetic and the loop below runs it per block, so the schedule cannot
   * reach inside one.
   *
   * **The scratch is the shape's size, and it used to be a fixed 1024 blocks.**
   * A `Q8KBlock` is 292 bytes, so 1024 of them is 299 KiB; a decode reaches 4-16
   * blocks and a 512-token prefill reaches 6144, so the fixed cap was a decode
   * size and every prefill longer than 256 tokens fell through to the exact
   * path. The fallback is *correct* and therefore silent -- nothing fails, only
   * slows -- which is why it went unnoticed. It is the largest single term in
   * the prefill gap: at `pp512` on 22 cores the integer path is ~180 t/s and
   * the exact path ~98 t/s on the same prompt.
   *
   * The decode shapes keep taking the stack buffer, because they are the hot
   * path and a 197-call-per-token decode cannot pay a heap round trip per GEMM
   * for a 4-block array. Above `kStackBlocks` the scratch is a heap allocation
   * sized to the shape, so one `q8` pointer is chosen and the walk below is
   * written once.
   *
   * The allocation is `nothrow` and a failure falls back to the exact path:
   * `gemm_quant` is reachable from the ABI with shapes this graph does not
   * produce, and a `bad_alloc` on the token path would be a worse answer than a
   * slower one. */
  constexpr int64_t kStackBlocks = 1024; /* 299 KiB of activation blocks */
  const bool exact = exact_gemm_forced();
  const int64_t total_blocks = m * row_blocks;
  const bool use_int = !exact && total_blocks > 0;
  alignas(64) quant::Q8KBlock stack_q8[kStackBlocks];
  std::unique_ptr<quant::Q8KBlock[]> heap_q8;
  quant::Q8KBlock *q8 = stack_q8;
  if (use_int && total_blocks > kStackBlocks) {
    heap_q8.reset(new (std::nothrow) quant::Q8KBlock[static_cast<std::size_t>(total_blocks)]);
    if (heap_q8 != nullptr) {
      q8 = heap_q8.get();
    } else {
      q8 = nullptr;
    }
  }
  const bool quantize_activations = use_int && q8 != nullptr;
  if (quantize_activations) {
    /* One block per task at the floor, which is where the measurement above was
     * taken: a block is 256 floats of work, so eight to a task is the point past
     * which a chunk stops being worth waking a worker for. */
    parallel_for(total_blocks, /*min_per_task=*/8, [&](int64_t lo, int64_t hi, int64_t) {
      for (int64_t b = lo; b < hi; ++b) {
        quant::quantize_q8_block(x + b * quant::kQ8KWeights, q8[b]);
      }
    });
  }
#endif

  /* The parallel axis is `j`, because at decode `m` is 1 and `j` is the only
   * axis wide enough to fill the machine -- and the only one that is safe: the
   * column `j` owns `blocks + j * row_bytes` outright, so its accumulation over
   * `k` stays whole inside one task.
   *
   * Do **not** be tempted to split that `k` walk across threads and add the
   * partials.  It is the obvious next step and it is wrong here: the sums would
   * be associated differently from the single-threaded path, so the result
   * would move with the thread count.  Everything below keeps `total` a single
   * serial chain, exactly as before. */
#if POCKETLLM_Q8K_HAVE_INT
  if (quantize_activations) {
    /* Grouped four rows to a weight walk.  The index space is the grouped part
     * first -- `groups * n` outputs, each computing four rows of column `j` --
     * and the rows that do not fill a group of four last.  At decode `m` is 1
     * and every row takes the one-row path, which is why the batched kernel
     * costs a decode nothing; at prefill `m % 4` is usually zero and the whole
     * GEMM walks its weights `m / 4` times instead of `m`. */
    const int64_t rpw = gemm_rows_per_walk();
    const int64_t groups = m / rpw;
    const int64_t tail = m - groups * rpw;
    parallel_for(groups * n + tail * n, /*min_per_task=*/32, [&](int64_t lo, int64_t hi, int64_t) {
      for (int64_t index = lo; index < hi; ++index) {
        const int64_t g = index / n;
        const int64_t j = index - g * n;
        const uint8_t *row_blocks_ptr = blocks + j * row_bytes;
        if (g < groups) {
          const int64_t r = g * rpw;
          /* The rows' whole dot in one call -- see `dot_Rrows_q8k` for why it is
           * one call and why its arithmetic is the one-row kernel's, not a
           * reassociation of it. */
          float totals[8];
          if (rpw == 8) {
            dot_Rrows_q8k<8>(type_id, q8 + r * row_blocks, row_blocks, row_blocks_ptr, k, totals);
          } else {
            dot_Rrows_q8k<4>(type_id, q8 + r * row_blocks, row_blocks, row_blocks_ptr, k, totals);
          }
          for (int64_t i = 0; i < rpw; ++i) {
            float value = totals[i];
            if (bias != nullptr) {
              value += bias[j];
            }
            const int64_t o = (r + i) * n + j;
            out[o] = accumulate ? out[o] + value : value;
          }
        } else {
          const int64_t r = groups * rpw + (g - groups);
          const float total =
              dot_row_q8k(type_id, reinterpret_cast<const uint8_t *>(q8 + r * row_blocks),
                          row_blocks_ptr, k);
          const float value = bias != nullptr ? total + bias[j] : total;
          out[r * n + j] = accumulate ? out[r * n + j] + value : value;
        }
      }
    });
    return;
  }
#endif

  parallel_for(m * n, /*min_per_task=*/32, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t index = lo; index < hi; ++index) {
      const int64_t r = index / n;
      const int64_t j = index - r * n;
      const float *row = x + r * k;
      const uint8_t *row_blocks_ptr = blocks + j * row_bytes;
      /* The whole row's dot in one call: the AVX2 path folds all `k / 256`
       * blocks into one set of accumulators and reduces once, rather than
       * reducing per block.  The decode stays hoisted per block, so the two
       * halves of the saving -- cheaper decode and fewer horizontal sums -- are
       * both here. */
#if POCKETLLM_Q8K_HAVE_INT
      float total =
          quantize_activations
              ? dot_row_q8k(type_id,
                            reinterpret_cast<const uint8_t *>(q8 + r * row_blocks),
                            row_blocks_ptr, k)
              : dot_row(type_id, row, row_blocks_ptr, k);
#else
      float total = dot_row(type_id, row, row_blocks_ptr, k);
#endif
      if (bias != nullptr) {
        total += bias[j];
      }
      out[r * n + j] = accumulate ? out[r * n + j] + total : total;
    }
  });
}

void embedding(const int32_t *tokens, int64_t n_tokens, const float *table, int64_t vocab,
               int64_t d, float *out) {
  /* Over tokens: each token writes its own row. */
  parallel_for(n_tokens, /*min_per_task=*/1, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t t = lo; t < hi; ++t) {
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
  });
}

void embedding_quant(const int32_t *tokens, int64_t n_tokens, const uint8_t *blocks, int64_t vocab,
                     int64_t d, int type_id, float *out) {
  const int block_bytes = quant::block_bytes_of(type_id);
  const int per_block = quant::kBlockWeights;
  const int64_t row_bytes = (d / per_block) * block_bytes;

  /* Over tokens, as the dense gather: a token reads only its own row of blocks.
   * Splitting a token's `d` as well would be safe but pointless -- at decode
   * `n_tokens` is 1 and the row is short, and the balance between the two is a
   * question for a larger model than this one. */
  parallel_for(n_tokens, /*min_per_task=*/1, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t t = lo; t < hi; ++t) {
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
  });
}

void silu_mul(const float *gate, const float *up, float *out, int64_t n) {
  /* Elementwise: every output depends on its own index only, so the grain is a
   * scheduling choice and not a numeric one -- the output is byte-identical at
   * every thread count, which is what `test_cpu_parallel.py` checks.
   *
   * The grain is 64 and it used to be 4096, which was a *prefill* size: the
   * hidden width is 3072, so at decode the whole op was one element short of a
   * task and ran on the caller's thread alone. `expf` is tens of cycles, so that
   * made a 3072-element map a ~20 us serial term on a token that calls it 28
   * times -- 600 us of a 16 ms token. Measured at 22 threads, ctx 512, the
   * decode term per token: 4096 -> 600 us, 256 -> 220, 64 -> 256, with the
   * end-to-end token moving with it. 64 and 256 are within noise of each other
   * and 64 is the one that still fills the pool on a narrower shape, so it is
   * the one that ships. */
  parallel_for(n, /*min_per_task=*/64, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t i = lo; i < hi; ++i) {
      const float g = gate[i];
      /* `expf` is called with the value the reference passes -- not a clamped
       * one. For a large negative g the exponential goes to zero and the result
       * to zero, which is the limit and not an overflow; a clamp would change the
       * answer in the one range where it is cheap to be right. */
      out[i] = (g / (1.0F + std::exp(-g))) * up[i];
    }
  });
}

void rope_neox(float *x, int64_t n_tokens, int64_t n_heads, int64_t d, int64_t start_pos,
               const float *cos_table, const float *sin_table) {
  const int64_t half = d / 2;
  const int64_t units = n_tokens * n_heads;
  /* Over `(token, head)`: each unit rotates its own `d`-wide vector in place.
   * The `half`-wide loop inside stays whole -- its two writes share `a` and `b`,
   * so splitting it would be splitting a single vector's elements across
   * threads, which is not the same computation. */
  parallel_for(units, /*min_per_task=*/8, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t unit = lo; unit < hi; ++unit) {
      const int64_t t = unit / n_heads;
      const int64_t position = start_pos + t;
      const float *cos_row = cos_table + position * half;
      const float *sin_row = sin_table + position * half;
      float *vec = x + unit * d;
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
  });
}

void attention(const float *q, int64_t q_len, int64_t n_heads, const void *k_cache,
               const void *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores, KVDtype kv_dtype) {
  const int64_t group = n_heads / n_head_kv;
  /* The score row for a query at position `q_offset + q_len - 1` is the widest
   * any row can be, and it is what spaces the rows within a per-task scratch
   * region apart.  `attention_scratch` and the callers that size the buffer use
   * the same quantity. */
  const int64_t max_span = q_offset + q_len - first_key;
  const int64_t blocks = (q_len + kAttentionRows - 1) / kAttentionRows;

  /* One cache row, widened out of the f16 cache and then read from L1 by every
   * kernel in the unit.  `f16` is false on an f32 cache, where this is the
   * cache pointer itself -- the f32 path keeps exactly the address arithmetic
   * it always had and pays one predicted branch per row, not per element.
   *
   * The scratch is declared *inside* the parallel body below, and that is not
   * a style choice: a buffer in this frame would be one buffer written by every
   * pool thread at once, which is a race whose symptom is a finite, plausible,
   * wrong score row -- and it measured as a real slowdown as well, because the
   * false sharing on one 512-byte line serialized the eight attention units.
   * Per task is per thread here: a task runs to completion on one thread.
   *
   * See `kv_row_to_float` for why widening is the design. */
  const bool f16 = kv_dtype == KVDtype::kF16;
  const int64_t kv_stride = n_head_kv * d;
  if (f16 && d > kAttentionSumMaxDim) {
    throw Error("attention: an f16 cache needs d <= " + std::to_string(kAttentionSumMaxDim) +
                ", got " + std::to_string(d));
  }

  /* Over `(block of kAttentionRows tokens, head)`.  Each unit owns one output
   * row per token it covers and one score row per token too, so the scratch is
   * private to the task that writes it -- `scores` is a shared buffer handed in
   * by the caller, and two tasks writing the same row would be the classic race
   * whose symptom is fluent, finite, wrong output.  The per-task region is
   * `scores + chunk * kAttentionRows * max_span`; nothing else is shared, and
   * the dot, the max scan and the weighted sum inside a unit stay serial so the
   * arithmetic is the same as the single-threaded path.
   *
   * The unit is also the *whole* of a task's work: there is nothing finer to
   * split, because the three passes inside it are chained -- `max_score` is
   * needed before the exponentials and `total` before the weighted sum -- so a
   * second thread on the same unit could only wait.  `kAttentionGrain` is 1 for
   * that reason; see its comment in the header.
   *
   * The score pass is where the tiling pays: the keys a whole block can see
   * (`s <= q_offset + t0`) are walked once for `kAttentionRows` query rows, in
   * `dot_tile_r`, which produces exactly the `dot4` results a row-at-a-time walk
   * would.  The causal *tail* -- the `kAttentionRows - 1` keys after that,
   * which only the later rows of the block can see -- takes `dot4` one row at a
   * time rather than a second, ragged tiled kernel, so there is exactly one
   * tiled code path to be right and the tail is the same code the whole kernel
   * used to be.
   *
   * A `q_len` that is not a multiple of `kAttentionRows` is not a special case:
   * the last block covers fewer rows, the shared region is the one those rows
   * share, and every one of them is handled -- including the single-row block a
   * decode step runs, which is the shipped one-`dot4`-per-row path. */
  /* Rounded up, not `n_heads / kAttentionHeadBatch`: a head count that is not a
   multiple of the batch leaves a final unit with one head in it, and truncating
   drops that unit on the floor -- the head's output is never written and the
   caller reads whatever the buffer held.  Qwen3's 16 heads divide evenly, which
   is exactly why the test parametrizes an odd count. */
  const int64_t groups = (n_heads + kAttentionHeadBatch - 1) / kAttentionHeadBatch;

  /* Deal the query blocks from both ends, and this is a load-balance fix rather
   * than a preference.
   *
   * Attention is causal, so block `b` walks `q_offset + b * kAttentionRows`
   * shared keys -- its cost grows with the block index, roughly linearly.  The
   * unit index used to be `b * groups + g`, so a contiguous split of it gave the
   * first worker the cheapest block and the last worker the most expensive one,
   * and `parallel_for` caps its chunk count at the thread count, so there is no
   * over-decomposition for the pool's work-stealing to even out.  The result was
   * measured: `attention` reached 8.6x on 22 cores where the GEMM reached
   * 13.6x, 39% of the ideal against the GEMM's 62%, which made it 24% of prefill
   * and the worst-scaling op in the tree.
   *
   * Ordering the blocks `0, blocks-1, 1, blocks-2, ...` makes the cumulative
   * cost linear in the unit index -- the first half of the sequence pairs a
   * cheap block with an expensive one, so any contiguous range carries about its
   * share -- and every worker then gets a similar total.  The mapping is a
   * permutation of which worker runs which unit; each unit's arithmetic, its
   * private scratch and the set of outputs it writes are untouched, so the
   * result is bit-identical and the thread-count test still holds.  A decode is
   * one block, so `blocks == 1` leaves it exactly where it was. */
  auto block_of = [blocks, groups](int64_t unit, int64_t &g) {
    const int64_t p = unit / groups;
    g = unit - p * groups;
    return (p & 1) != 0 ? (blocks - 1 - (p >> 1)) : (p >> 1);
  };

  /* ---- flash attention: one pass over the keys, no score row --------------
   *
   * Everything below materializes the score row: a store and a reload of
   * `q_len * max_span` floats per head, plus a separate sweep for the softmax.
   * That is the whole of the remaining gap to llama.cpp's shipped default,
   * which is flash attention -- it keeps a running max, denominator and
   * weighted sum and walks the keys once.
   *
   * The two paths differ in *how the span is cut*, and that is deliberate.
   * Batched prefill walks a query row's keys in order, keeping the shipped
   * kernel's row tiling (`dot_tile_r` shares one key row across
   * `kAttentionRows` query rows) because prefill is where that tiling pays.  A
   * one-token decode has only `n_head_kv` units -- the 8-unit limit the profile
   * keeps finding -- so it cuts the span into `kFlashSplit` chunks of
   * `kFlashBlock` keys and merges them with a log-sum-exp.  A span shorter than
   * one block is one chunk, so the two paths are the *same* reduction there,
   * which is what keeps a short prompt's batched result byte-identical to the
   * same tokens decoded one at a time.  At longer spans they agree to the
   * reassociation error and no closer, which is the price of the parallelism;
   * llama.cpp's own prefill and decode flash kernels differ the same way.
   */
  if (!scalar_weighted_sum_forced() && d <= kAttentionSumMaxDim) {
    constexpr int64_t kFlashBlock = 128;
    constexpr int64_t kFlashSplit = 8;
    const int64_t per_head = d + 2;
    const int64_t span = q_offset + q_len - first_key;
    const int64_t n_blk = (span + kFlashBlock - 1) / kFlashBlock;

    /* A widened K row in slot 0 and the matching V row in slot 1: flash needs
     * both live at once, where each shipped pass widened one at a time. */
    auto widen2 = [&](float *row_buf, const void *base, int64_t index, int64_t kv_head,
                      int slot) {
      if (f16) {
        float *const dst = row_buf + slot * d;
        kv_row_to_float(static_cast<const uint16_t *>(base) + index * kv_stride + kv_head * d, d,
                        dst);
        return static_cast<const float *>(dst);
      }
      return static_cast<const float *>(base) + index * kv_stride + kv_head * d;
    };

    /* Fold one key's score into a running (max, denominator, weighted sum).  A
     * key that does not raise the max costs one multiply-add per output
     * element; one that does pays the rescale of what has accumulated.  This is
     * the online softmax, and `P` is `[max, denominator, sum...]`. */
    if (q_len == 1) {
      const int64_t stride = group * per_head;
      const int64_t chunks = n_blk < 1 ? 1 : n_blk;
      /* The chunking is a function of the *span* and of nothing else, and that
       * is a correctness requirement and not tidiness.  An earlier draft took
       * the chunk count from what the caller's scratch could hold -- which the
       * backend sizes from `parallel_tasks`, and `parallel_tasks` is a function
       * of the thread count -- so the number of chunks, and with it the
       * reduction tree and the response's last bits, moved when the pool did.
       * The test that holds every kernel to the same bytes at one thread and at
       * eight caught it, which is what it is for.
       *
       * So the partials live here rather than in `scores`.  That costs the
       * allocation this path used to borrow, and it buys a chunk count that is
       * `kFlashSplit`-bounded but otherwise fixed, independent of the caller
       * and of the pool.  `kFlashSplit` is 8: decode attention has only
       * `n_head_kv` units, so 8 chunks is already 64 units on this model's 8 KV
       * heads, and more chunks split the same span into pieces too short to
       * amortize a merge. */
      const int64_t C = std::min<int64_t>(kFlashSplit, chunks);
      std::vector<float> partials(static_cast<std::size_t>(n_head_kv) *
                                  static_cast<std::size_t>(C) * static_cast<std::size_t>(stride));
      {
        parallel_for(n_head_kv * C, kAttentionGrain, [&](int64_t lo, int64_t hi, int64_t) {
        alignas(32) float row_buf[2 * kAttentionSumMaxDim];
        for (int64_t unit = lo; unit < hi; ++unit) {
          const int64_t u = unit / C;
          const int64_t c = unit - u * C;
          /* The `group` query heads of this KV head start at `h0`, not at the
           * front of `q`: head `h` reads KV head `h / group`, so the Q vector
           * for this unit is `q + (u * group + p) * d` and not `q + p * d`. */
          const float *const qh = q + (u * group) * d;
          float *const part = partials.data() + unit * stride;
          for (int64_t p = 0; p < group; ++p) {
            float *const P = part + p * per_head;
            P[0] = -INFINITY;
            P[1] = 0.0F;
            std::fill(P + 2, P + 2 + d, 0.0F);
          }
          for (int64_t b = (n_blk * c) / C; b < (n_blk * (c + 1)) / C; ++b) {
            const int64_t klo = first_key + b * kFlashBlock;
            const int64_t khi = std::min<int64_t>(q_offset, klo + kFlashBlock - 1);
            for (int64_t s = klo; s <= khi; ++s) {
              const float *const kvec = widen2(row_buf, k_cache, s, u, 0);
              float sc[kAttentionHeadBatch];
              if (group == 2) {
                dot_pair(qh, qh + d, kvec, d, sc);
              } else {
                sc[0] = dot4(qh, kvec, d);
              }
              const float *const vvec = widen2(row_buf, v_cache, s, u, 1);
              for (int64_t p = 0; p < group; ++p) {
                flash_fold(part + p * per_head, sc[p] * scale, vvec, d);
              }
            }
          }
        }
      });
      /* Merge the chunks.  The largest max is the exponent's reference point;
       * each chunk's denominator and sum are rescaled onto it. */
      parallel_for(n_head_kv, kAttentionGrain, [&](int64_t lo, int64_t hi, int64_t) {
        alignas(32) float gacc[kAttentionSumMaxDim];
        for (int64_t u = lo; u < hi; ++u) {
          const int64_t h0 = u * group;
          for (int64_t g = 0; g < group; ++g) {
            float gm = -INFINITY;
            for (int64_t c = 0; c < C; ++c) {
              const float mm = (partials.data() + (u * C + c) * stride + g * per_head)[0];
              if (mm > gm) {
                gm = mm;
              }
            }
            float gd = 0.0F;
            for (int64_t z = 0; z < d; ++z) {
              gacc[z] = 0.0F;
            }
            for (int64_t c = 0; c < C; ++c) {
              const float *const P = partials.data() + (u * C + c) * stride + g * per_head;
              const float w = std::exp(P[0] - gm);
              gd += P[1] * w;
              for (int64_t z = 0; z < d; ++z) {
                gacc[z] += P[2 + z] * w;
              }
            }
            const float inv = 1.0F / gd;
            float *const dst = out + (h0 + g) * d;
            for (int64_t z = 0; z < d; ++z) {
              dst[z] = gacc[z] * inv;
            }
          }
        }
      });
      return;
      }
    }

    /* Batched prefill: the shipped unit -- `kAttentionRows` query rows of one
     * batch of heads -- with the score row replaced by a per-row running triple.
     * The unit's own scratch is on its stack, so this path reads no caller
     * buffer at all; the row tiling and the causal tail keep exactly their
     * shipped structure. */
    parallel_for(blocks * groups, kAttentionGrain, [&](int64_t lo, int64_t hi, int64_t) {
      alignas(32) float row_buf[2 * kAttentionSumMaxDim];
      alignas(32) float part[kAttentionRows][kAttentionHeadBatch][kAttentionSumMaxDim + 2];
      for (int64_t unit = lo; unit < hi; ++unit) {
        int64_t g = 0;
        const int64_t b = block_of(unit, g);
        const int64_t t0 = b * kAttentionRows;
        const int64_t rows = std::min<int64_t>(kAttentionRows, q_len - t0);
        const int64_t h0 = g * kAttentionHeadBatch;
        const int64_t hb = std::min<int64_t>(kAttentionHeadBatch, n_heads - h0);
        const int64_t kv_head = h0 / group;
        const float *qvec[kAttentionHeadBatch][kAttentionRows];
        for (int64_t j = 0; j < rows; ++j) {
          for (int64_t p = 0; p < hb; ++p) {
            qvec[p][j] = q + ((t0 + j) * n_heads + h0 + p) * d;
            float *const P = part[j][p];
            P[0] = -INFINITY;
            P[1] = 0.0F;
            std::fill(P + 2, P + 2 + d, 0.0F);
          }
        }
        int64_t s = first_key;
        const int64_t shared_end = q_offset + t0;
        if (rows == kAttentionRows) {
          for (; s <= shared_end; ++s) {
            const float *const kvec = widen2(row_buf, k_cache, s, kv_head, 0);
            const float *const vvec = widen2(row_buf, v_cache, s, kv_head, 1);
            for (int64_t p = 0; p < hb; ++p) {
              float sc[kAttentionRows];
              dot_tile_r<kAttentionRows>(qvec[p], kvec, d, sc);
              for (int64_t j = 0; j < rows; ++j) {
                flash_fold(part[j][p], sc[j] * scale, vvec, d);
              }
            }
          }
        }
        if (s <= shared_end) {
          for (; s <= shared_end; ++s) {
            const float *const kvec = widen2(row_buf, k_cache, s, kv_head, 0);
            const float *const vvec = widen2(row_buf, v_cache, s, kv_head, 1);
            for (int64_t j = 0; j < rows; ++j) {
              float sc[kAttentionHeadBatch];
              if (hb == 2) {
                dot_pair(qvec[0][j], qvec[1][j], kvec, d, sc);
              } else {
                sc[0] = dot4(qvec[0][j], kvec, d);
              }
              for (int64_t p = 0; p < hb; ++p) {
                flash_fold(part[j][p], sc[p] * scale, vvec, d);
              }
            }
          }
        }
        for (int64_t k = 1; k < rows; ++k) {
          const int64_t abs = q_offset + t0 + k;
          const float *const kvec = widen2(row_buf, k_cache, abs, kv_head, 0);
          const float *const vvec = widen2(row_buf, v_cache, abs, kv_head, 1);
          for (int64_t j = k; j < rows; ++j) {
            float sc[kAttentionHeadBatch];
            if (hb == 2) {
              dot_pair(qvec[0][j], qvec[1][j], kvec, d, sc);
            } else {
              sc[0] = dot4(qvec[0][j], kvec, d);
            }
            for (int64_t p = 0; p < hb; ++p) {
              flash_fold(part[j][p], sc[p] * scale, vvec, d);
            }
          }
        }
        for (int64_t j = 0; j < rows; ++j) {
          for (int64_t p = 0; p < hb; ++p) {
            const float *const P = part[j][p];
            const float inv = 1.0F / P[1];
            float *const dst = out + ((t0 + j) * n_heads + h0 + p) * d;
            for (int64_t z = 0; z < d; ++z) {
              dst[z] = P[2 + z] * inv;
            }
          }
        }
      }
    });
    return;
  }

  parallel_for(blocks * groups, kAttentionGrain,
               [&](int64_t lo, int64_t hi, int64_t chunk) {
                 float *const task_scores = scores + chunk * kAttentionScoreRowsPerTask * max_span;
                 /* This task's own widened row -- see the note on `widen`.  A
                  * second row's worth exists so that a pass that ever needs two
                  * live rows at once does not have to grow this; today each pass
                  * widens, consumes and drops a single row. */
                 alignas(32) float row_buf[2 * kAttentionSumMaxDim];
                 auto widen = [&](const void *base, int64_t index, int64_t kv_head, int slot) {
                   if (f16) {
                     float *const dst = row_buf + slot * d;
                     kv_row_to_float(
                         static_cast<const uint16_t *>(base) + index * kv_stride + kv_head * d, d,
                         dst);
                     return static_cast<const float *>(dst);
                   }
                   return static_cast<const float *>(base) + index * kv_stride + kv_head * d;
                 };
                 for (int64_t unit = lo; unit < hi; ++unit) {
                   int64_t g = 0;
                   const int64_t b = block_of(unit, g);
                   const int64_t t0 = b * kAttentionRows;
                   const int64_t rows = std::min<int64_t>(kAttentionRows, q_len - t0);
                   /* The batch is `kAttentionHeadBatch` consecutive query heads
                    * that share a KV head.  They share one because attention is
                    * grouped and the batch never crosses a group boundary: head
                    * `h` reads KV head `h / group`, so any run of heads shorter
                    * than `group` lies inside one group.  `hb` handles a head
                    * count that is not a multiple of the batch. */
                   const int64_t h0 = g * kAttentionHeadBatch;
                   const int64_t hb = std::min<int64_t>(kAttentionHeadBatch, n_heads - h0);
                   const int64_t kv_head = h0 / group;
                   /* Causal: the query at absolute position `q_offset + t0 + j`
                    * sees cache rows `first_key .. q_offset + t0 + j`.
                    * `first_key` is where the cache's live span begins, which is
                    * always 0 today -- a sliding-window variant would move it and
                    * nothing else here would change. */
                   const float *qvec[kAttentionHeadBatch][kAttentionRows];
                   float *row_scores[kAttentionHeadBatch][kAttentionRows];
                   float max_score[kAttentionHeadBatch][kAttentionRows];
                   for (int64_t p = 0; p < hb; ++p) {
                     for (int64_t j = 0; j < rows; ++j) {
                       qvec[p][j] = q + ((t0 + j) * n_heads + h0 + p) * d;
                       row_scores[p][j] = task_scores + (p * kAttentionRows + j) * max_span;
                       max_score[p][j] = -INFINITY;
                     }
                   }

                   /* One `dot4` per batched head against one key vector, where
                    * `dot_pair` does two of them in a single walk over the key
                    * row.  This is the shape attention spends its time in:
                    * decode is `rows == 1`, and the score pass is `q_len * span`
                    * of these per head.  Leave it a `dot4` per head and the key
                    * row is read `group` times -- see `kAttentionHeadBatch`. */
                   auto score_row = [&](int64_t j, const float *kvec,
                                        float sc[kAttentionHeadBatch]) {
                     if (hb == 2) {
                       dot_pair(qvec[0][j], qvec[1][j], kvec, d, sc);
                     } else {
                       sc[0] = dot4(qvec[0][j], kvec, d);
                     }
                   };
                   /* The keys every row of the block can see, `kAttentionRows` at
                    * a time. */
                   int64_t s = first_key;
                   const int64_t shared_end = q_offset + t0;
                   if (rows == kAttentionRows) {
                     for (; s <= shared_end; ++s) {
                       const float *kvec = widen(k_cache, s, kv_head, 0);
                       for (int64_t p = 0; p < hb; ++p) {
                         float sc[kAttentionRows];
                         dot_tile_r<kAttentionRows>(qvec[p], kvec, d, sc);
                         for (int64_t j = 0; j < rows; ++j) {
                           const float score = sc[j] * scale;
                           row_scores[p][j][s - first_key] = score;
                           if (score > max_score[p][j]) {
                             max_score[p][j] = score;
                           }
                         }
                       }
                     }
                   }
                   if (s <= shared_end) {
                     /* The ragged block -- the tail of the chunk, or a decode
                      * step -- or the one-row decode. The rows see different
                      * spans here, so each takes one dot per row over the keys
                      * it shares with the rest, the batched heads together. */
                     for (; s <= shared_end; ++s) {
                       const float *kvec = widen(k_cache, s, kv_head, 0);
                       for (int64_t j = 0; j < rows; ++j) {
                         float sc[kAttentionHeadBatch];
                         score_row(j, kvec, sc);
                         for (int64_t p = 0; p < hb; ++p) {
                           const float score = sc[p] * scale;
                           row_scores[p][j][s - first_key] = score;
                           if (score > max_score[p][j]) {
                             max_score[p][j] = score;
                           }
                         }
                       }
                     }
                   }
                   /* The remainder of each row's span, past what the whole block
                    * shares: row `j` still needs `q_offset + t0 + 1 .. q_offset +
                    * t0 + j`, which is `j` keys. */
                   for (int64_t k = 1; k < rows; ++k) {
                     const int64_t abs = q_offset + t0 + k;
                     const float *kvec = widen(k_cache, abs, kv_head, 0);
                     for (int64_t j = k; j < rows; ++j) {
                       float sc[kAttentionHeadBatch];
                       score_row(j, kvec, sc);
                       for (int64_t p = 0; p < hb; ++p) {
                         const float score = sc[p] * scale;
                         row_scores[p][j][abs - first_key] = score;
                         if (score > max_score[p][j]) {
                           max_score[p][j] = score;
                         }
                       }
                     }
                   }

                   for (int64_t p = 0; p < hb; ++p) {
                     for (int64_t j = 0; j < rows; ++j) {
                       const int64_t span = q_offset + t0 + j - first_key + 1;
                       /* Shifted by the max, as the reference `softmax` is: the
                        * exponentials of a 1000-scale score row would otherwise all
                        * be zero and the row would normalize to 0/0.  The
                        * normalization is folded into the stored weights here
                        * rather than applied at the use site, so the weighted sum
                        * below reads one value per (row, key) either way. */
                       float total = 0.0F;
                       for (int64_t i = 0; i < span; ++i) {
                         const float w = std::exp(row_scores[p][j][i] - max_score[p][j]);
                         row_scores[p][j][i] = w;
                         total += w;
                       }
                       const float inv_total = 1.0F / total;
                       for (int64_t i = 0; i < span; ++i) {
                         row_scores[p][j][i] *= inv_total;
                       }
                     }
                   }

                   /* The untiled form is the fallback for two callers: the test
                    * that asks for it, and a head dimension wider than the
                    * tile's stack bound. */
                   if (scalar_weighted_sum_forced() || d > kAttentionSumMaxDim) {
                     for (int64_t p = 0; p < hb; ++p) {
                       for (int64_t j = 0; j < rows; ++j) {
                         float *dst = out + ((t0 + j) * n_heads + h0 + p) * d;
                         std::fill(dst, dst + d, 0.0F);
                         const float *wrow = row_scores[p][j];
                         const int64_t span = q_offset + t0 + j - first_key + 1;
                         for (int64_t i = 0; i < span; ++i) {
                           const float weight = wrow[i];
                           const float *vvec = widen(v_cache, first_key + i, kv_head, 0);
                           for (int64_t x = 0; x < d; ++x) {
                             dst[x] += weight * vvec[x];
                           }
                         }
                       }
                     }
                     continue;
                   }

                   /* The weighted sum, `kAttentionRows` output rows per walk
                    * over a V row.
                    *
                    * This pass had no work done to it when the score pass got
                    * its tiling, and it is the *second* largest term of the
                    * attention call: for every query row it streamed the whole
                    * V slab and did `dst[x] += weight * vvec[x]`, two memory
                    * operations per one FMA whose constants are a broadcast
                    * register.  The tiling is the same idea the score pass
                    * already uses, applied to the other end of the call.
                    *
                    * **The split between the two regions is the whole
                    * correctness argument and the first draft of the probe got
                    * it backwards.**  Row `j` owns keys `[0, span0 + j)`, where
                    * `span0 = q_offset + t0 - first_key + 1` is what every row
                    * of the tile shares.  So the *shared prefix* `[0, span0)` is
                    * walked with the V row outer and the rows inner -- the only
                    * shape in which the V load is shared at all -- and the
                    * *triangular remainder* `[span0, span0 + j)` is walked per
                    * row, because no other row sees those keys.  A version that
                    * looped `j` outside `i` gave every row only its own region
                    * and produced fluent output that was wrong by 0.8 of its own
                    * scale.
                    *
                    * The store to `out` is deferred to the end of the tile
                    * rather than done in row order, so the tile's results are
                    * copied once per row instead of once per key.  Both
                    * differences are silent -- no race, no NaN, just a different
                    * number -- which is why the two tests in
                    * `tests/native/test_cpu_parallel.py` compare the untiled
                    * path's token sequence and its bytes.
                    *
                    * `kAttentionRows * d` floats of scratch is what this costs:
                    * 2 KiB at the shipped constant, on the task's own stack. */
                   for (int64_t p = 0; p < hb; ++p) {
                     for (int64_t j0 = 0; j0 < rows; j0 += kAttentionRows) {
                       const int64_t jn = std::min<int64_t>(kAttentionRows, rows - j0);
                       /* `kAttentionSumMaxDim` and not `d`: the rows of this array
                        * are not `d` apart, so a single `fill` over `jn * d` floats
                        * from `&acc[0][0]` would zero only the first two rows of a
                        * four-row tile and leave the rest holding whatever the
                        * stack had.  That is a NaN-shaped bug, not a slow one. */
                       alignas(32) float acc[kAttentionRows][kAttentionSumMaxDim];
                       for (int64_t j = 0; j < jn; ++j) {
                         std::fill(acc[j], acc[j] + d, 0.0F);
                       }
                       const int64_t span0 = q_offset + t0 + j0 - first_key + 1;
                       for (int64_t i = 0; i < span0; ++i) {
                         const float *vvec = widen(v_cache, first_key + i, kv_head, 0);
                         for (int64_t j = 0; j < jn; ++j) {
                           const float weight = row_scores[p][j0 + j][i];
                           for (int64_t x = 0; x < d; ++x) {
                             acc[j][x] += weight * vvec[x];
                           }
                         }
                       }
                       for (int64_t j = 1; j < jn; ++j) {
                         const int64_t span = span0 + j;
                         const float *wrow = row_scores[p][j0 + j];
                         for (int64_t i = span0; i < span; ++i) {
                           const float weight = wrow[i];
                           const float *vvec = widen(v_cache, first_key + i, kv_head, 0);
                           for (int64_t x = 0; x < d; ++x) {
                             acc[j][x] += weight * vvec[x];
                           }
                         }
                       }
                       for (int64_t j = 0; j < jn; ++j) {
                         std::memcpy(out + ((t0 + j0 + j) * n_heads + h0 + p) * d, acc[j],
                                     static_cast<std::size_t>(d) * 4);
                       }
                     }
                   }
                 }
               });
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

void softmax(const float *x, float *out, int64_t rows, int64_t cols) {
  for (int64_t r = 0; r < rows; ++r) {
    softmax_row<float>(x + r * cols, out + r * cols, cols);
  }
}

void logits_temperature(const float *logits, float *out, int64_t n, float temperature) {
  if (!(temperature > 0.0F)) {
    throw Error("logits_temperature: temperature must be positive, got " +
                std::to_string(temperature));
  }
  for (int64_t i = 0; i < n; ++i) {
    out[i] = logits[i] / temperature;
  }
}

void topk_sample(const float *logits, int64_t vocab, float uniform, int64_t top_k, float top_p,
                 float min_p, int64_t *order, int64_t *out) {
  if (vocab <= 0) {
    throw Error("topk_sample: need at least one logit");
  }

  /* The same shifted softmax the `softmax` op computes, in double -- see the
   * helper for why the two callers take different accumulators. `probs[i]` is
   * the probability of token `i`, in token order. */
  std::vector<double> probs(static_cast<std::size_t>(vocab));
  softmax_row<double>(logits, probs.data(), vocab);

  /* Then ranked by probability, which is what the reference's
   * `argsort(-probs, kind="stable")` does. `ranked[i]` is the probability of
   * token `order[i]`. */
  rank_descending<double>(probs.data(), vocab, order);
  std::vector<double> ranked(static_cast<std::size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    ranked[static_cast<std::size_t>(i)] = probs[static_cast<std::size_t>(order[i])];
  }

  /* Truncation over the ranked list, each rule the reference's: top-k keeps the
   * first k, min-p drops below `min_p * ranked[0]` (inclusive, so `min_p=1`
   * keeps the argmax), and top-p keeps the smallest prefix whose cumulative mass
   * reaches p. */
  std::vector<uint8_t> keep(static_cast<std::size_t>(vocab), 1);
  if (top_k > 0 && top_k < vocab) {
    std::fill(keep.begin() + static_cast<std::ptrdiff_t>(top_k), keep.end(), static_cast<uint8_t>(0));
  }
  if (min_p > 0.0F) {
    const double cutoff = static_cast<double>(min_p) * ranked[0];
    for (int64_t i = 0; i < vocab; ++i) {
      if (!(ranked[static_cast<std::size_t>(i)] >= cutoff)) {
        keep[static_cast<std::size_t>(i)] = 0;
      }
    }
  }
  if (top_p < 1.0F) {
    /* The reference keeps `i` while the mass strictly above it is below p, and
     * also the first index that *reaches* p; when nothing reaches it the kept
     * index is 0, which is numpy's `argmax` of an all-false mask. The crossing
     * is found as a running sum in one pass, not a per-index prefix -- an O(n^2)
     * scan here would be a quarter of a minute per token at this vocabulary. */
    int64_t crossing = 0;
    {
      double cumulative = 0.0;
      for (int64_t i = 0; i < vocab; ++i) {
        cumulative += ranked[static_cast<std::size_t>(i)];
        if (cumulative >= static_cast<double>(top_p)) {
          crossing = i;
          break;
        }
      }
    }
    double above = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      const bool within = above < static_cast<double>(top_p);
      if (!within && i != crossing) {
        keep[static_cast<std::size_t>(i)] = 0;
      }
      above += ranked[static_cast<std::size_t>(i)];
    }
  }

  /* Renormalize over what was kept, then invert the CDF at `uniform` -- the
   * first index whose cumulative mass reaches the draw, numpy's `side="left"`. */
  double total_kept = 0.0;
  for (int64_t i = 0; i < vocab; ++i) {
    if (keep[static_cast<std::size_t>(i)]) {
      total_kept += ranked[static_cast<std::size_t>(i)];
    }
  }
  if (!(total_kept > 0.0)) {
    *out = order[0];
    return;
  }
  double cumulative = 0.0;
  int64_t chosen = vocab - 1;
  for (int64_t i = 0; i < vocab; ++i) {
    if (!keep[static_cast<std::size_t>(i)]) {
      continue;
    }
    cumulative += ranked[static_cast<std::size_t>(i)] / total_kept;
    if (cumulative >= static_cast<double>(uniform)) {
      chosen = i;
      break;
    }
  }
  *out = order[chosen];
}

/* Widen one f16 cache row into `dst` as floats.
 *
 * **This is how the f16 cache is read, and the decision to materialize rather
 * than to write a second family of dot kernels is the design.**  Every score
 * kernel in this file -- `dot`, `dot4`, `dot_pair`, `dot_tile_r` -- is the result
 * of a measured correctness argument about lane structure, and the score dot is
 * the one place a last-bit change flips tokens (see `dot4`).  A parallel
 * `_f16` family would double that surface and put two kernels under one
 * argument.  Widening first means a cache row is read by exactly the kernels an
 * f32 cache is, so the *only* new claim in the whole f16 path is "this
 * expansion equals each element's width", which is one line of hardware.
 *
 * It is also where the bytes go.  A head row is `d` = 128 halves = 256 bytes,
 * and the widened row is 512 bytes -- which is L1-resident five times over, so
 * the row is streamed from the cache once and then read and re-read out of L1
 * by whatever tiling the kernel uses.  `dot_pair` reads a key row twice (once
 * per query head) and `dot_tile_r` reads it once for `kAttentionRows` rows; both were
 * reading the same 512-byte row in the f32 case already, so this is the access
 * pattern the kernels were written against, not a new one.
 *
 * The tail below `AVX2` is `half_to_float` per element, which is where that
 * function is exercised: the vector branch is only compiled where `_mm_cvtph_ps`
 * exists, and the two agree bit for bit. */
void kv_row_to_float(const void *row, int64_t d, float *dst) {
#if POCKETLLM_HAVE_F16C
  const uint16_t *src = static_cast<const uint16_t *>(row);
  int64_t i = 0;
  for (; i + 8 <= d; i += 8) {
    const __m128i lo_bits = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i));
    const __m128i hi_bits = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + i + 4));
    _mm_storeu_ps(dst + i, _mm_cvtph_ps(lo_bits));
    _mm_storeu_ps(dst + i + 4, _mm_cvtph_ps(hi_bits));
  }
  for (; i < d; ++i) {
    dst[i] = half_to_float(src[i]);
  }
#else
  const uint16_t *src = static_cast<const uint16_t *>(row);
  for (int64_t i = 0; i < d; ++i) {
    dst[i] = half_to_float(src[i]);
  }
#endif
}

/* Narrow one f32 row into an f16 cache row.  The round trip is what makes the
 * cache a *lossy* copy of the f32 K/V the layer computes, and that is the same
 * loss the oracle's default basis takes -- see `Backend::attention`. */
void float_to_kv_row(const float *src, int64_t d, void *row) {
#if POCKETLLM_HAVE_F16C
  uint16_t *dst = static_cast<uint16_t *>(row);
  int64_t i = 0;
  for (; i + 8 <= d; i += 8) {
    const __m128i lo = _mm_cvtps_ph(_mm_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT);
    const __m128i hi = _mm_cvtps_ph(_mm_loadu_ps(src + i + 4), _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + i), lo);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + i + 4), hi);
  }
  for (; i < d; ++i) {
    /* One at a time through a lane rather than a masked store: `d` is 128 for
     * every model this runs, so the tail is dead code here and the cost of it
     * being slow is zero.  It exists so a width that is not a multiple of eight
     * is wrong in no way. */
    const __m128i one = _mm_cvtps_ph(_mm_set_ss(src[i]), _MM_FROUND_TO_NEAREST_INT);
    dst[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(one) & 0xFFFF);
  }
#else
  (void)src;
  (void)d;
  (void)row;
  throw Error("float_to_kv_row: this build has no f16 conversion (needs F16C)");
#endif
}

}  // namespace kernel
}  // namespace pocketllm