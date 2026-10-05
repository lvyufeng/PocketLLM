#include "kernel/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__AVX2__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define POCKETLLM_HAVE_AVX2 1
#else
#define POCKETLLM_HAVE_AVX2 0
#endif

#include "kernel/parallel.h"
#include "quant/blocks.h"
#include "quant/q8k.h"
#include "runtime/status.h"

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
   * The buffer is on this function's frame because the graph's shapes are
   * small -- the widest call is `m=32, k=3072`, 384 blocks -- and this is not
   * on a recursion path.  A shape that does not fit takes the exact path
   * instead: correct, only slower, and the check is here rather than an
   * assumption because `gemm_quant` is reachable from the ABI with shapes this
   * graph does not produce. */
  constexpr int64_t kMaxStackBlocks = 1024; /* 299 KiB of activation blocks */
  const bool quantize_activations = !exact_gemm_forced() && m * row_blocks <= kMaxStackBlocks;
  alignas(64) quant::Q8KBlock q8[kMaxStackBlocks];
  if (quantize_activations) {
    quant::quantize_row_q8_k(x, q8, m * k);
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
  /* Elementwise: every output depends on its own index only. */
  parallel_for(n, /*min_per_task=*/4096, [&](int64_t lo, int64_t hi, int64_t) {
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

void attention(const float *q, int64_t q_len, int64_t n_heads, const float *k_cache,
               const float *v_cache, int64_t n_head_kv, int64_t d, int64_t first_key,
               int64_t q_offset, float scale, float *out, float *scores) {
  const int64_t group = n_heads / n_head_kv;
  const int64_t cache_row = n_head_kv * d;
  /* The score row for a query at position `q_offset + q_len - 1` is the widest
   * any row can be, and it is what spaces the per-task scratch regions apart.
   * `attention_scratch_rows` and the callers that size the buffer use the same
   * quantity. */
  const int64_t max_span = q_offset + q_len - first_key;

  /* Over `(token, head)`.  Each unit owns one output row and one score row, so
   * the scratch is private to the task that writes it -- `scores` is a shared
   * buffer handed in by the caller, and two tasks writing the same row would be
   * the classic race whose symptom is fluent, finite, wrong output.  The
   * per-task region is `scores + chunk * max_span`; nothing else is shared, and
   * the dot, the max scan and the weighted sum inside a unit stay serial so the
   * arithmetic is the same as the single-threaded path.
   *
   * The unit is also the *whole* of a task's work: there is nothing finer to
   * split, because the three passes inside it are chained -- `max_score` is
   * needed before the exponentials and `total` before the weighted sum -- so a
   * second thread on the same unit could only wait.  `kAttentionGrain` is 1 for
   * that reason; see its comment in the header. */
  parallel_for(q_len * n_heads, kAttentionGrain,
               [&](int64_t lo, int64_t hi, int64_t chunk) {
                 float *const row_scores = scores + chunk * max_span;
                 for (int64_t unit = lo; unit < hi; ++unit) {
                   const int64_t t = unit / n_heads;
                   const int64_t h = unit - t * n_heads;
                   /* Causal: the query at absolute position `q_offset + t` sees
                    * cache rows `first_key .. q_offset + t`. `first_key` is where
                    * the cache's live span begins, which is always 0 today -- a
                    * sliding-window variant would move it and nothing else here
                    * would change. */
                   const int64_t end = q_offset + t;
                   const int64_t span = end - first_key + 1;
                   const float *qvec = q + (t * n_heads + h) * d;
                   const int64_t kv_head = h / group;

                   float max_score = -INFINITY;
                   for (int64_t s = 0; s < span; ++s) {
                     const float *kvec = k_cache + (first_key + s) * cache_row + kv_head * d;
                     const float score = dot(qvec, kvec, d) * scale;
                     row_scores[s] = score;
                     if (score > max_score) {
                       max_score = score;
                     }
                   }

                   /* Shifted by the max, as the reference `softmax` is: the
                    * exponentials of a 1000-scale score row would otherwise all
                    * be zero and the row would normalize to 0/0. */
                   float total = 0.0F;
                   for (int64_t s = 0; s < span; ++s) {
                     const float w = std::exp(row_scores[s] - max_score);
                     row_scores[s] = w;
                     total += w;
                   }

                   float *dst = out + (t * n_heads + h) * d;
                   std::fill(dst, dst + d, 0.0F);
                   const float inv_total = 1.0F / total;
                   for (int64_t s = 0; s < span; ++s) {
                     const float weight = row_scores[s] * inv_total;
                     const float *vvec = v_cache + (first_key + s) * cache_row + kv_head * d;
                     for (int64_t i = 0; i < d; ++i) {
                       dst[i] += weight * vvec[i];
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

}  // namespace kernel
}  // namespace pocketllm