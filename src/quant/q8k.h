/* The int8 activation block -- `block_q8_K` -- and the quantizer that makes it.
 *
 * A packed weight block holds a coarse value per weight, so a dot against a
 * float activation costs a decode per weight before any arithmetic happens.
 * llama.cpp's answer, and now this engine's, is to quantize the *activation*
 * instead: once per 256-weight block, to int8 with a single scale, and then take
 * the short path through both operands -- unpack the weight to 4 or 6 unsigned
 * bits, multiply against the signed activation byte with `_mm256_maddubs_epi16`,
 * fold the block's scale in with `_mm256_madd_epi16`, and apply the one float
 * scale per block.  Every product inside the dot is then exact integer
 * arithmetic, and the only rounding the product adds beyond the checkpoint's own
 * is the activation's.
 *
 * **That last part is a real precision change, not a last-bit one, and it is
 * stated rather than buried.** The activation's scale comes from the
 * largest-magnitude element of the block, so an activation element picks up up
 * to half a step of that scale as error.  Measured on this engine's conformance
 * shapes it moves the product by 0.004-0.006 of the output's magnitude, against
 * a `QUANTIZED_RTOL` of 0.05 -- an order of margin.  It is also, exactly, what
 * llama.cpp does, which is the point: `test_quantized_forward.py` records that
 * the two engines' quantized logits cannot be compared elementwise *because*
 * llama.cpp quantizes its activation and this tree did not.  After this they run
 * the same arithmetic.
 *
 * The layout is llama.cpp's `ggml-common.h`, and the quantizer is a port of
 * `quantize_row_q8_K_ref` (`ggml/src/ggml-quants.c`), term for term.  Being a
 * port matters more here than anywhere else in `quant/`: the two engines are
 * compared token for token, so the quantization decision -- which element sets
 * the scale, how the product rounds -- has to be the same decision and not
 * merely a similar one.
 *
 *     block_q8_K:  d(fp32)  qs[256 int8]  bsums[16 int16]      -- 292 bytes
 *
 * `bsums[j]` is the sum of `qs[16j .. 16j+15]`, which is what lets a weight's
 * per-group *minimum* be applied with one `madd` per block instead of a
 * per-weight subtraction: the correction is `min * sum(q8)` over the group, and
 * the sum was computed here for free.
 */

#ifndef POCKETLLM_QUANT_Q8K_H
#define POCKETLLM_QUANT_Q8K_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace pocketllm {
namespace quant {

/* Weights in an activation block.  The same 256 as a k-quant super-block, and
 * deliberately so: the two operands of a packed dot are blocked together, so a
 * weight block and the activation block it multiplies are aligned. */
constexpr int kQ8KWeights = 256;

/* The quantized activation block, byte for byte llama.cpp's `block_q8_K`.  The
 * members are in the file order of the format, which for this one is also the
 * order the vector kernel reads them in. */
struct Q8KBlock {
  float d;
  int8_t qs[kQ8KWeights];
  int16_t bsums[kQ8KWeights / 16];
};

static_assert(sizeof(Q8KBlock) == 292, "block_q8_K is 4 + 256 + 32 bytes");

/* Round to nearest, ties to even -- llama.cpp's `nearest_int`, kept as the
 * magic-number form rather than written with `nearbyint`, for two reasons.  The
 * result must be the reference's *decision*, so a rounding that differs on the
 * tie it happens to hit is a different token two hundred steps later; and the
 * trick is what the reference compiles to, so there is no risk of a library call
 * with a rounding mode this code does not set.  `12582912.0f` is 1.5 * 2^23:
 * adding it puts the value's integer part into the float's mantissa low bits,
 * which the mask then extracts. */
inline int nearest_int(float value) {
  const float shifted = value + 12582912.0F;
  int bits = 0;
  std::memcpy(&bits, &shifted, sizeof(bits));
  return (bits & 0x007FFFFF) - 0x00400000;
}

/* Quantize `k` floats -- a whole multiple of 256 -- into `out`, one
 * :c:type:`Q8KBlock` per 256.
 *
 * The scale is `-127 / max`, where `max` is the *signed* element of the largest
 * magnitude.  The negation is not a sign convention: it makes the largest-magnitude
 * element land on -127 (or +127, for a positive `max`), which is the int8 range's
 * edge, so the block uses all seven bits of precision rather than seven minus a
 * sign.  `d` is `1 / iscale`, the float the kernel multiplies back in.
 *
 * A block whose elements are all zero quantizes to `d = 0` with a zeroed `qs` --
 * the same answer the reference gives, and the one that keeps a dead activation
 * from dividing by zero. */
inline void quantize_row_q8_k(const float *x, Q8KBlock *out, int64_t k) {
  const int64_t n_blocks = k / kQ8KWeights;
  for (int64_t b = 0; b < n_blocks; ++b) {
    const float *row = x + b * kQ8KWeights;
    Q8KBlock &block = out[b];

    float max = 0.0F;
    float amax = 0.0F;
    for (int j = 0; j < kQ8KWeights; ++j) {
      const float magnitude = std::fabs(row[j]);
      if (magnitude > amax) {
        amax = magnitude;
        max = row[j];
      }
    }
    if (amax == 0.0F) {
      block.d = 0.0F;
      std::memset(block.qs, 0, sizeof(block.qs));
      std::memset(block.bsums, 0, sizeof(block.bsums));
      continue;
    }

    const float iscale = -127.0F / max;
    for (int j = 0; j < kQ8KWeights; ++j) {
      /* The clamp is defensive: `|iscale * x[j]| <= 127` by construction, so the
       * only way past it is a rounding at the top of the range. */
      const int value = nearest_int(iscale * row[j]);
      block.qs[j] = static_cast<int8_t>(value > 127 ? 127 : value);
    }
    for (int j = 0; j < kQ8KWeights / 16; ++j) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        sum += block.qs[j * 16 + i];
      }
      block.bsums[j] = static_cast<int16_t>(sum);
    }
    block.d = 1.0F / iscale;
  }
}

}  // namespace quant
}  // namespace pocketllm

#endif /* POCKETLLM_QUANT_Q8K_H */
