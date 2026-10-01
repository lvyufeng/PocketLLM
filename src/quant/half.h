/* Half-precision conversion, for reading f16 and bf16 weights.
 *
 * Every weight in a GGUF that is not quantized is stored in one of these two
 * formats, so the dense path needs both before it needs anything else. The
 * conversions are bit-exact -- a soft path for f16 because the host may not
 * have an f16 type, a shift for bf16 because that is all it is -- and they are
 * deliberately not vectorized: this is the loader's cost, paid once per weight,
 * not the kernel's.
 *
 * `float` is assumed to be IEEE 754 binary32, which is what every platform this
 * builds on provides and what the bit patterns below encode. A host where that
 * is false would fail the reader test long before this arithmetic mattered.
 */

#ifndef POCKETLLM_QUANT_HALF_H
#define POCKETLLM_QUANT_HALF_H

#include <cstdint>
#include <cstring>

namespace pocketllm {

/* The little-endian u16 at `p`.  GGUF is little-endian on disk, which is the
 * byte order of every host this runs on; a memcpy rather than a cast because a
 * misaligned load is undefined and a packed tensor's first element is exactly
 * that. */
inline uint16_t load_u16(const uint8_t *p) {
  uint16_t value = 0;
  std::memcpy(&value, p, sizeof(value));
  return value;
}

/* IEEE 754 binary16 -> binary32.
 *
 * The three cases are the whole of the format: zero and the subnormals (biased
 * exponent 0), the infinities and NaNs (31), and the ordinary numbers. The
 * subnormal branch renormalizes, which is why it is a loop rather than a
 * formula -- a hardcoded shift would be right for one input and wrong for the
 * rest.
 *
 * A subnormal h with mantissa m has value m * 2^-24; left-shifting until the
 * implicit leading bit is at position 10 makes it (1.f) * 2^(p-24) for a top
 * set bit p, whose float32 exponent field is p + 103. That is the `e` the loop
 * lands on, and `m & 0x3FF` after the shift is the fraction. */
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x03FFu;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign; /* +-0 */
    } else {
      uint32_t e = 127u - 15u + 1u;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --e;
      }
      mantissa &= 0x03FFu;
      bits = sign | (e << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13); /* inf, or a NaN payload */
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }

  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/* bfloat16 -> binary32.  The format is the top 16 bits of a float32, so the
 * conversion is exact by construction and needs no rounding: the low 16 bits
 * are zeros, which is what a left shift by 16 gives. */
inline float bf16_to_float(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace pocketllm

#endif /* POCKETLLM_QUANT_HALF_H */