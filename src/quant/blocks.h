/* The k-quant super-block formats, decoded one weight at a time.
 *
 * These are the formats `python/pocketllm/quant/k_quants.py` decodes with
 * numpy, and this file is the same arithmetic written the way the *kernel*
 * needs it. The Python decoders return a whole block as an array; that shape is
 * right for a loader and wrong for a packed GEMM, which never materializes a
 * row of weights at all -- it walks the block once per output element and
 * consumes each decoded value immediately.
 *
 * So the unit here is a single weight, addressed by its index within the
 * super-block, and the decoders are `POCKETLLM_HD` so the same source compiles
 * for the host and for a CUDA kernel. That single-source property is the point:
 * a device decoder written as a second transcription of the Python would be a
 * third place for the bit layout to be wrong, and the two C decoders would
 * agree with each other exactly where they shared a misreading.
 *
 * What is deliberately *not* shared is `kernel/kernels.cpp`. Those functions
 * are the reference the device kernels are checked against, and a reference
 * that shares its implementation with the thing it verifies cannot tell you
 * which of the two is wrong. A bit layout is different: it is a property of the
 * file, not of an algorithm, and two implementations of it can only disagree by
 * one of them being incorrect.
 *
 * The layouts, from `ggml-common.h`, in the order the bytes appear:
 *
 *   block_q4_K:  d(fp16) dmin(fp16) scales[12] qs[128]          -- 144 bytes
 *   block_q6_K:  ql[128] qh[64] scales[16 int8] d(fp16)        -- 210 bytes
 *
 * Both carry 256 weights. `q4_K` splits them into eight groups of 32, each with
 * a 6-bit scale and a 6-bit minimum packed into the twelve `scales` bytes;
 * `q6_K` splits them into sixteen groups of 16, each with an int8 scale.
 */

#ifndef POCKETLLM_QUANT_BLOCKS_H
#define POCKETLLM_QUANT_BLOCKS_H

#include <cstdint>

#include "quant/half.h"

/* The CPU translation unit and the CUDA one both include this. `POCKETLLM_HD`
 * is the marker that decides which callable a definition gets: plain `inline`
 * on the host, `__host__ __device__` under nvcc. nvcc defines `__CUDACC__` for
 * every file it compiles, including the host pass, so the test is the compiler
 * and not the target. */
#ifdef __CUDACC__
#define POCKETLLM_HD __host__ __device__
#else
#define POCKETLLM_HD inline
#endif

namespace pocketllm {
namespace quant {

/* Weights in a k-quant super-block. */
constexpr int kBlockWeights = 256;

/* Block sizes in bytes, matching `k_quants.py`'s constants of the same names. */
constexpr int kQ4KBlockBytes = 144;
constexpr int kQ6KBlockBytes = 210;

/* The byte at `p`, as an index. GGUF is little-endian and so is every host this
 * runs on, which is the assumption `half.h` already states for its u16 load. */
POCKETLLM_HD int as_byte(const uint8_t *p, int offset) {
  return static_cast<int>(p[offset]);
}

/* The fp16 at `p + offset`, widened. */
POCKETLLM_HD float as_half(const uint8_t *p, int offset) {
  return half_to_float(static_cast<uint16_t>(as_byte(p, offset) | (as_byte(p, offset + 1) << 8)));
}

/* The signed byte at `p + offset`. */
POCKETLLM_HD int as_int8(const uint8_t *p, int offset) {
  const int value = as_byte(p, offset);
  return value >= 128 ? value - 256 : value;
}

/* The 6-bit scale/min pair `j` out of a Q4_K block's twelve packed bytes.
 *
 * This is `ggml`'s `get_scale_min_k4`, and the first four pairs are the easy
 * case: byte `j` holds the scale in its low six bits and byte `j + 4` the
 * minimum. The upper four pairs have no bytes of their own left -- the twelve
 * are the whole budget -- so they borrow the *top two bits* of the first four
 * pairs' bytes and take their own low nibbles from bytes 8..11. That is why the
 * second branch reads offsets both above and below `j`, and why an
 * implementation that treated all eight pairs alike would decode the first half
 * of every block correctly and the second half as noise.
 *
 * The C reference kernel's version of this is `dequant_q4_k`'s loop below, not
 * a separate function: it is used eight times per block and inlining it by hand
 * is one less call in a loop that runs 256 times per row. */
POCKETLLM_HD void get_scale_min_k4(const uint8_t *scales, int j, int *scale, int *minimum) {
  if (j < 4) {
    *scale = as_byte(scales, j) & 63;
    *minimum = as_byte(scales, j + 4) & 63;
    return;
  }
  *scale = (as_byte(scales, j + 4) & 0x0F) | ((as_byte(scales, j - 4) >> 6) << 4);
  *minimum = (as_byte(scales, j + 4) >> 4) | ((as_byte(scales, j) >> 6) << 4);
}

/* One weight of a Q4_K super-block: `d * scale * q - dmin * minimum`.
 *
 * `index` is 0..255. The block holds four 32-byte runs of packed nibbles, and
 * each run serves *two* groups of 32 -- the low nibbles belong to the earlier
 * group and the high nibbles to the later one. So the run is `index / 64` and
 * which nibble to take is decided by `(index / 32) % 2`, not by `index / 32`
 * alone: reading all of a run's low nibbles and then all of its high ones is
 * the same thing as interleaving them 32 at a time, and only the interleaved
 * order puts them back where the quantizer wrote them. */
POCKETLLM_HD float dequant_q4_k(const uint8_t *block, int index) {
  const float d = as_half(block, 0);
  const float dmin = as_half(block, 2);
  const int group = index / 32;          /* which of the eight 32-weight groups */
  const int within = index % 32;         /* position inside that group */
  const int run = group / 2;             /* which 32-byte packed run */
  const int high = group % 2;            /* low or high nibble of that run */
  const int q = as_byte(block, 16 + run * 32 + within);
  const int nibble = high ? (q >> 4) : (q & 0x0F);
  int scale = 0;
  int minimum = 0;
  get_scale_min_k4(block + 4, group, &scale, &minimum);
  return d * static_cast<float>(scale) * static_cast<float>(nibble) -
         dmin * static_cast<float>(minimum);
}

/* One weight of a Q6_K super-block: `d * scale * (q - 32)`.
 *
 * The 256 weights are two halves of 128, and each half reads a different 64-byte
 * span of `ql` and a different 32-byte span of `qh`. Within a half the weights
 * fall into four 32-weight runs -- `sub` below. The four runs share the two
 * 32-byte nibble runs of `ql` in an interleaved way: runs 0 and 2 read the first
 * nibble run and 1 and 3 read the second, while runs 0 and 1 take the low nibble
 * of what they read and runs 2 and 3 the high one. The high bits come from
 * `qh`, two per weight, at bit position `2 * sub` -- so each `qh` byte is shared
 * by four weights and each `ql` byte by two.
 *
 * The scale index is `i / 16 + 2 * sub`: the sixteen scales of a half are
 * indexed by run as well as by position, which is the piece that a decoder
 * indexing them by position alone gets right for one run in four.
 *
 * The scales are signed bytes and the value is offset, hence `q - 32`: a weight
 * may be negative without a separate sign plane. */
POCKETLLM_HD float dequant_q6_k(const uint8_t *block, int index) {
  const int half = index / 128;      /* which 128-weight half */
  const int within = index % 128;    /* position inside that half */
  const int sub = within / 32;       /* which of the four 32-weight runs */
  const int i = within % 32;         /* position inside that run */

  const int ql = as_byte(block, half * 64 + (sub % 2) * 32 + i);
  const int qh = as_byte(block, 128 + half * 32 + i);
  const int high = ((qh >> (2 * sub)) & 3) << 4;
  const int low = sub < 2 ? (ql & 0x0F) : (ql >> 4);
  const int q = low | high;

  const int scale = as_int8(block, 192 + half * 8 + i / 16 + 2 * sub);
  const float d = as_half(block, 208);
  return d * static_cast<float>(scale) * static_cast<float>(q - 32);
}

/* Dispatch for a block a caller only knows by its type id. The ids are GGML's,
 * the same numbers `abi/spec.h` carries; the two the packed GEMM reads are
 * spelled out here rather than shared with `spec.cpp` because that table
 * answers "how many bytes" and this answers "which bit layout", and only the
 * latter is allowed to grow a decoder at a time. */
constexpr int kGgmlQ4K = 12;
constexpr int kGgmlQ6K = 14;

/* How many bytes one block of `type_id` occupies, or 0 if this build cannot
 * read it. */
POCKETLLM_HD int block_bytes_of(int type_id) {
  if (type_id == kGgmlQ4K) {
    return kQ4KBlockBytes;
  }
  if (type_id == kGgmlQ6K) {
    return kQ6KBlockBytes;
  }
  return 0;
}

/* One weight out of a block of `type_id`. Calling this with a type
 * `block_bytes_of` refused is a programming error; the callers check first. */
POCKETLLM_HD float dequant_block(int type_id, const uint8_t *block, int index) {
  return type_id == kGgmlQ4K ? dequant_q4_k(block, index) : dequant_q6_k(block, index);
}

}  // namespace quant
}  // namespace pocketllm

#endif /* POCKETLLM_QUANT_BLOCKS_H */