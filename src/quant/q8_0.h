/* The `q8_0` weight block: 32 signed bytes and one fp16 scale.
 *
 * This is the *storage* format a checkpoint's tensor arrives in -- not to be
 * confused with `q8k.h`, which is the int8 *activation* block the quantized GEMM
 * works in.  They share a name and nothing else: this one is 34 bytes per 32
 * weights with no packing at all, and it is the format the quantizer chooses for a
 * tensor that is small enough, or important enough, to keep at eight bits.
 *
 * Layout, which is llama.cpp's `ggml-common.h`:

 *     struct block_q8_0 {
 *         ggml_half d;        // the block's scale, 2 bytes
 *         int8_t    qs[32];   // 32 signed weights
 *     };                      // 34 bytes, 32 weights
 *
 * A weight is `d * qs[i]`, so decoding is a multiply and no bit unpacking -- which is
 * why this file is a header and a loop rather than a kernel.
 *
 * **The first caller is the audio tower**, whose mmproj stores its attention and
 * FFN weights this way while the vision tower's file is dense f16.  The tower's
 * loader refused anything but f32 and f16 by name, which was right for the file it
 * was written against and wrong for this one.
 *
 * The decode is exact in the sense that matters here: `d` is an fp16 widened to f32
 * and the product is one f32 multiply, which is what llama.cpp's own
 * `dequantize_row_q8_0` computes.  A GEMM that instead kept the weights in eight
 * bits and quantized the activation would round differently -- that is the other
 * path (`q8k.h`) and a separate change.
 */

#ifndef POCKETLLM_QUANT_Q8_0_H
#define POCKETLLM_QUANT_Q8_0_H

#include <cstdint>
#include <cstring>

#include "quant/half.h"
#include "runtime/status.h"

namespace pocketllm {
namespace quant {

/* Weights per block. */
constexpr int64_t kQ8_0BlockElems = 32;
/* Bytes per block: two for the fp16 scale and thirty-two for the values. */
constexpr int64_t kQ8_0BlockBytes = 34;

/* One block's scale, widened to f32. */
inline float q8_0_scale(const uint8_t *block) {
  return half_to_float(load_u16(block));
}

/* The block's 32 weights, widened.  `out` holds 32 floats. */
inline void q8_0_dequantize_block(const uint8_t *block, float *out) {
  const float scale = q8_0_scale(block);
  const int8_t *values = reinterpret_cast<const int8_t *>(block + 2);
  for (int64_t i = 0; i < kQ8_0BlockElems; ++i) {
    out[i] = static_cast<float>(values[i]) * scale;
  }
}

/* A whole tensor of packed `q8_0` bytes, widened to `out`.
 *
 * `elements` must be a multiple of 32, which every tensor this reads is: the format
 * has no way to store a partial block, and a file claiming one is malformed rather
 * than a case to pad. */
inline void q8_0_dequantize(const uint8_t *packed, int64_t elements, float *out) {
  if (elements % kQ8_0BlockElems != 0) {
    throw Error("a q8_0 tensor holds a whole number of 32-weight blocks, and " +
                std::to_string(elements) + " weights were declared");
  }
  const int64_t blocks = elements / kQ8_0BlockElems;
  for (int64_t block = 0; block < blocks; ++block) {
    q8_0_dequantize_block(packed + block * kQ8_0BlockBytes, out + block * kQ8_0BlockElems);
  }
}

/* The bytes a tensor of `elements` weights occupies. */
inline int64_t q8_0_packed_bytes(int64_t elements) {
  return (elements / kQ8_0BlockElems) * kQ8_0BlockBytes;
}

}  // namespace quant
}  // namespace pocketllm

#endif /* POCKETLLM_QUANT_Q8_0_H */