/* The repacked q4_K panel GEMM -- on by default, `$POCKETLLM_CPU_REPACK=0` for the
 * row kernel.
 *
 * `dot_Rrows_q8k<8>` walks **one** weight column against eight activation rows
 * and decodes each weight block once per walk.  A prefill GEMM re-walks the whole
 * weight matrix `m / R` times, so every weight byte is decoded `m / R` times.
 * The panel path's answer is llama.cpp's: pre-tile the weights into eight-column
 * panels (`Q4Kx8`) and the activations into four-row groups (`Q8Kx4`), so one
 * kernel pass produces an 8x8 tile of the output with every operand block
 * decoded once.
 *
 * The arithmetic is llama.cpp's, ported deliberately rather than re-derived: the
 * four `maddubs` products of a 32-weight sub-block accumulate in int16 and take
 * **one** `madd_epi16` scale multiply, against the four `madd_epi16` our own
 * layout needs because our four `j` iterations carry different scales.  Measured
 * 1.57x serially and 1.19-1.35x end-to-end on the real prefill shapes
 * (`docs/architecture/c_engine.md`).
 *
 * **This path is NOT bit-identical to the row kernel.**  The int16 accumulation
 * is a different association from the row kernel's four int32 accumulators, and
 * it is legal only because q4_K's 4-bit values (0-15) against int8 activations
 * (+/-127) give at most 1905 per `maddubs` lane, so four sum to 7620 < 32767.
 * q6_K's 6-bit values would overflow, which is why this path is q4_K-only and
 * everything else stays on the row kernel.
 *
 * The mutual-consistency invariant is untouched because this is a **selected
 * whole-GEMM path**, never mixed with `gemm_quant`: when it is on it computes the
 * entire sub-batch and the row kernel is not called at all, so
 * `test_the_four_row_weight_walk_is_the_one_row_walk_four_times` still describes
 * `gemm_quant`'s own schedule exactly.
 *
 * **On by default, and that is the stage's whole point**: the panel path is what
 * takes a `q4_k_m` prefill from 0.86x of llama.cpp to parity at the default 22
 * threads, and a default that leaves the engine 14% behind is not a default.
 * `POCKETLLM_CPU_REPACK=0` selects the row kernel, which is *not* the opt-in
 * convention the other selectors in this tree use -- those add a behaviour that
 * was not there, this one withholds one, and inverting the sense is what makes
 * the shipped binary the fast one. */
#ifndef POCKETLLM_KERNEL_REPACK_Q4K_H
#define POCKETLLM_KERNEL_REPACK_Q4K_H

#include <cstddef>
#include <cstdint>

#include "quant/q8k.h"

namespace pocketllm {
namespace kernel {

/* One 256-weight super-block of eight weight columns, interleaved.  Byte-for-byte
 * llama.cpp's `block_q4_Kx8` (ggml/src/ggml-cpu/repack.h) -- and, because
 * `1152 == 8 * 144`, **the same byte count** as the eight `q4_K` blocks it is
 * built from.  A repacked weight matrix is therefore the same size as the
 * original, which is what lets the loader repack in place. */
struct Q4Kx8 {
  uint16_t d[8];
  uint16_t dmin[8];
  uint8_t scales[96];
  uint8_t qs[1024];
};
static_assert(sizeof(Q4Kx8) == 1152, "block_q4_Kx8 size");

/* Four activation rows' 256-weight super-block, interleaved.  llama.cpp's
 * `block_q8_Kx4`. */
struct Q8Kx4 {
  float d[4];
  int8_t qs[1024];
  int16_t bsums[64];
};
static_assert(sizeof(Q8Kx4) == 1168, "block_q8_Kx4 size");

/* Is the repacked path selected?  Reads `$POCKETLLM_CPU_REPACK` once, the way the
 * other CPU selectors are read -- but with the opposite sense: unset selects the
 * panel path, `=0` withholds it.  See the file comment for why. */
bool repack_enabled();

/* Can this build run the panel kernel at all?  True only when the translation
 * unit was compiled with AVX2; the loader asks this rather than
 * `repack_enabled()` alone so a non-AVX2 build falls back to the row kernel
 * instead of calling the empty stub `gemm_q4k_8x8` below. */
bool repack_available();

/* Repack a whole q4_K weight matrix: `n` columns of `k` weights, row-major
 * `q4_K` blocks (`k / 256` per column), into `n / 8 * (k / 256)` panels.
 * `panels` must hold `n * k / 8` bytes, the same byte count as `blocks`. */
void repack_weights_q4k(const uint8_t *blocks, int64_t n, int64_t k, Q4Kx8 *panels);

/* Repack a whole q6_K weight matrix: `n` columns of `k` weights, row-major
 * `q6_K` blocks (`k / 256` per column), into the `Q6KRepacked` layout
 * (`quant/blocks.h`). `out` must hold `n * (k / 256) * 274` bytes. Unlike the
 * q4_K panels this **enlarges** the matrix (210 -> 274 bytes a block, +30%),
 * which is the cost of trading the per-weight bit assembly for a byte load.
 * `n` and `k` need no divisibility beyond the `k % 256` every q6_K tensor
 * already has. */
void repack_weights_q6k(const uint8_t *blocks, int64_t n, int64_t k, uint8_t *out);

/* Quantize `nr` activation rows of `k` floats (row stride `k`) into groups of
 * four rows, laid out `packed[g * (k/256) + b]` -- group-major, the order
 * `gemm_q4k_8x8` walks.  `nr` must be a multiple of 4. */
void repack_activations(const float *x, int64_t nr, int64_t k, Q8Kx4 *packed);

/* The GEMM: `nr` rows x `nc` columns, `k` deep.  `s[(row)*bs + col]` with `bs` a
 * FLOAT stride.  `nr` must be a multiple of 4, `nc` a multiple of 8.  `panels`
 * and `acts` are the group-major arrays the two packers above produce.
 *
 * Parallel across output-column panels, which is the split that wins: splitting
 * output **rows** instead makes every thread stream the whole weight matrix and
 * measures ~1.7x *slower* than the row kernel at 22 threads. */
void gemm_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels, const Q8Kx4 *acts, int nr,
                  int nc);

/* One activation row against the same panels -- the decode half of the layout,
 * and the reason a repacked weight is readable at every `m` rather than only at
 * `m % 8 == 0`.  `acts` is the row kernel's own `quant::Q8KBlock` array
 * (`nr * n / 256` blocks), so the activation packing is shared with
 * `dot_row_q8k`.  `s[y * bs + x * 8 .. + 7]`, one output axis.
 *
 * Not bit-identical to `dot_row_q8k`; see the file comment. */
void gemv_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels, const quant::Q8KBlock *acts,
                  int nr, int nc);

}  // namespace kernel
}  // namespace pocketllm

#endif  // POCKETLLM_KERNEL_REPACK_Q4K_H