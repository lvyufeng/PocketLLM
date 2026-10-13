/* The repacked q4_K panel layout -- on by default, `$POCKETLLM_CPU_REPACK=0`
 * for the row kernel.
 *
 * One activation row at a time against eight-column weight panels (`Q4Kx8`):
 * `gemv_q4k_8x8` below is the kernel every panel weight is read by, at every
 * call shape.  The layout is llama.cpp's, ported deliberately rather than
 * re-derived; the dense 8x8-tile GEMM (`gemm_q4k_8x8`, also below) is the
 * layout's other kernel and is **retained but no longer dispatched**.
 *
 * **One kernel at every `m` is a correctness constraint, not a simplification.**
 * `Qwen3Model::matmul` used to select the tile GEMM when the call's total row
 * count was a multiple of eight and the GEMV otherwise, which made a row's
 * arithmetic a function of the batch it rode in: a 16-token prompt and the same
 * prompt split 16 + one decode step produced different logits, so a stream
 * batched with others was not byte-equal to its solo run -- the serving
 * contract's `equals_solo`, silent wrong text rather than a perf note.  The two
 * kernels are not bit-identical (below), so no dispatch keyed on `m` can keep
 * that contract; the GEMV is the one canonical arithmetic, and the GEMM is kept
 * as the layout's own tile arithmetic -- the measured prefill win, and the
 * reference any future dispatch would have to reproduce bit-for-bit.
 *
 * `dot_Rrows_q8k<8>` walks **one** weight column against eight activation rows
 * and decodes each weight block once per walk, so a prefill GEMM re-walks the
 * whole weight matrix `m / R` times.  The tile GEMM's answer is llama.cpp's:
 * eight-column weight panels (`Q4Kx8`) and four-row activation groups (`Q8Kx4`),
 * one kernel pass producing an 8x8 tile with every operand block decoded once.
 * The arithmetic is llama.cpp's, ported deliberately rather than re-derived: the
 * four `maddubs` products of a 32-weight sub-block accumulate in int16 and take
 * **one** `madd_epi16` scale multiply, against the four `madd_epi16` our own
 * layout needs because our four `j` iterations carry different scales.  Measured
 * 1.57x serially and 1.19-1.35x end-to-end on the real prefill shapes with the
 * tile GEMM still dispatched (`docs/architecture/c_engine.md`).
 *
 * **Neither panel kernel is bit-identical to the row kernel.**  The GEMM's int16
 * accumulation is a different association from the row kernel's four int32
 * accumulators, and it is legal only because q4_K's 4-bit values (0-15) against
 * int8 activations (+/-127) give at most 1905 per `maddubs` lane, so four sum to
 * 7620 < 32767.  q6_K's 6-bit values would overflow, which is why this layout is
 * q4_K-only and everything else stays on the row kernel.  The row kernel is a
 * third arithmetic again; `POCKETLLM_CPU_REPACK=0` selects it as the A/B, and
 * the invariance tests run under both arms.
 *
 * The mutual-consistency invariant is untouched because the panel layout is
 * never mixed with `gemm_quant` inside a tensor: a weight whose bytes are panels
 * is read only by the panel kernel, so
 * `test_the_four_row_weight_walk_is_the_one_row_walk_four_times` still describes
 * `gemm_quant`'s own schedule exactly.
 *
 * **On by default, and the sense inversion stays**: what the panel layout ships
 * is the decode kernel -- the GEMV measured faster than the row kernel's decode
 * in the A/B `docs/architecture/c_engine.md` records -- and a caller who never
 * read this file should get the fast one.  `POCKETLLM_CPU_REPACK=0` selects the
 * row kernel, which is *not* the opt-in convention the other selectors in this
 * tree use -- those add a behaviour that was not there, this one withholds one,
 * and inverting the sense is what makes the shipped binary the fast one. */
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

/* Is the repacked panel layout selected?  Reads `$POCKETLLM_CPU_REPACK` once, the
 * way the other CPU selectors are read -- but with the opposite sense: unset
 * selects the panel layout, `=0` withholds it.  See the file comment for why. */
bool repack_enabled();

/* Can this build run the panel kernels at all?  True only when the translation
 * unit was compiled with AVX2; the loader asks this rather than
 * `repack_enabled()` alone so a non-AVX2 build falls back to the row kernel
 * instead of calling the empty stubs the panel kernels compile to below. */
bool repack_available();

/* Repack a whole q4_K weight matrix: `n` columns of `k` weights, row-major
 * `q4_K` blocks (`k / 256` per column), into `n / 8 * (k / 256)` panels.
 * `panels` must hold `n * k / 8` bytes, the same byte count as `blocks`. */
void repack_weights_q4k(const uint8_t *blocks, int64_t n, int64_t k, Q4Kx8 *panels);

/* Repack a whole q6_K weight matrix: `n` columns of `k` weights, row-major
 * `q6_K` blocks (`k / 256` per column), into the `Q6KRepacked` layout
 * (`quant/blocks.h`). `out` must hold `n * (k / 256) * 288` bytes -- the block's
 * 274 bytes of payload on a 16-byte stride, so the kernel's `uint4` reads are
 * aligned. Unlike the q4_K panels this **enlarges** the matrix (210 -> 288 bytes
 * a block, +37%), which is the cost of trading the per-weight bit assembly for
 * aligned vector byte loads.
 * `n` and `k` need no divisibility beyond the `k % 256` every q6_K tensor
 * already has. */
void repack_weights_q6k(const uint8_t *blocks, int64_t n, int64_t k, uint8_t *out);

/* Quantize `nr` activation rows of `k` floats (row stride `k`) into groups of
 * four rows, laid out `packed[g * (k/256) + b]` -- group-major, the order the
 * tile GEMM `gemm_q4k_8x8` walks.  `nr` must be a multiple of 4.  With the GEMM
 * not dispatched this is the GEMM's packing step alone; the shipped model packs
 * activations one row at a time into `quant::Q8KBlock` for the GEMV. */
void repack_activations(const float *x, int64_t nr, int64_t k, Q8Kx4 *packed);

/* The GEMM: `nr` rows x `nc` columns, `k` deep.  `s[(row)*bs + col]` with `bs` a
 * FLOAT stride.  `nr` must be a multiple of 4, `nc` a multiple of 8.  `panels`
 * and `acts` are the group-major arrays the two packers above produce.
 *
 * Parallel across output-column panels, which is the split that wins: splitting
 * output **rows** instead makes every thread stream the whole weight matrix and
 * measures ~1.7x *slower* than the row kernel at 22 threads.
 *
 * **No call shape dispatches to this kernel.**  It was selected for a call whose
 * total row count was a multiple of eight, which made a row's arithmetic a
 * function of its batch; that selection is removed (see the file comment) and
 * the GEMV is the model's only panel kernel.  It stays in the library as the
 * layout's own tile arithmetic and the measured large-`m` prefill win. */
void gemm_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels, const Q8Kx4 *acts, int nr,
                  int nc);

/* One activation row against the panels -- and, since the `m % 8 == 0` selection
 * was removed, the kernel **every panel weight is read by, at every call
 * shape**.  It was written as the decode half of the layout, so a repacked
 * weight is readable at every `m` rather than only at multiples of eight; it is
 * now the only reader, because the tile GEMM's arithmetic cannot be made to
 * agree with it bit-for-bit and a row's result must not depend on which rows
 * ride with it (`Qwen3Model::matmul`).
 *
 * `acts` is the row kernel's own `quant::Q8KBlock` array (`nr * n / 256`
 * blocks), so the activation packing is shared with `dot_row_q8k`.
 * `s[y * bs + x * 8 .. + 7]`, one output axis.
 *
 * Not bit-identical to `dot_row_q8k`; see the file comment. */
void gemv_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels, const quant::Q8KBlock *acts,
                  int nr, int nc);

}  // namespace kernel
}  // namespace pocketllm

#endif  // POCKETLLM_KERNEL_REPACK_Q4K_H