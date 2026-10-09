/* The repacked q4_K panel GEMM.  See repack_q4k.h for why it exists, what opts
 * in, and why it is not bit-identical to the row kernel.
 *
 * The kernel body is a deliberate port of llama.cpp's `ggml_gemm_q4_K_8x8_q8_K`
 * AVX2 path (ggml/src/ggml-cpu/arch/x86/repack.cpp).  Its shuffle offsets are
 * transcribed, not re-derived: reading them off the source and hand-writing the
 * masks was wrong every time it was tried, and a transcription diffed against
 * the original is something a reviewer can check.  The port is verified
 * bit-exact against llama.cpp's own compiled kernel on identical panels and
 * activations, and the packers against llama.cpp's own repacker and
 * `ggml_quantize_mat_q8_K_4x4`.
 *
 * Only the AVX2 path is here.  `-march=native` is already this tree's build
 * choice, so AVX2 is the path that runs on the project's own x86 machine; a
 * build without it falls back to the row kernel rather than to a scalar copy.
 */
#include "kernel/repack_q4k.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "kernel/parallel.h"
#include "quant/blocks.h"
#include "quant/q8k.h"

#if defined(__AVX2__)
#include <immintrin.h>

/* llama.cpp's own name for the fp16-column load its kernel uses. */
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))

namespace pocketllm {
namespace kernel {
namespace {

using block_q4_Kx8 = Q4Kx8;
using block_q8_Kx4 = Q8Kx4;
constexpr int QK_K = 256;

void gemm_q4k_8x8_avx2(int n, float *s, size_t bs, const void *vx, const void *vy, int nr,
                       int nc) {

    const int qk = QK_K;
    const int nb = n / qk;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    // ---- setup, lines 2066-2076 ----
    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    /* llama.cpp aligns this to 16 because its 16-row `y += 4` pass reads four
     * `Q8Kx4` groups (16 rows) per iteration and its narrower `y++` tail then
     * finishes the remainder -- so a `nr % 16 != 0` batch never over-reads the
     * activation array.  The port kept both AVX2 passes, so the split stands. */
    int anr = nr - nr % 16;

    // ---- AVX2 body, lines 2816-3486 ----

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                        __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                        __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                        __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                        __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                        __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                        __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                        const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                        const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                        const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                        const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                        __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                        __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                        __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                        __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                        __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                        __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                        __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                        __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);//GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                        __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                        __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                        __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    /* ---- AVX2 body, lines 3488-3490 ---- */

/* ---- AVX2 body, lines 3511-3526 ---- */
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];
        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }
        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_Kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_Kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when sb = 1
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                    const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                    const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                    const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                    const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    /* ---- AVX2 body, lines 3525-3526 ---- */
    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_Kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_Kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when sb = 1
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                    const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                    const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                    const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                    const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
}

}  // namespace
/* The q6_K repack is a plain byte transform with no AVX2.  It walks every
 * column and super-block of the tensor, decodes each weight to the `q - 32`
 * byte the kernel will read, and copies the sixteen scales and the half `d`
 * through unchanged.  The decode is the file decoder's own
 * `quant::dequant_q6_k`, read back as the integer it started from: rather than
 * re-derive the bit layout here (a second transcription, and the one the file
 * header warns against), it recovers `q` the same way the kernel does and
 * stores `q - 32`, which is exactly the range an `int8` holds.  The float `d`
 * and `scale` are *not* folded in -- they are carried to the kernel so the
 * per-weight product keeps the association `dequant_q6_k` used. */
void repack_weights_q6k(const uint8_t *blocks, int64_t n, int64_t k, uint8_t *out) {
  const int64_t nb = k / 256;
  const size_t row_bytes = static_cast<size_t>(nb * quant::kQ6KBlockBytes);
  const size_t out_row_bytes = static_cast<size_t>(nb) * quant::kQ6KRepackedBytes;
  parallel_for(n, 1, [&](int64_t r_lo, int64_t r_hi, int64_t) {
    for (int64_t r = r_lo; r < r_hi; ++r) {
      for (int64_t b = 0; b < nb; ++b) {
        const uint8_t *in = blocks + r * row_bytes + b * quant::kQ6KBlockBytes;
        uint8_t *o = out + r * out_row_bytes + b * quant::kQ6KRepackedBytes;
        int8_t *qs = reinterpret_cast<int8_t *>(o);
        for (int i = 0; i < 256; ++i) {
          qs[i] = static_cast<int8_t>(quant::q6_k_raw_value(in, i) - 32);
        }
        for (int s = 0; s < 16; ++s) {
          o[256 + s] = in[192 + s];
        }
        o[272] = in[208];
        o[273] = in[209];
      }
    }
  });
}

}  // namespace kernel
}  // namespace pocketllm

namespace pocketllm {
namespace kernel {
namespace {

/* The panel GEMV: one activation row against the eight-column panels, which is
 * the *decode* half of the same layout.  It exists because the panel path is
 * selected for the whole *session*, not per shape: a repacked weight has to be
 * readable by whatever `m` the graph hands it, and the row kernel's
 * `dot_row_q8k` reads the file's per-row blocks, not panels.  Without this, a
 * repacked checkpoint would decode from misread bytes -- and it would not fail,
 * it would answer fluently.
 *
 * The arithmetic is llama.cpp's `ggml_gemv_q4_K_8x8_q8_K` AVX2 path: the same
 * int16 accumulation as the panel GEMM, one activation row at a time, with the
 * result permuted back out of the interleaved column order.  It is therefore
 * *not* bit-identical to `dot_row_q8k`, for the same reason the GEMM is not.
 *
 * The activation operand is one `quant::Q8KBlock` per 256 weights -- exactly the
 * layout `gemm_quant` already builds for its rows -- so nothing extra is
 * quantized here.  `s[y * bs + x * 8 .. + 7]`, so `bs` is the output axis. */
void gemv_q4k_8x8_cols(int n, float *s, size_t bs, const Q4Kx8 *panels,
                       const quant::Q8KBlock *acts, int nr, int64_t x_lo, int64_t x_hi) {
  const int qk = QK_K;
  const int nb = n / qk;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  const Q4Kx8 *b_ptr_start = panels;
  const quant::Q8KBlock *a_ptr_start = acts;
// Lookup table to convert signed nibbles to signed bytes
__m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
// Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
__m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
__m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
// Permute mask used for easier vector processing at later stages
__m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

// Mask to extract nibbles from bytes
const __m256i m4b = _mm256_set1_epi8(0x0F);

int64_t b_nb = n / QK_K;


// Process Q8_K blocks one by one
for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of quant::Q8KBlock format
    const quant::Q8KBlock * a_ptr = a_ptr_start + (y * nb);

    // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
    for (int64_t x = x_lo; x < x_hi; x++) {

        // Pointers to RHS blocks
        const Q4Kx8 * b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_row = _mm256_setzero_ps();
        __m256 acc_min_rows = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb; b++) {

            // Load and convert to FP32 scale from quant::Q8KBlock
            const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

            // Load the scale values for the 8 blocks interleaved in Q4Kx8
            // col_scale_f32 rearranged so as to multiply with appropriate quants
            const __m256 col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(b_ptr[b].d)), deltamask));
            const __m256 col_dmin_f32 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b_ptr[b].dmin)));

            __m256i iacc_b = _mm256_setzero_si256();
            __m256i iacc_min_b = _mm256_setzero_si256();

            const __m256i q8sums = _mm256_loadu_si256((const __m256i * )(a_ptr[b].bsums));
            __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
            q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

            // Processes two sub blocks from each Q4_K in each iteration
            for (int sb = 0; sb < QK_K / 64; sb++) {

                // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                // 4-bit -> 8-bit
                // Values of the first sub block of eight block_q4_K structures for the sb loop
                const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                // Values of the second sub block of eight block_q4_K structures when sb = 1
                const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                uint32_t utmp_0[4], utmp_1[4];

                // Scales and Mins of corresponding sub blocks from different Q8_K structures are stored together
                // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp_0[1] & kmask1;
                utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                utmp_0[2] = uaux_0;
                utmp_0[0] &= kmask1;

                // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                const uint32_t uaux_1 = utmp_1[1] & kmask1;
                utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                utmp_1[2] = uaux_1;
                utmp_1[0] &= kmask1;

                // Scales of first sub block in the sb loop
                const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                // Scales of second sub block in the sb loop
                __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                // Mins of first and second sub block of Q4_K block are arranged side by side
                __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                // Load the two sub block values corresponding to sb in quant::Q8KBlock in batches of 16 bytes and replicate the same across 256 bit vector
                __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                // Dot product done within 32 bit lanes and accumulated in the same vector
                // First done for first sub block and then for second sub block in each sb
                // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                // ...........................................................................
                // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)


                __m256i iacc_0 = _mm256_setzero_si256();
                __m256i iacc_1 = _mm256_setzero_si256();

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                // Accumulate the iacc value for one sb
                __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                q8s = _mm256_bsrli_epi128(q8s, 4);

                // Accumulate for the complete block
                iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
            }

            // Multiply-Add with scale values for the complete super block
            acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

        }

        // Accumulated output values permuted so as to be stored in appropriate order post accumulation
        acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
        _mm256_storeu_ps(s + (y * bs + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
    }
}
}
}  // namespace

/* Split across output-column panels, the same split the GEMM uses and for the
 * same measured reason: splitting output **rows** makes every thread stream the
 * whole weight matrix.  Without this the decode path is single-threaded and
 * measured 3.4x *slower* than the row kernel it replaces. */
void gemv_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels,
                  const quant::Q8KBlock *acts, int nr, int nc) {
  parallel_for(nc / 8, 1, [&](int64_t lo, int64_t hi, int64_t) {
    gemv_q4k_8x8_cols(n, s, bs, panels, acts, nr, lo, hi);
  });
}

}  // namespace kernel
}  // namespace pocketllm

#endif  // __AVX2__

namespace pocketllm {
namespace kernel {

/* **The sense is inverted from every other selector in this tree, on purpose.**
 * `$POCKETLLM_CPU_KV_F32`, `$POCKETLLM_CPU_SCALAR_DOT` and the rest are opt-in:
 * unset means the shipped default and setting them adds a behaviour.  Here the
 * shipped default *is* the panel path, because it is what brings a `q4_k_m`
 * prefill to llama.cpp parity, and unset has to select it for a caller who never
 * read this file.  `=0` is the escape hatch back to the row kernel. */
bool repack_enabled() {
  static const bool on = [] {
    const char *v = std::getenv("POCKETLLM_CPU_REPACK");
    return !(v != nullptr && v[0] == '0');
  }();
  return on;
}

/* The panel path needs AVX2; the row kernel is always there.  The loader asks
 * this rather than `repack_enabled()` alone, so a build without AVX2 falls back
 * instead of calling the empty stub below. */
bool repack_available() {
#if defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

namespace {

/* The low six bits of `lo` and the top two of `hi`, the packing llama.cpp's
 * `make_block_q4_Kx8` uses for a q4_K scale/min pair. */
uint8_t six_bits(uint8_t lo, uint8_t hi) { return static_cast<uint8_t>((lo & 63) | ((hi & 48) << 2)); }

}  // namespace

/* `n` weight columns of `k` weights, row-major q4_K blocks (`k / 256` per
 * column), into `n / 8 * (k / 256)` eight-column panels.  llama.cpp's
 * `make_block_q4_Kx8`, with the panel and block indices explicit because our
 * caller owns the whole matrix. */
void repack_weights_q4k(const uint8_t *blocks, int64_t n, int64_t k, Q4Kx8 *panels) {
  const int64_t nb = k / 256;
  const size_t row_bytes = static_cast<size_t>(nb * quant::kQ4KBlockBytes);
  parallel_for(n / 8, 1, [&](int64_t p_lo, int64_t p_hi, int64_t) {
    for (int64_t p = p_lo; p < p_hi; ++p) {
      for (int64_t b = 0; b < nb; ++b) {
        Q4Kx8 *o = panels + (p * nb + b);
        const uint8_t *in[8];
        for (int r = 0; r < 8; ++r) {
          in[r] = blocks + (p * 8 + r) * row_bytes + b * quant::kQ4KBlockBytes;
        }
        for (int r = 0; r < 8; ++r) {
          std::memcpy(&o->d[r], in[r], 2);
          std::memcpy(&o->dmin[r], in[r] + 2, 2);
        }
        for (int i = 0; i < 128; ++i) {
          std::memcpy(&o->qs[i * 8], in[i % 8] + 16 + (i / 8) * 8, 8);
        }
        for (int i = 0; i < 4; ++i) {
          uint8_t sv[8];
          uint8_t mv[8];
          for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < 8; ++j) {
              const uint8_t *sc = in[j] + 4;
              if (pass == 0) {
                sv[j] = sc[i] & 63;
                mv[j] = sc[i + 4] & 63;
              } else {
                sv[j] = static_cast<uint8_t>(((sc[i] & 192) >> 2) | (sc[i + 8] & 15));
                mv[j] = static_cast<uint8_t>(((sc[i + 4] & 192) >> 2) | ((sc[i + 8] & 240) >> 4));
              }
            }
            uint8_t *q = o->scales + pass * 48 + i * 12;
            q[0] = six_bits(sv[0], sv[4]);
            q[1] = six_bits(sv[1], sv[5]);
            q[2] = six_bits(sv[2], sv[6]);
            q[3] = six_bits(sv[3], sv[7]);
            q[4] = six_bits(mv[0], mv[4]);
            q[5] = six_bits(mv[1], mv[5]);
            q[6] = six_bits(mv[2], mv[6]);
            q[7] = six_bits(mv[3], mv[7]);
            q[8] = static_cast<uint8_t>((sv[4] & 15) | ((mv[4] & 15) << 4));
            q[9] = static_cast<uint8_t>((sv[5] & 15) | ((mv[5] & 15) << 4));
            q[10] = static_cast<uint8_t>((sv[6] & 15) | ((mv[6] & 15) << 4));
            q[11] = static_cast<uint8_t>((sv[7] & 15) | ((mv[7] & 15) << 4));
          }
        }
      }
    }
  });
}

/* `nr` activation rows of `k` floats (row stride `k`) into `nr / 4` groups of
 * four rows, laid out group-major -- `packed[g * (k/256) + b]` -- so a thread
 * owning a group writes one contiguous run.
 *
 * **The interleave is llama.cpp's 4x8 one, not its 4x4 one.**  `block_q8_Kx4` is
 * written by both `ggml_quantize_mat_q8_K_4x4` and `..._4x8` in llama.cpp, and
 * the q4_Kx8 GEMM reads the **8**: its `i` spans eight packed bytes, which is
 * exactly
 *
 *   `qs[32 * (w / 8) + 8 * m + (w % 8)]` holds activation `w` of row `m`,
 *
 * with bsums at `(((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3)`.
 * Packing with the 4x4 rule instead produces a *self-consistent* wrong answer --
 * the values are a permutation of the right ones, so a multiset check passes and
 * the row-group scaling is unaffected, and only the pairing against the weights
 * is wrong.  That is why this was caught by a scalar reference off the panel and
 * not by inspection. */
void repack_activations(const float *x, int64_t nr, int64_t k, Q8Kx4 *packed) {
  const int64_t nb = k / 256;
  parallel_for(nr / 4, 1, [&](int64_t g_lo, int64_t g_hi, int64_t) {
    for (int64_t g = g_lo; g < g_hi; ++g) {
      for (int64_t b = 0; b < nb; ++b) {
        Q8Kx4 *o = packed + (g * nb + b);
        for (int i = 0; i < 64; ++i) {
          o->bsums[i] = 0;
        }
        for (int m = 0; m < 4; ++m) {
          const float *row = x + (g * 4 + m) * k + b * 256;
          /* Quantized by the *row kernel's own quantizer*, not by a second
           * transcription of it.  The two engines have to quantize an activation
           * the same way -- that is what makes the token-for-token match hold --
           * and the decision the quantizer makes (which element sets the scale,
           * ties-to-even via `nearest_int`, the defensive clamp at 127) is
           * exactly the kind of thing a copy drifts from.  Calling it makes the
           * agreement structural rather than a thing to re-check. */
          quant::Q8KBlock q8;
          quant::quantize_q8_block(row, q8);
          o->d[m] = q8.d;
          for (int w = 0; w < 256; ++w) {
            const int q = q8.qs[w];
            const int j = 32 * (w / 8) + 8 * m + (w % 8);
            o->qs[j] = static_cast<int8_t>(q);
            /* `index` is constant across a 16-weight group -- it reduces to
             * `4m + 16 * (w / 64) + ((w / 16) & 3)` -- which is why the
             * accumulator below lands one `bsums` entry per (row, group) and why
             * it equals `q8.bsums[w / 16]`. */
            const int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            o->bsums[index] = static_cast<int16_t>(o->bsums[index] + static_cast<int16_t>(q));
          }
        }
      }
    }
  });
}

#if defined(__AVX2__)
/* Split across output-column panels, which is the split that wins: each panel
 * walks the whole activation matrix once and one weight panel, so all `n / 8`
 * threads share the activation reads in cache.  Splitting output **rows**
 * instead gives every thread the whole weight matrix and measures ~1.7x
 * *slower* than the row kernel at 22 threads -- that is the trap this shape
 * exists to avoid. */
void gemm_q4k_8x8(int n, float *s, size_t bs, const Q4Kx8 *panels, const Q8Kx4 *acts, int nr,
                  int nc) {
  const int64_t nb = n / 256;
  parallel_for(nc / 8, 1, [&](int64_t lo, int64_t hi, int64_t) {
    for (int64_t p = lo; p < hi; ++p) {
      gemm_q4k_8x8_avx2(n, s + p * 8, bs, panels + p * nb, acts, nr, 8);
    }
  });
}
#else
void gemm_q4k_8x8(int, float *, size_t, const Q4Kx8 *, const Q8Kx4 *, int, int) {}
#endif

#if !defined(__AVX2__)
/* The decode half of the panel layout, empty for the same reason `gemm_q4k_8x8`
 * is: there is no scalar port of it, and `repack_available()` is false here, so
 * the caller takes the row kernel instead.  It still has to *exist*, because
 * `qwen3.cpp` calls it unconditionally and the symbol is resolved at link time
 * whether or not the path is taken -- without this, every non-AVX2 build of
 * `libpocketllm.so` fails to link. */
void gemv_q4k_8x8(int, float *, size_t, const Q4Kx8 *, const quant::Q8KBlock *, int, int) {}
#endif

#if !defined(__AVX2__)
/* Same shape of gap as `gemv_q4k_8x8` above, and for the q6_K repack this time:
 * the transform lives inside the `__AVX2__` block and has no scalar port, but
 * `qwen3.cpp` calls it whenever a CUDA backend asked for the repacked layout --
 * a runtime guard the linker cannot see, so the symbol has to exist for every
 * build.  `q6k_repack_enabled_` is set only on `cuda`, so no ascend or plain CPU
 * build ever reaches this; it is here so the link succeeds, and it does nothing
 * if called. */
void repack_weights_q6k(const uint8_t *, int64_t, int64_t, uint8_t *) {}
#endif

}  // namespace kernel
}  // namespace pocketllm
