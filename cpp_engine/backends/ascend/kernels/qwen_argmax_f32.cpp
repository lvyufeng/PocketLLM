// AscendC row-wise argmax over FP32 logits for first-generation 910.
//
// This is the greedy selection at the end of every forward. aclnnArgMax exists on
// this install, but it returns only the index: the engine needs the winning logit
// too (it reports it, and the TP top-1 merge compares logits across ranks), and it
// needs the global token id rather than the shard-local one. Chaining aclnnArgMax
// with a gather to recover the value costs two more ops and an index tensor, so
// this is written directly.
//
// Tie-break is the lower token id, matching argmax_fp32_rows_cuda. That is not
// cosmetic: under TP the ranks must agree on the same token, and a tie inside one
// shard resolved differently from the CUDA reference would make a cross-backend
// comparison diverge on exactly the inputs that look most benign.
//
// One block owns up to eight consecutive rows. Vector ReduceMax is float16-only
// on this SoC (see the note in qwen_ascend_kernel_common.hpp), so the scan is
// scalar over UB-resident tiles:
// MTE2 brings a tile in, the scalar unit walks it. At a 62,080-wide vocab slice
// that is one pass over 243 KiB per row, bandwidth-bound and comfortably cheaper
// than the lm_head GEMM that produced it.

#include "qwen_ascend_kernel_common.hpp"

namespace {

using namespace pocket;

// Floats per tile. 2048 * 4 = 8 KiB, small enough to double-buffer later without
// restructuring, large enough that the per-tile MTE2 issue cost disappears.
constexpr uint32_t kScanTile = 2048;

// Rows per block. Both results are 4 bytes wide and one per row, so a block that
// owned a single row would write 4 bytes of a 32-byte GM cache line that seven
// other blocks also write. Scalar GM stores are not coherent between cores at
// sub-line granularity: whichever core's line lands last wins and the other seven
// results are silently lost. Giving each block eight consecutive rows makes its two
// output spans exactly one cache line each, so no two blocks share a line and the
// row parallelism survives.
constexpr uint32_t kRowsPerBlock = kBlockBytes / sizeof(float);

}  // namespace

// logits: [rows, count] FP32. tokens: [rows] int32, count-relative index plus
// token_offset. values: [rows] FP32, the winning logit.
extern "C" __global__ __aicore__ void qwen_argmax_f32_rows_kernel(
    GM_ADDR logits, GM_ADDR tokens, GM_ADDR values, uint32_t rows,
    uint32_t count, int32_t token_offset) {
    AscendC::GlobalTensor<float> logits_gm;
    logits_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits));
    AscendC::GlobalTensor<int32_t> tokens_gm;
    tokens_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokens));
    AscendC::GlobalTensor<float> values_gm;
    values_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buf;
    pipe.InitBuffer(tile_buf, kScanTile * sizeof(float));
    AscendC::LocalTensor<float> tile = tile_buf.Get<float>();

    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t block_count = AscendC::GetBlockNum();

    const uint32_t groups = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    for (uint32_t group = block; group < groups; group += block_count) {
        const uint32_t group_end =
            min_u32(rows, (group + 1) * kRowsPerBlock);
        for (uint32_t row = group * kRowsPerBlock; row < group_end; ++row) {
        const uint64_t base = static_cast<uint64_t>(row) * count;
        float best = 0.0f;
        uint32_t best_index = 0;
        bool have_best = false;

        for (uint32_t start = 0; start < count; start += kScanTile) {
            const uint32_t width = min_u32(kScanTile, count - start);
            // A row is not 8-float aligned in general, and the scan reads only
            // `width` lanes, so the aligned bulk plus a scalar tail is exact.
            const uint32_t bulk = align_down_u32(width, kAlignFloat);
            if (bulk > 0) {
                AscendC::DataCopy(tile, logits_gm[base + start], bulk);
                wait_load_before_scalar();
            }
            for (uint32_t i = 0; i < bulk; ++i) {
                const float value = tile.GetValue(i);
                // Strict greater-than keeps the first maximum, which is the
                // lowest index, which is the lowest token id.
                if (!have_best || value > best) {
                    best = value;
                    best_index = start + i;
                    have_best = true;
                }
            }
            for (uint32_t i = bulk; i < width; ++i) {
                const float value = logits_gm.GetValue(base + start + i);
                if (!have_best || value > best) {
                    best = value;
                    best_index = start + i;
                    have_best = true;
                }
            }
            if (bulk > 0) {
                // The next iteration refills the same tile, and nothing but the
                // scalar unit read it, so this is S_MTE2 rather than MTE3_MTE2.
                wait_scalar_before_load();
            }
        }

        tokens_gm.SetValue(row, static_cast<int32_t>(best_index) + token_offset);
        values_gm.SetValue(row, best);
    }
    }
}
