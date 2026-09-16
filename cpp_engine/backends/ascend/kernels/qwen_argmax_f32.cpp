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
// One block owns up to eight consecutive rows.
//
// The reduction runs in two passes, both of which keep the scalar unit off the
// bulk of the data. The first pass walks every tile of a row and carries out one
// number, the tile's maximum: the vector unit folds the tile with a halving tree
// of `Max`, which leaves one lane per kFoldLanes-wide group, and the scalar unit
// reads only those lanes. The second pass re-reads the single tile that held the
// row maximum and walks it to find the earliest lane equal to it.
//
// Carrying the maximum and the index in one pass is what a paired reduce would
// do, and FP32 paired reductions are unavailable on this SoC (see the note in
// qwen_ascend_kernel_common.hpp). Splitting the work this way costs one extra
// pass over a single tile and removes a scalar walk over every other tile, which
// at a 62,080-wide vocab slice is 62,080 scalar reads per row reduced to about
// 2,100. The original single-pass scalar version measured 1.39 ms per row here,
// against a 248 KiB read that a whole row's MTE2 traffic could not account for:
// it was the scalar walk, not bandwidth, that set the cost.

#include "qwen_ascend_kernel_common.hpp"

namespace {

using namespace pocket;

// Floats per tile. 2048 * 4 = 8 KiB, small enough to double-buffer later without
// restructuring, large enough that the per-tile MTE2 issue cost disappears.
constexpr uint32_t kScanTile = 2048;

// Lanes the vector fold stops at. This is one full FP32 vector instruction
// (256-byte registers), which is also the granularity AscendC splits a longer
// vector op into, so every step of the fold stays within a single instruction's
// worth of lanes per group.
constexpr uint32_t kFoldLanes = 64;

// Standing in for "no maximum seen yet". -FLT_MAX rather than -inf so that the
// fold tree, which is pure `Max`, cannot turn a comparison against it into a NaN
// decision. Nothing in the model reaches this: logits are finite.
constexpr float kNoMaximum = -3.4028234663852886e38f;

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
        const uint32_t tiles = (count + kScanTile - 1) / kScanTile;

        // Pass one: the maximum of every tile, and which tile produced it. The
        // strict `>` keeps the first tile that reaches the maximum, which is the
        // one holding the row's lowest matching index.
        float best = kNoMaximum;
        uint32_t best_tile = 0;
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t start = t * kScanTile;
            const uint32_t width = min_u32(kScanTile, count - start);
            const uint32_t bulk = align_down_u32(width, kAlignFloat);
            if (bulk > 0) {
                AscendC::DataCopy(tile, logits_gm[base + start], bulk);
                wait_load_before_scalar();
            }

            // Fold the bulk down to one lane per kFoldLanes-wide group. `top` is
            // the largest power-of-two multiple of kFoldLanes with 2 * top <= bulk,
            // so every fold reads tile[top, 2 * top), which is loaded, and the tree
            // leaves the maximum of tile[0, 2 * top) spread over tile[0, kFoldLanes).
            // Each fold writes a range strictly below the range it reads, and the
            // steps are 64 lanes wide or wider, so the in-place aliasing is a
            // forward dependency the vector pipe already honours in order.
            uint32_t scanned = 0;
            if (bulk >= 2 * kFoldLanes) {
                uint32_t top = kFoldLanes;
                while (top * 4 <= bulk) top *= 2;
                for (uint32_t stride = top; stride >= kFoldLanes; stride /= 2) {
                    AscendC::Max(tile, tile, tile[stride], stride);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                wait_compute_before_scalar();
                scanned = 2 * top;
                for (uint32_t i = 0; i < kFoldLanes; ++i) {
                    const float value = tile.GetValue(i);
                    if (value > best) {
                        best = value;
                        best_tile = t;
                    }
                }
            }

            // `scanned` is 0 on a tile too short to fold, and 2 * top otherwise,
            // which leaves at most the last half-tile and the sub-8-float remainder
            // of the row to the scalar unit.
            for (uint32_t i = scanned; i < bulk; ++i) {
                const float value = tile.GetValue(i);
                if (value > best) {
                    best = value;
                    best_tile = t;
                }
            }
            for (uint32_t i = bulk; i < width; ++i) {
                const float value = logits_gm.GetValue(base + start + i);
                if (value > best) {
                    best = value;
                    best_tile = t;
                }
            }
            if (bulk > 0) {
                // The next iteration refills the same tile, and nothing but the
                // scalar unit read it, so this is S_MTE2 rather than MTE3_MTE2.
                wait_scalar_before_load();
            }
        }

        // Pass two: the earliest lane of the winning tile equal to the maximum.
        // The winning tile is the first one whose maximum reached it, so no earlier
        // tile can hold the value and the earliest lane here is the row's answer
        // even when the value repeats. `best` is finite whenever any lane is, so a
        // scan that finds nothing means every lane was NaN -- no comparison against
        // NaN is ever true -- and the first lane of the first tile is what a
        // forward walk would have been left holding.
        const uint32_t start = best_tile * kScanTile;
        const uint32_t width = min_u32(kScanTile, count - start);
        const uint32_t bulk = align_down_u32(width, kAlignFloat);
        uint32_t best_index = start;
        bool found = false;
        if (bulk > 0) {
            AscendC::DataCopy(tile, logits_gm[base + start], bulk);
            wait_load_before_scalar();
        }
        for (uint32_t i = 0; i < bulk; ++i) {
            if (tile.GetValue(i) == best) {
                best_index = start + i;
                found = true;
                break;
            }
        }
        for (uint32_t i = bulk; i < width && !found; ++i) {
            if (logits_gm.GetValue(base + start + i) == best) {
                best_index = start + i;
                found = true;
            }
        }
        if (!found && bulk > 0) {
            best = tile.GetValue(0);
        }
        if (bulk > 0) {
            wait_scalar_before_load();
        }

        tokens_gm.SetValue(row, static_cast<int32_t>(best_index) + token_offset);
        values_gm.SetValue(row, best);
    }
    }
}
