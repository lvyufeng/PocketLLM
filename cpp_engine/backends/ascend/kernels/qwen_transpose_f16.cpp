// Device-side fp16 transpose: dst[cols, rows] = src[rows, cols]^T.
//
// Why this kernel exists: the Cube contracts along the *column* axis of both of
// its ND operands, so O = P * V needs V staged as [d, j] while the KV cache
// stores it as [j, d]. Every cheaper route was ruled out before this one was
// written, and none of them by guessing:
//
//   - UB -> L1 with Nd2NzParams: DataCopyUB2L1ND2NZImpl has no dav_c100
//     definition at all (only c220, c310, m200), so it cannot even link.
//   - The ND2NZ staging path (DataCopyGM2L1ND2NZ): it copies each source row as
//     one contiguous run, so it can only permute 16x16 fractals -- it cannot move
//     an element off its own row. A transpose is not expressible in it.
//   - LoadData2DParams::ifTranspose: measured on this part. It reorders inside
//     each 16x16 fractal but leaves the fractal grid exactly as staged, so it is
//     a within-fractal fix only (see qwen_cube_transpose_probe.cpp).
//   - TransDataTo5HD on float: a stub. The half overloads are real, but they
//     consume an address list per lane, which is the same job this does.
//   - Pre-transposing V in the KV cache: correct, but it changes the cache layout
//     for every consumer and the decode append writes one row at a time, which a
//     block-granular store cannot express.
//
// What is left is the vector transpose unit: `Transpose<half>` lowers to
// vtranspose, a 16x16 uint16 transpose with no fractal precondition. It is
// reached here through one call per 16x16 block, gathering the block's rows on
// the load and scattering its columns on the store, so the input and the output
// are both plain row-major ND.
//
// One AI core per block column-strip? No: blocks are handed out grid-stride, so
// a tall matrix spreads over all 30 cores and a short one does not leave 29 idle.

#include "qwen_ascend_kernel_common.hpp"

namespace {

// vtranspose consumes 16 rows of 32 bytes: 16 halfs per lane, the unit every
// address in its list steps in. So a transpose block is exactly 16x16 halfs and
// the matrix is tiled in those.
constexpr uint32_t kTransposeLane = 16;
constexpr uint32_t kTransposeTile = 16 * 16;
constexpr uint32_t kTransposeTileBytes = kTransposeTile * sizeof(half);

// Both operands are handed to DataCopyParams in 32-byte blocks.
constexpr uint32_t kTransposeRowBlocks = kTransposeLane * sizeof(half) / pocket::kBlockBytes;

}  // namespace

// src is [rows, cols] row-major with a row stride of `src_pitch`, and dst is
// [cols, dst_pitch] row-major, holding dst[col][row] = src[row][col] for
// row < src_rows and zero for row >= src_rows.
//
// The three extra extents exist for the Cube attention path, and none is
// cosmetic. `dst_pitch` has to be the query extent rounded up to a whole number of
// Cube chunks while `src_rows` is the query extent itself, which is usually not;
// and the attention kernel reads V over that whole rounded-up range, because its
// chunk grid has no partial case either. Filling the difference with zeros is what
// lets it do that without a bounds check: those columns are multiplied by a
// probability that has already been masked to zero, so zero contributes nothing
// while uninitialised memory would contribute a NaN. `src_pitch` is the KV cache's
// head stride: one transpose covers one head, and the heads are interleaved in the
// cache rather than laid out one after another.
//
// `rows` (the destination row extent), `dst_pitch`, `src_pitch` and `cols` are all
// multiples of 16; `src_rows` may be any value up to `rows`. A ragged tile would
// otherwise read past the end of src rather than produce a wrong answer, so the
// partial source block gathers only the rows that exist and zeroes the rest.
extern "C" __global__ __aicore__ void qwen_transpose_f16_kernel(
    GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t rows, uint32_t cols, uint32_t dst_pitch,
    uint32_t src_pitch, uint32_t src_rows) {
    if (AscendC::GetBlockIdx() >= AscendC::GetBlockNum()) {
        return;
    }
    if (rows == 0 || cols == 0 || dst_pitch < rows || src_pitch < cols || src_rows > rows ||
        (rows % kTransposeLane) != 0 || (cols % kTransposeLane) != 0 ||
        (dst_pitch % kTransposeLane) != 0 || (src_pitch % kTransposeLane) != 0) {
        return;
    }

    const uint32_t col_blocks = cols / kTransposeLane;
    const uint32_t row_blocks = rows / kTransposeLane;
    const uint32_t total_blocks = row_blocks * col_blocks;
    if (total_blocks == 0) {
        return;
    }

    AscendC::GlobalTensor<half> src_g;
    AscendC::GlobalTensor<half> dst_g;
    src_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(src_gm),
                          static_cast<uint64_t>(rows) * src_pitch);
    dst_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dst_gm),
                          static_cast<uint64_t>(cols) * dst_pitch);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pack_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buf;
    pipe.InitBuffer(pack_buf, kTransposeTileBytes);
    pipe.InitBuffer(tile_buf, kTransposeTileBytes);
    AscendC::LocalTensor<half> pack = pack_buf.Get<half>();
    AscendC::LocalTensor<half> tile = tile_buf.Get<half>();

    for (uint32_t block = AscendC::GetBlockIdx(); block < total_blocks; block += AscendC::GetBlockNum()) {
        const uint32_t rb = block / col_blocks;
        const uint32_t cb = block % col_blocks;

        // How many of this tile's 16 source rows actually exist. The tiles past the
        // source end still run: their contribution is the zero fill below, which is
        // the whole point of `dst_pitch` exceeding `rows`.
        const uint32_t first_row = rb * kTransposeLane;
        const uint32_t live = first_row < src_rows
                                  ? pocket::min_u32(kTransposeLane, src_rows - first_row)
                                  : 0;

        if (live > 0) {
            // Gather the block's rows: one 32-byte piece per row, packed back to
            // back, which is the row-per-lane layout vtranspose reads.
            AscendC::DataCopyParams gather;
            gather.blockCount = static_cast<uint16_t>(live);
            gather.blockLen = kTransposeRowBlocks;
            gather.srcStride = static_cast<uint16_t>((src_pitch - kTransposeLane) * sizeof(half) /
                                                     pocket::kBlockBytes);
            gather.dstStride = 0;
            AscendC::DataCopy(pack, src_g[first_row * src_pitch + cb * kTransposeLane], gather);
            pocket::wait_load_before_compute();
        }
        if (live < kTransposeLane) {
            // The rows a partial tile does not fill are written as zero, both here
            // and, via the transpose, into the destination. Padding V with zeros
            // is safe where leaving it stale is not: the score it multiplies has
            // already been masked, so zero adds nothing and garbage adds a NaN.
            AscendC::Duplicate(pack[live * kTransposeLane], half(0.0f),
                               (kTransposeLane - live) * kTransposeLane);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Transpose(tile, pack);
        pocket::wait_compute_before_store();
        // The next gather writes `pack` again, so Vector has to be done reading it
        // as well -- the transpose above is the last reader.
        pocket::wait_compute_before_load();

        // Scatter the block's 16 columns: contiguous in `tile`, one 32-byte piece
        // per destination row, and destination rows are `dst_pitch` apart.
        AscendC::DataCopyParams scatter;
        scatter.blockCount = kTransposeLane;
        scatter.blockLen = kTransposeRowBlocks;
        scatter.srcStride = 0;
        scatter.dstStride = static_cast<uint16_t>((dst_pitch - kTransposeLane) * sizeof(half) /
                                                  pocket::kBlockBytes);
        AscendC::DataCopy(dst_g[cb * kTransposeLane * dst_pitch + first_row], tile, scatter);
        pocket::wait_store_before_compute();
    }
}
