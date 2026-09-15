// Cube (Mmad) GQA attention for the Qwen full-attention layers on
// first-generation 910.
//
// Why this kernel exists: the vector kernel beside it computes every Q * K^T dot
// product on the vector unit. Attention is 0.2% of the arithmetic in a prefill but
// was measured at 25% of the prefill time and 93% of the decode time, and the
// reason is that the whole product ran on a unit that has no business doing it.
// This kernel moves both products -- S = Q * K^T and O = P * V -- onto the Cube
// and leaves the vector unit the softmax, which is what it is for.
//
// Shape, per work item:
//
//   pass 1, per context chunk of kChunk positions
//     Q[rows, d] and K[kChunk, d] are staged GM -> L1 as ND -> NZ, then
//     S = Q * K^T is a single Mmad into L0C, read out to UB in fp16, scaled, and
//     written back to GM through Nz2Nd so pass 2 can read it one row at a time.
//
//   pass 2, per query row
//     max, exp, sum and the division by the sum, all on a row-major fp32 row.
//     The row is rewritten in GM as the normalised fp16 probability row, so the
//     PV product needs no post-division and O never has to be touched row-wise.
//
//   pass 3, per context chunk
//     P[rows, kChunk] and V^T[d, kChunk] are staged and O accumulates across
//     chunks in L0C itself, then is read out and stored.
//
// One work item is a whole (query head group, position tile) pair rather than a
// single head, and that is the difference between this kernel being usable and
// not. K and V do not depend on which query head asks for them, so an item per
// head re-reads the entire K cache once per head -- a factor of `repeat` more
// traffic than the arithmetic requires, on a part where attention is already
// bandwidth-bound rather than Cube-bound. Stacking the head group as extra rows
// of the same Mmad costs nothing: every head in a group shares one causal limit,
// one K operand and one V operand, and the stacked row order is exactly the order
// Q already has in GM (position-major, head-minor), so the A operand is one
// contiguous block and the store is one contiguous block back.
//
// The consequence is that `m` is rows_per_tile * repeat, and L0C has to hold the
// whole product: m * kChunk * 4 bytes inside 256 KiB. That is what fixes
// rows_per_tile for a given model rather than any cache argument.
//
// The GM round trip through the score scratch is deliberate rather than lazy.
// L0C leaves the Cube in a zN fractal image, and every row-wise reduction on that
// image is a gather across 16-element fractals; the one primitive that undoes the
// layout, Nz2Nd, only writes to GM on this SoC. At the sizes involved the round
// trip is 2 bytes per score in and 2 bytes back out, against 2 * d MACs per score
// on the Cube -- for d = 256 the arithmetic intensity is 64 MACs per byte moved.
//
// Two separate entries exist because the two call shapes are genuinely different,
// not because of a parameter:
//
//   causal != 0 (prefill): rows are sequence positions with individual causal
//     limits, and one work item covers rows_per_tile positions.
//   causal == 0 (decode): a single position, so the tile is the whole query head
//     group and every row shares one limit.
//
// First-generation 910 specifics this kernel depends on:
//   - no MMAD bias operand, so nothing here asks for one;
//   - no fp32 WholeReduceMax, hence the vmax halving fold below;
//   - L0C -> UB only through DataCopy under BLOCK_MODE_MATRIX, sized in KB of the
//     fp32 source rather than of the fp16 destination.

#include "qwen_ascend_kernel_common.hpp"

namespace {

// Context columns per Cube chunk. 512 fills the L1 B operand to 256 KiB at
// head_dim 256 and keeps the L0C tile inside its budget for a row tile of 96,
// which is what a 16-position tile is for the 6-head groups this model uses.
constexpr uint32_t kChunk = 512;

// Largest stacked row tile (positions * head group). The L1 and UB budgets below
// are sized from this and the host refuses to dispatch a larger one; it is also
// what L0C caps, since tile_rows * kChunk * 4 has to fit kMaxL0cBytes.
constexpr uint32_t kMaxStackedRows = 96;

// Largest head dimension the staging buffers are sized for.
constexpr uint32_t kMaxHeadDim = 256;

// The Cube's fractal axis.
constexpr uint32_t kFractal = 16;

// Stand-in for -infinity on the score rows. exp() of anything this negative is
// exactly zero in fp32, and it survives the mask without producing a NaN that
// would poison the P * V product.
constexpr float kMaskValue = -1.0e30f;

// Vector budget. The row buffers are sized by the fold width, so the host has to
// keep the score pitch inside 8 KiB; the vector kernel still covers the longer
// contexts this refuses.
constexpr uint32_t kMaxFoldWidth = 8192;
constexpr uint32_t kMaxVectorBytes = 240 * 1024;

// L0C budget for one readout. The scores and the output are read out separately,
// so this is the larger of the two -- the score image, which is tile_rows wide in
// columns of a whole chunk. Left short of the full 256 KiB because the readout
// has to cover the rounded image, not just the rows that carry data.
constexpr uint32_t kMaxL0cBytes = 192 * 1024;

__aicore__ inline uint32_t cube_round_up(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// Smallest power of two that is at least `value`, floored at the 8 lanes the
// scalar tail of the fold reads.
__aicore__ inline uint32_t cube_pow2_up(uint32_t value) {
    uint32_t power = pocket::kAlignFloat;
    while (power < value) power *= 2;
    return power;
}

// Largest of `width` contiguous fp32 lanes, where `width` is a power of two and at
// least 8. Destroys `work`.
//
// The counterpart of pocket::fold_sum and written for the same reason: this SoC
// lists `Intrinsic_vcmin` and `Intrinsic_vcmax` for float16 only, so there is no
// fp32 WholeReduceMax to call, while `Intrinsic_vmax` does have a float32 form.
// The fold is nothing but vmax.
__aicore__ inline float fold_max(const AscendC::LocalTensor<float>& work,
                                 uint32_t width) {
    for (uint32_t half = width / 2; half >= pocket::kAlignFloat; half /= 2) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(work, work, work[half], half);
    }
    pocket::wait_compute_before_scalar();
    float best = work.GetValue(0);
    for (uint32_t i = 1; i < pocket::kAlignFloat; ++i) {
        const float value = work.GetValue(i);
        if (value > best) best = value;
    }
    return best;
}

// Fill `work[limit, width)` with `fill`, leaving [0, limit) alone.
//
// The lanes above the limit carry real scores for positions the row must not
// attend to, so masking them is what makes the fold and the exp below correct --
// and it is also the only thing standing between a chunk whose K operand runs
// past the end of the cache and a NaN in the output, because those lanes are
// overwritten here before anything reads them.
//
// Duplicate covers the 8-aligned bulk and the scalar unit covers the fewer than
// eight lanes between the limit and the first aligned lane -- the alternative, a
// per-lane scalar fill, is unbounded when the limit sits just above a power of
// two.
__aicore__ inline void mask_tail(const AscendC::LocalTensor<float>& work,
                                 uint32_t limit, uint32_t width, float fill) {
    const uint32_t aligned = pocket::min_u32(
        pocket::pocket_align_up(limit, pocket::kAlignFloat), width);
    pocket::wait_compute_before_scalar();
    for (uint32_t i = limit; i < aligned; ++i) work.SetValue(i, fill);
    pocket::wait_scalar_before_compute();
    if (aligned < width) {
        AscendC::Duplicate(work[aligned], fill, width - aligned);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

}  // namespace

// Work item layout and the GM offsets it implies:
//
//   causal == 1: item = kv_head + kv_heads * block. The tile is `rows_per_tile`
//     sequence positions of one head group, stacked position-major so the Q rows
//     are `q_heads * head_dim` apart in GM, exactly as the model laid them out.
//   causal == 0: item = kv_head, one position. The tile is the whole head group,
//     whose Q rows are `head_dim` apart and therefore contiguous.
//
// `score_stride` is the row pitch of the per-core score scratch and also the row
// pitch of the transposed V image, both of which are the attention extent rounded
// up to a whole number of chunks. `score_rows` is the scratch's row count per
// core; `scratch` is laid out as one `score_rows * score_stride` region per AI
// core, because a core reuses its region across every work item it is handed and
// the total is then independent of the work count.
extern "C" __global__ __aicore__ void qwen_gqa_attention_cube_kernel(
    GM_ADDR q_rows, GM_ADDR k_cache, GM_ADDR vt_cache, GM_ADDR out_rows,
    GM_ADDR scratch, uint32_t seq_len, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t position_offset, uint32_t max_context,
    uint32_t rows_per_tile, uint32_t causal, uint32_t score_stride,
    uint32_t score_rows, float scale) {
    if (head_dim == 0 || head_dim > kMaxHeadDim || rows_per_tile == 0 ||
        score_stride == 0 || (score_stride % kChunk) != 0 ||
        (head_dim % kFractal) != 0 || q_heads == 0 || kv_heads == 0 ||
        (q_heads % kv_heads) != 0 || (q_heads / kv_heads) == 0 ||
        (head_dim % pocket::kAlignHalf) != 0) {
        return;
    }
    if (causal == 0 && (kv_heads != 1 || seq_len != 1)) {
        return;
    }

    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t cores = AscendC::GetBlockNum();
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t row_blocks = (seq_len + rows_per_tile - 1) / rows_per_tile;
    const uint32_t items = (causal != 0) ? row_blocks * kv_heads : kv_heads;
    if (items == 0) return;

    const uint32_t round_head = cube_round_up(head_dim, kFractal);
    const uint32_t fold_width = cube_pow2_up(score_stride);

    // The score scratch row count has to cover the stacked tile, and the vector
    // row buffers have to hold a whole score row plus the tile read out of L0C.
    const uint32_t max_rows = pocket::min_u32(rows_per_tile * repeat, kMaxStackedRows);
    const uint32_t max_tile_rows = cube_round_up(max_rows, kFractal);
    if (max_rows == 0 || max_rows < rows_per_tile * repeat || fold_width > kMaxFoldWidth ||
        fold_width * 6 + max_tile_rows * kChunk * sizeof(half) > kMaxVectorBytes ||
        max_tile_rows * kChunk * sizeof(float) > kMaxL0cBytes ||
        score_rows < max_tile_rows) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> q_buf;
    AscendC::TBuf<AscendC::TPosition::B1> k_buf;
    AscendC::TBuf<AscendC::TPosition::A1> p_buf;
    AscendC::TBuf<AscendC::TPosition::B1> v_buf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> tile_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stage_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    pipe.InitBuffer(q_buf, kMaxStackedRows * kMaxHeadDim * sizeof(half));
    pipe.InitBuffer(k_buf, kChunk * kMaxHeadDim * sizeof(half));
    pipe.InitBuffer(p_buf, kMaxStackedRows * kChunk * sizeof(half));
    pipe.InitBuffer(v_buf, kMaxHeadDim * kChunk * sizeof(half));
    pipe.InitBuffer(tile_buf, max_tile_rows * kChunk * sizeof(half));
    pipe.InitBuffer(stage_buf, fold_width * sizeof(half));
    pipe.InitBuffer(work_buf, fold_width * sizeof(float));

    AscendC::LocalTensor<half> q_l1 = q_buf.Get<half>();
    AscendC::LocalTensor<half> k_l1 = k_buf.Get<half>();
    AscendC::LocalTensor<half> p_l1 = p_buf.Get<half>();
    AscendC::LocalTensor<half> v_l1 = v_buf.Get<half>();
    AscendC::LocalTensor<half> tile = tile_buf.Get<half>();
    AscendC::LocalTensor<half> stage = stage_buf.Get<half>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();

    AscendC::GlobalTensor<half> q_g, k_g, v_g, out_g, score_g;
    q_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_rows),
                        static_cast<uint64_t>(seq_len) * q_heads * head_dim);
    k_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                        static_cast<uint64_t>(max_context) * kv_heads * head_dim);
    v_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vt_cache),
                        static_cast<uint64_t>(kv_heads) * head_dim * score_stride);
    out_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out_rows),
                          static_cast<uint64_t>(seq_len) * q_heads * head_dim);
    score_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scratch),
                            static_cast<uint64_t>(cores) * score_rows * score_stride);
    const uint32_t core_scores = core * score_rows * score_stride;

    // L0C is addressed directly: Gemm's c100 branch skips its marshalling copies
    // when the destination already sits on L0C, and its own UB readout is wrong on
    // this SoC -- it sizes the copy from the destination element width while L0C
    // always holds fp32, so it under-runs by 2x.
    AscendC::TBuffAddr l0c_addr;
    l0c_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::LocalTensor<half> c_l0c;
    c_l0c.SetAddr(l0c_addr);
    c_l0c.InitBuffer(0, AscendC::TOTAL_L0C_SIZE / sizeof(half));

    for (uint32_t item = core; item < items; item += cores) {
        const uint32_t kv_head = (causal != 0) ? (item % kv_heads) : item;
        const uint32_t block = (causal != 0) ? (item / kv_heads) : 0;
        const uint32_t row0 = block * rows_per_tile;
        const uint32_t positions =
            (causal != 0) ? pocket::min_u32(rows_per_tile, seq_len - row0) : 1u;
        const uint32_t rows = positions * repeat;
        const uint32_t tile_rows = cube_round_up(rows, kFractal);
        const uint32_t q_base = (row0 * q_heads + kv_head * repeat) * head_dim;
        const uint32_t q_pitch = q_heads * head_dim;
        const uint32_t limit_block = pocket::min_u32(
            position_offset + row0 + (causal != 0 ? positions : 1u), max_context);
        if (limit_block == 0 || rows > max_rows) continue;
        const uint32_t chunks = (limit_block + kChunk - 1) / kChunk;
        if (chunks * kChunk > score_stride) continue;

        // ---- Stage Q once: the same tile is the A operand of every chunk. ----
        AscendC::Nd2NzParams q_params;
        q_params.ndNum = 1;
        q_params.nValue = static_cast<uint16_t>(rows);
        q_params.dValue = static_cast<uint16_t>(head_dim);
        q_params.srcNdMatrixStride = 0;
        q_params.srcDValue = static_cast<uint16_t>(q_pitch);
        q_params.dstNzC0Stride = static_cast<uint16_t>(tile_rows);
        q_params.dstNzNStride = 1;
        q_params.dstNzMatrixStride = 0;
        AscendC::DataCopy(q_l1, q_g[q_base], q_params);

        // ---- pass 1: S = Q * K^T, chunk by chunk, into the score scratch. ----
        const AscendC::GemmTiling score_tiling =
            AscendC::GetGemmTiling<half>(rows, head_dim, kChunk);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t col0 = chunk * kChunk;
            // Positions past the end of the cache are not read: the last chunk is
            // the only one that can run over, and the lanes it leaves holding the
            // previous chunk's keys are all past `max_context`, which mask_tail
            // overwrites before anything folds them.
            const uint32_t live = pocket::min_u32(kChunk, max_context - col0);
            AscendC::Nd2NzParams k_params;
            k_params.ndNum = 1;
            k_params.nValue = static_cast<uint16_t>(live);
            k_params.dValue = static_cast<uint16_t>(head_dim);
            k_params.srcNdMatrixStride = 0;
            k_params.srcDValue = static_cast<uint16_t>(kv_heads * head_dim);
            k_params.dstNzC0Stride = static_cast<uint16_t>(kChunk);
            k_params.dstNzNStride = 1;
            k_params.dstNzMatrixStride = 0;
            AscendC::DataCopy(k_l1, k_g[(col0 * kv_heads + kv_head) * head_dim],
                              k_params);

            // The ND -> NZ conversion runs on the vector unit, so the L1 write is
            // MTE3 while the L0 load inside Gemm is MTE1.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Gemm(c_l0c, q_l1, k_l1, rows, head_dim, kChunk, score_tiling,
                          false, 0);
            AscendC::PipeBarrier<PIPE_ALL>();

            // L0C -> UB fp32 in, fp16 out. `blockLen` counts KB of the fp32 source.
            AscendC::DataCopyParams readout;
            readout.blockCount = 1;
            readout.srcStride = 0;
            readout.dstStride = 0;
            readout.blockLen = static_cast<uint16_t>(
                tile_rows * kChunk * sizeof(float) / 1024);
            AscendC::DataCopyEnhancedParams readout_mode;
            readout_mode.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
            AscendC::DataCopy(tile, c_l0c, readout, readout_mode);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Scale before the fp16 store, not after: the unscaled product of two
            // 256-wide fp16 rows can leave fp16 range, and the scaled one cannot.
            // For this model the factor is exactly 1/16, so the half rounding of
            // the scalar is exact as well.
            AscendC::Muls(tile, tile, static_cast<half>(scale),
                          tile_rows * kChunk);
            AscendC::PipeBarrier<PIPE_V>();

            // Undo the zN fractal order on the way to GM so pass 2 can address a
            // score row as one span. `srcNStride` counts 16-element units, so the
            // fractal-to-fractal distance of `tile_rows * 16` elements is
            // `tile_rows` here; `dstDStride` is the scratch row pitch.
            AscendC::Nz2NdParamsFull store_params(
                1, static_cast<uint16_t>(rows), static_cast<uint16_t>(kChunk), 1,
                static_cast<uint16_t>(tile_rows),
                static_cast<uint16_t>(score_stride), 1);
            AscendC::DataCopy(score_g[core_scores + col0], tile, store_params);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // ---- pass 2: softmax, one row at a time, in place. ----
        //
        // The row is left as exp(s - max) / sum, so pass 3 produces the finished
        // output and nothing has to divide a row of O afterwards -- which matters,
        // because O comes back out of L0C in the same fractal image the scores did.
        for (uint32_t s = 0; s < rows; ++s) {
            const uint32_t position = row0 + s / repeat;
            const uint32_t limit = pocket::min_u32(
                position_offset + (causal != 0 ? position + 1 : 1), max_context);
            if (limit == 0) continue;
            const uint32_t load_width =
                pocket::min_u32(cube_round_up(limit, pocket::kAlignHalf), fold_width);
            const uint32_t row_at = core_scores + s * score_stride;

            AscendC::DataCopy(stage, score_g[row_at], load_width);
            pocket::wait_load_before_compute();
            AscendC::Cast(work, stage, AscendC::RoundMode::CAST_NONE, load_width);
            AscendC::PipeBarrier<PIPE_V>();
            mask_tail(work, limit, fold_width, kMaskValue);
            const float maximum = fold_max(work, fold_width);

            // Second read of the same row: fold_max consumed the first copy, and
            // recomputing exp from the scratch is cheaper than keeping a copy of a
            // row that may be thousands of lanes wide.
            pocket::wait_compute_before_load();
            AscendC::DataCopy(stage, score_g[row_at], load_width);
            pocket::wait_load_before_compute();
            AscendC::Cast(work, stage, AscendC::RoundMode::CAST_NONE, load_width);
            AscendC::PipeBarrier<PIPE_V>();
            mask_tail(work, limit, fold_width, kMaskValue);
            AscendC::Adds(work, work, -maximum, fold_width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(work, work, fold_width);
            AscendC::PipeBarrier<PIPE_V>();

            // Keep the probabilities in fp16 before fold_sum destroys the fp32
            // row, then fold, then scale the stored copy by the reciprocal. The
            // division is a scalar one, which is exact on this part where the
            // vector Reciprocal is only ~2e-3.
            AscendC::Cast(stage, work, AscendC::RoundMode::CAST_NONE, fold_width);
            AscendC::PipeBarrier<PIPE_V>();
            const float total = pocket::fold_sum(work, fold_width);
            const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
            pocket::wait_scalar_before_compute();
            AscendC::Muls(stage, stage, static_cast<half>(inverse), fold_width);
            AscendC::PipeBarrier<PIPE_V>();
            pocket::wait_compute_before_store();
            // Only the scored columns are written: the fold runs over a power of
            // two, which can be wider than the scratch row, and letting it store
            // its full width would write into the next row.
            AscendC::DataCopy(score_g[row_at], stage, score_stride);
            pocket::wait_store_before_load();
        }

        // ---- pass 3: O = P * V, accumulated across chunks in L0C. ----
        //
        // V is contracted along its row axis, so the operand has to be the
        // transposed cache: [head_dim, positions]. The host produced it with
        // qwen_transpose_f16_kernel on the way in; nothing in the ND2NZ staging
        // path can transpose across fractals.
        const AscendC::GemmTiling value_tiling =
            AscendC::GetGemmTiling<half>(rows, kChunk, head_dim);
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t col0 = chunk * kChunk;
            AscendC::Nd2NzParams p_params;
            p_params.ndNum = 1;
            p_params.nValue = static_cast<uint16_t>(rows);
            p_params.dValue = static_cast<uint16_t>(kChunk);
            p_params.srcNdMatrixStride = 0;
            p_params.srcDValue = static_cast<uint16_t>(score_stride);
            p_params.dstNzC0Stride = static_cast<uint16_t>(tile_rows);
            p_params.dstNzNStride = 1;
            p_params.dstNzMatrixStride = 0;
            AscendC::DataCopy(p_l1, score_g[core_scores + col0], p_params);

            AscendC::Nd2NzParams v_params;
            v_params.ndNum = 1;
            v_params.nValue = static_cast<uint16_t>(round_head);
            v_params.dValue = static_cast<uint16_t>(kChunk);
            v_params.srcNdMatrixStride = 0;
            v_params.srcDValue = static_cast<uint16_t>(score_stride);
            v_params.dstNzC0Stride = static_cast<uint16_t>(round_head);
            v_params.dstNzNStride = 1;
            v_params.dstNzMatrixStride = 0;
            AscendC::DataCopy(v_l1,
                              v_g[static_cast<uint64_t>(kv_head) * head_dim *
                                      score_stride + col0],
                              v_params);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();

            // initValue 1 leaves L0C alone before the first k-iteration, which is
            // what makes the accumulation cross the chunk boundary; the first chunk
            // starts from a zeroed accumulator instead.
            AscendC::Gemm(c_l0c, p_l1, v_l1, rows, kChunk, head_dim, value_tiling,
                          false, chunk == 0 ? 0 : 1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::DataCopyParams readout;
        readout.blockCount = 1;
        readout.srcStride = 0;
        readout.dstStride = 0;
        readout.blockLen = static_cast<uint16_t>(
            tile_rows * round_head * sizeof(float) / 1024);
        AscendC::DataCopyEnhancedParams readout_mode;
        readout_mode.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(tile, c_l0c, readout, readout_mode);
        AscendC::PipeBarrier<PIPE_ALL>();

        // The stacked row order is the GM order, so the whole tile lands with one
        // store and the head groups need no reassembly.
        AscendC::Nz2NdParamsFull out_params(
            1, static_cast<uint16_t>(rows), static_cast<uint16_t>(head_dim), 1,
            static_cast<uint16_t>(tile_rows), static_cast<uint16_t>(q_pitch), 1);
        AscendC::DataCopy(out_g[q_base], tile, out_params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// Context-partitioned decode attention on the Cube, the many-core counterpart of
// the entry above.
//
// The entry above is one work item per KV head, which for the real TP4 shape is one
// work item. Measured at a 4097-token context it takes 1220 us, and the measured
// work is 9 chunks by 6 rows by 256 wide -- roughly 25 MFLOP on the Cube, or about
// 12 us of arithmetic. So the 1220 us is not arithmetic and not the K/V read
// either: it is a single core walking nine chunks through a strictly serial
// stage-store-reload-softmax-reload pipeline, with nothing to overlap against.
// Thirty cores are idle while that happens.
//
// This entry splits the chunk axis instead of the head axis. Each block takes a
// contiguous run of chunks, computes its own online softmax over just those
// positions, and writes an unnormalized partial -- (local max, local denominator,
// unnormalized output) -- for the reduce kernel below to merge. K and V are still
// read exactly once in total across the grid, so the traffic does not grow; only
// the latency does not have to be paid serially.
//
// The partials are written in the two pieces a core can actually address after an
// L0C readout:
//
//   o_partials   fp16 [num_partitions, q_heads, head_dim], one contiguous packed
//                row per query head, written by the same Nz2Nd store the single
//                core path uses. The cross-partition sum runs in fp32 in the
//                reduce kernel; only the per-partition term is rounded, which is
//                the same rounding the single core path already applies once to
//                its finished output.
//   stats        fp32 [num_partitions, q_heads, 2] holding {max, denominator},
//                written scalar side because two values do not justify a copy.
//
// The row order inside a block is the head group, so the only thing a partition
// changes is which columns of K and V it walks.
extern "C" __global__ __aicore__ void qwen_gqa_attention_cube_partial_kernel(
    GM_ADDR q_rows, GM_ADDR k_cache, GM_ADDR vt_cache, GM_ADDR o_partials,
    GM_ADDR stats, GM_ADDR scratch, uint32_t q_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t context_len, uint32_t max_context,
    uint32_t chunks_per_partition, uint32_t part_stride, uint32_t vt_stride,
    uint32_t num_partitions, float scale) {
    if (head_dim == 0 || head_dim > kMaxHeadDim || part_stride == 0 ||
        (part_stride % kChunk) != 0 || (head_dim % kFractal) != 0 || q_heads == 0 ||
        kv_heads == 0 || (q_heads % kv_heads) != 0 || chunks_per_partition == 0 ||
        num_partitions == 0 || (head_dim % pocket::kAlignHalf) != 0) {
        return;
    }
    const uint32_t item = AscendC::GetBlockIdx();
    if (item >= num_partitions * kv_heads) return;

    const uint32_t kv_head = item % kv_heads;
    const uint32_t part = item / kv_heads;
    const uint32_t repeat = q_heads / kv_heads;
    // One query position: the tile is the whole head group, as in the decode half
    // of the entry above.
    const uint32_t rows = repeat;
    if (rows == 0 || rows > kMaxStackedRows) return;

    const uint32_t tile_rows = cube_round_up(rows, kFractal);
    const uint32_t round_head = cube_round_up(head_dim, kFractal);
    const uint32_t fold_width = cube_pow2_up(part_stride);
    const uint32_t score_rows = tile_rows;
    const uint32_t partial_stride = 2 + head_dim;

    // The scratch row buffers are sized from the partition's own width rather than
    // from the whole context, which is the second reason this is faster than the
    // single core entry at long context: the fold runs over one partition's
    // columns instead of over every position in the cache.
    if (fold_width > kMaxFoldWidth ||
        fold_width * 6 + tile_rows * kChunk * sizeof(half) > kMaxVectorBytes ||
        tile_rows * kChunk * sizeof(float) > kMaxL0cBytes) {
        return;
    }

    const uint32_t chunks_total = (context_len + kChunk - 1) / kChunk;
    const uint32_t chunk_begin = part * chunks_per_partition;
    if (chunk_begin >= chunks_total) return;
    const uint32_t chunk_end =
        pocket::min_u32(chunks_total, chunk_begin + chunks_per_partition);
    const uint32_t active = chunk_end - chunk_begin;

    // Columns of this partition that carry real positions. Everything past it in
    // the scratch row is either padding the softmax left at zero or a lane the
    // K/V stores never touched, and masking to the fold width covers both.
    const uint32_t part_live = pocket::min_u32(
        context_len - chunk_begin * kChunk, part_stride);
    if (part_live == 0) return;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> q_buf;
    AscendC::TBuf<AscendC::TPosition::B1> k_buf;
    AscendC::TBuf<AscendC::TPosition::A1> p_buf;
    AscendC::TBuf<AscendC::TPosition::B1> v_buf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> tile_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stage_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    pipe.InitBuffer(q_buf, kMaxStackedRows * kMaxHeadDim * sizeof(half));
    pipe.InitBuffer(k_buf, kChunk * kMaxHeadDim * sizeof(half));
    pipe.InitBuffer(p_buf, kMaxStackedRows * kChunk * sizeof(half));
    pipe.InitBuffer(v_buf, kMaxHeadDim * kChunk * sizeof(half));
    pipe.InitBuffer(tile_buf, score_rows * kChunk * sizeof(half));
    pipe.InitBuffer(stage_buf, fold_width * sizeof(half));
    pipe.InitBuffer(work_buf, fold_width * sizeof(float));

    AscendC::LocalTensor<half> q_l1 = q_buf.Get<half>();
    AscendC::LocalTensor<half> k_l1 = k_buf.Get<half>();
    AscendC::LocalTensor<half> p_l1 = p_buf.Get<half>();
    AscendC::LocalTensor<half> v_l1 = v_buf.Get<half>();
    AscendC::LocalTensor<half> tile = tile_buf.Get<half>();
    AscendC::LocalTensor<half> stage = stage_buf.Get<half>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();

    AscendC::GlobalTensor<half> q_g, k_g, v_g, o_g, score_g;
    q_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_rows),
                        static_cast<uint64_t>(q_heads) * head_dim);
    k_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_cache),
                        static_cast<uint64_t>(max_context) * kv_heads * head_dim);
    v_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(vt_cache),
                        static_cast<uint64_t>(kv_heads) * head_dim * vt_stride);
    o_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o_partials),
                        static_cast<uint64_t>(num_partitions) * q_heads * head_dim);
    score_g.SetGlobalBuffer(
        reinterpret_cast<__gm__ half*>(scratch),
        static_cast<uint64_t>(num_partitions) * kv_heads * score_rows * part_stride);
    AscendC::GlobalTensor<float> stats_g;
    stats_g.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(stats),
        static_cast<uint64_t>(num_partitions) * kv_heads * q_heads *
            (partial_stride - head_dim));

    // Each block owns one scratch region for the whole of its life, so a lingering
    // MTE3 from a previous store cannot be read back by the next block.
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint32_t q_base = kv_head * repeat * head_dim;
    const uint32_t q_pitch = q_heads * head_dim;
    const uint32_t core_scores = item * score_rows * part_stride;

    AscendC::TBuffAddr l0c_addr;
    l0c_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::LocalTensor<half> c_l0c;
    c_l0c.SetAddr(l0c_addr);
    c_l0c.InitBuffer(0, AscendC::TOTAL_L0C_SIZE / sizeof(half));

    // ---- Stage Q once: the same tile is the A operand of every chunk. ----
    AscendC::Nd2NzParams q_params;
    q_params.ndNum = 1;
    q_params.nValue = static_cast<uint16_t>(rows);
    q_params.dValue = static_cast<uint16_t>(head_dim);
    q_params.srcNdMatrixStride = 0;
    q_params.srcDValue = static_cast<uint16_t>(q_pitch);
    q_params.dstNzC0Stride = static_cast<uint16_t>(tile_rows);
    q_params.dstNzNStride = 1;
    q_params.dstNzMatrixStride = 0;
    AscendC::DataCopy(q_l1, q_g[q_base], q_params);

    // ---- pass 1: S = Q * K^T over this partition's chunks. ----
    const AscendC::GemmTiling score_tiling =
        AscendC::GetGemmTiling<half>(rows, head_dim, kChunk);
    for (uint32_t c = 0; c < active; ++c) {
        const uint32_t col0 = (chunk_begin + c) * kChunk;
        const uint32_t local0 = c * kChunk;
        const uint32_t live = pocket::min_u32(kChunk, max_context - col0);
        AscendC::Nd2NzParams k_params;
        k_params.ndNum = 1;
        k_params.nValue = static_cast<uint16_t>(live);
        k_params.dValue = static_cast<uint16_t>(head_dim);
        k_params.srcNdMatrixStride = 0;
        k_params.srcDValue = static_cast<uint16_t>(kv_heads * head_dim);
        k_params.dstNzC0Stride = static_cast<uint16_t>(kChunk);
        k_params.dstNzNStride = 1;
        k_params.dstNzMatrixStride = 0;
        AscendC::DataCopy(k_l1, k_g[(col0 * kv_heads + kv_head) * head_dim],
                          k_params);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Gemm(c_l0c, q_l1, k_l1, rows, head_dim, kChunk, score_tiling,
                      false, 0);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyParams readout;
        readout.blockCount = 1;
        readout.srcStride = 0;
        readout.dstStride = 0;
        readout.blockLen = static_cast<uint16_t>(
            tile_rows * kChunk * sizeof(float) / 1024);
        AscendC::DataCopyEnhancedParams readout_mode;
        readout_mode.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(tile, c_l0c, readout, readout_mode);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Muls(tile, tile, static_cast<half>(scale),
                      tile_rows * kChunk);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Nz2NdParamsFull store_params(
            1, static_cast<uint16_t>(rows), static_cast<uint16_t>(kChunk), 1,
            static_cast<uint16_t>(tile_rows),
            static_cast<uint16_t>(part_stride), 1);
        AscendC::DataCopy(score_g[core_scores + local0], tile, store_params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // ---- pass 2: local softmax per row, left unnormalized. ----
    //
    // The single core entry divides by the row sum here and never has to touch the
    // output again. A partition cannot: its sum only covers its own columns, so the
    // division has to wait for the reduce. What is stored is therefore
    // exp(s - m_part), which is exactly the P operand the value product wants once
    // the reduce has rescaled by exp(m_part - M).
    for (uint32_t s = 0; s < rows; ++s) {
        const uint32_t row_at = core_scores + s * part_stride;
        // Clamped to the scratch row: the fold width can be up to twice the row
        // pitch, and reading the difference would walk off the end of the region.
        const uint32_t load_width = pocket::min_u32(
            cube_round_up(part_live, pocket::kAlignHalf), part_stride);

        AscendC::DataCopy(stage, score_g[row_at], load_width);
        pocket::wait_load_before_compute();
        AscendC::Cast(work, stage, AscendC::RoundMode::CAST_NONE, load_width);
        AscendC::PipeBarrier<PIPE_V>();
        mask_tail(work, part_live, fold_width, kMaskValue);
        const float maximum = fold_max(work, fold_width);

        pocket::wait_compute_before_load();
        AscendC::DataCopy(stage, score_g[row_at], load_width);
        pocket::wait_load_before_compute();
        AscendC::Cast(work, stage, AscendC::RoundMode::CAST_NONE, load_width);
        AscendC::PipeBarrier<PIPE_V>();
        mask_tail(work, part_live, fold_width, kMaskValue);
        AscendC::Adds(work, work, -maximum, fold_width);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(work, work, fold_width);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(stage, work, AscendC::RoundMode::CAST_NONE, fold_width);
        AscendC::PipeBarrier<PIPE_V>();
        const float total = pocket::fold_sum(work, fold_width);
        pocket::wait_compute_before_store();
        AscendC::DataCopy(score_g[row_at], stage, part_stride);

        pocket::wait_compute_before_scalar();
        const uint32_t head = kv_head * repeat + s;
        const uint32_t stat_at = (part * q_heads + head) * 2;
        stats_g.SetValue(stat_at, maximum);
        stats_g.SetValue(stat_at + 1, total);
        pocket::wait_scalar_before_compute();
        pocket::wait_store_before_load();
    }

    // ---- pass 3: unnormalized O = P * V, accumulated across the partition. ----
    const AscendC::GemmTiling value_tiling =
        AscendC::GetGemmTiling<half>(rows, kChunk, head_dim);
    for (uint32_t c = 0; c < active; ++c) {
        const uint32_t col0 = (chunk_begin + c) * kChunk;
        const uint32_t local0 = c * kChunk;
        AscendC::Nd2NzParams p_params;
        p_params.ndNum = 1;
        p_params.nValue = static_cast<uint16_t>(rows);
        p_params.dValue = static_cast<uint16_t>(kChunk);
        p_params.srcNdMatrixStride = 0;
        p_params.srcDValue = static_cast<uint16_t>(part_stride);
        p_params.dstNzC0Stride = static_cast<uint16_t>(tile_rows);
        p_params.dstNzNStride = 1;
        p_params.dstNzMatrixStride = 0;
        AscendC::DataCopy(p_l1, score_g[core_scores + local0], p_params);

        AscendC::Nd2NzParams v_params;
        v_params.ndNum = 1;
        v_params.nValue = static_cast<uint16_t>(round_head);
        v_params.dValue = static_cast<uint16_t>(kChunk);
        v_params.srcNdMatrixStride = 0;
        v_params.srcDValue = static_cast<uint16_t>(vt_stride);
        v_params.dstNzC0Stride = static_cast<uint16_t>(round_head);
        v_params.dstNzNStride = 1;
        v_params.dstNzMatrixStride = 0;
        AscendC::DataCopy(v_l1,
                          v_g[static_cast<uint64_t>(kv_head) * head_dim *
                                  vt_stride + col0],
                          v_params);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Gemm(c_l0c, p_l1, v_l1, rows, kChunk, head_dim, value_tiling,
                      false, c == 0 ? 0 : 1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::DataCopyParams out_readout;
    out_readout.blockCount = 1;
    out_readout.srcStride = 0;
    out_readout.dstStride = 0;
    out_readout.blockLen = static_cast<uint16_t>(
        tile_rows * round_head * sizeof(float) / 1024);
    AscendC::DataCopyEnhancedParams out_mode;
    out_mode.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(tile, c_l0c, out_readout, out_mode);
    AscendC::PipeBarrier<PIPE_ALL>();

    // Packed rather than pitched: the reduce kernel wants one contiguous row per
    // head, and the head group is already contiguous in the tile.
    AscendC::Nz2NdParamsFull out_params(
        1, static_cast<uint16_t>(rows), static_cast<uint16_t>(head_dim), 1,
        static_cast<uint16_t>(tile_rows), static_cast<uint16_t>(head_dim), 1);
    AscendC::DataCopy(
        o_g[static_cast<uint64_t>(part) * q_heads * head_dim +
            static_cast<uint64_t>(kv_head) * repeat * head_dim],
        tile, out_params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Number of partitions the reduce kernel's weight vector is sized for. The
// launcher never asks for more: the context is bounded by the cache pitch and each
// partition owns at least one 512-column chunk.
constexpr uint32_t kMaxCubePartitions = 64;

// Merge the per-partition partials written by the kernel above.
//
// One block per query head. The partition holding the largest local max sets the
// reference M, every partition contributes exp(m_p - M) to the denominator and the
// same weight to the unnormalized output sum, and the result is the ratio. That is
// the rescaling the online softmax inside one core already does, lifted one level:
// partitions are just more key ranges.
//
// The weights for one head are raised in a single vector Exp over the partition
// axis rather than one scalar_exp per partition, because scalar_exp costs a full
// vector round trip and there are up to sixty-four of them per head.
extern "C" __global__ __aicore__ void qwen_gqa_attention_cube_reduce_kernel(
    GM_ADDR o_partials, GM_ADDR stats, GM_ADDR out, uint32_t q_heads,
    uint32_t head_dim, uint32_t num_partitions) {
    if (q_heads == 0 || head_dim == 0 || head_dim > kMaxHeadDim ||
        num_partitions == 0 || num_partitions > kMaxCubePartitions ||
        (head_dim % pocket::kAlignFloat) != 0) {
        return;
    }
    const uint32_t width = cube_pow2_up(num_partitions);
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t cores = AscendC::GetBlockNum();

    AscendC::GlobalTensor<half> o_g;
    AscendC::GlobalTensor<float> stats_g;
    AscendC::GlobalTensor<half> out_g;
    o_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(o_partials),
                        static_cast<uint64_t>(num_partitions) * q_heads * head_dim);
    stats_g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(stats),
                            static_cast<uint64_t>(num_partitions) * q_heads * 2);
    out_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out),
                          static_cast<uint64_t>(q_heads) * head_dim);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weight_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accum_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> row_buf;
    pipe.InitBuffer(weight_buf, kMaxCubePartitions * sizeof(float));
    pipe.InitBuffer(accum_buf, kMaxHeadDim * sizeof(float));
    pipe.InitBuffer(work_buf, kMaxHeadDim * sizeof(float));
    pipe.InitBuffer(row_buf, kMaxHeadDim * sizeof(half));

    AscendC::LocalTensor<float> weights = weight_buf.Get<float>();
    AscendC::LocalTensor<float> accum = accum_buf.Get<float>();
    AscendC::LocalTensor<float> work = work_buf.Get<float>();
    AscendC::LocalTensor<half> row = row_buf.Get<half>();

    for (uint32_t head = core; head < q_heads; head += cores) {
        float global_max = -3.402823466e+38F;
        uint64_t live = 0;
        for (uint32_t p = 0; p < num_partitions; ++p) {
            const float denominator = stats_g.GetValue((p * q_heads + head) * 2 + 1);
            if (denominator <= 0.0f) continue;
            ++live;
            const float maximum = stats_g.GetValue((p * q_heads + head) * 2);
            if (maximum > global_max) global_max = maximum;
        }
        if (live == 0) {
            // Nothing in this head's context is attended to; the row is defined as
            // zero rather than left at whatever the caller had there.
            AscendC::Duplicate(accum, 0.0f, head_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(row, accum, AscendC::RoundMode::CAST_NONE, head_dim);
            AscendC::PipeBarrier<PIPE_V>();
            pocket::wait_compute_before_store();
            AscendC::DataCopy(out_g[head * head_dim], row, head_dim);
            pocket::wait_store_before_load();
            continue;
        }

        // One Exp for the whole partition axis. Empty partitions are given -1e30 so
        // their weight is exactly zero and the accumulation can skip them by index
        // without a second pass over stats.
        AscendC::Duplicate(weights, -1.0e30f, kMaxCubePartitions);
        AscendC::PipeBarrier<PIPE_V>();
        pocket::wait_compute_before_scalar();
        for (uint32_t p = 0; p < num_partitions; ++p) {
            if (stats_g.GetValue((p * q_heads + head) * 2 + 1) <= 0.0f) continue;
            weights.SetValue(
                p, stats_g.GetValue((p * q_heads + head) * 2) - global_max);
        }
        pocket::wait_scalar_before_compute();
        AscendC::Exp(weights, weights, width);
        pocket::wait_compute_before_scalar();

        float denominators[kMaxCubePartitions];
        float terms[kMaxCubePartitions];
        float total = 0.0f;
        for (uint32_t p = 0; p < num_partitions; ++p) {
            const float denominator = stats_g.GetValue((p * q_heads + head) * 2 + 1);
            denominators[p] = denominator;
            terms[p] = denominator > 0.0f ? weights.GetValue(p) : 0.0f;
            total += denominator * terms[p];
        }

        AscendC::Duplicate(accum, 0.0f, head_dim);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t p = 0; p < num_partitions; ++p) {
            if (denominators[p] <= 0.0f) continue;
            AscendC::DataCopy(
                row,
                o_g[static_cast<uint64_t>(p) * q_heads * head_dim +
                    static_cast<uint64_t>(head) * head_dim],
                head_dim);
            pocket::wait_load_before_compute();
            AscendC::Cast(work, row, AscendC::RoundMode::CAST_NONE, head_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(work, work, terms[p], head_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accum, accum, work, head_dim);
            AscendC::PipeBarrier<PIPE_V>();
            // The next partition overwrites `row`; the Cast above has to be done
            // reading it first.
            pocket::wait_compute_before_load();
        }

        const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
        pocket::wait_scalar_before_compute();
        AscendC::Muls(accum, accum, inverse, head_dim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(row, accum, AscendC::RoundMode::CAST_NONE, head_dim);
        AscendC::PipeBarrier<PIPE_V>();
        pocket::wait_compute_before_store();
        AscendC::DataCopy(out_g[head * head_dim], row, head_dim);
        pocket::wait_store_before_load();
    }
}
