// Single-tile Cube probe: C[m, n] = A[m, k] * B[n, k]^T on one AI core.
//
// This exists to settle, by measurement rather than by reading headers, the two
// things the hand-written Cube attention kernel depends on and that no code in
// this repository has ever exercised:
//
//   1. The L1 layout `AscendC::Gemm` expects for its operands. Gemm reads its
//      inputs through the v1 LoadData path, whose fractal offsets are only
//      documented indirectly; feeding it the layout produced by
//      `DataCopy(l1, gm, Nd2NzParams)` is either right or it is silently
//      wrong, so the host test compares against a CPU reference.
//   2. That the L0C -> UB readout works on dav_c100 at all. Fixpipe is an
//      unsupported stub on this SoC, and Gemm's own c100 branch replaces it
//      with a `DataCopy` under `BLOCK_MODE_MATRIX` after a full pipe barrier.
//
// The kernel is deliberately one core, one tile, no pipelining: it is a layout
// oracle, not a fast GEMM. Everything here is first-generation 910 only.

#include "qwen_ascend_kernel_common.hpp"

// The whole tile has to fit the on-chip budgets at once: L1 holds both operands
// (the ND->NZ staging is only used for GM -> L1, so there is no room for a
// second stage), and UB holds the fp16 accumulator.
constexpr uint32_t kProbeL1Bytes = 1024 * 1024;
constexpr uint32_t kProbeUbBytes = 256 * 1024;

// GetGemmTiling rounds both axes up to a multiple of GemmTiling::blockSize,
// which is 16 -- the comment inside GetGemmTiling that calls it "16 * 16" is
// wrong. The L0C buffer, and therefore the UB image of it, is allocated at the
// rounded size, and the zN fractal column stride is roundM * 16 elements, so a
// kernel that assumes the logical m gets the fractals wrong by exactly that
// factor. Measured: assuming 256 here left every shape whose m or n was not
// already a multiple of 256 off by an order of magnitude.
constexpr uint32_t kProbeTileAxis = 16;

__aicore__ inline uint32_t probe_round_up(uint32_t v, uint32_t multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

// m x k fp16 operand plus n x k fp16 operand must fit L1.
__aicore__ inline bool probe_l1_fits(uint32_t m, uint32_t n, uint32_t k) {
    return (m + n) * k * sizeof(half) <= kProbeL1Bytes;
}

__aicore__ inline bool probe_ub_fits(uint32_t m, uint32_t n) {
    const uint32_t round_m = probe_round_up(m, kProbeTileAxis);
    const uint32_t round_n = probe_round_up(n, kProbeTileAxis);
    return round_m * round_n * sizeof(half) <= kProbeUbBytes;
}

extern "C" __global__ __aicore__ void qwen_cube_gemm_probe_kernel(
    GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR c_gm,
    uint32_t m, uint32_t n, uint32_t k, uint32_t mode, uint32_t readout_len) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    if (!probe_l1_fits(m, n, k) || !probe_ub_fits(m, n)) {
        // Write the sentinel the host looks for so a too-large tile is a clear
        // failure instead of a garbage result.
        AscendC::GlobalTensor<half> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c_gm), m * n);
        out.SetValue(0, static_cast<half>(-32768.0f));
        return;
    }
    const uint32_t round_m = probe_round_up(m, kProbeTileAxis);
    const uint32_t round_n = probe_round_up(n, kProbeTileAxis);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a1_buf;
    AscendC::TBuf<AscendC::TPosition::B1> b1_buf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> out_buf;
    pipe.InitBuffer(a1_buf, m * k * sizeof(half));
    pipe.InitBuffer(b1_buf, n * k * sizeof(half));
    pipe.InitBuffer(out_buf, round_m * round_n * sizeof(half));

    AscendC::GlobalTensor<half> a_g, b_g, c_g;
    a_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a_gm), m * k);
    b_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_gm), n * k);
    c_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c_gm), m * n);

    AscendC::LocalTensor<half> a_l1 = a1_buf.Get<half>();
    AscendC::LocalTensor<half> b_l1 = b1_buf.Get<half>();
    AscendC::LocalTensor<half> c_ub = out_buf.Get<half>();

    // Plain row-major [rows, cols] global input, converted to the fractal L1
    // layout on the way in. `dstNzC0Stride` is the row count of the NZ image,
    // counted in elements, and the requirement follows from Gemm's LoadL0A /
    // LoadL0B: they index the 16x16 block (k-block, axis-block) at
    // (kBlock * axisBlockNum + axisBlock) * 256 elements, which is what
    // Nd2NzParams produces when `dstNzC0Stride` is that axis length.
    AscendC::Nd2NzParams a_params;
    a_params.ndNum = 1;
    a_params.nValue = static_cast<uint16_t>(m);
    a_params.dValue = static_cast<uint16_t>(k);
    a_params.srcNdMatrixStride = 0;
    a_params.srcDValue = static_cast<uint16_t>(k);
    a_params.dstNzC0Stride = static_cast<uint16_t>(round_m);
    a_params.dstNzNStride = 1;
    a_params.dstNzMatrixStride = 0;
    AscendC::DataCopy(a_l1, a_g, a_params);

    AscendC::Nd2NzParams b_params;
    b_params.ndNum = 1;
    b_params.nValue = static_cast<uint16_t>(n);
    b_params.dValue = static_cast<uint16_t>(k);
    b_params.srcNdMatrixStride = 0;
    b_params.srcDValue = static_cast<uint16_t>(k);
    b_params.dstNzC0Stride = static_cast<uint16_t>(round_n);
    b_params.dstNzNStride = 1;
    b_params.dstNzMatrixStride = 0;
    AscendC::DataCopy(b_l1, b_g, b_params);

    // The ND->NZ conversion bounces through UB on the vector unit, so the L1
    // write is MTE3 while the L0 load inside Gemm is MTE1.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();

    // Accumulate straight into L0C. Gemm's c100 branch skips both marshalling
    // copies when dst is already on L0C (it only pre-copies and reads back when
    // dst is UB), which matters because Gemm's own readout is wrong on this SoC:
    // it sizes the copy as `roundM * roundN * sizeof(T) / 1024`, using the
    // *destination* element size, while L0C always holds fp32. That under-runs
    // the copy by 2x and leaves UB half uninitialised -- measured, not inferred:
    // sweeping the unit in the host test shows the correct value is the fp32
    // byte count in KB, and anything smaller leaves the buffer short.
    AscendC::TBuffAddr l0c_addr;
    l0c_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::LocalTensor<half> c_l0c;
    c_l0c.SetAddr(l0c_addr);
    c_l0c.InitBuffer(0, AscendC::TOTAL_L0C_SIZE / sizeof(half));

    AscendC::GemmTiling tiling = AscendC::GetGemmTiling<half>(m, k, n);
    AscendC::Gemm(c_l0c, a_l1, b_l1, m, k, n, tiling, false, 0);
    AscendC::PipeBarrier<PIPE_ALL>();

    // L0C -> UB, fp32 in and fp16 out, `blockLen` counted in KB of the fp32
    // source. An explicit override exists only so the host test can sweep the
    // unit rather than take this one on faith.
    const uint32_t default_readout = round_m * round_n * sizeof(float) / 1024;
    AscendC::DataCopyParams ro_params;
    ro_params.blockCount = 1;
    ro_params.srcStride = 0;
    ro_params.dstStride = 0;
    ro_params.blockLen = static_cast<uint16_t>(readout_len == 0 ? default_readout : readout_len);
    AscendC::DataCopyEnhancedParams ro_enhanced;
    ro_enhanced.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(c_ub, c_l0c, ro_params, ro_enhanced);
    AscendC::PipeBarrier<PIPE_ALL>();

    // mode 0: hand back the logical [m, n] product. UB holds the zN fractal
    // image the Cube left behind -- element (row, col) sits at
    // (col / 16) * roundM * 16 + (row / 16) * 256 + (row % 16) * 16 + (col % 16)
    // -- so a row-major store would emit the fractals in that order rather than
    // the matrix. `DataCopyUB2GMNZ2NDImpl`, reached by passing Nz2NdParamsFull,
    // is the vendor's undo for exactly this layout: it walks the source one
    // 16-element column fractal at a time and writes each into a row-pitched
    // destination. `srcNStride` counts 16-element units, so the fractal-to-
    // fractal distance of roundM * 16 elements means srcNStride == roundM.
    if (mode != 1) {
        AscendC::Nz2NdParamsFull out_params(1, static_cast<uint16_t>(m),
                                            static_cast<uint16_t>(n), 1,
                                            static_cast<uint16_t>(round_m),
                                            static_cast<uint16_t>(n), 1);
        AscendC::DataCopy(c_g, c_ub, out_params);
        AscendC::PipeBarrier<PIPE_ALL>();
        return;
    }

    // mode 1: hand back the UB image untouched, in the layout the readout
    // actually left it in, so the caller can decode the fractal order from a
    // known product instead of inferring it from a row-major read.
    AscendC::GlobalTensor<half> raw_g;
    raw_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c_gm), round_m * round_n);
    const uint32_t total = round_m * round_n;
    const uint32_t max_elements = 4095u * 16u;
    AscendC::DataCopyParams raw_params;
    raw_params.blockCount = 1;
    raw_params.srcStride = 0;
    raw_params.dstStride = 0;
    for (uint32_t off = 0; off < total; off += max_elements) {
        const uint32_t left = total - off;
        raw_params.blockLen = static_cast<uint16_t>((left < max_elements ? left : max_elements) / 16);
        AscendC::DataCopy(raw_g[off], c_ub[off], raw_params);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
