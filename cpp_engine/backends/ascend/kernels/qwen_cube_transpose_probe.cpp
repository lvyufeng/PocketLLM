// Hand-rolled Cube path: C[m, n] = A[m, k] * B[k, n] with the B operand supplied
// in its natural [k, n] layout and transposed by the L1 -> L0B load.
//
// Why this probe exists: the hand-written attention kernel needs O = P * V, and
// the Cube contracts along the *column* axis of both ND operands. P is naturally
// [m, j] (contraction j is its column axis) but the V cache is naturally
// [j, d] -- j is V's *row* axis. So the second matmul is impossible without a
// transpose of V, and the transpose has to happen somewhere cheap. The other
// candidates were ruled out by reading the headers rather than by guessing:
//
//   - UB -> L1 with Nd2NzParams: DataCopyUB2L1ND2NZImpl has no dav_c100
//     definition at all (only c220, c310, m200), so it cannot link here.
//   - A vector-unit transpose: TransDataTo5HD for float is a stub, and the
//     half overloads need the matrix to already be in fractal form.
//   - Producing a transposed ND matrix in GM and staging that: correct, but it
//     costs a full extra read+write of V per layer per attention call.
//
// That leaves `LoadData2DParams.ifTranspose`. dav_c100/kernel_operator_mm_impl.h
// does pass it through to load_cbuf_to_cb for the v1 API (only the separate
// LoadDataWithTranspose entry points are stubbed), but nothing states what L1
// layout it expects. So: measure it.
//
// The kernel is one core, one tile, no pipelining. It is a semantics oracle.

#include "qwen_ascend_kernel_common.hpp"

constexpr uint32_t kTpL1Bytes = 1024 * 1024;
constexpr uint32_t kTpUbBytes = 256 * 1024;
constexpr uint32_t kTpTileAxis = 16;

__aicore__ inline uint32_t tp_round_up(uint32_t v, uint32_t multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

// A is [m, k] and B is [k, n], so L1 has to hold both in full.
__aicore__ inline bool tp_l1_fits(uint32_t m, uint32_t n, uint32_t k) {
    return (m * k + n * k) * sizeof(half) <= kTpL1Bytes;
}

// L0A holds the whole A tile as m/16 * k/16 fractals, L0B the same for B, and
// L0C the fp32 product. Any of the three overflowing would be a silent wrap.
__aicore__ inline bool tp_l0_fits(uint32_t m, uint32_t n, uint32_t k) {
    return (m / 16) * (k / 16) * 256 * sizeof(half) <= AscendC::TOTAL_L0A_SIZE &&
           (k / 16) * (n / 16) * 256 * sizeof(half) <= AscendC::TOTAL_L0B_SIZE &&
           static_cast<uint64_t>(m) * n * sizeof(float) <= AscendC::TOTAL_L0C_SIZE &&
           tp_round_up(m, kTpTileAxis) * tp_round_up(n, kTpTileAxis) * sizeof(half) <= kTpUbBytes;
}

extern "C" __global__ __aicore__ void qwen_cube_transpose_probe_kernel(
    GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR c_gm,
    uint32_t m, uint32_t n, uint32_t k, uint32_t transpose, uint32_t reserved) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::GlobalTensor<half> c_g;
    c_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c_gm), m * n);
    if ((m % 16) != 0 || (n % 16) != 0 || (k % 16) != 0 || m == 0 || n == 0 || k == 0 ||
        !tp_l1_fits(m, n, k) || !tp_l0_fits(m, n, k)) {
        c_g.SetValue(0, static_cast<half>(-32768.0f));
        return;
    }
    const uint32_t round_m = tp_round_up(m, kTpTileAxis);
    const uint32_t round_n = tp_round_up(n, kTpTileAxis);
    const uint32_t m_blocks = m / 16;
    const uint32_t n_blocks = n / 16;
    const uint32_t k_blocks = k / 16;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> a1_buf;
    AscendC::TBuf<AscendC::TPosition::B1> b1_buf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> out_buf;
    pipe.InitBuffer(a1_buf, m * k * sizeof(half));
    pipe.InitBuffer(b1_buf, n * k * sizeof(half));
    pipe.InitBuffer(out_buf, round_m * round_n * sizeof(half));

    AscendC::GlobalTensor<half> a_g, b_g;
    a_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a_gm), m * k);
    b_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b_gm), n * k);

    AscendC::LocalTensor<half> a_l1 = a1_buf.Get<half>();
    AscendC::LocalTensor<half> b_l1 = b1_buf.Get<half>();
    AscendC::LocalTensor<half> c_ub = out_buf.Get<half>();

    // A is always staged as the ND [m, k] it is. The fractal for (m-block, k-block)
    // lands at kBlock * m_blocks * 256 elements, which is the order LoadL0A walks.
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

    // The B operand is where the two hypotheses diverge. transpose == 0 stages
    // the natural [n, k] form, which is what AscendC::Gemm already consumes and
    // therefore doubles as a check that the hand-rolled L0A/L0B/Mmad sequence
    // below reproduces Gemm. transpose == 1 stages the natural [k, n] form --
    // the layout the V cache actually has -- and asks the load to transpose.
    AscendC::Nd2NzParams b_params;
    b_params.ndNum = 1;
    b_params.srcNdMatrixStride = 0;
    b_params.dstNzNStride = 1;
    b_params.dstNzMatrixStride = 0;
    if (transpose == 0) {
        b_params.nValue = static_cast<uint16_t>(n);
        b_params.dValue = static_cast<uint16_t>(k);
        b_params.srcDValue = static_cast<uint16_t>(k);
        b_params.dstNzC0Stride = static_cast<uint16_t>(round_n);
    } else {
        b_params.nValue = static_cast<uint16_t>(k);
        b_params.dValue = static_cast<uint16_t>(n);
        b_params.srcDValue = static_cast<uint16_t>(n);
        b_params.dstNzC0Stride = static_cast<uint16_t>(k);
    }
    AscendC::DataCopy(b_l1, b_g, b_params);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::TBuffAddr a_l0a_addr;
    a_l0a_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
    AscendC::LocalTensor<half> a_l0a;
    a_l0a.SetAddr(a_l0a_addr);
    a_l0a.InitBuffer(0, AscendC::TOTAL_L0A_SIZE / sizeof(half));

    AscendC::TBuffAddr b_l0b_addr;
    b_l0b_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
    AscendC::LocalTensor<half> b_l0b;
    b_l0b.SetAddr(b_l0b_addr);
    b_l0b.InitBuffer(0, AscendC::TOTAL_L0B_SIZE / sizeof(half));

    AscendC::TBuffAddr l0c_addr;
    l0c_addr.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
    AscendC::LocalTensor<half> c_l0c;
    c_l0c.SetAddr(l0c_addr);
    c_l0c.InitBuffer(0, AscendC::TOTAL_L0C_SIZE / sizeof(half));

    // L0A: one call per m-block, mirroring Gemm's kBlocks > 1 path with a single
    // m-tile. The source fractals for m-block `index` are index, index + m_blocks,
    // index + 2 * m_blocks, ... -- that is (k-block, m-block) with the k-block
    // outer, which is exactly the order the ND [m, k] staging produced.
    for (uint32_t index = 0; index < m_blocks; ++index) {
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(k_blocks);
        params.srcStride = static_cast<uint16_t>(m_blocks);
        params.ifTranspose = false;
        AscendC::LoadData(a_l0a[index * k_blocks * 256], a_l1[index * 256], params);
    }

    // L0B: the whole operand in one call. Without a transpose the L1 fractals are
    // already in (k-block outer, n-block inner) order, so consecutive works. With
    // a transpose the question is whether the hardware reorders the fractal grid
    // as well as transposing inside each fractal; if it does, the same consecutive
    // walk reproduces the same L0B image from the [k, n] staging.
    {
        AscendC::LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint8_t>(k_blocks * n_blocks);
        params.srcStride = 1;
        params.ifTranspose = transpose != 0;
        AscendC::LoadData(b_l0b, b_l1, params);
    }

    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::MmadParams mmad_params;
    mmad_params.m = static_cast<uint16_t>(m);
    mmad_params.n = static_cast<uint16_t>(n);
    mmad_params.k = static_cast<uint16_t>(k);
    mmad_params.isBias = false;
    mmad_params.cmatrixInitVal = true;
    AscendC::Mmad(c_l0c, a_l0a, b_l0b, mmad_params);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyParams ro_params;
    ro_params.blockCount = 1;
    ro_params.srcStride = 0;
    ro_params.dstStride = 0;
    ro_params.blockLen = static_cast<uint16_t>(round_m * round_n * sizeof(float) / 1024);
    AscendC::DataCopyEnhancedParams ro_enhanced;
    ro_enhanced.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
    AscendC::DataCopy(c_ub, c_l0c, ro_params, ro_enhanced);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Nz2NdParamsFull out_params(1, static_cast<uint16_t>(m), static_cast<uint16_t>(n), 1,
                                        static_cast<uint16_t>(round_m), static_cast<uint16_t>(n), 1);
    AscendC::DataCopy(c_g, c_ub, out_params);
    AscendC::PipeBarrier<PIPE_ALL>();
}
