// The activation side of the ternary checkpoint's incoherence transform, on the
// 910B.
//
// The weights in the file are in a rotated frame -- each matrix was multiplied by
// `R^-1` before it was quantized to three values, which is what spreads the
// quantization error instead of concentrating it -- so the activation that meets
// them has to be rotated into the same frame. `R = (1/sqrt(N)) H_N diag(s)` with
// `H_N[i][j] = (-1)^popcount(i AND j)` the natural-order Sylvester Hadamard
// matrix, applied independently to each `block` run of the last axis.
//
// Two directions, and the checkpoint says which tensor takes which:
//
//   forward  `x |-> (1/sqrt(N)) H (s * x)`  signs, then the butterflies
//   inverse  `z |-> (1/sqrt(N)) s * (H z)`  the butterflies, then the signs
//
// The scale multiplies the *input* rather than the output because that is the
// fork's own order (`dst = src * scale`, then the passes) and it is the reason the
// two agree bit for bit rather than nearly: scaling after a ten-deep butterfly
// tree rounds ten times where this rounds once.
//
// The accumulation is fp32 and only the result is narrowed, which is the one place
// the arithmetic can afford to be wide: the rounding error of a ten-pass butterfly
// in fp16 would land in exactly the part of the activation that the 1.75-bit
// weights cannot afford.
//
// ## The three levels the vector unit cannot reach
//
// A Walsh-Hadamard transform over `block` elements is `log2(block)` butterfly
// passes whose pairing stride runs 1, 2, 4, ... `block/2` *elements*, and the first
// three of those are 4, 8 and 16 bytes in FP32 -- below the 32-byte block every UB
// vector operand is addressed in. This is not a rounding question. A vector
// instruction handed a source operand offset by 1..7 elements takes an aicore
// exception and the launch is lost (`error code = 0x10`, "Illegal instruction,
// which is usually caused by unaligned UUB addresses"), measured on this part;
// whole-block offsets are the only ones it has. `qwen_vec_offset_probe.cpp` is
// that measurement and `tests/test_qwen_ascend_vec_offset.cpp` is the record.
//
// So the ten levels split. Strides 8, 16, ... `block/2` are offset-tensor `Add`
// and `Sub` with an explicit count, the same form `fold_sum` folds a row with, and
// strides 1, 2 and 4 are computed on the scalar unit one 8-element group at a
// time. The scalar unit is the only place in UB that addresses an element rather
// than a block, and three levels is what it has to carry: WHT-8 needs 24 additions
// and the standard network spends them on 12 butterflies over the group, while the
// seven remaining levels cost one instruction per pair of runs.
//
// The cost of that is real, and it is this kernel's known weakness: 128 groups of
// eight per 1024-element run, each a load of eight, twelve butterflies and a store
// of eight. The width is the *columns* of a folded projection and the rotation is
// memoised per layer, so it is bounded rather than open-ended -- but it is not
// free, and the measurement belongs beside the kernel before this is called done.
//
// ## Shape
//
// One AI core per (row, block-run) pair, handed out grid-stride so a one-row
// projection still spreads over all 30 cores. `block` elements live in UB at a
// time: the fp16 source, the fp32 signs, and two fp32 tiles that the butterfly
// levels ping-pong between. Those buffers are sized at pocket::kHadamardMaxBlock,
// because the vector unit has no dynamic allocation and `pipe.InitBuffer` wants
// its size at compile time, so the launcher refuses a `block` above it. The
// checkpoint's `block` is 1024.
//
// `x` and `y` may be the same buffer: one tile is loaded whole before anything is
// written, and a tile's write range is exactly its read range, so no core can see
// another's output.

#include "qwen_ascend_kernel_common.hpp"

// The tile size and the block bounds, shared with the host launcher so the size it
// admits and the size allocated here cannot drift apart. See the header.
#include "qwen_hadamard_geometry.hpp"

namespace {

// A butterfly level whose pairing stride is at least one whole 32-byte block.
// `dst` and `src` are different tiles, so the two instructions of a run both read
// the pair they were given and neither depends on the other's write.
__aicore__ inline void butterfly_block_level(const AscendC::LocalTensor<float>& dst,
                                             const AscendC::LocalTensor<float>& src,
                                             uint32_t block, uint32_t step) {
    for (uint32_t base = 0; base < block; base += 2 * step) {
        AscendC::Add(dst[base], src[base], src[base + step], step);
        AscendC::Sub(dst[base + step], src[base], src[base + step], step);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// The three levels whose pairing stride is under one block: 1, 2 and 4 elements in
// FP32, all of them inside a single 8-element group. They run on the scalar unit,
// group by group, with the group held in registers between the loads and the
// stores -- three passes over the whole run here would be three times the UB
// traffic for the same twelve butterflies.
//
// This is the merge WHT-8 makes possible and no other level does: strides 1, 2 and
// 4 pair elements that share an aligned group of eight, so a group can be finished
// once it is in registers, whereas stride 8 reaches across groups and has to go
// back to memory. The group is `kAlignFloat` because that is also the operand
// granularity of the vector unit, so nothing here depends on a value the rest of
// the file does not already use.
__aicore__ inline void butterfly_sub_block(const AscendC::LocalTensor<float>& dst,
                                           const AscendC::LocalTensor<float>& src,
                                           uint32_t block) {
    pocket::wait_compute_before_scalar();
    for (uint32_t base = 0; base < block; base += pocket::kAlignFloat) {
        float lane[pocket::kAlignFloat];
        for (uint32_t i = 0; i < pocket::kAlignFloat; ++i) {
            lane[i] = src.GetValue(base + i);
        }
        // `base` walks the groups of `2 * step` and `j` walks inside one, which is
        // what makes level `step` reach every pair whose lower index has that bit
        // clear -- pairs (0,2) and (4,6) *and* (1,3) and (5,7) at step 2. Walking
        // `i` by `2 * step` alone would pair only the aligned ones and quietly
        // return a linear map that is not a Hadamard transform at all.
        for (uint32_t step = 1; step < pocket::kAlignFloat; step <<= 1) {
            for (uint32_t base = 0; base < pocket::kAlignFloat; base += 2 * step) {
                for (uint32_t j = 0; j < step; ++j) {
                    const float low = lane[base + j];
                    const float high = lane[base + step + j];
                    lane[base + j] = low + high;
                    lane[base + step + j] = low - high;
                }
            }
        }
        for (uint32_t i = 0; i < pocket::kAlignFloat; ++i) {
            dst.SetValue(base + i, lane[i]);
        }
    }
    pocket::wait_scalar_before_compute();
}

// All ten levels, from `a` into whichever of the two tiles ends up holding the
// result. The destination alternates, so the caller cannot name the buffer it
// wants and gets the tensor back instead.
__aicore__ inline AscendC::LocalTensor<float> butterfly(
    const AscendC::LocalTensor<float>& a, const AscendC::LocalTensor<float>& b,
    uint32_t block) {
    butterfly_sub_block(b, a, block);
    bool in_a = false;
    for (uint32_t step = pocket::kAlignFloat; step < block; step <<= 1) {
        if (in_a) {
            butterfly_block_level(b, a, block, step);
            in_a = false;
        } else {
            butterfly_block_level(a, b, block, step);
            in_a = true;
        }
    }
    return in_a ? a : b;
}

template <bool kForward>
__aicore__ inline void hadamard_body(GM_ADDR x_gm, GM_ADDR signs_gm, GM_ADDR y_gm,
                                     uint32_t rows, uint32_t width, uint32_t block,
                                     float scale) {
    if (AscendC::GetBlockIdx() >= AscendC::GetBlockNum()) {
        return;
    }
    // The host launcher rejects all of these too. They are repeated because a
    // kernel that indexes GM with unsigned arithmetic has no bounds check to fall
    // back on, and a launch is cheaper to lose than a core is to fault.
    if (rows == 0 || width == 0 || block < pocket::kHadamardMinBlock ||
        block > pocket::kHadamardMaxBlock || (width % block) != 0 ||
        (block & (block - 1)) != 0) {
        return;
    }

    const uint32_t runs = width / block;
    const uint32_t total_runs = rows * runs;

    AscendC::GlobalTensor<half> x_g;
    AscendC::GlobalTensor<half> y_g;
    AscendC::GlobalTensor<float> sign_g;
    x_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x_gm),
                        static_cast<uint64_t>(rows) * width);
    y_g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y_gm),
                        static_cast<uint64_t>(rows) * width);
    // The sign vector covers the whole feature axis and one `block` of it is read
    // per run, so it is indexed by the run rather than by the row.
    sign_g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(signs_gm), width);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sign_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> a_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_buf;
    pipe.InitBuffer(sign_buf, pocket::kHadamardMaxBlock * sizeof(float));
    pipe.InitBuffer(a_buf, pocket::kHadamardMaxBlock * sizeof(float));
    pipe.InitBuffer(b_buf, pocket::kHadamardMaxBlock * sizeof(float));
    pipe.InitBuffer(x_buf, pocket::kHadamardMaxBlock * sizeof(half));
    pipe.InitBuffer(y_buf, pocket::kHadamardMaxBlock * sizeof(half));

    AscendC::LocalTensor<float> sign = sign_buf.Get<float>();
    AscendC::LocalTensor<float> a = a_buf.Get<float>();
    AscendC::LocalTensor<float> b = b_buf.Get<float>();
    AscendC::LocalTensor<half> x = x_buf.Get<half>();
    AscendC::LocalTensor<half> y = y_buf.Get<half>();

    for (uint32_t run_index = AscendC::GetBlockIdx(); run_index < total_runs;
         run_index += AscendC::GetBlockNum()) {
        const uint32_t row = run_index / runs;
        const uint32_t run = run_index - row * runs;
        const uint32_t offset = row * width + run * block;
        const uint32_t sign_offset = run * block;

        // `x` and `sign` are refilled here and the previous run's Vector pass was
        // their last reader. MTE2 is free the instant its own copy retires, so
        // without this the next tile lands under a butterfly that is still
        // reading the last one.
        pocket::wait_compute_before_load();
        AscendC::DataCopy(x, x_g[offset], block);
        AscendC::DataCopy(sign, sign_g[sign_offset], block);
        pocket::wait_load_before_compute();

        // The signs and the scale are elementwise and commute -- a sign is +-1 and
        // so exact -- so folding them into one pass reproduces the reference's two
        // multiplies exactly, and in the reference's order.
        AscendC::Cast(a, x, AscendC::RoundMode::CAST_NONE, block);
        AscendC::PipeBarrier<PIPE_V>();
        if (kForward) {
            AscendC::Mul(a, a, sign, block);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Muls(a, a, scale, block);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> result = butterfly(a, b, block);

        if (!kForward) {
            AscendC::Mul(result, result, sign, block);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, block);
        pocket::wait_compute_before_store();
        AscendC::DataCopy(y_g[offset], y, block);
        // `y` is rewritten by the next run's Cast, which is Vector, not MTE2.
        pocket::wait_store_before_compute();
    }
}

}  // namespace

// The two directions. Both are file-scope entry points because each one compiles
// to its own module binary, and the direction is a template argument rather than a
// runtime flag so the untaken half is dropped before codegen.

extern "C" __global__ __aicore__ void qwen_hadamard_forward_f16_kernel(
    GM_ADDR x_gm, GM_ADDR signs_gm, GM_ADDR y_gm, uint32_t rows, uint32_t width,
    uint32_t block, float scale) {
    hadamard_body<true>(x_gm, signs_gm, y_gm, rows, width, block, scale);
}

extern "C" __global__ __aicore__ void qwen_hadamard_inverse_f16_kernel(
    GM_ADDR x_gm, GM_ADDR signs_gm, GM_ADDR y_gm, uint32_t rows, uint32_t width,
    uint32_t block, float scale) {
    hadamard_body<false>(x_gm, signs_gm, y_gm, rows, width, block, scale);
}
