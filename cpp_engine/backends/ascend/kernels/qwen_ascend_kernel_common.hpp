// Shared device-side helpers for the hand-written AscendC kernels.
//
// Everything here is first-generation 910 specific (Short_SoC_version=Ascend910:
// 30 AI cores, 32 MB L2, 256 KiB UB, no BF16). Three constraints shape every
// kernel in this directory and are worth stating once:
//
//   1. TBuf inserts no synchronization. Every hand-off between hardware pipes has
//      to be written out. Omitting one does not crash: the kernel returns success
//      with numbers that are simply wrong, because Vector read UB before MTE2's
//      copy landed.
//   2. There are no scalar transcendentals. `expf`, `cosf`, `rsqrtf` and friends
//      exist in the CANN headers but are gated behind the SIMT API and are not
//      visible from a classic __aicore__ kernel. Scalar `sqrt`, `+ - * /` are, and
//      they are fully precise. Anything else has to run as a vector op over a tile,
//      even when only one value is wanted.
//   3. Vector `Reciprocal` and `Rsqrt` are ~2e-3 hardware approximations, not FP32.
//      Where a reciprocal has to be exact, take it as a scalar division instead.
//
// FP32 block/pair reductions are unavailable: their generic paths instantiate
// vcgadd/vcpadd, which this SoC supports for float16 only. FP32 WholeReduceSum
// lowers to vcadd and is supported (`Intrinsic_vcadd|float16,float32` in
// Ascend910B.ini); use it only after arranging each logical row as one repeat.
// The halving folds below use nothing but vadd.

#ifndef POCKET_QWEN_ASCEND_KERNEL_COMMON_HPP
#define POCKET_QWEN_ASCEND_KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace pocket {

// Vector ops want 32-byte alignment: 16 halfs or 8 floats.
constexpr uint32_t kAlignHalf = 16;
constexpr uint32_t kAlignFloat = 8;

// Bytes per 32-byte DataCopy block, the unit DataCopyParams counts in.
constexpr uint32_t kBlockBytes = 32;

__aicore__ inline uint32_t pocket_align_up(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

__aicore__ inline uint32_t align_down_u32(uint32_t value, uint32_t multiple) {
    return value / multiple * multiple;
}

__aicore__ inline uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Pipe hand-offs. Named for what they guard rather than for the event, because the
// event names alone read as noise at the call site.
__aicore__ inline void wait_load_before_compute() {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

__aicore__ inline void wait_compute_before_scalar() {
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
}

__aicore__ inline void wait_scalar_before_compute() {
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
}

__aicore__ inline void wait_compute_before_store() {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void wait_scalar_before_store() {
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
}

// A store has to land before the same buffer is refilled on the next iteration.
__aicore__ inline void wait_store_before_load() {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// A store has to drain before Vector reuses the same UB. This differs from the
// MTE3->MTE2 hand-off above: the next producer is Vector, not another load.
__aicore__ inline void wait_store_before_compute() {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}

// Same lifetime rule when the scalar unit is the next user of the stored tile.
__aicore__ inline void wait_store_before_scalar() {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

// A load has to land before a scalar unit reads the same UB.
__aicore__ inline void wait_load_before_scalar() {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
}

// A scalar read has to finish before a load refills the same UB. This is the
// counterpart to wait_load_before_scalar for a tile the scalar unit consumes
// directly, where wait_store_before_load would name a store that never happened.
__aicore__ inline void wait_scalar_before_load() {
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
}

// A Vector read has to finish before a load refills the same UB.
//
// This is the anti-dependency of wait_load_before_compute and it is easy to miss: a
// staging tile that is loaded, consumed by Vector, then loaded again needs a hand-off
// in *both* directions. MTE2 is free the instant its previous copy retires, so
// without this the next copy lands on top of a tile the Vector pipe is still reading,
// and the consumer silently sees the following iteration's data.
__aicore__ inline void wait_compute_before_load() {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
}

// Exact-length fp16 transfers.
//
// DataCopy moves whole 32-byte blocks, so a tile whose length is not a multiple of
// 16 halfs cannot be moved by DataCopy alone without reading or writing past the
// payload. Rounding the length up is what the first draft of these kernels did, and
// it is an out-of-bounds access on both directions at the end of a plane.
// DataCopyPad would be the answer on a second-generation part, but every GM<->UB
// overload of it reports "not support" on dav_c100. So the tail moves on the scalar
// unit: at most 15 elements, once per tile.
//
// Both helpers own their pipe hand-offs. Callers must not add their own
// wait_load_before_compute / wait_compute_before_store around them, or the flag
// pairs will nest and deadlock.

__aicore__ inline void load_half_exact(const AscendC::LocalTensor<half>& dst,
                                       const AscendC::GlobalTensor<half>& src,
                                       uint32_t offset, uint32_t count) {
    const uint32_t bulk = align_down_u32(count, kAlignHalf);
    if (bulk > 0) {
        AscendC::DataCopy(dst, src[offset], bulk);
    }
    for (uint32_t i = bulk; i < count; ++i) {
        dst.SetValue(i, src.GetValue(offset + i));
    }
    if (bulk > 0) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    if (bulk < count) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
}

__aicore__ inline void store_half_exact(AscendC::GlobalTensor<half>& dst,
                                        uint32_t offset,
                                        const AscendC::LocalTensor<half>& src,
                                        uint32_t count) {
    const uint32_t bulk = align_down_u32(count, kAlignHalf);
    if (bulk < count) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = bulk; i < count; ++i) {
            dst.SetValue(offset + i, src.GetValue(i));
        }
    }
    if (bulk > 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dst[offset], src, bulk);
    }
}

// Sum of `width` contiguous floats, where `width` is a power of two and at least
// kAlignFloat. Folds the buffer in half repeatedly with vadd, then finishes the
// last 8 lanes on the scalar unit. Destroys `work`.
//
// Costs log2(width/8) vector instructions: 4 for a 128-wide key row.
__aicore__ inline float fold_sum(const AscendC::LocalTensor<float>& work,
                                 uint32_t width) {
    for (uint32_t half = width / 2; half >= kAlignFloat; half /= 2) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(work, work, work[half], half);
    }
    wait_compute_before_scalar();
    float sum = 0.0f;
    const uint32_t tail = min_u32(width, kAlignFloat);
    for (uint32_t i = 0; i < tail; ++i) sum += work.GetValue(i);
    return sum;
}

// Column-wise sum of a [rows, width] tile, leaving the result in the first `width`
// lanes. `rows` must be a power of two; `width` a multiple of kAlignFloat. Same
// halving fold as fold_sum, but folding whole rows so all `width` columns reduce at
// once. Destroys `work`.
__aicore__ inline void fold_rows(const AscendC::LocalTensor<float>& work,
                                 uint32_t rows, uint32_t width) {
    for (uint32_t half = rows / 2; half >= 1; half /= 2) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(work, work, work[half * width], half * width);
    }
}

// exp() of a single scalar. There is no scalar exp on this part, so this runs the
// vector unit over one 8-lane tile. Used where a per-token gate is needed and
// batching the whole sequence is not worth the buffer.
__aicore__ inline float scalar_exp(const AscendC::LocalTensor<float>& work,
                                   float value) {
    wait_scalar_before_compute();
    AscendC::Duplicate(work, value, kAlignFloat);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(work, work, kAlignFloat);
    wait_compute_before_scalar();
    return work.GetValue(0);
}

// Broadcast `src[i]` across a row of `width` lanes, for i in [0, rows).
//
// This is the shape a scalar-per-key-row multiply needs, and it is the hot spot of
// the recurrence: one Duplicate per row. The dsts do not overlap, so no barrier is
// needed between them, but it is still `rows` instruction issues. Brcb would reduce
// that issue count, but it is an unsupported stub on first-generation dav_c100, so
// the scalar broadcast loop is intentional for this target.
__aicore__ inline void broadcast_rows(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& src,
                                      uint32_t rows, uint32_t width) {
    for (uint32_t i = 0; i < rows; ++i) {
        AscendC::Duplicate(dst[i * width], src.GetValue(i), width);
    }
}

// Copy a [rows, width] column window out of a [rows, stride] GM tile.
//
// DataCopyParams counts in 32-byte blocks, so this needs width*sizeof(T) and
// (stride-width)*sizeof(T) to both be block multiples. Callers validate that on
// the host rather than silently truncating here.
template <typename T>
__aicore__ inline AscendC::DataCopyParams window_params(uint32_t rows,
                                                        uint32_t width,
                                                        uint32_t stride) {
    AscendC::DataCopyParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint16_t>(width * sizeof(T) / kBlockBytes);
    params.srcStride = static_cast<uint16_t>((stride - width) * sizeof(T) / kBlockBytes);
    params.dstStride = 0;
    return params;
}

}  // namespace pocket

#endif  // POCKET_QWEN_ASCEND_KERNEL_COMMON_HPP
