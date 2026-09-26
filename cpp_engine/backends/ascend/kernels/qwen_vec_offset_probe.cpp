// Does a UB vector instruction honour a source operand that is offset by less
// than a 32-byte block?
//
// `AscendC::Add(dst, src0, src1, count)` hands the three operands to the vector
// unit as raw `__ubuf__` pointers and does no alignment arithmetic of its own, so
// the C++ layer is silent on the question. Every other kernel in this directory
// only ever offsets a UB operand by whole 32-byte blocks -- `fold_sum` stops its
// halving fold at kAlignFloat and finishes the last eight lanes on the scalar
// unit, and the attention row fold does the same -- so none of them answers it.
//
// It matters because the ternary checkpoint's incoherence rotation is a
// Walsh-Hadamard transform over 1024-point runs: ten butterfly passes whose
// pairing stride runs 1, 2, 4, ... 512 *elements*, and the first three of those
// are 4, 8 and 16 bytes in FP32. If the unit rounds the address down, a
// straight-line port of the CUDA kernel computes `2 * src[i]` instead of
// `src[i] + src[i + shift]` -- a different transform, silently, at three of the
// ten levels. If it faults, it faults. Either answer decides the kernel's shape,
// which is why this exists as a kernel rather than as a reading of the docs.
//
// Measured on the 910A (2026-09-26): it faults. Offsets of 1 through 7 elements
// all raise an aicore exception -- `error code = 0x10`, `errorStr: Illegal
// instruction, which is usually caused by unaligned UUB addresses` -- while the
// eight-element offset, one whole block, returns the sum. The three sub-block
// levels are therefore unreachable from the vector unit, and the rotation has to
// reach them on the scalar unit, the one place in UB that addresses an element
// rather than a block.
//
// One core, one array, `dst[i] = src[i] + src[i + shift]` over `count` elements.
// `shift = 8` is the control: that offset is exactly one block and is the one
// every kernel here already relies on. `shift` in 1..7 is the question.

#include "qwen_ascend_kernel_common.hpp"

namespace {

// Two fp32 staging buffers of this many elements each: 32 KiB of the 256 KiB UB,
// which leaves room for the pipeline and keeps the probe independent of every
// other kernel's budget. One extra block is reserved past the input buffer so the
// shifted operand of the last lane is still inside it.
constexpr uint32_t kProbeChunk = 4096;
constexpr uint32_t kProbeSlack = pocket::kAlignFloat;

}  // namespace

extern "C" __global__ __aicore__ void qwen_vec_offset_probe_kernel(
    GM_ADDR source_gm, GM_ADDR dest_gm, uint32_t count, uint32_t shift) {
    // One core. The probe is a question about addressing, not throughput, and a
    // partitioned range would only add a second way for it to be wrong.
    if (AscendC::GetBlockIdx() != 0 || count == 0 || shift == 0 ||
        shift > pocket::kAlignFloat) {
        return;
    }

    AscendC::GlobalTensor<float> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source_gm),
                           static_cast<uint64_t>(count) + kProbeSlack);
    AscendC::GlobalTensor<float> dest;
    dest.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dest_gm),
                         static_cast<uint64_t>(count));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, (kProbeChunk + kProbeSlack) * sizeof(float) +
                            kProbeChunk * sizeof(float));
    AscendC::LocalTensor<float> work = buf.Get<float>();
    AscendC::LocalTensor<float> out = work[kProbeChunk + kProbeSlack];

    for (uint32_t base = 0; base < count; base += kProbeChunk) {
        const uint32_t lanes = pocket::min_u32(kProbeChunk, count - base);
        // The load carries one whole block past the lanes that are computed, so
        // `work[i + shift]` is defined for every `i < lanes` whatever `shift` is
        // -- and so the copy length stays a block multiple, which is all
        // DataCopy can express.
        pocket::wait_compute_before_load();
        AscendC::DataCopy(work, source[base],
                          static_cast<int32_t>(lanes) +
                              static_cast<int32_t>(pocket::kAlignFloat));
        pocket::wait_load_before_compute();
        AscendC::Add(out, work, work[shift], static_cast<int32_t>(lanes));
        pocket::wait_compute_before_store();
        AscendC::DataCopy(dest[base], out, static_cast<int32_t>(lanes));
        // The next iteration's Add writes `out` again, so the store has to drain
        // before Vector, not before another load: `work` and `out` are separate
        // halves of the buffer and the next load cannot touch the store's source.
        pocket::wait_store_before_compute();
    }
}
