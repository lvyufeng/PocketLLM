// HBM read-rate probe: stream `tile_count` 64x256 FP16 tiles and touch every
// element once, so the achievable read rate can be measured directly instead of
// inferred from a GEMV.
//
// The number matters because the decode target is a bandwidth question before it
// is a kernel question: a decode step re-reads every resident weight, so
// (weight bytes) / (read rate) is a floor no amount of kernel work goes below.
// The two instruments that existed before this answered it badly.
//
//   - `memcpy_d2d` is not a memory probe on this stack. A 512 MiB device-to-
//     device copy runs at 8.7 GB/s, which is PCIe-shaped and 40x under what the
//     GEMV already reaches, so a copy rate says nothing about HBM either way.
//   - A one-row GEMV mixes the kernel's own efficiency into the figure. Its
//     plateau across a 256x weight-size range rules out a per-call launch cost,
//     but not a per-core load-issue limit that would also be flat.
//
// This measures the memory system alone: MTE2 streams the tiles, no vector or
// Cube op ever runs, and the only consumer is one scalar read per batch, which
// exists to keep the copies alive rather than to compute anything.
//
// It is a lower bound by construction. The staging buffer holds six tiles and is
// reused, so each batch pays one MTE2->S hand-off and the S->MTE2 anti-dependency
// that lets the next batch refill it. Those two round trips are the only bubbles
// in a loop that is otherwise a back-to-back MTE2 queue, and they are charged
// against the transfer.
//
// First-generation 910 only (30 AI cores, 256 KiB UB, FP16).

#include "qwen_ascend_kernel_common.hpp"
#include "qwen_hbm_probe_geometry.hpp"

namespace {

// Six tiles = 192 KiB staged at once, under the 256 KiB UB with room for the
// pipe. Larger batches amortise the two hand-offs better; this is the largest
// that is still safe to allocate.
constexpr uint32_t kHbmProbeBatchTiles = 6;
constexpr uint32_t kHbmProbeUbBytes =
    pocket::kHbmProbeTileElems * kHbmProbeBatchTiles * sizeof(half);

// The elements summed per tile. The first and last bracket the transfer, so a
// DataCopy that moved less than the whole tile leaves the tail stale and the
// check in bench_qwen_ascend_bandwidth sees a shortfall.
__aicore__ inline uint32_t sample_index(uint32_t i) {
    constexpr uint32_t k = pocket::kHbmProbeTileElems;
    return i == 0 ? 0u : (i == 1 ? k / 3u : (i == 2 ? (2u * k) / 3u : k - 1u));
}

}  // namespace

extern "C" __global__ __aicore__ void qwen_hbm_read_probe_kernel(
    GM_ADDR source_gm, GM_ADDR sink_gm, uint32_t tile_count) {
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t per_core =
        (tile_count + pocket::kHbmProbeCores - 1u) / pocket::kHbmProbeCores;
    const uint32_t first = block * per_core;
    if (first >= tile_count) {
        return;
    }
    const uint32_t last = pocket::min_u32(first + per_core, tile_count);

    AscendC::GlobalTensor<half> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(source_gm),
                           static_cast<uint64_t>(tile_count) *
                               pocket::kHbmProbeTileElems);
    AscendC::GlobalTensor<half> sink;
    sink.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sink_gm),
                         pocket::kHbmProbeCores);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stage;
    pipe.InitBuffer(stage, kHbmProbeUbBytes);
    AscendC::LocalTensor<half> buf = stage.Get<half>();

    // Summing the per-tile first elements into one value keeps the whole read
    // live without a vector op: a scalar read of `buf` is ordered against the
    // copies by the MTE2->S flag, and the result is carried to the sink, so no
    // copy can retire unused.
    float carried = 0.0f;
    for (uint32_t tile = first; tile < last; tile += kHbmProbeBatchTiles) {
        const uint32_t live = pocket::min_u32(kHbmProbeBatchTiles, last - tile);
        for (uint32_t i = 0; i < live; ++i) {
            AscendC::DataCopy(buf[i * pocket::kHbmProbeTileElems],
                              source[static_cast<uint64_t>(tile + i) *
                                     pocket::kHbmProbeTileElems],
                              pocket::kHbmProbeTileElems);
        }
        pocket::wait_load_before_scalar();
        for (uint32_t i = 0; i < live; ++i) {
            const AscendC::LocalTensor<half> tilebuf =
                buf[i * pocket::kHbmProbeTileElems];
            for (uint32_t s = 0; s < pocket::kHbmProbeSamples; ++s) {
                carried += static_cast<float>(tilebuf.GetValue(sample_index(s)));
            }
        }
        // The next batch refills the same staging buffer, so the scalar reads
        // above have to finish first.
        pocket::wait_scalar_before_load();
    }

    sink.SetValue(block, static_cast<half>(carried));
}
