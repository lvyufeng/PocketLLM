// Grid and tile geometry for the HBM read probe, shared by the device kernel and
// the host launcher.
//
// Without this the core count would be written twice: once as the divisor the
// kernel uses to hand each core a contiguous span of tiles, and once as the
// block dimension the launcher passes to `aclrtlaunch_*`. A drift between them
// does not fail — it silently duplicates or drops tiles, which reads as a
// bandwidth number that is merely a little off, exactly the failure mode the
// probe exists to rule out.
//
// Host and device both include this, so it must stay free of `kernel_operator.h`
// and of anything else that only compiles on one side.

#ifndef POCKET_QWEN_HBM_PROBE_GEOMETRY_HPP
#define POCKET_QWEN_HBM_PROBE_GEOMETRY_HPP

#include <cstdint>

namespace pocket {

// Every AI core on a first-generation 910, so the figure is the whole part's
// read rate rather than one core's scaled up.
constexpr uint32_t kHbmProbeCores = 30;

// 64 x 256 FP16 = 32 KiB per tile. Wide enough that one DataCopy is a real burst
// rather than a measurement of per-call overhead.
constexpr uint32_t kHbmProbeTileElems = 64u * 256u;

// How many elements of each tile the kernel sums. Sampling is what keeps the read
// free of Vector and Cube work, but it also has to be enough to show the copy
// actually moved the tile: the first and last element bracket the transfer, so a
// short DataCopy leaves the tail holding stale UB and the check in
// bench_qwen_ascend_bandwidth sees it. The two interior points guard the middle.
constexpr uint32_t kHbmProbeSamples = 4;

}  // namespace pocket

#endif  // POCKET_QWEN_HBM_PROBE_GEOMETRY_HPP
