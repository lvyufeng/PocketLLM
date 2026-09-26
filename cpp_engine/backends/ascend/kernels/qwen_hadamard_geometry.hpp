// Block-size bounds for the incoherence rotation, shared by the device kernel and
// the host launcher.
//
// The kernel sizes its UB tiles at `kHadamardMaxBlock` elements and addresses them
// with a runtime `block`, because the vector unit has no dynamic allocation and
// `TPipe::InitBuffer` wants its size at compile time. The launcher is the only
// thing that can refuse a larger one. A drift between the two does not fail: the
// copy lands past the end of a tile, into the next one, and the rotation returns
// plausible numbers computed from someone else's data.
//
// Host and device both include this, so it must stay free of `kernel_operator.h`
// and of anything else that only compiles on one side.

#ifndef POCKET_QWEN_HADAMARD_GEOMETRY_HPP
#define POCKET_QWEN_HADAMARD_GEOMETRY_HPP

#include <cstdint>

namespace pocket {

// Five tiles of this many elements -- three fp32 for the signs and the two
// ping-pong butterfly tiles, two fp16 for the load and the store -- is 64 KiB of
// the 256 KiB UB, which leaves the pipeline room without asking the allocator for
// anything clever. The checkpoint's block is 1024; this is the budget, not a
// property of any checkpoint.
constexpr uint32_t kHadamardMaxBlock = 4096;

// The floor is the fp16 DataCopy granularity, not a mathematical one: 16 halfs is
// exactly one 32-byte block, so a shorter run would have the hardware move a whole
// block anyway -- reading past the row it was given and writing past the row it was
// asked for. The butterfly itself would be correct down to 8, and the fp32 tiles
// only need 8; it is the fp16 copy that sets this.
//
// A power of two at or above it completes the transform: the scalar unit runs the
// three levels below 8 elements as a WHT-8 over each aligned group of eight, and
// the first block level pairs groups, so any power of two from 16 up is closed.
constexpr uint32_t kHadamardMinBlock = 16;

}  // namespace pocket

#endif  // POCKET_QWEN_HADAMARD_GEOMETRY_HPP
