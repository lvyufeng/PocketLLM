// Geometry of the gated-delta recurrence, shared by the kernel and its launcher.
//
// This header exists because the two need to agree on one number and there is no
// way to check that agreement at compile time across translation units: the device
// kernel derives (head, value slice) from the block index with kSlices, and the
// launcher sizes the grid as heads * slices. If the two copies of that number
// drifted apart the kernel would still launch and still return success -- it would
// just leave part of every head's state unwritten, which reads as a silent accuracy
// bug rather than a failure. Keeping one definition removes the possibility.
//
// Deliberately includes nothing: `kernel_operator.h` cannot be compiled by the host
// compiler and this file has to be visible to both.
//
// No dependency on the AscendC headers means it is also safe for host-side
// benchmarks and tests that want to describe the same shape.

#ifndef POCKET_QWEN_GATED_DELTA_GEOMETRY_HPP
#define POCKET_QWEN_GATED_DELTA_GEOMETRY_HPP

#include <cstdint>

namespace pocket {
namespace gated_delta {

// Qwen3.5's linear-attention head geometry. Both dimensions are 128, and the host
// launcher rejects anything else, so the state tile size, the number of Axpy rows
// and the fold depth are all compile-time constants that depend on them.
constexpr uint32_t kKeyDim = 128;
constexpr uint32_t kValueDim = 128;

// A whole head's state in GM, [kKeyDim, kValueDim].
constexpr uint32_t kStateElems = kKeyDim * kValueDim;

// Value columns per core. 64 is a vector repeat of fp32, so one repeat is exactly
// one state row and the strided ops in the kernel have a natural unit; it also
// divides 128 evenly into the two slices that turn a 12-head TP4 shard into 24
// blocks on a 30-core part.
constexpr uint32_t kValueSlice = 64;
constexpr uint32_t kSlices = kValueDim / kValueSlice;

// The AI cores are 30 on this part and the launcher never asks for more. Held here
// so host-side reporting sizes its grid by the same cap the launcher clamps to.
constexpr uint32_t kMaxBlocks = 30;

}  // namespace gated_delta
}  // namespace pocket

#endif  // POCKET_QWEN_GATED_DELTA_GEOMETRY_HPP
