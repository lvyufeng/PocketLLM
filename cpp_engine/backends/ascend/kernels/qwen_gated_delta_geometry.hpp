// Geometry of the gated-delta recurrence, shared by the kernel, its launcher and
// the benchmark that prices it.
//
// This header exists because the three need to agree on one number and there is no
// way to check that agreement at compile time across translation units. The device
// kernel derives (head, value range) from the block index with the value split, and
// the launcher sizes the grid as `heads * split`. If the two copies of that number
// drifted apart the kernel would still launch and still return success -- it would
// just leave part of every head's state unwritten, which reads as a silent accuracy
// bug rather than as a failure. Keeping one definition removes the possibility.
//
// Deliberately includes nothing but <cstdint>: `kernel_operator.h` cannot be
// compiled by the host compiler and this file has to be visible to both.

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

// Widest value-axis cut the kernel is instantiated for.
//
// The state is separable down the value axis -- column j of S evolves without
// reference to column j' -- so a head can be cut into independent column ranges
// with no cross-core traffic. The cut is worth taking and this is roughly how
// much: holding the key rows fixed at 128 and halving the value width took the
// operator from 41.3 to 36.0 ms/layer, while halving the key rows instead took it
// to 22.6. The row count is the instruction count and an `Axpy` per key row is the
// shape of all three passes over the state, so a cut that leaves the rows alone
// halves every instruction's repeat count and nothing else. That this part charges
// far more for the instruction than for the bytes it moves is what makes the cut
// worth 13% rather than 50%.
constexpr uint32_t kMaxValueSplit = 2;

// The AI cores are 30 on this part. Defined here rather than in the launcher
// because the value split is the one place a kernel's item count and the
// launcher's grid size are two derivations of the same number.
constexpr uint32_t kMaxBlocks = 30;

/// The value-axis cut to run, given how many heads this rank owns and how many
/// cores it has to put them on.
///
/// Two, only when the doubled item count still fits in one pass over the cores.
/// A split halves the work of an item but not its instruction count, so it pays
/// only while the extra items are free; once they are not, it buys a whole extra
/// pass and loses badly. At TP4 the shard is 12 heads, so 24 items fit on the 30
/// cores and the operator drops 13% (41.3 -> 36.0 ms/layer at 4433 tokens). At TP2
/// the shard is 24 heads, so 48 items need two passes and the same cut takes the
/// operator from 41 to 72 ms. Hence a rule and not a constant: the head count is
/// what decides it, and the head count is not known until the call.
constexpr uint32_t value_split_for(uint32_t heads, uint32_t cores) {
    return heads * kMaxValueSplit <= cores ? kMaxValueSplit : 1u;
}

}  // namespace gated_delta
}  // namespace pocket

#endif  // POCKET_QWEN_GATED_DELTA_GEOMETRY_HPP
