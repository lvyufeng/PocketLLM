// Stamp layout of the hand-written cross-process all-reduce, shared by the
// device-side arrival wait and the host collective that issues it.
//
// The two have to agree on where a peer's stamp for a round sits and what value it
// carries. Neither is checkable at compile time across translation units, and a
// drift does not fail: the wait would spin until its bound on every call and the
// step would read as uniformly slow, with nothing to point at. So the constants
// live here once and `collective/ipc_allreduce.cpp` takes its own from this header
// rather than restating them.
//
// Deliberately includes nothing but <cstdint>: `kernel_operator.h` cannot be
// compiled by the host compiler and this file has to be visible to both.

#ifndef POCKET_QWEN_IPC_ARRIVE_GEOMETRY_HPP
#define POCKET_QWEN_IPC_ARRIVE_GEOMETRY_HPP

#include <cstdint>

namespace pocket {

// Buffer sets the payload and the stamps alternate between, so a peer that
// reaches round N+1 cannot overwrite the slot this rank is still reading for
// round N. Two is the whole requirement; see ipc_allreduce.hpp for why a peer can
// never lead by more than one round.
constexpr int kIpcSets = 2;

// Rank-to-rank spacing of one stamp, in uint16 words. 32 words is 64 bytes, so no
// two ranks' stamps share a 32-byte DataCopy block, let alone a cache line: the
// hand-written store path on this part loses writes when two cores share one.
constexpr int kIpcStampStride = 32;

// Rounds the stamp table holds before it wraps. The table is indexed by round and
// the payload alternates by parity, so the wrap only has to be longer than any
// window in which a stale stamp could still be outstanding.
constexpr long long kIpcStampTable = 4096;

// Ranks this layout addresses. The stamp array is `world * kIpcStampStride` words
// and the wait kernel is instantiated for any world up to this.
constexpr int kIpcMaxWorld = 8;

// The stamp rank `who` sends at round `round`, which is also the value every peer
// expects from it. Odd, so it can never be mistaken for the zeroed slot a rank
// starts with, and distinct per rank, so a slot that received the wrong rank's
// data fails the check instead of passing it.
//
// A macro, not a function, and not by taste. AscendC marks every function a kernel
// calls with `__aicore__`, which `sys_macros.h` defines as the token `[aicore]`
// that only the device compiler understands; the host compiler has no such
// spelling and would choke on it. So a single *function* definition cannot be read
// by both compilers, while a single expression can. The arguments are each used
// once and fully parenthesised, which is the only hazard a macro adds.
//
// `world` is the width the stamp was computed for, and the peer checks against the
// width it knows. They have to agree, which they do: a rank and its peers are the
// same group.
#define POCKET_IPC_STAMP_VALUE(round, who, world)                                 \
    (static_cast<uint16_t>(                                                       \
        (((((round) % pocket::kIpcStampTable) * (world)) + (who)) & 0x7fff) * 2 + \
        1))

// Spin iterations the device-side wait is allowed per 1000 ms of the host
// deadline, so `limit` can be derived from
// `POCKET_ASCEND_IPC_ALLREDUCE_DEADLINE_MS` instead of being a second deadline
// nobody would remember to change. An iteration is a handful of scalar GM loads
// and lands well under 5 us, so this bound is generous by a large factor; it
// exists to make a lost peer a reported failure rather than a hung device, which
// only needs it to be finite.
constexpr uint32_t kIpcArriveIterationsPerMs = 200;

}  // namespace pocket

#endif  // POCKET_QWEN_IPC_ARRIVE_GEOMETRY_HPP
