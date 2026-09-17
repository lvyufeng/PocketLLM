// Device-side arrival wait for the hand-written cross-process all-reduce: one
// kernel that spins on the peer stamps in GM until every one of them carries this
// round's value, so the *device* pays the rendezvous and the host never blocks.
//
// The collective it belongs to polls the same stamps from the host, with a
// blocking `aclrtMemcpy` whose drain of the default stream is what makes the read
// succeed on its first iteration. That drain is not free the other way round: the
// host is inside the runtime for the whole wait, so it cannot enqueue anything
// behind it, and the device runs dry. Measured at 129 collectives a decode step,
// deleting the host wait altogether -- `POCKET_ASCEND_IPC_ALLREDUCE_NOPOLL`, which
// is wrong by construction and exists only to price this -- takes the step from
// 76.9 ms to 51.9 ms. That 25 ms is host time the device could have been given
// work across, so the same wait moved onto the device should recover it, provided
// the peers do arrive while the device is still busy with the layer before.
//
// Three things were meant to make it safe rather than merely faster, and the
// measurement says one of them does not hold.
//
//   * It waits for exactly the condition the host poll waits for -- every peer's
//     stamp for this round -- and that part is right. Across ten interleaved pairs
//     it never hit its iteration bound and never had to write a failure status, so
//     the stamps arrive when the host loop would have seen them too.
//   * It is bounded. `limit` iterations and then it gives up and reports the
//     pending count, so a lost peer becomes a nonzero status the host raises
//     rather than a device that spins forever.
//   * It runs on the caller's stream, in front of the pushes and the reduce that
//     the host already enqueued behind it, so stream order is the only ordering
//     this needs. The protocol that keeps a peer from overwriting a slot being
//     read survives the move: a rank's push for round N+1 is issued after its wait
//     for round N+1 on the same stream, and that wait cannot complete before the
//     peer's round N push, which is issued after the peer's round N wait. The
//     device executing in order is what enforces it, not the host.
//
// What does not hold is the first one's conclusion: waiting for the stamp is not
// waiting for the plane. A peer's stamp can be readable here while the plane it
// announces is not yet readable by the reduce behind this kernel, and the host round
// trip this kernel exists to delete was -- without ever being designed as one -- the
// window that covered it. Delete the round trip and the cover goes with it. Ten
// interleaved pairs fail the launcher's reproducibility gate ten times in ten
// against the host poll's zero in ten, and putting the delay back on the host
// (`POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US`) puts the failure rate on a dose-response:
// 10 of 10 runs at 0 us, 2 of 4 at 20, 1 of 4 at 100, 0 of 4 at 500 -- with the 20 us
// arm still running *above* the shipped step time, so mere slowness does not explain
// it. That is why this kernel is a bound on what the host round trip costs and not a
// candidate: it takes the step from 77.117 to 53.745 ms and the decode from 12.968 to
// 18.607 TPS, and the token it produces is not the shipped path's token.
//
// The shipped host-poll path has the same defect at a lower rate -- it fails the same
// gate 8 times in 36 -- so this is not a new bug introduced here. It is that one with
// the cover taken off, which is also why the fix belongs in the collective rather
// than in this kernel.
//
// Reads are scalar loads of `uint16_t`. They are not hoistable: the load is the
// loop's exit condition, so no iteration can be folded into another.
//
// No cache maintenance around them, on purpose. The stamps this reads are written
// by a peer's `aclrtMemcpyAsync` D2D push, which is the identical mechanism that
// puts the peer's *plane* into this rank's memory, and that plane is already read by
// a device kernel -- `qwen_add_inplace_f16_ascend`, in the reduce below -- on every
// collective the engine issues today. If a device load could not see a peer's D2D
// write at all, the shipped engine's sums would be wrong, not merely slow. So the
// load path is coherent for peer writes and this kernel inherits that; adding a DCCI
// here would be insulating against a hazard the reduce behind it would still be
// exposed to.
//
// What that argument does not cover is *when* the write becomes visible, which is
// the window above, and a *stale* line this rank's own core read in an earlier round,
// since the table is reused every round. If this ever spins to its bound on a group
// that is clearly alive, the stale-line reading is the one to test, by invalidating
// the block with `AscendC::DataCacheCleanAndInvalid<uint16_t,
// CacheLine::ENTIRE_DATA_CACHE>` before re-reading. It is not the reading the
// measurements support: the observed failure is a wrong sum, not a spin to the bound.
// Two readings survive there -- the sender's plane copy is still in flight when its
// stamp lands, which wants a real release ordering on the sender, or the reduce's own
// loads hit a line the peer's write has not invalidated yet, which is what a DCCI on
// the reduce's input would address. Neither is implemented; both are priced by the
// dose-response rather than sized.
//
// First-generation 910 only, like every kernel in this directory.

#include "qwen_ipc_arrive_geometry.hpp"

#include "kernel_operator.h"

extern "C" __global__ __aicore__ void qwen_ipc_arrive_wait_kernel(
    GM_ADDR stamps_gm, GM_ADDR status_gm, uint32_t world, uint32_t rank,
    uint32_t round, uint32_t limit) {
    // One core spins. The other blocks would only add their own scalar traffic to
    // the cache line the peers are writing.
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    // Written on failure and only on failure, so the host does not have to read it
    // after every call to be sure of seeing one. A successful wait leaves whatever
    // was there alone; the host zeroes the word when it reads a zero back. See
    // kStatusCheckEvery in ipc_allreduce.cpp.
    AscendC::GlobalTensor<uint32_t> status;
    status.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status_gm), 1);
    if (world <= 1 || rank >= world) {
        return;
    }

    AscendC::GlobalTensor<uint16_t> stamps;
    stamps.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(stamps_gm),
                           static_cast<uint64_t>(world) *
                               pocket::kIpcStampStride);

    for (uint32_t spin = 0; spin < limit; ++spin) {
        uint32_t pending = 0;
        for (uint32_t k = 0; k < world; ++k) {
            if (k == rank) {
                continue;
            }
            const uint16_t want = POCKET_IPC_STAMP_VALUE(round, k, world);
            const uint16_t got =
                stamps.GetValue(static_cast<uint64_t>(k) * pocket::kIpcStampStride);
            if (got != want) {
                ++pending;
            }
        }
        if (pending == 0) {
            return;
        }
    }

    // Every iteration of the bound above was spent with at least one peer silent.
    // Hand the count back rather than a boolean so the host can say how many are
    // missing, the same way the host-side poll reports `pending` of `world - 1`.
    // Never zero: this write is the only record of the failure, and the host may be
    // kStatusCheckEvery calls away from reading it.
    uint32_t missing = 0;
    for (uint32_t k = 0; k < world; ++k) {
        if (k == rank) {
            continue;
        }
        if (stamps.GetValue(static_cast<uint64_t>(k) * pocket::kIpcStampStride) !=
            POCKET_IPC_STAMP_VALUE(round, k, world)) {
            ++missing;
        }
    }
    status.SetValue(0, missing == 0 ? 1u : missing);
}
