// Hand-written cross-process all-reduce for the Ascend backend.
//
// `HcclAllReduce` on this stack prices at 0.4810 ms per call whatever the payload
// size, and a rows=1 Qwen decode step issues 129 of them -- 62 of the step's
// 104.3 ms. The price is the host cost of handing the call to the runtime, not
// wire time: 10 KB and 640 KB move at the same rate. The way out is therefore a
// cheaper call, not a faster fabric.
//
// `cpp_engine/tests/bench_qwen_ascend_ipc_exchange.cpp` measured what a
// replacement made only of `aclrtMemcpyAsync` over IPC-imported buffers costs: a
// complete world-4 all-reduce at 10 KB in **0.259 ms**, 1.9x cheaper. This file is
// that barrier, wired to the contract `tp_comm.hpp` already declares.
//
// The three findings that shape it, all from that probe or from running this
// implementation in the engine:
//
//   * Nothing AscendCL offers can order the rounds across a process boundary. An
//     imported notify cannot be waited on (`aclrtWaitAndResetNotify` returns
//     107000) and a cross-process event cannot be created (207000, then 107000).
//     So the arrival signal is carried in the payload: the sender stamps a value
//     that changes every round next to the data, and the receiver polls for it.
//   * One receive buffer per (peer, source) pair is not enough. A peer that
//     reaches round N+1 before this rank has looked for round N has overwritten
//     the plane or the stamp being waited for, and the poll stalls until its
//     deadline and then resynchronises a round late -- measured as 21 stalled
//     rounds in 5000. Two buffer sets used on alternate rounds remove it: zero
//     stalls in 20000. A peer can never lead by more than one round, because its
//     round N+1 push is issued before it polls for round N+1 and that poll needs
//     this rank's round N+1 push, which is only issued after this rank's round N
//     completes.
//   * The stamps live in their own contiguous array, exported and imported like a
//     payload slot, rather than in the payload buffer. A blocking 2-byte D2H read
//     costs tens of microseconds, so reading one stamp per peer per poll iteration
//     is the single largest term in the loop; one read of the whole array -- two
//     bytes per rank, eight bytes at world 4 -- replaces all of them. The array is
//     indexed by (set, source) so the parity that protects the payload protects
//     the stamps too.
//
// Everything here runs on the caller's stream, the default stream included, and
// never substitutes one of its own. That is not a simplification: an earlier
// version staged the payload on a private stream, which made the copy unordered
// against the producer kernels and forced a `aclrtSynchronizeStream(nullptr)`
// before every one of the 129 calls. The device then had to go idle while the host
// woke up and enqueued the push, once per collective, and the whole point of the
// rewrite is not to add bubbles. On the caller's stream the pushes are enqueued
// behind the compute that produces them and the reduce is ordered behind the
// pushes, so the caller sees the result exactly as it would from `HcclAllReduce`.
//
// Everything here is opt-in. `POCKET_ASCEND_IPC_ALLREDUCE=1` selects it, and any
// call outside the configured envelope falls back to HCCL rather than failing.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pocket {

// Whether this call would use the hand-written collective: the feature is on, the
// world is larger than one, and the plane is within the size ceiling. A pure
// function of the arguments and the environment, so every rank answers it the
// same way -- which is what keeps the decision from desynchronising the group.
bool ascend_ipc_allreduce_f16_applies(int world, int count);

// All-reduce `count` FP16 elements in place across `world` ranks, one process per
// rank, rendezvousing through `id_path`'s directory. Every rank must call it in
// the same order, exactly as `HcclAllReduce` requires.
//
// Returns true when it ran, false when the call is outside the envelope and the
// caller must use HCCL instead. Throws when it ran and failed, including when a
// peer stops answering: a stall is reported rather than waited out indefinitely,
// because in a decode loop a lost peer and a lost round look identical from here.
bool ascend_ipc_allreduce_f16_inplace(int world, int rank, int device,
                                      const char* id_path, uint16_t* values,
                                      int count, void* stream);

}  // namespace pocket
