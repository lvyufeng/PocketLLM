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
//
// What justified building it took arms that are not the shipped configuration, so
// they are listed here rather than left to be found in the source. All default off
// and none of them widens the envelope:
//
//   * `POCKET_ASCEND_IPC_ALLREDUCE_STATS` prints a per-call split of the
//     collective's host time every 32 calls. Side-effect free.
//   * `..._SKIP` and `..._NOPOLL` do not compute the all-reduce at all -- the
//     first issues nothing, the second reduces whatever the peer slot happened to
//     hold -- so a token from such a run is meaningless. They bound what the
//     barrier costs by deleting parts of it.
//   * `..._POLL_SLEEP_US`, `..._SETTLE_US` and `..._RSTREAM` compute the right
//     answer and pay host or device time the shipped path does not, so a token
//     from one of them is meaningful and a step time from one is a bound.
//   * `..._DEVWAIT` moves the arrival wait onto a kernel on the caller's stream. It
//     computes that wait correctly and it is still a **wrong-result** arm: the host
//     round trip it deletes is also what covers the window between a peer's stamp
//     landing and its plane being readable, so ten interleaved pairs put it at 10
//     of 10 gate failures against the host poll's 0 of 10, at 77.117 -> 53.745 ms
//     and 12.968 -> 18.607 TPS. A token from it is not a token and its step time is
//     a bound on what removing the round trip could be worth, not a candidate. The
//     kernel is bounded and the host reads its status word every 32 calls, so a
//     lost peer is still reported rather than waited out; `..._DEADLINE_MS` bounds
//     the device spin from the same value it bounds the host poll with.
//     `..._SETTLE_US` is what turns it back into a correct arm -- the delay after
//     the arrival is exactly the window, and the dose-response is in 5.5.4 -- but
//     by then it is slower than the arm it was going to replace.
//   * `..._DEADLINE_MS` and `..._MAX_ELEMENTS` change when the barrier gives up and
//     which calls take this path at all.
//
// The numbers they produced are in docs/performance/ascend_single_request_tps.md
// 5.5.3 and 5.5.4; a process with one of them set says so on stderr before its
// first collective. One earlier switch, `..._ASYNCPOLL`, was measured and removed
// -- the blocking stamp read is faster -- and the .cpp carries why.
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
