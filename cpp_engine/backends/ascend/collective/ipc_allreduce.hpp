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
// This is the shipped collective on this backend, and the arrival wait it runs is the
// shipped wait. It is on unless `POCKET_ASCEND_IPC_ALLREDUCE` says `0`, `false`,
// `FALSE`, `off` or `OFF`, and any call outside the configured envelope falls back to
// HCCL rather than failing. So an off switch and a size ceiling, not an on switch.
//
// The wait itself runs on the device -- a kernel on the caller's stream spins on
// the peer stamps -- unless `POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT` says `0`, which
// selects the host poll. The host poll blocks inside `memcpy_d2h` for the whole
// rendezvous, so nothing is enqueued behind it and the device runs dry; deleting
// that round trip is 23.4 ms of a 77.1 ms step at 129 collectives a decode step. It
// defaults to the device for the same reason the barrier defaults on: it is
// measured, and its failure mode is bounded twice over -- the kernel spins a fixed
// number of iterations and gives up, and the host reads the word it writes on
// failure every `STATUS_EVERY` calls -- 128, one decode step -- and raises it as
// an error. `..._DEADLINE_MS` bounds both from one value.
// docs/performance/ascend_single_request_tps.md 5.5.3-5.5.4 has the
// measurement and docs/performance/serving_throughput_scaling.md the serving ladder.
//
// It used to be opt-in, and the reason recorded for that was a reproducibility
// gate: a batched path that is not run-to-run reproducible is a rare bad token in
// production rather than an error, which is worse than a slow one. That reason has
// since been discharged. The gate failures were the RoPE table aliasing the
// `Intermediate` workspace slot, not this barrier; with the table on its own slot
// the gate is failed 0 times in 22 interleaved runs across both arms
// (docs/performance/ascend_rope_table_workspace_aliasing.md). The size ceiling was
// then raised to 512 x 5120 so the barrier reaches the serving ladder's planes at
// all, which is where the 1.53x it is worth lives
// (docs/performance/serving_throughput_scaling.md). Leaving it opt-in after that
// would have shipped a measured win to nobody.
//
// The `atoi` reading that the on switch used to take is gone with it: under it
// `POCKET_ASCEND_IPC_ALLREDUCE=true` parsed as 0 and meant *disabled*, the
// opposite of what it says. Harmless while every document and script spelled it
// `=1`, and not something to keep once the variable governs the default.
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
//     from one of them is meaningful and a step time from one is a bound. SETTLE_US
//     and RSTREAM both compose with the device wait, which is the default.
//   * `..._DEADLINE_MS` and `..._MAX_ELEMENTS` change when the barrier gives up and
//     which calls take this path at all.
//   * `..._STATUS_EVERY` sets the interval, in collectives, between two host reads
//     of the device arrival wait's status word -- 128 by default, which at 129
//     collectives a decode step is one check a step. `0` stops reading
//     it at all, which is wrong by construction because a peer that never arrives
//     then stops being reported, and exists only to price the read in one run: the
//     word is in device memory and reading it is a blocking copy, so it drains the
//     default stream, which is the host wait the device arm was built to delete.
//
// `..._DEVWAIT` used to be on that list -- an arm that computed the wait correctly
// but was not what shipped. It is what ships now, so the two spellings have swapped
// places: `=0` is the arm that is off the shipped path, and it is not a run anyone
// needs to be warned about.
//
// The numbers these produced are in docs/performance/ascend_single_request_tps.md
// 5.5.3 and 5.5.4; a process with one of them set says so on stderr before its
// first collective. One earlier switch, `..._ASYNCPOLL`, was measured and removed
// -- the blocking stamp read is faster -- and the .cpp carries why.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pocket {

// Whether this call would use the hand-written collective: the feature is not
// switched off, the world is larger than one, and the plane is within the size
// ceiling. A pure function of the arguments and the environment, so every rank
// answers it the same way -- which is what keeps the decision from desynchronising
// the group.
bool ascend_ipc_allreduce_f16_applies(int world, int count);

// The interval, in collectives, between two host reads of the device arrival wait's
// status word -- `POCKET_ASCEND_IPC_ALLREDUCE_STATUS_EVERY`, 128 by default.
//
// Exposed for the same reason the predicate above is: it is a pure function of the
// environment with no ACL behind it, so a host-only test can pin its readings. What
// that test has to pin is the one reading that is not a spelling but a meaning --
// `0` is the interval, not "unset" -- because the alternative reading silently turns
// the off arm into the default arm, which is the trap `..._MAX_ELEMENTS` was already
// bitten by in the other direction.
long long ascend_ipc_status_check_every();

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
