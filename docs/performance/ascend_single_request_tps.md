# Ascend 910A Single-Request Decode: What The 104 ms Step Is Made Of

Qwen3.8-27B, 4 x Ascend 910 first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0,
checkpoint `/mnt/data1/modelscope/Qwen/Qwen3.8-27B`, TP4 ranks on devices 0-3. Every number below is
a measurement on that machine, and the command that produced it is named in the section that uses
it.

[The all-reduce A/B](ascend_decode_collective_ab.md) ends on the batch table: 113.3 TPS at rows=16,
214.2 TPS at rows=64. That is the throughput answer for a server, and it is not the answer to the
question this page asks, which is why **one** request is 9.6 TPS and what is left to do about it.

The short version: the step is 104.3 ms, **101.2 ms of it does not depend on the row count at all**,
and of that 101.2 ms about **62 ms is 129 `HcclAllReduce` calls whose per-call price is host latency
rather than bandwidth**. The remaining ~42 ms is the weight streaming and the recurrence. The copy
primitives needed to hand-write a collective are cheap and they do work across the process boundary
the engine runs behind. The three ordering primitives AscendCL offers across that boundary are *not*
usable here — an imported notify cannot be waited on and a cross-process event cannot be created —
but a barrier that reads its arrival signal out of the payload itself is, and it prices a complete
all-reduce at **0.26 ms against `HcclAllReduce`'s 0.4810** in an empty loop (§5.4). That barrier is
now built as a kernel-level replacement for the collective (the backend's default since the flip;
`POCKET_ASCEND_IPC_ALLREDUCE=0` selects HCCL instead) and it
has been measured inside the engine against the shipped path **interleaved run for run**, so the two
arms share a session: rows=1 goes from **106.33 ms / 9.408 TPS to 77.32 ms / 12.935 TPS**, +37.7%
(§5.5.4). It shipped opt-in, and the reason this page gave for that — the launcher's hard gate,
`batch_repeat_mismatches`, failed by **both** arms, 3 of 36 runs under HCCL and 8 of 36 under the
replacement — is retracted. Those failures were the partial-RoPE table aliasing a pooled workspace
buffer with a kernel the host had already queued, which is a separate defect with a separate fix and
nothing to do with the collective. With it fixed the gate is failed 0 times in 22 runs across both
arms, and the switch that was waiting on it has been flipped: **the hand-written collective is the
backend's default**, and `POCKET_ASCEND_IPC_ALLREDUCE=0` is the way back to HCCL. The device-side
wait of §5.5.3 also passes the gate once the table has its own slot, and takes rows=1 to
**53.0-53.9 ms and 18.56-18.86 TPS** — 18.6 TPS on a single request at the shipped row count, with
the tokens the reference produces. That one was left opt-in by this flip, because flipping a default
is its own change and it was not this one. It has since been flipped in turn, on
its own three-arm serving ladder — 1.22x clear of the host poll at 16 rows and
1.08x at 112 — so it is the backend's default too now, and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0` is what reaches the host poll
([serving concurrency](serving_throughput_scaling.md)).

What those same pairs do not leave in doubt is the step. Half of the saving came from deleting a
bracket rather than from the barrier: the first integration kept the engine's `begin`/`end` pair
around the new collective and paid 12.9 ms of host round trips for ordering it did not need (§5.5.1).
The empty-loop probe projected 13.2 TPS, so the barrier itself delivered what it promised; §5.5.3
prices what is left, and it is the arrival poll rather than the payload.

That is the collective side. §3 named the other side — 42.3 ms of non-collective per-layer work,
3.6x off the memory floor — and left it unscoped, on the reading that the rows=1 shape was what a
single request is and therefore not a lever. It is a lever: the Cube's M tile is sixteen rows, a
decode step has one, and the fifteen the tile wants do not have to be real. §6 is the layer's own
projections issued with the activation's batch dimension broadcast over the row it already has, and
it is worth **13.05 -> 17.59 TPS at rows=1 over three interleaved pairs**, +34.8%, on top of the
hand-written collective rather than against it. On the merged tree the two levers stack: the widths
generate one 32-token sequence — the reference's — across ten runs at both collectives and both
arrival waits, and with the device-side wait the step is **39.4-39.8 ms, 25.15-25.37 TPS**, still on
that sequence (§6.3).

## 1. The row sweep: single-request cost is batch-independent cost

8 decode steps, a 32-token prompt, verifier off, `QWEN_BATCH_ROWS` swept 1 to 8:

| rows | ms/step | decode TPS | TPS per row |
|---|---|---|---|
| 1 | 104.344 | 9.58 | 9.58 |
| 2 | 106.539 | 18.77 | 9.39 |
| 4 | 110.082 | 36.34 | 9.08 |
| 8 | 122.193 | 65.47 | 8.18 |

A least-squares fit over those four points is **101.2 ms that does not depend on the row count, plus
2.56 ms per row**. Throughput is 6.8x higher at rows=8 for 1.17x the time, because 97% of the step is
work that one row already pays for in full.

So the single-request number is not a scaling problem, it is a fixed-cost problem: **the 101.2 ms
fixed term alone would be 9.9 TPS, and the one row that has to be decoded adds 2.6% on top of it.**
Nothing about the row axis can move it, and the rest of this page is about the 101.2 ms.

Rows=1 here is the batched-decode path at one row, not the serial path — 104.3 ms against the
108.2 ms [the attention page](ascend_attention_optimization.md) measured. The batched wrapper is marginally cheaper at M=1
than the per-row wrapper, which is a side effect rather than a separate result.

```bash
# one arm of the table
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=0 QWEN_BATCH_PROMPT_LEN=32 \
  scripts/run_qwen_ascend_tp4.sh "" 8
```

## 2. Where the 101.2 ms goes

### 2.1 129 collectives

A decode step issues **129 `HcclAllReduce` calls**: 64 `ar.mlp` and 48 `ar.lin.out` and
16 `ar.full.out` from the three projection sites in `qwen_layer_components.inl`, plus
1 `ar.hidden_a` for the embedding output. Every one of them reduces a 5120-element fp16 plane
(`hidden_size`), so **10240 B is the message at all 129 sites** and one price covers the whole step.
`bench_qwen_ascend_allreduce` prices one call at world=4, 100 iterations:

| variant | ms/call | what it is |
|---|---|---|
| `sync` | 0.3994 | internal communication stream, no host synchronize |
| `comm` | 0.4810 | the sequence the engine runs: event record, stream wait, `HcclAllReduce`, `stream_synchronize` |
| `comm`, no trailing synchronize | 0.4740 | isolates the synchronize |
| `enqueue` | 0.4105 | API call only, in a queue kept deep |
| `pure-enq` | 0.3735 | enqueue with the drain removed entirely |

`0.4810 x 129` is **62.0 ms of the 104.3 ms step, or 59%**, and the ablation in
[§1 of the A/B page](ascend_decode_collective_ab.md) confirms it from the other side: three
independent `POCKET_TP_AR_NOOP` deltas of 65.5, 63.3 and 57.4 ms, measured there at rows=16, 32 and
64. Those are batched steps, but the ablation deltas are flat across the three because the collective
is per-*call*, and the same 129 calls sit in the rows=1 step. The per-call figure has a run-to-run
spread of about 10% — an earlier run of this same bench gave `comm=0.4256` and the engine's own phase
attribution is 0.4430 — so read the share as 55-59% of the step, not as three significant figures.

That spread is also the honest answer to "is 129 calls really the whole story": the collective is the
single largest term in the step by a factor of two over anything else, but it is not 100% of it and
the 41% that is not collective is what a collective-side fix leaves untouched.

### 2.2 The price is host latency, not bandwidth

The same call with the message scaled by 100x:

| bytes | ms/call (`comm`) | GB/s |
|---|---|---|
| 10 240 | 0.4810 | 0.02 |
| 81 920 | 0.5018 | 0.16 |
| 655 360 | 0.4681 | 1.4 |
| 5 242 880 | 2.1282 | 2.5 |
| 41 943 040 | 3.1954 | 13.1 |

Flat to 640 KB — sixty-four times the payload for less than nothing, since 655360 B is *faster* than
10240 B — and only then does the fabric start to show. The `bytes=10240` call moves 10 KB in
0.48 ms, which is 0.02 GB/s. Whatever this call costs, it is not the wire.

The last row is also the one place where the trailing synchronize is visible in the payload:
`comm=3.1954` against `comm-nosync=2.1314` at 40 MB, where the drain finally exceeds the host's
ability to hide it.

### 2.3 The platform floor for the same shape of work

One device operation, enqueued and drained, no fabric involved:

| probe | per call |
|---|---|
| `device_synchronize` alone | 0.0026 ms |
| `aclrtMemset` 4 B + synchronize | 0.0196 ms |
| `aclrtMemset` 10 KB + synchronize | 0.0195 ms |
| `aclrtSetDevice` | 0.0024 ms |

A 10 KB memset, issued and drained, is 0.0195 ms. The 10 KB all-reduce is 0.4810 ms — **24.7x the
cost of the operation it performs**, on a payload where the arithmetic itself is free. This is the
number that makes it tempting to replace the collective outright, and §5 is what happened when that
was measured rather than assumed.

### 2.4 The remaining 42-49 ms

`129 x 0.4810 = 62.0 ms` at one end of the spread and `129 x 0.4256 = 54.9 ms` at the other, so
**42.3 to 49.4 ms of the 104.3 ms step is not collective**. This is 64 layers of weight streaming
plus the gated-delta recurrence, the norms, and the sampling. §3 prices it.

## 3. The ceiling is 85 TPS of HBM, not of compute

The HBM read probe in `bench_qwen_ascend_decode_ops` measures **1148 GB/s**. `resident_weight_bytes`
is 13 449 011 456 B per card at TP4, so reading the weights once costs
`13449011456 / 1148e9 = 11.7 ms` — **85 TPS with zero compute and zero communication**.

The step is 104.3 ms and the memory floor is 11.7 ms. 9.58 TPS against 85 TPS is a **8.9x gap**, and
none of it is the memory system: the 42.3 ms of non-collective work is already 3.6x the streaming
floor, and the collectives are on top of that.

The size of the collective prize is bounded and worth stating explicitly: **make every collective
free and the step is 42.3 ms, or 23.6 TPS.** That is a 2.5x on single-request throughput — but it is
not the only term left, because the 42.3 ms of per-layer work is itself above its own floor. §6 is
that term: the layer's own projections, issued so the Cube's sixteen-row tile is full, for 20.1 ms of
a 76.6 ms step.

## 4. Two levers that measured to nothing

### 4.1 MTP is a loss on this checkpoint

`--qwen-mtp-tokens 3` against plain greedy decode, 8 tokens, 5-token prompt, same machine and
checkpoint:

| arm | decode TPS | drafts proposed | accepted |
|---|---|---|---|
| plain | 9.62 | — | — |
| `--qwen-mtp-tokens 3` | 4.28 | 15 | 0 |

A longer arm (64 generated tokens) gives the same picture with better statistics: **9.35 -> 3.65
TPS**, `mtp_accept_rate=0.0112994`, 177 drafts proposed and 2 accepted. `spec_accept_length=1.03333`
against `mtp_tokens=3` is a draft head that is not predicting the target's own next token, and every
rejection costs a full 3-token verify pass plus a replay — `mtp_verify_seconds=9.55` and
`mtp_replay_seconds=6.43` against 6.74 s of plain decode. Speculative decoding is a **2.2x to 2.6x
loss** on this checkpoint, and that is a property of the checkpoint rather than of the Ascend port:
the draft module is the same one [the Qwen drafter acceptance page](qwen_drafter_acceptance.md)
measures on the GPU path.

### 4.2 Verifier placement

`QWEN_VERIFY_DEVICE_TOP1` moves the top-1 reduction off the host and onto the device. On/off on the
same 8-token run: **9.29808 TPS against 9.3055 TPS**. The difference is 0.08%, which is inside the
run-to-run spread. The verifier is not where the step goes, and this lever should be treated as
retracted rather than pending.

## 5. Writing the collective by hand

The arithmetic is 10 KB of fp16 summed four ways and broadcast back, and it costs 0.48 ms. Three
things have to be true before a hand-written replacement is even possible here, and none of them is
evident from the API:

1. Can a rank's buffer be reached from another rank?
2. Does that survive the process boundary, given the engine runs **one process per TP rank**
   (`HcclCommInitAll` segfaults on this stack, so the ranks rendezvous through a shared id path and
   never share an address space)?
3. Is the resulting collective cheaper than 0.48 ms?

### 5.1 The primitives, in one process

`bench_qwen_ascend_peer_copy --device 0`, all eight cards visible from one process (the engine
deploys on 0-3, but the peer topology is worth seeing whole):

| primitive | result |
|---|---|
| `aclrtDeviceCanAccessPeer` + `aclrtDeviceEnablePeerAccess`, peers 1-7 of the 8 devices the host reports | err 0, all enabled |
| `aclrtMemcpy` D2D across devices (synchronous) | **err 507899 `RT_DRV_INTERNAL_ERROR`, destination unchanged** |
| `aclrtMemcpyAsync` D2D across devices | err 0, data matches |
| `aclrtMemcpyAsync` cross-device, enqueue price | 0.0032 ms |
| `aclrtMemcpyAsync` same-device, enqueue price | 0.0032 ms |
| `aclrtMemcpyAsync` 4-way batch (`aclrtMemcpyBatchAsync`) | **err 207000 `RT_FEATURE_NOT_SUPPORT`** |
| `aclrtRecordNotify` + `aclrtWaitAndResetNotify`, same device | 0.0059 ms |

The first line is the one that has to be known before anything is written: **the synchronous
device-to-device copy does not work between cards on this CANN/driver pair.** It reports an error and
it does not transfer. The asynchronous form of the same copy is fine, and at 0.0032 ms to enqueue
there is nothing wrong with its price. Any code that reaches for `aclrtMemcpy` because the copy is
"trivially small" will get an error and stale data.

A four-way host-driven all-reduce built on those primitives, in one process, produces the correct
sum (`correct=1` — every rank reads back `n(n+1)/2` in fp16) at **2.5468 ms per round**. That number
is not a verdict on the algorithm: the breakdown is `push_ms=0.0543`, `local_ms=2.4087`,
`barrier_ms=0.0838`, `device_set_ms=0.0208`, `per_rank_ms=0.6315`, and the dominant window is the one
where the probe interleaves `aclrtSetDevice` with aclnn calls on four cards at once —
`add_launch` prices the same add at 0.0105 ms enqueued alone and **0.2463 ms once a device switch is
interleaved with each call**. That cost belongs to the single-process experiment. The engine has one
process per rank and pays the device bind once at startup, so the measurement that decides anything
is the next one.

### 5.2 The primitives, across processes

`bench_qwen_ascend_ipc_exchange`, four processes, one device each, keys exchanged through files in
`--dir` (rendezvous only; the transfer is the thing under test):

| step | result |
|---|---|
| `aclrtIpcMemGetExportKey`, 4 slots per rank | ok 4/4 |
| `aclrtIpcMemImportByKey`, default flags | **err 507899 `RT_DRV_INTERNAL_ERROR`** |
| `aclrtIpcMemImportByKey` after exporting with `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION` | **ok 3/3** |
| `aclrtIpcMemSetImportPid` | **err 507899 on both sides** |
| `aclrtNotifyGetExportKey` / `aclrtNotifyImportByKey` | ok, 3/3 |
| `aclrtMemcpyAsync` into an imported peer buffer, 600 calls | err 0 on every call |
| `result=ok slots_ok=4/4` on all four ranks | every slot holds its owner's value |

So the copies, the keys, and the notifies all cross the process boundary, and the arithmetic comes
out right. Two operator notes fall out of the table:

- **Import only works with the PID whitelist off**, and the flag that turns it off
  (`ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION`) has to be set by the **exporter**. The
  importer-side `aclrtIpcMemSetImportPid` that the header suggests as the alternative does not work:
  it returns 507899 itself. The exporter's PID is embedded in the export key, so a rank that exports
  with the default flag produces a key only its own process can import.
- The failure mode is a clean error code at import, not a later fault — which is the good version of
  this problem.

### 5.3 A notify-based barrier loses

With the channel proven, the four barrier shapes, 200 iterations, world=4, `--bytes 10240`:

| barrier | push_ms | wait_ms | total_ms |
|---|---|---|---|
| none (`--no-wait`) | 0.0169 | 0.0000 | 0.0922 |
| own notify only (`--self-wait`) | 0.0173 | 0.0026 | 0.0986 |
| one peer notify (`--wait-one`) | 0.0271 | 0.2473 | 0.2859 |
| all three peers (`--peer-wait`) | 0.0310 | 0.7212 | 0.7666 |

Every arm is `result=ok slots_ok=4/4`. The `total_ms` includes the closing `stream_synchronize`,
which is why the no-wait arm is 0.0922 and not 0.0169 — draining three cross-process copies is
~0.075 ms, and that part is fine.

The reading is unambiguous:

- **Pushing is cheap and flat.** 0.0169 ms for three cross-process copies plus a notify record, and
  it does not grow with the message: at 1 MB the whole peer-wait arm is 0.7609 ms against 0.6986 ms
  at 10 KB, i.e. **100x the payload for 9% more time**. Same flatness as `HcclAllReduce`.
- **Waiting on a remote notify is the whole cost**: 0.0026 ms for your own notify, **0.2473 ms for
  one peer's, 0.7212 ms for three** — per wait, not a global stall, and `n-1` remote waits is
  `(n-1) x 0.23 ms` almost exactly.
- The notify import flag makes no difference (`--notify-no-peer-access` changes nothing), so it is
  the cross-process signal path and not the peer-access mapping behind it. `--set-import-pid` does
  not work, so it cannot be tested the other way.

**A world-4 notify barrier therefore costs 0.0310 + 0.7212 = 0.75 ms of host time**, against
`HcclAllReduce`'s 0.4810 — a 1.6x **loss**, 129 times per decode step. But that is a verdict on *this*
barrier and not on hand-writing the collective. The notify arms are the ones that need a host-visible
signal, and on this stack a host-visible signal from another process is ~0.23 ms, one per peer. §5.4
is the barrier that does not need one.

### 5.4 The barrier that works: the arrival signal is the payload

Which leaves one signal path that is neither a notify nor an event: the data. Push the payload into the
peer's slot, then poll your own slot for it. Nothing has to cross a process boundary except the value
that was going to cross it anyway.

The form that works is `--poll-wait` in `bench_qwen_ascend_ipc_exchange`. Each rank stamps element 0
with a counter only it writes — `0x3c00 + 0x400*rank + it`, so the bands are disjoint and the round is
recoverable from the slot — pushes its buffer to all three peers, and then loops over its own receive
slots doing a 2-byte D2H read until the stamp of the round it is in is there. The counter is what makes
the poll mean anything: a fixed sentinel is already in the slot from the previous round, so the poll
would return before the peer had pushed. It is a genuine barrier and not four processes happening to
run in lockstep: with `--skew-us 50`, which makes rank r sleep r×50 µs before each round, the four
ranks' `wait_ms` separate cleanly along the induced arrival order — **0.271 / 0.173 / 0.134 / 0.081**
for ranks 0-3, against 0.097-0.148 with no skew — so the barrier absorbs the 150 µs rank 0 arrives
early with and the last rank pays nothing.

Two receive buffers per (peer, source) pair are required, used on alternate rounds, and that is not a
detail. With one, a peer that gets a round ahead overwrites the stamp being waited for, and the poll
then cannot match at all — it burns its deadline and resynchronises a round late, which is a stall
rather than a wrong number. `--poll-single-set` selects the one-buffer layout so the difference is
measured rather than asserted; at 5000 rounds it hits the probe's stall cap on **21 rounds** while the
two-buffer layout stalls **zero times in 20000**:

| layout | rounds run | `wait_ms` | stalled rounds | `total_ms` |
|---|---|---|---|---|
| one buffer per pair (`--poll-single-set`) | 2 760 of 5 000 | **7.33** | 21 | 7.65 |
| two buffers, alternating | 5 000 | 0.092-0.122 | 0 | 0.330 |
| two buffers, alternating | 20 000 | 0.096-0.116 | 0 | **0.259** |

Every arm reports `result=ok slots_ok=4/4` and `reduce sum_ok=5120/5120 first=0x4b80 (15.0)
want=15.0` on all four ranks, so the exchange and the reduce are both correct, not only fast.

The steady state at 20000 rounds is `push_ms=0.044`, `wait_ms=0.109`, `reduce_ms=0.070` and a 0.032 ms
trailing drain, for **0.259 ms for a complete world-4 all-reduce at 10 KB**. The `reduce_ms` figure is
three `aclnnInplaceAdd` calls and one device copy and it still carries the aclnn first-call
initialisation: the same number is 4.6655 ms at 100 rounds, 2.4237 at 200 and 1.4158 at 400, i.e. a
fixed ~480 ms amortised over the run, so the steady per-call reduce is nearer 0.046 ms. Steady buckets
are flat (0.086-0.118 ms per quarter-run, first quarter included), so the barrier settles immediately
and the single averaged figure is a fair description of it.

Against `HcclAllReduce`'s 0.4810, that is **1.9x cheaper per call**. Extrapolated to the engine's 129
calls that would take the collective from 62.0 ms to 33.4 ms and the step from 104.3 ms to 75.7 ms, or
9.6 TPS to **13.2 TPS**. The caveat this number has to be read under: this probe reduces on an idle
device, and the engine's collective never is, so the projection is an upper bound on what the barrier
is worth in place. §5.5 puts the same barrier in the engine and lands at 77.32 ms / 12.935 TPS — 2.1%
short of the projection, which is close enough that the barrier was never the problem. What made the
first engine integration miss it by 17% was not the barrier but the bracket the engine wrapped it in.

What this probe cannot settle: its loop is empty, so the ranks stay within a fraction of a round of
each other by construction and every push is issued into an idle device. The engine has a layer's worth
of compute between collectives. What is settled is the thing §5.3 could not settle: the barrier itself,
at world=4 on four processes, costs less than the collective it would replace. §5.5 is what happens
when the same barrier is put where the collective actually is.

### 5.5 The same barrier inside the engine

The barrier is now `cpp_engine/backends/ascend/collective/ipc_allreduce.{hpp,cpp}`, reached through
`tp_all_reduce_sum_f16_inplace`. It is the default for any call with world > 1
and a plane of at most `kDefaultMaxElements` FP16 elements, and `POCKET_ASCEND_IPC_ALLREDUCE=0` turns
it off; a call outside that envelope falls through to HCCL unchanged, and every rank evaluates the
same predicate on the same arguments, so the choice cannot desynchronise the group. That ceiling was
40 960 when this section was written, chosen by a
sweep §5.5.4 then re-tested and could not separate the sizes between, and it has since been re-derived
from the other end: [serving_throughput_scaling.md](serving_throughput_scaling.md)
measures the two barriers against each other as whole-prefill TTFT, finds the hand-written one
cheaper to 421 rows and dearer from 629, and the default now sits at 512 rows — 2 621 440 elements.
So the envelope covers prefill and batched decode as well as the one-row plane it was written for,
which is a change to what a serving run does and not only to what a diagnostic can measure.
Everything in this section was measured under the old ceiling and none of it moves: §5.5.1-5.5.4 are
all rows=1 steps, where both ceilings admit the hand-written path.

**The fall-through is a run that still ends in the reference's tokens.** The predicate is covered
device-free by `cpp_engine/tests/test_ipc_allreduce_envelope.cpp`; what that cannot show is the
envelope doing its job inside a decode. Two interleaved pairs at 16 rows, `POCKET_ASCEND_IPC_ALLREDUCE_STATS=1`
on both arms, one arm at the shipped ceiling and one with `..._MAX_ELEMENTS=4096` so that a 81 920-element
plane is over it:

| arm | barrier calls in `rank0.log` | `step_ms` | gate |
|---|---:|---:|---|
| shipped ceiling | 697 | 104.03, 104.14 | 0 mismatches, 48 of 48 agreed |
| `..._MAX_ELEMENTS=4096` | **0** | 139.78, 143.54 | 0 mismatches, 48 of 48 agreed |

The refused arm's step is the HCCL step, its four runs produce **one identical token stream**
(md5-equal across all four `ref_step` logs), and its seed line is `seed_mismatches=0`. So an
environment that lowers the ceiling does not corrupt a batched decode, it moves it to the other
collective — which is the property the default depends on, since the ceiling is the only thing
keeping a plane the barrier was not measured for off it.

#### 5.5.1 The bracket cost more than the collective saved

The first engine integration wrapped the new collective in the same `begin_nccl_collective()` /
`end_nccl_collective()` bracket the HCCL path uses, and that is the version that was measured first:

| arm | `step_ms` | decode TPS | vs the arm it replaced |
|---|---|---|---|
| HCCL, bracketed (the shipped default when these ran) | 105.783 | 9.45333 | — |
| hand-written, bracketed | 91.5055 | 10.9283 | +15.6% |
| hand-written, bracket removed | **78.593** | **12.7238** | **+34.6%** |

`end_nccl_collective()` ends with `stream_synchronize(nccl_comm_stream)` — a host round trip that waits
for the default stream to drain and *then* for the collective to finish — and around a call that
already runs on the caller's stream it buys no ordering at all. It is worth 12.9 ms of a 91.5 ms step,
0.100 ms per call over 129 calls, and it is the whole difference between the middle row and the last
one. Removing it turns the replacement from a 13.6 ms saving into a 27.2 ms one.

The bracket cannot simply be deleted, because the HCCL path does need it: a null-stream `HcclAllReduce`
drains the default stream and then runs on a private one, so the ready-event plus the synchronize is
the only thing ordering it against the compute stream. What the caller has to know is *which*
collectives it is wrapping, which the backend knows and the engine does not, so the contract gained
`pocket::tp_all_reduce_f16_on_caller_stream(world, count)`. Ascend's hand-written collective answers
true, because everything it issues is stream-ordered on the stream it is handed; the Ascend HCCL path
and NCCL on CUDA both answer false. `all_reduce_half` brackets only where that answer is false. The
predicate failing in the other direction is a correctness hazard rather than a slow step, which is
why it is the backend that answers and not the caller.

#### 5.5.2 The gate it has to pass

Rows=1, 8 decode steps, 32-token prompt, one process per rank on devices 0-3, `QWEN_BATCH_VERIFY=3`:

| arm | `step_ms` | decode TPS | `verify_agreed` | seed / verify / repeat / batch-repeat mismatches | `verify_logit_abs_matched` |
|---|---|---|---|---|---|
| `POCKET_ASCEND_IPC_ALLREDUCE=0` (HCCL) | 105.102 | 9.51456 | 3 | 0 / 0 / 0 / 0 | 0.076725 |
| `POCKET_ASCEND_IPC_ALLREDUCE=1` (before 5.5.1) | 91.5055 | 10.9283 | 3 | 0 / 0 / 0 / 0 | 0.693588 |
| `POCKET_ASCEND_IPC_ALLREDUCE=1` (with 5.5.1) | 78.593 | 12.7238 | 3 | see 5.5.4 | — |

The middle row is the replacement before the bracket was removed and the last row is what it costs
today. The last row's mismatch column is a pointer forward rather than a zero, because the number in
it is not a gate. `verify_mismatches` compares a batched row against the single-row reference on a
checkpoint whose per-step top-1 is a near-tie, and the launcher's gate explicitly does not require it
to be zero; an earlier revision of this page read it as a correctness gate and concluded from it that
the replacement was clean. The number that does gate is `batch_repeat_mismatches`, and §5.5.4 is what
it says about the replacement: the interleaved A/B fails it 8 times in 36 under the hand-written
collective and 3 times in 36 under the shipped HCCL path, which is not a separation, and 106.33 ms /
9.408 TPS against 77.32 ms / 12.935 TPS is. The failure counts are a workspace race and not the
collective — the retraction opening §5.5.4 says what they were — so neither arm can be read off them
at all.

What does move between the arms is the distance between the batched and the single-row arithmetic:
`verify_logit_abs_matched` is 0.69 against HCCL's 0.077, and the checksum distance 0.032 against
0.023. That is a comparison *inside* each arm — a batched decode against a single-row decode through
the same collective — so it is not the two paths disagreeing with each other. It is the same fp16 sum
taken in a different order, three pairwise `InplaceAdd` accumulations here against HCCL's ring,
sitting under a batched-versus-single-row gap both arms already have.

The timing rows above carry `QWEN_BATCH_VERIFY=0`; the gate rows below carry the verifier. Both are
separate runs because the verifier re-prefills and re-decodes the grid three times over, which would
land inside the clock window.

#### 5.5.3 What the hand-written collective still costs

Against the projection in §5.4 (75.7 ms) the barrier in place is 77.2-78.6 ms across runs, so the
empty-loop probe was 2-4% optimistic about its own lever and the barrier itself was never the problem.
What is left is measurable against two floors. Both come from diagnostic switches that produce wrong
results and exist only to price the parts: `SKIP` drops the push, the poll and the reduce; `NOPOLL`
keeps the push and the reduce and drops the poll. Same shape, same devices, same bracket-off
configuration:

| arm | `step_ms` | decode TPS | what it contains | in-kernel total per call |
|---|---|---|---|---|
| `SKIP` + `NOPOLL` | 50.7318 | 19.7115 | the layer, no collective at all | 0.0026 ms |
| `NOPOLL` | 51.9047 | 19.2661 | payload push and three `InplaceAdd`s, no wait | 0.079 ms |
| the real collective | 78.3432 | 12.7644 | the same, plus the poll | 0.367 ms |

**The poll is what the hand-written collective spends.** Its payload path costs the step 1.2 ms over the
collective-free floor even though its in-kernel total is 0.079 ms per call — nine tenths of that is
hidden behind device work that was queued anyway. The poll is the other 26.4 ms, and it is not hidden at
any level.

The same barrier on an idle device with an empty loop is 0.109 ms of `wait_ms` (§5.4). In the engine
the steady-state bucket is **0.285 ms** per call — `POCKET_ASCEND_IPC_ALLREDUCE_STATS`, the last 32-call
window of three decode runs, 0.2799 / 0.2836 / 0.2909. Two candidate causes for the 0.18 ms difference,
and these numbers no longer leave them undecided: the first has been built and refuted.

**The read is not the cost, and removing its drain makes it worse.** `pocket::memcpy_d2h` drains the
default stream before its blocking `aclrtMemcpy`, and the collective is issued on that same stream, so a
poll iteration looked like it was waiting out the whole layer queued behind it — a host round trip of
exactly the class §5.5.1 removed, reintroduced once per iteration, and unable to order the write being
waited for in any case, because that write belongs to another process. The read was replaced with an
`aclrtMemcpyAsync` into pinned host memory on a stream this collective owns, draining only that stream,
and the two were interleaved eight pairs against each other: the asynchronous read is slower in **eight
of eight**, 76.73 → 85.96 ms and 13.035 → 11.634 TPS, a 10.7% loss. The switch was removed rather than
shipped.

The same diagnostic says why. The blocking read finishes in 1.0-1.31 iterations (1.31 / 1.00 / 1.28) at
0.285 ms of `wait_ms` per call; the asynchronous one takes 2.97-3.19 iterations (3.03 / 2.97 / 3.19) at
0.322. The drain is not dead time — it *is* the wait, and it is a cheap way to spend it. Waiting out the
layer is what makes the peers arrive before the read, so the loop runs once; skipping the drain returns
the read promptly enough to find nothing, and a repeat does cost less than half of a blocking iteration
(0.105 ms against 0.238, over the three-run means) but three of them cost more than one. That accounts
for 4.8 ms of the 9.24 ms regression (0.0371 ms per call over 129 calls); the other 4.4 ms is outside
the barrier's own timing, which is what a host that no longer paces itself to the device looks like from
the step clock.

**The wait is the host's round trip, and the peers are already there when it is made.** The paragraph
that used to close this section read the 26.4 ms as a peer-compute wait that only rescheduling could
recover, and that is retracted. The switch that moves the wait onto the device says what it actually
is. `POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT` enqueues `qwen_ipc_arrive_wait_kernel` on the caller's stream
— one core spinning on the same stamps against the same values the host loop checks — and returns
without blocking. Ten interleaved pairs, one run of each arm alternating, the launcher, shapes and gate
of §5.5.4's pairs:

| arm | `step_ms` | decode TPS | `wait_ms` per call | poll iterations | gate |
|---|---|---|---|---|---|
| host poll (the arm shipped then; `..._DEVWAIT=0` now) | 77.117 (76.551-77.648) | 12.968 | 0.285 | 1.00-1.31 | 0 of 10 |
| device wait | 53.745 (53.069-54.247) | 18.607 | 0.130 | — | **10 of 10** |

23.4 ms of a 77.1 ms step, 30.3% of it, is the host's round trip through the runtime, and it is device
idle time behind that round trip rather than anything the group is waiting for. The poll's own 1.00-1.31
iterations say the peers have arrived before the first blocking read completes. What the drain buys is
that ordering in front of the read — which is why `ASYNCPOLL`, which removed the drain, lost 10.7%, and
why `DEVWAIT`, which removes the round trip and keeps the ordering, gains 30%. The device wait lands
1.8 ms above the `NOPOLL` floor of 51.905 ms, the same order as the run-to-run spread between two
sessions; what is left is the part of the wait the device does have to do for itself.

**The device wait is correct, and the 10 of 10 in that table is not about the collective.** The gate
column above was read as the device wait being wrong, and the paragraph that used to follow it — kept
as the retraction below — built a whole mechanism out of it. It was wrong. The gate failures were the
partial-RoPE table's workspace aliasing, a race in a pooled buffer between a blocking H2D copy and an
attention kernel the host had already queued, and the device wait was failing the gate more often than
the poll for the same mundane reason the replicated rows were: it moves the timing of that window.
With the table given its own slot (the RoPE table's workspace slot page, `#280`), five interleaved
pairs of poll against device wait, the gate of §5.5.4 on both arms:

| arm | `step_ms` | decode TPS | gate failures |
|---|---|---|---|
| host poll | 76.10-77.66 | 12.88-13.14 | 0 of 5 |
| device wait | 53.02-53.88 | 18.56-18.86 | **0 of 5** |

and the same 32 tokens come out of all four runs made for it — poll twice, device wait twice — which
are the reference's own: `11751 13 198 760 6511 314 9564 369 19241 ...`. So the 26.4 ms is recovered
rather than bounded, at the shipped row count, and the switch is the shape it ships in
now that the workspace fix is in the same tree — the flip itself is measured on the
serving ladder in [serving concurrency](serving_throughput_scaling.md), and
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0` is the way back to the host poll.

**It was read as a bound rather than a candidate, and that reading is retracted.** What used to stand
here: every one of those ten device-wait runs fails the launcher's reproducibility gate, §5.5.4 has
the dose-response that identifies the cause, it is the same event as the hand-written arm's own 8
failures in 36, so the 26.4 ms is recoverable in principle and this particular way of recovering it is
not correct — the host round trip is not only waiting, it is also the fence the arrival signal does not
have. The first three clauses were true and the last one does not follow from them. The failures were
the RoPE table's workspace aliasing, which lands on the same comparison; the device wait did not lack
a fence, it changed the window. The paragraph above has the re-measurement.

At rows=16 the replacement was not in the picture when these runs were taken: 16 rows x 5120 is 81 920
elements, twice the 40 960-element ceiling then in force, so a batched decode fell back to HCCL and the
rows=16 arm was HCCL against itself. The ceiling has since been raised to 512 rows
([serving_throughput_scaling.md](serving_throughput_scaling.md)), so that plane is on the replacement
now; the retraction is unaffected either way, since it is about the RoPE table and not the collective.
An earlier revision of this page quoted +33% here (141.184 ms / 113.327 TPS becoming
106.365 ms / 150.425 TPS). That measurement was taken with the ceiling at 1 Mi elements, i.e. before
the reproducibility sweep, and with `QWEN_BATCH_VERIFY=0`, so it is a timing number from a
configuration that did not gate. It is retracted rather than corrected: under the ceiling in force
when it was taken there was nothing to measure, and the ceiling itself is a default rather than a
bound (§5.5.4), so re-raising it to re-measure would be choosing a configuration to produce a number
rather than measuring the shipped one.

#### 5.5.4 The reproducibility gate, and which arm it actually separates

> **Retracted: this gate was never measuring the collective.** Every failure recorded in this
> subsection — both shapes, both arms, all thirty-six interleaved pairs and the pooled runs behind
> them — is the partial-RoPE table aliasing the `Intermediate` workspace slot, a race between a
> blocking H2D copy and an attention kernel the host had already queued. It is a defect in the
> workspace pool with a two-file fix, and it has nothing to do with which collective is in the step.
> With it fixed the gate is failed 0 times in 22 interleaved runs across both arms, including the
> device-wait arm of §5.5.3 that used to fail it 10 times in 10. The analysis below is kept because
> the rates, the shapes and the argument about interleaving are a correct description of what the gate
> *looked* like, and because the retraction is easier to read against them; the conclusions it draws
> about the platform, the barrier and the two collectives are withdrawn. A batched pass at rows=1 is
> reproducible on this platform.

`batch_repeat_mismatches` is the batched pass run twice from a reset engine with nothing else
changed — the reproducibility the batched path owes its callers, and a hard zero in the launcher's
gate (`scripts/run_qwen_ascend_tp4.sh`). Two earlier revisions of this page read it wrong in opposite
directions, and the way it is easy to get wrong is worth stating before the numbers.

**It is not `verify_mismatches`.** That one compares the batched pass against a single-row reference
built from a different set of kernels, and on this checkpoint's near-ties that comparison is a coin
flip the launcher does not gate on. At rows=1, on the token the two paths *both* choose, their logits
differ by 0.05 to 0.61. When the top-1 margin is narrower than that they choose differently, and every
step after the divergence differs for a reason that is not a defect. That is where the
`worst_logit_abs` of 23.76-23.98 comes from, and an earlier revision of this page read it as "a whole
contribution missing".

**It is not a gate that separates the two collectives.** An earlier revision of this section asserted
the opposite — that HCCL reproduces the batched pass 20 times in 20 while the replacement fails it 7
times in 32 — and that reading does not survive interleaving. Same launcher, same checkpoint, same
shapes, `QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=3 QWEN_BATCH_PROMPT_LEN=32`, 8 decode steps, one process
per rank on devices 0-3, the two arms alternating run for run:

| arm | `batch_repeat_mismatches` failures | runs | step | decode TPS |
|---|---|---|---|---|
| `POCKET_ASCEND_IPC_ALLREDUCE=0` (HCCL; the shipped default when these ran) | 3 | 36 | 106.33 ms | 9.408 |
| `POCKET_ASCEND_IPC_ALLREDUCE=1` (hand-written; the default since the flip) | 8 | 36 | 77.32 ms | 12.935 |

**HCCL fails this gate too.** The point estimates are 8.3% against 22.2% and the rates are still not
separated (Fisher two-tailed p = 0.19), so this gate neither rejects the replacement nor validates it.
Thirty-six interleaved pairs is not a sample that could have separated them: at those two rates it has
about 37% power at the 5% level, and roughly 120 pairs per arm would be needed for 85%. The interleaved
pairs are the comparison that counts, because it is the only one where the two arms
share a session; pooling every rows=1 gate-on run of the session — HCCL 3 of 62 across several
configurations, hand-written 12 of 56 — does separate them (p = 0.011), and that number is not
trustworthy for the same reason: the arms are not matched, and every run in the pool that is not one
of the thirty-six interleaved pairs was launched whenever it was convenient. The honest statement is
the one the interleaved runs support, which is that the batched path at rows=1 is not run-to-run
reproducible on this platform under **either** collective, and that the rate at which it is not has
not been pinned down for either.

That last sentence is withdrawn with the rest of the subsection: what the rates are rates *of* is the
RoPE table's workspace race, which is in both arms, and with it fixed the same gate the same way is
failed 0 times in 22 runs. What survives here is the arithmetic about samples — the interleaved pairs
are still the only matched comparison, and still too few to have separated two rates this close.

**What the failures look like.** Both arms fail it in one of two shapes, and the two shapes are the
same event observed through two comparisons.

- *A top-1 flip at step 0.* The batched pass and the single-row reference put the same two tokens on
  top — `248046` and `197`, at the last prompt position, 33 — and disagree about which of the two wins.
  Across the four runs that do this in the thirty-six interleaved pairs, the single-row path's logit
  for its choice is 6.269-6.282 and the batched path's for its own is 6.329-6.477; every run reports
  the identical pair, which is what a *systematic* near-tie looks like when a run-to-run perturbation
  is large enough to cross it. Nothing is corrupted. The gate does not record the top-1/top-2 margin
  inside either path, so the pair is what identifies this shape rather than the margin.
- *A repeat-only flip.* The first batched pass agrees with the single-row reference on all three
  gated steps and the second batched pass, launched from a reset engine with the same tokens and
  slots, does not. `verify_mismatches=0`, `verify_agreed=3`, `worst_logit_abs` 0.048 to 0.95 across the
  recorded runs. This shape is the cleanest evidence available, because the single-row reference
  reproduces itself in almost every one of these runs: what is not reproducible is specifically the
  batched path.

Both shapes are present, and every recorded failure of either shape is real: in all of them
`batch_repeat_mismatches=1`, which is the batched path disagreeing with itself. What is *not* evenly
distributed is the first shape. All four step-0 flips in the thirty-six pairs are in the hand-written
arm; no HCCL run has one. That is a difference in the direction the section's earlier claim needed, so
it is worth saying why it is not evidence for it: the hand-written sum is a different order of
operations from the single-row reference's, §5.5.2 already measures it landing further from that
reference (`verify_logit_abs_matched` median 0.24 against HCCL's 0.12), and a near-tie flips in
proportion to the distance to the thing being compared against. The sixty-two-against-fifty-six pool
does show the same lean — one trip of `verify_mismatches` in the HCCL arm against ten in the
hand-written one — and it is the same unmatched pool that cannot support a conclusion about the rate.
The gate for this shape is `verify_mismatches`, and the launcher explicitly does not gate on it.

Named runs of the record: `rate_hccl_2` (HCCL, repeat-only, `worst_logit_abs` 0.607),
`rate2_hccl_6` (HCCL, repeat-only, 0.048), `rate_ipc_1` (hand-written, repeat-only, 0.436),
`rate_ipc_9` (hand-written, step-0 flip, 23.759) and `rate2_ipc_14` (hand-written, step-0 flip,
24.212).

**What the gate has been catching — and the answer, which is not the collective.** This subsection was
written to read the dose-response below as a peer-stamp/plane ordering defect in the barrier and to
attribute the gate failures to it. The table's own numbers are real and are kept; the reading is
retracted. It was the partial-RoPE table aliasing the `Intermediate` workspace slot, and any lever that
moved when the host reached the copy moved the failure rate — which is what the delay column below
actually sorts on.

The device wait of §5.5.3 is what made the gate legible. It deletes the host's round trip and nothing
else, and on the unfixed build it failed the gate 10 times in 10 while the host poll failed it 0 times
in 10, same launcher, same shapes, same session, alternating run for run. The round trip is a ~0.25 ms
delay *after* the last peer's stamp is seen and *before* the reduce is enqueued, so the switch that
inserts exactly that delay and nothing else — `SETTLE_US`, on the same device-wait arm; with `DEVWAIT`
it drains the device wait first, or the sleep lands before the arrival and opens no window at all —
put the failure rate on a dose-response:

| post-arrival delay | `batch_repeat_mismatches` failures | runs | `step_ms` | decode TPS |
|---|---|---|---|---|
| none, device wait | 10 | 10 | 53.7 | 18.61 |
| none, host poll — the shipped arm then | 0 | 10 | 77.1 | 12.97 |
| 20 us | 2 | 4 | 84.3-85.3 | 11.7-12.0 |
| 100 us | 1 | 4 | 94.7-95.4 | 10.5-10.6 |
| 500 us | 0 | 4 | 145.5-146.6 | 6.8-6.9 |

The delay is not a clean knob and the 20 and 100 us rows are four runs each, so the shape of the curve
is much weaker than its direction. What the direction is a direction *in* is the workspace race, not
the barrier: the delay sits in front of work the host then queues, so it changes what is in flight on
the stream when the rope table is uploaded, and a longer delay drains more of it. The 20 us arm running
at 84.3-85.3 ms — *above* the shipped arm's 77.1 ms — and failing 2 of 4 where the shipped arm fails 0
of 10 is consistent with that and is not evidence about a stamp. The same reading covers the numbers
above it: the shipped arm's 8 failures in 36 are the same race on a narrower window, the failures
tracking plane size is the failures tracking how the workspace is churned, and the step-0 flip shape
appearing in the hand-written arm and in none of HCCL's 36 is a difference in what the two arms queue.
With the table given its own slot the gate is failed 0 times in 22 runs across both arms, and the
device wait passes it 10 of 10 — see §5.5.3 and the RoPE table's workspace slot page (`#280`).

**The size ceiling, re-tested against the control it was set against.** The replacement is scoped to
planes of at most 40 960 FP16 elements, and that number came from a sweep read against "HCCL failed
none of 10 gate runs": 5120 clean in 16 of 17, 40 960 clean in 4 of 4, 81 920 failing 3 of 7,
163 840 failing 4 of 9. The control is now measured failing 3 times in 36, so the same logs have to be
read against that instead:

| plane | arm | `batch_repeat_mismatches` failures | runs |
|---|---|---|---|
| ≤ 40 960 | hand-written, rows=1 | 9 | 41 |
| > 40 960 | hand-written, rows=1 | 7 | 16 |
| 81 920 | hand-written, rows=16 | 2 | 3 |
| 40 960 | HCCL, rows=16 — the matched control | 0 | 3 |

7 of 16 against 9 of 41 is Fisher two-tailed p = 0.11, and the matched rows=16 pair is 2 of 3 against
0 of 3, p = 0.40. Displacing the hand-written baseline by the pooled 22% does not change the reading,
because the rows=1 arms below the ceiling already sit at 22% themselves. So the sweep does not
establish a bound: the direction is the same at every size above the ceiling, but no arm in it is large
enough to separate from a baseline that is itself this high, and one arm that ought to be worst — a
rows=1 run with the ceiling raised so the 32×5120 prefill plane also goes hand-written — fails 2 of 5,
which is not above the hand-written baseline at all. The ceiling is kept on three weaker grounds: the
plane the replacement exists for is two orders of magnitude inside it; every large-plane arm that has
been run leans the same way rather than contradicting; and raising it would change a shipped default's
behaviour on no evidence. It costs nothing that §5.5 measures, because rows=1 planes are 5120.

**This is a property of the stack, not of the replacement.** It was measured independently, on the
same checkpoint in a different phase, in
[the gated-delta slice work](ascend_gated_delta_slice.md#the-generated-tokens-are-not-a-usable-ab-signal-here):
four runs of two identical binaries over a 4966-token prompt produced three distinct greedy step-0
tokens and top logits spanning 10.42-10.94, "well above fp16 rounding", with the two runs either side
of the kernel swap no more alike than two runs of the same kernel. The failures above are the same
observation through a different gate. The gap that work left is the same one here: neither measurement
can drop the TP collective from the path — `QWEN_TP_WORLD=1` OOMs on this checkpoint — so both say
"not reproducible with a collective active" and neither can say it is the collective.

That paragraph is left standing as what was measured and is not relied on. The banner retraction above
re-opens it: the mechanism it needed to be independent of the collective is a race in a pooled
workspace slot that any phase uses, and the tree it was measured on had the same one. **That has since
been measured: the prefill instability does not survive the RoPE table's fix.** Eight runs of one
binary at the original 4966-token prompt length, 9 new tokens, the TP all-reduce enabled — one
identical step-0 token, one identical top logit, and one identical 9-token sequence across all eight,
where the pre-fix tree gave three distinct tokens in four runs. The repeat and the caveat that bounds
it are recorded in
[the gated-delta slice page](ascend_gated_delta_slice.md#the-generated-tokens-are-not-a-usable-ab-signal-here);
the short of it is that it shows the spread is gone on the fixed tree, not that the pooled slot was
the only thing producing it. The operational consequence it drew is recorded as rule 8 in
[the benchmarking rules](../guides/benchmarking.md), which is the right rule to keep either way:
establish run-to-run stability before comparing generated tokens across configurations.

**Four candidate mechanisms for the hand-written collective, tested and ruled out.** They were worth
testing while the failure looked collective-specific; with the failure present in both arms none of
them is load-bearing, and they are kept because each closes off a real hazard for the next person.

- *The arrival window.* A peer's stamp rides behind its plane on one stream, so a stamp that has
  landed ought to mean the plane before it has landed — unless two cross-device D2D copies issued on
  one stream do not complete in order at the destination. `POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US`
  empties exactly that window, sleeping between the poll succeeding and the reduce reading. 500 µs was
  tried; 3000 µs was tried again with the failure still in place. The arithmetic confirms the switch
  was doing what it says — 3000 µs across 129 calls is 387 ms, and the step goes from 77 to 472 ms.
- *Cache-line contention between the four stamp writers.* A stamp is two bytes and four processes
  write into every peer's stamp array concurrently, one slot each; at their natural spacing all four
  land inside a single 32-byte block, which is the hazard this backend has already been bitten by for
  scalar GM stores. `kStampStride` now pads each stamp out to its own 64-byte block. The rate does not
  move (4 of 20 padded against 2 of 6 packed). The padding stays, because the hazard it removes is
  real and it costs nothing measurable — 76.2-78.5 ms either way — but it is not the cause.
- *The null stream.* The validating probe always creates a stream; the collective runs on whatever
  the caller passed, and at all 129 engine sites that is nothing.
  `POCKET_ASCEND_IPC_ALLREDUCE_RSTREAM` substitutes a private stream with a drain on either side —
  the shape `resolve_stream` gives tp_comm's HCCL branch — and it costs half of what the replacement
  is worth (91.4 ms against 77.0 ms). It changes the failure's shape and not its rate.
- *The reduction's arithmetic.* The sum is three pairwise `aclnnInplaceAdd` accumulations in rank
  order into the accumulating buffer, with no atomics, no reduction tree and no racing accumulator,
  so two runs that receive the same three planes must produce the same sum bit for bit. An
  FP32-accumulator form was built on the theory that FP16 rounding was the perturbation; it made the
  measured deviation wider and cost 6.4 ms of a 77 ms step, and it was removed.

**What this leaves.** The replacement is worth 29.0 ms of a 106.3 ms step, +37.5%, and this section
gives no reason to prefer HCCL's reproducibility to its own. It shipped opt-in for the reason above —
a batched path that is not run-to-run reproducible is exactly the kind of defect that shows up in
production as a rare bad token rather than as an error, and the one number that would settle it, the
two arms' true rates at matched configurations, was not in hand. That reason is discharged rather
than answered: the rate was the RoPE table's workspace race and not the collective, 0 of 22 failures
with it fixed, so the switch has been flipped and **the hand-written collective is the backend's
default**. `POCKET_ASCEND_IPC_ALLREDUCE=0` is the way back to HCCL.

Two controls could not be run and are named so the gap is visible. `QWEN_TP_WORLD=1` — no collective
in the program — cannot host this checkpoint at all: one 32 GB card OOMs on
`model.language_model.layers.7.mlp.up_proj.weight`, so there is no collective-free arm, and the
batched path's reproducibility cannot be measured without a collective in it. And the isolated probe
of §5.4 never sees the failure at all: `bench_qwen_ascend_ipc_exchange --poll-wait` runs 20 000 rounds
with `slots_ok=4/4` and no mismatch, which is what makes the engine context rather than the barrier
the place the remaining answer is.


## 6. The Cube's M tile is sixteen rows and a decode step has one

§3 left the 42.3 ms of non-collective work named and unscoped, on the reading that rows=1 is what a
single request is and so not something a lever can act on. The row count is not the shape the
hardware sees. `bench_qwen_ascend_decode_ops` sweeps M over the two weight shards the layer's
projections actually use — the TP4 MLP shard (`4352 = 17408/4`) and its fused gate+up (`8704`) — with
the weight read once per call:

| M | 5120x4352 device us | GB/s | 5120x8704 device us | GB/s |
|---|---|---|---|---|
| 1 | 167.1 | 266.7 | 301.2 | 295.9 |
| 4 | 165.2 | 269.7 | — | — |
| 16 | 101.1 | 440.7 | 156.4 | 570.0 |
| 64 | 328.9 | 135.5 | — | — |

**Sixteen times the rows for 1.65x the rate**, and M=64 — four tiles of them — is slower than M=1.
That is not a compute curve: the weight bytes are the same at every row of that table, so what the M
axis is changing is the rate at which they are read, and the shape that reads them fastest is the one
that fills the tile's sixteen rows exactly. Against the 1148 GB/s the HBM probe reaches, a decode
step's 267-296 GB/s is the whole of the gap §3 attributed to "per-layer work".

### 6.1 The fifteen rows do not have to be real

A decode step has one row and no second sequence to put in rows 1-15. What makes the table above a
lever is that those rows do not have to contain anything: a zero stride on the activation's row
dimension describes a batch that repeats one row of storage, so nothing is copied, nothing is read
twice, and all sixteen outputs come back equal to the one the caller wanted. The bench checks both
halves of that at both widths — row 0 against a separate single-row product, and all sixteen rows
against row 0 — and reports `identical: yes` / `row0=ok all 16 rows=ok` throughout.

| shape | `matmul 1x` | `broadcast x 16x` (zero input stride) | `matmul 16x` (real rows) |
|---|---|---|---|
| 5120x4352 | 167.1 us / 266.7 GB/s | **112.0 us / 397.9 GB/s** | 101.1 us / 440.7 GB/s |
| 5120x8704 | 301.2 us / 295.9 GB/s | **149.0 us / 598.1 GB/s** | 156.4 us / 570.0 GB/s |

The broadcast form is within 11% of a genuine sixteen-row matmul at 4352 and *faster* than one at
8704. Filling the tile with copies is what buys the rate; the arithmetic that gets thrown away was
never what the unit was waiting for.

**The destination has to stay tall.** The other way to leave the caller's buffer untouched is to
collapse the output as well — a zero stride on the output's row dimension, so all sixteen writes land
on row 0 and the consumer reads the one row it always read. It is the tidier shape and it is a loss:

| shape | `broadcast x` (tall y) | `broadcast xy` (zero output stride) |
|---|---|---|
| 5120x4352 | 112.0 us, enqueue 13.7 us | 302.6 us, enqueue 115.3 us |
| 5120x8704 | 149.0 us, enqueue 14.0 us | 348.5 us, enqueue 109.9 us |

2.7x and 2.3x, and the enqueue column says where it comes from: a zero output stride turns the store
into a scatter the aclnn side cannot batch. The engine grows the allocation instead (§6.2), which
costs one `qwen_copy_rows_strided_f16` that it does not have to run: nothing downstream ever reads
row 1.

### 6.2 The engine: one rule at the allocator, one condition at the projection

Two changes, and both are shaped so that the batched path cannot see them.

**The allocation.** `allocate_elements` is the single point every activation allocation in the layer
passes through, so the rule lives there: a one-row FP16 activation is allocated with room for
`QWEN_ASCEND_REPLICATE_ROWS` rows. Its `shape` is deliberately left alone — every consumer in the
layer reads row 0 through the extent it reads today, so the copies are never observed and no
downstream kernel has to know they exist. Only the allocation grows.

"One row" is both spellings of it, a rank-1 shape and a rank-2 shape whose first dimension is 1,
because a rule that depended on how a call site punctuated its shape is exactly the gap a replicated
write turns into an overflow. Rank 3 and above is where the rule stops, and that boundary is the KV
cache rather than an artifact: a cache is `{slots, context, heads, dim}`, its append and attention
kernels index those strides themselves, nothing replicated ever writes one, and growing one would
cost `replicate` times the cache — 2.1 GB against 134 MB at an 8192-token context — to buy nothing.
No activation in the layer is spelled that way.

**The projection.** `Linear::forward` in `qwen_layer_components.inl` is the one site every one of the
layer's linear projections goes through — the 17 call sites for `lin.qkv`, `lin.ab`, `lin.a`/`lin.b`,
`lin.z`, `lin.out`, `full.q`/`full.kv`, `full.k`/`full.v`, `full.out`, `mlp.gate_up`,
`mlp.gate`/`mlp.up`, `mlp.down`, plus `mtp.fc` and the target head. It takes `replicate` rows and a
zero activation stride when the caller asked for one row, and the unchanged call otherwise:

```cpp
const int replicate = rows == 1 ? ascend_replicate_rows() : 1;
require_launch(qwen_fp16_matmul_rows_f16(
    input, linear.weight.f16_data(), output, replicate, output_rows,
    columns, replicate > 1 ? 0 : columns, output_rows, columns), ...);
```

`rows != 1` leaves `replicate` at 1, so prefill and batched decode take the identical path they took
before, by construction rather than by a check. The zero-stride case is not a special case in the
backend either: `aclnn_ops.cpp`'s `matmul_rows` accepts a zero stride on either operand's row
dimension as a broadcast, and every other stride it validates is unchanged.

### 6.3 What it is worth: 13.05 -> 17.59 TPS at rows=1

Three interleaved pairs, one arm alternating with the other inside one session, the TP4 launcher, one
process per rank on devices 0-3, the hand-written collective on in both arms so that what is
measured is this lever and not §5's:

| pair | `QWEN_ASCEND_REPLICATE_ROWS=1` | `=16` | step, rep=1 | step, rep=16 |
|---|---|---|---|---|
| 1 | 13.1333 TPS | 17.4303 TPS | 76.9-76.5 ms | 56.9-56.3 ms |
| 2 | 13.0327 TPS | 17.6771 TPS | — | — |
| 3 | 12.9902 TPS | 17.6566 TPS | — | — |

**13.05 -> 17.59 TPS, +34.8%**, and 76.6 ms -> 56.5 ms per step. Both arms exit
`qwen_ascend_tp4_status=0` on all four ranks. The control band is 12.99-13.13 across the session and
the replicated band is 17.43-17.71, so the separation is far outside the spread — unlike §5's
collective A/B, this one does not need thirty-six pairs to be legible.

That session predates the RoPE-table fix, so it is a timing result and nothing more: it makes no
claim about the tokens, which is the claim that later turned out to be wrong. The bands survive it —
the merged-tree matrix further down measures the same two, 12.96-13.06 and 17.46-17.71 — and it is
there, not here, that the widths are shown to generate what the default generates.

The width is swept rather than assumed, and 16 is where it stops:

| `QWEN_ASCEND_REPLICATE_ROWS` | 1 | 8 | 16 | 32 | 1 | 16 |
|---|---|---|---|---|---|---|
| decode TPS | 12.927 | 12.4983 | **17.6432** | 16.8139 | 12.7466 | **17.6649** |

8 is *below* the control and 32 is below 16, which is the tile boundary showing through: the unit
has sixteen rows and any count that is not a multiple of sixteen pays for a partly-filled tile, while
32 pays for two of them at a rate (321.7 GB/s, `replicated 32x5120x4352`) that does not cover the
second. The bench's own `replicated {4,8,16,32}` row — the same matmul with the sixteen rows actually
materialised on the host — is 166.1 / 166.0 / 101.0 / 138.5 us, the same shape of curve, and it is
what makes the sweep's 8 and 32 rows a hardware result rather than a scheduling one.

The cost is workspace, and it is small because only one-row activations grow: `activation_workspace_peak_bytes`
is 1825704 at the default and 3094824 at 16, +1.2 MB, against 13.45 GB of resident weights per card.
`gpu_memory_used_bytes` moves by the same 4 MB over four ranks.

**The accuracy question, since this is the width that exposed one.** An earlier revision of this
page could not show that `QWEN_ASCEND_REPLICATE_ROWS=16` generated what the default generated: three
runs produced three different first tokens where widths 2, 4 and 8 were stable. That was not the
replication. It was the partial-RoPE cos/sin table sharing a workspace slot with an attention kernel
the host had already queued -- the table is written by a blocking `aclrtMemcpy` H2D, which the stream
does not order against work queued before it, so a kernel issued ahead of the upload lands on the
table between the copy and the rotation's read. Replication changed the timing of that window rather
than inventing it. The table has a slot of its own now
([the RoPE table's workspace slot](ascend_rope_table_workspace_aliasing.md), #280).

The corruption was localized independently of that attribution, from this branch's own dumps, before
the fix was in. At `QWEN_DUMP_KV_LAYER=3`: the pre-norm K projection is **bit-exact** between widths
1 and 16, while the post-RoPE K and Q differ by up to 7.61 in exactly the 32-byte lanes `[0..7]` and
`[32..39]` of the 256-wide row. The divergence opens between the projection and the rotation, which
is inside RoPE and not in the projection the lever changes.

With the table in its own slot, one session, the TP4 launcher, one process per rank on devices 0-3:

| arm | `QWEN_ASCEND_REPLICATE_ROWS=1` | `=16` |
|---|---|---|
| HCCL (the shipped collective) | 9.63 TPS / 103.8 ms | 9.76 TPS / 102.5 ms |
| host poll (hand-written collective) | 12.96-13.06 TPS / 76.6-77.1 ms | 17.46-17.71 TPS / 56.5-57.3 ms |
| device wait (same collective, §5.5.3) | 18.66-18.79 TPS / 53.2-53.6 ms | 25.15-25.37 TPS / 39.4-39.8 ms |

All ten runs emit `11751 13 198 760 6511 314 9564 369 19241 ...` -- the reference's 32 tokens, one
sequence, at both widths, at both collectives, and at both of the hand-written collective's arrival
waits. The four hand-written configurations were then repeated at 128 tokens, two runs each: all
eight emit one identical 128-token sequence, whose first 32 are the reference's. The lever is worth
the same 13.05 -> 17.59 TPS with the accuracy intact; it is not paid for with it.

An earlier revision of this page quoted two pairs at 13.0088 / 17.5931 and 12.9874 / 17.2694 TPS and
claimed the reference's tokens for all four runs. That pair is **withdrawn rather than corrected**:
it is not reproducible from the tree it is printed against, because `Linear::forward` was handing the
replication factor to a multi-row projection as the batch at the time, so a prefill chunk projected
row 0 alone and the first token could not have been the reference's. The session above replaces it.
The 34.8% this section is named for survives the correction unchanged; what did not survive is the
claim that the tokens were already right.

### 6.4 Where the 20.1 ms went

`QWEN_PHASE_PROFILE` brackets every scope with a `device_synchronize`, so its numbers are device work
at that site and its scopes are fully serialized; that means the sum of the scopes is larger than the
step and only the deltas between the two arms are readable. Seven decode steps on rank 0, both arms
in one session:

| scope | rep=1 ms | rep=16 ms | delta |
|---|---|---|---|
| `pd.mlp.gate_up` | 137.38 | 86.33 | **-51.05** |
| `pd.mlp.down` | 103.96 | 60.89 | **-43.07** |
| `pd.lin.qkv` | 54.72 | 41.13 | -13.59 |
| `pd.lin.out` | 45.87 | 36.40 | -9.47 |
| `pd.lin.z` | 47.70 | 38.97 | -8.73 |
| `pd.full.q` | 19.66 | 14.15 | -5.51 |
| `pd.full.out` | 15.77 | 12.53 | -3.24 |
| `pd.lin.ab` | 42.56 | 44.15 | +1.59 |
| `pd.full.k` / `pd.full.v` | 12.86 / 12.68 | 12.76 / 13.07 | -0.10 / +0.39 |
| **all `pd.*` scopes** | **493.16** | **359.39** | **-133.77** |
| `full_attention` | 184.62 | 177.15 | -7.47 |
| `attn_resid_norm` | 37.27 | 40.83 | +3.56 |
| `tp_all_reduce` (the collective) | 261.44 | 245.03 | -16.40 |
| `STACK.d` (the step) | 1188.34 | 1051.69 | -136.65 |

**-133.77 of the -136.65 ms is the projections**, and the collective moves by -16.40 ms, which is
inside the run-to-run spread these arms already show (an earlier session of the same pair put it at
+5.50). The lever does what it says and touches nothing else.

`pd.lin.ab` is the one projection that does not move, and it is the one that should not: it is a
gated-delta site whose input row is produced at the width the fused path expects, and its `+1.59` is
the same order as `attn_resid_norm`'s `+3.56`. Both are the cost of the larger activation
allocations, and both are a tenth of what the two MLP projections return.

The host profile says the same thing from the issuing thread's side, with no synchronization at all.
Its `tp_all_reduce` scope falls 123.91 ms over the same seven steps, which is not the collective
getting faster — the peers' arrival work is unchanged — but the blocking `memcpy_d2h` in the poll
waiting out less queued layer, the mechanism §5.5.3 already identified. The projections' own host
cost falls 2.4-3.7 ms each. Neither number is a share of the step; they are what the thread that
issues the step spends, and the step itself is §6.3.

### 6.5 The gate that proves the rule fired

The rule of §6.2 is stated over shape rank and dtype, and a destination spelled some third way would
be sized for one row and then written sixteen deep, with nothing else in the engine noticing. How
much room a destination has is also the one thing the projection cannot work out for itself: it is
handed a bare pointer, and whether that buffer holds one row or sixteen was decided at an allocation
somewhere else.

`QWEN_ASCEND_REPLICATE_CHECK=1` closes that. Every activation remembers its **declared extent** at
`allocate_elements` — not the block's capacity, which is a workspace slot's largest-ever extent and
therefore a bound on what the hardware can take rather than on what this write was sized for. Every
replicated projection looks its destination up and declines to launch if the tall write does not fit,
naming the site and both extents. A buffer that reached the device through a bare `allocate` rather
than through `allocate_elements` is not in the map and is refused rather than assumed fine, which is
the same failure the check exists to catch. It is off by default — a hash lookup per allocation, and a
check on the code rather than on the hardware.

Both controls were run. With the rule in place and `QWEN_ASCEND_REPLICATE_CHECK=1`, `replicate=16`
completes at **17.5072 TPS, `qwen_ascend_tp4_status=0`**: every one of the 17 sites found a
destination with room. With the rule temporarily disabled (`ascend_replicated_elements` returning
its argument) and the check still on, the run fails at the first projection:

```
error: Qwen replicated projection at lin.qkv has 49152 destination bytes for 81920
qwen_ascend_tp4_status=1
```

The second extent is `replicate x output_rows x 2` at that site: 16 x 2560 x 2 = 81920 B, where 2560
is `lin.qkv`'s output width — `2 * key_dim + value_dim` at TP4, i.e. `2 * (16/4) * 128 +
(48/4) * 128`. The first is what the slot held with the rule off: less than one replicated row, so
the launch is refused instead of writing past the end of it. The rule was restored, the engine
rebuilt, and the A/B re-measured; the matrix in §6.3 is from that restored build merged with the
RoPE-table fix, and the projection's batch width was corrected in the same pass.

What this gate does not establish is that the sixteen rows carry the right *numbers*. It verifies the
room, not the arithmetic. The rest of the arithmetic is covered elsewhere rather than here: the
broadcast's parity at both widths is the bench's own check (§6.1), the extent is what this gate
checks, and the tokens are what §6.3 has — the two arms, and the widths either side of them, emitting
one 32-token sequence that is the HF reference's. Before the workspace fix that third leg was not
available and this paragraph said so; it is available now. The bench grew a fourth leg with it: each
batch width's row 0 against the same product accumulated in double on the host, where `m=1` and `m=16`
both come out at a max relative error of 3.45e-4 of scale on `4352x5120` and 3.24e-4 on `8704x5120`.
The broadcast is the single-row form's arithmetic, to the figures the measurement resolves.

Three further things are not established, and are named so they are not read as done. The
`full_attention` and `attn_resid_norm` deltas above are a single session's, on a step that has 112
attention calls in it, and the interaction between a 1.2 MB larger workspace and the attention
kernels' own allocations is not separated from noise. The HCCL row in §6.3 is the arm
`POCKET_ASCEND_IPC_ALLREDUCE=0` reaches and is measured at both widths — 9.63 and 9.76 TPS, +1.3% —
so the lever is nearly free there but nearly
worthless too: a 103.8 ms step leaves the ~20 ms the tile buys buried in the collective. The 34.8%
§6.3 reports is this lever on top of the hand-written path, which the flip has since made the
default, and not on the HCCL arm. And the fused versus
unfused `mlp.gate_up` choice the engine makes from the checkpoint's own layout is not swept against
`QWEN_ASCEND_REPLICATE_ROWS`; only the layout this checkpoint selects (fused, 8704) is measured
here.

## 7. What is left, in order of size

| lever | measured size | state |
|---|---|---|
| Replace the 129 collectives with the hand-written one | 29.0 ms of 106.3, i.e. 9.41 -> **12.94 TPS**, interleaved (§5.5.4) | taken; the backend's default since the flip, with `POCKET_ASCEND_IPC_ALLREDUCE=0` as the way back to HCCL. The gate failures this row used to cite (3 of 36 HCCL, 8 of 36 hand-written) were a workspace race elsewhere and are 0 of 22 with it fixed, so the gate no longer argues either way |
| The poll the hand-written collective still does | 26.4 of the 27.6 ms it costs over the collective-free floor (§5.5.3); 23.4 of a 77.1 ms step as the host round trip, measured against a device-side wait on the same stamps | taken; the device-side form is 30% faster, 53.0-53.9 ms and 18.56-18.86 TPS, and passes the gate 10 of 10 once the RoPE table has its own workspace slot (§5.5.3), and its own three-arm ladder put it 1.22x/1.08x clear of the host poll at 16 and 112 serving rows, so it is the backend's default in turn, with `POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0` as the way back ([serving concurrency](serving_throughput_scaling.md)) |
| ~~The arrival signal has no release ordering~~ | **retracted**: the 10-of-10 rate that exposed it was the RoPE table's workspace aliasing, and it goes to 0 of 10 without the barrier changing at all (§5.5.3, §5.5.4) | withdrawn |
| The bracket that was eating half of it | 12.9 ms of a 91.5 ms step, 0.100 ms/call (§5.5.1) | removed; the predicate that scopes it is now part of the contract |
| The 42.3 ms of non-collective per-layer work | 133.8 of the 136.7 ms step saving, i.e. 13.05 -> **17.59 TPS**, interleaved (§6.3, §6.4) | taken for the one-row activations; opt-in with `QWEN_ASCEND_REPLICATE_ROWS=16`. What is left of the 42.3 ms is not priced here |
| MTP / speculative decoding | **-2.2x** | measured, ruled out on this checkpoint (§4.1) |
| Verifier placement | 0.08%, noise | retracted (§4.2) |
| The row axis | 1.17x the time for 6.8x the throughput | already the shipped answer, rows=16+ |

## 8. Reproducing this

The engine numbers come from the repository's TP4 launcher, one process per rank on devices 0-3:

```bash
# the rows=1 arm of §1
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=0 QWEN_BATCH_PROMPT_LEN=32 \
  scripts/run_qwen_ascend_tp4.sh "" 8
```

The §5.5 A/B is the same launcher with the hand-written collective on one arm and off the other,
`QWEN_BATCH_VERIFY=3` so that both arms are gated. The `=1` arm is what an unset environment runs
now that the collective is the default, and it is spelled out here because the measurement names
both arms:

```bash
# HCCL arm, then the hand-written collective (the default since the flip)
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=3 QWEN_BATCH_PROMPT_LEN=32 \
  POCKET_ASCEND_IPC_ALLREDUCE=0 scripts/run_qwen_ascend_tp4.sh "" 8
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=3 QWEN_BATCH_PROMPT_LEN=32 \
  POCKET_ASCEND_IPC_ALLREDUCE=1 scripts/run_qwen_ascend_tp4.sh "" 8
```

The §5.5.3/§5.5.4 device wait is the same again, and its dose-response rows add
`POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US`. The collective is named there because the wait stacks on
it. It was opt-in when these rows were measured and is the backend's default now, so unset runs
the pair and only the `=0` arm needs spelling out:

```bash
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=3 QWEN_BATCH_PROMPT_LEN=32 \
  POCKET_ASCEND_IPC_ALLREDUCE=1 POCKET_ASCEND_IPC_ALLREDUCE_STATS=1 \
  POCKET_ASCEND_IPC_ALLREDUCE_DEADLINE_MS=3000 \
  POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0 [POCKET_ASCEND_IPC_ALLREDUCE_SETTLE_US=100] \
  scripts/run_qwen_ascend_tp4.sh "" 8
```

The `=0` spelling is the host-poll arm those rows compare against; dropping it runs the device wait
those rows measure. Everything else on the line is a diagnostic and is still opt-in.

The §6 A/B is the plain single-request path on both arms — 5-token prompt, 8 new tokens — with the
row replication on one arm and off the other, interleaved and repeated. It was measured on the
hand-written collective, which the `POCKET_ASCEND_IPC_ALLREDUCE=1` below now spells out only to pin
it; unset would run the same arm:

```bash
# §6.3, three interleaved pairs against the same build
for pair in 1 2 3; do
  for rep in 1 16; do
    env QWEN_ASCEND_REPLICATE_ROWS=${rep} QWEN_ASCEND_DEVICES=0,1,2,3 \
        POCKET_ASCEND_IPC_ALLREDUCE=1 \
        scripts/run_qwen_ascend_tp4.sh "The capital of France is" 8
  done
done

# §6.5's gates: the same with the check on, and the same one-row width the
# negative control failed at
QWEN_ASCEND_REPLICATE_ROWS=16 QWEN_ASCEND_REPLICATE_CHECK=1 \
  POCKET_ASCEND_IPC_ALLREDUCE=1 scripts/run_qwen_ascend_tp4.sh "The capital of France is" 8
```

The MTP A/B of §4.1 is the same launcher without `QWEN_BATCH_ROWS`, plus `--qwen-mtp-tokens 3` on
one arm:

```bash
cpp_engine/build-ascend/pocketllm_engine \
    --ckpt /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
    --smoke-forward --smoke-layers 0 --resident-bench --max-new-tokens 8 \
    --tp-world 4 --tp-rank R --device R --nccl-id-path <shared> \
    --prompt "The capital of France is" [--qwen-mtp-tokens 3]
```

The four benches below are separate processes and need the four devices to themselves. They are
benches rather than tests, so `scripts/build_ascend.sh` does not build them; build each by name, and
`source scripts/ascend_env.sh` first — an ACL binary launched without it does not fail, it hangs
before `aclInit` returns.

```bash
cmake --build cpp_engine/build-ascend -j --target \
    bench_qwen_ascend_allreduce bench_qwen_ascend_peer_copy \
    bench_qwen_ascend_ipc_exchange bench_qwen_ascend_decode_ops
```

```bash
# the per-call price, the size sweep, and the platform floor of §2
cpp_engine/build-ascend/tests/bench_qwen_ascend_allreduce \
    --nccl-id-path <shared> --tp-world 4 --tp-rank R --device R --iters 100

# the primitives in one process, §5.1
cpp_engine/build-ascend/tests/bench_qwen_ascend_peer_copy --device 0

# the primitives across processes, §5.2/§5.3 -- four processes, and
# --no-pid-check is required or every import returns 507899
for r in 0 1 2 3; do
  cpp_engine/build-ascend/tests/bench_qwen_ascend_ipc_exchange \
      --world 4 --rank $r --device $r --dir <shared> --iters 200 --no-pid-check &
done; wait

# the poll barrier of §5.4, and the one-buffer layout it replaces
for r in 0 1 2 3; do
  cpp_engine/build-ascend/tests/bench_qwen_ascend_ipc_exchange \
      --world 4 --rank $r --device $r --dir <shared> --no-pid-check \
      --poll-wait [--poll-single-set] --iters 20000 &
done; wait

# the HBM read probe of §3, the M/width sweeps of §6.1, and the broadcast
# shapes the replicated launch is
cpp_engine/build-ascend/tests/bench_qwen_ascend_decode_ops --device 0
```

`bench_qwen_ascend_ipc_exchange` exports all `world` slots of its own buffer and imports the slot
that belongs to *this* rank out of every peer, so `slots_ok=4/4` means slot `k` holds rank `k`'s
value on every rank. A rank's own slot is legitimately `0x0000`: nothing pushes into it, because the
local term of the sum never leaves the rank.

## 9. Files

- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — the per-call price, the size sweep, the
  platform floor, and the overlapped-vs-serial comparison, §2.1-§2.3.
- `cpp_engine/tests/bench_qwen_ascend_peer_copy.cpp` — the primitives in a single process and the
  first host-driven all-reduce prototype, §5.1.
- `cpp_engine/tests/bench_qwen_ascend_ipc_exchange.cpp` — the cross-process export/import probe, the
  four notify barrier shapes, and the payload-carried poll barrier with its one-buffer control,
  §5.2-§5.4.
- `cpp_engine/tests/bench_qwen_ascend_decode_ops.cpp` — the HBM read probe behind the 85 TPS
  ceiling (§3), and the M sweep, the replicated-launch parity checks, and the broadcast/broadcast-xy
  comparison of §6.1.
- `cpp_engine/engine/qwen_engine.cpp` — `all_reduce_half_rows`, the one-wide-collective decision that
  makes 129 the count; and, for §6, `ascend_replicate_rows`, the allocation rule
  `ascend_replicated_elements`, and the `QWEN_ASCEND_REPLICATE_CHECK` gate.
- `cpp_engine/engine/qwen_layer_components.inl` — `Linear::forward`'s Ascend branch, the single site
  the replicated projection is issued from, §6.2.
- `cpp_engine/backends/ascend/kernels/aclnn_ops.cpp` — `matmul_rows`' stride guard, which is where a
  zero on either operand's row stride is admitted as a broadcast rather than rejected as a short
  pitch, §6.2.
- `cpp_engine/engine/main.cpp` — `--batch-decode`, `QWEN_BATCH_ROWS`, and the `batch_decode=1` line
  §1 is read from.
- `scripts/run_qwen_ascend_tp4.sh` — the TP4 launcher used for every engine number here.
