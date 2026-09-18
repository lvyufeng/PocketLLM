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
all-reduce at **0.26 ms against `HcclAllReduce`'s 0.4810** (§5.4). So the 129 collectives are 62.0 ms
of the step today and 33.4 ms of it if they are replaced — 28.6 ms saved, and 9.6 TPS against **13.2**.
That replacement is a design with a measured core, not a shipped one: §5.4 says what is measured and
what is still assumption.

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
free and the step is 42.3 ms, or 23.6 TPS.** That is a 2.5x on single-request throughput, and it is
the entire remaining headroom on this axis — the rest of the gap to 85 TPS is in the 42.3 ms of
per-layer work, which is a different project.

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

Against `HcclAllReduce`'s 0.4810, that is **1.9x cheaper per call** — 129 calls go from 62.0 ms to
33.4 ms, and the step from 104.3 ms to 75.7 ms, or **9.6 TPS to 13.2 TPS**.

What this is not yet: the loop around it is empty, so the ranks stay within a fraction of a round of
each other by construction. The engine has real work between collectives, and the design that follows
from these numbers — one pair of buffers per rank rotating through a global round counter, exported
once at startup through the rendezvous the HCCL id already uses, with the reduce staying on the device
— has not been built or measured. The 0.259 also includes a trailing `stream_synchronize` that the
engine may not need, since the next layer's work is enqueued on the same stream and only the device
reads the result; that variant is untested and would be cheaper, not more expensive. What is settled is
the thing §5.3 could not settle: the barrier itself, at world=4 on four processes, costs less than the
collective it would replace.

## 6. What is left, in order of size

| lever | measured size | state |
|---|---|---|
| Replace the 129 collectives with the hand-written one | 129 x (0.4810 - 0.259) = **28.6 ms** of 104.3, i.e. 9.6 -> **13.2 TPS** | barrier measured (§5.4); engine integration not built |
| The 42.3 ms of non-collective per-layer work | 3.6x above the 11.7 ms memory floor | not scoped on this page |
| MTP / speculative decoding | **-2.2x** | measured, ruled out on this checkpoint (§4.1) |
| Verifier placement | 0.08%, noise | retracted (§4.2) |
| The row axis | 1.17x the time for 6.8x the throughput | already the shipped answer, rows=16+ |

## 7. Reproducing this

The engine numbers come from the repository's TP4 launcher, one process per rank on devices 0-3:

```bash
# the rows=1 arm of §1
QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=0 QWEN_BATCH_PROMPT_LEN=32 \
  scripts/run_qwen_ascend_tp4.sh "" 8
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
```

`bench_qwen_ascend_ipc_exchange` exports all `world` slots of its own buffer and imports the slot
that belongs to *this* rank out of every peer, so `slots_ok=4/4` means slot `k` holds rank `k`'s
value on every rank. A rank's own slot is legitimately `0x0000`: nothing pushes into it, because the
local term of the sum never leaves the rank.

## 8. Files

- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — the per-call price, the size sweep, the
  platform floor, and the overlapped-vs-serial comparison, §2.1-§2.3.
- `cpp_engine/tests/bench_qwen_ascend_peer_copy.cpp` — the primitives in a single process and the
  first host-driven all-reduce prototype, §5.1.
- `cpp_engine/tests/bench_qwen_ascend_ipc_exchange.cpp` — the cross-process export/import probe, the
  four notify barrier shapes, and the payload-carried poll barrier with its one-buffer control,
  §5.2-§5.4.
- `cpp_engine/tests/bench_qwen_ascend_decode_ops.cpp` — the HBM read probe behind the 85 TPS
  ceiling, §3.
- `cpp_engine/engine/qwen_engine.cpp` — `all_reduce_half_rows`, the one-wide-collective decision that
  makes 129 the count.
- `cpp_engine/engine/main.cpp` — `--batch-decode`, `QWEN_BATCH_ROWS`, and the `batch_decode=1` line
  §1 is read from.
- `scripts/run_qwen_ascend_tp4.sh` — the TP4 launcher used for every engine number here.
