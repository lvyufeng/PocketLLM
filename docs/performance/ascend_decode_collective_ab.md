# Ascend Decode: The All-Reduce A/B, The Event Probe, And Batch Scaling

Qwen3.8-27B, 4 x Ascend 910 first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0,
checkpoint `/mnt/data1/modelscope/Qwen/Qwen3.8-27B`. Every number below is a measurement on that
machine; the command that produced it is named in the same section.

`docs/performance/ascend_attention_optimization.md` measures the decode step at **9.2 TPS**
(108.2 ms) and attributes 129 TP all-reduce calls per token — 64 `ar.mlp`, 48 `ar.lin.out`,
16 `ar.full.out`, 1 `ar.hidden_a` — to a per-call `stream_synchronize` in `end_nccl_collective`.
Two routes follow from that attribution, and this page records what each one actually bought:

1. **Hide the collectives.** Issue them from a second stream and order their consumers with events.
   This is the textbook construction, it was built, and it does not work here — §3 and §4 give the
   ordering table that rules it out and the probe that says why the failure is not in the event.
2. **Amortise the step over rows.** The collective cost is per-call latency, so it does not grow with
   the batch, while the work it is interleaved with does. This is the route that reaches the target:
   **113.3 TPS at rows=16** against a 70 TPS goal, 214.2 TPS at rows=64. §6.

The second result does not make the first one uninteresting. It is what says the two are not
alternatives: the collectives are 57-65 ms of a step that is 107 ms at rows=1 and 299 ms at rows=64,
so their share of the step falls as the batch grows, and the batching route is bounded by the
per-row work that remains.

## 1. Where the step's time goes

The layer ladder from the attention page, re-read as what it implies about a batch:

| layers | decode TPS | ms/step |
|---|---|---|
| 1 | 105.7 | 9.5 |
| 4 | 67.2 | 14.9 |
| 8 | 58.8 | 17.0 |
| 16 | 33.5 | 29.9 |
| 32 | 18.1 | 55.4 |
| 64 | 9.24 | 108.2 |

A linear fit gives **~7.8 ms fixed per step + ~1.63 ms per layer**. The fixed term is per-step, not
per-layer, and the per-layer term contains both the weight streaming and the collectives. Both
terms are candidates for amortisation over a batch; only the first is free of per-row work.

The 129 collectives at the standalone 0.3923 ms per call measure **50.6 ms** of the 108.6 ms step,
and the direct ablation agrees: removing the collective calls entirely from an otherwise unchanged
step moves 139.7 -> 74.2 ms at rows=16, 188.0 -> 124.6 ms at rows=32 and 298.8 -> 241.4 ms at
rows=64. Those three deltas — 65.5, 63.3 and 57.4 ms — are the same number three times, which is
what a per-call cost looks like. The ablation was measured through a temporary `POCKET_TP_AR_NOOP`
gate in a scratch build; the gate is not in this tree and the numbers are recorded here rather than
left reproducible, because the same ablation is also the easiest way to make the engine produce
wrong tokens. It does: with the reductions removed every row decodes something else, and the harness
reports it (`verify_mismatches=16/48`).

## 2. The collective is host latency, not device time

`cpp_engine/tests/bench_qwen_ascend_allreduce.cpp`, 4 processes on devices 0-3, `--iters 100`,
one rank per process, 5120-fp16 elements (10 KB):

| variant | 10 KB | 80 KB | 640 KB |
|---|---|---|---|
| `sync` (issue and synchronize, as `end_nccl_collective` does) | 0.4017 ms | 0.4286 ms | 0.4495 ms |
| `enqueue` (host issue loop, event bracket, no synchronize) | 0.3965 ms | 0.4653 ms | 0.4564 ms |
| `pure-enq` (the same with no event bracket at all) | 0.3530 ms | 0.4058 ms | 0.4090 ms |
| `host-med` (median of the per-call issue times) | 0.3863 ms | 0.4514 ms | 0.4504 ms |
| `drain` (whatever is left after the issue loop, all 100 iterations) | 1.85 ms | 2.36 ms | 2.33 ms |

`enqueue` sits within 0.010 ms of `host-med` and `pure-enq` within 0.003 ms of `pure-med`, so the
event bracket is not what costs: with it removed entirely the host still spends 0.353 ms inside the
call. `HcclAllReduce` **blocks the calling host thread for ~0.35 ms per call**; there is no queue to
hand the work to and return from, and the 1.85 ms of drain after 100 issued calls says the device
never falls behind the host. Against a device-op floor of 0.0180 ms measured on the same part
(`memset-10kb+sync`, `sync-only` 0.0019 ms), **95% of the collective is host time inside the call**.

That is what makes the second-stream construction attractive: if the host is blocked but the device
is not, a communication thread could carry the block while the compute thread runs layer work. The
bench's overlap test asks the question directly rather than assuming it — a second thread issues a
real Cube GEMM (`1024x4352x5120`) on its own stream throughout the collective loop:

```
allreduce rank=0 overlapped ar=  0.3982 ms  gemm-alone=  0.6409 ms  gemm-concurrent=  0.3943 ms  n=101
```

101 GEMMs completed inside the 39.8 ms the 100 collectives took, i.e. **0.394 ms each against
0.641 ms standalone**. The concurrent rate coming out *faster* than the standalone one is a host-side
artefact rather than a contention effect — the standalone loop queues 20 GEMMs behind a single
`device_synchronize` while the side thread synchronizes after each one — but whichever way that
resolves, the direction is what matters here: a collective that occupied the Cube would slow the side
thread down, and it did not. **The device has room during the collective; the 0.35 ms is host
latency.** So the construction §3 attempts is not ruled out by device contention, and its failure
has to be explained some other way.

## 3. The async overlap attempt

The attempt is on record as a set of orderings, each measured end-to-end on the TP4 engine with the
token-parity oracle (`POCKET_BATCH_VERIFY`, which compares each row's decoded token against the
synchronous reference). SL=4 rows=1 except where noted.

| ordering | oracle result |
|---|---|
| synchronous (control) | `mismatches=0, cs=0` |
| async, no barrier | `mismatches=1, cs=0.0242121` |
| barrier=1 (wait for the collective to be issued) | `mismatches=1, cs=0.061088` |
| barrier=3 (issue-thread sync after) | `mismatches=1, cs=0.309403` |
| barrier=4 (issue-thread sync before and after) | SL=4: `mismatches=0, cs=0.143357`; wide (rows=4, 12 verify steps, 48 comparisons): `mismatches=4, cs=0.389465`, repeat `cs=6.74798e+06` |
| barrier=2 (drain the communication stream) | SL=4: `mismatches=0, cs=0.00310186`; wide: `mismatches=0, cs=0.0714827`, `worst_logit_abs=0.461189` — bit-identical to the synchronous control |
| WINDOW=64/4096 event-ring size, SL=8 | `mismatches=1` |

Only the ordering in which the **compute thread host-synchronises the communication stream** before
reading the reduced values is correct, and that is exactly what `end_nccl_collective` already does.
The machinery therefore buys nothing: every correct ordering is the synchronous path with extra
steps, and every ordering that removes the host wait is wrong at every depth and every event-ring
size tried.

## 4. The event primitive itself is sound

A wrong result through an event can mean the event is not a dependency, or it can mean the event is a
dependency and the collective is not ordered by it. `cpp_engine/tests/bench_qwen_ascend_event_order.cpp`
separates the two. It produces a buffer with a real Cube matmul (a memset is far too fast to lose a
race), records an event, and consumes through a second stream, 40 iterations:

| ordering | consumer | stale |
|---|---|---|
| `comm-wait kernel` — producer on the secondary stream, consumer on the default stream | aclnn kernel | **0** |
| `default-wait kernel` — the same pair with the streams exchanged | aclnn kernel | **0** |
| `host-synced kernel` — the control, host waits on the event before enqueue | aclnn kernel | **0** |
| `comm-wait copy` — producer on the secondary stream, consumer a D2D copy | `aclrtMemcpyAsync` | 40 |

The event is sound in **both** directions, which is the direction the second-stream construction
needs. The copy row is not a reading about events: `aclrtMemcpyAsync` does not support
device-to-device on this part, so that consumer reports stale every iteration by simply not being a
transfer, which is why the probe keeps both consumers and labels them.

So the defect behind §3 is not the event and not the consumer. It is what the collective makes
visible to the event — `HcclAllReduce` returning before the reduction is complete, with the
completion only observable through the host synchronize that the synchronous path performs anyway.

Read against §2, the negative result is sharp rather than vague: the 0.35 ms is host latency, the
device is idle while it is paid, and the whole cost is therefore of the kind a second thread exists
to hide — and hiding it is nonetheless wrong at every ordering tried. The removable-in-principle cost
is not removable in fact on this CANN/910 pair, which is why §6 is the route that reaches the target
rather than a fallback from it.

## 5. What the A/B rules out

- **A communication thread that hides the collective.** Seven orderings, none correct without the
  same host wait the synchronous path already does.
- **A larger event ring.** `WINDOW=64` and `WINDOW=4096` both fail at SL=8.
- **Blaming the event or the cross-stream dependency.** The probe says both are sound.
- **A faster collective.** 10 KB and 640 KB both measure 0.3923/0.3989 ms — flat in payload, so it
  is latency. `HCCL_BUFFSIZE`, `HCCL_MULTI_QP_THRESHOLD` and `HCCL_INTRA_PCIE_ENABLE` were each
  tried and each left it within noise of 0.39 ms. `HCCL_ALGO=HD` and `NHR` **hang** on this stack;
  do not retry them.
- **Fewer collectives as the whole answer.** Merging them removes at most 50.6 ms of a 108.6 ms step,
  which is 17 TPS at rows=1 even if the merge were free — see the roadmap's own correction of the
  20-23 TPS figure.

## 6. The route that reaches the target: batch scaling

`--batch-decode --max-batch-size N`, full 64 layers, 32-token prompt, TP4, one row per arena slot,
each row's KV cache and recurrent state pinned to that slot for the whole run. The per-row prompt is
a rotation of one 32-token list, so no two rows decode the same sequence — see the seeding note at
the end of §6.1 for why that matters to the oracle.

| rows | TPS | ms/step | ms/row marginal |
|---|---|---|---|
| 1 | 9.35 | 107.0 | — |
| 16 | 114.5 | 139.7 | ~2.0 |
| 32 | 170.2 | 188.0 | ~3.0 |
| 48 | 202.2 | 237.4 | ~3.1 |
| 64 | 214.2 | 298.8 | ~3.8 |
| 96 | 181.8 | 528.1 | ~7.2 |
| 128 | 200.0 | 639.8 | ~3.5 |

Re-run of the low half of the same table on the cleaned tree — the measurement scaffolding removed,
16 steps rather than 8 — gives 9.35 / 17.9 / 35.7 / 65.1 / **112.8** / **169.4** TPS at rows
1 / 2 / 4 / 8 / 16 / 32, i.e. 107.0 / 111.5 / 111.9 / 122.9 / 141.8 / 188.9 ms. The two runs agree
within the spread this engine shows run to run — 114.5 and 112.8 at rows=16, 170.2 and 169.4 at
rows=32 — and the step cost is nearly flat from rows=1 to rows=8 (107 to 123 ms for eight times the
rows), which is the fixed per-step half being shared.

A third set of runs at rows=16 on the tree as it stands now — three consecutive runs, same command,
`QWEN_BATCH_VERIFY=16 QWEN_BATCH_ROWS=16 scripts/run_qwen_ascend_tp4.sh "" 16` — gives 113.295,
113.873 and 112.197 TPS at 141.2, 140.5 and 142.6 ms/step. The number is stable to about 1%.

The **70 TPS goal is cleared at rows=16 with ~113 TPS**, and 100 TPS with it. The shape of the
result is the point: the step grows 107 -> 299 ms while the row count grows 64x, so the fixed
per-step cost — the 50.6 ms of collectives, the per-step norms and the fixed launch overhead — is
being shared rather than repeated. Between rows=16 and rows=64 the marginal cost of a row is 2-4 ms
against a fixed cost of 100-190 ms per step, so throughput is still limited by the fixed half at
every batch size on this table. The rows=96 entry breaks the pattern — 7.2 ms per added row against
3.5 ms for the next 32 — and is recorded as measured rather than explained: the two runs either side
of it do not form a line, and attributing it to KV cache or attention without a separate measurement
would be a guess.

The same batch read the other way — bytes streamed per second of step — is the reason this route has
room left. At rows=64 the step spends 241 ms of work on a single pass over the 13.45 GB of resident
weights, i.e. **56 GB/s effective**, against 270 GB/s at M=1 and 422 GB/s at M=16 (`matmul
Mx5120x4352`, `bench_qwen_ascend_decode_ops`) and a 1148 GB/s HBM read probe on the same part. That
gap is per-row decomposition, not weight streaming: `qwen_causal_depthwise_conv_silu_f16_batched_ascend`,
`qwen_gated_delta_step_batched_f16_ascend` and `qwen_append_kv_cache_f16_batched_ascend` still loop
over rows internally, and those three are the next thing to fix.

### 6.1 What the batch oracle actually says

The driver compares the batched pass against a synchronous single-row reference and reports four
counts, and reading them as one number is what made this the hardest part of the work. They are:

| count | what it compares | must be |
|---|---|---|
| `seed_mismatches` | the two prefills, both from a reset engine | 0 |
| `batch_repeat_mismatches` | the batched pass against **itself**, second pass | 0 |
| `repeat_mismatches` | the single-row reference against **itself**, second pass | the baseline |
| `verify_mismatches` | the batched pass against the single-row reference | above the baseline |

Two separate effects are in play, and they were separated by adding the two self-comparisons.

**The single-row reference is not reproducible.** Two in-process passes of it, same binary, same
inputs, disagree with each other on 6-9 of 16 rows over a 16-step grid. It is horizon-dependent: at
`verify_steps=3` the same control reads 0 in every run tried, so the reference is deterministic for
the first few positions and only starts varying later. The same behaviour is visible on the plain
single-row `--generate` path without any batching: identical through step 4, then diverging, with
`|dlogit|` of 0.007-0.53 at token margins of 15-22, so it is not a borderline argmax. Neither
`HCCL_DETERMINISTIC=true` nor forcing the synchronous collective path (`QWEN_NCCL_COMM_STREAM=0`)
changes it. The batched path, by contrast, reproduces itself on every row of every run.

**The batched path also differs from the single-row path systematically**, and this one is
deterministic rather than noise: two independent runs diverge on the *identical* set of rows at the
same step, while both report `repeat_mismatches=0`, i.e. the reference reproduced itself exactly on
those runs. It localizes to the first full-attention layer. Sweeping the layer count at rows=16 and
`verify_steps=3` (48 compared pairs), with the KV cache size the engine reports in the same line:

| layers | `kv_cache_bytes` | `verify_mismatches` | max abs dlogit |
|---|---|---|---|
| 1 | 0 | 0/48 | 4.8e-07 |
| 2 | 0 | 0/48 | 4.8e-07 |
| 3 | 0 | 0/48 | 7.3e-04 |
| 4 | 720896 | 0/48 | 4.8e-01 |
| 5 | 720896 | 1/48 | 6.4e-01 |
| 6 | 720896 | 1/48 | 1.53 |
| 7 | 720896 | 1/48 | 5.6e-01 |
| 8 | 1441792 | 0/48 | 5.6e-01 |
| 64 | — | 9/48 | 2.65 |

Below the first full-attention layer every layer is linear attention and the two paths agree to
rounding level; the deviation appears in the same step that `kv_cache_bytes` stops being zero, and
then compounds through 64 layers into ~2.6 logits and 9 of 48 greedy tokens. The three operators
that run for the first time there are the batched partial RoPE, the batched KV append and the
batched GQA decode, all in `qwen_layer_components.inl`; which of the three is not yet isolated, and
it is the open correctness item on this path.

The consequence for the harness is that `verify_mismatches` is a count to be read against
`repeat_mismatches` from the same run, not a zero test, and that is how
`scripts/run_qwen_ascend_tp4.sh` now treats it: `seed_mismatches` and `batch_repeat_mismatches` are
hard gates, and a run whose `verify_mismatches` exceeds its own `repeat_mismatches` prints a NOTICE
naming both. The parity configuration that produced the earlier `verify_mismatches=0` figures in this page was
weaker in a second way as well. The rows were seeded from a constant token list, and a per-row
rotation of a constant list is the same list: every row decoded the same sequence, so a slot mix-up
between two rows cancelled out of the comparison — which is the one failure that comparison exists
to catch. The launcher now builds the list from consecutive ids and the rotation separates the rows.

### 6.2 The M=1 GEMV rate is not the hardware ceiling

`ascend_attention_optimization.md` §6 reads the `bench_qwen_ascend_gemm --scan` table — flat at
~320 GB/s over a 256x weight-size range — as "**~320 GB/s is the streaming ceiling** for this access
pattern on this part, not a launch-cost artefact", and derives a 42.0 ms/token = 23.8 TPS hard floor
at TP4 from it. The flatness argument is sound as far as it goes, but the scan only covers **M=1**,
and it is flat because M=1 is the shape the Cube cannot fill: the unit is 16 rows tall, so a batch of
one leaves fifteen sixteenths of every Mmad tile idle. Widening the batch is what the hardware was
waiting for, and it is 30% faster at M=16.

The ceiling that matters is the one every remaining target has to clear, and it is the memory system,
not the GEMV: the HBM read probe measures **1148 GB/s**, so 13.449 GB of resident weights is
**11.7 ms/token = 85 TPS at TP4** with zero compute cost. The batch table above is already at 214
TPS, above that number, because a step's weight read is shared by every row in it — which is the
whole argument for batching, and the reason the 23.8 TPS "bound" was never one.

## 7. Reproducing this

Everything on this page other than the two benches comes from the engine itself, one process per rank
on devices 0-3, through the repository's TP4 launcher:

```bash
# 70 TPS: rows=16, 16 steps, full 64 layers
QWEN_BATCH_ROWS=16 scripts/run_qwen_ascend_tp4.sh "" 16

# the knee of the table
QWEN_BATCH_ROWS=64 scripts/run_qwen_ascend_tp4.sh "" 8
```

The launcher sets `HCCL_WHITELIST_DISABLE=1`, raises `POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS` (rank
startup is skewed by minutes of checkpoint loading), and passes `--smoke-layers 0` for full depth.
`QWEN_BATCH_ROWS=N` adds `--batch-decode --max-batch-size N --token-ids <one 32-token prompt>`,
which is how the batched path is seeded: the prompt is replicated across the rows with a per-row
rotation so no two rows decode the same sequence. The result is the `batch_decode=1 ...` line the
engine prints at the end of the run. `QWEN_BATCH_VERIFY` sets the numerator of the four parity
counts on that line; see §6.1 for how to read them, and note that the launcher's hard gates are
`seed_mismatches` and `batch_repeat_mismatches`.

The layer sweep in §6.1 is the same command with `QWEN_LAYERS` set:

```bash
# the first full-attention layer is where the two paths start to differ
QWEN_LAYERS=4 QWEN_BATCH_VERIFY=3 QWEN_BATCH_ROWS=16 scripts/run_qwen_ascend_tp4.sh "" 4
```

The two benches are separate processes, four ranks each, and need the devices to themselves:

```bash
# the host-vs-device split and the Cube-GEMM overlap test
cpp_engine/build-ascend/tests/bench_qwen_ascend_allreduce \
    --nccl-id-path <shared> --tp-world 4 --tp-rank R --device R --iters 100

# the cross-stream event probe, single device
cpp_engine/build-ascend/tests/bench_qwen_ascend_event_order --device 0 --iters 40
```

## 8. Files

- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — the host-vs-device split and the Cube-GEMM
  overlap test, §2.
- `cpp_engine/tests/bench_qwen_ascend_event_order.cpp` — the cross-stream event probe, §4.
- `cpp_engine/tests/bench_qwen_ascend_decode_ops.cpp` — the M sweep (`matmul Mx5120x4352`) and the
  HBM read probe, §6.
- `cpp_engine/engine/qwen_engine.cpp` — `all_reduce_half_rows`, which is where the one-wide-collective
  decision lives.
- `cpp_engine/engine/main.cpp` — `--batch-decode`, the row-to-slot assignment and the parity oracle.
- `scripts/run_qwen_ascend_tp4.sh` — the TP4 launcher and its `QWEN_BATCH_ROWS` mode.
