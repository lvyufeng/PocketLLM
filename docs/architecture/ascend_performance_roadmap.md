# Ascend 910A Performance Roadmap

Qwen3.8-27B, 4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`, 30 AI cores, 32 MB L2,
FP16 only, no BF16), TP=4, CANN 9.0.0.

Measured status and the ranked next steps. All figures and their commands are in
[`performance/ascend_attention_optimization.md`](../performance/ascend_attention_optimization.md); this
document only draws conclusions from them.

## Targets and current standing

| | target | measured now | gap |
|---|---|---|---|
| Prefill | >= 2000 TPS | 1261.6 TPS (4096-token prompt), 878.8 TPS (512-token prompt) | 1.6x |
| Decode | >= 100 TPS | 9.2 TPS (TP4, 64 layers) | 10.9x |

Decode is not 1.6x away and is not a tuning problem. See "Decode" below.

## Prefill: the wall is the collectives, not the math

An earlier version of this document attributed 84% of prefill to attention and planned a 10.9x
speedup on that basis. **That attribution was wrong and has been retracted.** A second version
correctly blamed the projections' share and then stopped at `STACK.r`, the whole decoder layer, as an
"unattributed black box". **That is now measured too, and the answer is not compute:**

| component, 512-token prompt | share of prefill wall |
|---|---|
| 129 TP all-reduce calls (2 per layer + 1 hidden) | **87%** |
| every other operation in the 64-layer stack | 10% |
| `top1_allreduce`, 1 call | 2% |

At a 4096-token prompt the same profile gives 84% for 513 collectives, so this is the shape of a
64-layer TP4 pass rather than a property of short prompts. Attention is 9.8% at 512 tokens and 9.3%
at 4096; all projection GEMMs together are ~1% at 512 and ~0.3% at 4096.

The consequence is that prefill cannot be fixed by making matmuls faster, and the single-card GEMM
measurement (74-115 TFLOPS at batch 4096) is not the binding constraint. The lever is the number of
times the layer stops at a collective.

**Done:**

- Branch `perf/ascend-overlap-slice-by-rows`: the overlapped all-reduce now slices by row count
  instead of a constant 4 (`comm_overlap_slices_for_rows`), worth **2.05x at a 512-token prompt**
  (428.1 -> 878.8 TPS) and neutral at 4096. See
  [`performance/ascend_tp_collective_overlap.md`](../performance/ascend_tp_collective_overlap.md).

**Next step, in order:**

1. **Cut the collective count per layer.** Two per layer — one `mlp.down` reduce in every layer and
   one attention-output reduce — at ~3.8 ms each against a layer whose every other operation
   together costs 0.90 ms. Removing one of the two per layer takes 0.245 s off the 0.584 s wall at
   512 tokens. This is the same change decode needs, and it is the only identified prefill lever
   that is worth more than a few percent.
2. **Explain `top1_allreduce`.** A single call costs 0.243 s at 4096 tokens (7.5% of that prefill)
   against 0.011 s for the same scope in the decode block of the same run. It is measured and not
   yet explained; a 20x phase difference on one call is worth an afternoon.
3. **Re-check the 2000 TPS target at 512-token prompts separately.** The 1261 TPS figure is a
   4096-token prefill, where the per-token cost is dominated by the layer GEMMs. Shorter prompts pay
   the same fixed per-step costs over fewer tokens, so a single 2000 TPS target across prompt lengths
   may need to be restated per length rather than pursued as one number.

## Decode: quantization is required, not optional

The hard bound is bandwidth, and it is measurable rather than arguable. Per-rank resident weights are
13.45 GB and every linear is read once per token. The standalone single-row GEMV plateaus at ~320 GB/s
(flat over a 256x weight-size range, so this is the streaming ceiling and not a launch-cost artefact).
That gives:

| | ms/token | TPS |
|---|---|---|
| perfect streaming, TP4 | 42.0 | 23.8 |
| perfect streaming, TP8 | 21.0 | 47.6 |
| **required for 100 TPS** | **<= 10** | **100** |

100 TPS at TP8 would need 673 GB/s per card with zero collective cost. Two separate deficits sit on
top of the bound, and neither is small:

1. **129 TP all-reduce calls per decode token, at ~0.40 ms each** — roughly 52 ms of the ~104 ms
   step, and confirmed by the post-fix host profile of a 4096-context token: the 129 calls measure
   0.068 s of the 0.114 s step, **60%**. A previous version of this document blamed the per-call
   `stream_synchronize` in
   `end_nccl_collective` for ~70 ms of it. **That attribution is wrong.** In
   `bench_qwen_ascend_allreduce` the per-call figure is flat in message size (0.404 ms of pure
   host enqueue at 10 KB, 0.412 ms at 640 KB) and the synchronized round trip is not slower than the
   unsynchronized one (0.388 vs 0.456 ms at 10 KB — the "sync" variant is measured inside the loop
   and the "comm" variant pays an extra per-iteration synchronize, so the difference is the
   measurement, not the fence). The probe line settles it: enqueueing an event pair costs 0.011 ms,
   enqueueing one bare collective costs 0.382 ms. The cost is host-side issue of the collective
   itself, it is paid once per call regardless of payload, and neither `HcclCommConfig`'s
   `hcclOpExpansionMode` (swept 0/1/2/3, all within noise) nor `HCCL_OP_EXPANSION_MODE=AIV` moves
   it. The fix is fewer collectives, not a cheaper fence around each one.
2. **The per-layer cost is ~2.4x the pure weight-streaming cost** (13.45 GB / 108.6 ms = 124 GB/s
   effective vs the ~320 GB/s ceiling). The excess is layer work that moves no weights: the
   gated-delta matrix operations — two 128x128 reductions and a rank-1 update currently expressed as
   128-wide vector ops, with `broadcast_rows` the documented hot spot — plus the norms and
   transposes. Those are Cube work being done on the vector unit, and unlike attention this one is
   worth 128x128 scale arithmetic.

**Ranked next steps:**

1. **Merge the per-layer collectives.** 129 calls at ~0.5 ms is the largest single identified cost in
   the decode step and needs no new hardware capability. Removing it should land decode at the
   streaming bound of roughly 20-23 TPS at TP4.
2. **Move the gated-delta matrix work onto the Cube.** Same argument as the attention kernel: it is a
   matrix operation running on the vector unit.
3. **Weight quantization (int8, then int4).** This is the only route to 100 TPS. int8 halves the
   42 ms floor to 21 ms (47 TPS at TP4), int4 quarters it to 10.5 ms (95 TPS at TP4, ~190 at TP8).
   Accuracy validation is the cost, not the kernel.
4. **Continuous batching.** Note this multiplies *throughput*, not per-token latency: it is currently
   blocked by a throw in the scheduler path and by the missing `runtime.batch_rows` branch in
   `qwen_layer_components.inl`. It is a throughput lever on top of the above, not a substitute for it.

## Retracted from the previous version of this document

The following were estimates presented as a plan and are not supported by measurement. They are
listed so they are not picked up again:

- "attention is 84% of prefill" — measured at 9.8% at 512 tokens, and the phase is separately
  instrumented from the decoder layer. The layer itself turned out to be 87% TP all-reduce, so the
  error was not in measuring attention but in never opening the scope that contained everything
  else.
- Multiplicative projections of phase gains ("Phase 1 x Phase 2 = ~720 TPS") — the phases are not
  independent and the base attribution was wrong.
- "Cube is ~2x faster than Vector at matmul, so expect 3-5x" — the measured decode attention result is
  257 us vs 5224 us (20.3x), because the vector path's cost was a scalar loop over positions, not a
  matmul. Gains of this kind come from removing a scalar loop, not from the unit's peak rate.
- "FlashDecoding reduce to <20 ms gives 3.4x decode" — FlashDecoding is 5224 us in decode attention
  and loses to both Cube paths; it is a fallback for shapes the Cube path refuses.
- The week-by-week schedule with per-week TPS targets.
