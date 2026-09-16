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

## Prefill: the wall is the linear-attention recurrence

Two earlier versions of this document got this wrong in different ways, and both retractions matter.
The first attributed 84% of prefill to attention. The second replaced it with the collectives at
84-87%, on the strength of a host-wall profile whose spans nest — an outer `ar.*` scope absorbed
everything that ran between two collective issues, and `TOTAL` came out above the wall it was
measuring (2.77 s of scopes against a 1.25 s wall). **A device-synced profile at three prompt lengths
gives a third answer**, and it is the first one the collectives do not own:

| share of `STACK.r` | 1105 tokens | 2223 tokens | 4433 tokens |
|---|---|---|---|
| **`gated_delta`** (48 calls) | **44.9%** | **51.9%** | **56.2%** |
| 129 TP all-reduce calls | 24.7% | 16.4% | 12.7% |
| all 384 projection GEMMs | 13.3% | 15.3% | 14.0% |
| `full_attention` (16 calls) | 11.1% | 11.0% | 12.1% |

The recurrence is 10.35 us per token per head per layer and is **exactly linear in sequence length**
(496.9 / 495.1 / 496.5 us per token across a 4x span), so it is not a fallback path and it does not
amortise. An ablation of the kernel shows what the time is: removing its `broadcast_rows` calls and
its 128-iteration rank-1 `Axpy` loop — both Cube-shaped arithmetic expressed as per-row vector
instructions — moves the end-to-end 4433-token prefill from **1128 to 1734 TPS**, so **62% of the
recurrence's 2.2 s is vector-instruction issue cost, not state bandwidth.** Both groups are
independent, to 0.7%.

The consequence is that prefill cannot be fixed by making the collectives fewer, and the projections
are not the binding constraint either at 13-15%. The lever is one kernel.

**Done:**

- Branch `perf/ascend-overlap-slice-by-rows`: the overlapped all-reduce now slices by row count
  instead of a constant 4 (`comm_overlap_slices_for_rows`), worth **2.05x at a 512-token prompt**
  (428.1 -> 878.8 TPS) and neutral at 4096. See
  [`performance/ascend_tp_collective_overlap.md`](../performance/ascend_tp_collective_overlap.md).
  That document's section 5 also carries the corrected attribution above and the ablation.

**Next step, in order:**

1. **Move the gated-delta recurrence onto the Cube.** It is 2.20 s of the 3.95 s wall at 4433 tokens
   and 62% of it is issue cost of per-row vector instructions whose shapes are matrix shapes: the
   `broadcast_rows` / `fold_rows` pair computes `k^T S` as a broadcast, a multiply and a seven-step
   fold, and the rank-1 update `S += k (x) delta` is 128 `Axpy` calls. `Brcb` is an unsupported stub
   on first-generation `dav_c100`, so the scalar `Duplicate` loop is deliberate — but it is 36% of the
   wall at that length. This is the only identified prefill lever worth more than a few percent.
2. **Cut the collective count per layer.** Two per layer — one `mlp.down` reduce in every layer and
   one attention-output reduce — at a measured ~1.85 ms of fixed host-side issue each. At 129 calls
   that is 0.24 s of the 3.95 s wall at 4433 tokens, and it is the same change decode needs. Worth
   doing, but it no longer reaches the target on its own.
3. **Explain `top1_allreduce`.** A single call costs 0.243 s at 4096 tokens (7.5% of that prefill)
   against 0.011 s for the same scope in the decode block of the same run. It is measured and not
   yet explained; a 20x phase difference on one call is worth an afternoon.
4. **Re-check the 2000 TPS target at 512-token prompts separately.** The 1261 TPS figure is a
   4096-token prefill, where the per-token cost is dominated by the layer GEMMs. Shorter prompts pay
   the same fixed per-step costs over fewer tokens, so a single 2000 TPS target across prompt lengths
   may need to be restated per length rather than pursued as one number.

At 4433 tokens, 2000 TPS is a 2.217 s wall. The recurrence is 2.202 s of the 3.948 s measured, so it
has to fall 4.7x before any other term matters; turning the two ablated groups into Cube ops gives a
2.6 s wall or about 1700 TPS, and the remainder is then the collectives' fixed term plus the state
passes that survive.

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
   worth 128x128 scale arithmetic. The prefill ablation prices the same two groups at **62% of the
   recurrence's device time** (see "Prefill" above), so the estimate is no longer a cost model.

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
  instrumented from the decoder layer.
- "the collectives are 84-87% of prefill" — the replacement for the line above, and wrong for the
  same reason in the other direction: the spans it was read from nest, so an `ar.*` scope was
  charged whatever ran between two collective issues. Device-synced, the collectives are 12.7% of
  the decoder stack at 4433 tokens against the linear-attention recurrence's 56.2%.
- Multiplicative projections of phase gains ("Phase 1 x Phase 2 = ~720 TPS") — the phases are not
  independent and the base attribution was wrong.
- "Cube is ~2x faster than Vector at matmul, so expect 3-5x" — the measured decode attention result is
  257 us vs 5224 us (20.3x), because the vector path's cost was a scalar loop over positions, not a
  matmul. Gains of this kind come from removing a scalar loop, not from the unit's peak rate.
- "FlashDecoding reduce to <20 ms gives 3.4x decode" — FlashDecoding is 5224 us in decode attention
  and loses to both Cube paths; it is a fallback for shapes the Cube path refuses.
- The week-by-week schedule with per-week TPS targets.
