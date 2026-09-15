# Ascend 910A Performance Roadmap

Qwen3.8-27B, 4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`, 30 AI cores, 32 MB L2,
FP16 only, no BF16), TP=4, CANN 9.0.0.

Measured status and the ranked next steps. All figures and their commands are in
[`ascend_attention_optimization.md`](ascend_attention_optimization.md); this document only draws
conclusions from them.

## Targets and current standing

| | target | measured now | gap |
|---|---|---|---|
| Prefill | >= 2000 TPS | 1265 TPS (4096-token prompt) | 1.6x |
| Decode | >= 100 TPS | 9.2 TPS (TP4, 64 layers) | 10.9x |

Decode is not 1.6x away and is not a tuning problem. See "Decode" below.

## Prefill: 1.6x is available, but not where it was assumed

An earlier version of this document attributed 84% of prefill to attention and planned a 10.9x
speedup on that basis. **That attribution was wrong and has been retracted.** The measured split at a
512-token prompt is:

- `STACK.r`, the whole decoder layer, **65.9%** — and it is still an unattributed black box
- `full_attention`, measured separately, 9.8%
- all projection GEMMs together, ~3%

The projection GEMMs being 3% is the load-bearing fact: prefill cannot be fixed by making matmuls
faster, and the single-card GEMM measurement (74-115 TFLOPS at batch 4096) says the target needs only
24-38% of what the hardware already delivers. The missing 1.6x is inside `STACK.r`.

**Next step, in order:**

1. **Instrument inside `decoder_layer_component.forward`.** `STACK.r` wraps the whole layer as one
   scope, so the norms, transposes, gated-delta work and elementwise passes are all merged into one
   17.2 ms/layer figure. Nothing further can be aimed until this is broken down. This is the only
   step that is currently unblocked and worth doing first.
2. **Re-check the 2000 TPS target at 512-token prompts separately.** The 1265 TPS figure is a
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

1. **129 TP all-reduce calls per decode token**, ~540 us each, serialised by the per-call
   `stream_synchronize` in `end_nccl_collective` — roughly 70 ms of the 108 ms step. The collective
   floor is flat in payload size up to 640 KB (0.3923 ms at 10 KB, 0.3989 ms at 640 KB against a
   0.0183 ms device-op floor), so this is latency, and the fix is fewer collectives.
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
  instrumented from the 65.9% `STACK.r` black box.
- Multiplicative projections of phase gains ("Phase 1 x Phase 2 = ~720 TPS") — the phases are not
  independent and the base attribution was wrong.
- "Cube is ~2x faster than Vector at matmul, so expect 3-5x" — the measured decode attention result is
  257 us vs 5224 us (20.3x), because the vector path's cost was a scalar loop over positions, not a
  matmul. Gains of this kind come from removing a scalar loop, not from the unit's peak rate.
- "FlashDecoding reduce to <20 ms gives 3.4x decode" — FlashDecoding is 5224 us in decode attention
  and loses to both Cube paths; it is a fallback for shapes the Cube path refuses.
- The week-by-week schedule with per-week TPS targets.
