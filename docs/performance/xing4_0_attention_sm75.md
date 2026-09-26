# Xing4.0-29B-A4B's MLA attention on sm_75

What the attention costs on one RTX 2080 Ti, and what measuring it changed. The port it measures
is `src/models/xing4_0/attention.py` — the staging form of [#391][issue], not a kernel.

| | |
|---|---|
| Hardware | 1 x RTX 2080 Ti (sm_75, 22528 MiB, 616 GB/s peak) |
| Checkpoint | `XingChen-AGI/Xing4.0-29B-A4B`, the released BF16 shard 3 — layer 2 |
| Weights | `model.layers.2.*`, cast to fp16; the release's own tensors, unmodified |
| Invocation | `python scripts/bench_xing4_0_attention.py --device cuda:2` |
| Date | 2026-09-26 |

One layer is every layer's attention, so the per-layer figures below are scaled by the trunk's **40
layers** to give a per-token cost — the number that decides whether the attention can be the decode
bottleneck. Prefill and decode are reported separately and neither is carried across; the numbers
here are all single-request, batch 1.

## The volume, and its floor

The attention's weights are 54.2 MiB per layer in fp16, and a decode token streams all of them:
**2.117 GiB/token for the trunk**, which at 616 GB/s is **3.690 ms/token** — a 271 tok/s ceiling
that no implementation can beat on this card. That is the reference line for everything below.

The three projections that dominate the volume, as plain fp16 GEMMs at their real shapes:

| projection | shape | M=1 ms | M=1 weight-read GB/s | M=512 ms | M=512 weight-read GB/s |
|---|---:|---:|---:|---:|---:|
| `q_b_proj` | 6144x768 | 0.066 | 142 | 0.201 | 47 |
| `kv_b_proj` | 8192x512 | 0.065 | 129 | 0.156 | 54 |
| `o_proj` | 3584x4096 | 0.090 | 325 | 0.443 | 66 |

At M=1 these skins read the weights at 21–53% of peak: a GEMV on this architecture is not a
bandwidth benchmark. At M=512 the same bytes are amortised over 512 rows and the rate falls again,
because the GEMM has become compute-bound — the M=512 column is a FLOPs number wearing a bandwidth
label. Both are the same fact from two sides: **the reachable rate for this attention's decode
shapes is a few hundred GB/s, not 616**, so the effective floor is two to four times 3.69 ms.

## Decode: one token against a filled cache

Median of 20 steps, per layer, with the GPU's own time taken from the profiler and the wall time
from CUDA events.

| context | form | wall ms | GPU ms | 40-layer GPU ms | 40-layer tok/s |
|---:|---|---:|---:|---:|---:|
| 1 | absorbed | 2.68 | 0.222 | 8.9 | 113 |
| 1 | expanded | 2.36 | 0.252 | 10.1 | 99 |
| 4096 | absorbed | 2.47 | 0.267 | 10.7 | 94 |
| 4096 | expanded | 3.27 | 1.413 | 56.5 | 18 |
| 32768 | absorbed | 2.46 | 0.516 | 20.7 | 48 |
| 32768 | expanded | 14.29 | 10.35 | 414 | 2.4 |

Three things in that table are the stage's findings.

**The absorbed form is not an optimisation, it is the only form that survives a long context.** At
32K the expanded form costs 413 ms/token for the trunk — 20x the absorbed one — because it
materialises a 10240-wide key and value for every cached token on every step, while the absorbed
form reads the 576-wide latent. The two agree numerically (`tests/test_xing4_0_attention.py`) and
the choice between them is purely this table.

**The natural way to write the absorbed score is 5.8x too slow.** `(b, h, s, k) @ (b, k, tokens)`
broadcasts the cache across heads, and a broadcast operand is read once *per head* — 32x the
traffic, 1.07 GiB per layer at 32K instead of 33 MiB. Folding the head axis into the GEMM's `M`
dimension gives one matrix whose second operand is read once. Measured effect at 32K: **2.991 ->
0.516 ms per layer**; at 4096, 0.464 -> 0.267 ms. The arithmetic is untouched — each output
element is still the same per-head sum over the same 512 terms — which is why the parity tests did
not move. This was found by the measurement and not by reading the code.

**The port is host-bound, by an order of magnitude.** The wall column is flat at ~2.5 ms per layer
per step while the GPU column is 0.22–0.52 ms: about 250 eager kernel launches at roughly 7–10 us
of host time each. The attention's own ceiling here is 271 tok/s, the port reaches 113, and the
difference is launch latency — so the performance work this stage leaves behind is a fused decode
kernel or a CUDA graph, not a better contraction.

## Prefill

One pass, no cache, per layer.

| sequence | expanded wall | absorbed wall | expanded GPU | absorbed GPU |
|---:|---:|---:|---:|---:|
| 128 | 1.79 | 2.74 | 0.524 | 0.558 |
| 512 | 2.82 | 3.58 | 1.485 | 2.201 |
| 2048 | 6.48 | 18.36 | 5.72 | 18.15 |

Scaled to the trunk, a 2048-token prefill is 229 ms of attention (expanded) — about 8.9K tok/s if
the attention were the only thing running, which it is not. The prefill recommendation is the
mirror of the decode one: **expanded at prefill, absorbed at decode**, and the gap widens with
sequence length (3.2x at 2048).

## What this says about the bottleneck

[#391][issue] expected the attention not to be the bottleneck, and at short context it is not: 8.9 ms
of GPU per token at 1 token of context and 10.7 ms at 4K are 113 and 94 tok/s of headroom, against
a MoE that has to stage 0.877 GiB of routed experts per token through this card. At 32K the
attention's own figure rises to 20.7 ms/token (48 tok/s), which is no longer obviously clear of the
model's decode rate — so the honest statement is that the attention is not the bottleneck **at the
contexts this card can hold**, and that a long-context decode on one 2080 Ti would have to be
measured against a fitted kernel rather than against this port.

The stage's remaining gap is launch overhead, and it is the same gap the rest of the repository has
already closed elsewhere with graphs and fused kernels.

## What is not here

- **End-to-end logits.** The acceptance for #391 asks for attention-only parity and *then*
  end-to-end logits. The parity is done and is against the checkpoint's own tensors at the released
  widths (`tests/test_xing4_0_attention.py`); the logits comparison cannot run yet, because the
  hyper-connection block and the MoE are [#392][issue392] and [#393][issue393] and there is no
  trunk to logits from. It is deferred to #393 with its reason, not dropped.
- **A kernel.** Everything here is eager PyTorch, so the wall column is an upper bound on what a
  kernel would cost and the GPU column is a lower one.
- **A quantized attention.** These are the released BF16 tensors in fp16. The 4-bit GGUF path will
  move less per token and is #393's measurement.

[issue]: https://github.com/lvyufeng/PocketLLM/issues/391
[issue392]: https://github.com/lvyufeng/PocketLLM/issues/392
[issue393]: https://github.com/lvyufeng/PocketLLM/issues/393
