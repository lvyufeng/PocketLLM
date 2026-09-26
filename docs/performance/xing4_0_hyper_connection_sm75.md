# Xing4.0-29B-A4B's hyper-connection on sm_75

What the block costs, measured because [#392][issue] asked whether it is small. It is not, and the
reason is dispatch count rather than arithmetic.

| | |
|---|---|
| Hardware | 1 x RTX 2080 Ti (sm_75, 616 GB/s peak) |
| Checkpoint | `XingChen-AGI/Xing4.0-29B-A4B`, released BF16 shard 3 — layer 2's `attn_hc` |
| Weights | real, at the released widths; cast to fp16 |
| Invocation | `python scripts/bench_xing4_0_block.py --device cuda:2` |
| Date | 2026-09-26 |

One call is `attn_hc` or `ffn_hc`; the trunk makes two per layer over 40 layers, so a per-token
figure is **80 calls**. The extrapolations below are that multiplication, and the port is the eager
one in `src/models/xing4_0/hyper_connection.py`.

## Bytes

`hc_fn` is 24 x 14336, 672 KiB in bf16, plus 24 + 3 scalars; 0.656 MiB per call, **52.5 MiB per
token** for the trunk. At 616 GB/s that is **0.0894 ms/token**. Against the attention's 2.117 GiB and
the MoE's 0.877 GiB, this block is 2.5% of a decode token's weight traffic, and the audit's note that
it can be kept unquantized in fp16 at no memory cost is confirmed by the same arithmetic.

## Time

| rows | wall ms | GPU ms | kernel launches | GPU us/launch | 80 calls, GPU ms | 80 calls, wall ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.292 | 0.382 | 300 | 1.27 | **30.6** | **183.4** |
| 512 | 3.249 | 1.025 | 300 | 3.41 | 82.0 | 259.9 |
| 2048 | 3.366 | 2.699 | 301 | 8.96 | 215.9 | 269.3 |

**300 kernel launches per call, every one of them averaging 1.27 us of GPU time at decode.** The
arithmetic is nothing — the block is 4x4 matrices and two 24-wide gates — and the GPU column is 340x
the byte floor, which is what 300 dispatches of a trivial kernel costs. A decode token therefore
issues about **24,000 launches** into this block alone, and 30.6 ms of GPU time to do 0.09 ms of work.

The Sinkhorn accounts for 120 of the 300: 20 iterations of (sum, add, divide) on each axis, and each
of those six operations is its own kernel. The rest is the norm, the gate's own sigmoid/scale/bias
path and the three casts, all of which are pointwise over a `(rows, 4)` tensor.

This is the finding the issue said to look for, and it is the *second* largest decode-time item after
the MoE, ahead of the attention's 8.9 ms. It is also the most fixable thing in the stage: nothing here
needs new arithmetic, only fewer dispatches.

## What would fix it

Three options, in the order this repository would take them:

1. **One fused kernel.** The entire block is a reduction plus pointwise work over `(rows, 4)` and
   `(rows, 4, 4)`. The 20-iteration loop has a data dependency between the two normalizations, so it
   cannot be folded into fewer *steps* — but it can be folded into one *kernel* that runs all 40
   normalizations in registers. That is the shape `hc_*` wants, and the audit already established it
   may be fp16 with fp32 gates at no memory cost.
2. **A CUDA graph over the decode step.** 24,000 launches per token is exactly the case a graph
   exists for, and the repository already has a graphed decode path for V4.1 to copy. It fixes the
   wall column (183 ms/token) without touching the GPU column (30.6 ms/token), so it is a partial
   answer on its own.
3. **Do nothing, if the MoE dominates by more than this.** At 30.6 ms of GPU per token the block caps
   decode at 33 tok/s, which is above nothing else in the stage only if the MoE lands under that.
   [#393][issue393] measures the MoE; if the MoE's decode is slower than 33 tok/s then this block is
   not the first thing to fix and the number to keep is the launch count, not the milliseconds.

## What is not here

- **A kernel, or a graph.** This is a measurement of the port. Nothing in this page changes
  `hyper_connection.py`.
- **A prefill story.** At 2048 rows the block is 2.7 ms of GPU per call, i.e. 216 ms per 2048-token
  prefill across the trunk — the same order as the attention's 229 ms and worth the same treatment,
  but the decode case is where the launch count is pathological.
- **End-to-end confirmation.** The block's parity is established against the reference
  (`tests/test_xing4_0_hyper_connection.py`); what it does to a real token's latency is a question
  for a trunk, which is [#393][issue393].

[issue]: https://github.com/lvyufeng/PocketLLM/issues/392
[issue393]: https://github.com/lvyufeng/PocketLLM/issues/393
