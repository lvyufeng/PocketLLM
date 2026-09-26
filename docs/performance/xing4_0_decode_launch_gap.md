# Xing4.0-29B-A4B: the decode step's launch count, and what a graph buys

[§6 of the design record](../architecture/xing4_0_29b_a4b_design.md#6-where-a-decode-steps-time-goes)
leaves decode at 6.5 tok/s against a 152 tok/s byte floor and attributes the gap to host submission:
178 ms of host against 46.5 ms of device work, 11,536 launches a step. That attribution is a
*difference between two figures taken different ways*, and the number the next stage needed is not the
difference — it is how much of the host time a CUDA graph actually recovers. **This page is that
number: a whole-step graph runs in 38.0 ms against an eager step of 160–215 ms, at bit-identical
logits** — measured at a frozen position, before any of the work a real decode loop would need.

| | |
|---|---|
| Hardware | 1 x RTX 2080 Ti (sm_75), `--device cuda:2` |
| Checkpoint | `/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf` (18.72 GiB, 17.84 GiB resident) + the release directory's tokenizer and `config.json` |
| Context | **4,096 tokens**, the prompt the e2e bench builds, `--max-model-len 8192`, greedy, one request |
| Commit | `perf/xing4-0-decode-graph-probe`, the tree of PR #428 |
| Question | [#427][issue]: what is the ceiling on a graphed decode step, before committing to the work of building one |
| Run | 2026-09-26, `scripts/probe_xing4_0_decode_graph.py --device cuda:2 --context 4096 --steps 20` |

## Where a step's time goes, by name

Profiling one step (`scripts/profile_xing4_0_decode_launches.py --context 4096`, both activities)
gives a shape that is different from what §6 assumed, in a way that changes what the fix is:

**The step hands the host 22,155 ATen dispatches, and 10,508 of them launch no kernel at all.**

| aten op | calls | cuda ms | cpu µs |
| --- | ---: | ---: | ---: |
| `aten::as_strided` | 3,220 | 0.00 | 4,275 |
| `aten::to` | 2,037 | 0.00 | 3,040 |
| `aten::view` | 1,787 | 0.00 | 4,640 |
| `aten::reshape` | 1,668 | 0.00 | 3,115 |
| `aten::copy_` | 1,233 | 2.66 | 12,668 |
| `aten::empty_strided` | 1,117 | 0.00 | 9,652 |
| `aten::mul` | 1,080 | 1.87 | 16,833 |
| `aten::_to_copy` | 1,079 | 0.00 | 5,508 |
| `aten::unsqueeze` | 961 | 0.00 | 2,915 |
| `aten::empty` | 820 | 0.00 | 5,086 |
| `aten::slice` | 678 | 0.00 | 3,097 |
| `aten::add` | 475 | 0.71 | 7,856 |
| `aten::expand` | 439 | 0.00 | 1,312 |
| `aten::permute` | 400 | 0.00 | 1,061 |
| `aten::matmul` | 398 | 0.00 | 3,596 |
| `aten::transpose` | 358 | 0.00 | 927 |
| `aten::bmm` | 280 | 3.33 | 10,268 |
| `aten::arange` | 230 | 0.13 | 1,444 |
| `aten::cat` | 200 | 0.42 | 3,618 |
| `aten::linear` | 198 | 0.00 | 686 |
| `aten::mm` | 198 | 4.32 | 6,709 |
| `aten::mean` / `pow` / `rsqrt` | 162 / 161 / 161 | 0.73 / 0.24 / 0.25 | 3,194 / 3,333 / 2,225 |
| `aten::cos` / `sin` | 80 / 80 | 0.15 / 0.17 | 1,252 / 1,111 |
| everything else (≈2,900 calls) | — | ~1 | ~15,000 |

`view`, `reshape`, `as_strided`, `unsqueeze`, `permute`, `transpose`, `slice`, `expand`,
`_unsafe_view`, `_reshape_alias` and `t` are **metadata-only**: they change a stride array and
return. They are **10,508 of the 22,155 dispatches — 47% — and 24.0 ms of the step's host time for
zero device work.** The `aten::to` / `_to_copy` / `copy_` chains add 4,349 more dispatches and
**21.2 ms** of host time for 2.66 ms of device: they are the fp16↔fp32 traffic around the residual
split §2 of the design record introduces, because the four residual streams are fp32 and the
sublayers are fp16, so every sublayer boundary is two casts and a contiguous copy.

**Those two families are 14,857 dispatches and 45 ms of the step, and they exist to feed 1,074 GEMM
calls** (`matmul`/`bmm`/`mm`/`linear`, 21.3 ms of host and 7.65 ms of device between them).

What actually reaches the card, from a CUDA-only profile of the same step:

| | per step |
| --- | ---: |
| `cudaLaunchKernel` calls — the host's bill | **5,346** |
| GPU kernel instances | 5,577 |
| device-to-device copies | 308 |
| the card's own time | 45.9 ms |

The largest kernels, by device time, from the by-name table:

| kernel | count | device ms |
| --- | ---: | ---: |
| `xing4_hyper_connection_kernel<float>` | 80 | 8.19 |
| `gguf_moe_w13_kernel<half>` | 38 | 6.29 |
| `gguf_quant_gemm_kernel<float>` | 41 | 3.54 |
| `gguf_moe_w2_scatter_kernel<float>` | 38 | 3.43 |
| `gemvx::kernel` (the LM head) | 120 | 3.21 |
| `gguf_quant_gemm_kernel<half>` | 80 | 3.21 |
| `unrolled_elementwise_kernel` | 559 | 1.52 |
| `gemv2T_kernel_val<...half...>` | 80 | 1.35 |
| `elementwise_kernel<128, 2, ...>` | 641 | 1.22 |

So: **22,155 host dispatches to submit 5,346 launches, for 5,577 kernels that take 46 ms.** Between a
quarter and a fifth of the step is the card. The rest is the host walking a dispatch tree that is half
bookkeeping.

**One count on this page does not reconcile with §6's, and the discrepancy is stated rather than
smoothed.** §6 says 11,536 launches a step; the two scripts above say 5,346 host submissions and 5,577
kernels. They are not the same instrument — §6's count was taken in the stage-5 profiling round with a
different profiler view — and it cannot be re-derived from that stage's artifacts. What is
reproducible is the pair of commands in the run record, and it is what the rest of this page uses.

## The one device-to-host read in the path

Capturing the step did not work at first, and what blocked it is worth its own paragraph because it
was invisible to every other kind of measurement.

`plan_routes` (`src/models/xing4_0/mlp.py`) called `torch.bincount(flat_expert, minlength=n_experts)`.
`minlength` already fixes the output's length, but `bincount` still sizes its output from the data's
own maximum, which it reads back to the host. It runs **once per MoE block, 38 times a step**, and
measured **76 device-to-host copies and 78 stream synchronisations a step** — two of each a call.
It is also a CUDA graph's flat refusal: `cudaErrorStreamCaptureUnsupported`, reported asynchronously
at whatever call happened to come next.

The replacement is the same histogram over a buffer whose length is known, a `scatter_add_`, and
`test_the_plan_is_a_csr_over_the_route_table` already asserted the counts against
`torch.bincount(..., minlength=experts)` — so the parity was pinned before the change was made. Its
own cost is now measured: **3.6 ms of the step** (bincount 179.8 ms/step against scatter 176.2, four
interleaved rounds each, 1.020×). The guards are `test_the_plan_never_reads_back_from_the_card` —
which fails on the reintroduced `bincount` rather than merely asserting a count — and the capture,
which now succeeds.

**The sync was real, it happened 38 times a step, and it was not the bottleneck.** Removing it moves
the step 2%. That is the argument for measuring before rewriting.

## The three arms

One process, the same weights, the cache resumed to the same state, the position frozen at 4,096.
Twenty steps an arm with one `synchronize` around each arm's loop, and the arms interleaved so a clock
change or a warm-up lands on both columns.

| Arm | ms/step | tok/s | host submissions | device ms | device-busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 178.6 | 5.60 | 5,346 | 45.9 | 26% |
| **graph, frozen position** | **37.9** | **26.37** | **0** | 38.0 | **100%** |
| `torch.compile(mode="reduce-overhead")` | 155.2 | 6.44 | 1,228 | 43.5 | 28% |

Interleaved, in the order they were run:

```
     eager:    211.7 ms/step    4.72 tok/s        eager:    213.4 ms/step    4.69 tok/s
     graph:     38.3 ms/step   26.11 tok/s        graph:     39.8 ms/step   25.10 tok/s
     eager:    212.3 ms/step    4.71 tok/s
```

**The graph's row is the whole result.** The host submits **zero** `cudaLaunchKernel` calls — the
replay is one submission of a different kind — and the card's own time becomes **100% of the step**.
The step stops being a host artefact and becomes the device work it always was.

**The ratio depends on which eager you take, and both are honest.** In this run the *first* timed
eager arm — measured before the graph had been captured — is 178.6 ms, and the interleaved ones after
it are 212. Across eight runs of the probe the graph column never left 37.8–39.9 ms while the eager
column ranged 160–218 ms. So: **38 ms against an eager step of 160–215 ms is 4.2× to 5.7×, and the
stable quantity is the 38**. Which eager figure is the right denominator is a question about the host,
not about the graph: the eager path is slower in a process that has also built and replayed a graph,
and this page does not claim to have explained that beyond measuring it.

**Run with the host free to run ahead, one graphed step is 38.0 ms (26.28 tok/s)** — 200 replays
submitted back to back with no synchronise between them, so whatever the host can queue is hidden and
the number is the card's. That is what a real decode loop converges to.

## The graph is exact

`max |Δlogits| = 0.000e+00` against the eager step at the same position, compared element for element
across all 131,072 of them. Not "within fp16 rounding" — zero. The capture pass runs the real forward
against the real cache and the replay reproduces it, so at this position there is no rounding
question to argue about.

**The position is frozen, and that is the whole remaining problem.** `start_pos` is a Python int that
picks the RoPE row, the causal mask offset, the cache slice the attention reads and the range it
writes; a capture bakes all four in. Every replayed step here is the same position at the same token.
Making them index tensors the graph reads is stage 2, and this page is the authorization for it
rather than a substitute for it.

## `torch.compile` is not the instrument

`mode="reduce-overhead"` was tried first, because it is the cheap version of the same idea, and it
does not work here: **6.44 tok/s against eager's 5.60**, 1.15×. Three reasons, all structural rather
than tunable:

- It **refused cudagraphs**: `skipping cudagraphs due to mutated inputs`. A decode step writes the KV
  cache in place, and Dynamo sees the mutation.
- It **cannot trace the two ops that matter**: `gguf_quant_gemm_forward` and
  `gguf_moe_prefill_grouped_forward` are pybind functions, and Dynamo has no rule for them, so they
  become graph breaks and the MoE stays eager either way.
- The position is a Python int, so a real decode loop would recompile per position.

It did do something — 22 Inductor graphs, and the host submission count falls from 5,346 to 1,228 —
but the step stays bound by the ops it could not touch, and those two pybind calls plus the scatter
they feed are **16.5 ms of its device time on their own**.

## What this authorizes

- **Stage 2 is worth doing.** The ceiling is 4.2× or better and the mechanism is already demonstrated
  at one position with bit-identical output.
- **A per-position graph is affordable if the position has to be bucketed.** The whole step's graph
  pool is **24.0 MiB** — a one-token capture body is narrow. V4.1's page carries 6.5 MiB *a layer* for
  a body of the same width, but that capture was per layer, so the two are not the same measurement;
  the comparison is only to say that 24 MiB for a forty-block step is small. Even 64 buckets would be
  1.5 GiB, and a decode loop needs far fewer than that.
- **The next thing to fuse is the dispatcher, not a kernel.** 47% of the step's dispatches launch
  nothing, and the largest single device item, the hyper-connection, is already one kernel. Once the
  graph removes the *submission* cost, what is left is 46 ms of device time, and that is the number
  §3's cast and norm fusions and §4's attention kernel have to be priced against — not 167.

## What is not here

- **A decode loop.** The position is frozen. A graphed loop needs `start_pos` as a tensor in four
  places, and the hardest is the attention GEMM's `N`, which is the cache length: a graph needs it
  static, so it is either a bucket plus masking or a kernel that reads it from the device. That
  choice is stage 2's, and this page's 24 MiB says both are affordable.
- **End-to-end rates.** Every number here is a step at one position in a probe, not a generation. The
  rates a report wants are the e2e bench's, and they move when the loop does.
- **Prefill.** Unchanged and out of scope: the launch count is a decode property.

## Run record

| Claim | Command |
| --- | --- |
| The dispatch distribution, per step | `scripts/profile_xing4_0_decode_launches.py --device cuda:2 --context 4096 --steps 3` |
| The three arms, the parity, the pool, the submissions | `scripts/probe_xing4_0_decode_graph.py --device cuda:2 --context 4096 --steps 20` |
| The bincount sync, its cost and its guards | `python -m pytest tests/test_xing4_0_moe.py -q`, and the A/B on `plan_routes` recorded above |

One thing about the numbers a reader should not have to reconstruct: **the eager column is a host
measurement and it moves with the host; the graph column is the card's and does not.** Eight runs of
the probe gave graph 37.8–39.9 ms every time and eager anywhere from 160 to 218. The within-run
spread is under 3% in both columns, so the interleaved pairs above are the figures to quote, and the
graph's own 38 ms is the figure to carry forward.
