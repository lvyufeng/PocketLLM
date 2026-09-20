# DeepSeek-V4.1-Flash: the decode step with the tree half inside a CUDA graph

The [remaining-bottlenecks page](deepseek_v4_1_flash_remaining_bottlenecks.md) leaves Lever 6 — the
per-layer CUDA graph — as the largest item on its list and the only one it could not size: the graph
is the right instrument for a step that launches **7,076 kernels** with **111–122 ms** outside every
named call at **11%** device-busy, but that page's own rule, 先搬，量完再说, says a lever is not
authorized until its number is in. **This page is that number**, taken end to end on the real
checkpoint with the real forty layers and the real `DeviceRoutedExperts` call in the middle.

**The step falls from 833.6 to 561.6 ms** at a frozen `start_pos` of 1024, and from 843.8 to 566.1 ms
at 1025 — **1.48× and 1.49×**, which is **272.0 ms** and **277.7 ms** of the step. The logits are
**bit-identical** at both positions (`max |diff| = 0.000e+00`), and all forty layers' graphs share a
pool of **0.254 GiB**: **6.5 MiB a layer**, not the 3.7 GiB a layer the round's own plan projected off
a 512-token activation, because a decode capture body is one token wide.

Two things are deliberately outside this page and are stated here rather than left to be inferred.
**The position is frozen**, so this is not yet a decode loop: `start_pos` is a Python int that picks
both the `freqs_cis` slice and the ring-buffer slot, so a capture freezes it and every replayed step
is the same position at the same token. And **prefill is not in scope** — the prompt is one forward
with no chunking, and its graph is a different mechanism for a separate round. What the page
establishes is the two Phase A gate conditions: one layer's graph fits in the shared pool, and the
capture pass reproduces the eager step exactly at a fixed position.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards; **no resident bank attached** (`DEEPSEEK_V41_RESIDENT_EXPERTS` unset, `resident_engram=False`, `DEFAULT_EXPERT_CACHE`) |
| Commit | `e8ecf89` on `perf/v41-decode-graph` (the tree of PR #278); extension `cuda_kernel.cpython-311-x86_64-linux-gnu.so`, built 2026-09-17 00:46 |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `--max-seq-len 2048`, `--steps 8 --warmup 3 --max-graphed 40`, `temperature = 0.0` |
| Prompt | 1024 and 1025 tokens of the e2e probe's prose, then a **fixed** token re-forwarded at the frozen position |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, cross-pairs SYS; all four idle (1 MiB, 0%) before and after |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Probe | `/tmp/probe_v41_split_graph.py`, written for this round; capture machinery reused from `/tmp/probe_v41_graph_size.py`. Run 2026-09-18 04:32–05:12 (41 min), `/tmp/split_graph_full2.log` |

The resident bank is **not** attached, so every absolute ms/step below is a warm-page-cache figure by
the rule the [device-experts page](deepseek_v4_1_flash_device_experts.md) sets, and the numbers are
not comparable to that page's 722.0 ms or to the remaining-bottlenecks page's 722.0 / 3.0 tok/s
sittings. It does not affect the graph's own delta: the dense tree reads no experts, and the expert
call is the same eager call on both sides of the comparison.

`temperature = 0.0` is not a convenience. The sampler is Gumbel-max over the logits, its draws differ
between columns because the capture pass runs extra bodies, and pinning it to argmax removes the draws
from the comparison without touching the logits or either column's cost. The two tokens reported at
the end — 295 at 1024 and 793 at 1025 — are the argmax at those positions, not a generation, and the
parity signal on this page is the logits column rather than any token id.

## The split is forced, and it is half the step

The step is **graph A → eager expert call → graph B**, per layer, forty times, and the split is not a
design choice. `routed.forward` cannot be captured because the host has to act in the middle of it:
`_route_ids` does a pinned D2H plus a `torch.cuda.current_stream(...).synchronize()`, and
`_resolve_row`'s `[int(e) for e in ids_row.tolist()]` is a host read per row
(`src/models/deepseek_v4_1/device_experts.py:1794`, **2.9 ms a row**). That read is the point of a
host-resident expert bank, so a graph over the whole step is not available at any price — and the
half that is available is the half this page measures.

Both halves are captured per layer against preallocated buffers: **A** = block input + `pre_mix` →
the ffn-pre activation plus the gate's `(weights, indices)`; **B** = the ffn-post activation → the
block output and its `pre_mix`. Eighty graphs in all, two a layer, sharing one
`torch.cuda.graph_pool_handle()`.

Three pageable H2D copies are capture-hostile and are patched **only for the duration of the graphed
column**, so the eager column pays them exactly as the shipped path does: `ops._fp4_levels`
(`src/kernels/ops.py:115`), `kernels._fp4_codes` / `_fp4_values` (`kernels.py:144` / `:158`), and
`attention.get_window_topk_idxs`'s CPU-built table. The codebook pair is reimplemented rather than
wrapped — the pageable `torch.tensor` is inside the function — so it is diffed against the library's
own construction on random inputs first: `codebook diff against the library's own: 0.0` in both
rounds.

## The position is frozen, and two of them cover the compressor

A capture records the position into the graph: the `freqs_cis` slice and the ring slot are Python at
record time. So each frozen position is its own capture round, and the probe decodes **as a real
step** — runs the layer bodies for real, records the graphs against the activation the model actually
produced, replays immediately so the capture pass is also a correct forward, then `.copy_`s the
per-layer caches (`window_kv_cache`, `compress_kv_cache`, `k_cache`, `kv_state`, `score_state`) back
from a snapshot. Without the rewind the capture pass's own writes would be the graphed column's
starting state and the two columns would be one fixed-point iteration apart, which on a recurrence
reads exactly like a wrong graph.

Two positions are measured because the compressor is the one place where the position changes what the
graph **contains**. It emits a row only when `(start_pos + 1) % compress_ratio == 0`, and
`compress_ratios` is **0 for layers 0–1, 2 for layers 2–19, and 1 for layers 20–39**, with
`kv_source_layer_ids = [2, 8, 14, 20]` — four layers carry the source, the other thirty-six read the
cache.

At **1024** (1025 % 2 = 1) the ratio-2 sources capture a **fill** body with **no quantizer node at
all**, and only layer 20 quantizes; at **1025** (1026 % 2 = 0) all four sources capture the **emit**
body with the quantizer in. Both variants are therefore in the tables below, and the configuration
that captured only the fill body — layer 8 at 1024, the 1.01 ms/step layer — is not what these numbers
come from.

## The measurement

Forty of forty layers graphed, four ranks, eight timed steps a column, three warmups, the position
frozen; `ms/step` measured with one synchronize around the whole loop:

```text
40 of 40 layers graphed, 4 ranks, 8 steps a column, 3 warmups, start_pos frozen
   pos       column   ms/step   graph A   routed   graph B  pool GiB  resv GiB    max diff
  1024        eager     833.6                                  0.000     0.000   0.000e+00
  1024        graph     561.6      32.2   5993.1      23.6     0.254     0.633   0.000e+00
  1024  eager-again     828.5                                  0.000     0.000   0.000e+00
  1025        eager     843.8                                  0.000     0.000   0.000e+00
  1025        graph     566.1      37.4   6112.5      27.0     0.000     0.021   0.000e+00
  1025  eager-again     856.8                                  0.000     0.000   0.000e+00
```

**The A-A controls are the bar the delta has to clear, and they are not small.** `eager` against
`eager-again` is 833.6 against 828.5 (**0.6%**, 5.1 ms) at 1024 and 843.8 against 856.8 (**1.5%**,
13.0 ms) at 1025, so the same binary on the same input disagrees with itself by up to 13 ms. Read
against each position's first eager column the falls are **272.0 ms** and **277.7 ms** (1.48× and
1.49×); read against the controls they are 266.9 ms and 290.7 ms. The honest bar for the change is
therefore **~267–291 ms** and the honest ratio **1.48–1.51×** — 20–50× the spread, so the effect is
real and the direction is not in doubt, but a second decimal on the ratio would be pretending to a
precision the controls do not have.

**`max diff` is the graphed column's logits against the first eager column at the same position**, and
it is exactly zero at both positions — not "close", not "within fp16 rounding". The capture pass is a
real forward over the same state, so the two columns start from the same place and a missing,
reordered or wrong-shape kernel would show up as a bit rather than as a drift.

**The three marks cannot be read as the split**, and the table says so at its own foot: the expert
call synchronizes on the host, so graph A's GPU work continues into that window and its tail is
charged to `routed` (5993.1 ms over 40 layers × 8 steps ≈ 18.7 ms a layer-step), while graph B's
replay overlaps the next layer's A. `graph A` and `graph B` are therefore launch clocks, and the step
total — one synchronize around the whole loop — is the only number on the table that means anything.
The per-layer accounting the per-layer probe already has (213 eager kernels + 244 host API calls
against 214 + 4 replayed) is the right instrument for the mechanism; this table is the right
instrument for the end-to-end price.

The graph column's 0.633 GiB of newly reserved memory is everything that column reserved, of which
0.254 GiB is the graphs' own pool. The steps themselves allocate nothing new.

## What the graph pool costs, and why the projection was 580× wrong

The plan's risk was memory: ~3.7 GiB a layer × 40 is ~148 GiB against 22 GiB a card, which would have
ended the idea before it started. The measured pool is **0.254 GiB for all forty layers — 6.5 MiB a
layer**, which is 580× under the projection. The projection was taken from a 512-token synthetic
activation; the decode capture body is **one token wide**, and a graph's private pool holds what the
recorded body allocated, not what the layer's weights occupy (those are outside it and already
resident).

Two reading rules for that column:

- **Read the 1024 row for the pool's size.** The column is a delta measured around the capture, and
  the 1025 round is the second round in the same process, capturing the same eighty graphs over the
  same forty layers into a fresh handle after the first round's release — so it reports what the delta
  saw (0.000 GiB, 0.021 GiB newly reserved) rather than what its pool holds. That is an artifact of
  measuring a delta twice in one process, not a second round that is free.
- **The eager rows' `0.000` is "not applicable", not "measured zero."** The pool and reservation
  columns are only taken in the capture branch; the eager columns allocate no graph memory by
  construction.

## A pool id cannot be reused once its graphs die

This is the one trap this round cost a run to, and it is worth its own paragraph because Phase B's
rebuild path walks straight into it. `torch.cuda.graph_pool_handle()` is a bare `(id, generation)`
tuple (`torch/cuda/graphs.py:64`) with no lifetime semantics at all — holding a reference to it pins
nothing. The allocator's entry for that id lives only as long as some graph captured into it does:
`capture_begin(pool=…)` → `create_or_incref_pool` increments a `use_count`, destroying the `CUDAGraph`
decrements it, and at zero the entry moves to `graph_pools_freeable` and is erased only if
`release_cached_blocks` can free *all* of its memory.

So a second capture round with the same id, after the first round's graphs have been dropped — which
is exactly what a second frozen position is — can die inside the allocator:

```
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED
  at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":2630
```

It landed on the first layer of position 1025's graph column on the first attempt
(`/tmp/split_graph_full.log`), **after** position 1024's three columns had all completed — and it took
the finished round's numbers with it, because the report was printed at the end of the process. Both
halves of that are fixed in the probe as it now stands: **a fresh `graph_pool_handle()` per capture
round** before the `empty_cache()`, and the table printed at the end of each position rather than
after the loop. Reproduced in isolation one experiment a process (the allocator is unusable after a
failed `capture_begin`, which raises `Offset increment outside graph capture encountered
unexpectedly`): with a tensor allocated *inside* the capture still held, the same id asserts; with
that tensor dropped, or after an `empty_cache()`, it is reusable; a fresh handle works over three
rounds unconditionally. So the trap is intermittent — it depends on whether anything still holds pool
memory — and the rule that is safe without checking is the fresh handle.

## What this page does not say

- **It is not a decode loop.** Every step re-forwards the same token at the same position. There is no
  token-to-token comparison here and no TG-n decode rate; `start_pos` has to reach the card before
  either exists.
- **It says nothing about prefill.** The prompt forward is never graphed (`driver.enabled` stays off
  until it is done), because a capture recorded at `start_pos = 0` over 1024 tokens is a
  prefill-shaped body and replaying it before a decode hands the next layer a 1024-token activation
  when it expects one.
- **It is a warm-page-cache figure with no resident bank attached.** Per the device-experts page's
  rule, that is a different configuration from the 722.0 ms step, and the two must not be subtracted
  from each other.
- **It does not price the expert half**, which is untouched and still the majority of what is left:
  the 561.6 ms step is the graphed tree half plus the same eager `DeviceRoutedExperts` call. The tree
  half's own residual after the graph is not derivable from this table for the reason above — the
  marks overlap by construction — and is the first thing the next round should measure directly.
- **It does not authorize anything by itself.** It clears Phase A's gate: forty of forty layers fit,
  and the capture pass reproduces the eager step bit for bit at a fixed position.

## What it unblocks

The library changes that let the position actually advance, which are the only thing missing between
this measurement and a real graphed decode: a `Pos` abstraction over "Python int or 0-dim device
tensor" used at the six position-shaped sites in `attention.py`; the `compress_len` prefix — which
grows every step and therefore cannot be captured as a slice — replaced by the full cache width plus a
mask, the way the prefill branch already masks; `Indexer`'s `topk = min(index_topk, end_pos // ratio)`
made `index_topk` on the device path, where unreachable slots are already `-1`; the compressor's emit
and fill bodies selected as two captured variants; the per-device codebook caches, which are a fix
worth having on their own; a `DecodeGraphs` owning the per-layer pairs and the shared pool, taking a
**fresh pool handle on any rebuild**; and a flag on the existing `src/cli/generate_v41.py`, default
off, behind which 64 tokens of real greedy decode is compared against the eager path token by token.

## Reproducing

```bash
# Phase A in full: both frozen positions, three columns each, forty layers, four ranks.
PYTHONPATH=/mnt/data1/dsv4_inference torchrun --nproc_per_node=4 \
    /tmp/probe_v41_split_graph.py --positions 1024,1025
#   defaults: --checkpoint /mnt/data3/DeepSeek-V4.1-Flash --steps 8 --warmup 3 --max-graphed 40
#             --threads 22 --max-seq-len 2048. One capture round a position, ~20 min each.

# The pool-id trap on its own, one experiment a process -- the allocator is unusable after a failed
# capture_begin, so `fresh`, `dropped` and `emptied` cannot share one. Only `dropped`/`emptied` are
# expected to be reusable with the same id; `fresh` takes a new handle each round and always is.
python /tmp/pool_probe2.py fresh
python /tmp/pool_probe2.py dropped
python /tmp/pool_probe2.py emptied
```

`torch.distributed.run` sets `OMP_NUM_THREADS=1` for every worker, so the probe calls
`torch.set_num_threads(22)` itself and `--threads` is what makes the thread count mean anything. Read
`max diff` and the A-A controls before reading any ratio on this page.
