# DeepSeek-V4.1-Flash: a real graphed decode, token for token

The [frozen-position page](deepseek_v4_1_flash_decode_graph.md) ends with a list of what it unblocks
and one sentence about what it does not have: **it is not a decode loop**, because `start_pos` is a
Python int that picks both the `freqs_cis` slice and the ring-buffer slot, and a capture records the
int. **This page is that loop.** Sixty-four tokens of greedy decode past a 1024-token prompt, the
position advancing 1024 → 1087, three legs each in its own process, and the tokens compared one by
one against the eager path.

**The decode step falls from 638 to 408–431 ms a token — 1.43–1.56× — and all 64 tokens are
bit-identical**: `max |logit diff| = 0.000e+00` on every row of a 64 × 129280 logits matrix, 64/64
argmax agreement, and `read_bytes` flat at 0 for every token of every leg.

Two columns of the same binary, three legs:

| Leg | Flag | Decode | Step | Whole call |
| --- | --- | --- | --- | --- |
| `eager_a` | `--no-decode-graphs` | 40.9 s / 64 | **638 ms/token** | 81.6 s |
| `eager_b` | `--no-decode-graphs` | 39.6 s / 64 | **618 ms/token** | 74.5 s |
| `graph` | `--decode-graphs` | 27.6 s / 64 | **431 ms/token** | 70.3 s |
| `graph` (repeat) | `--decode-graphs` | 26.1 s / 64 | **408 ms/token** | 62.9 s |

`Step` is the decode loop alone, which is the column the two paths are comparable on; the whole call
carries a 23–26 s load and the 1024-token prompt forward as well, and it is 5–9 s shorter on the
graphed legs only because a second sitting warms the page cache further. **Prefill is untouched and
not in scope.**

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, **resident bank attached** (`DEEPSEEK_V41_RESIDENT_EXPERTS=1`, 457.8 GiB attached) |
| Commit | `e70cc12` + the Phase B working tree on `perf/v41-decode-graph`; extension `cuda_kernel.cpython-311-x86_64-linux-gnu.so` |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `--max-seq-len 1088`, `--max-new-tokens 64`, `temperature 0.0` (the default) |
| Prompt | 1024 tokens of `/tmp/prompt1024.txt`; context 1088 |
| Decode | 64 steps, positions 1024 → 1087, first token 271, last 21779 |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, all four idle before and after |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Driver | `/tmp/run_parity.sh`; comparison `/tmp/cmp_logits.py`; `/tmp/b_parity_*.log`, `/tmp/dump_*.pt` |
| Gate probe | `/tmp/probe_v41_graph_nodes.py`, `/tmp/b_nodes3.log`, `/tmp/v41nodes3/` |

The resident bank is attached here, which the frozen-position page's run did not have, so **the two
pages' absolute ms/step are different configurations and must not be subtracted from each other**.
It does not touch the comparison: the dense tree reads no experts, and the expert call is the same
eager call in both columns.

`temperature = 0.0` matters for a different reason than it did on that page. The 64 tokens are a real
greedy continuation — the text below — and greedy is what makes a token-by-token comparison possible
at all: a Gumbel draw would differ between columns for reasons that have nothing to do with the graph.

```text
ote in his notebook that a city is a question its river has already answered. Nobody in the hall
looked up.

Chapter 12. The traveller reached Stockholm on a wet morning in late autumn, having followed the
Elbe for the better part of a week. The station was quiet, the platform lamps still burning
```

## What had to change for the position to advance

Six sites in `attention.py` are shaped by the position, and every one of them had to accept either a
Python int — what the eager path passes today — or a 0-dim CUDA int64 tensor, without the eager path
changing at all. `src/models/deepseek_v4_1/decode_pos.py` is that: a `Pos` with `row`, `span`,
`slot`, `pick`, `upto`, `group`, `first` and `emits`, each returning the *same kind of index object
the site already builds by hand*.

| Site | Before | Now |
| --- | --- | --- |
| `freqs_cis[start_pos : start_pos + seqlen]` | a Python slice | `pos.span(seqlen)` |
| `window_kv_cache[:bsz, start_pos % win] = v` | an int index | `pos.slot(win)` |
| `get_window_topk_idxs`, decode branch | a CPU `cat` of two `arange`s, `.to(device)` | rotation, `(arange(win) + pos.slot(win) + 1) % win`, then `where(idxs > pos.row(), -1, idxs)` |
| `_compress_kv`'s cache write | `start_pos // ratio` prefix | `pos.group(ratio, seqlen)` |
| `compress_len` | `(start_pos + seqlen) // ratio`, a **growing** prefix | the **full cache width**, with the tail masked in the score |
| `Indexer`'s `topk` | `min(index_topk, end_pos // ratio)` | `index_topk` always, because unreachable slots already become `-1` |

`compress_len` is the one that is not a translation. The old value is a prefix length that grows by
one every `ratio` steps, so it cannot be captured as a shape — a graph has one shape. The new value
is the cache's full width, a per-config constant, and the slots beyond the live prefix are masked to
`-inf` exactly the way the prefill branch already masks them. That is why every downstream shape in
the graphed step is fixed.

The compressor's **emit** and **fill** bodies are captured as two variants, because the position
decides which one runs and a graph can only hold one: `(start_pos + 1) % ratio == 0` is a host bool,
so `Compressor.forward` takes it as `compress_variant` and the capture pass records both.

Three pageable H2D copies were on the hot path in their own right and are cached per device now, which
is a fix worth having outside the graph: `ops._fp4_levels` and `kernels._fp4_codes` / `_fp4_values`
built a codebook with `torch.tensor(<python list>, device=...)` on every call. Inside a capture that
is not slow, it is a hard failure.

`src/models/deepseek_v4_1/graphs.py` owns the forty pairs, the shared pool, the capture pass and the
snapshot/restore; `Block.forward` hands itself to it when a decode step is being replayed and
otherwise runs unchanged. `--decode-graphs` on the existing `src/cli/generate_v41.py` selects the
path, and it is **off by default**.

## The A-A control is the bar, and it is 3%

`eager_a` and `eager_b` are the same command twice in two processes: 638 against 618 ms a token, a
**3.2%** spread with identical output. So the change is read as a range rather than a ratio: the
graphed step is **207–230 ms** below the first eager column and **187–210 ms** below the second,
which is **1.43–1.56×**. The graphed column's own two sittings (431 and 408) differ by 5.3%.

That is 12–20× the spread, so the direction is not in doubt and a second decimal on the ratio would
be pretending to a precision these controls do not have. The headline this page is willing to defend
is **the step falls by roughly 190–230 ms, about 1.5×**, on a step whose expert half is untouched.

The three host marks say where the rest is. On the graphed legs, `2.4 + 369.4 + 2.7`, `2.0 + 378.9 +
2.2`, `2.1 + 370.1 + 2.4` and `1.8 + 379.5 + 2.0` ms (graph A / eager experts / graph B): the tree
halves together are **~4.5 ms of the 408** and the eager expert call is **~370 ms of it**, which is
89% of what a graphed decode step still costs. The marks are launch clocks and the expert call
synchronizes, so they are a share and not a profile — but the split is now unambiguous, and it is the
same statement the frozen-position page could only make by inference.

## The gate: what the recordings actually contain

The plan's rule is that the compressor variants and the `compress_len` width **are the two places a
wrong graph is silent** — the round before this one measured an 11.58× that was really 10.76× because
a quantizer node was missing from a capture. Both get a node-level check rather than a timing check:
every graph is dumped with `cudaGraphDebugDotPrint`, the kernels are demangled out of the DOT, and
the counts are printed a layer at a time. `/tmp/b_nodes3.log`, all four ranks.

**Every layer, forty of forty.** `a` is a graph-A node count, `b` graph B:

| Layers | Graph A | Graph B |
| --- | --- | --- |
| 0–1 | 145 | 18 |
| 2, 8, 14 | **emit 307 / fill 208** | 18 |
| 3–7, 9–13, 15–19 | 149 | 18 |
| 20 | 310 | 18 |
| 24, 28, 32, 36 | 204 | 18 |
| the rest of 21–39 | 149 | 18 |

Forty layers, 260.0 MiB of shared pool, captured in 1.6 s per rank — the 0.254 GiB / 6.5 MiB a layer
the frozen-position page measured, unchanged.

**`[2, 8, 14]` and not `[2, 8, 14, 20]` is the correct answer to the variant question.** `Compressor`
is constructed only when `is_kv_source`, and `kv_source_layer_ids = [2, 8, 14, 20]` — but `graphs.py`
takes `dual = compressor.compress_ratio > 1`, and layer 20's ratio is **1**. Layers 3–19 have ratio 2
and no compressor at all, because they are not kv sources. The three dual layers are exactly the
ratio-2 kv sources.

**The 99-node gap inside each of them is the row-production path, and the compressed-KV quantizer
is inside it.** An exhaustive per-symbol diff of demangled names for layer 2 gives **96 distinct
symbol names in the emit graph against 84 in the fill graph**, the fill set a strict subset of the
emit set — **no symbol is larger in fill** — and `Σ(emit − fill) = 99`, byte-identical across layers
2, 8 and 14.

That the gap is not one op is shown by the graph beside it. **Layer 20 is a kv source whose ratio is
1**, so it has no pooling and no `kv_state` to fill, but it produces a row on *every* step and
therefore runs the row-production path unconditionally: its graph is `a = 310` over 101 symbols
against emit's 307 over 96, and the two differ by **two symbols in one direction and seven in the
other, each with a mechanical reason**. Emit-only: `cunn_SpatialSoftMaxForward` (ratio 1 has nothing
to pool) and `index_copy_kernel` (ratio 1 has no `kv_state`/`score_state` to write a slot into).
Layer-20-only: `radixSortKVInPlace` and `ReduceOp<c10::BFloat16>` — layer 20 is
`candidate_source_layer_id`, so it is the layer that runs `select_candidate_blocks` over its 2048
blocks — plus the `FillFunctor`/`compare_scalar` companions of that. So the 99 nodes are the pooling,
the RoPE and quantizer on the latent, and the cache write, and the indexer's extra work over the
position the row created.

**The quantizer is in the emit variant and absent from the fill variant, which is exactly what this
gate exists to catch.** `fp4_act_quant_e4m3` is not a CUDA kernel — it is ATen ops in `kernels.py` —
so its nodes carry ATen names, and **25 of the 99 are its family**:
`searchsorted_cuda_kernel<float, long>` is `_fp4_codes`' codebook lookup (`torch.searchsorted`,
`kernels.py:160`) and it is **emit-only, 1 → 0**; the two `indexSelectSmallIndex<c10::complex<float>>`
are `levels[lower]` / `levels[upper]`; `index_elementwise_kernel` is the `_fp4_values` gather and the
`index_put` the cache write lowers to; `launch_clamp_scalar`, `compare_scalar_kernel<float>` and the
`AUnaryFunctor<long>` / bool-functor casts are the rounding and the tie-break.

**There is no `quantize_rows_kernel` on either side, and that is not a missing node**: it belongs to
the C++ engine's quantizer, not to `kernels.py`, whose E4M3 path is composed of ATen ops. The
**indexer's** own E8M0 scale (`2 ** ceil(log2(·))`, `ops.py:102`) is the reason to trust the split
rather than the names: it shows as a **shared** increase (`ceil` / `log2`, 3 → 2) and not an emit-only
one, because `_compress_topk_idxs` — indexer included — runs in both variants and only
`_compress_kv`'s `if latent is not None:` block is emit-only. Every emit-only symbol is accounted for
that way, and none of them is a *missing* one.

**The step's own kernel multiset.** The probe replays a whole step — graph A, the eager routed call,
graph B, per layer — and profiles it against the same step run eagerly:

```text
eager:  7515 kernels with device time, 9140 host API calls, 546 copies/memsets
replay: 8359 kernels with device time,  962 host API calls, 360 copies/memsets
```

`cudaLaunchKernel 6801 -> 320`, `cudaStreamWaitEvent 304 -> 40`, `cudaMemcpyAsync 546 -> 360`,
`Memcpy DtoD 218 -> 80`, `cudaStreamIsCapturing 298 -> 200`, `cudaStreamSynchronize 168 -> 120`. The
replay's kernel count is **higher** while its host-API count is 9.5× lower: a graph replays every node
it recorded, including ones the eager column elides at a given position, and the long tail of `-1`/`-2`
deltas is the profiler's warm-up, not a missing node.

Three corrections had to be made to the probe before this diff was readable at all, and each of them
is a way the check could have lied:

- **The replay column has to run the same computation as the eager column.** The first version left
  out the routed expert call on the theory that it is eager in both columns and cancels. It does not:
  the two columns then differ by the experts' forty kernels a step, and all forty `moe_single_*` and
  `quantize_rows_kernel` entries come back as "missing from the replay" and bury the answer.
- **The replay column has to run under `torch.inference_mode()`**, the way `generate.py` wraps the
  decode loop. Without it the run died on all four ranks with *Inplace update to inference tensor
  outside InferenceMode is not allowed*.
- **`cudaGraphDebugDotPrint` needs `enable_debug_mode()` before capture and `keep_graph=True`.**
  Without both, the raw `cudaGraph_t` is destroyed once the executable is instantiated, printing it
  writes no file, and raises nothing. The DOT writer escapes `<<<` to `\<\<\<`, so a regex looking for
  a bare `<<<` parses **zero nodes** — and a dump that parses to zero nodes looks exactly like a graph
  with no kernels in it.

With those three fixed, what is left "in the eager step and not in the replay" is host API calls
(`cudaEventRecord` 352, `cudaEventQuery` 255, `cudaThreadExchangeStreamCaptureMode` 168,
`cuLaunchKernelEx` 168, `cudaDeviceGetAttribute` 160, …) plus a small tail that is each attributable
and none of which is a missing computation: `compare_scalar_kernel<long>` 20 is the probe's own
`int(logits[0].argmax())` from the prefill, `_scatter_gather_elementwise_kernel` 4 with its four
`Memcpy DtoH` is `_route_ids`' pinned readback once at capture against once a step eagerly,
`CatArrayBatchedCopy_alignedK_contig` 2 is the prefill branch's `freqs_cis` slice, and
`where_kernel_impl` / `launch_clamp_scalar` 5 → 1 with `BinaryFunctor<bool,…>` 9 → 1 is the five
`get_window_topk_idxs` calls (eight launches each eagerly, one each in the replay).

## Residency, counted

The graphed leg's own report, identical on all four ranks:

```text
card memory: 14.73 GiB allocated, 15.28 GiB reserved, 15.06 GiB peak allocated
decode graphs: 40 layers, 65 replays each, 260.0 MiB of shared pool
```

`reserved` is the column to read against the 22 GiB budget (`gpu_memory_budget`): what the card no
longer has for anything else. 15.28 GiB of 22.0, with the expert arena (288 pooled rows a card), the
KV caches and both graph halves already inside it. The graphs' own share is **260.0 MiB**, 1.7% of
what is reserved. `peak allocated` (15.06) sits just under `allocated` + the pool, so the capture pass
does not spike: the forty layers are captured into one shared pool and nothing else is allocated
around them.

The pool is a total, not a layer figure, and it is 580× under the plan's own 148 GiB projection for
the reason the frozen-position page gives: a decode capture body is one token wide.

## What this page does not say

- **It says nothing about prefill**, which is not graphed and is a separate round. The 1024-token
  prompt forward is eager on every leg, and the graph driver stays disabled until it is done.
- **The expert half is untouched.** ~370 of the 408 ms is the eager `DeviceRoutedExperts` call, and a
  graph cannot reach it: `_route_ids` synchronizes on the host and reads a row's ids back as Python
  ints. The tree half is now ~4.5 ms; the step is the expert call plus a rounding error — and that
  term has since moved on its own, by registering the resident bank and deleting the copy `_stage` made
  into the pinned arena: the same CLI, prompt and 64 tokens at **200/202 ms a token with the
  registration against 341–348 without it**, 1.70–1.72×, of which the expert call is 178.5–184.3
  against 309.0–323.2. Read that as
  [the device-experts page's closing section](deepseek_v4_1_flash_device_experts.md#the-pin-and-the-copy-removal-landed-200-ms-a-decode-token-and-160-s-a-prefill);
  this page's 408 is the same configuration with the copy still in the path.
- **It is a warm-page-cache figure with the resident bank attached.** Per the
  [device-experts page](deepseek_v4_1_flash_device_experts.md)'s rule, that is a different
  configuration from both that page's 722.0 ms and the frozen-position page's 833.6 ms.
- **The ratio is 1.43–1.56×, not a point estimate.** The A-A control is 3.2% and the graphed column's
  own two sittings differ by 5.3%; a run with a wider prompt or a longer generation would narrow it.
- **`--decode-graphs` is off by default** and this page is not, by itself, an argument to turn it on.
  What it is: the gate is cleared — real positions, real greedy decode, 64 tokens token-aligned, the
  node counts accounted for.

## Reproducing

```bash
# The three legs, one configuration a process, 64 tokens each, logits dumped a row a token.
/tmp/run_parity.sh          # ~5 min; -> /tmp/b_parity_{eager_a,eager_b,graph}.log, /tmp/dump_*.pt
python /tmp/cmp_logits.py /tmp/dump_eager_a.pt /tmp/dump_eager_b.pt /tmp/dump_graph.pt
#   read_bytes flat on every token of every leg, 64/64 tokens identical to eager_a,
#   max |logit diff| = 0.000e+00 on all 64 rows, argmax 64/64.

# One graphed leg on its own, for the residency line.
/tmp/run_graph_mem.sh       # -> /tmp/b_parity_graph2.log, /tmp/dump_graph2.pt

# The node-level gate: forty layers dumped to DOT and parsed, the variant diff, the step multiset.
/tmp/run_nodes2.sh          # ~2 min; -> /tmp/b_nodes3.log, /tmp/v41nodes3/
/tmp/diffvars.py            # the per-symbol diff of any two DOT dumps (c++filt on stdin, not argv)
/tmp/sets.py                # that diff's symbol sets against layer 20's, and the L20-only rows
```

`torch.distributed.run` sets `OMP_NUM_THREADS=1` for every worker, so `--threads 22` is what makes the
thread count mean anything. Read the A-A control before reading any ratio on this page.
