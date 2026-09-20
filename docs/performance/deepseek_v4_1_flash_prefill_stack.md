# DeepSeek-V4.1-Flash: the three prefill optimizations, stacked

The [chunked-prefill page](deepseek_v4_1_flash_chunked_prefill.md) published the phase table of a
4096-token chunk at 32768 and named its three largest rows. The three passes that followed each took
one of them — [#296](https://github.com/lvyufeng/PocketLLM/pull/296) the sparse-attention score pass,
[#297](https://github.com/lvyufeng/PocketLLM/pull/297) the two grouped fp4 kernels' weight reads, and
[#298](https://github.com/lvyufeng/PocketLLM/pull/298) the reduction inside them — and each was
measured against the tree it was cut from. **This page is the composition instead of the parts:** one
arm on the 256K branch, one on the 256K branch with all three merged, the identical geometry, the
identical probe, all four ranks.

The chunk goes **65.94 → 48.48 s instrumented** and **56.87 → 38.07 s with the taps off (‑33.1%)**.
The three changes' own measured deltas — 5.33 s from #296's A/B, 8.98 s from #297's, 4.58 s from
#298's — add to 18.89 s against the 18.80 s measured here, so **they stack additively**: no overlap,
no interference, and the two that share a kernel launcher (#297 and #298) are independent in the
composition as well as in their own measurements. What the chunk is *made of* changes shape rather
than only shrinking. The sparse pass falls to a fifth of its former size and the grouped GEMM's two
kernels lose two thirds of theirs, while the expert H2D goes from 16.71 to 17.38 s — **the copies are
now 46% of the instrumented chunk and the largest single row in it**, and the third lever the
chunked-prefill page named is not the host's row loop, which this run measures at 1.52 s of 163,840
calls in both arms. The logits move and the decisions do not: the stacked tree picks the same token
as the base tree at all nine dump positions, and the stacked tree run twice is bit-identical.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | base arm: `feature/v41-256k-context` tip `98e828f` (itself `perf/v41-hc-token-tile` `83ed600` plus the 256K work). Stacked arm: `perf/v41-prefill-integration-256k` tip `b7f633a`, which is `98e828f` with three merges — `3d64eca` (`perf/v41-prefill-sparse-attn-warp-dot`, tip `ef53553`), `4f27695` (`perf/v41-moe-multi-coalesced-weights`, tip `0d5a4ec`), `b7f633a` (`perf/v41-moe-reduce-csr`, tip `38d453c`) |
| The two arms | the base tree's own `cuda_kernel.cpython-311-x86_64-linux-gnu.so` (12709712 bytes, md5 `8cb40d947ae3a9a3e23fcfe0a43781cc`) against the stacked tree's (12853624 bytes, md5 `b4022f063272ee23ad242c8eb7f88c5c`) — and, unlike the [weight-staging page](https://github.com/lvyufeng/PocketLLM/pull/297), the Python differs too, because the coalesced and CSR call sites live in it; **this is two trees run in turn, not one binary swapped** |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--at 32768 --chunk 4096 --prefill-chunk 4096 --max-seq-len 41024 --pool-rows 148 --buffers 2 --threads 22`, resident bank on |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, cross-pairs SYS |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env, `CUDA_HOME=/usr/local/cuda-12.4`, `TORCH_CUDA_ARCH_LIST=7.5`, `POCKETLLM_BUILD_CPP=0` |
| Probe | `/tmp/probe_v41_chunk_profile_host.py`, once an arm: warm up eight chunks to 32768, then 22 taps with the barrier split out of each number over one 4096-token chunk, then the same width again with the taps off. Logs `/tmp/chunk_profile_base.log` (23:40) and `/tmp/chunk_profile_all.log` (23:24) |
| Parity | `/tmp/pr_b_parity.py`, three arms through `/tmp/parity_stack.sh`: the base tree, the stacked tree, and the stacked tree again as the A-A control, compared per rank with `--compare` |

**On the two binaries.** The control this page has is the tree, not a byte-identical extension: both
arms had their `cuda_kernel` rebuilt *after* their probe run, so the installed `.so` of each arm is
whatever was in place at 23:24 and 23:40 and is not in the payloads, and neither of the two md5s above
is the one #297 or #298 recorded for its own A/B — those were 9,490,256 and 9,416,528 bytes, against
12,709,712 and 12,853,624 here, the gap being the `.nv_fatbin` arch list rather than the kernels. What
makes the composition readable anyway is that the two trees differ in the Python as well, so "one
binary swapped" was never the design, and that the rows neither change touches — `attn.window`
0.81 → 0.82, `hc_mixes`/`hc_pre`/`hc_post`, `norm`, `engram` — agree across the arms to within 2%,
which is the shared code doing the same thing in both runs. The three changes' own A/Bs, in the
section below, were each taken on the tree they name.

## The chunk, both arms

Rank 0, seconds, `body` being the call between the tap's two barriers and `sync` the barriers
themselves. The column order in the payload is `[total, calls, sync, body]`; `total = body + sync`.
Nested rows are indented, and a nested row's `total` is contained in its parent's `body`.

| phase | calls | base total | base body | base sync | stacked total | stacked body | stacked sync |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hc_mixes` | 80 | 0.40 | 0.13 | 0.27 | 0.40 | 0.13 | 0.27 |
| `hc_pre` | 81 | 0.29 | 0.03 | 0.25 | 0.29 | 0.03 | 0.25 |
| `attn` | 40 | 13.04 | 11.34 | 1.70 | **6.79** | 5.23 | 1.56 |
| `  attn.sparse` | 40 | 8.07 | 0.00 | 8.06 | **1.98** | 0.00 | 1.97 |
| `  attn.window` | 40 | 0.81 | 0.40 | 0.41 | 0.82 | 0.42 | 0.41 |
| `  attn.compress_kv` | 38 | 2.14 | 2.14 | 0.00 | 2.11 | 2.11 | 0.00 |
| `    attn.compressor` | 4 | 0.02 | 0.02 | 0.00 | 0.02 | 0.02 | 0.00 |
| `    attn.indexer` | 8 | 2.11 | 2.11 | 0.00 | 2.09 | 2.08 | 0.00 |
| `hc_post` | 80 | 0.96 | 0.05 | 0.91 | 0.96 | 0.05 | 0.92 |
| `moe` | 40 | 50.64 | 47.78 | 2.86 | **39.39** | 36.44 | 2.95 |
| `  moe.gate` | 40 | 0.08 | 0.02 | 0.06 | 0.08 | 0.02 | 0.06 |
| `  moe.routed` | 40 | 47.28 | 47.28 | 0.00 | **35.95** | 35.95 | 0.00 |
| `    routed.route_ids` | 40 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `    routed.resolve` | 163,840 | 7.00 | **1.52** | 5.53 | 7.26 | **1.52** | 5.80 |
| `    routed.stage` | 163,840 | 19.77 | 17.04 | 2.77 | 20.72 | 17.75 | 3.02 |
| `      routed.upload` | 8,634 / 8,978 | 16.71 | 2.34 | 14.37 | 17.38 | 2.37 | 15.02 |
| `      routed.buffer` | 8,634 / 8,978 | 0.20 | 0.04 | 0.16 | 0.22 | 0.04 | 0.18 |
| `    routed.issue` | 156 / 165 | 18.18 | 0.57 | **17.61** | **5.49** | 5.24 | **0.25** |
| `    routed.drain` | 156 / 165 | 0.35 | 0.34 | 0.01 | 0.38 | 0.37 | 0.01 |
| `  moe.shared` | 40 | 0.39 | 0.03 | 0.36 | 0.38 | 0.02 | 0.36 |
| `norm` | 169 | 0.54 | 0.03 | 0.51 | 0.54 | 0.03 | 0.51 |
| `engram` | 2 | 0.47 | 0.38 | 0.09 | 0.49 | 0.41 | 0.09 |
| **five phases** | | **65.33** | 59.33 | | **47.83** | 41.88 | |

| | base | stacked |
| --- | ---: | ---: |
| instrumented chunk, rank 0 | 65.94 | 48.48 |
| the same width with the taps off | 56.87 | 38.07 |
| what the instrument costs | 9.07 | 10.41 |
| names / calls recorded | 22 / 346,042 | 22 / 346,748 |
| five phases cover / their bodies cover | 99.1% / 90.0% | 98.7% / 86.4% |
| warm-up: first chunk, the other seven, spread | 74.80–75.83, 65.71, ≤1.39 | 57.79–60.75, 46.84, ≤1.26 |
| expert rows staged | 9,509 | 9,843 |
| peak allocated | 15.05 GiB | 15.12 GiB |

All four ranks report the same wall in both arms — 65.94/65.94/65.94/65.95 and
48.48/48.48/48.48/48.48 — with the identical taps-off chunk (56.87/56.86 and 38.07/38.06), and the
tables above are rank 0's; ranks 1 and 2 differ in where the same seconds sit, which is what the
per-rank `staged` column (9,509/9,516/5,029/5,020 against 9,843/9,918/5,106/5,042) is: the expert
split is 2/2/1/1 rows a layer, so ranks 2 and 3 stage half as much and their routed taps read
smaller.

Two reading notes that the `sync` column forces.

- **The instrument's own price is not constant across the arms.** The same 346,748-call tap costs
  10.41 s of the stacked chunk against 9.07 s of the base one, so the two instrumented walls are
  65.94 and 48.48 while the two *quiet* walls are 56.87 and 38.07. The quiet pair is the honest
  ratio: **‑33.1%**, against the instrumented pair's ‑26.5%, which is a floor.
- **The `routed.issue` row's columns move between the arms and its total does not stay put.** In the
  base chunk the tap's postamble is the grouped GEMM and it reads 17.61 s of `sync` against 0.57 s of
  `body`; in the stacked chunk those are 0.25 and 5.24. The function is byte-identical between the two
  trees — `diff` of the extracted `_issue_chunk` is empty — so what moved is what it waits on: with
  the kernels three times faster, the issue path's own time is the host's launches and the
  `wait_stream` on the copy stream it depends on, not the GEMM. The **sum** of the two columns is the
  number to compare across arms: 18.18 against 5.49.

## Where the 18.80 quiet seconds went

The three passes' own A/Bs, the base each was taken against, and what the composition here reads.
Each delta is the other page's, measured on the real checkpoint at this same geometry:

| change | its own A/B | its delta | the row it lands in, here |
| --- | ---: | ---: | --- |
| #296, the sparse score pass as a warp dot | 56.88 → 51.55 s | ‑5.33 | `attn.sparse` 8.07 → 1.98 (‑6.09 total) |
| #297, the two fp4 kernels' weight reads coalesced | 56.49 → 47.51 s | ‑8.98 | `routed.issue` 18.18 → 5.49 (‑12.69 total) |
| #298, the reduction grouped per token into a CSR | 51.46 → 46.88 s | ‑4.58 | the same row: #297's two kernels go 13.72 → 4.44 s of its own device table, and this finishes the reduction inside the second |
| the three added | | **‑18.89** | measured here: **‑18.80** on the quiet chunk |

The last row is the point of the page. The deltas are taken against three different bases — #296
against the 256K branch, #297 against `master`-based kernels inside the 256K worktree, #298 against
the tree with #296 already in it — so their sum is a prediction that only holds if the three do not
interact, and it holds to **0.1 s in 18.8**, five parts in a thousand. The one interaction worth
looking for is between #297 and #298, which are stacked: #297 widens the token tile and stages the
weights through shared, #298 changes the reduction the second kernel does afterwards, and the two
touch the same launcher and the same two symbols. Their being additive says the reduction loop #298
replaced was not covered by anything #297 did, which its own page claims by construction (the CSR is
a different loop over the same pairs) and this page confirms by measurement.

The third row of the table is where the two differ in kind rather than in size. `attn.sparse` is a
single kernel whose 8.06 s of measured device time becomes 1.97; `routed.issue` is a *path* whose
total time more than triples its own kernels' improvement, because the 12.69 s it loses is not all
kernel time — 9.28 s of it is #297's two kernels and the rest is the reduction and the host's share
of the launch loop, which is only exposed once the card stops being the limit.

## What the chunk is made of now

Attributing the quiet chunk's 38.07 s out of the measured device rows:

| row | base, device | stacked, device | share of the stacked quiet chunk |
| --- | ---: | ---: | ---: |
| expert H2D into the arena (`routed.upload`'s sync, 170.5 GiB at ~12 GB/s) | 14.37 | 15.02 | **39%** |
| the grouped fp4 GEMM (`routed.issue`'s sync, base) | 17.61 | ≤4.4 | ~12% |
| sparse attention | 8.06 | 1.97 | 5% |
| the TP combine at the end of `MoE.forward` (`moe`'s own sync) | 2.86 | 2.95 | 8% |
| the rest of the measured device time (window, indexer, hc, norm, engram, gate, shared) | 8.56 | 8.55 | 22% |
| the routed path's own host bodies (see below) | 5.11 | 5.11 | 13% |

Three things follow.

- **The lever has moved to the bytes.** 15.02 s of H2D is now the largest single row, and it is not a
  kernel: it is 170 GiB of expert rows crossing PCIe 3.0 at the 12 GB/s this host measures, 8,978
  calls of ~19 MiB. #297's own docstring says so about its own change — "the staged rows are the same,
  so what moved is the call count and not the floor" — and with the kernels out of the way the floor
  is what is left. The reduction that would move it is the number of rows a chunk stages, which is
  the same lever the [chunked-prefill page](deepseek_v4_1_flash_chunked_prefill.md) prices at 1.51 ms
  a row and the pool width at 148 rows.
- **The host's row bookkeeping is not the third lever.** The chunked-prefill page read a 6.7–9.5 s
  band off `_resolve_row`'s row of its table and attributed it to the per-row Python loop at 20–29 µs
  a call over 327,680 calls. The barrier split says that row's `body` is **1.52 s over 163,840 calls
  in the base chunk, 1.52 s in the stacked chunk, and 1.52–1.55 s in the base warm-up's eight chunks**
  — four independent 163,840-call groups agreeing to 2%, the cleanest one-instance measurement in
  this run — so its 7.00 s row is 1.52 s of loop, ~2.0 s of the probe's own two barriers, and the
  balance a wait on H2D issued by `_stage_misses`. What the routed path spends on the host is its
  *bodies*, and they sum to 5.11 s: `_upload` 2.34 s over 8,634 calls (271 µs a call), the per-row
  loop 1.52 s, `DeviceRoutedExperts.forward`'s own glue 1.97 s, `_issue_chunk` 0.57 s,
  `_drain_chunk` 0.34 s, `_stage_misses` own 0.13 s. The per-row loop is 30% of it, and the shim the
  chunked-prefill page built prices the floor of that loop at 4.6 µs a call — 0.75 s of the 1.52 —
  so the row loop is worth at most 1.4% of a chunk, against the copies' 39%. (That page's band is
  corrected in place.)
- **The TP combine has become visible.** `MoE.forward`'s own postamble is 2.95 s of the stacked chunk
  over 40 calls — 73.8 ms a layer — and nothing in it is a kernel this pass touched: it is
  `tp.reduce` at the end of the MoE (`modules.py:485`), one all-reduce of a `[4096, 5120]` bf16
  activation, 41.9 MiB, across four cards whose fabric is two PHB pairs and two NV2 pairs joined by
  SYS. At 8% of the chunk it is now larger than the sparse pass, and it is the one row here that no
  amount of kernel work inside a layer can shrink.

## The logits

`/tmp/pr_b_parity.py` dumps the last-position logits after every chunk — nine positions from 4096 to
36864, 129,280 entries each — and it was run three times through `/tmp/parity_stack.sh`: the base
tree, the stacked tree, and the stacked tree again. The third arm is the control, and it comes back
**bit-identical over all nine positions on all four ranks, worst `max |diff| = 0.000e+00`**. There is
no host-side nondeterminism in this run to discount, so base against stacked is the three kernel
changes and nothing else.

| | base against stacked | stacked against stacked again |
| --- | --- | --- |
| positions with the same argmax | 9 of 9, every rank | 9 of 9, every rank |
| bit-identical positions | 0 of 9 | **9 of 9** |
| worst `max \|diff\|` | 4.564e+00, at 8192 | 0.000e+00 |
| `max \|logit\|` there | 25.28 | 25.28 |
| top-2 gap there | 12.162 | 12.162 |

The perturbation is not rounding dust. The mean `|diff|` across the vocabulary is 0.33–0.67 at the
nine positions against a largest logit of 25.3–27.9, so the distribution moves by a fiftieth of its
own scale — which is what reordering fp32 accumulation through forty layers, a router and an expert
pool does, and the same shape the [chunked-prefill
page](deepseek_v4_1_flash_chunked_prefill.md) reported when it localized its own first difference to
layer 0's gate and routed experts. What it does not do is move the decision. The gap between the top
two logits is 9.10 to 13.64 at the nine positions, standing at **2.7× to 6.0×** the worst
perturbation there — 12.162 against 4.564 at 8192, the tightest of the nine — and the argmax is the
same on both trees at every one of them. Nine decisions out of nine is a weak test on nine positions,
and it is the test this run has.

Two footnotes. Every rank writes the same nine rows, byte for byte, in all three arms: the expert
split is reduced back to one activation per layer, and the four cards agree to the bit, which is why
the log repeats the same comparison four times. And the arms' walls agree with the profile to the
second — the base arm's nine chunks are 65.23 s for the first and 55.2–56.9 for the rest, the stacked
arm's 38.5 — so the run that produced these logits is the run this page's tables describe.

## Reproducing

```bash
export DEEPSEEK_V41_RESIDENT_EXPERTS=1
# the base arm, on the 256K branch
V41_TREE=/tmp/pr_b /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
    /tmp/probe_v41_chunk_profile_host.py --at 32768 --chunk 4096 --max-seq-len 41024 \
    --pool-rows 148 --buffers 2 --threads 22 --out /tmp/chunk_profile_base
# the stacked arm, on 98e828f with the three merges
V41_TREE=/tmp/prefill_all /home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
    /tmp/probe_v41_chunk_profile_host.py --at 32768 --chunk 4096 --max-seq-len 41024 \
    --pool-rows 148 --buffers 2 --threads 22 --out /tmp/chunk_profile_all
```

The stacked tree is the three branches merged onto `feature/v41-256k-context`, which is what
`perf/v41-prefill-integration-256k` is:

```bash
git checkout -b perf/v41-prefill-integration-256k feature/v41-256k-context
git merge perf/v41-prefill-sparse-attn-warp-dot
git merge perf/v41-moe-multi-coalesced-weights
git merge perf/v41-moe-reduce-csr
```

`--out X` writes `X.r0` … `X.r3`, one a rank; the parity arms are `/tmp/pr_b_parity.py` with the same
`--at/--chunk/--pool-rows/--max-seq-len`, and `--compare a b` reads two of its dumps back.
