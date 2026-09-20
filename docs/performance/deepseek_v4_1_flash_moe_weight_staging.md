# DeepSeek-V4.1-Flash: the batched prefill MoE call's weight reads, coalesced

The [remaining-bottlenecks page](deepseek_v4_1_flash_remaining_bottlenecks.md) ended by shipping the
batched expert call (`--expert-batched`, `moe_multi_token_fp4_forward`) and taking a 512-token
prefill **27.0%** down. The [device-experts page](deepseek_v4_1_flash_device_experts.md) and the
[graphed-decode pages](deepseek_v4_1_flash_decode_graph_live.md) then took the decode side as far as
the tree inside a CUDA graph. This page is the pass that follows the batched call into its kernels:
a 4096-token chunk at 32768 spends **9.331 s of its 65.36 s in `moe_multi_w1w3_fp4_kernel` and 4.390 s
in `moe_multi_w2_partial_fp4_kernel`** — 21% of the chunk in two kernels that read weights — and both
of them read the checkpoint's per-expert layout with the *columns* strided, so every warp's 32 lanes
touch 32 separate 32-byte sectors for 16 useful bytes each.

The change is two things at once, and the sweep below prices them apart. The token tile goes from 8
rows to 32 — a slot averages 34.4 rows, so the old kernel walked the weights **4.3 times a call** and
the new one **1.08** — and the weight packs are staged through shared so that a warp's 32 scattered
sectors become 512 contiguous bytes. Of the **3.89×** the two together measure off the model,
**1.66× is the tile and 2.34× is the coalescing**; in the model they are worth **3.06× on
`moe_multi_w1w3_fp4_kernel`** (59.81 → 19.54 µs a call) and **3.14× on
`moe_multi_w2_partial_fp4_kernel`** (28.14 → 8.95 µs) on the real checkpoint, on the same tree, in
consecutive runs. The chunk's wall follows it down almost exactly — 9.276 s of kernel removed, 9.00 s
of profiled wall and 8.98 s of unprofiled chunk mean gone with it — because these two kernels are on
the critical path rather than overlapped: **56.49 → 47.51 s a 4096-token chunk at 32768, 72.5 → 86.2
tok/s**, with the seven other device-table rows unmoved to within 3%.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | `perf/v41-moe-multi-coalesced-weights` on top of `master` at `645e36b`; both arms run from the `feature/v41-256k-context` worktree at `/tmp/pr_b` (tip `a5327fb`), which is the tree that carries the chunked-prefill path a 4096-wide forward at position 32768 needs |
| The two arms | master's `cuda_kernel_impl.cu` built to `cuda_kernel.cpython-311-x86_64-linux-gnu.so` (9416528 bytes, md5 `37aba9d7c6e932d026b1dcab2f00dd92`) against the same file with this change (9490256 bytes, md5 `0830e3d7bb657c61fd3e683b7104a42a`), copied into `/tmp/pr_b/build/extensions/` in turn — one binary differs, nothing else |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `--pool-rows 148 --buffers 2`, resident bank on |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, cross-pairs SYS |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env, `CUDA_HOME=/usr/local/cuda-12.4`, `TORCH_CUDA_ARCH_LIST=7.5`, `POCKETLLM_BUILD_CPP=0` |
| Probe | `/tmp/probe_v41_kernel_split.py` (this pass's, unchanged: prefill to `--at` the way the sweep does, then profile **one** `--chunk` with the torch profiler), run once an arm into `/tmp/kernel_split_before.log` and `/tmp/kernel_split_after_256k.log` |
| Microbench | `/tmp/run_moe_multi_bench.py` driving `/tmp/moe_multi_bench.cu`, off the model, at the measured per-call geometry |

The extension is orthogonal to the 256K work: `git diff --stat master feature/v41-256k-context --
src/csrc/` is empty and both trees' `cuda_kernel_impl.cu` hash to `84a36a1a4fa1ecba99fd2f8fb7ec31e3`,
so the same `.so` serves either. The measurement could not be taken against `master` itself: master
has no chunked-prefill path, and a 4096-wide forward at position 0 takes `_window_kv`'s decode branch
(`Pos.first()` is `self._host == 0`) and dies in `decode_pos.py:76` with `RuntimeError: The expanded
size of the tensor (1) must match the existing size (4096) at non-singleton dimension 0. Target
sizes: [1, 512]. Tensor sizes: [4096, 512]`. The arms were therefore swapped inside the 256K
worktree, which is why the commit and the arm builds are named separately above.

## The layout, and what a warp's load costs

`moe_multi_w1w3_fp4_kernel` is one block to a `(slot, column tile)` pair, 128 columns of `dim` a
block over 18 blocks and one slot each — a grid of `(18, 61)`. It loops over the tokens the slot was
routed, and for each K block multiplies that token's int8 activations by the expert's fp4 weights.
The weights come from the checkpoint as `[expert][inter_dim][blocks_k][16]` bytes: **one k-block is
16 contiguous bytes, and consecutive columns are `dim / 2 = 2560` bytes apart.** The kernel's inner
loop is over columns (the lanes), so the lanes of a warp address 32 consecutive columns at the same
k-block — 32 addresses 2560 bytes apart. Each 32-byte sector delivers 16 useful bytes: the other 16
belong to two other columns no lane in this warp wants. At 61 slots and 2100 (token, expert) pairs a
call that is **729 MiB of w1/w3 weight bytes read through a sector stream that is 50% useful**, and
it is what the 59.81 ms a call measures.

Two things follow. First, the reads want to be column-major. Second, they cannot be made so by
addressing alone: the bytes are where they are, and the checkpoint's layout is the checkpoint's. So
the strip has to be restaged, in shared, into the shape a warp can read — and doing that restaging
per K strip rather than per k-block is what keeps it affordable.

## The standalone price

`/tmp/moe_multi_bench.cu` is the same kernel with the staging parameterised: `TILE` tokens in
registers a thread, `STRIP` K blocks staged a pass. It is driven at the measured call's geometry —
**1050 rows, 61 slots (2100 pairs, 2 routes a row, 34.4 pairs a slot, 4.3 passes of 8), dim 5120,
inter_dim 2304, `blocks_k` 160, grid (18, 61), 1094 MiB of weight bytes a call, 729 MiB of it
w1/w3** — with a
bit-identity check against the shipping kernel before anything is timed. Two runs of that sweep are
in hand and they agree to within 1% on every row; the table is the second, each variant checked
`exact` against `bench_v0`:

| variant | ms a call | speed | blocks/SM | shared |
| --- | --- | --- | --- | --- |
| `moe_multi_w1w3_fp4_kernel` (shipping, `tile=8`) | 50.787 | 1.00× | 1 | 40 KiB |
| strip through shared, `tile=8 strip=16` | 111.195 | 0.46× | 7 | 72 KiB |
| strip through shared, `tile=16 strip=32` | 51.715 | 0.98× | 4 | 152 KiB |
| strip through shared, `tile=32 strip=16` | 30.526 | 1.66× | 3 | 84 KiB |
| **smem stage, `tile=32 strip=4`** | **13.065** | **3.89×** | **3** | **21 KiB** |
| smem stage, `tile=16 strip=4` | 14.737 | 3.45× | 3 | 19 KiB |
| smem stage, `tile=32 strip=2` | 17.794 | 2.85× | 3 | 10 KiB |
| smem stage, `tile=32 strip=6` | 15.654 | 3.24× | 2 | 32 KiB |
| smem stage, `tile=32 strip=8` | 26.103 | 1.95× | 1 | 42 KiB |
| smem stage, `tile=48 strip=4` | 19.858 | 2.56× | 2 | 23 KiB |
| smem stage, `tile=64 strip=4` | 24.190 | 2.10× | 2 | 25 KiB |
| u2-load, v0-layout, `tile=32 strip=16` | 28.344 | 1.79× | 3 | 84 KiB |
| u4-load, v0-layout, `tile=32 strip=16` | 31.782 | 1.60× | 3 | 84 KiB |
| transposed weights, `tile=8 strip=16` | **8.639** | **5.88×** | 8 | 72 KiB |
| transposed weights, `tile=16 strip=32` | 9.348 | 5.43× | 4 | 152 KiB |
| transposed weights, `tile=64 strip=8` | 17.391 | 2.92× | 2 | 50 KiB |
| staging only (loads, no math), `tile=8 strip=16` | 0.392 | — | 8 | 72 KiB |
| math only (constant weight, no loads), `tile=8 strip=16` | 24.282 | — | 8 | 72 KiB |
| math only, `tile=32 strip=16` | 9.697 | — | 3 | 84 KiB |

Considered as a set, the table rules three routes out and picks a fourth:

- **The 3.89× is two changes, and the table separates them.** `strip v0-layout tile=32 strip=16`
  (30.526 ms, 1.66×) is the shipping addressing — 2-byte reads of the checkpoint's own layout —
  with only the token tile widened from 8 to 32: a slot averages 34.4 rows, so the weights are
  walked **4.3 times a call at `tile=8` and 1.08 at `tile=32`**, and the weight traffic falls with
  them. The strip's coalescing is the remaining **2.34×**, 30.526 → 13.065. Both effects are inside
  the in-model 3.06×, because the model's call has the same 34.4 rows a slot — which is worth
  saying plainly, since a reader who only widened the tile would already take the cheaper 1.66× of
  it. (The 84 KiB that row reports is the bench's own allocation for a staging this mode does not
  use; the tile alone needs its 16 KiB of activations.)
- **Load width is not the problem.** Re-reading the same scattered addresses as `uint2` or `uint4`
  instead of 16 bytes moves 30.526 ms to 28.344 and 31.782. Each warp still asks for one sector a
  lane; a wider request does not put a neighbour's bytes into the same sector.
- **Staging without the strip is worse than not staging.** At `strip=16` the staging *is* the load —
  the same scattered sectors, now through shared — and `tile=8 strip=16` measures 111.195 ms, 0.46×,
  the shipping kernel's own cost with a barrier added. The win appears only when the strip is small
  enough that the staging pass's shared traffic is cheap against the compute that reuses it, and
  `strip=4` (13.065 ms) against `strip=6` (15.654) and `strip=8` (26.103) is that trade: **barriers
  against occupancy.** Widening the strip quarters the barrier count but the weights staged are
  `2 · strip · 128 · 17` bytes, so occupancy falls faster than the barriers fall — 3 blocks/SM at 21
  KiB, 2 at 32 KiB, 1 at 42 KiB.
- **`tile=64` is out of registers, not out of shared.** It stages only 25 KiB and still loses 2.10×,
  because the accumulators are `4 · tile = 256` registers a thread for the gate/up pairs, past what a
  thread has: the compiler splits them across two passes and the reuse the tile was bought for goes
  with it. `tile=48` costs the same way (2.56× at 2 blocks), and the measurements bracket the
  shipped 32 from both sides.
- **The loads are nearly free when they are coalesced.** Staging alone — the same loops with the
  arithmetic removed — is 0.392 ms, 0.8% of the shipping kernel. This is the honest statement of
  where the 50.787 ms goes, and it is *not* a floor to subtract: the 24.282 ms "math only" row is the
  same arithmetic with the weight operand a compile-time constant, and with the loads gone the
  compiler can hoist every `__dp4a` into a dependency chain that no load stalls; the transposed
  variant does the same arithmetic *and* its own coalesced loads in 8.639 ms, below it. The two
  diagnostic rows say "the scattered loads are what the kernel is spending on", not "here is the
  budget left over".
- **The transposed checkpoint layout is the faster route, and it is not this change.** 8.639 ms
  (5.88×) at **8 blocks/SM**: if the arena held each expert's weights as `[blocks_k][inter_dim]`, a
  warp would read its own 512 contiguous bytes with no staging, no barriers and no shared to hold.
  It needs a one-time rewrite of the weight bytes in the bank, which is a separate decision from this
  kernel's addressing — recorded here as the lever the staged route is 1.5× short of.

The reduction, priced on the same 1050-row call: `moe_multi_reduce_partials_kernel` rescans all 2100
pairs for each of the 1050 tokens to find its own, **11962.6 µs**, where a per-token CSR of the same
additions in the same ascending pair order — bit-identical (`exact` in the sweep) — is **172.8 µs**,
69.24×. That is the next lever and it is a follow-up rather than this PR: it changes what the call is
passed, not what the kernel reads. It has since shipped, as
[the reduction page](deepseek_v4_1_flash_moe_reduce_csr.md); the grouping the microbench built in
Python is built there **on the device, inside the launcher**, because one of the two call sites
resolves its routing on the GPU and reading it back would be the D2H sync that path exists to avoid.

The scan's 11962.6 µs is 0.50× the 23930 µs a call the in-model profile reports, i.e. about 1.9 s of
the 3.733 s a chunk; the discrepancy is the microbench's idle card against the chunk's concurrency,
the same direction as the 3.89× against 3.06× above.

## The change

Three kernels get the same treatment — `moe_multi_w1w3_fp4_kernel`,
`moe_multi_w2_partial_fp4_kernel` and its non-deterministic twin `moe_multi_w2_accum_fp4_kernel`
(`DEEPSEEK_MOE_DETERMINISTIC_REDUCE` unset or 0; the partial path is the default). Two things
changed, and the sweep prices them apart: the token tile went from `kMaxTokens = 8` to
`kMultiTile = 32`, and the weight packs went from per-column global reads to a K strip staged
through shared. A call's shared buffer is now

```
tile · strip · 8 · int        the token tile's activations, restaged per K strip
matrices · strip · 128 · 16   one strip of each weight matrix's fp4 packs
matrices · strip · 128        and its scale bytes
```

which is 21,504 bytes for `w1w3` (`matrices = 2`) and 12,800 for `w2` — at `kGemmThreads = 128`
threads that is **3 blocks a multiprocessor**, 64,512 of the 65,536 bytes a 2080 Ti's SM offers.

The two structural points that make it correct rather than merely fast:

1. **The strip only reorders the reads.** Each `(token, k-block)` product is still accumulated into
   the same register, in ascending k-block order, by the same `__dp4a` on the same unpacked fp4
   codes, with the same block scale multiplied once per k-block; the token tile's outer loop is
   untouched. Every rounding the kernel performed before it performs now, in the same order. The
   microbench's `exact` column is that claim tested: all eight computing variants agree with the
   shipping kernel bit for bit on the same inputs.
2. **The weight rows a block may touch are bounded by `rows_here`.** The staging loop loads
   `strip · min(cols, inter_dim - col0)` entries, not `strip · cols`, so a block whose column tile
   runs past `inter_dim` does not read a neighbour's rows; the compute for columns past `inter_dim`
   is skipped where it always was. `w2` walks `dim` the same way.

## The in-model A/B

Same checkpoint, same tree, same probe, same `--pool-rows 148`, consecutive runs, one binary apart.
Rank 0, a 4096-token chunk at 32768, profiler on:

| device table row (rank 0) | before | after | |
| --- | --- | --- | --- |
| `moe_multi_w1w3_fp4_kernel` | 9.331 s / 59813.9 µs a call | 3.048 s / 19541.0 µs | **3.06×** |
| `moe_multi_w2_partial_fp4_kernel` | 4.390 s / 28140.7 µs | 1.397 s / 8952.9 µs | **3.14×** |
| `moe_multi_reduce_partials_kernel` | 3.733 s / 23930.0 µs | 3.626 s / 23246.4 µs | 0.97× |
| `moe_multi_swiglu_quant_fp4_kernel` | 0.016 s | 0.016 s | — |
| `prefill_sparse_attn_he...` | 8.147 s / 203663.2 µs | 8.079 s / 201967.0 µs | 0.99× |
| `Memcpy HtoD (Pinned -> Device)`, 57054 calls | 15.636 s | 15.610 s | 1.00× |
| `aten::copy_`, 66709 calls | 16.854 s | 16.742 s | 1.01× |
| `cudaEventSynchronize` (host), 8790 calls | 29.159 s | 19.813 s | 0.68× |

The rows that were not touched are the control, and they move by at most 3% — including the two
biggest host and device terms in the step. That is what makes the two rows that did move attributable:
the difference cannot be a page-cache or clock difference between sittings, because `Memcpy HtoD`
and the sparse-attention kernel read the same checkpoint through the same paths in both runs and hold
still.

The wall follows the two kernels almost exactly, which is the more interesting number:

| | before | after |
| --- | --- | --- |
| chunk wall under the profiler | 65.36 s | 56.36 s (−13.8%) |
| 4096-token chunk mean, unprofiled, eight chunks to 32768 | 56.49 s | **47.51 s (−15.9%)** |
| fastest of those eight | 55.10 s | 45.61 s |
| chunk throughput | 72.5 tok/s | **86.2 tok/s** |

9.276 s of kernel time was removed and 9.00 s of profiled wall went with it (8.98 s unprofiled). No
other row grew to absorb the saving, which says these two kernels are serialized against everything
else in the chunk rather than overlapped with it — and therefore that removing the weight-read cost
the next section prices is worth wall time one for one, with no ceiling hiding behind a balance of
streams.

Both figures are for a 4096-token chunk arrived at **32768**, which is the configuration the split
probe is written for, and they are not a 262144-token number: the chunked-prefill work in review
(PR #293, which this branch does not carry and therefore cannot link) measures 57.98 s a chunk at
262144, where the attention and the KV traffic are a different fraction of it. That has to be
re-taken there rather than scaled from here.

One number does *not* transfer: the standalone sweep prices the same change at 3.89× and the model
pays 3.06× on `w1w3`. The microbench runs the kernel on an otherwise idle card where the coalesced
loads have the L2 to themselves; in the chunk it shares the L2 with the sparse-attention kernel's
203 ms-a-call working set and 57054 `Memcpy HtoD` calls. The in-model figure is the one to quote.

## Correctness

- **Bit-identity inside the microbench**: all eight computing variants of
  `bench_strip` — every `tile`/`strip` combination that does the arithmetic, in both the checkpoint
  and the transposed layout — report `exact` against `bench_v0`, over a call with 2100 pairs across
  61 slots, i.e. the real call's routing shape.
- **Bit-identity in the model**: the same two arms, every chunk of the run to 32768 plus one,
  `max |diff|` on the full `[1, 129280]` last-position logits row, with the argmax compared as
  well. `probe_v41_kernel_split.py` does not keep logits, so the parity arm is
  `/tmp/pr_b_parity.py` — the same harness with the profiler block replaced by a per-chunk logits
  dump — and the two arms' files are compared position by position. **`max |diff| = 0.000e+00` at
  each of the nine positions on all four ranks, and the same argmax with it**: rank 0 reads
  `223, 5198, 223, 223, 13822, 17227, 13918, 18014, 455` in both arms, and ranks 1–3 read the same
  nine. The parity harness carries no profiler, so it is also a second, cleaner A/B of the wall:
  `prefilled to 36864, chunks at 56.04 s mean from the second on` against **`46.76 s mean`**, all
  four ranks printing the same figure to the hundredth.
- **The repository's own kernels tests** pass on the changed build:
  `tests/test_moe_multi_token_fp4.py` and `tests/test_moe_single_token_fp4.py`, 13 passed, which
  includes `test_agreement_at_production_shapes` and the register-tile and non-local-expert cases
  against the single-token reference.

## What this leaves

The chunk at 32768 is now dominated by terms this change does not touch: `Memcpy HtoD` 15.610 s and
`aten::copy_` 16.742 s of a 56.36 s wall (57% between them), the sparse-attention kernel 8.079 s, and
the reduction 3.626 s. Of those, the reduction's CSR has since shipped as
[its own page](deepseek_v4_1_flash_moe_reduce_csr.md) — built on the device rather than host-side as
the microbench had it, which is the correction the second call site forced. The two copy terms are the
same [H2D and `_stage` accounting](deepseek_v4_1_flash_device_experts.md) the earlier pages priced — a
different lever on a different path, not a kernel to rewrite. Inside the weight reads that remain, the
transposed arena is the 1.5× the shared-staging route is short of, at the cost of a one-time rewrite
of the bank — and it is the last weight-read lever this page can name.
