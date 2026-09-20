# DeepSeek-V4.1-Flash on the four cards: what is left to optimize

The [routed-experts page](deepseek_v4_1_flash_device_experts.md) ends with a configuration: the dense
tree cut across four 2080 Ti, one process a card, **722.0 ms a decode step** and **3.0 tok/s** a
prefill. This page is the pass that follows it. It takes that step apart by measurement and asks what
each remaining lever is actually worth, including the ones that turn out to be worth nothing — and it
carries one change that the measurement did earn, the batched expert call in
[Lever 5](#lever-5--the-prefill-is-a-batch-shape-problem-and-the-fix-is-in-27-at-512-tokens), which
took a 512-token prefill down 27.0% with the logits unmoved. Every number here is either re-taken for
this pass or is quoted from the sitting that produced it, with the file it came from named.

The frame the whole page rests on is the one the TP4 sitting established: **this step is
Python-dispatch-bound, not compute-bound.** Device-busy is 11% (≈1.5 ms of a 13.7 ms block probe),
the step launches **7,076 kernels**, 122 distinct ones, of which the sixteen heaviest are 47% and the
remaining ~3,750 are a tail over 106 names. A rank running a quarter of the weights at TP4 runs a
quarter of the launches **at the same price each**, which is why the tree's factor is 1.9× and not 4×.
Anything framed as a bandwidth win on this path has to explain how it removes launches or removes
work from the critical path, because bytes are not what the step is spending.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | `c628927` on `master` (PR #274 merged); the extension is the repo-root `cuda_kernel.cpython-311-x86_64-linux-gnu.so`, built 2026-09-17 00:46, carrying the sparse-attention fence from `85f7585` |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, resident bank on, `--hot-rows 0 --pool-rows 0` unless a row says otherwise |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, cross-pairs SYS |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Probes | This pass's are `/tmp/probe_v41_decode_locality.py` (new: `--locality`'s LRU/LFU replay and `--stage-probe`) and `/tmp/bench_hc_mixes.py`, re-run to `/tmp/hc_mixes.out`. Everything else is named at the point it is used |

The extension's build time predates the fence commit's own timestamp by 1 h 13 m, which invites the
suspicion that these numbers were taken on a binary without the barrier — the barrier is what makes
the sparse attention reproducible. It was checked rather than assumed: `85f7585` adds exactly one
`__syncthreads()` to each of six sparse-attention kernels (7/7/7/10/7 inline barriers at `HEAD`
against 6/6/6/9/6 at its parent), and `cuobjdump -xelf` + `nvdisasm` on the shipped `.so` counts
**7/7/7/7/10/7 `BAR.SYNC`** in those same six — the post-fence source, built from the working tree
before the commit was made.

Three sittings are new here, and they are what the levers below are priced off:
`/tmp/locality2.out` (24 decode steps, the draw and union curve), `/tmp/locality3.out` (the same run
plus the resident-set replay), and `/tmp/stage.out` (`--stage-probe 2`, the in-situ `_stage` clock and
its thread sweep). All three are four-rank `torchrun` runs of the probe in its checked-in form.

## The step this page is against

Uninstrumented, 22 threads, warm, `/tmp/probe_v41_tp4_e2e.py`:

| Context | Prefill | Prefill tok/s | Decode | in `DeviceRoutedExperts` | in the tree | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 tokens | 3.05 s | 2.6 | **746.8 ms** | 444.3 ms | 302.5 ms | 1.34 |
| 128 tokens | 42.86 s | 3.0 | **722.0 ms** | 408.8 ms | 313.2 ms | 1.39 |

and the same step with `_stage`, `_upload` and `_launch` wrapped (`--lengths 8,128`), which is the
column every lever below is priced against:

| | 8 tokens | 128 tokens |
| --- | ---: | ---: |
| `_stage` | 224.7 ms | 256.0 ms |
| `_upload` | 16.0 ms | 17.8 ms |
| `_launch` | 192.9 ms | 177.6 ms |
| unattributed inside the class | 59.3 ms | 39.7 ms |
| whole step | 804.5 ms | 815.9 ms |

For a decode step at 128 tokens the 128-token column is the one to read: **`_stage` 256.0 of 815.9,
`_launch` 177.6, `_upload` 17.8, and 39.7 the class cannot place.** Instrumenting costs 60–90 ms, so
the split is read for its ratios and the totals come from the table above; scaled to the 722.0 ms
step the four terms are 226.5, 157.2, 15.8 and 35.1 ms.

The per-op instrument (`/tmp/probe_dense_tree.py --tree cuda`, `/tmp/dt_cuda.out`) puts the same step
at 688.9 ms with the class at 414.45 and the named dense calls at 191.32, so the two instruments
agree on the class to 1.3% and on the step to the run's own first-frame effect.

## What is already spent

Five levers on this path are closed, and the first is closed by a change that landed inside this
work rather than before it. They are listed here because the plan they came from is now a *record*
and its phases should not be re-opened as if they were still open.

| Lever | Measured | Where |
| --- | --- | --- |
| `hc_split_sinkhorn`'s 19-iteration Python loop, fused into one kernel | the loop's **154.45 ms/step becomes 7.19**; on an idle card, 80.0 µs against the loop's 1719.1 µs, 21.5× | [the dense tree's largest launch source](deepseek_v4_1_flash_device_experts.md#the-trees-largest-launch-source-was-one-python-loop-and-cutting-the-tree-could-not-cut-it), and `/tmp/hc_mixes.out` below |
| The disk out of `_stage`: the resident bank | an emptied page cache is **782.9 ms a step against 17.01 s** (21.7×); warm it is a wash | [the bank](deepseek_v4_1_flash_device_experts.md#the-resident-bank-takes-the-disk-out-of-_stage-and-it-is-the-copy-into-pinned-that-was-left), and [the copy it left, since removed](deepseek_v4_1_flash_device_experts.md#the-pin-and-the-copy-removal-landed-200-ms-a-decode-token-and-160-s-a-prefill) |
| Per-layer residence for a prefill: `--expert-hot-rows` and `--expert-pool-rows` | **2.6–2.7× and 4.0–4.2×** a 512-token prefill at 148 rows | [the set](deepseek_v4_1_flash_device_experts.md#a-per-layer-resident-set-is-worth-27-on-a-prefill-and-it-is-the-fill-that-pays-for-it), [the pool](deepseek_v4_1_flash_device_experts.md#the-pool-spends-the-same-arena-on-what-the-pass-draws-and-its-first-key-answered-the-wrong-layer) |
| The row loop, one row deep | **1.098× at 32 tokens, 1.108× at 128** — a prefill lever only, because a decode step is one row a layer and a row-deep pipeline has nothing to overlap there | [the row loop](deepseek_v4_1_flash_device_experts.md#the-row-loop-runs-one-row-deep-and-it-is-worth-110-on-a-prefill) |
| `--threads`, the launcher's flag | **1.13× with the source resident, 6.8× without** — and the 6.8× is the disk's, not the flag's | [the threads table](deepseek_v4_1_flash_device_experts.md#--threads-is-worth-113-with-the-source-resident-and-68-without-it) |

One of these deserves a correction here rather than in the page it came from, because this pass
measured the thing that looked uncertain: **`hc_split_sinkhorn`'s fused path is already the default,
not an opt-in.** `_HC_SPLIT_IMPL` defaults to `"auto"` and `_auto_impl("hc_split")` returns
`"triton"` whenever triton imports (`src/kernels/ops.py:84`), so
`hc_split_sinkhorn_torch` — the 1719.1 µs body — runs only on a CPU tensor, a non-power-of-two
`hc_mult`, or an environment without triton. The fused kernel's own dispatch is 80.0 µs a call, which
is the 7.19 ms a step the table above records. There is nothing left to fuse here.

## Lever 1 — `_stage` is not slow, it is on the critical path, and the thread count is spent

**⚠ Closed, and the third reading below is why.** Everything this section measures about `_stage` is
reproduced: it is 145.8 / 185.2 / 192.1 / 220.2 / 241.2 ms on rank 0 across five sittings against the
198.5 ms recorded here, and it is on the critical path. What does not survive is the inference drawn
from the in-situ/replay gap — that the gap is *scheduling*, and therefore recoverable by issuing layer
`k+1`'s stage under layer `k`'s tree. `/tmp/probe_stage_sched.py`, five sittings at 128 tokens:
the gap is a **one-time pool warm-up** (rank 0's back-to-back replay runs 467.5 → 295.8 → 207.3 →
153.3 ms over four passes of the *same* copies, and this section's 91.8 ms is the *seventh* pass of a
thread sweep that ends at 22), and re-inserting the recorded 13.3–15.6 ms gap between the calls costs
**679–940 ms a step across the four ranks against the step's own 116–225 ms** — the stage-after-tree
shape this section proposes is 3–8× *more* expensive, not 15% cheaper. The other two candidate
mechanisms are excluded with it: CPU/wall is **8.84–13.91×**, so the calling thread is not the copy's
bottleneck and a background thread buys nothing, and spending the gap sleeping with the pool released
is no better than spinning (698–736 against 679–940 ms), so pool sleep/wake is not it either. The
sweep, the per-call wall times and the gapped replays are in `/tmp/sched.log`–`/tmp/sched5.log`. **The
two readings that do survive are 1 and 2** — the 14 GiB/s is not a machine floor, and a per-rank
thread count is worth up to 9.8% of the term; the ranked table at the top of this page carries both.

The class's own docstring for `_stage` says "It is not a faster `copy_`: 14 GiB/s either way."
This pass put a clock on the real call instead of arguing from that sentence, and the answer is that
the sentence is right about the rate at 22 threads and wrong about it being a floor.

`--stage-probe 2` wraps `DeviceRoutedExperts._stage`, times the real call on step 2 of a decode run,
snapshots its arguments, and then replays **the same copies** back to back under other thread counts
with nothing else in flight. `/tmp/stage.out`, four ranks, 128 tokens of context:

| rank | in situ, 22 threads | the same copies replayed, 22 threads | replayed, 1 thread |
| ---: | ---: | ---: | ---: |
| 0 | 480 `copy_`, 1.401 GiB, **198.5 ms, 7.58 GB/s** | 91.8 ms, 16.39 GB/s | 516.3 ms, 2.91 GB/s |
| 1 | 480 `copy_`, 1.401 GiB, **165.4 ms, 9.09 GB/s** | 124.6 ms, 12.07 GB/s | 381.8 ms, 3.94 GB/s |
| 2 | 240 `copy_`, 0.700 GiB, **148.3 ms, 5.07 GB/s** | 72.8 ms, 10.32 GB/s | 188.2 ms, 4.00 GB/s |
| 3 | 240 `copy_`, 0.700 GiB, **103.3 ms, 7.28 GB/s** | 76.9 ms, 9.78 GB/s | 202.4 ms, 3.72 GB/s |

The 198.5 ms is within 15% of the 224.7–256.0 ms the phase clock records for the same term across a
whole run — the probe times one step where the clock averages a run — which is what says the two
instruments are measuring the same work and not one of them the wrong call.

Note what the 480 against 240 `copy_` is: **only ranks 0 and 1 draw two rows a layer**, so those two
stage twice the bytes and pay roughly twice the clock. That is the `sorted` deal's 2/2/1/1 split
showing up in the staging term — this sitting is `sorted`, which was the default then and is one flag
off it now, and the `id` deal that is the default removes the split — and it is why the step's gate is
rank 1 rather than a round-robin rank.

**Three readings, and only the third is a lever.**

1. **The 14 GiB/s is not a machine floor.** That clause comes from the class's own staging table
   ("host staging, page cache → pinned, 4.20 GiB | 0.30 s, 14 GiB/s"), which is an **aggregate**: the
   four copy chains' 4.20 GiB over a step's wall. This pass's per-rank in-situ rates are **7.58 / 9.09
   / 5.07 / 7.28 GB/s** — i.e. 4.20 GiB over the slowest rank's 198.5 ms is 22.7 GB/s aggregate, so
   the sentence and this measurement are the same order and not the same quantity. What settles it is
   the floor: replayed at **one thread** the identical copies run at **2.91 GB/s** on rank 0, a 5.6×
   spread against its own 16.39 at 22 threads. At the low end the loop is per-call overhead and not
   DRAM, so "14 GiB/s either way" is true at the thread count the class uses and false as a property
   of the operation.
2. **The thread count is spent.** The replay's own optimum is 22 threads on ranks 0 and 3, 12 on rank
   2 and 8 on rank 1 — 91.8 / 115.8 / 65.7 / 76.9 ms against the uniform 22 threads' 91.8 / 124.6 /
   72.8 / 76.9 — so a per-rank constant is worth **up to 9.8%** of the term, and exactly 0% on two of
   the four ranks. The step is gated by the slowest rank, which at 22 threads is rank 1, so the
   tunable part is rank 1's 124.6 → 115.8: **7.1% of a 256 ms term, ~16 ms a step**, and it would
   need the launcher to know its rank's contention. The full sweep is in `/tmp/stage.out`. Its shape
   is worth reading before anything else is tried here: 1 thread is 202–516 ms, 4 threads 79–190, 8
   threads 69–124, and the four ranks disagree about where the optimum is, so a thread count is not
   one number on this host — but nothing above 8 threads is ever more than 1.6× off a rank's own best.
3. ~~**The in-situ/replay gap is the lever, and it is a scheduling gap.**~~ **Falsified — see the
   banner at the top of this section.** The gap and the per-layer burst below are measured correctly,
   and the bytes and calls are the same; what does not survive is that the gap is recoverable. The
   same bytes at the same
   thread count cost **2.2× on rank 0, 1.3× on rank 1, 2.0× on rank 2 and 1.3× on rank 3** in the
   real step. One decode layer draws **2 rows = 35.9 MiB** on the two-draw ranks, so a layer's
   `_stage` is **12 `copy_` calls of 35.9 MiB total — six of 5.625 MiB (the packed `w1`/`w2`/`w3`)
   and six of 352 KiB (their scale rows)** — and 40 of those calls in sequence are what the 198.5 ms
   is. Back to back the identical 40 calls take **91.8 ms**, i.e. **2.30 ms a layer's copies against
   the 4.96 the step pays for them**. The per-layer burst is far too short to reach steady state on
   its own and there is a fork-join at each one, which is what the 2.66 ms a layer of gap is.
   **The sentence that followed — that layer `k+1`'s stage should be issued while layer `k`'s tree
   computes — is falsified; see the banner at the top of this section.** The burst and the fork-join
   are real; what the replay column is not is a floor the step fails to reach, because 91.8 ms is the
   seventh of seven consecutive replays of the same copies and a one-time pool warm-up decays across
   them.

   **The ceiling that followed, kept for the record and not as a price.** The part of `_stage` that is
   not the copy is **106.7 / 40.8 / 75.5 / 26.4 ms a step** by rank, and the step is gated by the
   slowest rank, so the most a perfect overlap could return is **~107 ms of a 722 ms step, 15%**. On
   V4-Flash the same idea was tried as cross-layer prefetch and **the configuration with it off was
   the fastest one** — that caution was the finding and it should have been read as the answer rather
   than as a reason to A/B. Measured here it is not 15% recovered but 3–8× spent: the copies do not
   overlap with the tree, they contend with it for the same DRAM and the same intra-op pool, and
   issuing them from under the tree's own dispatch is the worst of the three orders rather than the
   best. The other two candidate mechanisms are excluded by the same probe — CPU/wall is
   **8.84–13.91×**, so the main thread is not the copy's bottleneck and a background thread buys
   nothing, and spending the gap *sleeping with the pool released* is no better than spinning on it
   (698–736 against 679–940 ms), so pool sleep/wake is not the mechanism either.

**A caution that belongs with the lever.** Do not fold the `_stage` copies and the `_upload` copies
into one another. Merging the **host-side** `shm → pinned` copies into fewer, larger `copy_` calls is
safe and is not what has been tried; merging the **pinned → device** H2D calls into one is a measured
regression on this hardware — V4-Flash's decode went from 3.4 to 1.5 with a single large H2D, because
the transfer then had no overlap left to hide behind.

## Lever 2 — a decode-resident set is worth 1.40× on a decode, and this section priced it at 1.09–1.15×

**This section's original price of this lever was wrong by a factor of three, and the correction comes
first.** 61–97 ms was applied to a *draw-count* percentage from the offline replay further down, and
the direct measurement — a
256-step decode at 128 tokens, A-B-A-B with the control at both ends, `--expert-pool-rows 600` against
`--expert-pool-rows 0` — reads **0.780 / 0.750 s a token against 0.528 / 0.566**: **765.0 against
547.0 ms on the pair means, 1.399×, 218 ms off the step**, and the top-32 logits are identical at
`|dlogit| 0.000e+00` on all four ranks of all four legs, so this is a cache and not an approximation.
The full sitting — the A-B-A-B table, the phase split, the across-length sweep, the arena identity,
the parity column and the reproduce lines — is
[the pool's decode subsection](deepseek_v4_1_flash_device_experts.md#the-pool-is-a-decode-lever-and-a-one-token-decode-is-the-measurement-that-hid-it)
on the device-experts page.

**What the offline replay could not see.** It replays the draws faithfully — the same 240 draws a
step, the recorded `indices`, the same set width, a real LRU and a real LFU — and it is a model of the
*draws* rather than of the **path**: it counts hits and never prices a step. And the price is not
simply the `_stage` term a hit removes. **In the decode window a hit is worth 5.0–6.4 ms of the step
a row** across eight sittings, against the 2.7–4.2 ms the two legs cost at their own measured rates,
because a row is 12 `copy_` through `at::parallel_for` and a step that stops staging 36 of them stops
paying 432 fork-joins. The decomposition is a decode-window one and should be read as such: all eight
sittings divide a decode phase's saving by the rows that phase stopped staging, and the same
instrument has not been run on the prefill phase's rows, so **5.0–6.4 is a property of the decode
window and not a price for a hit in both phases.** Two of its columns are also low against the real arena on the column that compares:

| resident set | bytes | LRU, all steps (replay) | **measured, decode-phase** | replay vs measured |
| ---: | ---: | ---: | ---: | ---: |
| 300 rows | 5.25 GiB, 5415 MiB of arena | 25.8% | **29.9%** | −4.1 |
| 600 rows | 10.51 GiB, 10794 MiB of arena | 39.6% | **44.6%** | −5.0 |

Four to five points, the same direction at both widths, and the same order as the two pooled legs'
own spread against each other (45.5% against 44.9%). The likely reason is the cold start — the
replay's set is empty at step 1 while the real arena has 128 tokens of prefill behind it — and rather
than settle it, the useful conclusion is that **the replay is a floor on this lever and not a price.**
The rows below 300 are the one regime the pool has not tested: the 0.0% there is a statement about a
set narrower than one step's 240 draws, and the narrowest width the real arena has been run at is 300.

`--expert-hot-rows` and `--expert-pool-rows` are 2.6–4.2× levers on a **prefill**. This pass asked
the decode question directly, by recording what a decode step actually draws and replaying it through
a cache, and got a negative-looking result whose *shape* is worth knowing even though its price is
not — the shape is the boundary at one step's width, and the price is above.

A row is **18,800,640 B = 17.93 MiB**. `/tmp/probe_v41_decode_locality.py --locality` records the
gate's own `indices` for 24 decode steps, which is **240 draws a step** (40 layers × 6 slots) and is a
global measurement — it is the layer's routing, not the process's deal. The cumulative distinct union
grows **240, 391, 511, 638, 746, 869, 959, 1042, 1112, 1205, 1316, 1428, 1535, 1609, 1703, 1788,
1871, 1974, 2041, 2106, 2158, 2236, 2307, 2374** — still adding ~94 rows a step at step 24, so the
union has not converged at 42.02 GiB. Per-step, the fraction already seen goes 0.0%, 37.1%, 50.0%,
… 78.3% at step 20 and 67.5–72.1% at steps 21–23.

`/tmp/locality3.out` replays those same draws through a least-recently-used and a least-frequently-used
set of `N` rows:

| resident set | bytes | LRU, all steps | LRU, last 8 | LFU, all steps | LFU, last 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 rows | 1.75 GiB | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| 200 rows | 3.50 GiB | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| 300 rows | 5.25 GiB | 25.8% | 21.2% | 24.7% | 20.6% |
| 400 rows | 7.00 GiB | 27.2% | 21.4% | 31.0% | 26.5% |
| 600 rows | 10.51 GiB | 39.6% | 39.0% | 42.6% | 41.5% |
| 900 rows | 15.76 GiB | 46.1% | 46.0% | 49.1% | 52.0% |
| 1400 rows | 24.51 GiB | 54.5% | 59.5% | 54.6% | 59.6% |
| 2400 rows | 42.02 GiB | 58.9% | 70.3% | 58.9% | 70.3% |

**The 0.0% rows are the shape worth keeping.** A set narrower than one step's own width is not a
smaller cache, it is no cache at all: a step inserts 240 rows with no repeat *inside* the step, so 100
or 200 sequential insertions evict everything that would have been a hit before the next step asks for
it, and the boundary is sharp rather than gradual — 200 rows is 0.0% and 300 is 25.8%. **The real
path has not been run below 300 rows**, so this is a bound on the replay and not a measured property
of the arena; it belongs with the two columns above as a reason to read the replay as a floor. Note
also that the real arena is keyed on `(layer, expert)` — a fix that had to be made before any of the
measured columns here were quotable, because the earlier key was the bare expert id, which answers a
different layer's row with a confidently wrong tensor.

**What it is worth, and what it costs, as measured.** The sitting above returns **218 ms of a 765 ms
step, 1.399×**, for **10794 MiB of arena a card** and 18.76 GiB of `cuda` against the no-arena step's
9.71 — so it takes **9.05 GiB more than the step already holds**, which is a real charge against a
22 GiB card and the KV cache's share rather than the "8–10 GiB spare" this section originally
budgeted. **That charge is what sizes the shipped default, and the default is 288.** 600 buys its
1.399× for 18.76 GiB of the card and leaves the KV cache 3–4 GiB; 288 buys the 1.224× the 300-row
column below measures — 288 is a few rows under that sitting — for 5200 MiB of arena and leaves about
8.5 GiB, and a default that OOMs on a long prompt is not a default. 600 stays reachable by name for a
decode-heavy short-context run. What settles the *direction* of the default is that 0 is not the
neutral one: `expert_batched` cannot run without a pool, because the batched call reads each arena row
it is handed as one expert's bytes for a whole chunk, so `--expert-pool-rows 0` drops the prefill's
two best-measured mechanisms at once. On a 512-token prompt that coupling is **179.67 s against
21.28 s, 8.44×**, in [the acceptance sitting
on the device page](deepseek_v4_1_flash_device_experts.md#the-launchers-default-is-288-and-the-acceptance-sitting-behind-it).

**Half the arena buys 69% of the win.** At 128 tokens and 64 steps the same sitting carries 300 rows
against 600: **1.224× against 1.320×**, a decode-phase hit rate of 29.9% against 44.6%, for 5415 MiB
of arena against 10794 and 13.51 GiB of card against 18.76. So the width is a tuning knob with a
knee in it rather than an on/off switch, and the arena a decode wants is one it can fill and refill
rather than the prompt-wide width a prefill wants. What that says about the earlier conclusion here —
that 600 rows is a quarter of the 2374-row union and therefore cannot pay for its card, and that "the
same knob is a 2.7× on one and a 1.1× on the other" — is that both are falsified by the direct
sitting. The prefill result does transfer, at 1.40×, and the two columns above are why the replay did
not show it.

**Why the two readings differ, stated once.** A prefill's floor is *a layer's distinct experts* — ~142
of 384 on rank 0 there — because a prefill's rows are in flight together and a layer asks for its
whole draw at once. A decode *step*'s floor is its own 240 scattered draws, and that is the floor the
replay measures. A decode *generation*'s floor is the working set it re-draws, which is smaller and
is what the pool holds across steps: 42.9–45.5% of every step's draws at every prompt length, 61.3%
at an eight-token prompt, and 69.3–86.1% of a prefill's. The knob is a 4.0–4.2× on a prefill and a
1.40× on a decode, and the difference between those two numbers is the number of steps, not the shape
of the access pattern.

## Lever 3 — `_upload` is the leg that becomes the wall

`_upload` is **16.0–17.8 ms** of the step, and it runs at the link's own rate. Re-measured for this
pass on this host (`/tmp/bench_stage2.py --part c`, `cuda:0`, 128 MiB, 22 threads):

| H2D source, 128 MiB | time | rate |
| --- | ---: | ---: |
| pinned, sync | 11.69 ms | **11.49 GB/s** |
| pageable, sync (the driver's bounce) | 11.87 ms | 11.31 GB/s |
| the pageable source, now `cudaHostRegister`-ed | 11.87 ms | **11.31 GB/s** |

**11.3–11.5 GB/s ≈ 91 Gb/s is the link and registering the source buys exactly nothing** — 11.31
before and 11.31 after, to the hundredth. The same probe's part A also says the trick is *unavailable*
for this workload rather than merely useless: registering a pageable tensor, an anonymous private
mmap, and a `/dev/shm` `MAP_SHARED` mapping all succeed (`rc=0`), and **the read-only `/dev/shm`
mapping the bank is fails `rc=712` "part or all of the requested memory range is already mapped"** —
the bank is already mapped, so it cannot be re-registered as a DMA source. The `shm → pinned` hop is
therefore not removable by that route; it is removable by not having a hop, which is Lever 2.

Across four cards the aggregate is **38.56 GiB/s** against one card's **10.47** (`/tmp/probe_h2d.py`),
and that aggregation is the entire reason the split exists: the same 4.20 GiB of a step is **0.11 s of
transfer on four cards against 0.51 s on one**. One link is already saturated by a single card.

So the ordering of the two largest terms is a fact about the step rather than a consequence of Lever 1,
which is now closed: today the host copy is the longer leg by ~14×, and removing it makes the H2D the
longer leg. The only way past either is **not to move the bytes**, which is Lever 2 — and Lever 2 is
**1.399× measured**, so the honest statement is that the expert path's floor on this host is one queue
of 4.2 GiB a step through a PCIe 3.0 x16 link, that a decode-resident set answers ~45% of that queue
at a width the card can pay for, and that every remaining lever is about how much of the queue is on
the critical path rather than how fast it goes.

## Lever 4 — `hc_mixes` has 208 µs a call that is inside no kernel

`hc_mixes` is **27.72 ms/step** over 80 calls — **346.5 µs a call** for an op whose arithmetic is a
`[1, 20480] @ [20480, 24]` GEMV and a handful of elementwise passes over 20,480 numbers. Re-run for
this pass to `/tmp/hc_mixes.out`, on one idle card, no checkpoint and no second rank:

| piece | time | rate |
| --- | ---: | ---: |
| `F.linear([1, 20480], [24, 20480])` fp32 — the GEMV | 30.0 µs | 65.51 GB/s |
| `x4.flatten(2).float()` | 2.3 µs | 72.73 GB/s |
| `rsqrt(square().mean(-1)) + 1e-6` on `[1, 4, 5120]` | 60.2 µs | 2.72 GB/s |
| `hc_split_sinkhorn` (the fused kernel, the default path) | 80.0 µs | — |
| `hc_split_sinkhorn_torch` (the 19-iteration loop, now a fallback only) | 1719.1 µs | — |
| **sum of those pieces** | **172.5 µs** | |
| **the method as written, timed as one call** | **138.5 µs** | 14.20 GB/s |
| **the op as the step measures it** | **346.5 µs** | |

**Half of the op is not in any of its pieces.** Timed as one call, the method the module actually runs
is **138.5 µs** — less than the sum of its parts, because the parts were each timed with their own
sync — and the step charges 346.5. The **208 µs a call** between those two is the method's own Python
around a real plan: the flatten, the view, the `rsqrt` argument build, the module wrapper, the ATen
dispatches between them, and the extra sync points a real call has and a bench does not. At 80 calls a
step that residue is **16.6 ms of a 27.72 ms term**, and no kernel change addresses it. The 80.0 µs
the sinkhorn costs is the only part of the op that is a kernel at all.

The GEMV's 65.51 GB/s is the **N=24 shape, not the layout and not the dtype**, and three variants say
so on the same bytes: `torch.mv(w, x[0])` is 31.4 µs, `x @ w.t()` is 29.4, and the transposed-contiguous
layout form is 33.9 — all within 15% of each other, where widening the output to N=256 is **40.8 µs
and 514.64 GB/s**, N=1024 is 565.49 and N=4096 is 581.07. So the wall is ~580 GB/s and 24 columns is
11% of it; and because the op is called on one row with `hc_mult = 4` in every call, four rows could
be fused into one N=96 GEMV without changing an answer.

The `rsqrt` line is the second-largest piece and the worst rate: **60.2 µs at 2.72 GB/s** for a
mean-square over 20,480 numbers and a reciprocal square root. It is one elementwise kernel's worth of
work being done by several ATen ops.

**What the whole op is worth: 16.6 ms a step, and it is dispatch.** The op is 346.5 µs a call of which
the method timed as one call is 138.5, so the reachable part is the **208 µs of glue — 16.6 ms of the
27.72 ms term**, 2.3% of the step. The remaining 138.5 is already the arithmetic plus the fused
sinkhorn's own 80 µs, so a fusion that also beat the sinkhorn would be bounded by the whole 27.72 ms
(3.8%) and no better — and the same tax is what Lever 6's graph removes wholesale, on every op rather
than this one. Sized that way it is the least valuable of the six levers; it is listed because it is
the cheapest to try and because it is a prerequisite-free way to see the glue's size before paying
for a graph.

## Lever 5 — the prefill is a batch-shape problem, and the fix is in: −27% at 512 tokens

A 128-token prefill is **42.86 s, 3.0 tok/s, 41.85 of it inside `DeviceRoutedExperts`** — because a
prefill of `n` rows stages a layer's 4.2 GiB *n* times, once per row, and 128 rows is what that says.
Everything that has been done to it so far moves the constant: the resident bank 6.3 tok/s, the
per-layer set 2.6–2.7×, the pool **11.4 tok/s**, the row-deep pipeline 1.10×.

What none of them changes is the **shape** of the floor. The pool's is *a layer's distinct experts* —
~142 of 384 on rank 0 — which is why its width is not prompt-independent: 148 and 288 rows stage the
same 5700 rows and 96 stages 6579, and shortening the pass to a quarter moves the floor to ~79 experts
a layer and the right width to 64 rows rather than 148, for 2.3–2.4× instead of 4.0–4.2×. A longer
prefill raises the floor toward 384 rows, **6.9 GiB a card**, past what the four cards can give once
the tree and the caches are on them.

**`moe_multi_token_fp4_forward` (`src/csrc/cuda_kernel_impl.cu:3175`) was the change this page bet
would make the floor a function of the batch instead of the layer**, and it is now wired in — the
`--expert-batched` path in `DeviceRoutedExperts`, which resolves the whole pass before staging any of
it and then issues one call a chunk instead of one call a row. **The bet was half right.** It is the
largest prefill lever measured on this path, but not through the floor: the staged rows come out
**identical to the per-row path** (3255/3288/1937/1963 at 128 tokens, 5623/5673/3791/3740 at 512, and
the pool's took-in/evicted columns with them), so the floor is still a layer's distinct experts. What
collapses is the number of kernel invocations that each read those rows: 128 rows a layer is 42–80
calls instead of 128, and 512 rows is **40 calls — one a layer, 512.0 rows a call**.

End to end, `--decode 0` so the prefill's own counters are what is compared, against the same sitting
of the per-row path:

| | r0 | r1 | r2 | r3 | mean | tok/s | chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 tokens, `--expert-pool-rows 288`, per-row | 45.4 | 47.3 | 47.4 | 50.4 | 47.63 s | 10.75 | — |
| 512 tokens, 288 rows, batched | 36.5 | 35.6 | 33.6 | 33.3 | **34.75 s** | **14.73** | 40/40/40/40 |
| 128 tokens, `--expert-pool-rows 64`, per-row | 22.4 | 20.5 | 19.8 | 23.9 | 21.65 s | 5.91 | — |
| 128 tokens, 64 rows, batched | 18.9 | 21.0 | 18.1 | 18.8 | **19.20 s** | **6.66** | 75/80/42/42 |
| 128 tokens, 64 rows, batched again | 19.0 | 19.8 | 19.8 | 18.7 | **19.33 s** | **6.62** | 75/80/42/42 |

**−27.0% at 512 tokens and −11.3% at 128**, and the logits are **bit-identical on all four ranks at
both widths**: `32/32 of the top 32 identical, worst |dlogit| 0.000e+00` of a max |logit| of 2.915e+01
and 2.782e+01 respectively, with the pool counters exactly equal. The two repeat legs cut the same
chunk counts on every rank, so the partition is a property of the routing and not of a race. The arena
grows by what a chunk holds in flight: 13.31 → 13.90 GiB a card at 288 rows, 9.38 → 9.53 at 64.

Three traps carried from the work that got here, the third of which this round paid for:

- The pool's key must carry the **layer** as well as the expert — it was keyed on the bare expert id
  once, which returned a wrong answer deterministically.
- The routed experts' expert-parallel partial has to land in the same all-reduce as the shared
  expert's row-parallel partial rather than beside it.
- **A chunk may not contain two rows that read the same arena row for different experts**, and the
  rule that decides it must claim the rows a row *reads* and not only the rows it *stages*. A row
  whose draws all hit the pool stages nothing and still reads arena rows, so a rule built from the
  misses leaves such a row beside the row that is about to overwrite what it reads. At
  `--expert-pool-rows 64` the misses-only rule leaves **two arena rows holding two experts inside one
  call**, on two of the four ranks, and the run it measured moved **12 of the top 32 logits by
  1.220e+00 of a max |logit| of 2.782e+01 (4.39e-02 relative)**; at 288 rows it leaves none, which is
  why a sweep at the width above never caught it. Both readings are from the same probe, the same
  prompt and the same routing, differing only in which rule cut the chunks
  (`/tmp/probe_v41_oldrule.py`, `/tmp/probe_v41_keyalias.py`).

## Lever 6 — the graph, which is bounded by the same thing every other lever is

The per-layer CUDA graph is the only lever that reaches the 1.5 ms of device-busy TP4 measured, and it
is the reason the tree moved onto the cards at all. It is **still gated** — the round it belongs to
was run as 先搬，量完再说, and this page is the 量 rather than an authorisation. Its bound is worth
recording accurately so it is not over-sold:

- The step launches **7,076 kernels**, sixteen of which are 47% and the rest a tail over 106 names.
  A graph is the right instrument for that tail, and there is nothing else in this page that is.
- **111–122 ms** of the phase clock's total is outside every named call, and Lever 4 priced 208 µs a
  call of the same kind inside one op. That residue — Python, `torch.cuda` calls, argument marshalling
  — is what a graph removes and it is the same work the per-op table's `hc_mixes` row cannot see.
- **88 `ncclDevKernel_AllReduce` calls a step** are the part a graph has to capture as collective
  nodes or leave outside it. Three all-reduces a layer is a property of the split and not something a
  graph changes.
- The contrast with Qwen on this same hardware is the reason this is not a small number here: there a
  graph was worth **3.7%** because decode was already 96.3% GPU-busy, with 1,125 launches a step
  leaving a 1.0 ms gap. Here device-busy is 11% and the launches are six times as many, so the same
  instrument is bounded by the step and not by a gap.

## A reading rule the per-op table needs

The per-op table's `GB/s` column is **unsharded bytes over sharded time**. It is consistent for every
row — `wq_a`'s 255.9 is its whole 5120 x 1280 bf16 over 51.25 µs, `wkv`'s 112.9 is its whole
512 x 5120 — but at TP4 an op the split cuts moves a **quarter** of the bytes the column counts, so
`wq_b`'s **1790.9 GB/s** is a full `[32768, 1280]` bf16 (83.9 MB) over a 46.75 µs sharded call whose
shard is 20.97 MB: **448.5 GB/s**, which is what the card's memory system actually delivered. It is
not a timing artifact and the column is arithmetically right; it is a convention that reads ~4× the
card's rate for any cut op. **Read `ms/step` for cost and never read that column as a roofline** — and
for scale, a bf16 GEMV of the same byte count measures **515.79 GB/s** on this card
(`/tmp/hc_mixes.out`), which is what makes the 448.5 a plausible number rather than a suspicious one.

## Ranked

Decode, against 722.0 ms; prefill against 42.86 s / 3.0 tok/s. The worth column is what the measurement
supports as an **upper bound** at the configuration named, and the two gated rows say so.

| # | Lever | Measured basis | Worth | Cost / gate |
| ---: | --- | --- | ---: | --- |
| 1 | ~~Overlap layer `k+1`'s `_stage` with layer `k`'s tree~~ | five sittings of the same probe: the 91.8 ms replay is the seventh pass of a pool that is warming, and re-inserting the recorded gap between the copies costs 679–940 ms a step against the step's own 116–225 | **falsified** — 3–8× spent, not 15% returned | closed; the banner in Lever 1 and `/tmp/sched5.log` |
| 2 | A decode-resident set, `--expert-pool-rows` | A-B-A-B at 128 tokens, 256 decode steps, control at both ends: 765.0 against 547.0 ms a token at 600 rows, top-32 logits bit-identical; and at the shipped default's own 288 rows the same sitting reads **0.601 / 0.578 against 0.769 s** | **218 ms** (1.399×) at 600 rows, **~0.19 s (1.28×)** at 288 for half the arena | **default 288** — 5200 MiB of arena, 13.3 GiB of `cuda`, leaves ~8.5 GiB for the KV cache; 600 is 10794 MiB and 9.05 GiB a card above the no-arena step's own 9.71 |
| 3 | **Wire in `moe_multi_token_fp4_forward`** | `--decode 0`, per-row against batched on the same sitting: 512 tokens at 288 rows is 47.63 → 34.75 s a rank, 128 at 64 rows is 21.65 → 19.20/19.33 s, top-32 logits bit-identical on all four ranks and the pool counters equal | **−27.0%** on a 512-token prefill, −11.3% on a 128-token one; the staged rows are unchanged, so it is the call count (512 → 40, one a layer) and not the floor | **spent**; default on, and it falls back to the per-row path without a pool |
| 4 | Per-layer CUDA graph | 7,076 launches, 111–122 ms unattributed, 11% device-busy | the largest, and **not sized** — bounded by the step, ~2–4× on the tree's own terms is the shape | **gated** on the round's own numbers, per 先搬，量完再说 |
| 5 | A fused `hc_mixes` | 346.5 µs a call against 138.5 µs of the method timed as one call | **~17 ms** (2.3%) | row 4's graph eats the same glue — do one, not both |
| 6 | Per-rank `--threads` | replay optimum 22/8/12/22 against a uniform 22 | **≤16 ms** on `_stage` | the thread count is otherwise spent; 8 is the cliff |
| 7 | Anything on `_upload` | 11.3–11.5 GB/s, `cudaHostRegister` measured not to help | **0 ms** today; it becomes the longer leg only if the host copy is removed | only #2 addresses it, by not moving the bytes |

**Every row above is priced at a short prompt, and only one row of a chunk scales with context.** Rows
2 and 3 are 512- and 128-token prefill levers; a 4096-token chunk at 262144 is a different list, and on
it **`attn.indexer` plus `attn.compress_kv` are 3.19 s of a chunk at 32768 against 10.24 s of one at
262144 — 11.6% to 32.7%** — while every other row of the two phase tables is flat or lower over the
same span. Its two levers are the indexer's `all_reduce`, which upcasts to fp32 and so puts 33.6 MB on
the wire for a 16.8 MB tile — **5.47 ms a level-one tile measured on this fabric against 2.88 ms in
bf16, 1.79 s of the prefix path's 3.53 s at 262144, and the prediction lands on the tap's own `sync`
column to 0.89 at 32768 and 0.84 at 262144** — and a one-tile lookahead over that collective, which
has no numerics gate and can hide the whole of it. The wire dtype's gate is **closed**: fp16 on it
picks a different set on all eight indexer layers, and the candidate path's half of the row does not
grow with context at all (2.035 s at 32768 against 2.084 at 262144).
[the row that grows with context](deepseek_v4_1_flash_chunked_prefill.md#the-one-row-that-grows-with-context)
takes it apart and names what gates each.

## Falsified, so do not re-run these

- **Cross-layer `_stage` overlap.** The in-situ/replay gap is a one-time pool warm-up and not a
  schedule: re-inserting the recorded 13.3–15.6 ms gap between the same copies costs **679–940 ms a
  step against the step's own 116–225**, and the calling thread is not the copy's bottleneck
  (CPU/wall 8.84–13.91×), so a background thread buys nothing either. Five sittings,
  `/tmp/probe_stage_sched.py`, `/tmp/sched.log`–`/tmp/sched5.log`.
- **A resident set narrower than one step.** 100 and 200 rows are exactly **0.0%**, not "less". A
  decode step inserts 240 rows with no repeat inside it, so anything under ~300 thrashes to zero.
  This is a replay result: the real arena has not been run below 300 rows, so read the boundary as a
  bound on the mechanism and not as a measured width.
- **The thread count as a lever.** One thread is 2.91 GB/s and 22 is 16.39, but 22 is already where
  every rank's curve flattens; the per-rank optimum is worth **7.1% of the step's own 256 ms term,
  ~16 ms**, and exactly 0% on the two ranks whose 22-thread curve is already their best.
- **`cudaHostRegister` for the H2D.** **11.31 registered against 11.31 unregistered**, the same to
  the hundredth — and the bank's read-only `/dev/shm` mapping cannot be registered at all (`rc=712`).
- **Merging the H2D calls.** V4-Flash's decode went 3.4 → 1.5 with one large H2D; the transfer needs
  calls to hide behind each other, not fewer of them.
- **Fusing `hc_split_sinkhorn`.** It is fused and it is the default path (`src/kernels/ops.py:84`);
  the 1719.1 µs body is a fallback that a CUDA tensor never reaches.
- **The per-op table's `GB/s` column as a roofline.** It counts unsharded bytes against sharded time;
  at TP4 it reads ~4× high for every op the split cuts.
- **That the prefill's resident set does not transfer to a decode.** The old reason given here was the
  shape of the access pattern. A prefill's floor is a layer's distinct experts and a decode *step*'s is
  its own 240 draws, but a decode *generation*'s is the working set it re-draws, which the pool holds
  across steps. The knob is 4.0–4.2× on a prefill and **1.399×** on a decode, and the difference
  between those two numbers is the step count, not the access pattern.
- **Caching `_stage`'s key strings and its `checkpoint.packed()` lookup.** The proposition was that
  the 501 µs a copy held ~230 µs of Python, which would have been worth seconds a pass.
  `/tmp/probe_v41_stage_micro.py` splits it: the two f-strings are **0.5 µs**, `packed()` +
  `view()` **8.3 µs**, the whole `_stage` body **130.2 µs** and `copy_` alone into pinned **77.3 µs**,
  so the Python is **~16 µs of the 501** and the ceiling is **~0.3 s**. The rest of the 501 is
  contention for the host's memory system: the identical loop measures **133.8 µs a copy solo** and
  **5353 / 5416 / 7258 / 6180 µs** as four processes on disjoint expert bands, a 40–54× spread that is
  asymmetric across the four. Both probes are
  [in the expert page](deepseek_v4_1_flash_device_experts.md#a-pool-hit-used-to-take-a-staging-buffer-and-the-copy-behind-it-was-left-uncovered).
  The corollary is a reading rule for every term on this page: **a wait on this path is worth less than
  the bytes behind it**, which is why the buffer-rotation fix below is bit-exact and still inside the
  noise.
- **The two-buffer rotation as a `_take_buffer` cost.** `_take_buffer` is 10–29 ms against a stage of
  4–8 s *with no pool*, where every row stages; a pool makes the rotation and the row count disagree,
  and against a 72.5%-empty pass it walks onto a slot whose copy has just been issued. It cost
  **6.89 s of a 30.35 s class wall** at 512 tokens and 288 rows. The fix — a row that moves nothing
  takes no slot — is bit-exact and drops the wait to 0.09–0.25 s on the low-pressure ranks, but the
  end-to-end A/B is −6% with **overlapping ranges** (33.95 against 31.90 s mean) and `_stage` and
  `_upload` unmoved in both columns, so the honest claim is the mechanism and not the percentage.
- **That the indexer's candidate path gets slower as `index_k` leaves L2.** The gather's index is 4 MiB
  at 32768 (inside this card's 5.5 MiB) and 68 MiB at 262144 (outside it), which predicted a
  per-iteration cost that grows with width — and that growth was the explanation offered for the
  candidate path's 2.084 s at 262144. The 32768 half of the same probe settles it: **2.035 s over the
  same 1024 tiles, flat to 2.4%, 1.99 ms a c-iteration against 2.04**. The candidate path is 77% of the
  2.6 s indexer row at 32768 and does not scale with context at all; the row's growth is the prefix
  path's — +3.000 s of the +3.049 s — and half of that growth is its collective, which
  [the chunked-prefill page](deepseek_v4_1_flash_chunked_prefill.md#the-one-row-that-grows-with-context)
  is corrected on. `/tmp/chunk_indexer_steps_32768.log`.
- **fp16 on the indexer's reduce wire.** Halving the message is worth ~0.93 s of the 5.10 s row at
  262144 and the arm is a dtype on a closure, so it looked free. It is not: `INDEXER_REDUCE_BITS=16`
  moves the selected *set* on all eight indexer layers, against a null — two arms with no knob moved —
  whose own disagreement reproduces to the digit across runs, so the baseline is subtractable and this
  clears it. Layer 2 goes from 3083 differing rows of 4096 to all 4096, and from 621210 differing
  elements to 1640362. Layer 2 is the one that settles it rather than 8/14/20/24/28/32/36: being the
  first indexer, it has no predecessor whose differing selection it could be inheriting.
  `/tmp/indexer_parity_reduce16.log` against `_null.log`.

## Reproducing

```bash
# the draw and union curve over 24 decode steps, and the resident-set replay off the same run.
# Four ranks, one process a card; --out takes a rank suffix so the payloads do not collide.
PYTHONPATH=/mnt/data1/dsv4_inference torchrun --nproc_per_node=4 \
    /tmp/probe_v41_decode_locality.py \
    --length 128 --decode 24 --locality --threads 22 --out /tmp/locality3.pt

# the in-situ _stage clock and its thread sweep: times the real call on step 2, snapshots its
# arguments, replays the same copies at 1/2/4/8/12/16/22 threads, and restores expert_rows and
# drawn_rows afterwards so the replay is not counted twice. `--locality` off, because the point is
# the timing and not the draws.
PYTHONPATH=/mnt/data1/dsv4_inference torchrun --nproc_per_node=4 \
    /tmp/probe_v41_decode_locality.py \
    --length 128 --decode 4 --stage-probe 2 --threads 22 --out /tmp/stage.pt

# `hc_mixes` taken apart: the GEMV at four output widths, the three other ops the method pays,
# the fused sinkhorn against the loop it replaced, and a bf16 GEMV of the same byte count for
# scale. One idle card, no checkpoint, no second rank.
python /tmp/bench_hc_mixes.py

# the resident set priced on the path instead of in a replay: A-B-A-B at 128 tokens, 256 decode
# steps a leg, --expert-pool-rows 600 against 0 with the control at both ends.
for pool in 600 0 600 0; do
  DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_hot_ab.py \
    --length 128 --decode 256 --threads 22 --hot-rows 0 --pool-rows "$pool" --out /tmp/pr_$pool.pt
done

# and what closed Lever 1: the same 40 calls replayed four times over (the pool's warm-up), then
# with the recorded gap re-inserted between them, spinning and sleeping. Five sittings.
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 \
    /tmp/probe_stage_sched.py --length 128 --decode 4 --probe-step 2 --out /tmp/sched.pt

# the batched A/B and the chunk rule behind it. Five serial legs at --decode 0 -- so the prefill's
# own counters are what is compared, the sampler's Gumbel first token not being fed back -- per-row
# then batched at each width, with the 128-token batched leg repeated. Each leg logs to
# /tmp/diag_<tag>.log and writes a payload with a rank suffix; --compare pairs one rank's logits
# between two sittings.
bash /tmp/v41_diag.sh
python /tmp/probe_v41_hot_ab.py --compare /tmp/diag_n512row.pt.r0 /tmp/diag_n512batch.pt.r0

# the same program, prompt and routing with the misses-only rule cutting the chunks instead, which
# is the 12/32 and 4.39e-02 run; and the counter that explains it, one arena row held for two
# different experts inside one call, under both rules in a single sitting. probe_v41_chunk_diag2.py
# is the same comparison per call rather than in aggregate.
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_oldrule.py \
    --length 128 --pool-rows 64 --decode 0 --batched --out /tmp/old128.pt
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_keyalias.py \
    --length 128 --pool-rows 64 --decode 0 --batched --out /tmp/ka.pt

# the buffer-rotation fix, its A/B and the two probes that price what is left of `_stage`. The A/B is
# interleaved -- before, before2, after, before3, after2 -- with the two variants swapped under the
# shipped path between legs and diffed at the end, so the host's drift lands on both columns.
bash /tmp/v41_guard_ab.sh
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python /tmp/probe_v41_stage_micro.py
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python /tmp/probe_v41_stage_contend.py --tag solo
for i in 0 1 2 3; do DEEPSEEK_V41_RESIDENT_EXPERTS=1 \
  python /tmp/probe_v41_stage_contend.py --tag "q$i" --offset "$i" & done; wait
```

`torch.distributed.run` sets `OMP_NUM_THREADS=1` for every worker unless the environment already has
one, so `--threads` is what makes the thread count in these runs mean anything; the two probes above
call `torch.set_num_threads` directly for their sweeps and restore it afterwards. The resident bank
must be attached (`DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `/tmp/resident_bank.py`) or every `_stage` figure
here becomes a disk figure.
