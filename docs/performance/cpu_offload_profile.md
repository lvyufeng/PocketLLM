# CPU offload and prefetch: the measured ceiling

**Date:** 2026-09-15
**Commit:** `043b981`
**Script:** `scripts/profile_cpu_offload.py`
**Artifact:** `/tmp/cpu_offload_profile.json` from
`python scripts/profile_cpu_offload.py --stage all --json /tmp/cpu_offload_profile.json`
**Issue:** #156 (CPU Offloading with Prefetch Pipeline), roadmap entry 15 of #151

Issue #156 promises four things for layer-granularity CPU offloading: a 70B FP4 model
runnable on a 2080 Ti, decode at **≥ 5 tok/s**, prefetch hiding **> 80 %** of the H2D
traffic, and token parity. None of the four had been measured on this host, and the
arithmetic behind them does not close on its own terms. The roadmap's section 2.1
budgets "Decode: ~5 tok/s（每层 ~200ms，其中 H2D ~100ms + 计算 ~100ms）"
(`docs/architecture/pocketllm_roadmap_old_hardware.md:170`) — ~200 ms per layer, of
which H2D ~100 ms and compute ~100 ms — for a 70B FP4 layer that is 0.42 GiB. At the
measured 10.53 GiB/s that is 39 ms of copy, not 100 ms; and the compute half is 3.6 ms
per layer at one row, not 100 ms (Result 4).

Roadmap entry 15 of #151 asks for exactly this measurement before any throughput is
promised:

> Start with real-checkpoint memory and transfer profiles, distinguish cold storage
> from warm host-cache behavior, and establish the overlap ceiling before promising
> throughput.

This page records that measurement. Nothing here changes a runtime path, and no
engine code was written: the numbers are the go/no-go input for whether
layer-granularity offload should be built at all.

## What was measured, and the distinction that makes it a new question

Every CPU/GPU mixed path in this repository is **expert**-granular: GLM's
active-expert staging, the Qwen4-Exp host-resident expert shard, the cpp_engine GGUF
Q2 staging. Expert granularity works because MoE routes each token to a handful of
experts, so most expert bytes never move at all — Qwen4-Exp moves one 56.25 GiB/rank
shard once and then only the routed experts, which is why it gained 2.07× on prefill
([Qwen4-Exp heterogeneous TP4 performance](qwen4_exp_performance.md)).

**Layer** granularity — llama.cpp's `--gpu-layers N` — moves every weight of every
offloaded layer on every token. Nothing in this repository implements it: a
whole-repository grep for `gpu_layers|gpu-layers|n_gpu_layers` matches only this page
and the script behind it, both naming the flag to say what is *not* implemented. The
two are not comparable budgets, and only the second is what #156 asks for.

The difference is arithmetic, not a matter of degree. A GLM-5.2 MoE block is
6,070,800,384 B, of which 5,838,471,168 B is `ffn_{gate,up,down}_exps`. With
`expert_used_count = 8` out of 256 experts, the arithmetic reads 0.386 GiB of that
block — so offloading the layer moves 14.6× what the layer needs. Under expert
granularity the same block moves 1.0× of what it needs. That ratio, reported as
`move_amplification`, is the whole question this page answers.

## Measurement conditions

| Item | Value |
| --- | --- |
| Host | x86_64 CUDA machine, `root` |
| GPU | 4 × RTX 2080 Ti, 22528 MiB each, sm_75 |
| PCIe / topology | Gen3; GPUs 0–1 on NUMA node 0 (PHB pair), GPUs 2–3 on node 1 (NV2 pair); every cross-pair SYS |
| CPU / RAM | 2 × Intel Xeon E5-2696 v4 @ 2.20 GHz, 2 NUMA nodes, ~1 TiB RAM |
| OS / Python | Ubuntu 22.04.5, kernel 5.15, CPython 3.11.14 (conda `deepseek`) |
| torch / CUDA | torch 2.9.1+cu128, CUDA runtime 12.8 |
| Checkpoints | GLM-5.2-GGUF-UD-Q4_K_M (`/mnt/data3/GLM-5.2-GGUF/UD-Q4_K_M`, 11 shards), Qwen3.8-Flash-Next (`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`), Qwen3.8-27B-FP8 (`/mnt/data2/Qwen3.8-27B-FP8`) — GGUF/safetensors headers and directory listings only, no tensor data read. No 70B checkpoint of any kind is present. |
| Stage wall time | storage 46.07 s, h2d 4.11 s, overlap 33.41 s, budget 0.22 s |

`--stage storage` samples 2 GiB per mount from a fixed offset of 64 MiB into the
largest file on each mount — for this run an unrelated 95.7 GiB tarball on `/mnt/data1`,
the Qwen3.8-27B-NVFP4 checkpoint on `/mnt/data2`, and a GLM-5.2 Q2_K_XL shard on
`/mnt/data3` — and evicts only that range
(`posix_fadvise(POSIX_FADV_DONTNEED)`) rather than dropping the whole page cache,
which would perturb unrelated work on a shared machine. `--stage h2d` stays under
1 GiB per GPU. `--stage overlap` runs one GPU at a time and peaks at ~13 GiB of
device memory, because the GLM spec's double buffer is two 6.07 GiB staging buffers
plus the layer's own weights. `--stage budget` reads GGUF/safetensors headers only and
never touches tensor data.

Every rate below is wall time, with the target device drained by
`torch.cuda.synchronize(device)` after the loop. A bare `torch.cuda.synchronize()`
drains only the *current* device, and timing `cuda:3` that way reports rates four
orders of magnitude too high because nothing waits for the copy just enqueued; that
bug was found and fixed during this work and the rates below are from after the fix.

## Result 1: every checkpoint mount on this host is a spinning disk

| Mount | Device | Model | Kind | Cold GiB/s | Warm GiB/s | Warm, 4 threads GiB/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `/mnt/data1` | `/dev/sdc` | ST4000VX015-3CU1 | rotational | **0.096** (20.86 s) | 3.00 | 10.77 |
| `/mnt/data2` | `/dev/sdb1` | ST1000DM003-1SB1 | rotational | **0.194** (10.28 s) | 3.08 | 11.65 |
| `/mnt/data3` | `/dev/sda` | HGST HSH721414AL | rotational | **0.209** (9.59 s) | 4.15 | 11.86 |

Producing command: `python scripts/profile_cpu_offload.py --stage storage`.

Cold is single-threaded after evicting the sampled range; warm is the same read
repeated (page cache); 4-thread is four threads over disjoint 8 MiB chunks of the
same range. All three mounts report `ROTA=1` in `lsblk`.

This corrects a premise of the plan this work was scoped under, which recorded
`/mnt/data2` as an SSD on the strength of a 2.92 GiB/s reading. That reading was a
warm page-cache hit, not a disk rate: `/mnt/data2` is a 931 GiB Seagate
ST1000DM003 on `/dev/sdb`, and its cold read is 0.194 GiB/s — the same order as the
other two, not 15× faster. The real SSDs on this host are `sdd`, `sde` and `nvme0n1`,
and none of them holds a checkpoint. The three cold figures are also the noisiest
numbers in this record: on rotating media the rate depends on which part of the
platter the sampled region happens to occupy, and repeat runs of this stage have
differed by around a quarter at the same offset. The ordering and the conclusion
survive that; the third digit does not.

The consequence is the one the roadmap already anticipated: **the host can only feed
PCIe out of page cache.** Reading weights from disk is 50–110× slower than the PCIe
link, so any offload design here has to be "host RAM resident, PCIe transferred", and
prefetching from disk cannot be hidden behind anything. 1 TiB of RAM is enough to
hold GLM-5.2 Q4 (434 GiB) entirely, so the residency part is affordable; the transfer
budget below is what is left to argue about.

## Result 2: H2D sweep — where copies stop being latency-bound

| Copy size | Pinned GiB/s | Pageable GiB/s | Ratio |
| ---: | ---: | ---: | ---: |
| 4 KiB | 0.46 | 0.46 | 1.00 |
| 16 KiB | 1.57 | 1.57 | 1.00 |
| 64 KiB | 6.81 | 4.01 | 1.70 |
| 256 KiB | 9.82 | 6.71 | 1.46 |
| 1 MiB | 10.29 | 10.06 | 1.02 |
| 4 MiB | 10.38 | 10.13 | 1.02 |
| 16 MiB | 10.44 | 10.17 | 1.03 |
| 64 MiB | 10.51 | 9.40 | 1.12 |
| 256 MiB | 10.53 | 8.27 | 1.27 |

Producing command: `python scripts/profile_cpu_offload.py --stage h2d`.

One GPU (`cuda:0`), pinned source allocated with `pin_memory=True` and pageable source
with a plain `torch.empty`; both sources are written before timing, because a
never-written anonymous page reads back as the shared zero page and would make the
read side of the copy free, flattering the pageable column.

Two conclusions that a double-buffered pipeline depends on:

- **At 64 KiB and below a copy is latency-bound, not bandwidth-bound.** 64 KiB
  reaches only 6.8 GiB/s of the 10.53 GiB/s plateau, 16 KiB reaches 1.6 GiB/s and
  4 KiB reaches 0.46 GiB/s. A staging ring whose chunks are small loses most of the
  link to per-copy latency; the chunk that matters here is a whole layer, hundreds of
  MiB to GiB, so this is not a problem in practice — but it does rule out "prefetch in
  small pieces" as a design.
- **Pinned is never slower than pageable, and pageable only keeps up in a narrow
  window.** Pinned holds 10.3–10.5 GiB/s from 1 MiB upwards. Pageable matches it to
  within 3 % at 1–16 MiB, and then falls away to 9.40 GiB/s at 64 MiB and 8.27 GiB/s
  at 256 MiB — 11 % and 21 % behind, at exactly the chunk sizes a layer-granular
  pipeline would use. The staging ring has to be pinned.

Four GPUs, 64 MiB each, concurrent:

| Mode | Aggregate GiB/s | Per GPU GiB/s |
| --- | ---: | ---: |
| Pinned | **40.80** | 10.20 |
| Pageable | 16.50 | 4.13 |

This is 40.80 GiB/s against the recorded 42.42 GiB/s aggregate in
[Qwen4-Exp heterogeneous TP4 performance](qwen4_exp_performance.md), and three
further runs of this stage landed at 40.99, 41.26 and 41.64 GiB/s — within 4 % of the
recorded figure. The link is not the constraint on four concurrent copies: each GPU
gets its own ~10.2 GiB/s.

**How much of this table is reproducible:** the three repeat runs put the pinned
plateau from 1 MiB up at 10.3–11.1 GiB/s (± 4 %) and the aggregate at 40.8–41.6
GiB/s, but they moved the latency-bound points and the pageable tail by as much as
30 % — 256 MiB pageable was 6.7–8.3 GiB/s across the four runs, and 64 KiB pinned was
6.8–9.2. The shape is identical in every run and only the shape is load-bearing here.

## Result 3: NUMA placement is not a lever on this host

64 MiB pinned buffers, 5 timed iterations each after 2 warm-ups, page placement
verified from `/proc/self/numa_maps`:

| Buffer pages on node | Target | GiB/s |
| --- | --- | ---: |
| 0 | `cuda:0` | 10.76 |
| 0 | `cuda:3` | 10.59 |
| 1 | `cuda:0` | 10.52 |
| 1 | `cuda:3` | 10.64 |

Both local pairings are ahead of their remote counterparts — node-0 pages to `cuda:0`
by 0.24 GiB/s (2.3 %) and node-1 pages to `cuda:3` by 0.06 GiB/s (0.5 %) — which is
the expected direction but an order of magnitude below the 10–20 % a remote-node
penalty usually costs, and well inside the ± 4 % run-to-run spread of the pinned
plateau measured in Result 2. The link, not the page placement, is the limit, and the
reason is worth recording because it is not what the plan assumed:

- **`torch.empty(..., pin_memory=True)` ignores thread NUMA affinity here.** Under
  both node-0 and node-1 CPU affinity, a 64 MiB pinned buffer lands entirely on
  **node 1** — recorded as `{'1': 16384}` pages in both cases. `cudaHostAlloc` places
  the pages itself and, on this driver, picks node 1 regardless.
- To actually pin pages on a chosen node, the script allocates with
  `numa_alloc_onnode`, verifies the placement, and then calls `cudaHostRegister` on
  that anonymous buffer (class `_NodePinned`). Only then does the A/B vary the node
  at all, and the answer is that it does not matter.

So the honest conclusion is not "NUMA pinning is a win" but "NUMA pinning is
achievable and buys nothing on this host": whichever node the pages sit on, the copy
runs at the PCIe plateau. Any future plan that lists NUMA placement as a lever for
H2D bandwidth has been measured and should be dropped.

**A constraint that must survive into any implementation:**
`cpp_engine/tests/probe_host_register.cpp:1-11` bans registering an entire
checkpoint's file-backed mmap, because pinning 80 GB+ of file pages poisons Linux
dirty-page accounting and stalls unrelated `fsync`, build and git I/O system-wide.
#156's "CPU weights must be pinned" can therefore only be built as a **bounded
anonymous pinned staging ring** — which is exactly what `_NodePinned` is, and nothing
like pinning a checkpoint.

## Result 4: the overlap ceiling — the core number

Real layer byte counts and real GEMM shapes, on `cuda:0`, fp16, timed over 4 layers
per iteration of a double-buffered two-stream pipeline modelled on
`src/models/qwen4_exp/moe.py:360-426`. `hidden` is
`(serial − overlap) / copy_only`: zero when the pipeline saves nothing, and 1.0 when
the copy is entirely hidden, which needs `compute ≥ copy`. When `compute < copy` the
best it can reach is `compute / copy`, because a pipeline cannot beat the transfer it
is hiding behind — it can only spend as much time as it has compute to spend. That
bound is the last column.

**GLM-5.2 Q4 MoE block — 6.07 GiB per layer, `move_amplification` 14.6×**

| Rows | Copy ms/layer | Compute ms/layer | Serial ms/layer | Overlap ms/layer | Hidden | Max hideable (`compute / copy`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (decode) | 540.48 | 1.18 | 541.79 | 541.31 | **0.09 %** | 0.22 % |
| 512 (prefill) | 540.24 | 6.26 | 546.83 | 542.35 | **0.83 %** | 1.16 % |

**Llama-70B FP4 dense layer — 0.42 GiB per layer, `move_amplification` 1.0×**

| Rows | Copy ms/layer | Compute ms/layer | Serial ms/layer | Overlap ms/layer | Hidden | Max hideable (`compute / copy`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (decode) | 39.79 | 3.58 | 43.54 | 40.88 | **6.67 %** | 9.01 % |
| 512 (prefill) | 39.89 | 15.25 | 55.02 | 43.83 | **28.04 %** | 38.23 % |

Copy rate is 10.46–10.64 GiB/s in all four rows, matching the sweep's plateau.

Producing command:
`python scripts/profile_cpu_offload.py --stage overlap --overlap-rows 1 512`.

The GLM row carries 6.07 GiB through PCIe to run 1.18 ms of arithmetic. The pipeline
cannot hide what is not there to hide behind: when `compute < copy` the most a
perfect pipeline can hide is `compute / copy`, and in all four rows the measured
fraction sits below that bound — recovering 40 % of it in the GLM decode case and
72–74 % in the other three. The 0.8–3.9 ms/layer by which the measured overlap
exceeds `max(copy, compute)` is per-layer event and launch overhead, and it does not
shrink with the ratio. So the gap to the bound is the pipeline's own cost, not a shape
that a better implementation could recover, and the bound itself is what #156 is
arguing with.

**The consequence is that "prefetch hides >80 % of H2D" is not a tuning target that
was missed — it is unreachable for these shapes.** On the column above, hiding 80 %
of the transfer needs `compute ≥ 0.8 × copy`; on the more generous reading, hiding
80 % of the whole step (`(serial − overlap) / serial ≥ 0.8`) needs `compute ≥ 4 × copy`.
For the GLM block at decode, compute is 1/458th of the copy, and its ceiling is
0.22 % under either formula. Prefill is the only phase with a meaningful ratio — 28 %
measured, 38 % attainable on the 70B layer — which is the opposite of what #156
targets: prefill is already amortized over many tokens, and decode is the phase where
the layer must be re-fetched for every single token.

## Result 5: what the arithmetic allows, per checkpoint and TP degree

`--stage budget` reads headers only and computes, for each checkpoint, how many whole
layers fit in a 22 GiB card, how many must be offloaded, and the resulting token-rate
ceiling. It charges whole layers against the GPU budget with no allowance for KV
cache, activations or compute buffers, so the resident count is an **upper** bound and
the offloaded count a lower bound — the real numbers are worse, not better.

| Checkpoint | TP | GiB/layer/rank | Resident | Offloaded | Transfer GiB/token | Serial tok/s | Perfect-overlap tok/s | Compute share of serial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM-5.2 Q4 (434 GiB, 79 L, 256 exp, top-8) | 1 | 5.47 | 3 | 76 | 415.75 | 0.02526 | 0.02532 | 0.23 % |
| | 2 | 2.74 | 7 | 72 | 196.93 | 0.05321 | 0.05346 | 0.45 % |
| | 4 | 1.37 | 14 | 65 | 88.89 | 0.1174 | 0.1184 | 0.90 % |
| Qwen3.8-Flash-Next (335 GiB, 48 L, 512 exp, top-10) | 1 | 4.83 | 4 | 44 | 212.46 | 0.04942 | 0.04955 | 0.26 % |
| | 2 | 2.41 | 9 | 39 | 94.16 | 0.1112 | 0.1118 | 0.51 % |
| | 4 | 1.21 | 18 | 30 | 36.21 | 0.2877 | 0.2907 | 1.02 % |
| Qwen3.8-27B-FP8 (28.75 GiB, 64 L, dense) | 1 | 0.35 | 61 | 3 | 1.06 | 9.555 | 9.890 | 3.38 % |
| | 2 | 0.18 | 64 | 0 | — | — | — | — |
| | 4 | 0.09 | 64 | 0 | — | — | — | — |
| Llama-70B FP4 (arithmetic model, 33.87 GiB, 80 L) | 1 | 0.42 | 51 | 29 | 12.28 | 0.8330 | 0.8575 | 2.85 % |
| | 2 | 0.21 | 80 | 0 | — | — | — | — |
| | 4 | 0.11 | 80 | 0 | — | — | — | — |

Producing command: the `--stage all` run above. The budget stage takes its bandwidth
and compute from the stages that ran before it, so run on its own it has neither —
`--stage budget` alone falls back to 10.61 GiB/s and 1.0 ms/layer and says so in the
artifact's `bandwidth_source` / `compute_source` fields, which is how a standalone run
is told apart from this table. `--bandwidth-gib-s` and `--compute-ms-per-layer`
override either.

Assumptions recorded in the artifact, and both of them measured rather than borrowed
from a document: bandwidth **10.53 GiB/s**, the best pinned point of the sweep in
Result 2, and compute **1.18 ms per layer per token**, the GLM-5.2 MoE block at one
row from Result 4 (plus 22 GiB per card). The `—` rows are not missing data: at those
TP degrees zero layers need to be offloaded, so there is no transfer to ceiling
against.

Two published figures come back out of the metadata, which is the check that the
header reading is not inventing structure: Qwen3.8-Flash-Next's routed-expert set is
225.0 GiB, exactly **56.25 GiB per rank at TP4** as
[Qwen4-Exp heterogeneous TP4 performance](qwen4_exp_performance.md) records, and
GLM-5.2-UD-Q4_K_M reads back as **79 layers, hidden 6144, 256 experts, top-8** with a
5.65 GiB median layer.

**The one input that is measured on the wrong thing is the compute term.** 1.18 ms
per layer is the GLM-5.2 MoE block at one row, and the budget charges it to every
checkpoint. For the 70B dense layer the same stage measured 3.58 ms instead, so
substituting its own number:

| Case | Compute charged | Compute measured | Serial tok/s (charged → measured) |
| --- | ---: | ---: | --- |
| GLM-5.2 Q4, TP4, 65 layers offloaded | 76.7 ms | 76.7 ms | 0.1174 → 0.1174 |
| Llama-70B FP4, TP1, 29 layers offloaded | 34.2 ms | 103.9 ms | 0.833 → 0.787 |

The GLM row is already using its own measurement. The 70B ceiling falls 5.5 % and its
compute share rises from 2.85 % to 8.2 %. Neither changes a conclusion below, and on
the numbers a real 70B implementation would have to quote the second column.

## Verdict on #156's four acceptance claims

| Claim | Verdict |
| --- | --- |
| 70B FP4 runs on a 2080 Ti | **Partly, and not for the stated reason.** 33.87 GiB at 4-bit weight-only fits in 22 GiB at TP2 (16.93 GiB/rank) and TP4 (8.47 GiB/rank) with *zero* offloaded layers — so #156's premise holds only at TP1. At TP1 the ceiling is 0.79–0.83 tok/s. **Not verifiable end to end on this host**: no 70B FP4 checkpoint exists here, so the entry is published Llama-70B geometry, not a measurement. |
| decode ≥ 5 tok/s | **Refuted by bandwidth, not by implementation quality.** 5 tok/s allows 200 ms/token, i.e. 2.11 GiB of offloaded traffic per token at the measured 10.53 GiB/s. One 70B FP4 layer is 0.42 GiB, so at most 4 of 80 could be offloaded — leaving 76 layers, 32.2 GiB, resident on a 22 GiB card. The roadmap's own split (16 layers resident, 64 offloaded, `:168`) moves 27.1 GiB/token, which is 2.57 s at the measured rate: **0.39 tok/s**, 13× short. The best offloaded ceiling measured on any checkpoint here is 0.29 tok/s (Flash-Next TP4) and 0.83 tok/s for the 70B FP4 model at TP1. |
| prefetch hides > 80 % of H2D | **Refuted for decode, unreachable in principle.** Maximum hideable is `compute / copy`; measured 1.18 ms of compute against a 540 ms copy for a GLM MoE block (0.09 % hidden, 0.22 % attainable), and 3.58 ms against a 39.8 ms copy for a 70B dense layer (6.67 %, 9.01 % attainable). 512-row prefill — the phase #156 does not target — is the best case at 28.0 % of 38.2 %. |
| token parity | **Not measured.** No layer-granularity offload path exists in this repository to hold parity, and this work deliberately added none. |

Everything that points the other way is worth stating too: the *prefill* half of the
roadmap's arithmetic does hold. At TP1 the offloaded 70B FP4 layers move 12.28 GiB per
512-row chunk, 1.17 s at the measured 10.53 GiB/s if the pipeline is perfect, which
is 439 tok/s and the right order for the roadmap's "~200 tok/s (H2D-bound)"
(`:169`). Prefill is transfer-bound and amortizes the transfer over the chunk; decode
is the same transfer with nothing to amortize it over.

## Why the ceiling is where it is, in one line

For a layer that must move `B` bytes and compute `t` seconds, decode throughput is
bounded by `1 / (B / 10.53 GiB/s)` no matter how good the overlap is, once
`B / 10.53 GiB/s > t`. Every offloaded configuration in the table above is in that
regime. The only way out is to move fewer bytes per token, which is precisely what
expert granularity already does — and that is a design this repository has already
shipped and measured, not one that is still available to win.

## Reproduction

```bash
# The full run behind this page (~84 s wall, GPU-side allocations of a few GiB)
python scripts/profile_cpu_offload.py --stage all --json /tmp/cpu_offload_profile.json

# Individual stages
python scripts/profile_cpu_offload.py --stage storage --storage-sample-mb 2048
python scripts/profile_cpu_offload.py --stage h2d
python scripts/profile_cpu_offload.py --stage overlap --overlap-rows 1 512
python scripts/profile_cpu_offload.py --stage budget
```

`--stage budget` looks for the checkpoints it knows about at fixed paths; pass
`--checkpoint <path>` (repeatable) to add one, `--no-synthetic-70b` to drop the
published-geometry 70B entry, and `--bandwidth-gib-s` / `--compute-ms-per-layer` to
replace the constants it derives from the other stages. No reference model was run,
so there is no parity comparison in this record and none is claimed.
