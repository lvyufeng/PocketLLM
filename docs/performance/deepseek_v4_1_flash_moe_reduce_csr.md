# DeepSeek-V4.1-Flash: the batched prefill MoE call's reduction, grouped per token

The [weight-staging page](deepseek_v4_1_flash_moe_weight_staging.md) took the batched expert call's
weight reads — `moe_multi_w1w3_fp4_kernel` and `moe_multi_w2_partial_fp4_kernel` — and left one row
of the same device table untouched: `moe_multi_reduce_partials_kernel`, the sum that turns a call's
per-(token, expert) partial products into each token's output. It does that sum by scanning **every**
pair for **every** row. A call is `tokens × pairs` compares — **11.3 G** at the call the profile
measures — and of the 2.205 M (token, pair) tests those compares make, exactly 2100 find a pair:
**0.095%**, or one in `tokens`. The page priced the alternative off the model at **11962.6 µs** for
the scan against **172.8 µs** for a per-token CSR of the same additions in the same order, 69.24×,
bit-identical, and recorded it as this page's work.

It ships here, with one correction that the second call site forced. The microbench built its CSR on
the host, in Python, out of routing it already had; the shipped grouping is built **on the device,
inside the launcher**, because one of the two live call sites resolves its routing with tensor ops
and never brings it to the host. The build is three ATen ops and costs **50.8 µs of device time a
call**; the kernel it feeds replaces **23183.4 µs** a call on rank 0 at the same position, and the
4096-token chunk the two sit in moves by the reduction's own 3.4–3.6 s and by nothing this host can
resolve past that.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | two changes, one page. The CSR reduce shipped as `perf/v41-moe-reduce-csr` on `perf/v41-moe-multi-coalesced-weights` (`0d5a4ec`, which is master at `a533a0a` plus #297), merged as PR #298 / `b394ddb`; the `bincount` substitution on that block shipped separately as `perf/moe-csr-device-counts` on master at `fbd139f`, and the sections from [The build's last host synchronization](#the-builds-last-host-synchronization) on are its record |
| The two arms | **one** binary, `DEEPSEEK_MOE_CSR_REDUCE=1` against `DEEPSEEK_MOE_CSR_REDUCE=0`. `env_int_default` reads the variable per call, so a single sitting can take both arms with the page cache, the pool's fill and the clock held constant, and no binary is swapped between them. Build `cuda_kernel.cpython-311-x86_64-linux-gnu.so`, 9494904 bytes, md5 `4f14b337b7c064b2db042b4dda283cd6`, byte-identical in `/tmp/pr_b/build/extensions/` where the run loads it, in this repository's `build/extensions/` and at the repository root (against #297's two-binary A/B, 9416528 / `37aba9d7c6e932d026b1dcab2f00dd92` before and 9490256 / `0830e3d7bb657c61fd3e683b7104a42a` after) |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `--pool-rows 148 --buffers 2`, `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, resident bank on |
| The third arm | [The build's last host synchronization](#the-builds-last-host-synchronization) is a second change to the same block, and it is a **third** switch on it: `DEEPSEEK_MOE_CSR_BINCOUNT=1` puts `at::bincount` back in place of the `searchsorted`. Build `cuda_kernel.cpython-311-x86_64-linux-gnu.so` 9503488 bytes, md5 `5361aac05f6a77516157326624d5ede3` in this repository's `build/extensions/` and `09949222e6a448e629db8de491b538a0` in `/tmp/prefill_csr/build/extensions/`, where the model-level sessions load it — same source and same size, different md5 because the build path is compiled into the object |
| Workload | `--at 32768 --chunk 4096 --max-seq-len 53248`: prefill to 32768 eight chunks at a time, then profile one 4096-token chunk at 32768 |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, cross-pairs SYS |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env, `CUDA_HOME=/usr/local/cuda-12.4`, `TORCH_CUDA_ARCH_LIST=7.5`, `POCKETLLM_BUILD_CPP=0` |
| Probes | `/tmp/probe_csr_ab.py` (four profiled chunks, walking the position forward with the arms) and `/tmp/probe_csr_ab_fixed.py` (every arm rewound to **one** position); `/tmp/csr_build_cost.py` and `/tmp/csr_build_ops.py` price the build off the model |
| Microbench | `/tmp/run_moe_multi_bench.py` driving `/tmp/moe_multi_bench.cu` at the measured per-call geometry |
| Parity | `/tmp/pr_b_parity.py` at `--at 32768 --chunk 4096 --pool-rows 148`, both arms under the same configuration as the probes, nine positions a rank; `/tmp/bench_issue_kernel.py` carries the three-way parity on random routing for the `bincount` substitution |
| The chunk, for the substitution | `/tmp/probe_v41_csr_abab.py --at 32768 --chunk 4096` on a mirrored two-arm period, 16 chunks an arm, two sittings (`/tmp/csr_abab.log`, `/tmp/csr_abab_taps.log`), against the taps-off comparator `/tmp/chunk_profile_nobarrier.log` at the same geometry |

## What a scan costs, in the shape the routing has

`moe_multi_reduce_partials_kernel` is one block to a `(token, column tile)` pair: `blockIdx.y` is the
row, `blockIdx.x · blockDim.x + threadIdx.x` the column, and each thread walks the **whole** pair
list looking for the entries whose `slot_tokens` field names its own row. So the loop's trip count is
`pairs` whatever the routing is, and the call's work is `tokens × pairs` tests over `dim` columns.

The profile's call measures 1050 rows, 61 slots and 2100 pairs — 2 routes a row on this rank, 34.4
pairs a slot, `dim` 5120 — so:

- **11.29 G compares a call**: 1050 rows × 2100 pairs × 5120 columns.
- **2100 of the 2.205 M tests find a pair**, because a pair names exactly one token and there are
  `pairs` of them. That is **0.095%** — 1 in `tokens`, not 1 in `pairs`, because the same 2100
  tests are made by all 1050 rows between them. Each thread's 2100 iterations contain the 2 pairs
  that name its row, and nothing else does any work.
- **The ratio is the routing's, not the shape's.** A row has as many pairs as it has routes to this
  rank's experts — 2 on ranks 0 and 1, 1 on ranks 2 and 3, given the 2/2/1/1 deal this page's
  sittings run (`sorted`, one flag from the default since 2026-09-20) — and it does not
  move when `dim` or `inter_dim` changes. Only `tokens` does, and `tokens` is what a chunk feeds
  the call.

That is why the fix is not a wider tile or a better load: there is no data movement to improve, only
a loop whose iteration count should never have been `pairs`. Grouping the pairs by token and walking
each row's own is the same sum with the trip count it wanted.

The scan is not deleted. `kCsrMinWork = 50000` is the product `tokens × pairs` where the two cross,
and the speculative-verify call sits far below it — 8 to 48 pairs over at most 6 rows is 288 of the
50000 — where a flat build against a scan over 48 pairs would be a regression. Both paths are in the
profile at once: in the CSR arm at 32768, rank 0 shows `moe_multi_reduce_csr_kernel` on 154 of the
156 calls and `moe_multi_reduce_partials_kernel` on the other **2**, at 34.2 µs a call.

## The grouping, and where it is built

The pair list arrives as `slot_tokens[pair] → token`, tagged by slot rather than by row. Grouping it
by row, in ascending pair order within each group, is four lines in the launcher:

```cpp
auto sorted = at::sort(slot_tokens, /*stable=*/true, /*dim=*/-1, /*descending=*/false);
auto row_pair = std::get<1>(sorted).to(torch::kInt32);
auto row_ptr = at::zeros({tokens + 1}, slot_tokens.options());
row_ptr.narrow(0, 0, tokens).copy_(at::searchsorted(
    std::get<0>(sorted), at::arange(static_cast<int64_t>(tokens), slot_tokens.options()),
    /*out_int32=*/true, /*right=*/false));
row_ptr.narrow(0, tokens, 1).fill_(pairs);
```

`row_pair` is the pairs in token order and `row_ptr` that order's exclusive prefix sum, so
`moe_multi_reduce_csr_kernel` walks `[row_ptr[token], row_ptr[token + 1])` — the row's own pairs, no
test at all. `searchsorted`'s leftmost insertion index of a token into the sorted tokens *is* that
token's group start, i.e. the counts' exclusive prefix sum; the `fill_(pairs)` closes the last group,
because `pairs` pairs cover `tokens` tokens and every one of them names a token.

The four lines were five in the first version, and the op that left was `at::bincount`:

```cpp
auto counts = at::bincount(slot_tokens.to(torch::kLong), {}, tokens);
row_ptr.narrow(0, 1, tokens).copy_(counts.cumsum(0, torch::kInt32));
```

It is an ATen tensor op and it is written as one, but on CUDA it is **not a device op**, and in a
launcher that both call sites reach with `tokens` and `pairs` already resolved it was the only thing
in the block that stopped the host. [The section below](#the-builds-last-host-synchronization) prices
it; `DEEPSEEK_MOE_CSR_BINCOUNT=1` puts it back in process.

**The stable sort is the correctness argument, not an implementation detail.** `slot_tokens` is
already in ascending pair order, and a stable sort keeps each token's pairs in the order they
arrived, which is ascending pair index — the exact sequence the scan adds them in. A row's
accumulator therefore sees the same float additions in the same order, and the two kernels are
**bit-identical** rather than nearly so. An unstable sort would reorder a row's pairs among
themselves and the outputs would differ in the last bits, which is a different claim and a worse one.

**Built on the device because one of the call sites is there.** `moe_multi_token_fp4_forward` has two
live callers and they resolve their routing in opposite ways:

- `src/models/deepseek_v4_1/device_experts.py:1471`, inside `_issue_chunk`, passes `tokens` and
  `slots` as **Python lists** it has just built — the routing is on the host, and a host-side CSR
  would be free there.
- `src/components/moe/gpu_prefill_backend.py:636` builds the same arguments entirely from **tensor
  ops** — `torch.argsort(pair_experts, stable=True)`, `torch.unique_consecutive`, `cumsum` — and
  never calls `.cpu()`. It is reached from the DeepSeek-V4 (non-4.1) runtime, where
  `src/models/deepseek_v4/runtime.py:3524` constructs the backend and the prefill calls it at
  3009, 3678, 3955 and 3972. Reading that routing back to build a CSR on the host would be a device
  synchronisation on the one path that exists to avoid one.

The microbench is the correction. `/tmp/run_moe_multi_bench.py`'s `reduce_bench` assembles `row_pair`
by iterating **slots** and sorting per row, under the comment *"The CSR the host could hand down for
free: `_route_ids` already has the routing"* — true of the first call site and false of the second.
A launcher that took the first caller's convenience would have silently added a D2H to the second.

**It adds no ATen call.** Measured over the same chunk, `aten::narrow` runs 62,116 times in the CSR
arm against **62,650** in the scan arm: the path borrows an existing call rather than adding one,
because `slot_tokens` is already `int32` and `.to()` is a no-op there, and `row_ptr`'s allocation is
one `aten::zeros` the chunk already makes for `y`. The kernels the CSR arm adds over the scan arm are
exactly the stable sort and the two ~1 KB elementwise ops whose outputs are `tokens`-sized —
`searchsorted`'s indices and the `fill_`, into the `bincount`'s and `cumsum`'s places.

## The build's price

`/tmp/csr_build_cost.py` runs the four ops in a loop on an idle card, 200 iterations an arm, at the
two geometries the call sees — and at two that are deliberately tiny, to show the cost is flat:

| tokens | pairs | build |
| --- | --- | --- |
| 1050 | 2100 | 294.53 µs |
| 6 | 48 | 293.60 µs |
| 1 | 8 | 291.52 µs |
| 128 | 256 | 318.96 µs |

291.5–319.0 µs across a 500× range of work, so the four ops are launch-bound and the tail is all
launch. `/tmp/csr_build_ops.py` splits them, at 2100 pairs and at 48:

| op | 2100 pairs | 48 pairs |
| --- | --- | --- |
| `torch.sort(..., stable=True)` | 47.46 µs | 45.30 µs |
| `torch.sort(...)` (unstable) | 46.99 µs | 44.44 µs |
| `argsort` | 43.79 µs | 42.16 µs |
| `.long()` | 11.62 µs | 10.93 µs |
| `bincount` | 84.79 µs | 84.16 µs |
| `zeros(tokens + 1)` | 12.31 µs | 11.79 µs |
| `empty(tokens + 1)` | 3.28 µs | 3.31 µs |
| `cumsum` | 13.87 µs | 13.78 µs |

Two things follow. **`sort` and `bincount` are the build**, and both are launch-bound (the same 84 µs
at 48 pairs as at 2100). And **294 µs is not what the chunk pays**, because a standalone sweep prices
a launch-bound sequence with nothing to overlap it against. The chunk has 20 ms of kernels a layer to
hide it behind. From the profiler, on rank 0 over the same 4096-token chunk:

| | 32768, CSR=1 | 45056, CSR=1 |
| --- | --- | --- |
| `aten::sort` | 164 calls, **44.61 µs** device / 43.74 µs host | 173 calls, 42.62 / 38.08 µs |
| `aten::bincount` | 154 calls, **3.04** / 37.77 µs | 163 calls, 2.93 / 31.90 µs |
| `aten::cumsum` | 154 calls, **3.14** / 37.49 µs | 163 calls, 3.05 / 34.11 µs |
| `aten::narrow` | 157 calls, 0.00 / 5.53 µs | 166 calls, 0.00 / 4.78 µs |
| **build, device** | **50.8 µs a call** | **48.6 µs a call** |
| **build, host** | **124.6 µs a call** | **118.9 µs a call** |

**50.8 µs a call, 7.9 ms over the chunk's 156 calls**, against the **3.2–3.6 s the scan spends at the
same position on the four ranks** — 0.22% of what it saves, and 0.016% of a 47 s chunk. The 294 µs the
standalone sweep reports is that same work with nothing behind it to overlap, the identical direction
as the weight-staging page's 3.89× off the model against 3.06× in it. In the scan arm the same table
has **10** `aten::sort` calls at ~309 µs — the harness's own, not the launcher's — and no `bincount` or
`cumsum` at all, which is what makes the five rows above attributable to this change rather than to
the chunk.

The profiler's host column is what a *profiler* sees, and for one row above that is not the whole
cost. `aten::bincount`'s 37.77 µs is ATen's own CPU time for the op; the op's blocking device-to-host
reads are not a kernel and the profiler's host column stops at the launch. The next section is about
that difference, and it is the reason the launcher no longer calls `at::bincount`.

## The build's last host synchronization

Every op in the block above is an ATen tensor op, and the first version of it read as though that were
the same as being a *device* op. `at::bincount` is not one. In
`aten/src/ATen/native/cuda/SummaryOps.cu`, `_bincount_cuda` computes

```cpp
const int64_t nbins = std::max(self.max().item<input_t>() + (int64_t)1, minlength);
```

and, ahead of that, bounds-checks `*self.min().cpu().const_data_ptr<input_t>() < 0`. `item()` and
`.cpu()` are both blocking device-to-host reads, so **the op drains the stream twice** — 80 times over
a chunk's 40 calls — for an answer the stable sort on the line above already holds. That is the one
thing this block's comment was written to say it does not do ("reading it back to build the CSR in
Python would be the D2H sync this path exists to avoid"), and it was doing it in C++ instead of
Python.

`searchsorted` gives the same boundaries with no host read: the leftmost insertion index of a token
into the sorted tokens *is* that token's group start, so it is the counts' exclusive prefix sum.
`row_ptr[tokens]` cannot be read off that result — it would come back `0` — and is filled with
`pairs`, which is exact rather than a clamp: `pairs` pairs cover `tokens` tokens and every one of them
names a token, so the counts sum to `pairs`.

### What the substitution is worth

`/tmp/bench_issue_kernel.py` drives `moe_multi_token_fp4_forward` at the call's own geometry — 4096
tokens, `dim` 5120, `inter_dim` 2304, 6144 pairs over 100 rows of a 148-row arena — with nothing else
on the card, and measures every arm twice: with an empty queue, and with a queue of known width laid
in front of the call (4 fp32 `4096²` matmuls on the same stream, ~46 ms). A sync inside the call has
to absorb that queue, so **an arm whose host time grows by the stuffer's width is an arm with a sync
in it**. `--rounds 30`, seconds are ms a call:

| arm | host | wall | host, stuffed | wall, stuffed |
| --- | --- | --- | --- | --- |
| shipped (det=1, csr=1) | **0.32** | 42.82 | **0.34** | 89.25 |
| as `bincount` (csr=1) | **43.31** | 43.72 | **89.26** | 89.85 |
| scan reduce (csr=0) | 0.16 | 177.70 | 0.13 | 228.84 |
| atomic (det=0) | 0.09 | 42.83 | 0.08 | 90.71 |
| shipped again | 0.31 | 43.66 | 0.32 | 91.60 |

The host column is the whole experiment. **42.4 ms → 0.32 ms a call, ~132×, and the shipped arm does
not move when the stuffer is laid in front of it** (0.32 → 0.34), where the `bincount` arm grows by
the stuffer's own width (43.31 → 89.26 against a ~46 ms stuffer; its stuffed host time lands on its
stuffed wall, which is a stream drained to its end). The `wall` columns say the device side of the
substitution is free: 42.82 against the atomic path's 42.83 and the construction it replaces at
43.72. The A B C D A order puts the machine's drift on the repeat, which reads 0.31 against the
opening 0.32. The `scan` arm is the table's outlier at 177.70 because it carries the scan reduce; that
is the price the [first section](#what-a-scan-costs-in-the-shape-the-routing-has) already put at
69.24×, not a term in this comparison, and it is why the scan is not in the model-level rotation
below.

The mechanism is also checked the other way, on the kernel's own output: on **random** routing it
prints `bit-identical` for both the `bincount` construction and the `scan` against the new one, so the
three are the same float additions in the same order and the substitution changed no value.

### The chunk

1.70 s over a 4096-token chunk's 40 calls is below what four separate processes resolve — the
taps-off comparator for this geometry is 26.22–26.59 s with a 2.1–2.3 s spread across warm-up chunks
(`/tmp/chunk_profile_nobarrier.log`). So the arms alternate **inside one load**, on the machine's own
drift, which is what a lever this size needs:

```
DEEPSEEK_V41_RESIDENT_EXPERTS=1 V41_TREE=<tree> torchrun --nproc_per_node=4 \
    /tmp/probe_v41_csr_abab.py --at 32768 --chunk 4096
```

`/tmp/probe_v41_csr_abab.py` binds `_issue_chunk` and `forward` with `perf_counter` and nothing else —
**no `synchronize` in the instrument**, which is the very thing the arms differ in — and walks a
**mirrored period**: the arms, then the arms reversed, so that each arm's mean position inside a
period is identical (a,b,b,a for two arms; a,b,c,c,b,a for three). The context grows across the run,
so a chunk late in the run is slower than one early in it; a plain a,b,a,b rotation leaves each arm's
mean position off by half a chunk and books that trend as an arm difference, where the mirror cancels
it exactly. Sixteen chunks an arm, rank 0, two independent sittings:

| sitting | arm | wall, s a chunk (four positions) | mean | `_issue_chunk` | `forward` |
| --- | --- | --- | --- | --- | --- |
| 1 | `device` | 26.75 27.02 27.56 28.52 (×4) | **27.46** | **0.74 s** | 17.64 s |
| 1 | as `bincount` | 27.03 26.98 28.41 28.49 (×4) | 27.73 | **4.61 s** | 17.52 s |
| 2 | `device` | 26.91 26.78 27.65 27.71 (×4) | **27.26** | **0.76 s** | 17.69 s |
| 2 | as `bincount` | 26.80 26.67 27.68 27.97 (×4) | 27.28 | **4.66 s** | 17.62 s |

**The `_issue_chunk` column is the mechanism; the wall column is not the price.** 0.74 s against
4.61 s over 40 calls is 3.87 s a chunk of host stall removed, and it reproduces (0.76 against 4.66).
The wall moves by **+0.27 s of 26.75 in the first sitting and +0.02 s in the second** — against the
1.70 s the bench predicts — and both walls are position-dominated: each arm repeats its four position
values and climbs ~1.5 s across the run, which is exactly what the mirror cancels, so ±0.15 s is this
instrument's floor. Two sittings of the same design landing 0.25 s apart is the effect being zero at
this geometry.

The second sitting also taps the sub-phases of `forward`, to ask whether the seconds *moved* rather
than left (rank 0, seconds a chunk, mean of 16):

| arm | `_route_ids` | `_upload` | `_issue_chunk` | the three | `forward` |
| --- | --- | --- | --- | --- | --- |
| `device` | 4.68 | 0.60 | 0.76 | 6.03 | 17.69 |
| as `bincount` | 4.69 | 0.59 | 4.66 | 9.93 | 17.62 |

The three rows account for 3.90 s of the difference, and `_route_ids` — the launcher's other sync, and
the largest single host wait in the chunk at 4.68 s — is **identical across the arms** (4.69), so the
`bincount` sync was not pre-paying that wait. `forward`'s own total is 0.07 s *lower* on the arm with
the stall in it, so nothing inside the MoE absorbed it either: the 3.9 s appears in one phase, does
not appear as a compensating row anywhere, and does not appear at the wall. Read against the
no-barrier profile of the same geometry — 95.1% of the chunk inside the host's calls, 1.28 s
device-only, per rank — the only reading that survives its own arithmetic is that at TP4 the chunk's
wall is set by the collective path's convergence and not by one rank's serial host chain, so a
rank-local stall of 4 s in a 27 s chunk can sit in slack the collectives already have. The honest
statement of the change is therefore the narrow one: **it removes two stream drains from the batched
call's default path, measured in the call and in the phase, and it does not buy wall time at
32768/4096** (it removes a rank-*asymmetric* host stall, which is worth more at a geometry where the
ranks' host chains are what converge). What it is worth is what a synchronization is worth
structurally: it is one fewer capture-hostile site in the launcher, and a drained stream is a
prerequisite for any overlap that would put the fp4 GEMM behind the host's next rank of work.

## The in-model A/B

`/tmp/probe_csr_ab.py` walks four arms — `("1", "0", "0", "1")` — and profiles a 4096-token chunk in
each. It puts chunk *i* at `--at + i · --chunk`, so **the arm and the position move together** and the
four walls cannot be read as an arm effect:

| arm | position | chunk wall | `moe_multi_reduce_*` |
| --- | --- | --- | --- |
| CSR=1 | 32768 | 53.56 s | csr 154 calls, 138.4 µs a call |
| CSR=0 | 36864 | 73.04 s | scan 156 calls, 23410.1 µs a call |
| CSR=0 | 40960 | 72.58 s | scan 136 calls, 27963.8 µs a call |
| CSR=1 | 45056 | 62.36 s | csr 163 calls, 128.5 µs a call |

Its reduce rows are usable anyway — the two kernels differ by 169–239× at every position on every
rank — but its walls are not, so the section is measured by a second probe.

`/tmp/probe_csr_ab_fixed.py` is the one that can hold an arm still: every arm rewinds the state
(`front.reset_state(1)`) and re-prefills to the same `--at`, profiles **one** chunk there, and the
arms interleave `("1", "0", "1", "0")`, so both settings are sampled twice at each of the eight
chunks and every arm's eight-chunk prefill is a second, unprofiled measurement of identical work.
Rank 0, all four arms at 32768:

| arm | CSR | 8-chunk prefill mean | fastest chunk | profiled chunk | reduce, device |
| --- | --- | --- | --- | --- | --- |
| a | 1 | 44.36 s | 42.16 | 53.17 s | csr 154 calls, **0.021 s** |
| b | 0 | 49.78 s | 43.39 | 75.37 s | scan 156 calls, **3.617 s** |
| c | 1 | 46.23 | 43.75 | 78.17 s | csr 154 calls, **0.021 s** |
| d | 0 | 50.98 | 43.11 | 86.29 s | scan 156 calls, **3.599 s** |

**The profiled chunk is still unusable, and the position is now provably not the reason.** Arms a and
c are the same setting at the same position, and their walls are 25.0 s apart — more than either
setting is from the other. What moved under them is the collective: `record_param_comms`, the same
1152 calls of the same bytes in all four arms, in seconds:

| rank | a: CSR=1 | b: CSR=0 | c: CSR=1 | d: CSR=0 |
| --- | --- | --- | --- | --- |
| 0 | **4.256** | 25.959 | **22.106** | 25.872 |
| 1 | **4.606** | 13.491 | **33.371** | 23.689 |
| 2 | 13.751 | 24.976 | **33.207** | 47.511 |
| 3 | 14.350 | 37.109 | 32.593 | 37.407 |

Rank 0's two CSR=1 arms are 4.256 and 22.106 s of collective under walls 25.0 s apart, so the wall's
whole spread is the collective's, and no rank is stable — rank 1 reads 4.606 / 13.491 / 33.371 /
23.689 for the same 1152 calls. **No wall ratio is quoted from the profiled chunks.**

The eight-chunk prefill mean is a different measurement — outside the profiler, over eight identical
chunks, which is what averages a per-chunk spread that reaches 10 s on its own. Rank 0, chunk by
chunk:

| chunk | a: CSR=1 | b: CSR=0 | c: CSR=1 | d: CSR=0 |
| --- | --- | --- | --- | --- |
| 0 | 54.14 | 55.75 | 44.49 | 68.87 |
| 1 | 42.38 | 43.62 | 53.22 | 43.40 |
| 2 | 43.31 | 56.90 | 45.99 | 57.28 |
| 3 | 42.16 | 47.33 | 46.25 | 45.89 |
| 4 | 43.59 | 43.67 | 44.62 | 43.40 |
| 5 | 43.39 | 55.96 | 44.21 | 59.74 |
| 6 | 42.91 | 43.39 | 47.29 | 43.11 |
| 7 | 42.99 | 51.61 | 43.75 | 46.14 |
| **mean** | **44.36** | **49.78** | **46.23** | **50.98** |

Chunk 4 is 43.4–44.6 s in all four arms and chunk 2 is 43.3–57.3 s; the per-chunk scatter is larger
than the difference between the settings, which is exactly what the mean over eight is for. Per rank,
its two CSR arms against its two scan arms:

| rank | CSR=1 arms | CSR=0 arms | difference |
| --- | --- | --- | --- |
| 0 | 45.29 s | 50.38 s | **+5.09 s (+11.2%)** |
| 1 | 46.97 | 49.60 | **+2.63 s (+5.6%)** |
| 2 | 48.00 | 52.89 | **+4.89 s (+10.2%)** |
| 3 | 47.26 | 52.95 | **+5.69 s (+12.0%)** |

Four ranks of four favour the CSR setting, +4.58 s a chunk on the pooled mean (46.88 against 51.46 s,
9.8%). **And the same arms' device rows say that all of it, and nothing beyond it, is the kernel.**
The reduce is 3.60, 3.62, 3.32 and 3.16 s a chunk on ranks 0 to 3 against 0.021, 0.021, 0.016 and
0.016 — 3.42 s a chunk removed — and the four chunk-level differences scatter between **0.73× and
1.80×** of their rank's own kernel time. The scan's 42,000-block grid does run concurrently with the
all-reduce kernels, and the scan arm is the expensive-collective arm in three of the four pairs, but
that correlation does not survive the two arms that share a setting, so no collective bonus is
claimed. What the arms say is that the reduction's 3.4–3.6 s a chunk leaves the chunk, and that this
host resolves nothing past that.

So the measured effect is the reduction itself: **3.6 s of the 51.46 s chunk it comes out of, 7.0% of
it**, with `aten::copy_` 17.1 s, `Memcpy HtoD (Pinned -> Device)` 15.9 s, the sparse attention 8.1 s
and the collectives above untouched by this change and irreducible to it.

## Correctness

- **A test that reaches the new kernel.** `tests/test_moe_kernel_determinism.py`'s
  `test_csr_reduce_matches_the_scan_it_replaced` runs the same batched call twice, once with
  `DEEPSEEK_MOE_CSR_REDUCE=1` and once with `=0`, at `tokens = 160` and 480 pairs — `76800` of
  `kCsrMinWork`'s `50000`, so the CSR path is the one under test — and asserts `torch.equal`, not a
  tolerance, with the message naming the way the test goes stale if the geometry moves. It is also
  the only test in the suite that reaches the CSR kernel: every other MoE test's `tokens × pairs` is
  below the threshold. `/tmp/csr_engaged.py` is the check on that claim, printing the reduce kernels
  the profiler saw under each arm.
- **The two paths are bit-identical in the test, and the test is why the page quotes a reduction
  rather than a speedup**: at one position the two arms' reduce totals differ by **171–209×**
  (3.60/3.62/3.32/3.16 s against 0.021/0.021/0.016/0.016 s on ranks 0 to 3) and by nothing at all in
  the numbers they produce.
- **Bit-identity in the model, four ranks over nine positions.** The reduce's contribution is
  `sum(partials)`, so a CSR that adds the same values in the same order contributes the same float —
  the scan's order is the pair index's, and the stable sort is what preserves it.
  `/tmp/pr_b_parity.py` is the end-to-end check: it prefills to 32768 the way the sweep does and
  keeps the last-position logits row after **every** chunk, so its two arms — the same binary with
  `DEEPSEEK_MOE_CSR_REDUCE` set to `1` and to `0` — are compared position by position. Its chunks are
  4096 tokens over ~8192 pairs, three orders of magnitude above `kCsrMinWork`, so the arm being
  compared against the scan is the CSR kernel and not the fallback. The result is
  `max |diff| = 0.000e+00` at all nine positions on all four ranks — nine rows of 129280 logits,
  1.16 M floats a rank — with argmax
  agreeing at each: 223, 5198, 223, 223, 13822, 17227, 13918, 18014, 455. There is no tolerance in
  that number and no rounding to attribute to the reordering, because there is no reordering.
- **The scan is still the default wherever it should be.** `DEEPSEEK_MOE_CSR_REDUCE=0` reaches the
  old kernel from the same binary, and the sub-threshold path is live in the CSR arm itself — the 2
  calls at 34.2 µs a call in the 32768 arm, the 6 at 96.9 µs in the 45056 arm — rather than dead
  code kept for symmetry.

## What this leaves

The reduction is no longer a term in the chunk: at the prefill geometry it was 3.4–3.6 s of a 4096-token
chunk, and the chunk the A/B measured it in moves by that much in the same direction on all four ranks.
What is left of that chunk is the host's copy path and the collectives — `aten::copy_` 17.1 s,
`Memcpy HtoD (Pinned -> Device)` 15.9 s, `nccl:all_reduce` moving between 4.3 and 25.9 s of it across
arms that differ in neither arm nor position — and the sparse attention's 8.1 s, none of which this
change touches. The weight reads the
[weight-staging page](deepseek_v4_1_flash_moe_weight_staging.md) coalesced are the other half of
the batched call, and its transposed-arena lever (5.88× off the model) is still the one worth a
one-time rewrite of the bank.
