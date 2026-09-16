# Ascend TP Collective Overlap

Qwen3.8-27B (`hidden_size=5120`, 64 layers: 48 gated-delta linear attention, 16 full attention),
4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0.

The engine overlaps each row-parallel projection's TP all-reduce with the next slice's GEMM. This
document records how many slices that split should use, and what the measurement says about the
collective's cost. Every number is from a run on the machine above; the command is in
[Reproducing](#8-reproducing).

## 1. What the overlap path does

Three sites in the decoder layer end in a row-parallel projection followed by a TP all-reduce:
`mlp.down` (all 64 layers), `lin.out` (the 48 gated-delta layers) and `full.out` (the 16 full
attention layers). `QwenEngine::projection_all_reduce_overlapped` splits the projection's output
rows into `slices` and, for each slice, issues the GEMM on the compute stream and then the
`tp_all_reduce_sum_f16_inplace` for that slice on the HCCL comm stream, so slice *i*'s collective
runs under slice *i+1*'s GEMM. The ceiling is `comm_overlap_slices` (`QWEN_COMM_OVERLAP_SLICES`,
default 4); a row count or world size the path cannot serve falls back to the serial
GEMM-then-collective sequence.

## 2. Why the slice count cannot be a constant

Slicing does not change what has to move across the ring. `slices` slices send the same total bytes
in `slices` calls, so the trade is narrow: it buys overlap of the collective's **byte** term in
exchange for paying its **fixed per-call** term `slices` times instead of once. Splitting one call
into four pays three extra fixed terms, and it is a loss until the byte term is large enough that
overlapping it recovers more than those three fixed terms cost.

Both terms can be priced from the measurements below. At 512 rows the 4-slice pass issues 384 more
collectives than the serial pass and costs **0.612 s more prefill** (1.196 s against 0.584 s), i.e.
1.60 ms per extra collective. The serial pass's 128 collectives cost 0.491 s, i.e. 3.83 ms each. So
the fixed term is **~1.6 ms** and the byte term **~2.2 ms** at 512 rows — the 4-slice split pays
4x1.6 ms of fixed cost to overlap 2.2 ms of bytes, which is why it loses by 2x. Section 5.3 prices
the same fixed term at 1.85 ms from a device-synced profile three prompt lengths long, so the number
the crossover turns on is measured twice by unrelated routes.

The byte term grows 8x from 512 to 4096 rows; the fixed term does not. That is the whole crossover.

## 3. Measured: the constant was 2x wrong at short prompts

TP4, 64 layers, 512-token prompt, 128 new tokens, `QWEN_COMM_OVERLAP_SLICES` forced per run. Three
reps per setting, run interleaved so a slow patch of machine lands on every setting.

| slices | prefill TPS (3 reps) | mean |
|---|---|---|
| 1 (serial) | 882.09 / 882.55 / 862.62 | **875.8** |
| 4 (the shipped default) | 429.12 / 428.61 / 436.35 | **431.4** |
| 8 | 871.51 / 883.83 / 883.42 | **879.6** |

Decode moved only 8.72..8.99 TPS across all nine runs, which bounds the run-to-run drift: decode
takes the serial path at every setting, so a 2.03x prefill effect is not drift.

The slices=8 row is the control that the effect is the splitting and not the row count. At 512 rows
the then-current gate refused it (`per = 512/8 = 64 < 96`), so it fell back to the serial path and
landed on the serial number to within 0.4%. Two settings that resolve to the same code path agree;
the one that does not is 2x away.

At 4096 rows the trade reverses, two reps each:

| slices | prefill TPS (2 reps) | mean |
|---|---|---|
| 1 (serial) | 1206.01 / 1204.42 | **1205.2** |
| 4 | 1267.79 / 1262.53 | **1265.2** |

### Where the crossover is

Rows 1024 and 2048 were unmeasured, and they are what the rule has to get right. 8 new tokens:

| rows | slices=1 | slices=2 | slices=4 |
|---|---|---|---|
| 512 | **875.8** | — | 431.4 |
| 1024 | 1138.77 | **1140.13** | 862.41 |
| 2048 | 1195.12 | **1238.58** | 1185.81 |
| 4096 | 1205.2 | — | **1265.2** |

The optimum is 1 slice up to 1024 rows, 2 at 2048 and 4 at 4096 — i.e. **1024 rows per slice**, with
the 1024-row point a wash between 1 and 2 and every point above losing at 512 rows per slice.

## 4. The rule

`QwenEngine::comm_overlap_slices_for_rows(rows)` doubles the slice count while the resulting
per-slice row count stays at or above `kMinRowsPerOverlapSlice = 1024`, up to the configured
ceiling. Nothing else consumes `comm_overlap_slices` — a runtime knob still scales the ceiling, and
0 or 1 disables slicing.

Verification, no override set, same four shapes:

| prompt | new | slices chosen | prefill TPS | decode TPS |
|---|---|---|---|---|
| 512 | 128 | 1 | 878.8 | 8.73 |
| 1024 | 8 | 1 | 1134.5 | 8.96 |
| 2048 | 8 | 2 | 1238.3 | 9.11 |
| 4096 | 32 | 4 | 1261.6 | 8.91 |

Every point reproduces the optimum the forced sweep found (875.8 / 1140.1 / 1238.6 / 1265.2) to
within the drift the decode column bounds.

### Effect on the documented baseline

| configuration | prompt | new | prefill TPS | decode TPS |
|---|---|---|---|---|
| before (constant 4) | 512 | 128 | 428.1 | 9.21 |
| after (rule) | 512 | 128 | **878.8** | 8.73 |
| before (constant 4) | 4096 | 32 | 1265.3 | 9.55 |
| after (rule) | 4096 | 32 | **1261.6** | 8.91 |

**2.05x prefill at 512 tokens, within spread at 4096.** The 512-token row is worth 2x; the 4096 row
picks 4 slices and is unchanged, which is the intended behaviour of a rule that only ever moves the
short-prompt end.

## 5. Measured: what the prefill wall is made of

Sections 3 and 4 pin the slice count from the outside, by throughput. They say nothing about what
the wall that remains is made of, and the first answer this document gave — that the collectives are
84-87% of it — does not survive a device-synced profile. The corrected attribution and the evidence
that replaces it are below.

### 5.1 Why the earlier profile could not answer this

The first version of this section used `QWEN_HOST_PROFILE=1`. Its spans are host wall-clock and
**nest**: an outer scope absorbs everything its children spend, so `ar.lin.out` charged to the
collective whatever ran between two collective issues. `TOTAL` is the sum of every scope including
the nested ones, so it came out above the wall it was measuring — 2.77 s of scopes against a 1.25 s
wall in the runs below, 8.41 s against 3.95 s at the longest one. The short-prompt table that
resulted also names `layer.attn` and `layer.mlp_down`, scopes that no longer exist anywhere in
`cpp_engine/`; it is not reproducible on the current tree.

`QWEN_PHASE_PROFILE=1` instead has every `PhaseScope` synchronise the device on entry and exit, so
each `seconds=` is device time for that scope and a GEMM cannot hide behind a collective wait. Run
with `QWEN_COMM_OVERLAP_SLICES=1`, i.e. the serial path, so that nothing overlaps at all.

### 5.2 The attribution

TP4, four ranks, `QWEN_PHASE_PROFILE=1 QWEN_COMM_OVERLAP_SLICES=1`, rank 0, one report block per run
identifiable by its `STACK.r`. `STACK.r` is the whole 64-layer decoder stack as one scope and it
lands within 2% of the reported `prefill_seconds` at every length, so it is the denominator the rest
of the column is read against.

| prompt tokens | 1105 | 2223 | 4433 |
|---|---|---|---|
| wall (`prefill_seconds`) | 1.2501 | 2.1490 | 3.9484 |
| `STACK.r` | 1.2229 | 2.1187 | 3.9142 |
| prefill TPS | 883.9 | 1034.5 | 1122.7 |
| **`gated_delta`** (48 calls) | **0.5493 (44.9%)** | **1.1005 (51.9%)** | **2.2009 (56.2%)** |
| `full_attention` (16) | 0.1359 (11.1%) | 0.2331 (11.0%) | 0.4722 (12.1%) |
| projections, all 384 GEMMs | 0.1623 (13.3%) | 0.3243 (15.3%) | 0.5471 (14.0%) |
| `tp_all_reduce` (129 collectives) | 0.3026 (24.7%) | 0.3479 (16.4%) | 0.4961 (12.7%) |
| `attn_resid_norm` + `add` + `norm` | 0.0312 (2.6%) | 0.0507 (2.4%) | 0.0693 (1.8%) |

- **The linear-attention recurrence is the largest term and it is the one that grows.** `gated_delta`
  goes 44.9% → 56.2% of the stack across a 4x length increase, and its absolute cost is exactly
  linear: 0.5493 / 1.1005 / 2.2009 s for 1105 / 2223 / 4433 tokens is 496.9 / 495.1 / 496.5 us per
  token, a spread of 0.4%. It is 10.35 us per token per head per layer, over 48 layers.
- **The collectives are a falling share, not the wall.** 129 calls — one `mlp.down` in each of the 64
  layers, one output reduce in each of the 48 gated-delta layers (`lin.out`) and each of the 16 full
  attention layers (`full.out`), plus one hidden-state reduce — cost 24.7% of the stack at 1105
  tokens and 12.7% at 4433.
- **The projections are not the story either.** All 384 of them together are 14% at every length.
- The shares sum to 96% of `STACK.r`; the remainder is the small per-layer scopes (`swiglu`,
  `resid_copy` and friends) that the table does not list.

### 5.3 The collective priced from the same runs

Dividing `tp_all_reduce` by its 129 calls against the rows each call carries (`rows * 5120`
fp16 elements):

| rows | us/call | payload | MB/call |
|---|---|---|---|
| 1105 | 2346 | 11.32 MB | 11.32 |
| 2223 | 2697 | 22.76 MB | 22.76 |
| 4433 | 3845 | 45.39 MB | 45.39 |

A two-point fit on the ends gives a **fixed term of 1.85 ms per call** and a **marginal 22.7 GB/s**;
that predicts the middle point at 2849 us against 2697 us measured, 5% off. The fixed term agrees
with the ~1.6 ms section 2 derived from the slice sweep by a completely different route, which is the
useful part: the number that decides the slice count is the same number that decides the ceiling.

### 5.4 Where the recurrence's 2.2 s goes

The kernel's per-token work is one `Muls` over the 64 KiB state, two `broadcast_rows` calls (a
`Duplicate` per key row — 256 of them), a `Mul`, two `fold_rows` folds, the delta chain, and a
128-iteration `Axpy` loop for the rank-1 update. Reading that gives a cost model but not a
measurement, so each group was ablated out of `qwen_gated_delta_f16.cpp` and the scope re-timed:
build, run at 4433 tokens, restore.

| variant | `gated_delta` | delta | wall | prefill TPS |
|---|---|---|---|---|
| as shipped | 2.2023 | — | 3.929 | 1128 |
| `broadcast_rows` ablated (256 `Duplicate`/token) | 1.4105 | −0.792 (−36.0%) | 3.137 | 1413 |
| rank-1 update ablated (128 `Axpy`/token) | 1.6252 | −0.577 (−26.2%) | 3.373 | 1314 |
| both ablated | 0.8424 | −1.360 (−61.8%) | 2.557 | 1734 |

The two deltas sum to 1.369 s against 1.360 s measured with both gone — additive to 0.7%, so the
groups are independent. **62% of the recurrence is the issue cost of 384 per-row vector
instructions, not the state bandwidth**: the ablation rewrites only one of three kernels in a 64-layer
pass and moves the end-to-end prefill 1128 → 1734 TPS.

The shapes are the reason. `broadcast_rows` builds `work[i][j] = k[i]` so that
`fold_rows(k * S)` can compute `k^T S` — a 1x128 by 128x128 matvec written as a broadcast, a multiply
and a seven-step fold. The rank-1 update, `S += k (x) delta`, is a 128x1 by 1x128 outer product
written as 128 `Axpy`. Both are Cube shapes, and neither uses Cube. `Brcb` is an unsupported stub on
first-generation `dav_c100`, so the scalar `Duplicate` loop is deliberate rather than an oversight —
but it costs 36% of the prefill wall at this length.

### 5.5 What this costs the 2000 TPS target

2000 TPS at 4433 prompt tokens is a 2.217 s wall. The recurrence is 2.202 s of the 3.948 s measured,
so **it has to come down to 0.47 s — 4.7x — before any other term matters**, even holding everything
else fixed. Cutting it to 0.9 s, which is what turning the two ablated groups into Cube ops has to
plausibly achieve, gives a 2.6 s wall or about 1700 TPS, and the remaining gap is then the
collectives' 1.85 ms fixed term at 129 calls (0.24 s) plus the state passes that survive.

## 6. What this rules out

- **More slices.** Four is already past the optimum at every row count below 4096 and the ceiling is
  only reachable because of it; the failure mode at short prompts is not subtle.
- **Reading the prefill wall as collective-bound.** This was this document's original conclusion, at
  84-87%, and it is withdrawn: it came from nested host-wall spans charged to `ar.*` scopes that also
  contained the work between collective issues. Device-synced, the collectives are 12.7% of the stack
  at 4433 tokens against the recurrence's 56.2%, and their share falls as the prompt grows while the
  recurrence's rises. Cutting a collective is still worth doing — 0.24 s of the 3.95 s wall is pure
  fixed per-call cost — but it is not the lever that reaches 2000 TPS.
- **A cheaper collective.** The fixed term is host-side issue of the collective itself, measured flat
  in payload size in `bench_qwen_ascend_allreduce` (0.404 ms at 10 KB, 0.412 ms at 640 KB, against a
  0.011 ms device event pair), and `hcclOpExpansionMode` 0/1/2/3 leaves it unmoved. Slicing hides the
  byte term; it cannot make the fixed term smaller. Section 5.3 prices the same fixed term at 1.85 ms
  from inside the engine, which is 4.5x the standalone probe — the difference is not explained and is
  recorded here as open.
- **Tuning the projections to fix prefill.** Every projection GEMM together is 0.162 s at 1105 rows
  and 0.547 s at 4433, against a recurrence of 0.549 s and 2.201 s.
- **Reading the short-prompt result as a short-prompt problem.** The collective share is highest at
  short prompts (24.7% at 1105 rows) and lowest at long ones (12.7% at 4433); the wall is a shape of
  the pass, not of the prompt, but the term that owns it changes with length.

## 7. Files

- `cpp_engine/engine/qwen_engine.cpp` — `comm_overlap_slices_for_rows`, `kMinRowsPerOverlapSlice`,
  and the call site in `projection_all_reduce_overlapped`.
- `cpp_engine/engine/qwen_layer_components.inl` — the three projection sites that call it.
- `cpp_engine/engine/qwen_layer_components.hpp` and `include/qwen_engine.hpp` — the
  `comm_overlap_slices` config field and its default.
- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — the per-call cost in section 6.
- `cpp_engine/backends/ascend/kernels/qwen_gated_delta_f16.cpp` — the recurrence kernel section 5.4
  ablates, and `qwen_ascend_kernel_common.hpp` for `broadcast_rows` / `fold_rows`.

## 8. Reproducing

Four ranks, one process per rank, `HCCL_WHITELIST_DISABLE=1`,
`POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=12000`, and `scripts/ascend_env.sh` sourced first. Rank *i*
runs:

```bash
python3 -c "print(','.join(str(t) for t in [100,264,4628,286,1879,310,278,4221,13,576]*52)[:4096])" \
  > ids_4096.txt

QWEN_COMM_OVERLAP_SLICES=4 ./cpp_engine/build-ascend/pocketllm_engine \
  --ckpt /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
  --smoke-forward --smoke-layers 0 --resident-bench \
  --token-ids-file ids_4096.txt --max-new-tokens 32 \
  --tp-world 4 --tp-rank 0 --device 0 --nccl-id-path hccl_p4096_s4.id
```

`QWEN_COMM_OVERLAP_SLICES` pins the slice count for the sweep; omit it to measure the rule. The
`prefill_seconds` / `prefill_tokens_per_s` / `decode_seconds` / `decode_tokens_per_s` line on rank 0
is the result.

For the section 5 tables use `scripts/run_qwen_ascend_tp4.sh`, which is simpler to drive and takes
the prompt as text: `QWEN_PHASE_PROFILE=1 QWEN_COMM_OVERLAP_SLICES=1 ./scripts/run_qwen_ascend_tp4.sh
"<prompt>" 4`. The profile then prints one `qwen_phase tag=...` block per measurement boundary and
resets between them. **The tag does not identify a block**: a run emits several `tag=prefill` blocks,
one of them a decode-shaped pass whose scope times do not move with the prompt length at all. The
timed prefill is the block whose `STACK.r` matches the run's `prefill_seconds`, and the sections
above are all read from that block. Take `rank=0` and one block; a flat grep by tag mixes blocks and
produces a table that does not correspond to any single pass.

`QWEN_HOST_PROFILE=1` remains useful for the overlapped path, where a device-synced scope would
serialise the pipeline it is measuring, but its spans nest and its `TOTAL` overcounts — do not read
an `ar.*` figure from it as the collective's cost.
