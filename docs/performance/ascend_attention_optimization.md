# Ascend Attention Optimization

Qwen3.8-27B, 4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0.

This document records what was changed, what was measured, and what the measurements rule out.
Every number below comes from a run on the machine described above; the command that produced it is
named in the same section. Where a figure is an inference rather than a measurement, it is labelled
as one.

## 1. Why the attention kernel had to be rewritten

The original full-attention kernel computed every `Q * K^T` dot product on the **Vector** unit, as a
scalar loop over positions with an element-wise multiply plus a fold reduction inside. The Cube
(matrix) unit sat idle during both attention products.

For Qwen3.8-27B the ratio makes this indefensible:

- attention is **0.2% of the arithmetic** in a prefill (the feed-forward and projection GEMMs
  dominate the FLOP count), but it is not 0.2% of the time;
- 16 of the 64 layers are full attention; the other 48 are gated-delta linear attention.

Moving both products onto the Cube is therefore not a throughput play, it is the removal of a scalar
loop that has no business being on the vector unit at all.

## 2. The Cube (Mmad) kernel

`cpp_engine/backends/ascend/kernels/qwen_attention_cube_f16.cpp`.

Three passes per work item, with the vector unit left the softmax:

1. **`S = Q * K^T`** — Q and K staged GM -> L1 as ND, converted to NZ, one `Mmad` into L0C, read out
   to UB in fp16, scaled, written back to GM through `Nz2Nd`.
2. **Softmax per query row** — max, exp, sum and division on a row-major fp32 row. The row is written
   back as the already-normalised fp16 probability row, so pass 3 needs no post-division.
3. **`O += P * V`** — P and `V^T` staged, accumulated across chunks **in L0C itself**, then read out
   and stored.

One work item is a whole `(query head group, position tile)` pair, not a single head. K and V do not
depend on which query head asks for them, so a per-head item re-reads the entire K cache once per
head — a factor of `repeat` more traffic than the arithmetic requires, on a part where attention is
bandwidth-bound rather than Cube-bound. Stacking the head group as extra rows of the same `Mmad`
costs nothing and keeps the A operand one contiguous block.

The GM round trip through the score scratch is deliberate: L0C leaves the Cube in a zN fractal image,
and every row-wise reduction on that image is a gather across 16-element fractals. `Nz2Nd` only
writes to GM on this SoC. For `d = 256` the round trip is 2 bytes per score in and 2 bytes back out
against 64 MACs per score, so the arithmetic intensity absorbs it.

First-generation 910 specifics the kernel depends on, each of which cost a debugging round:

- no `MMAD` bias operand, so nothing asks for one;
- no fp32 `WholeReduceMax`, hence the vmax halving fold;
- L0C -> UB only through `DataCopy` under `BLOCK_MODE_MATRIX`, sized in KB of the **fp32 source**
  rather than of the fp16 destination.

Two entries exist because the call shapes genuinely differ: `causal != 0` for prefill (rows are
sequence positions with individual causal limits) and `causal == 0` for decode (a single position, so
the tile is the whole query head group and every row shares one limit).

## 3. Measured decode attention

Same machine, 4097-token context, `bench_qwen_ascend_attention`. Three kernels are dispatchable and
the winner is not the one with the most cores:

| kernel | time | relative |
|---|---|---|
| Cube, context-split | **257 us** | 1.0x |
| Cube, single core | 1225 us | 4.8x slower |
| FlashDecoding (vector, split) | 5224 us | 20.3x slower |

The split-Cube path wins because the single-core Cube kernel leaves 29 of 30 cores idle while
streaming the same K/V, and FlashDecoding drives its per-head partial and reduce state through
scalar `GetValue`/`SetValue` loops.

Dispatch (`cpp_engine/engine/qwen_layer_components.inl`): split Cube when the shape is expressible
and `context_length > 512`; single-core Cube below that, where there is nothing to split; the
FlashDecoding path is kept only for shapes the Cube path refuses. It is not a preference — it is a
fallback.

## 4. Measured prefill attribution

`QWEN_HOST_PROFILE=1`, 64 layers, 512-token prompt, `--resident-bench`. The profile emits four
blocks: `tag=prefill` (startup kernel warmup), `tag=warmup_decode` (a boundary
`QwenEngine::warmup_kernels` now emits so the warmup pass cannot be mistaken for the measured one),
`tag=prefill` again for the real prompt, and `tag=decode`. The real-prefill block reports `calls=64`
uniformly on the layer sub-scopes; without that boundary a contaminated run reported 128.

An earlier version of this section carried a three-block table and concluded that `STACK.r` at
17.2 ms/layer was "an unattributed black box". **That conclusion is superseded.** The layer was
instrumented, and the black box is not compute — it is the TP all-reduce:

| component | seconds | share of prefill wall |
|---|---|---|
| 129 TP all-reduce calls (2 per layer + 1 hidden) | 0.5072 | **87%** |
| everything else inside the 64-layer stack | 0.0577 | 10% |
| `top1_allreduce` (1 call) | 0.0111 | 2% |

`STACK.r` measures 0.548 s of the 0.584 s prefill wall, and 0.491 s of that is the collectives it
contains. Every other operation in the layer — norms, transposes, gated-delta, swiglu, and all the
projection GEMMs — sums to **0.90 ms per layer**. The full per-phase table, the pre-fix comparison
and the collective's measured fixed/byte cost split are in
[`ascend_tp_collective_overlap.md`](ascend_tp_collective_overlap.md).

Two conclusions this settles:

- **All projection GEMMs together are ~1% of a 512-token prefill.** The prefill gap is not in the
  matmuls, so further GEMM tuning cannot close it.
- **Prefill is collective-bound, and the fix is fewer calls.** The all-reduce is a measured ~1.6 ms
  of host-side issue plus a byte term, so the lever is the number of calls per layer, not their
  payload or their enqueue mechanism. `full_attention` remains 9.8% and is measured separately.

The same profile at a 4096-token prompt gives 84% for the collectives (84 vs 87 is inside the noise
of two different passes), so this is the shape of a 64-layer TP4 pass rather than a property of
short prompts. At 4096 rows `full_attention` grows to 9.3% and a single `top1_allreduce` call costs
0.243 s, 7.5% of that prefill — measured, and not yet explained. See
[`ascend_tp_collective_overlap.md`](ascend_tp_collective_overlap.md).

## 5. Measured throughput

All runs: four ranks, `--smoke-forward --resident-bench`, `tp-world 4`, 64 layers unless noted. `POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=12000`, `HCCL_WHITELIST_DISABLE=1`.

| configuration | prompt | new | prefill TPS | decode TPS |
|---|---|---|---|---|
| before startup kernel warmup | 512 | 128 | 234.3 | 9.38 |
| after startup kernel warmup | 512 | 128 | 428.1 | 9.21 |
| after startup kernel warmup | 4096 | 32 | 1265.3 | 9.55 |
| after the row-count slice rule | 512 | 128 | **878.8** | 8.73 |
| after the row-count slice rule | 1024 | 8 | 1134.5 | 8.96 |
| after the row-count slice rule | 2048 | 8 | 1238.3 | 9.11 |
| after the row-count slice rule | 4096 | 32 | 1261.6 | 8.91 |
| 64-layer reference, Cube split decode | 512 | 5 | 435.6 | 9.23 |
| 4096 prefill, prior to this branch | 4096 | 1-16 | 287-325 | 3.20 |

The slice rule is worth 2.05x at 512 tokens and is neutral at 4096; see
[`ascend_tp_collective_overlap.md`](ascend_tp_collective_overlap.md).

The startup-warmup fix (`QwenEngine::warmup_kernels`) brackets exactly one full forward pass with
8-row prefill and one decode step. It is worth 512/128: 2.185 s -> 1.196 s (**1.83x** prefill) and
4096/32: 965.9 -> 1265.3 TPS. Without it the first measured step pays the aclnn kernel-selection and
`aclrtMemcpy` warmup cost, which on this part lands outside the measured region. It is a fix to
*measurement validity* as much as to speed: before it, run-to-run prefill varied by 1.8x on identical
input.

### Decode scaling with layer count

| layers | decode TPS | ms/step |
|---|---|---|
| 1 | 105.7 | 9.5 |
| 4 | 67.2 | 14.9 |
| 8 | 58.8 | 17.0 |
| 16 | 33.5 | 29.9 |
| 32 | 18.1 | 55.4 |
| 64 | 9.24 | 108.2 |

A linear fit gives **~7.8 ms fixed per step + ~1.63 ms per layer**. That fixed term is why a 1-layer
run at 105 TPS says nothing about the 64-layer model, and it is worth stating explicitly because it
is easy to misread a short-layer smoke run as a solved decode target.

## 6. Measured hardware ceilings

`bench_qwen_ascend_gemm --scan` (single token row against a growing weight matrix):

| weight size | GB/s |
|---|---|
| 15 MB | ~356 |
| 60 MB | ~300 |
| 240 MB | ~290 |
| 960 MB | ~276 |
| 3840 MB | ~320 |

The rate is **flat over a 256x weight-size range**, so a single-row GEMV has no amortisable per-call
overhead: **~320 GB/s is the streaming ceiling** for this access pattern on this part, not a
launch-cost artefact. That the 15 MB case (fully L2-resident, 32 MB L2) is no faster than the 3840 MB
case is the strongest evidence — the limit is not HBM capacity traffic.

`bench_qwen_ascend_gemm --mem`, 512 MB D2D `aclrtMemcpy`: **8.7 GB/s**, i.e. **37x slower than the
GEMV**. Two consequences: any hot path using this copy is unaffordable, and this probe must not be
used to estimate HBM bandwidth.

`bench_qwen_ascend_allreduce`, HCCL f16 all-reduce, one process per rank:

| payload | time |
|---|---|
| 10 KB | 0.3923 ms |
| 640 KB | 0.3989 ms |
| 40 MB | 3.2064 ms |

Flat from 10 KB to 640 KB, against a device-op floor of 0.0183 ms. It is a pure **latency** wall, so
the fix is fewer collectives, not a faster collective. TP4 is the sweet spot (TP2 0.4554 ms, TP8
0.5156 ms at 10 KB).

### Decode is not at the bandwidth ceiling

Per-rank resident weights, from the engine's own startup line: `resident_weight_bytes=13449011456` =
**13.45 GB**. Every linear is read once per token, so at the measured 320 GB/s streaming rate the hard
floor is **42.0 ms/token = 23.8 TPS at TP4** (21.0 ms -> 47.6 TPS at TP8), before any collective cost.

Measured decode is **9.2 TPS** = 108.6 ms/token, i.e. 13.45 GB / 108.6 ms = **124 GB/s effective**,
about 40% of what the standalone GEMV achieves. So there are two separate deficits, and both must be
named:

- the per-layer cost is ~2.4x the pure weight-streaming cost, because `STACK.d` also carries the
  gated-delta matrix work (two 128x128 reductions and a rank-1 update currently done as 128-wide
  vector ops, with `broadcast_rows` the documented hot spot), the norms, and the transposes;
- 129 TP all-reduce calls per decode token (64 `ar.mlp` + 48 `ar.lin.out` + 16 `ar.full.out` +
  1 `ar.hidden_a`) at ~540 us each, roughly 60% of the step. A previous version of this document
  blamed the per-call `stream_synchronize` inside `end_nccl_collective` for ~70 ms of it; **that
  attribution is wrong.** In `bench_qwen_ascend_allreduce` the per-call figure is flat in message
  size (0.404 ms of pure host enqueue at 10 KB, 0.412 ms at 640 KB), and the probe line shows
  enqueueing an event pair costs 0.011 ms against 0.382 ms for one bare collective. The cost is
  host-side issue of the collective itself, paid once per call regardless of payload, so the fix is
  fewer collectives and not a cheaper fence around each one.

**100 TPS is not reachable on this part at fp16.** It needs <= 10 ms/token; the perfect-streaming
bound at TP8 is 21 ms with zero collective cost, which would already require 673 GB/s per card.
Reaching the target requires weight quantization — int8 for 2x, int4 for 4x — plus removing the
per-step collectives. Note also that 8 cards is the practical maximum here:
`HcclCommInitAll` is unusable on this stack and one process per rank is required.

## 7. What this rules out

- **More GEMM tuning for prefill.** Projections are ~1% of a 512-token prefill.
- **Blaming prefill on attention.** `full_attention` is 0.054 s of the 0.584 s wall at 512 tokens and
  is measured separately from the collectives; the 87% is the TP all-reduce, and pointing the next
  optimisation at the attention kernels would miss it.
- **A faster collective.** The HCCL floor is flat in payload size up to 640 KB.
- **`aclrtMemcpy`-based bandwidth work.** 8.7 GB/s.
- **Reading a short-layer smoke run as decode throughput.** See the layer scaling table.

## 8. Files

- `cpp_engine/backends/ascend/kernels/qwen_attention_cube_f16.cpp` — the Cube/Mmad attention kernel.
- `cpp_engine/backends/ascend/kernels/qwen_attention_f16.cpp` — the vector kernel, now the fallback;
  position tile 16 -> 64.
- `cpp_engine/backends/ascend/kernels/qwen_ascend_ops_launch.cpp` — Cube dispatch and geometry checks.
- `cpp_engine/engine/qwen_layer_components.inl` — the three-way decode dispatch.
- `cpp_engine/tests/bench_qwen_ascend_attention.cpp` — the kernel comparison in section 3.
- `cpp_engine/tests/bench_qwen_ascend_gemm.cpp` — `--scan` and `--mem`, section 6.
- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — section 6.
- `cpp_engine/tests/test_qwen_ascend_cube_attention.cpp` — numeric parity against the vector path.
