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

`QWEN_HOST_PROFILE=1`, 64 layers, 512-token prompt, `--resident-bench`. The profile emits three
blocks (kernel warmup, the real prefill, decode) and the per-phase cost is the difference between
block 0 and block 1. Prefill wall for the differenced pair: **1.6742 s**.

| phase | seconds | calls | us/call | share |
|---|---|---|---|---|
| `STACK.r` (whole decoder layer) | 1.1031 | 64 | 17236 | **65.9%** |
| `full_attention` | 0.1646 | 32 | 5144 | 9.8% |
| `STACK.d` | 0.0967 | 64 | 1512 | 5.8% |
| `top1_allreduce` | 0.0814 | 2 | 40688 | 4.9% |
| `tp_all_reduce` | 0.0779 | 130 | 599 | 4.6% |
| every `pr.*` / `pd.*` projection | ~0.050 | — | — | ~3% |

Two conclusions this settles:

- **All projection GEMMs together are ~3% of prefill.** The prefill gap is not in the matmuls, so
  further GEMM tuning cannot close it.
- **`STACK.r` at 17.2 ms/layer is 66% of prefill and is still an unattributed black box.** It wraps
  `decoder_layer_component.forward(...)` as a single scope, so it absorbs the layer's norms, transposes,
  gated-delta work and elementwise passes without naming any of them. Instrumenting inside it is the
  next measurement, and it is a prerequisite for any further prefill work — attributing this to
  "attention" would be wrong, since `full_attention` is measured separately and is 9.8%.

## 5. Measured throughput

All runs: four ranks, `--smoke-forward --resident-bench`, `tp-world 4`, 64 layers unless noted. `POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=12000`, `HCCL_WHITELIST_DISABLE=1`.

| configuration | prompt | new | prefill TPS | decode TPS |
|---|---|---|---|---|
| before startup kernel warmup | 512 | 128 | 234.3 | 9.38 |
| after startup kernel warmup | 512 | 128 | **428.1** | 9.21 |
| after startup kernel warmup | 4096 | 32 | **1265.3** | 9.55 |
| 64-layer reference, Cube split decode | 512 | 5 | 435.6 | 9.23 |
| 4096 prefill, prior to this branch | 4096 | 1-16 | 287-325 | 3.20 |

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
  1 `ar.hidden_a`) at the measured 0.3923 ms per call — flat in payload, so per-call and not
  per-byte — serialised by the per-call `stream_synchronize` inside `end_nccl_collective`. That is
  **50.6 ms** of the 108.6 ms step. (An earlier revision of this page put the same calls at ~540 us
  each and the total at ~70 ms; the 0.3923 ms figure in the table above is the measured one.)

**100 TPS is not reachable on this part at fp16.** It needs <= 10 ms/token; the perfect-streaming
bound at TP8 is 21 ms with zero collective cost, which would already require 673 GB/s per card.
Reaching the target requires weight quantization — int8 for 2x, int4 for 4x — plus removing the
per-step collective serialisation. Note also that 8 cards is the practical maximum here:
`HcclCommInitAll` is unusable on this stack and one process per rank is required.

## 7. What this rules out

- **More GEMM tuning for prefill.** Projections are ~3% of prefill.
- **Blaming prefill on attention.** `full_attention` is 9.8% at 512 tokens and is measured separately
  from `STACK.r`; attributing the 66% to it would point the next optimisation at the wrong unit.
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
