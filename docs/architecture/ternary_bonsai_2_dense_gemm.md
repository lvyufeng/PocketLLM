# Ternary-Bonsai-2-27B: the sm_75 dense GEMM

This is task 5 of 6 in [stage 1 of the checkpoint roadmap](pocketllm_new_model_roadmap.md#stage-1--ternary-bonsai-2-27b)
([#386](https://github.com/lvyufeng/PocketLLM/issues/386) under [#381](https://github.com/lvyufeng/PocketLLM/issues/381)).
The [reference gate](ternary_bonsai_2_reference_gate.md) measured what the upstream ternary runtime does on one
2080 Ti and drew the stage's map: the packing is real, the trits expand to signed bytes so the tensor-core dot is
an ordinary integer dot, and *"the decode kernel is where the format can be beaten, and the prefill number is a
target for a profiler rather than for a kernel"*. This page is what was built on that map, what it measures, and
what it does not yet beat.

**Verdict: correct, and faster than the FP8 path on decode but not on prefill.** One 2080 Ti running only the
dense projections of this model decodes at **21.8 tok/s** where the same projections at FP8 width — the shape the
Qwen3.8-27B runtime uses — afford **9.2 tok/s** on the same card, a **2.4×** difference for the 1.75-bit packing.
Prefill is the other way round: **600 tok/s** of dense projections at a 512-token prompt against **761** for the
FP8-width arm, so the packing buys nothing there, and both phases are behind what the upstream reference's *whole
model* reaches on this card (30.7 tok/s and 665 tok/s). The kernels are correct and the numbers are honest; the
work left is kernel work, and this page says exactly where it is.

## What was built

Two kernels in `src/csrc/llama_mmq/gguf_mma_wrapper.cu`, behind the two dispatchers the rest of the engine already
uses, keyed on GGML type id **143**:

| Phase | Entry point | Structure |
| --- | --- | --- |
| Prefill | `gguf_ptq1_0_mma_prefill_forward` | INT8 MMA over a tile the loader expands from trits to signed bytes |
| Decode | `gguf_ptq1_0_dp4a_decode_forward` | `__dp4a` GEMV, one thread per output row |

Both are dispatched unconditionally rather than behind a gate. There is no float path behind either one: the
blocks are 1.75-bit packed trits, so falling through would read them as if they were fp16 or fp32 — plausible
output from the wrong numbers, which is the failure mode this repository has been bitten by before.

### Prefill follows the fork's own trick

The loader (`load_tiles_ptq1_0` in `src/csrc/llama_mmq/mmq.cuh`) is a port of the fork's
`ggml_cuda_mmq_load_tiles_ptq1_0`, and the trick it copies is worth stating: **it does not invent a ternary dot
product at all.** Each 128-weight block is expanded, one signed byte per trit, into the shared-memory tile
layout that `vec_dot_q8_0_q8_1_mma` already expects — so `PTQ1_0` becomes an ordinary `Q8_0` tile, `VDR` is 2
(two 128-weight blocks per 32-int rung), and the DS layout is `D4`. Nothing downstream of the loader knows the
weights were ternary.

The stage walk the loader implements is the same one `src/loader/gguf/ptq1_0.py` decodes: a 16-wide stage over
`qs[0..15]` carrying weights 0..79 at a stride of 16, an 8-wide stage over `qs[16..23]` carrying weights 80..119
at a stride of 8, and `qh` carrying 120..127 with the parity interleaved. Lane roles within the loader's 8-lane
group are `0..3` → `qs` words 0..3, `4,5` → words 4,5, `6` → `qh`, `7` → idle.

### Decode is a GEMV whose only problem is bytes

A one-token step reads all 5.474 GiB once and does one multiply-accumulate per weight, so it is a bandwidth
problem and nothing else. The kernel gives **one thread one output row**, and each thread unpacks a whole
128-weight block per iteration with four integer accumulators — one per 32-element activation block, which is the
fork's `sumi[4]` shape. That shape is what keeps the scale out of the inner loop: it is applied once per block
instead of once per four weights, and the activation's `d` and the float multiply both leave the 32-step loop.
Reading the whole 28-byte block per lane per iteration, rather than four bytes, is what buys the memory-level
parallelism that covers DRAM latency.

The activation is staged in shared memory — only the running chunk of it, because at K = 17,408 staging all of it
costs 19.6 KiB per block and caps the resident blocks per SM at two. K is also split whenever one thread per row
would not fill the card, which on this host is most shapes: 17,408 rows is 136 blocks, two per SM. Split chunks
land in separate accumulators and a second kernel sums them in a fixed order, so the result does not depend on
the schedule.

## Correctness

`tests/test_gguf_ternary_gemm.py`, 23 cases, all passing. The evidence is layered so that a failure localises:

- **The packing is not in question.** A synthetic encoder is built by inverting the decoder's own digit table,
  asserting it round-trips, so every synthetic case reads blocks that `src/loader/gguf/ptq1_0.py` decodes back to
  the trits that were packed. `tests/test_ptq1_0_layout.py` separately pins that decoder against ten blocks read
  out of the released file.
- **Two references, deliberately at different precisions.** The loose one is the unquantized dot product — decode
  the blocks to fp32 and multiply — which is what the issue's acceptance asks for; it can only be as tight as the
  activation quantization the kernels perform (int8 per 32 elements, exactly as llama.cpp does), and it holds to
  6.7e-3 relative over eight seeds at K ∈ {5120, 6144, 17408}. The tight one reproduces that quantization in
  torch and isolates the weight unpack from the activation path.
- **The tight comparison is stated in ulps of the output.** Both entry points return bf16 — the engine's carrier,
  and what the Q4_K/Q5_K paths beside them return — so the reference is rounded to bf16 as well and the bound is
  two ulps. **Measured, the two agree bit for bit on every shape tested**, decode and prefill, synthetic and
  released weights; the ulp of slack is there for a summation order a future warp count could change. An unpack
  error — a wrong stage, a wrong `qh` parity, a scale on the wrong block — moves the answer by units, not ulps.
- **The released checkpoint is exercised**, not just synthetic blocks: 64 rows of `blk.0.ffn_gate.weight` from the
  5.9 GiB artifact, decoded and prefill-batched, both bit-exact. The case skips when the file is absent.

## Throughput

`tests/bench_ptq1_0_dense.py`. One representative tensor per distinct shape is read and timed, then multiplied by
how many tensors share that shape, because the cost depends on `[N, K]` and on nothing else. The sum is the dense
projection cost of one forward pass, and `rows / total` is the token rate it affords with the attention stack and
the launch overhead taken out — a ceiling, not a model measurement.

| | |
| --- | --- |
| Checkpoint | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` — 402 `ptq1_0` tensors, **5.474 GiB**, 8 distinct shapes |
| Card | RTX 2080 Ti, physical index 0, 22528 MiB, `CUDA_VISIBLE_DEVICES=0` |
| Measured read ceiling | **526 GiB/s** over 2 GiB (`torch.sum`), 540 GiB/s at 512 MiB |
| Call round trip | 21.9 µs empty kernel; 89.0 µs for the smallest PTQ1_0 decode call |

Two arms, interleaved within each shape rather than blocked, because the card drifts downward within a series
(it sits at its 260 W cap):

- **ternary** — the two entry points above, 1.75 bits per weight;
- **cublas fp16** — the same weights dequantized to fp16 and handed to cuBLAS. This is what the FP8 Qwen3.8-27B
  path does after its online unpack (fp8 → fp16 → cuBLAS FP16, since Turing has no fp8 tensor core), so it is the
  like-for-like FP8-width arm: same shapes, same precision the FP8 path computes in, **ten times the bytes**.

### Dense projections of one forward pass, summed over the checkpoint

| rows | ternary | tok/s | GiB/s | cublas fp16 | tok/s | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (decode) | 45.90 ms | **21.8** | 119.3 | 108.56 ms | 9.2 | **2.37×** |
| 128 | 253.79 ms | 504.4 | 21.6 | 212.40 ms | 602.6 | 0.84× |
| 512 | 852.92 ms | **600.3** | 6.4 | 672.80 ms | 761.0 | 0.79× |

The GiB/s column is the useful byte rate: at 512 rows the same weights are read once for 512 tokens of work, so
the column falls by design and the compute rate is the reading that matters (33–45 TFLOP/s of nominal ternary
multiply-accumulate).

Per shape, the two ends of the decode column are worth seeing — the spread is not noise, it is the difference
between a shape that fills the card and one that does not:

| shape | count | decode | prefill @512 | cublas decode | cublas prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=248,320 K=5,120 | 2 | 171.8 GiB/s | 45.2 TFLOP/s | 0.35× | 1.26× |
| N=17,408 K=5,120 | 128 | 136.0 GiB/s | 33.5 TFLOP/s | 0.39× | 1.30× |
| N=5,120 K=17,408 | 64 | 125.6 GiB/s | 29.2 TFLOP/s | 0.41× | 1.35× |
| N=10,240 K=5,120 | 48 | 120.3 GiB/s | 31.9 TFLOP/s | 0.42× | 1.11× |
| N=1,024 K=5,120 | 32 | 15.9 GiB/s | 17.0 TFLOP/s | 1.10× | 1.10× |

### Against the FP8 Qwen3.8-27B path, at the same prompt and context

Bonsai-2-27B and Qwen3.8-27B are the same architecture — 48 Gated DeltaNet plus 16 full-attention layers, hidden
5120, dense MLP 17,408, 262,144 positions — which is why stage 1 reuses that runtime, and why the FP8 numbers are
a fair yardstick at the same prompt length.

| | ternary, 1 card | FP8 Qwen3.8-27B, TP4 |
| --- | --- | --- |
| Weights | 5.474 GiB, all on one card | 6.86 GiB resident **per rank** |
| Dense projections, 512-token prompt | **600 tok/s** (this page) | — |
| Dense projections, decode | **21.8 tok/s** (this page) | — |
| Whole model, 512-token prompt | — | 864.54 tok/s (4.5 s for 8,192) |
| Whole model, decode at a 512 prompt | — | 43.22 tok/s |
| Whole model, decode per card | — | 10.8 tok/s |

The honest reading of that table: **one card's dense projections alone, with the attention stack and the launch
cost not yet added, reach about twice what four FP8 cards reach together on decode, and about 70% of what they
reach on prefill.** The decode half is the packing paying for itself exactly as the gate predicted. The prefill
half is the packing not paying, which is the next section.

The model-level number — this runtime, all 64 layers, the serving surface — is task 6
([#387](https://github.com/lvyufeng/PocketLLM/issues/387)) and is not claimed here.

## What is left, and why

Two measured gaps, with the mechanism for each.

### Decode: 119 GiB/s against a 526 GiB/s ceiling

The big shapes reach 136–172 GiB/s, so the kernel is not thrashing and the aggregate is dragged down by the small
ones. But even the best shape is a third of what the card can read, and the reason is the shape of the load: a
lane reads *its own* 28-byte block with seven 4-byte loads, so a warp-level load instruction presents 32
scattered addresses 28 bytes apart, one per 32-byte sector. Four useful bytes per lane per transaction is the
whole problem, and it is measurable without writing a kernel — a torch reduction over every seventh float is
exactly that pattern:

| access pattern | time | useful | touched |
| --- | ---: | ---: | ---: |
| dense 512 MiB read | 0.950 ms | 526.4 GiB/s | 526.4 GiB/s |
| every 7th float (28-byte stride) | 0.977 ms | **73.1 GiB/s** | 511.5 GiB/s |

Same DRAM traffic, one seventh of it useful. So the fix is not a different unpack — the unpack is the fork's and
it is already bit-exact — it is to stop issuing strided 4-byte loads at all: **stage the weight rows into shared
memory with coalesced 128-byte reads and unpack from there**, which is what this repository's MMQ prefill loader
already does and what the fork's own decode path does not. The stub of that change is visible in the file: the
chunk-local activation staging was added for exactly this reason, and the weights are the other half of it.

Three smaller readings belong with it:

- **The small shapes are launch-bound, not bandwidth-bound.** N=1,024 K=5,120 is 32 tensors and 2.3 MB each; at
  15.9 GiB/s the kernel spends most of its 67 µs inside a 22 µs round trip.
- **The entry point costs 89 µs of host time per call** against 22 µs for an empty kernel. A model pass makes 402
  such calls, so the fixed cost is ~27 ms — the same order as the 46 ms of kernel time. Batching the shapes that
  read the same activation (`ffn_gate` and `ffn_up` are the same `[N, K]`) or capturing the step into a graph is
  worth more here than another 10% of bandwidth, and that is the wiring task's call rather than this one's.
- **The K-split costs a second kernel launch.** It is only taken when one thread per row would leave the card
  idle, and the partials are 0.2% of the bytes, so it is not the reason for anything above — but it is why the
  decode column has two launches per call where the prefill column has one.

### Prefill: 600 tok/s, and *slower* than the FP16 arm it replaces

At 512 rows the ternary prefill is 1.27× slower than handing the same weights to cuBLAS as fp16 — ten times the
bytes, more of the time. This is not a surprise in kind: the reference gate found the fork's own MMQ 3.2% slower
than its cuBLAS fallback on this card. It is a surprise in degree, because the loader here is a line-by-line port
of the fork's. The tile loader is therefore probably not where the difference is; the launch configuration is
where to look first — `launch_prefill` picks `mmq_x` from divisibility alone, and this type's `VDR` of 2 is the
only one in the table, so the tile geometry this type lands on is the least-tested cell in the matrix. The
ternary prefill is a target for a profiler, in the gate's words; it is not a target for another packing.

## What this settles, and what it does not

Settled:

- `PTQ1_0` runs on sm_75 in both phases, bit-exact against its own reference arithmetic over the released weights
  and over synthetic blocks that the decoder round-trips;
- the type is wired through the existing dispatchers, so a caller does not have to know which quant it holds —
  only that 1.75-bit blocks are not fp16;
- decode is where the format pays, measured rather than asserted: 2.4× the FP8-width arm at 10× fewer bytes;
- both phases are behind the upstream reference's whole-model numbers on this card, so task 6 cannot be a wiring
  exercise alone.

Not settled, and deliberately so:

- **No model-level throughput and no quality claim.** Nothing here runs 64 layers or compares a generated
  sequence. The quality figures in the roadmap belong to the authors' evaluations.
- **`PQ2_0` was not run.** Its block geometry is declared next to `PTQ1_0`'s in
  `src/csrc/llama_mmq/ggml-common.h`, but the kernels are `PTQ1_0` only.
- **No claim about the attention stack.** 16 full-attention layers plus 48 Gated DeltaNet layers are outside this
  kernel's scope entirely, and on this card they are a large part of the reference's step: the gate measured
  decode losing 9% from an empty context to 32,768 tokens of it.

## Evidence

```bash
# the extension, which is what the kernels are built into
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 \
  TORCH_CUDA_ARCH_LIST=7.5 python setup.py build_ext --inplace
# exits 1 on the unrelated pocketllm_cpp NVFP4 sources; read the .so timestamp instead

# correctness, and the shape/tolerance suite the issue asks for
python -m pytest tests/test_gguf_ternary_gemm.py -q          # 23 passed

# the acceptance number: dense projections of one forward pass, both arms
python -m tests.bench_ptq1_0_dense --rows 1 128 512 --iters 15 --warmup 4

# the mechanism behind the decode gap: one access pattern at a 28-byte stride
# (inline; there is no script, it is three lines against a 512 MiB randn)
python -c "import torch,time;a=torch.randn(128*1024*1024,device='cuda');v=a[::7];v.sum()"
```

- `docs/architecture/ternary_bonsai_2_reference_gate.md` — what the upstream reference measures, and the four
  facts these kernels were built on
- `docs/architecture/qwen3_8_27b_fp8_design.md` — the FP8 runtime the comparison is against, and its projectors
- [Benchmarking and reporting rules](../guides/benchmarking.md) — the convention every number here follows
