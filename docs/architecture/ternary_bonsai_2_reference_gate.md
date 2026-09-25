# Ternary-Bonsai-2-27B: the reference gate

This is task 1 of 6 in [stage 1 of the checkpoint roadmap](pocketllm_new_model_roadmap.md#stage-1--ternary-bonsai-2-27b)
([#382](https://github.com/lvyufeng/PocketLLM/issues/382) under [#381](https://github.com/lvyufeng/PocketLLM/issues/381)).
The roadmap declared this stage's gate as *measure the upstream reference on this card, and if the ternary path is
slower than the memory it saves, close the stage on that measurement*. This page is that measurement, the four
answers the stage's later tasks depend on, and the verdict.

**Verdict: the gate passes, and not narrowly.** On one 2080 Ti the reference decodes a 26.90 B-parameter model at
**30.7 tokens/s** while streaming all **5.53 GiB** of its weights every token, and prefills at **665 tokens/s**.
`Qwen3.8-27B-FP8`, the runtime this stage reuses, needs four cards to reach 44 tokens/s. The ternary artifact is not
a slower way to run the same model; it is the only way this model runs on this card at all.

## What was run

| | |
| --- | --- |
| Checkpoint | `prism-ml/Ternary-Bonsai-2-27B-gguf`, `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes |
| Reference | [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp) at `842b188`, branch `prism` |
| Build | `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF`, CUDA 12.4, gcc 11.4.0 |
| Card | RTX 2080 Ti, physical index 2, 22528 MiB, compute capability 7.5, `CUDA_VISIBLE_DEVICES=2` |
| Host | the x86_64 2080 Ti machine, no NPU, `-t 16` CPU threads for the graph |
| Tool | `llama-bench -ngl 99 -r 5`, which times the prompt forward as prefill and later single-token forwards as decode |

`llama-bench`'s convention is the repository's: the first generated token belongs to prefill and is not counted in
decode. Every row below is a fresh process, weights loaded once, no cache prewarm beyond the load itself.

Two properties of this host are part of the reading. The checkpoint is mmapped and the box has a terabyte of RAM, so
after the first run the 5.53 GiB of weights live in page cache and every run after that is a warm-cache number; a
cold read off `/mnt/data2` is not measured here. And as the next table shows, the card is at its power limit, so
"what the reference does" has a few percent of slack in it either way.

### The card is power-capped, and the first run of a series reads high

Repeating one command six times in six processes settles it rather than measuring noise:

| series | run 1 | 2 | 3 | 4 | 5 | 6 | warm spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pp512` prefill only | 663.6 | 664.4 | 665.4 | 665.3 | 666.6 | 666.1 | 0.3% |
| `pp512` with a decode row | 700.8 | 680.7 | 668.0 | 665.2 | 664.9 | 664.8 | 2.4% |
| `tg128` @ depth 0 | 32.90 | 32.10 | 31.21 | 30.92 | 30.71 | 30.67 | 0.8% (runs 4–6) |

The card sits at its **260 W power cap** — `nvidia-smi` reports 257 W of 260 W during a run — and it is 58 °C before
that series and 83 °C after, so a run that starts on a cold card holds a higher SM clock for as long as the thermal
mass allows. The numbers below are the settled ones. A first run after an idle period reads up to 5% high, which is
larger than most of the effects the later kernel task will be chasing, so anything measured here has to be compared
warm and preferably interleaved.

### Reference throughput, one card

| test | tokens/s |
| --- | ---: |
| pp512 | 665.4 ± 1.1 |
| pp2048 | 668.5 ± 2.8 |
| pp4096 | 642.5 ± 8.1 |
| pp8192 | 615.2 ± 1.0 |
| tg128 @ depth 0 | 30.7 ± 0.1 |
| tg128 @ depth 4096 | 30.3 ± 0.1 |
| tg128 @ depth 32768 | 27.9 ± 0.1 |
| pp512 @ depth 4096 | 623.7 ± 9.2 |
| pp512 @ depth 32768 | 456.1 ± 5.0 |

Decode loses 9% from an empty context to 32768 tokens of it, which is what a 48-layer gated-DeltaNet trunk plus 16
full-attention layers should cost — the recurrent layers do not grow with context, and the 16 that do are a quarter
of the model. Prefill is flat from 512 to 2048 and falls off from 4096 onward.

### What fits, at what context

The checkpoint's `context_length` is 262,144. On this card the weights take 5.53 GiB, so the KV cache has roughly
16 GiB to live in — and the 16 full-attention layers want 64 KiB per token in FP16 (4 KV heads × 256 × 2 tensors ×
2 bytes).

| context | KV dtype | peak GPU 2 | generation |
| ---: | --- | ---: | ---: |
| 131,072 | FP16 | 14,216 MiB | 32.6 t/s |
| 196,608 | FP16 | 18,376 MiB | 32.6 t/s |
| 262,144 | FP16 | *does not fit* | — |
| 262,144 | `q8_0` | 15,836 MiB | 32.3 t/s |

So the model card's 262,144 is real on a 22 GiB card, but only with a quantized KV cache: at FP16 the cache alone is
16 GiB. That is a memory statement, not a speed one — the last column is flat across the rows that ran, and it is
llama-cli's own per-run figure from a 64-token generation, not the `llama-bench` decode above. The two tools differ
by about 5% here (32.3 against 30.7) and the gap is the cold-first-run effect: each of these memory runs was the
first run of its own series.

### The comparison the gate asked for

| | Ternary-Bonsai-2-27B | Qwen3.8-27B-FP8 |
| --- | --- | --- |
| Weights | 5.53 GiB | ~28 GB |
| Cards | **1** | **4** (TP4) |
| Prefill at 8,192 | 615 t/s | 1,819 t/s |
| Decode | 30.7 t/s | ~44 t/s (×1.67–3.21 with MTP) |
| 262,144 context | yes, with `q8_0` KV | yes |
| Runs on one 2080 Ti | **yes** | no |

Per card the ternary path wins on both axes — 615 t/s of prefill and 30.7 t/s of decode against 455 and 11 — but
that framing flatters it, because four cards are four cards. The claim that survives scrutiny is narrower and
stronger: **the FP8 path cannot run this model on one card, and the ternary path can.** That is the whole reason
the stage exists, and the measurement says it delivers.

## The four things the stage needed read out of the fork

Everything below was read from the fork's source at `842b188`, and the parts that a later task will reimplement were
first reproduced and checked against the fork's own reference code rather than against a name or a model card.

### 1. The `PTQ1_0` block: 128 weights in 28 bytes

`QK_PTQ1_0 = 128`, and the block is `uint8_t qs[24]`, `uint8_t qh[2]`, `ggml_half d` — 1.75 bits per weight, one
scale per 128 weights. `src/loader/gguf/ptq1_0.py` implements the decoder and `tests/test_ptq1_0_layout.py` pins it
against ten blocks read out of the released file at known rows and decoded by the fork's reference.

The layout is upstream `TQ1_0`'s base-3 packing with the scale group halved from 256 to 128, and the packing has
three features that a reader is likely to get wrong:

- **Trits are stored as a ceiling division, not a modulo.** The encoder writes
  `q = (q * 256 + 242) / 243` and the decoder recovers trit *n* with `xi = ((uint16_t)(byte * 3**n) * 3) >> 8`.
  The intermediate is `uint8_t`, so `byte * 3**n` **wraps** — that wrap is part of the format.
- **`qs` is walked in stages, not as one run.** The declared stages are `{32, 16, 8}`; a 32-byte stage does not fit
  in 24 bytes, so the walk reduces to a 16-wide stage covering bytes 0..15 (weights 0..79) and an 8-wide stage
  covering bytes 16..23 (weights 80..119). Reading the 24 bytes as a continuous 120-trit run puts different trits on
  different weights and produces a model that generates plausible nonsense.
- **`qh` carries the last 8 weights interleaved by parity.** `qh[0]` holds weights 120, 122, 124, 126 and `qh[1]`
  holds 121, 123, 125, 127, four trits per byte.

### 2. The Hadamard, and how it is applied

The weights in the file are **pre-rotated**. The checkpoint stores `W' = W · R⁻¹` for

```text
R = (1/√N) · H_N · diag(s),   H_N[i][j] = (-1)^popcount(i ∧ j),   N = 1024
```

`H_N` is the natural-order (Sylvester) Hadamard matrix, materialised as an `N × N` FP32 tensor of `±1/√N`, and `s` is
a per-width sign vector of `±1` declared in the metadata — one each for widths 5120, 6144 and 17408, concatenated
into `prism.hadamard.sign_values`.

At runtime the *activation* is transformed instead: before every folded weight's matmul, `x` becomes `R x`, i.e.
multiply by `s` and then apply the Walsh–Hadamard transform, blockwise along the last dimension. The rotation is
memoised per activation (`hadamard_memo` in `llama-graph.cpp`) so that two weights reading the same activation —
`ffn_gate` and `ffn_up`, say — rotate it once. Since the weight was folded with `R⁻¹`, the model computes exactly
what it computed before rotation. The rotation changes nothing about the function; it is the incoherence transform
that makes 1.75-bit weights able to represent that function at all.

`prism.hadamard.gdn_v_grouped = true` adds one more step, for the 48 `ssm_out.weight` tensors only: the 6144-wide
gated-DeltaNet output arrives tiled as `[head_dim 128, groups 16, rep 3]` and the folded weight expects the grouped
`[128, rep 3, groups 16]` order, so a reshape-and-permute runs before the signs. (Those two orders come from the
fork's own comment on the permute, `llama_hadamard_transform` in `llama-graph.h`;
`tests/test_prism_hadamard_transform.py` pins our reading of them against the fork's `ggml_permute` itself, because
the two orders are easy to state the wrong way round.) Getting this wrong permutes features within each head rather
than scrambling the model, which is the kind of bug that shows up as a small quality loss.

### 3. `token_embd.weight` gets the *inverse*, and only it

`prism.hadamard.weight_names` lists **401** tensors: 288 in the 48 linear-attention blocks (six each — `attn_qkv`,
`attn_gate`, `ssm_out`, `ffn_down`, `ffn_gate`, `ffn_up`), 112 in the 16 full-attention blocks (seven each — `attn_q`,
`attn_k`, `attn_v`, `attn_output` replacing `attn_qkv` and `attn_gate`), and `output.weight`.

`prism.hadamard.inverse_weight_names` lists exactly one: `token_embd.weight`. The reason is the difference between a
matmul and a lookup. Every other folded weight is a *matrix* that will be multiplied by an activation, so the
rotation lands on the activation and the weight stores `W R⁻¹`. An embedding table is not multiplied by anything —
it is indexed. The stored rows are `R z` for a latent `z`, so the rotation has to be undone *after* the lookup:

```text
forward (401 weights):  x ↦ R x  = (1/√N) · H · (s ⊙ x)      signs first, then H
inverse (token_embd):   z ↦ R⁻¹ (R z) = (1/√N) · s ⊙ (H · z)  H first, then signs
```

The two orders differ, and the fork's code says so explicitly — the inverse path applies the transform and *then*
multiplies by the signs. `output.weight` is a matmul even though it is the other end of the same table, so it takes
the forward form like everything else.

### 4. A real ternary kernel, not an FP16 upcast

`ggml_cuda_should_use_mmq` returns `turing_mma_available(cc)` for `GGML_TYPE_PTQ1_0`, and **compute capability 7.5
satisfies it**: `GGML_CUDA_CC_TURING` is 750 and the build was compiled for 75. That choice also selects the MMA
shared-memory layout, so the dot product is `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` and not the `..._dp4a` variant the
other quant types take on Turing. The tile loader `ggml_cuda_mmq_load_tiles_ptq1_0` unpacks each 128-weight block
into *packed int8* in shared memory (`__vsub4(__byte_perm(...), 0x01010101)`) against activations quantized to int8
per 32 — an exact integer product with an int32 accumulator, not a float approximation.

There *is* an FP16-dequantize-then-cuBLAS fallback, reached by lowering `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH`, and the
fork's own comment calls that path the source of "PTQ1_0's extra error on CUDA". So the accurate path is the default
and the fallback is the opt-in. On this card the fallback is also marginally *faster* on prefill and identical on
decode, which is worth writing down because it means the ternary packing is not buying prefill throughput here:

| adjacent pair (order alternated) | MMQ pp512 | fallback pp512 | MMQ tg128 | fallback tg128 |
| --- | ---: | ---: | ---: | ---: |
| 1 (MMQ first) | 703.3 | 711.8 | 33.09 | 32.53 |
| 2 (fallback first) | 668.7 | 700.4 | 31.48 | 32.05 |
| 3 (MMQ first) | 668.0 | 689.9 | 31.17 | 31.09 |
| 4 (fallback first) | 665.4 | 689.1 | 30.84 | 30.97 |
| mean effect | | **+21.5 t/s (+3.2%)** | | −0.02 t/s (0) |

Two details make those columns readable. The absolute numbers drift down over the series because the card is at its
power cap, which is why the order alternates; only the within-pair difference means anything, and it is positive in
all four pairs. And the decode column has no effect at all because **the switch does not reach decode**:
`ggml_cuda_should_use_mmvq` returns `ne11 <= 7` for `PTQ1_0` specifically, and that test runs *before* the MMQ test,
so a one-token step takes the vector-matrix kernel either way. The switch only ever changes a multi-token batch.

The two arms are genuinely different code, not the same kernel under two names: the fallback arm peaks 184 MiB
higher (6,512 against 6,328 MiB), which is the F16 copy cuBLAS needs of one weight matrix — 5,120 × 17,408
elements — allocated from the pool and reused node to node. So the finding is real, but it is a finding about
*accuracy versus 3% of prefill*, and the fork's default is the right one to inherit: `#386` should be trying to beat
both, not choosing between them.

## What the reference leaves on the table

A gate measurement is worth as much for what it says about the remaining task as for the verdict, and the two phases
point in opposite directions.

**Decode is memory-bound and the ternary weights are already winning there.** A one-token step takes the
vector-matrix kernel (see above) and reads all 5.53 GiB once: at 30.7 tokens/s that is a 32.6 ms step and **170
GiB/s, or 30% of the card's 574 GiB/s**. Streaming the same weights at peak bandwidth would take 9.6 ms and run at
~104 tokens/s, so two thirds of the step is something other than DRAM traffic — launch overhead across 64 blocks,
the full-attention layers, the DeltaNet scan. That residual is what `#386`'s decode half gets to attack. The size of
the prize is worth stating plainly: at FP16 the same weights would be 53.8 GB per token, which at the same achieved
bandwidth is 3.4 tokens/s, so the 2-bit packing is worth about nine times on decode — and decode is the phase where
the format pays for itself.

**Prefill is not.** The `MMQ`-versus-`cuBLAS` table above is the evidence: routing the projections to a
tensor-core F16 GEMM over an F16 copy of the weights — ten times the bytes — moves prefill by 3% and in the *faster*
direction. Whatever is bounding pp512 at 665 tokens/s, it is not the weight format, and it is not integer throughput
either: 26.9 B parameters × 2 × 512 tokens is 27.5 TFLOP, which at the 2080 Ti's 215 int8 TOPS would be 128 ms
against the 770 ms measured. Something structural is in the way — most likely the MMQ tile shapes at 512 tokens, or
the hybrid attention stack, which is 64 blocks of gated DeltaNet whose sequential scan does not care how the
projections are stored.

That is the useful thing this gate found for `#386`: **the decode kernel is where the format can be beaten, and the
prefill number is a target for a profiler rather than for a kernel.**

## What this settles, and what it does not

Settled:

- the stage proceeds; tasks 2 through 6 are worth doing;
- the artifact is self-describing. `prism.hadamard.*` names every folded tensor, declares the block size, the sign
  vectors and the GDN permutation, so nothing has to be inferred from a model card;
- the packing and the transform are pinned by executable tests rather than by prose;
- the type ids are `143` (`PTQ1_0`) and `142` (`PQ2_0`), and both now decode their geometry in
  `src/loader/gguf/reader.py` instead of reading as `unknown_` with a zero block size. Dispatch is still refused:
  knowing a block's size is not the same as being able to run it, and the loader must keep failing loudly until
  `#384` teaches it otherwise.

Not settled, and deliberately so:

- **No parity claim.** Nothing here compares a generated sequence against an FP16 parent, because this repository
  has not run the model yet. The reference generating a correct answer from a real prompt is a smoke test, not a
  correctness result — the roadmap's quality figures belong to the authors' evaluations.
- **No claim about the model's quality.** 84.78 average over the authors' 14 thinking-mode benchmarks is quoted in
  the roadmap as a reason to do this; it is not a measurement this repository made.
- **`PQ2_0` was not run.** Only the `PTQ1_0` artifact was measured. `PQ2_0` at 6.71 GiB is 21% larger for about 0.4
  more bits per weight; if it wins on quality that is a later comparison, not part of this stage.
- **No tuning was attempted.** Every number is the stock reference with `-ngl 99`. The
  `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` table above is the only switch touched, and it is the fork's own.

## Evidence

```bash
# the reference build
cd /mnt/data1/llama_cpp_prism
PATH=/usr/local/cuda-12.4/bin:$PATH cmake -B build-sm75 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build-sm75 --target llama-cli llama-bench -j 32

# one process per published row, so no row inherits the previous row's state
LD_LIBRARY_PATH=$PWD/build-sm75/bin CUDA_VISIBLE_DEVICES=2 ./build-sm75/bin/llama-bench \
  -m /mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -t 16 -r 5 -p 512 -n 128 -d 0     # and -p 2048/4096/8192 -n 0, -d 4096/32768

# the repeatability series: the same command six times, which is what shows the
# card settling rather than the kernel varying
for i in $(seq 6); do ... llama-bench -r 5 -p 512 -n 128 -d 0; done

# the switch A/B, order alternated because the card drifts within a series
GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0 llama-bench -r 5 -p 512 -n 128 -d 0   # the other arm

# which path each arm takes: the fallback needs an F16 copy of one weight matrix
nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits   # sampled at 0.25 Hz

# the layout test, which needs neither the checkpoint nor the fork
python -m pytest tests/test_ptq1_0_layout.py -q
```

- `docs/architecture/pocketllm_new_model_roadmap.md` — the stage this gate belongs to
- [Qwen3.8-27B-FP8 design and measurements](qwen3_8_27b_fp8_design.md) — the runtime, and the comparison above
- [Benchmarking and reporting rules](../guides/benchmarking.md) — the convention every number here follows
