# The fleet support matrix

Three accelerator classes run Qwen3 through this project, against one reference. This page is the
single place that states **what each one runs, how accurately, and how fast** — it is an index into
the measurements, not a new one: every number below is taken from a page that carries it with its
method, and where those pages disagree with this summary, they win.

The three are different machines with different surfaces, so the honest statement is per-device and
not one "matches to 1e-3". The reference is set below, and each section says what it can and cannot
be compared to.

## The reference: the 2080ti C engine

The **2080ti** host runs the native engine [`libpocketllm.so`](c_engine.md) — the GGUF reader, the
BPE tokenizer, the Qwen3 forward pass and the decode loop, all in C. Qwen3-0.6B is checked against
llama.cpp **token for token** on both `cpu` and `cuda`, for the f16 checkpoint and for the `q4_k_m`
one quantized from it. It exposes the **full logit vector** (`--print-top K`) and a
`pocketllm-bench` harness, so it is the one host where a **logit-level 1e-3** statement is even
expressible — and it is the oracle the other two boards are scored against.

| | value |
|---|---|
| CPU (`1.7B` q4_k_m, 22 threads, vs llama.cpp) | `pp512` **~0.99×** (parity), `tg128` @depth 512 **1.21–1.23×** |
| CUDA (`1.7B` q4_k_m) | `pp512` **292.6 t/s**, `tg128` **66.30 t/s** |

The CUDA kernels are naive (prefill is slow next to a tuned BLAS), so the number to quote for the
reference is the **CPU** one, where the engine is at parity with llama.cpp on prefill and ahead on
decode.

## The fleet at a glance

| | **2080ti (CUDA, sm_75)** | **S600 (Horizon Nash BPU)** | **Orange Pi (Ascend 310B1)** |
|---|---|---|---|
| Runtime | CUDA runtime + torch | Horizon OpenExplorer `libhbrt4.so` | CANN 8.3.RC2, `libascendcl.so` |
| Path | eager kernels | **AOT**: host compiles a `.hbm`, board runs it | aclnn/ACL + AscendC custom ops |
| Weight format | GGUF `f16` / `q4_k_m` | `leap_llm` symmetric per-channel int4 | GGUF `q4_k`/`q6_k`, faithfully decoded |
| Qwen3-0.6B | ✅ f16, q4_k_m | ✅ w8 | ✅ q4_k_m |
| Qwen3-1.7B | ✅ q4_k_m, f16 | ✅ w4 | ✅ q4_k_m |
| Qwen3-4B | ✅ q4_k_m, f16 | ✅ w4 | ✅ q4_k_m |
| Qwen3-8B | ✅ q4_k_m, f16 | ✅ w4 (87.7% of the pool) | ⚠️ runs, but at the memory/swap edge |
| Sampling | temperature/top-k/top-p/min-p/seed | same (RNG on the host) | same (RNG on the host) |
| `serve` | ✅ | ✅ board-tested | ✅ board-tested (`supports_batch = False`) |
| Accuracy surface | **logits** (1e-3 expressible) | text/tokens only | tokens only |

## Accuracy against the 2080ti oracle

The metric is **shared-prefix token agreement** over a fixed 50-prompt corpus (the harness in
[s600_qwen3.md](../models/s600_qwen3.md#accuracy-against-the-2080ti-oracle)), scored against a
near-lossless **f16** checkpoint (truth) and against the **q4_k_m** checkpoint the board runs (the
floor). A board whose agreement with truth equals the q4_k_m checkpoint's *own* agreement with truth
is measuring the quantizer's error, not the board's.

| Device | Result | What "1e-3" means here |
|---|---|---|
| **2080ti** | the oracle itself — token-identical to llama.cpp on both widths | logit-level 1e-3 (`--print-top K`) |
| **310B** | ✅ **at the quantizer's own floor, never below** (see below) | **token agreement only** (no logits surface) |
| **S600** | 🟡 a real provenance gap vs truth; cause not yet identified (calibration ruled out) | **token agreement only** (`xlm` returns text, no logits, no seed) |

**310B.** Against f16 the board reaches Σ prefix **1078 / 939 / 1620** (0.6B / 1.7B / 4B) where the
`q4_k_m` checkpoint's own distance from truth is the floor **995 / 956 / 1552** — ratios **1.08 /
0.98 / 1.04**. So the board's agreement with near-lossless weights is *as good as the quantized
checkpoint's own*, and it adds no token-agreement deficit of its own. This is the token-level
confirmation that the faithful `q4_k`/`q6_k` decode + cube is what actually executes: the W4A16 op's
**~1.2e-1** requantization error is real but is **not the path run** (it is env-gated off at
`backend.cpp:762`). Full section: [ascend_310b.md](ascend_310b.md#end-to-end-accuracy-the-310b-sits-at-the-quantizers-own-floor).

**S600.** Against f16 the provenance gap holds at every size — shipped beats ours on total shared
prefix (**1327 vs 973**, **724 vs 394**, **990 vs 536**, **1196 vs 729**). The `q4_k_m`-vs-f16 floor
*grows with the model* and overtakes **both** graphs from 1.7B up, so a GGML k-quant is a near-truth
reference only at 0.6B. The gap is **real against truth, and its cause is not yet identified.** The
first candidate — the compile-time calibration corpus, which defaults to a set with ~0 code and ~0 CJK
— was **tested and falsified**: a corpus matched to the parity categories moved the KV-cache scales
*away* from the vendor's on 44 of 49 tensors (median distance to the vendor's scales: **1.8%** for the
default corpus, **16.4%** for the matched one), so the default calibration was already vendor-matched
and calibration is not the lever. With the KV scales near-identical and the weights data-free-identical
(`w_max/7`), the remaining candidates are the compiler graph's operator rounding and accumulation
order, or a vendor weight-quant scheme that is not data-free. See
[the native compile chain](s600_native_chain.md) and the S600 model page for the record.

## Performance

### S600 — decode t/s (three runs, <0.8% spread)

| Model | Prefill (t/s) | Decode (t/s) | Load | Pool held |
|---|---|---|---|---|
| Qwen3-0.6B w8 (shipped) | 6169–7111 | **88.3** | 5.26 s | 27.9% |
| Qwen3-1.7B w4 (shipped) | 5172–5818 | **70.2** | 5.70 s | 34.8% |
| Qwen3-4B w4 (shipped) | 2415–2573 | **42.8** | 6.44 s | 62.3% |
| Qwen3-8B w4 (shipped) | 1939–2040 | **29.7** | 7.66 s | **87.7%** |
| Qwen3-0.6B w8 (ours c1024) | 11636–13128 | **121.6** | 5.31 s | 15.9% |
| Qwen3-1.7B w4 (ours c1024) | ~8982 | **88** | — | — |
| Qwen3-4B w4 (ours c1024) | 4339–4571 | **49.9** | 6.35 s | 39.6% |

Our own `cache_1024` builds sit **above their shipped twins on both axes** — faster and smaller in
the pool. Read against [s600_qwen3.md](../models/s600_qwen3.md#the-ladder-across-sizes): the 8B is
this board's ceiling.

### Orange Pi — Ascend 310B (marginal decode, `--steps 8` vs `32`)

| Model | Prefill | Decode | One-time plane build |
|---|---|---|---|
| Qwen3-0.6B | 0.065 s/token | **0.26 s/token** (3.8 t/s) | ~18.5 s |
| Qwen3-1.7B | 0.065 s/token | **0.39 s/token** (2.6 t/s) | ~91.5 s |
| Qwen3-4B | 0.129 s/token | **0.587 s/token** (1.7 t/s) | ~212–225 s |
| Qwen3-8B | — | not cleanly separable (swap) | ~500 s |

The wall is almost entirely the **one-time plane build**, not decode. Batching is the throughput
lever: aggregate 1.42 → 8.16 t/s over B=1…16, with a step of `0.1378 + 0.0641·(B−1)` s. The bound is
**interior m=B kernel work, not op-launch count**, which is why the fused attention kernel was
declined — see [the batched-decode bound](ascend_310b_batch_decode_bound.md) and
[ascend_310b.md](ascend_310b.md#current-state-the-canonical-numbers-after-the-series).

## One line each

| Device | Accuracy | Performance |
|---|---|---|
| **2080ti** | the oracle — token-exact vs llama.cpp | CPU at parity with llama.cpp on prefill, ahead on decode; CUDA kernels naive |
| **S600** | real provenance gap vs truth; cause under investigation (calibration falsified) | 88 t/s @0.6B → 30 t/s @8B; our compiled graphs beat the shipped ones |
| **310B** | ✅ **at the floor, never below** — the goal is met on this board | 3.8 → 1.7 t/s; batching lifts aggregate to ~8 t/s |

## Scope

This page summarizes; it does not re-measure. The authority for each row is the page named beside it
([the C engine](c_engine.md), [Qwen3 on the S600](../models/s600_qwen3.md),
[ascend_310b](ascend_310b.md)), and a number that moves there should move here in the same commit.
Two honest limits carry across the table: neither board surface exposes **logits**, so "aligned to
1e-3" is expressible on them **only as token agreement**; and the boards differ in **weight scheme**
(`leap_llm` int4 on the S600, GGUF `q4_k_m` on the 310B), so a cross-board token-agreement number is a
*quantizer-provenance* comparison, not a ranking of the hardware.