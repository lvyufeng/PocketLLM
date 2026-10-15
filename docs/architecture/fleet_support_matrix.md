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
| CPU (`1.7B` q4_k_m, 22 threads, vs llama.cpp) | `pp512` **1.037×**, `pp2048` **1.034×** (6/8 interleaved rounds ahead), `tg64` @depth 512 **1.37×** |
| CUDA (`1.7B` q4_k_m) | `pp512` **1031.3 t/s**, `tg128` **106.8 t/s** (0.107× / 0.404× of llama.cpp) |

The CUDA kernels are naive (prefill is slow next to a tuned BLAS — a tenth of llama.cpp's rate), so
the number to quote for the reference is the **CPU** one, where the engine is *ahead* of llama.cpp on
both halves: prefill past parity and decode by 1.37×. The CUDA decode rate has climbed — the decode
GEMM's k-split took it 0.259× → **0.404×**, and the `dp4a` activation path that would close more of
the gap has since been built, measured (decode **0.41×**, i.e. a loss) and closed: see
[the C engine](c_engine.md#the-int8-activation-path-was-built-and-measured-and-it-does-not-pay-do-not-re-attempt-it-blind).

## The fleet at a glance

| | **2080ti (CUDA, sm_75)** | **S600 (Horizon Nash BPU)** | **Orange Pi (Ascend 310B1)** |
|---|---|---|---|
| Runtime | CUDA runtime + torch | Horizon OpenExplorer `libhbrt4.so` | CANN 8.3.RC2, `libascendcl.so` |
| Path | eager kernels | **AOT**: host compiles a `.hbm`, board runs it | aclnn/ACL + AscendC custom ops |
| Weight format | GGUF `f16` / the k-quant rung `q2_k`–`q6_k` / `iq2_xxs`, `iq2_xs` | `leap_llm` symmetric per-channel int4 | GGUF `q4_k`/`q6_k`, faithfully decoded |
| Qwen3-0.6B | ✅ f16, q4_k_m, q2_k/q3_k/q5_k, iq2_xxs/iq2_xs | ✅ w8 | ✅ q4_k_m |
| Qwen3-1.7B | ✅ q4_k_m, f16 | ✅ w4, w8 | ✅ q4_k_m |
| Qwen3-4B | ✅ q4_k_m, f16 | ✅ w4, w8 | ✅ q4_k_m |
| Qwen3-8B | ✅ q4_k_m, f16 | w4 (87.7% of the pool, **not deployable for long generation**); w8 fits (54.2%) but is still not the deployable long-gen graph | ✅ q4_k_m, release on (fits; 0.65 t/s, no swap in the fit probe) |
| Sampling | temperature/top-k/top-p/min-p/seed | same (RNG on the host) | same (RNG on the host) |
| `serve` | ✅ (batch when `max_batch_size > 1`) | ✅ board-tested — **no batch, see below** | ✅ board-tested (batch when `max_batch_size > 1`) |
| Accuracy surface | **logits** (1e-3 expressible) | text/tokens only | tokens only |

**S600 `serve` — no batch.** A measured wall, not a design choice: the delegate accepts
`request_num <= 4` but its capacity for Qwen3 is 1 (`max_batch_num is: 1`, a table keyed by model
type inside `libxlm.so`), and a 2- or 4-request batch of NORMAL-priority requests SIGSEGVs inside
the runtime. Cancellation is a wall of the same kind — the SDK's public surface has no cancel
symbol. Full record: `python/pocketllm/server/xlm_backend.py`, `capabilities()`.

## Accuracy against the 2080ti oracle

The metric is **shared-prefix token agreement** over a fixed 50-prompt corpus (the harness in
[s600_qwen3.md](../models/s600_qwen3.md#accuracy-against-the-2080ti-oracle)), scored against a
near-lossless **f16** checkpoint (truth) and against the **q4_k_m** checkpoint the board runs (the
floor). A board whose agreement with truth equals the q4_k_m checkpoint's *own* agreement with truth
is measuring the quantizer's error, not the board's.

| Device | Result | What "1e-3" means here |
|---|---|---|
| **2080ti** | the oracle itself — token-identical to llama.cpp on both widths | logit-level 1e-3 (`--print-top K`) |
| **310B** | ✅ **at the quantizer's own floor at every size** — the 8B cell is the furthest under the floor's Σ (−3.6%, see below) | **token agreement only** (no logits surface) |
| **S600** | ✅ **at the quantizer's own floor at 0.6B `w8`** (973 ≈ 995) **and above it at 1.7B `w8`** (Σ **1174** vs floor **956**) **and 4B `w8`** (Σ **1656** vs floor **1552**) — the first cells at or above the floor at 1.7B and 4B; the provenance gap vs the shipped graphs still holds at every size (cause not yet identified); **the `w8` lever falls short at 8B** — built, fits (54.2%), Σ **1619** (best of any 8B graph) but still **below the floor's 1901**, so the size stays `w4` on deployability | **token agreement only** (`xlm` returns text, no logits, no seed) |

**310B.** Against f16 the board reaches Σ prefix **1078 / 939 / 1620 / 1832** (0.6B / 1.7B / 4B / 8B)
where the `q4_k_m` checkpoint's own distance from truth is the floor **995 / 956 / 1552 / 1901** —
ratios **1.08 / 0.98 / 1.04 / 0.96**. Through 4B the board's agreement with near-lossless weights is
*as good as the quantized checkpoint's own*; the new 8B cell lands furthest under the floor (Σ 1832
vs 1901, −3.6%), and its row-level shape — 35 of 50 prompts part from truth at the *same* token as
the checkpoint does, and the −69 net is eight early splits against seven later ones — reads as
board arithmetic scatter, not a lost path. This is the token-level confirmation that the faithful
`q4_k`/`q6_k` decode + cube is what actually executes: the W4A16 op's **~1.2e-1** requantization
error is real but is **not the path run** (it is env-gated off at `backend.cpp:762`). Full section:
[ascend_310b.md](ascend_310b.md#end-to-end-accuracy-the-310b-sits-at-the-quantizers-own-floor).

**S600.** Against f16 the provenance gap holds at every size — shipped beats ours on total shared
prefix (**1327 vs 973**, **724 vs 394**, **990 vs 536**, **1196 vs 729**). The `q4_k_m`-vs-f16 floor
*grows with the model* and overtakes **both** graphs from 1.7B up, so a GGML k-quant is a near-truth
reference only at 0.6B. The gap is **real against truth, and its cause is not yet identified.** The
first candidate — the compile-time calibration corpus, which defaults to a set with ~0 code and ~0 CJK
— was **tested and falsified**: a corpus matched to the parity categories moved the KV-cache scales
*away* from the vendor's on 44 of 49 tensors (median distance to the vendor's scales: **1.8%** for the
default corpus, **16.4%** for the matched one), so the default calibration was already vendor-matched
and calibration is not the lever. The **cache** has since been ruled out too: at the 1.7B, our own
`cache_4096` build scored against the vendor graph at the *same* cache trails it **377 vs 724** against
f16 — less than half — and lifting our cache from 1024 to 4096 moved nothing (**394 → 377**), so the
confound is removed and the gap is the build, not the cache. (The cache remains the lever for the 1.7B
`France?` *runaway loop*, a separate axis — the deeper cache closes the loop but not the corpus gap.)
With the KV scales near-identical and the weights data-free-identical
(`w_max/7`), the remaining candidates are the compiler graph's operator rounding and accumulation
order, or a vendor weight-quant scheme that is not data-free. See
[the native compile chain](s600_native_chain.md) and the S600 model page for the record.

**One axis now reaches the floor, and it is `w8`.** Our own 1.7B `w8` build — one flag
changed against our 1.7B `w4` `cache_1024` graph (`--w_bits 4 → 8`) — scores Σ **1174** against f16
truth where the size's floor is **956**: the first cell at 1.7B at or above the floor (ours `w4`
**394**, the vendor's `w4` `cache_4096` **724**), and its confident-wrong tail collapses with it
(23 wide-margin splits → **2**). The 4B follows it: our 4B `w8`
build scores Σ **1656** against f16 truth where that size's floor is **1552** (+104, +6.7%), with **0**
wide-margin splits. **The 8B is where the lever falls short.** The 8B `w8` graph was built and it
**fits** — 9.75 GiB at 54.2% of the `bpu_first` pool (its `cache_2048` arm 10.46 GiB / 58.1%), so the
earlier "`w8` doubles the weights past the pool" expectation was wrong — and its accuracy is the
highest Σ any 8B graph on this page reaches: **Σ 1619 against the floor's 1901** (+890 over our own
8B `w4`'s 729). It is still **below the floor**, and its long-generation battery fails a different
way (a `1..200` list closes, a `1..400` one does not), so **the S600 stays `w4` at 8B on
deployability grounds, and *that* row — not the `w8` one — is the size's lowest-Σ arm.** This makes the
`w8` reaching-the-floor claim a **0.6B/1.7B/4B statement, not a fleet-wide one**. Full record, with
build provenance and scope notes:
[s600_qwen3.md](../models/s600_qwen3.md#the-17b-at-w8-above-the-floor-against-truth) and
[the 8B at `w8`](../models/s600_qwen3.md#the-8b-at-w8-the-width-lever-falls-short).

## Performance

### S600 — decode t/s (three runs, <0.8% spread)

| Model | Prefill (t/s) | Decode (t/s) | Load | Pool held |
|---|---|---|---|---|
| Qwen3-0.6B w8 (shipped) | 6169–7111 | **88.3** | 5.26 s | 27.9% |
| Qwen3-1.7B w4 (shipped) | 5172–5818 | **70.2** | 5.70 s | 34.8% |
| Qwen3-4B w4 (shipped) | 2415–2573 | **42.8** | 6.44 s | 62.3% |
| Qwen3-8B w4 (shipped) | 1939–2040 | **29.7** | 7.66 s | **87.7%** |
| Qwen3-0.6B w8 (ours c1024) | 11636–13128 | **121.6** | 5.31 s | 15.9% |
| Qwen3-1.7B w4 (ours c1024) | 8828–9846 | **89.1** | 5.62 s | 22.7% |
| Qwen3-1.7B w8 (ours c1024) | 8127–9143 | **65.4** | 6.05 s | 29.3% |
| Qwen3-4B w4 (ours c1024) | 4339–4571 | **49.9** | 6.35 s | 39.6% |
| Qwen3-4B w8 (ours c1024) | 4163–4491 | **33.1** | 7.34 s | 56.6% |
| Qwen3-8B w8 (ours c1024) | 2909–3030 | **20.5** | 9.41 s | **54.2%** (`bpu_first`) |
| Qwen3-8B w8 (ours c2048) | 2580–2681 | **20.2** | 9.40 s | **58.1%** (`bpu_first`) |

Our own `cache_1024` `w4` builds sit **above their shipped twins on both axes** — faster and smaller
in the pool. The `w8` rebuilds keep the pool edge but not the rate one: `w8` doubles the packed
weights, so they read slower per token than the `w4` graph of the same size while still beating the
shipped graph's pool footprint ([s600_qwen3.md](../models/s600_qwen3.md#measured-ladder)).
The `cache_2048` 8B `w8` row is the width arm's one negative: a deeper cache that **fits and does not
help** — decode 20.5 → 20.2 t/s, pool 54.2% → 58.1%, so no `cache_4096` build was spent and the 8B
`w8` deployability question is **closed** (see
[s600_qwen3.md](../models/s600_qwen3.md#the-8b-w8-at-cache_2048-the-cache-closes-the-short-list-not-the-long-one)).
The 8B is this board's ceiling.

### Orange Pi — Ascend 310B (marginal decode, `--steps 8` vs `32`)

| Model | Prefill | Decode | One-time plane build | `--steps 8` wall |
|---|---|---|---|---|
| Qwen3-0.6B | 0.065 s/token | **0.129 s/token** (7.8 t/s) | 2.1 s | 12.8 s ⚠ |
| Qwen3-1.7B | 0.065 s/token | **0.39 s/token** (2.6 t/s) | 5.4 s | 18.9 s |
| Qwen3-4B | 0.129 s/token | **0.640 s/token** (1.6 t/s) | 14.4 s | 41.2 s |
| Qwen3-8B | — | **1.55 s/token** (0.65 t/s; release on) | — | — |

⚠ The 0.6B wall sits near a band edge of the process wall's ~3.07 s quantization and can read 9.7 or
12.8 s for the same work; the plane-build column is the reproducible one. See
[the quantized wall](ascend_310b.md#the-process-wall-is-quantized-in-307-s-bands).

The plane build was **4–8× faster** as of 2026-10-14 (was ~24.2 / ~85.6 / ~202 / ~430 s): the weights
are now decoded a whole super-block at a time, the plane is transposed with the A55's `vcvt_f16_f32`,
and both loops are threaded across the board's four cores. The transpose was then rewritten to do its
8×8 block **in the vector registers** (three-stage `vtrnq` butterfly, one 16-byte store per
destination row), which took the whole prologue another 1.5× down (3.3 → 2.1 s at 0.6B, 8.1 → 5.4 s at
1.7B, 21.7 → 14.4 s at 4B). The `--steps 8` wall roughly halved-to-thirded at every size across both
steps (28.4 → 10.0 → 12.8 s, 92.9 → 19.2 → 18.9 s, 228.5 → 43.8 → 41.2 s), byte-identical output. See
[ascend_310b.md](ascend_310b.md#current-state-the-canonical-numbers-after-the-series). The wall is now
split between the prologue, decode, and a fixed ~3.3 s ACL process-exit tax rather than being almost
entirely the prologue. Decode is still
the throughput question, and batching is its lever: aggregate 1.42 → 8.16 t/s over B=1…16, with a step of
`0.1378 + 0.0641·(B−1)` s. The bound is
**interior m=B kernel work, not op-launch count**, which is why the fused attention kernel was
declined — see [the batched-decode bound](ascend_310b_batch_decode_bound.md) and
[ascend_310b.md](ascend_310b.md#current-state-the-canonical-numbers-after-the-series).

## One line each

| Device | Accuracy | Performance |
|---|---|---|
| **2080ti** | the oracle — token-exact vs llama.cpp | CPU **ahead** of llama.cpp on both halves (prefill 1.037×, decode 1.37×); CUDA prefill still a tenth of llama.cpp's, decode 0.404× |
| **S600** | ✅ **at the floor at 0.6B `w8`, above it at 1.7B `w8`** (Σ **1174** vs floor 956) **and 4B `w8`** (Σ **1656** vs floor 1552); `w4` builds still trail the vendor; **8B stays `w4`** — below the floor (its `w8` arm reaches Σ 1619, best of the size, but still short) | 88 t/s @0.6B → 30 t/s @8B; our compiled graphs beat the shipped ones (the `w8` rebuilds keep the pool edge, not the rate one) |
| **310B** | ✅ **at the floor at every size (0.6B–8B measured)** — the goal is met on this board, with the 8B the furthest under the floor's Σ (−3.6%) | 7.8 → 1.6 t/s; batching lifts aggregate to ~8 t/s; 8B now fits: 0.65 t/s, no swap in the fit probe; 0.53 GiB touched once in the 4 h accuracy capture |

## Scope

This page summarizes; it does not re-measure. The authority for each row is the page named beside it
([the C engine](c_engine.md), [Qwen3 on the S600](../models/s600_qwen3.md),
[ascend_310b](ascend_310b.md)), and a number that moves there should move here in the same commit.
Two honest limits carry across the table: neither board surface exposes **logits**, so "aligned to
1e-3" is expressible on them **only as token agreement**; and the boards differ in **weight scheme**
(`leap_llm` int4 on the S600, GGUF `q4_k_m` on the 310B), so a cross-board token-agreement number is a
*quantizer-provenance* comparison, not a ranking of the hardware.
