# Filling m on the 310B — two levers, and the one that pays

[The m=1 rate is the shape, not the device](ascend_310b.md#the-m1-rate-is-the-shape-not-the-device-so-batching-is-the-lever)
(#599, #600) closed the 310B host-side question: a decode step runs at **~18 GFLOP/s** because every
GEMV in it presents an **m = 1** activation, the cube's M dimension sits idle, and the *same weights*
do **~108 GFLOP/s at m = 8** and **~230 GFLOP/s at m ≥ 16**. Launch count and sync placement were
measured and moved nothing; the device is fine; the *shape* is the limit. That leaves exactly one
lever — **put more tokens on the cube at once** — and it has exactly two forms. This page decides
between them. It is a scoping document: it changes no code, and its recommendation is **do neither**
— *on cost and constraint, not payoff*. [The bound is now measured](#the-bound-measured-and-it-is-not-the-3-the-model-above-derived):
batch serving buys **3.7× per-sequence at k=4 and 6.6× at k=8**, well past the level the opening
model suggested.

The gap, in one sentence: **a decode step is one token, so it is 36 layers plus a head of m=1 GEMMs,
and there is no way to widen m within one request's one token** — the tokens have to come from
somewhere, and there are only two places: *other requests* (batch serving) or *a draft's guesses*
(speculative decoding).

## What is actually m = 1, and why it is not the whole step

Before either lever is priced, the honest bound has to be stated, because "up to 5.7×" (19 → 108) and
"up to 12×" (19 → 230) are pure-GEMM ceilings and neither is the end-to-end win.

The 4B decode profile ([where a 4B decode step goes](ascend_310b.md#where-a-4b-decode-step-actually-goes-and-why-nothing-was-changed))
splits the step as:

| band | share | is it a cube GEMM? |
|---|---|---|
| `op_sync` — the NPU's own work, dominated by the cube | **85.5%** | **mostly** — the GEMMs plus the attention kernel |
| `to_f16` / `to_f32` — the graph's per-op f32↔fp16 convert | **9.7%** | no — per-op activation work |
| `sdma_down` / `sdma_up` — the chunk read-backs and uploads | **3.0%** | no — but the read-backs shrink with a wider m (see below) |
| `op_enqueue` + `aclTensor` glue | **1.9%** | no — host, and *grows* with a batch |

The one number the two sections above do **not** give is the split of `op_sync` between the GEMVs and
`AttentionStepCustom`. It is not measured, and it matters here: a batch of B tokens shares **one**
weight pass per layer, so the GEMM cost per token falls roughly B-fold between m=1 and m=8, while
attention's *weight* traffic falls B-fold too but its *per-token score* work does not. A rough model
of the op (the attention kernel's arithmetic is `q_len × span × d` per head, against the GEMMs'
`2 × params` per token) puts attention well under a fifth of `op_sync` at the 4B's short context, so
the batchable fraction is on the order of **0.85–0.9 of the step**. That is a derived bound, not a
measurement, and it is marked as such.

**The end-to-end bounds, then.** If a batch of B lifts the GEMM half from the m=1 rate to the rate at
m = B — the m-scale table's own numbers, 19 → 108 at B=8 — then:

- the pure-GEMM ceiling at B=8 is `1 / (0.1 + 0.9/5.7) = 3.4×` (and `3.8×` at B=16, `4.0×` at B=32);
- **accounting for the fixed per-op work** (the 14.5% that is not `op_sync`, which at most amortizes
  rather than scales), the realistic end-to-end ceiling at B=8 is **≈ 2.9×**.

Say **`< 3×` at B=8, `< 4×` at B=32**, not 5.7×. And that ceiling assumes every one of those fractions
holds a per-token constant and only the GEMM band moves — the most favorable reading. A measured 4B
step of 0.587 s/token would land near **0.20 s/token** if the whole GEMM band took the m=8 rate.

**That model is superseded below.** [The measured bound](#the-bound-measured-and-it-is-not-the-3-the-model-above-derived)
drives the same question through `pocketllm-mscale` on the 4B's five real drive shapes and the decode
profile's per-step split, and it lands at **3.7× at k=4 and 6.6× at k=8** — larger than this model's
`< 3×` at 8, because the model used a single fixed shape for the m-rate and the whole-step 14.5% for
the floor. Read this section as the derivation the measurement later corrected, not as the number.

One correction worth recording, because it is tempting and wrong: **the chunked read-back does not
divide by B.** `run_cube_weight` walks N in 8192-column chunks and reads back a partial sum per
chunk (`sdma_down`, 2.2%), and a larger m means `m × 8192` floats per chunk instead of `1 × 8192` —
so those bytes grow with B even as the per-token share of them falls. The read-back stops being a
bottleneck (0.4×/token at B=8) without disappearing from the step.

## The bound, measured — and it is not the `< 3×` the model above derived

The ceiling above was **asserted from a model**, and the page said so: the fraction of the step that
scales with m was a derived `0.85–0.9`, the m-rate curve was `m=1 → 19 GFLOP/s` while a decode step
actually runs at **~17 GFLOP/s** (the m-scale table's own m=8 at a *fixed* shape is not the *weighted*
rate over the 4B's five real drive shapes), and the per-op floor was assumed to scale with m. Each of
those is worth more than a factor of one, and together they move the answer. Re-measured on the board,
the honest end-to-end bound at **k=4 is 3.7× and at k=8 is 6.6×** — well above the `< 3×`-at-8 the
model gave — and the per-token rate *rises with m when weighted correctly*, which is the direction the
fixed-shape table obscured.

### The weighted GEMM curve — the 4B's real shapes, not a proxy

The m-scale table uses **one fixed (n, k) per row**. A 4B decode step drives **five** distinct shapes,
and the m-rate is not uniform across them: the wide-`n` shapes (the head at `n = 151936`, the ffn at
`n = 9728`) reach ~100–106 GFLOP/s at m=8 while the square `q/k/v/o` shape (`n = k = 2560`) only
reaches ~108 by m=16. So the right measurement is the *sum over the step's drives* at each m, from
`pocketllm-mscale` on exactly the shapes `qwen3.cpp` runs:

| drive (the 4B's own) | per layer | m=1 | m=2 | m=4 | m=8 | m=16 | m=32 |
|---|---|---|---|---|---|---|---|
| `q/k/v/o`, n=k=2560 | ×4 | 0.692 | 0.729 | 0.812 | 0.966 | 1.244 | 1.949 |
| `ffn_gate`/`up`, n=9728, k=2560 | ×2 | 3.246 | 3.324 | 3.521 | 3.884 | 4.602 | 6.829 |
| `ffn_down`, n=2560, k=9728 | ×1 | 2.344 | 2.415 | 2.614 | 2.938 | 3.688 | 6.120 |
| `output` head, n=151936 | ×1 | 52.845 | 51.839 | 54.000 | 58.797 | 68.206 | 99.871 |
| **step total (ms, all 36 layers + head)** | | **470.6** | **483.1** | **518.5** | **583.3** | **711.5** | **1092.5** |

The per-call figures are `pocketllm-mscale --n … --k … --rep 20`; the step total applies the call
counts (36×4 + 36×2 + 36×1 + 1). Two facts fall out and both matter:

- **The measured m=1 total, 470.6 ms, is the cube's own share of `op_sync`** — against the step's
  ~8.04 GFLOP its rate is **17.1 GFLOP/s**, the decode profile's ~17 GFLOP/s, which ties the synthetic
  harness to the real step. `op_sync` measured 452.8 ms, so the cube's GEMMs *are* essentially all of
  it; the ~18 ms of mesh is `AttentionStepCustom` **and** `rope_neox` (a cube drive too — it fell
  outside `gemm_quant`'s op scope, so it is in the gap, not in the cube total).
- **`r(k) = GEMM(k)/GEMM(1)` is `1.027` at k=2, `1.102` at k=4, `1.240` at k=8.** The GEMM time is
  *nearly flat* from m=1 to m=4 and only 24% higher at m=8 — so the per-*token* GEMM cost divides
  almost exactly by k, which is the whole lever, and it is larger than the fixed-shape model implied.

### The split at 4B, measured

The page's split above is `op_sync` 85.5% vs the rest 14.5%. That is right but it is a *whole-step*
average; what the bound needs is the same split **per decode step**, and the per-op dump gives it. The
`--steps 1` and `--steps 8` runs differ by exactly seven decode steps (they share the one-time prologue
and the 5-token prefill, as the [4B profile](ascend_310b.md#where-a-4b-decode-step-actually-goes-and-why-nothing-was-changed)
establishes), so the difference over seven **is** a decode step:

| stage | per decode step | share | scales with m? |
|---|---|---|---|
| **the cube** (the weighted GEMM table above, m=1) | **470 ms** | **88.2%** | **yes — the lever** |
| `to_f16` + `to_f32` (per-op activation convert) | **~42 ms** | 7.9% (14.5% of the activation half) | no |
| `rope_neox` + `attention` + `rms_norm` + `silu_mul` (the mesh) | **~18 ms** | 3.4% (11.5%) | no |
| `sdma_up` / `sdma_down` / `op_enqueue` / glue | **~3–10 ms** | ~2% | no (the transfer is tiny here) |

**The cube's GEMMs are 88% of a 4B decode step; the non-GEMM floor is ~77 ms.** That is a *stronger*
case than the 85.5/14.5 the page modelled, because the ~85% `op_sync` figure conflated the cube with
the attention/rope mesh. The one caution this split carries: the `to_f16`/`sdma` figures are
*activation-side*, and a real batch's larger activations convert and transfer proportionally more, so
a full m-fold growth of that band would roughly triple it per step — but the measurement above bounds
how much of the step it can ever be, and at m=1 it is under 8%.

### The bound — k ∈ {2, 4, 8}

`T(k) = 470.6 · r(k) + 77.2` ms per step for **all k sequences together**; the per-sequence wall is
that over k. `r(k)` is the table above.

| k | GEMM(k) ms | non-GEMM ms | **aggregate step** (all k) | **per-sequence step** | **per-seq speed-up** | aggregate tok/s |
|---|---|---|---|---|---|---|
| **1** | 470.6 | 77.2 | 548 ms | **548 ms/token** | 1.00× | 1.8 |
| **2** | 483.1 | 77.2 | 560 ms | **280 ms/token** | **1.96×** | 3.6 |
| **4** | 518.5 | 77.2 | 596 ms | **149 ms/token** | **3.68×** | 6.7 |
| **8** | 583.3 | 77.2 | 661 ms | **83 ms/token** | **6.64×** | 12.1 |
| 16 | 711.5 | 77.2 | 789 ms | 49 ms/token | 11.2× | 20.3 |
| 32 | 1092.5 | 77.2 | 1170 ms | 37 ms/token | 15.0× | 27.4 |

Two caveats, and neither is small enough to ignore. First, **m=1 here is 548 ms where the page's
canonical decode is 587 ms** — the 39 ms gap is real (a `--steps 8/32` marginal is measured deeper
into the sequence, where attention's window is longer) and it means these are ratios, anchored to the
profile's own numbers, not a promise about the canonical marginal. Second, **the ~14.5% floor is
modelled as fixed**, which is the *conservative* direction: the k-fold growth of `to_f16`/`sdma` sits
*inside* that floor and at m=1 is under 8% of the step. Third, KV residency: `per_layer` at 4096
positions is 2 MiB/layer, so 36 layers is **72 MiB per sequence** — at k=4 that is 288 MiB added to
the 4B's **~8.2 GiB** residency, i.e. ~3.5%, which the ~16.7 GB usable pool absorbs.

### The decision — the bound justifies batch serving, and the page's "do neither" must move

**There is a k where per-sequence decode is meaningfully faster, well past the 1.3× bar:** k=2 gives
1.96×, k=4 gives 3.68× and k=8 gives 6.64× per-sequence. The page's own derived model said `< 3×` at
B=8; the measured bound is **~6.6× at B=8**, and at **k=4 it is already 3.7×** — the same number the
model put at B=8. The model was wrong in the *useful* direction because it under-credited two things:
the GEMM's *weighted* rate rises more with m than a single fixed shape showed, and the 4B step's
per-op floor is a smaller fraction of the step than the `op_sync`-based 14.5% suggested. So the honest
reading is the opposite of the page's summary: **(a)'s payoff is not "short of 2×" observational, it is
3.7× at four sequences and 6.6× at eight.**

That does **not** reverse the page's *recommendation*, and the distinction is the point of this
section. The payoff was never the reason to say neither — the page said so itself
("the decision not to do it is purely about cost and constraint, not payoff"). What the measurement
changes is the *weight* on the trade: the page priced a ~2.9×-at-B=8 feature and got a lean against
it; the real feature is ~6.6× at B=8, which changes the calculus for anyone with a real multi-request
workload. So the honest conclusion is:

- **The bound justifies batch serving** — decide for it if a multi-request serving workload is the
  goal, because 3.7×/6.6× on a single card is a feature, not a rounding error.
- **The cost is unchanged and still the deciding factor** — a new batched ragged-window `ascend310b`
  attention kernel in the adjacent tree, the KV-slab rewrite, batch `Session`/`native`/adapter APIs
  and a scheduler, all to preserve the `supports_batch=False` correctness contract whose violation is
  silently wrong text (measured 0/6). None of that gets cheaper because the payoff is bigger.
- **The minimal sketch** (a design, not an implementation) is the page's own PR list, and it stays
  right: (1) the batched ragged `AttentionStepCustom` for `ascend310b` — the gating item and the only
  board-side one; (2) `qwen3.cpp`'s KV slab to `[layer][sequence][position][kv_width]`; (3) `Session`
  + `native.py` N positions and a batch `forward`, B=1 preserved so the gate is unchanged; (4)
  `native_backend.py`'s scheduler with `supports_batch` flipped true only when real per-sequence
  isolation replaces the lock. Its size is the page's own estimate: **two repositories, four PRs, and
  a new 310B kernel** — the same feature, now with a measured 6.6× to justify it.

The one-sentence verdict: **the lever is worth 3.7× at four sequences and 6.6× at eight — measured, not
modelled — so if 310B serving throughput becomes a real goal, build (a); nothing about the cost
changed, only the size of the prize.**

## Option (a) — batch serving

Several requests' decode steps share one forward pass, so the cube sees **m = (live requests)** at
once. This is the lever that *directly* fills the dimension the m-scale table measured, and it is the
one the last two sections named. It is also the one that collides with the whole shape of this tree.

### What has to change

**The KV cache and the position — the heart of it.** `src/model/qwen3.cpp` allocates
`k_cache_`/`v_cache_` as **one flat arena of `n_layer_ × cache_capacity_ × n_head_kv_ × head_dim_`**,
addressed by two scalars that are *per-session, not per-sequence*: `cache_capacity_` (the slab pitch)
and `position_`/`cache_length_` (the KV length, `src/model/qwen3.cpp:644` and `src/model/qwen3.cpp:738`).
Both the append (`kv_append` into `layer_k` at `start_pos`) and the attention window (`first_key=0`,
`start_pos` as the span, `src/model/qwen3.cpp:718`) are functions of that single scalar. **A batch
needs N caches and N positions**, and the slab layout has to change from `[layer][position][kv_width]`
to `[layer][sequence][position][kv_width]` — a pitch change that touches every element offset in
`forward`, including the per-chunk `DeviceBuffer{base + offset, bytes}` slices the chunked cube path
already builds.

**`AttentionStepCustom` is the hard floor, and it is not a host change.** The op the 310B backend
drives for attention takes `q_len == 1` per sequence today (the Ascend backend loops a prefill as
`q_len` decode steps, [as built in the 4B/8B round](ascend_310b.md)). Batching decode needs one call
to attend **B queries, each to its own visible window** — a *ragged* cone: request 1 may be at
position 40 and request 2 at 300, so there is no single `first_key`/`span` that describes both. The
custom op has no batched, per-sequence-span entry point today. **This has to be built in the adjacent
`minicpm-o-4.5-orangepi` tree as a new AscendC op**, and it has to be built for `ascend310b`
specifically — which is exactly the class of thing that, in the custom-ops record, had to be written
from scratch because the built-in `aclnnMm` ships no 310B kernel. A new 310B custom kernel is not a
tuning task; it is the reason this option is expensive.

**`Session` and the bridge.** `src/runtime/session.h` owns **one** `position_`, documented as "the KV
cache's length… one per `Session`", and one `Qwen3Model`. Batching needs N positions and a scheduler
that decides, each step, which sequences are live, how many token slots exist, and how they pack into
one m. The ctypes bridge (`python/pocketllm/native.py`) and the serving adapter
(`python/pocketllm/server/native_backend.py`) each expose one session with one sequence; both grow a
batch dimension, and the adapter's per-request decode loop becomes a *scheduler loop*. `future_mask`
/ position bookkeeping (injectable per sequence) has to thread through every `forward`.

**The executor is nearly free.** The op-by-op walk does not care that m = B instead of 1; `matmul`
already takes a batch dimension. This is the one part of the stack that is *not* the cost.

### What it buys, and what it breaks

It buys the bound above: **3.7× per-sequence at k=4 and 6.6× at k=8** (measured; see
[the measured bound](#the-bound-measured-and-it-is-not-the-3-the-model-above-derived)), and it is
*architecturally* the right
lever — it provably fills m, which is the measured constraint. The decision not to do it is purely
about cost and constraint, not payoff.

What it **breaks**, and the first two are not negotiable:

1. **`supports_batch = False` is a correctness contract, not a placeholder.** [Serving from the C
   engine](serving.md) records the measurement: six concurrent requests on one session give **6/6
   correct with the lock and 0/6 without — and no error either way.** The failure is silently wrong
   text with a 200. Batching does not remove the lock to *gain* overlap; it replaces the lock with
   *real* per-sequence state. That is the only way to keep the guarantee, and it is a real design.
2. **One process owns one device.** Batching keeps this — it is N sequences on *one* card, not a
   split — but it is new state (N caches, N positions, a scheduler) inside a runtime whose documented
   shape is `one Session with one position_ and one KV cache`. This is the rule being *stretched to
   its limit*, not violated; the page should say so rather than pretend.
3. **Memory.** The 4B already banks **~8.2 GiB** of device residency against a **~16.7 GB** usable
   pool (23.73 GB minus the driver's ~7 GB carve-out). Each extra sequence adds a KV cache — small
   against 8.2 GiB at short context, but the *scheduler's* whole point is many sequences, and at the
   8B's **~15.3 GiB** the board is already at the edge. Batch serving is a feature for the *small*
   checkpoints, and its memory math must be stated as such.
4. **The accuracy gate.** The token-identical-to-CPU gate must still hold per sequence, now under
   ragged windows and shared weight passes. Attention masked at the wrong span produces wrong text
   *silently*, the same failure class as the missing lock.
5. **The cost is spread across the tree.** A `minicpm` AscendC op, `qwen3.cpp` slabs and windows,
   `Session`/`native`/adapter batch APIs, and a scheduler — several PRs, two repositories, and a new
   board-side kernel with the `161001`-class environment risks every custom op here has carried.

## Option (b) — speculative decoding

A cheap draft proposes the next k tokens; the target verifies all k in **one m = k pass**, so the
expensive model does *one* m=k forward per k accepted tokens instead of k m=1 forwards. It fills m
**within a single request** — the serving contract is untouched — and for greedy decoding it is
**exactly output-preserving**. The problem is the draft.

### Why greedy is still token-identical (and what it costs)

Greedy decoding is `argmax` of the target's logits, and the target makes the *final* decision on every
position: the verify pass is the target running at m = k, and each position's argmax is taken from the
target's own logits. A draft token is **accepted only if it equals the target's argmax at that
position**; the branch ends at the first mismatch and the target's argmax there is taken instead. So
the emitted sequence is **bit-identical to what greedy would have produced one token at a time** — no
RNG is involved in greedy at all, and the acceptance rule is a pure function of the target's logits.
**The gate holds.** (The caution is real only for *sampling*: speculative sampling needs the corrected
distribution `p_target` and an acceptance test, and a naive draft-sample-verify changes the output
distribution. This engine's default is greedy and the sampling path is a separate, opt-in surface, so
the honest statement is: **greedy is exactly preserved; sampling would need the rejection-sampling
correction and is not what this option is priced for.**)

### What has to change, and the draft problem

- **`Session`/`Qwen3Model`** grow `forward_batch` / verify (the m=k forward and acceptance loop) —
  the graph already takes an m, so the *verify* is a new caller of existing shapes.
- **`native.py`** grows a speculative decode loop; the CLI's `run` loop and the serving adapter's
  decode loop both learn it. **The serving contract is untouched** — still one request, still
  serialized, still `supports_batch = False`.
- **The draft.** Three candidates, and this is where the option dies on *this* board:
  1. **The same model at a smaller width.** The target here is already **`q4_k_m`** — the bottom of
     the width ladder for a usable checkpoint. There is no smaller width that is still *this model*;
     the ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, and a draft that far down shares almost no
     distribution with the target and proposes almost nothing that verifies.
  2. **A second checkpoint** — a smaller Qwen3 — as **a second `Session` on the same device**. That is
     not "one process owns one device" violated (still one card), but it **doubles the weight
     residency** the board is already at the edge of (8.2 GiB at 4B, 15.3 GiB at 8B), and it adds a
     second plane build — which the profile says is the *dominant* wall (~212–225 s at 4B). On a
     single board whose headline cost is *loading the plane once*, a second checkpoint is the heaviest
     thing you can add.
  3. **A tiny draft on the host.** The host is the machine that spent ~200 s building a plane for the
     target and pages under memory pressure at 8B; a host draft trades an NPU win for host CPU that is
     not the bottleneck.

### What it buys

Speculative decoding's real speedup is `accepted_tokens / (forward_cost at m=k + draft_cost)`. The
m-scale table says the forward at m=k is **cheaper per token** than at m=1 — 19 → 108 GFLOP/s from
m=1 to m=8, so the verify pass at m=8 costs roughly `8 × 1/5.7 ≈ 1.4` m=1-forwards of compute for 8
tokens *if all 8 are accepted*. That is the entire upside, and it is **conditional on acceptance**:
real-world greedy acceptance on a 0.6B/4B pair is typically a fraction of k, the draft costs its own
forwards, and the end-to-end win is `tokens_per_verify / verify_cost` — plausibly **1.5–2.5×**, not
3×, and **zero** if the draft is bad. Nothing here is measured; on a board where the draft's own plane
build is minutes and the draft candidates are all weak, the honest expectation is **low**, and it
would take a real acceptance-rate measurement to raise it.

## The recommendation — neither, and here is the reasoning

**Do neither *as the next thing this tree does* — and note that the measurement above made the case
for (a) stronger, not weaker.** The recommendation stands on *cost and constraint*, which the
measurement did not touch; its payoff is now 3.7×/6.6× rather than the modelled `< 3×`, so the page
is choosing against a *larger* prize than it first priced. That is the honest state: not "the lever is
weak" but "the lever is strong and the price is a new board-side kernel plus a correctness-critical
rewrite, for a board that serves one request at a time today".

- **Batch serving is the right lever and the wrong cost.** It *provably* fills the dimension the
  measurement identified — nothing else in this tree can — but it needs a **new `ascend310b` custom
  attention kernel** (ragged per-sequence windows; today's `AttentionStepCustom` is `q_len==1`), a
  **slab-layout rewrite of the KV cache** from per-session to per-sequence, batch APIs in `Session`,
  `native.py` and the serving adapter, and a scheduler — spread over two repositories, and against a
  memory pool the 8B already fills. Its payoff (measured **3.7× at k=4, 6.6× at k=8**) is real but it
  is **serving throughput
  for a single-card board that currently serves one request at a time**, and the contract it must
  preserve (`supports_batch = False`, protect-the-lock) is the one whose absence has already produced
  silently wrong text measured 0/6. That is a large, correctness-critical feature for a board whose
  own page calls the 18 GFLOP/s honest and understood.
- **Speculative decoding preserves the serving contract and the greedy gate beautifully** — it is the
  *architecturally* cheaper of the two, touching no KV slab and no custom kernel. **But on this board
  it has no viable draft**: the target is already at the bottom width, a second checkpoint doubles the
  residency the board is near the edge of and adds a second multi-minute plane build, and a host draft
  spends the wrong resource. Its upside is acceptance-conditional and unmeasured, plausibly **1.5–2.5×
  and possibly zero**.
- **The honest statement is the one [the m=1 section](ascend_310b.md#the-m1-rate-is-the-shape-not-the-device-so-batching-is-the-lever)
  already made: this is a shape limit, not a device limit, and a single-card engine that serves one
  request at a time has nothing to fill the shape with.** That is a complete answer, not a deferred
  one.

### If one is ever done, it is (a), in this order

The recommendation is not "never". If 310B throughput becomes the priority — most plausibly *because*
a real multi-request serving workload appears on this board — **batch serving is the one to build**,
and it decomposes into these PRs. It should not be started until that workload exists.

1. **`minicpm-o-4.5-orangepi`: a batched, ragged-window `AttentionStepCustom` for `ascend310b`.**
   The gating item and the only board-side one. One custom op: B queries, B `(first_key, span)`
   pairs, one call. Without it there is no batched decode, and its `161001`-class env trap is the
   known risk.
2. **`qwen3.cpp`: the KV slab to `[layer][sequence][position][kv_width]`**, and `forward` to accept a
   per-sequence position/span vector. The pitch change touches every offset in the file; the
   chunked-cube `DeviceBuffer` slices inherit it.
3. **`Session` + `native.py`: N positions and a batch `forward`**, keeping one process / one device
   and the existing single-sequence path as the B=1 case so the gate is unchanged at B=1.
4. **`server/native_backend.py`: the scheduler** — live-set selection, token-slot packing, and
   `supports_batch` flipped `True` **only** once the per-sequence state is what makes it safe (i.e.
   the lock is replaced by real isolation, not removed).
5. **The gate:** token-identical to CPU per sequence under ragged windows, and the 6-concurrent-request
   measurement repeated — the 6/6-vs-0/6 check that started this whole constraint.

PRs 1–3 are a self-contained batched decode reachable from `pocketllm run`; PR 4 is what turns it into
serving. That ordering is the point of this page: **(a) is worth doing as a serving feature, and only
then; (b) is not worth doing on this board at any width.**