# Filling m on the 310B — two levers, and the one that pays

[The m=1 rate is the shape, not the device](ascend_310b.md#the-m1-rate-is-the-shape-not-the-device-so-batching-is-the-lever)
(#599, #600) closed the 310B host-side question: a decode step runs at **~18 GFLOP/s** because every
GEMV in it presents an **m = 1** activation, the cube's M dimension sits idle, and the *same weights*
do **~108 GFLOP/s at m = 8** and **~230 GFLOP/s at m ≥ 16**. Launch count and sync placement were
measured and moved nothing; the device is fine; the *shape* is the limit. That leaves exactly one
lever — **put more tokens on the cube at once** — and it has exactly two forms. This page decides
between them. It is a scoping document: it changes no code, and its recommendation is **do neither**.

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

One correction worth recording, because it is tempting and wrong: **the chunked read-back does not
divide by B.** `run_cube_weight` walks N in 8192-column chunks and reads back a partial sum per
chunk (`sdma_down`, 2.2%), and a larger m means `m × 8192` floats per chunk instead of `1 × 8192` —
so those bytes grow with B even as the per-token share of them falls. The read-back stops being a
bottleneck (0.4×/token at B=8) without disappearing from the step.

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

It buys the bound above: **≈ 2.9× at B=8, ~3.8× at B=32**, and it is *architecturally* the right
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

**Do neither. The 310B's ~18 GFLOP/s is a real shape limit, and neither lever clears it cheaply
enough to be the next thing this tree does.**

- **Batch serving is the right lever and the wrong cost.** It *provably* fills the dimension the
  measurement identified — nothing else in this tree can — but it needs a **new `ascend310b` custom
  attention kernel** (ragged per-sequence windows; today's `AttentionStepCustom` is `q_len==1`), a
  **slab-layout rewrite of the KV cache** from per-session to per-sequence, batch APIs in `Session`,
  `native.py` and the serving adapter, and a scheduler — spread over two repositories, and against a
  memory pool the 8B already fills. Its payoff (~2.9× at B=8) is real but it is **serving throughput
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