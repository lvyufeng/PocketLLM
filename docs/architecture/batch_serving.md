# Batch serving on the 310B: concrete changes, the honest ceiling, and the smallest version

[Filling m on the 310B](ascend_m_filling.md) (#600) priced the shape limit and chose **neither** lever:
batch serving's payoff is real (measured **3.7× per-sequence at k=4, 6.6× at k=8**) but its cost was a
**new `ascend310b` attention kernel** plus a correctness-critical rewrite, for a board that serves one
request at a time. This page is the engineering companion to that recommendation: it reads the runtime
and the serving adapter for **exactly what a batch changes**, checks whether the new kernel is really
the gate, and prices the smallest version. It is a scoping document — no code, and the "do neither"
verdict is **revised, not overturned**: the work splits into a cheap first step and an expensive
serving layer, and only the second is the old blocker.

## 1. What a batch actually has to change

Four seams, in dependency order. The short version: **`n` already is a batch dimension**; what is
missing is a *second* dimension — one per sequence.

### 1a. The GEMMs are already batched — this is not the work

`Qwen3Model::forward` calls `matmul(..., n, ...)` with `n` = the tokens in the call, so a **prefill is
already a batched forward** (`n` = prompt length) and only per-token decode runs `n = 1`. The
ms-scale table ([m = 1 is the shape](ascend_310b.md#the-m1-rate-is-the-shape-not-the-device-so-batching-is-the-lever))
is the proof that the cube does not care how the rows got there: it does 19 GFLOP/s at m=1 and
107.5 at m=8 on the same call. `rms_norm`, `silu_mul`, `rope_neox`, `gemm`/`gemm_quant` and the head
projection are all shape-agnostic already. **No GEMM kernel changes for a batch.**

### 1b. The KV cache — one arena, two per-session scalars

`qwen3.h` allocates `k_cache_`/`v_cache_` as **one flat arena** whose layout is
`[layer][position][kv_head][head_dim]`, addressed by two scalars that are **per-session, not
per-sequence**:

- `cache_capacity_` — the slab pitch (the row stride is `n_head_kv * head_dim` wide, `cache_capacity_`
  rows deep);
- `cache_length_` / `position_` — the KV length, which is *also* the attention span (the graph passes
  `first_key = 0` and `start_pos` as the span).

Every element offset in `forward` is a function of those. A batch means **N caches and N positions**,
so the layout has to gain a sequence axis: `[layer][sequence][position][kv_head][head_dim]`. That is
a **pitch change**, and it is not confined to an allocation line — it touches the two
`kernel::DeviceBuffer{base + offset, bytes}` slices built per layer, every `layer_offset`/`start_pos`
offset, and the `kv_append` destination the graph hands the backend. `cache_capacity_` can stay
**shared** if the scheduler pads each sequence's slab to the batch's max length (the whole batch then
advances together and no ragged window arises *within a layer*); giving each sequence its own
capacity is the alternative if packing is to be tight.

### 1c. Attention — the one thing that genuinely needs a new shape

Attention is the hard seam because a batch decode is **ragged by construction**. If sequence A is at
position 40 and sequence B at 300, there is no single `q_offset`/span that describes both. The op
signature (`backend.h`) already assumes this: `attention(q, q_len, ..., first_key, q_offset, ...)`
scores **one sequence's contiguous chunk against one window**. Batching it means one call per layer
with **B `(q_offset, span)` pairs**.

The good news, and the reason this page revises #600's cost: **on this board that loop already
exists**. `src/kernel/ascend/backend.cpp` runs a `q_len > 1` prefill as `q_len` calls to the
single-query `AttentionStepCustom` op, each against the window `[0, q_offset + t + 1)` — its own
comment says driving `q_len > 1` directly "would let a position" skip the per-position loop, and that
this is exactly what decode does not need. A batched decode at `q_len = 1` per sequence is **B such
calls per layer**: the **same B kernel-launches** the loop already issues in prefill, just with B
different `(q_offset, span)` pairs instead of one contiguous cone.

So #600's **"new `ascend310b` custom attention kernel — the gating item"** is **off the critical path
for correctness and for a first win.** The kernel is needed only for the *efficiency* half:
`AttentionStepCustom` re-reads B windows from device each call, so the attention read grows with B and
does not amortize. That cost is real at large B and long context (attention's KV read is `span * d` per
sequence), but it is bounded by the same non-GEMM floor this page's ceiling honors — and it is the
difference between the batched-loop win and the new-kernel win, **not** between working and not
working. The new kernel is an optimization on a path that runs without it.

### 1d. Session, the bridge, and the serving adapter

- **`Session`** (`session.h`) owns **one** `position_` (documented as "the KV cache's length … one per
  `Session`"), one `Qwen3Model`, and one `forward`. A batch needs N positions and a `forward` that
  takes per-sequence `(tokens, start_pos, span)`, plus a live-set. The single-sequence path stays as
  the **B=1 case** so the existing identity gate is unchanged.
- **The ABI** (`pocketllm_forward`) and **`native.py`** each expose one sequence. The additive move
  is a *new* entry point — e.g. `pocketllm_forward_batch(session, per-seq token/pos arrays, B,
  int32_t *out_tokens)` returning B greedy ids through the `Session::forward` argmax path that
  [#646](ascend_310b.md) just opened — leaving `pocketllm_forward` exactly as it is, so no existing
  caller or gate moves. `native.py` gains a `next_tokens(list_of_lists) -> list[int]` beside
  `next_token`.
- **`server/native_backend.py`** is where the batch becomes *serving*: the per-request `_run`/`stream`
  loops become **one scheduler loop** that, each step, selects the live set, packs each request's
  next token into a slot, calls the batched forward once, and fans the B results back out. The lock
  that serializes today is **replaced by real per-sequence isolation** — that is the only way to keep
  the `supports_batch` contract, and `supports_batch = True` may be declared **only** once that
  isolation is what makes it true, never before.

## 2. The honest ceiling at m = 2, 4, 16

The curve this uses is `ascend_m_filling.md`'s re-sweep — `ms(m) = 0.657 + 0.0393·m`, a large fixed
tile cost and a small per-row cost, so an m-row GEMM takes **1.06× the time of an m=1 GEMM at m=2,
1.19× at m=4, 1.83× at m=16** while doing m rows of work. The end-to-end model is
`speedup = 1 / (G/R_m + (1 − G))`, with `G` the GEMM share of the m=1 step (0.68 for the 0.6B,
0.855 for the 4B, per the decode profiles) and `R_m = m·ms(1)/ms(m)` the effective GEMM rate-up. On
the **0.6B** (`G = 0.68`), per-sequence:

| m | GEMM rate-up `R_m` | per-sequence speedup | aggregate (full batch) | what dominates the batched step |
|---|---|---|---|---|
| 2 | 1.89× | **1.47×** | 2.9× | the GEMM still reads the full weight per row, and attention has already doubled (two windows) |
| 4 | 3.37× | **1.92×** | 7.7× | the non-GEMM floor is now **61%** of the step — the GEMM's amortization is nearly spent |
| 16 | 8.76× | **2.51×** | 40× | the non-GEMM floor (**80%**) and the per-op enqueue work, which *grows* with B |

At the 4B's `G = 0.855` the same model gives **1.67× at m=2, 2.51× at m=4, 3.39× at m=8** — the
conservative row of `ascend_m_filling.md`'s own table (2.5× at k=4, 3.4× at k=8). That page's
*headline* bound, **3.7× at k=4 and 6.6× at k=8**, is larger because it drove `pocketllm-mscale`
over the 4B's five real weighted shapes rather than one fixed shape, with a different treatment of
the floor; read the two as a conservative model and a measured bound, not a contradiction.

The honest framing, and the one this page was asked to give: the **12× pure-GEMM headroom does not
survive contact**, and it flattens fast. At **m=2** the GEMM amortizes only 1.9× while **attention's
read has already doubled** — the per-sequence win is real but small (**~1.5×**). By **m=4** the
GEMM's amortization is nearly spent: the **non-GEMM floor is 61% of the batched step** and further
batching buys little per sequence. At **m=16** the ceiling is set entirely by the **non-GEMM floor**
(attention + KV traffic + the per-op host round-trips, which *grow* with B) and by the per-sequence
correctness of packing B unequal-length windows — the GEMM is no longer the limit. The
**per-sequence ceilings are ~1.5×, ~1.9×, ~2.5× at m = 2, 4, 16**; because each step serves m
sequences at once, the **aggregate throughput** at a full batch is m× higher again (2.9×, 7.7×,
40× — latency and throughput are different questions and both are real).

## 3. The smallest useful version

**The m=2 case is not a smaller multi-slot cache — it is the *same* multi-slot cache.** Slot 2 is
not a special case of the layout change; it *is* the layout change, entered at N=2. So the honest
answer is: there is **no "just two requests" shortcut** that sidesteps the sequence axis.

But the sequence axis is far cheaper than #600's "new custom kernel" framing implied, because the
**attention kernel is not required** for a correct, already-faster batched decode (section 1c). That
re-prices the smallest version:

- **Stage 1 — batched decode reachable from the engine** (~2 PRs). `qwen3.cpp`: the sequence axis on
  the KV slab and per-sequence `forward`, single-sequence path unchanged at B=1. `Session` + `native.py`:
  N positions and a `next_tokens` batch call. Attention: **loop the existing op, B calls per layer.**
  Landed behind a *new* entry point, so **nothing existing moves** and the 0.6B identity gate is
  re-run at B>1 (ragged windows, shared weight pass) as the acceptance test. This is the minimum that
  pays off, and it pays the whole per-sequence GEMM amortization.
- **Stage 2 — the serving scheduler** (~1 PR). `native_backend.py`: live-set selection, slot packing,
  `supports_batch = True`. This is what turns a batched decode into *batch serving*, and only here
  does the **lock** have to be replaced by real per-sequence state — the correctness-critical part.
- **Stage 3 — the batched attention kernel** (the expensive, other-repo item). Only to stop the B
  loop re-reading windows; it is an **optimization on top of a working path**, and it can be deferred
  until a workload shows the KV read is the bound.

Stages 1–2 are a self-contained batch-serving path and are **one project, not the old two-repo
project**. The `161001`-class custom-op risk that dominated #600's estimate attaches to Stage 3
alone.

## 4. Verdict

**Worth doing — as a serving feature, on the small checkpoints, in the Stage 1–2 shape above — and it
is now cheaper than #600 priced it, because the custom kernel was mis-scoped as a precondition.**

- **Cost.** ~2–3 missions: Stage 1 (slab + batch forward + bridge) and Stage 2 (scheduler). Stage 3
  (the board-side kernel) is a separate, larger, deferrable mission in the `minicpm-o-4.5-orangepi`
  tree.
- **Risk — what regresses.** (1) The **attention span** is the subtle one: a wrong `first_key`/span
  in a batched op returns *fluent, silently wrong* text — the same failure class as the missing lock
  (`0/6` measured, 200 status), which is why the per-sequence identity gate at B>1 is the gate and
  not a nicety. (2) **Memory**: the 0.6B is comfortable (weights ~0.4 GiB; the pool is ~16.7 GB
  usable, and a short-context KV slab is a few MB, so many slots fit), but the ceiling has to be
  stated in **batch × context** terms — at long context the KV slabs, not the weights, are what fill
  the pool, and a scheduler that admits sequences by count alone will OOM. (3) The **`supports_batch`
  contract**: it may flip `True` **only** when per-sequence isolation is what makes it true, not when
  it is convenient.
- **Minimal first step.** Stage 1, and specifically: **land the sequence axis on the KV slab with the
  single-sequence path physically unchanged at B=1, and re-run the 0.6B identity gate at B=2 with two
  distinct requests at different positions.** If that gate holds, the rest is scheduling; if it does
  not, the failure is in the one place worth knowing about before any scheduler is written.

**What is *not* worth doing:** removing the lock without per-sequence state (measured to return wrong
text with a 200), or building the Stage 3 kernel before a workload exists to justify it, or batching
for the 4B/8B where device residency is already at the pool's edge.
