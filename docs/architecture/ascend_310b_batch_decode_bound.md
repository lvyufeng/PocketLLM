# The batched-decode bound on the 310B is interior work, not op launches

[Filling m on the 310B](ascend_m_filling.md) priced batch serving and concluded the batched decode needed a
**new `ascend310b` attention kernel**; [Batch serving on the 310B](batch_serving.md) re-priced that kernel as
"stop the B loop re-reading windows"; and the Stage-3 scoping that followed measured the KV re-read at
**~0.4%** of a step and re-attributed the wall to **the B op launches per layer** — 28 layers × B calls to
`AttentionStepCustom`, one per sequence. This page measures that attribution directly and finds it is also
wrong: **the per-sequence cost of a batched decode step is interior work in the batch-sized ops — the m = B
GEMM and the host f32↔f16 conversion and transfer of m = B activations — and the attention launches are ~8%
of it.** The fused kernel does not earn its multi-week cost on this board, and the cheaper batched-primitive
assembly the scoping recommended would remove at most that 8%.

It is a measurement, not a model: it was taken *before* any batched code was written, because the number that
decides whether to write it — what fraction of the per-sequence term is launch count — needs no batched code
to read. That is the point of doing it first: **a launch count is the one cost you cannot batch away by
building the batch**, so if it is not the bound, the batch does not pay.

## 1. What was tested, and why it needed no new code

The mission's hypothesis: the decode step's per-sequence term is the `1 × B` `AttentionStepCustom` calls per
layer, and folding those B launches into ~3 batched ops (`BatchMatMulV2` + `SoftmaxV2` + a mask — all three
registered for `ascend310b`, per the scoping's registry dump) would collapse the step toward the fixed
weight-read floor.

Both halves of that claim are measurable against the **existing** tree, with the per-stage clock the Ascend
backend already carries (`POCKETLLM_ASCEND_PROFILE`, [ascend_310b.md](ascend_310b.md)):

- the batched decode the `serve` path drives is already reachable through the ctypes `Engine`
  (`python/pocketllm/native.py`, `next_tokens`), so the step can be timed at any B with no serving layer and
  no new kernel;
- the profile splits that step into per-op walls and per-stage walls (`to_f16`, `to_f32`, `sdma_up`,
  `sdma_down`, `op_enqueue`, `op_sync`), so the launch term (`op_enqueue` — the host cost of issuing an op,
  which the op's own wall also carries) can be read off separately from the kernel work (`op_sync`).

So the decision is made from a knee-high experiment, not a kernel.

## 2. Method: subtract two runs

The per-stage totals are summed over the whole process, so a raw run carries the **one-time plane build**
(~18.5 s for the 0.6B, `ascend_310b.md`) and the prefill, which are large and B-dependent. Both are removed
by taking **two runs with identical warm-up and prefill that differ only in the number of decode steps** —
`S = 1` and `S = 65` — and dividing the difference by the step count. Every number below is that difference
over the step count: the plane build, the prefill, the tokenizer and the process start cancel exactly, and
what is left is one decode step at that B.

Board and shape, stated because a decode figure does not transfer:

| | |
|---|---|
| device | Orange Pi AIpro 20T, Ascend **310B1**, CANN **8.3.RC2** |
| checkpoint | `Qwen3-0.6B-Q4_K_M` (confirmation: `Qwen3-1.7B-Q4_K_M`) |
| decode | greedy, `batch_kv_window` 2048, fixed **53-token** prompt |
| span measured | positions ~53–118 (0.6B, `S=65`); ~53–85 (1.7B, `S=33`) |
| build | `POCKETLLM_CORE_LIB=src/build/libpocketllm.so`, `libpocketllm.so` at #661 |

## 3. The step is B-linear

Decode-step wall (ms), 0.6B, from the subtract-two-runs difference:

| B | `AttentionStepCustom` calls/step | ms/step |
|---|---|---|
| 1 | 28 | 163.1 |
| 2 | 56 | 247.6 |
| 4 | 112 | 419.6 |
| 8 | 224 | 811.5 |

The per-sequence increment is **84.5, 86.0, 98.0 ms/sequence** (B=1→2, 2→4, 4→8): linear, if anything
mildly superlinear. So the Stage-1/2 scoping's step fit (`0.138 + 0.064·(B−1)`) reproduces — the wall *is* a
fixed floor plus a per-sequence term. The question the fit does not answer, and the one that decides Stage 3,
is what that term *is*.

## 4. The per-sequence term is not the launches (and not the KV read)

Decomposition of the per-sequence slope for the 0.6B, B=1→B=4, from the per-op walls (85.5 ms/sequence
total). Every op's wall is the sum of its kernel work (`op_sync`, where the NPU runs) and its host work
(`op_enqueue`, the two conversions, the transfers):

| term | ms/sequence | share of slope |
|---|---|---|
| `gemm_quant` (the m = B GEMM) | 25.9 | 30% |
| **`attention` (the 1 × B `AttentionStepCustom` loop)** | **6.9** | **8%** |
| `rms_norm` | 2.2 | 3% |
| `rope_neox` | 1.6 | 2% |
| `silu_mul` | 0.8 | 1% |
| `kv_append` | 0.5 | 1% |
| **op walls, subtotal** | **37.9** | **44%** |
| host glue between ops (converts, transfers, per-op aclnn setup) | 47.6 | 56% |

Two readings settle the hypothesis:

- **The launch term is 8%.** The whole `attention` op — which is exactly the `1 × B` `AttentionStepCustom`
  loop, since the graph issues one call per sequence per layer — is **6.9 of 85.5 ms/sequence**. Replacing B
  of those calls with three batched ops removes what is left of that term *after* the batched ops' own kernel
  time, so **at most 8% of the slope**, and plausibly less.
- **The pure-launch cost is under 1%.** `op_enqueue` — the host cost of issuing an op, the cost a launch
  collapse targets directly — slopes at **0.47 ms/sequence (0.6%)**. The clock says the launches are cheap;
  what is expensive is the kernel work each launch sets off.

And the boundary condition that makes this conclusive: **the per-sequence slope depends on context, and a
launch count does not.** Timed at a short span (~12–49 positions) the slope is **52.6 ms/sequence**; at the
measured span (~53–118) it is **92.6 ms/sequence** — the per-sequence cost roughly doubles as the attended
span roughly doubles, while the `AttentionStepCustom` call count (28 × B) is **identical in both**. A term
that scales with the window is not a term that scales with the number of launches.

The 1.7B confirms the shape, with the GEMM's share larger and the attention's smaller (B=1→B=4, 102.7
ms/sequence): `gemm_quant` **46.1 (45%)**, `attention` **5.2 (5%)**, `rms_norm` 3.2, `rope_neox` 1.5,
`silu_mul` 1.4, `kv_append` 0.5.

## 5. What the term actually is, split

The profiler was extended to record a **per-op × per-stage matrix**
(see [ascend_310b.md](ascend_310b.md#the-per-op-stage-matrix-the-profilers-second-table)),
so the residual this section used to leave unattributed is now split. Two views, both measured on the same
0.6B board runs.

**The step** (B=8, plen 53, 829.9 ms/step) splits:

| component | ms/step | share |
|---|---|---|
| kernel — `op_sync`, the NPU running the op | 585.0 | 70.5% |
| `sdma_up`/`sdma_down` — the host transfers | 86.1 | 10.4% |
| `to_f16`/`to_f32` — the host f32↔f16 convert | 58.4 | 7.1% |
| `op_enqueue` — the host cost of the op call | 6.0 | 0.7% |
| unattributed — per-op `aclTensor` setup, `GetWorkspaceSize`, the loops that feed a launch | 90.4 | 10.9% |

**The per-sequence slope** (the term itself, 85.4 ms/sequence at B=1→4) splits:

| component | ms/sequence | share |
|---|---|---|
| kernel | 66.4 | 77.7% |
| `sdma` transfers | 11.4 | 13.3% |
| conversion | 7.3 | 8.5% |
| enqueue | 0.4 | 0.5% |

Ranked per-op host work on the slope: **`attention` 8.8, `gemm_quant` 5.1, `rms_norm` 2.2, `rope_neox` 1.6,
`silu_mul` 0.8, `kv_append` 0.5** ms/sequence. Two readings:

- **The slope is 78% kernel work** — the batch's own arithmetic (the m = B GEMM and the m = B rows through
  the row-wise ops). The kernel cannot remove it; the GEMM's own curve is `0.657 + 0.0393·m`
  ([ascend_m_filling.md](ascend_m_filling.md)), so it grows with m by construction.
- **The largest *removable* host term is `attention`'s, and it is a device transpose, not a round-trip.**
  Its host cost is **context-growing** — 63.9 ms/step at a 53-token context, **155.7 ms/step at 253** — while
  the GEMM's is flat. That is the layout bridge in the file header: `attention` gathers the whole KV window
  `[0, context)` to the host, repacks it token-major → head-major, and sends it back, once per (sequence,
  layer) per step. The #632/#643 pattern ("keep an activation in its own width across a boundary") **does not
  apply**: #643's `prefers_kv_projection_in_cache_dtype` already removed the re-widen, and what is left is a
  transpose, which no width flag removes. The `gemm_quant` conversion (5.1 ms/sequence) is the other
  candidate, and it needs an on-device f32↔f16 cast the board does not ship.

So the "host glue" this page used to leave as a 56% residual is, measured, **a device transpose plus per-op
`aclNN` setup — not a boundary round-trip, and not op-launch count.** No *attention* kernel touches it: the
transpose is a host memory walk, and a fused attention op that reads the cache in place would remove it — but
that is Stage 3, which does not otherwise earn its cost.

## 6. The consequence: Stage 3 does not earn its cost, and the batched assembly removes ≤8%

- **The fused AscendC kernel (Stage 3) is not justified** on the 310B1 / CANN 8.3.RC2 board. Its premise —
  fold B launches into one — targets an **8%** term (the attention op) and a **0.6%** term (the pure enqueue).
  The multi-week cost is authoring and registering a new `ascend310b` AscendC op, packaging a `custom_opp_*.run`
  and wiring it into `backend.cpp`. That cost does not buy the wall.
- **The batched-primitive assembly (`BatchMatMulV2` + `SoftmaxV2` + mask, the scoping's item 2) removes at most
  the same 8%**, and only if its three ops' own kernel time is negligible against the one op they replace —
  which the numbers do not support: a batched `QK^T`/`PV` in fp16 over B windows still moves B windows' worth
  of data. It is a real, cheap experiment, but its ceiling is 8%, not "the wall".
- **The recommendation is to STOP at Stage 1–2 for the 310B.** Batched decode is landed, correct and ~5.8×
  aggregate at B=16; the remaining per-sequence cost is interior work, and the two levers that would move it —
  keeping activations in f16 on the device instead of round-tripping them (item 2 above), and a quantized GEMM
  that does not re-expand m — are both larger, separate, and unrelated to attention.

## 7. This supersedes both earlier framings

Both earlier attributions are corrected here, and neither quietly:

- [#600 / `ascend_m_filling.md`](ascend_m_filling.md) priced the kernel as needed to **stop the B loop re-reading
  windows**. The re-read is real but small — the Stage-3 scoping measured it at ~0.4% of a step — and this page
  shows the whole attention term it belongs to is 8% of the slope. **The KV re-read was not the wall.**
- **The Stage-3 scoping** that followed measured the re-read down and re-attributed the wall to **"the B op
  launches per layer."** This page measures that directly and finds the launches are **8% of the slope as the
  op, 0.6% as the enqueue**, and that the slope *grows with context* where launch count does not. **The op
  launches are not the wall either.** The wall is the m = B work inside the batch, which no attention kernel
  addresses.

What remains true from both: batching pays (5.76× aggregate at B=16 through Stage 1–2), the 4B/8B are not to be
batched (residency at the pool edge), and `161001` is an env trap rather than a numerics risk — it simply never
has to be spent now.

## 8. Reproduce

No new code is needed — the probe is a short script over the existing ctypes engine, and it is the two-run
difference, not the script, that is the method. On the board:

```bash
export POCKETLLM_ASCEND_PROFILE=1 POCKETLLM_CORE_LIB=$REPO/src/build/libpocketllm.so
# for each B and each S in {1, 65}: warm the plane build, reserve(window, B),
# prefill the 53-token prompt, then time S greedy decode steps
python probe.py <B> <S> 2048 53     # prints one JSON line + the per-stage/per-op profile
# per-sequence wall  = (wall(S=65) - wall(S=1)) / 64   -> ms/step, B-linear (see §3)
# per-op and per-stage slopes: the same difference over the profile totals (§4)
```

The script opens the engine with `pocketllm.native.Engine.open(ckpt, "ascend")`, runs one untimed forward to
pay the one-time plane build, `reset`s and `reserve`s, runs the prompt once (prefill), then times `S` calls to
`next_token` (B=1) or `next_tokens` (B>1) with `time.perf_counter()`. Both runs for a given B share the warm-up
and prefill, so their difference is pure decode. `S=1` and `S=65` were used for the B=1/2/4/8 decomposition of
§4; `S=16` for the standalone sweep of §3. The per-sequence slopes are read from the B=1 and B=4 differences;
the B=8 point is the linearity check.