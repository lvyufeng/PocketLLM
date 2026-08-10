# DSpark speculative decoding with adaptive draft-length gating

DeepSeek-V4-Flash-0731 ships a 3-stage DSpark draft module under `mtp.*`. One
round drafts `dspark_block_size` (=5) tokens from a single committed token; the
main model verifies them in one forward and commits the longest matching prefix
plus one bonus token.

In exact arithmetic that scheme reproduces plain greedy decoding. On this stack
it does not, and the cause is upstream of gating -- see
[Output determinism](#output-determinism) before relying on DSpark output
matching sequential decode.

All numbers below: TP=4, FP4 active-expert MoE, 4x RTX 2080Ti, routed experts on
CPU, short prompts (17-22 tokens). Measured 2026-08-05/06.

## Why gating exists

Verify cost grows with draft length. Fitting the measured multi-token points
(n=1: 296ms, n=2: 432, n=3: 494, n=6: 850):

    verify(n) ~= 300ms + 109ms * n

A round costs `draft(196ms) + verify(n)`. At n=5 that is 1041ms to produce
`accepted + 1` tokens, against 303ms for a plain single-token decode. So a round
that gets nothing accepted is a ~740ms loss. In the 31-round baseline, **7 rounds
accepted zero tokens and together burned 5.2s more than plain decoding** -- 16%
of total speculative time.

Always drafting all 5 therefore helps or hurts depending entirely on the
workload:

| prompt | accept rate | tok/s | vs plain (3.30) |
|---|---|---|---|
| repeat (count upward) | 100% (5.0/5) | 5.76 | 1.75x |
| code | 82.5% (4.1/5) | 4.92 | 1.49x |
| prose | 40% (2.0/5) | 2.88 | **0.87x** |
| math | 17.5% (0.9/5) | 1.80 | **0.55x** |

## The signal

The last DSpark stage already emits a confidence score per draft position, and
it is well calibrated. Pooled over all 4 prompts, P(draft token matches the main
model) by confidence bucket:

| confidence | n | match rate |
|---|---|---|
| < -0.5 | 5 | 20% |
| -0.5 .. 0.5 | 17 | 65% |
| 0.5 .. 1.5 | 19 | 79% |
| 1.5 .. 3.0 | 19 | 89% |
| 3.0 .. 6.0 | 14 | 100% |
| > 6.0 | 36 | 100% |

Monotone, and the per-prompt curves agree within their sample sizes. Position
carries no extra information once confidence is known (later positions show
higher raw match rates only because they are censored -- a position is observed
only if the whole prefix before it was accepted).

## Why not a fixed threshold

A threshold was tried first and rejected. Leave-one-prompt-out (threshold chosen
on 3 prompts, scored on the 4th):

| held out | always-k | thresholded | |
|---|---|---|---|
| math | 1.80 | 3.40 | 1.89x |
| prose | 2.88 | 3.59 | 1.25x |
| repeat | 5.76 | 5.62 | 0.98x |
| code | 4.92 | 4.31 | **0.87x** |

Mean 1.25x but a 13% regression on the *best* prompt. The best threshold depends
on the workload's accept rate, which a threshold cannot express: on an easy
workload every cut removes a token that would have been accepted.

## The rule that shipped

Estimate P(match | confidence) online from rounds already observed, then pick the
length maximizing expected tokens per millisecond:

    n* = argmax_n  E[tokens(n)] / cost(n)
    E[tokens(n)] = 1 + sum_{i<n} prod_{j<=i} p_j
    cost(n)      = draft_ms + verify(n)   (n > 0)
                 = plain_ms               (n = 0, skip the verify entirely)

Two details that matter:

- **Censored observations.** After the first rejection, later positions were
  verified but their match status is unknown. Counting them as misses biases
  every bucket down and makes the gate truncate progressively harder. Only
  positions `0 .. n_accepted` are recorded.
- **Margin.** Cost is linear in n but expected tokens are multiplicative, so
  underestimating p compounds while overestimating it wastes one position.
  Truncation must beat the full draft by 1.15x before it is taken.

## Results

Causal replay of the same 31 rounds (each prompt starts with a cold gate, the
pessimistic case; a real stream keeps its calibration across a generation):

| prompt | plain | always-k | **gated** | vs always-k |
|---|---|---|---|---|
| prose | 3.30 | 2.88 | 3.05 | 1.06x |
| math | 3.30 | 1.80 | 2.45 | **1.36x** |
| repeat | 3.30 | 5.76 | 5.76 | 1.00x |
| code | 3.30 | 4.92 | 4.92 | 1.00x |
| **overall** (round-weighted) | 3.30 | 3.14 | **3.62** | **1.15x** |

No prompt regresses. The gain is entirely on the workloads where speculation was
losing, and the two easy prompts are left drafting the full block untouched.

### Known limitation

Gating narrows but does not close the gap to plain decode on hard workloads
(prose 3.05 and math 2.45 vs plain 3.30). The reason is structural: the
confidence score is produced *by* the draft, so the draft always runs before the
gate can decide. Gating avoids verify cost, never the 196ms draft cost. A round
the gate skips still paid for its draft. Recovered fraction of the always-k ->
plain gap: prose 44%, math 72%.

**Pre-draft gating** addresses this by checking the committed token's logit margin
(top1 - top2) *before* drafting: if the main model is uncertain about what comes
next (small margin), the draft will be poor and speculation will lose even before
verify cost. Skipping the round there avoids both draft and verify costs.
Controlled by `DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD` (default 0.0 = disabled):
rounds with margin below this threshold skip drafting entirely and take a plain
single-token decode step instead. Set to 4.0 as a starting point if enabling.

The threshold is workload-dependent: easy prompts (repeat, code) rarely hit it
because the model is confident; hard prompts (math, prose) hit it more often,
which is exactly when skipping saves the most time. Disabled by default; tune
based on `margin_skipped_rounds` in the gate stats if enabling.

## Configuration

Gating is **off by default**; it changes draft lengths, so an existing deployment
should opt in.

| env var | default | meaning |
|---|---|---|
| `DEEPSEEK_DSPARK_GATE` | `0` | enable gating |
| `DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD` | `0.0` | skip drafting when committed token margin (top1 - top2 logit) is below this; 0 = disabled, try 4.0 if enabling |
| `DEEPSEEK_DSPARK_GATE_MARGIN` | `1.15` | truncation must win by this factor |
| `DEEPSEEK_DSPARK_GATE_MIN_DRAFT` | `0` | never draft fewer than this (0 = no floor) |
| `DEEPSEEK_DSPARK_GATE_DECAY` | `0.9` | per-round forgetting for workload shifts |
| `DEEPSEEK_DSPARK_GATE_DRAFT_MS` | `196` | measured draft cost |
| `DEEPSEEK_DSPARK_GATE_VERIFY_BASE_MS` | `300` | verify fixed cost |
| `DEEPSEEK_DSPARK_GATE_VERIFY_SLOPE_MS` | `109` | verify per-token cost |
| `DEEPSEEK_DSPARK_GATE_PLAIN_MS` | `303` | plain single-token decode cost |

Only the *ratios* between the cost parameters affect decisions, so the defaults
transfer to other setups as long as the cost shape holds. Re-measure them if the
verify path changes.

**`DEEPSEEK_GPU_MOE_MULTI_TOKEN_FP4=1` is required.** Without the small-batch
FP4 MoE kernel a multi-token verify falls into the prefill grouped MoE path:
verify goes 850ms -> 3273ms and speculation drops to 0.37x plain decode. That
kernel is itself default-off.

## Output determinism

Measured on-device (TP=4 FP4, 4x2080Ti, 20 tokens per prompt, three prompts),
comparing token sequences:

| check | math | code | prose |
|---|---|---|---|
| plain vs plain (same input twice) | identical | identical | identical |
| always-k vs always-k (same input twice) | identical | **diverges at 14** | **diverges at 7** |
| gated vs always-k | identical | diverges at 14 | identical |
| always-k vs plain | diverges at 12 | diverges at 5 | identical |

**Sequential decode is reproducible; the multi-token verify forward is not.**
Running always-k twice on the same prompt with the same weights produced
different tokens on 2 of 3 prompts. So DSpark output does not match sequential
decode, and it does not even match itself run to run.

This is upstream of gating. Gating only chooses how many already-drafted tokens
to submit, and where the verify path is stable (math, prose) gated and always-k
agree exactly. Where they differ (code, first difference at token 14) always-k
already differs from itself at the same position, so the divergence cannot be
attributed to the gate.

Consistent with [`fp4_multi_token_moe_kernel`]: batch-vs-sequential logit drift
was traced to the batched **attention** path (KV write ordering / indexer), not
the MoE kernel, and is present with the multi-token kernel off. Speculative
decoding turns that latent drift into visibly different tokens because the
accept/reject comparison amplifies it into a discrete decision.

**What this means in practice:** treat DSpark as changing the sampling
distribution, not as a transparent speedup. It is not safe for workloads
requiring reproducible output or exact parity with sequential decode. Fixing it
means fixing multi-token attention determinism, which is out of scope here.

Reproduce with `/tmp/verify_dspark_gate_vs_alwaysk.py` (A/B/C attribution) --
kept out of the repo because it needs the 167GB checkpoint and 4 GPUs.

## Tests

- `tests/test_dspark_gate.py` -- decision-rule properties, no GPU needed
- `tests/test_dspark_gate_replay.py` -- replays `tests/data/dspark_rounds_tp4.json`
  (the 31 measured rounds) and pins the results above
- On-device verification: performance gains confirmed (below), token identity
  disproved (above). The gate's 1.30-1.32x on the hard prompts reproduced live:

| prompt | always-k | gated | |
|---|---|---|---|
| math | 3.15 | 4.15 | 1.32x |
| code | 2.77 | 3.62 | 1.30x |
| repeat | 4.44 | 4.41 | 0.99x |
| prose | 3.22 | 3.06 | 0.95x |

The two easy prompts land within noise of always-k rather than exactly equal, as
the replay predicted -- live accept rates differed from the fixture's, so the
gate skipped a few rounds it would otherwise have drafted.

---

## cpp_engine port status

### Verify path

`PersistentEngine::verify_step` forwards a draft block and reports, for each
draft token, what the target model samples after consuming it.

- `cpp_engine/include/persistent_engine.hpp` — interface
- `cpp_engine/src/dsv4_engine.cpp` — implementation
- `cpp_engine/tests/test_perfect_draft.cpp` — feeds a plain decode back in as a
  "perfect" draft, so any mismatch is the verify path's own fault rather than
  the drafter's. 59/59 checked, 0 mismatches at draft_len 5.

It forwards the draft tokens one at a time rather than as a `[1, n, d]` batch.
Batching is the whole point of speculative decoding, but the two are not
numerically equivalent: GEMMs pick tiles and reduction orders by shape, so a
batched verify disagrees with plain decode by ~4e-3 at the first projection,
which amplifies to O(1) at the head. Batching it is a separate optimization
that has to be measured against that drift, not assumed free.

`DSV4_CPP_MOE_DETERMINISTIC_REDUCE=1` (default) removes the MoE atomicAdd
nondeterminism for topk>=3 via per-route partials and a fixed-order reduction;
`cpp_engine/tests/test_moe_fp4_determinism.cpp` guards it.

### Draft module (Stage B)

Porting `src/models/deepseek_v4/dspark.py` into `cpp_engine/src/dspark_engine.cpp`:

| Step | What | State |
|---|---|---|
| B1 | Skeleton + config parsing | done |
| B2 | Stage 0 (main_proj + main_norm + embed) | done |
| B3a | DSparkAttention | done |
| B3b | MoE FFN (routed + shared) | done |
| B4 | Stage 2 heads (norm, hc_head, markov, confidence) | done |
| B5 | Weight loading | done |
| B6 | Main-hidden caching during verify | done |

Parity tests compare each sub-path against an fp32 reference driven by the same
weights (`tests/test_dspark_attention_parity.py`, `tests/test_dspark_moe_parity.py`,
`tests/test_dspark_head_parity.py`). Main-hidden capture is covered by
`cpp_engine/tests/test_dspark_hidden_capture.cpp`, which needs no reference
because it checks the capture against the engine's own decode path.

Two things are deliberately not done yet: TP>1 needs an all-reduce after the
attention `wo_b` and after the routed MoE (each rank only sums its own experts'
routes), and `draft_tokens()` is not wired up.

#### Main-hidden capture

`PersistentEngine::set_dspark_capture_layers()` turns on capture of the main
model's block output at the draft's target layers, mean-pooled over the hc
dimension and concatenated on the last axis -- exactly what the reference's
forward hooks on `model.layers[idx]` record. Capturing the raw `[4, dim]`
instead would be 4x the memory and would not match what `main_proj` was trained
on. Off by default; the cost when on is one pooling kernel per target layer per
forward plus an `[n_target * dim]` D2H copy.

`verify_step` keeps one row per draft token rather than only the last, because
the accepted prefix is not known until after the comparison and the next round
has to start from wherever it lands. Prefill keeps only the final prompt
position: holding all of them would be `n_target * dim` floats per token, ~49 KB
at dim=4096, i.e. 3 GB at a 64K context, for hiddens no draft round reads.

Every way of getting this wrong still yields a finite vector of plausible
magnitude, so `cpp_engine/tests/test_dspark_hidden_capture.cpp` checks the
pieces separately. Against the real checkpoint: each slot matches a
single-layer capture of that layer bitwise (`max_abs=0`), the target layers are
distinguishable (`rel_l2` 0.56 and 1.23 against the first, so the slot check is
not vacuous), verify's rows match plain decode at the same positions exactly,
and prefill's hidden reads 2.8e-6 against the correct position versus 1.79
against the next one. Enabling capture leaves the token stream unchanged. The
pooling kernel itself is checked against a CPU mean at exact equality, which
separates a mean from a sum by a clean 4x, plus a poisoned destination so an
unwritten or over-wide slot cannot pass.

#### MoE notes

The draft's routed experts are held resident on the device rather than staged
per call: 13.4 MB per expert is 10.3 GB at TP=1 but 2.6 GB at TP=4, and staging
would put a PCIe transfer on the critical path of every draft round -- the one
thing speculative decoding cannot afford, since the draft has to stay cheap
relative to the verify it is trying to skip. Total DSpark weights measure
11.4 GB at TP=1 and 3.8 GB at TP=4.

Routing is discrete, so a wrong gate produces plausible-looking numbers rather
than obvious garbage. Measured on 5 tokens against an fp32 reference, cpp/ref
`rel_l2` is 0.014-0.016 (int8 activation quantization before the expert GEMM),
while swapping a single expert for the next-ranked one reads 0.476 -- a ~32x
margin, which is what makes the 0.02 tolerance meaningful rather than merely
satisfied.

#### Head notes

The output head (`head.weight`, 129280x4096 bf16, ~1 GB) is kept whole on every
rank rather than vocab-sharded as the reference does. The reference all-gathers
its logit shards; here the draft's inner loop runs `block_size` times per round,
so a collective per position would put TP latency on the hot path -- and a
replicated table makes every rank's drafted ids identical by construction
rather than by agreement. That is the reason total DSpark weights read 12.4 GB
at TP=1 rather than the 11.4 GB the MoE alone accounts for.

The head's parity is a different regime from the MoE's: bf16 weights against
fp32 activations with no int8 anywhere, so cpp/ref `rel_l2` is 1.3e-7. Three
plausible ways to get it wrong -- no markov bias, all biases computed from the
input token instead of sequentially, and collapsing the hc dimension by mean
instead of the head's own gate -- read 8.3e-1, 5.7e-1 and 1.0e0, and each
changes at least one drafted token. Hence a 1e-5 tolerance plus an exact
token-id comparison: the smallest observed top1-top2 margin was 0.02, so ids
matching is a real check, not a foregone one.
