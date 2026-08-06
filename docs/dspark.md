# DSpark speculative decoding with adaptive draft-length gating

DeepSeek-V4-Flash-0731 ships a 3-stage DSpark draft module under `mtp.*`. One
round drafts `dspark_block_size` (=5) tokens from a single committed token; the
main model verifies them in one forward and commits the longest matching prefix
plus one bonus token. Output is identical to plain greedy decoding.

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

Closing it would need a decision made *before* drafting -- e.g. predicting from
the committed token's own entropy whether drafting is worth attempting. Not
implemented.

## Configuration

Gating is **off by default**; it changes draft lengths, so an existing deployment
should opt in.

| env var | default | meaning |
|---|---|---|
| `DEEPSEEK_DSPARK_GATE` | `0` | enable gating |
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

## Tests

- `tests/test_dspark_gate.py` -- decision-rule properties, no GPU needed
- `tests/test_dspark_gate_replay.py` -- replays `tests/data/dspark_rounds_tp4.json`
  (the 31 measured rounds) and pins the results above
- On-device A/B lives outside the repo; the replay covers the decision rule but
  cannot confirm end-to-end token identity.
