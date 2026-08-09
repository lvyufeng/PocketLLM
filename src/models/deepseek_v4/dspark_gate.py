"""Adaptive draft-length gating for DSpark speculative decoding.

A DSpark round drafts k tokens and verifies all of them in one main-model
forward. Verify cost grows with the draft length -- measured on TP=4 FP4
(4x2080Ti) it is close to linear:

    verify(n) ~= 300ms + 109ms * n          (fit over n = 1, 2, 3, 6)

so a round that gets nothing accepted costs ~1046ms to produce the one bonus
token that a 303ms plain decode would have produced anyway. In a 31-round
measurement, 7 rounds accepted zero tokens and together burned 5.2s more than
plain decoding -- 16% of total speculative time.

The last DSpark stage already emits a confidence score per draft position, and
it carries real signal (pooled over 4 prompts, P(match) rises monotonically from
20% in the lowest confidence bucket to 100% in the highest). The gate turns that
score into a draft length.

Why not a fixed threshold: the best threshold depends on the workload's accept
rate, which a threshold cannot express. Leave-one-prompt-out, a threshold tuned
on 3 prompts gained 1.89x on the worst prompt but *lost* 13% on the best one,
because on an easy workload every cut is a cut into a draft that would have been
accepted. So the gate instead estimates P(match | confidence) from the rounds it
has already run and picks the length maximizing expected tokens per millisecond:

    n* = argmax_n  E[tokens(n)] / cost(n)
    E[tokens(n)] = 1 + sum_{i<n} prod_{j<=i} p_j     (bonus token + expected prefix)
    cost(n)      = draft_ms + verify(n)   for n > 0,   plain_ms   for n = 0

The estimate is learned online from observed accepts, never from a fitted prior,
so an easy workload's high-confidence buckets converge to p~1.0 and the gate
stops truncating. Replayed causally over the same 31 rounds (each prompt
starting cold, which is the pessimistic case) this is 1.31x plain decode with no
per-prompt regression against the current always-draft-k behaviour.

Pre-draft gating: The confidence score is produced *by* the draft, so gating can
only avoid the verify cost, never the 196ms draft cost. This is why a skipped
round still costs draft_ms and the gate cannot fully close the gap to plain
decode on a hard workload. To close it, the gate checks the committed token's
logit margin (top1 - top2) *before* drafting: if the main model is uncertain
about what comes next (small margin), the draft will be poor and speculation
will lose even before verify cost. Controlled by
DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD (default 0.0 = disabled, set >0 to enable).

See docs/dspark.md for the full measurement tables.
"""
from __future__ import annotations

import os
from collections import defaultdict

import torch

# Confidence bucket edges. Coarse on purpose: with only a handful of resolved
# positions per round, finer buckets would spend the whole generation on the
# prior instead of on observations.
_BUCKET_EDGES = (-1e9, -0.5, 0.5, 1.5, 3.0, 6.0, 1e9)

# Cold-start pseudo-counts. PRIOR_P is deliberately optimistic so a fresh stream
# drafts the full k until it has evidence to do otherwise -- guessing low would
# truncate good drafts on an easy workload before ever observing one.
_PRIOR_N = 2.0
_PRIOR_P = 0.75

# Per-round forgetting, so a mid-generation shift (prose -> code block) is
# tracked rather than averaged away.
_DECAY = 0.9

# Keeping the full draft is the safe default: cost is linear in n but expected
# tokens are multiplicative, so an underestimate of p compounds while an
# overestimate only wastes one position of verify. Require truncation to win by
# this factor before taking it.
_MARGIN = 1.15


def _bucket_of(conf: float) -> int:
    for i in range(len(_BUCKET_EDGES) - 1):
        if _BUCKET_EDGES[i] <= conf < _BUCKET_EDGES[i + 1]:
            return i
    return len(_BUCKET_EDGES) - 2


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


class DSparkGate:
    """Chooses how many of the k drafted tokens to verify.

    Cost parameters are in milliseconds and describe the *shape* of the cost
    curve, not absolute performance: only the ratio between the fixed verify
    cost and the per-token slope affects the decision, so the defaults measured
    on TP=4 FP4 transfer to other setups as long as the shape holds. They are
    overridable per deployment via the env vars named in `from_env`.

    Not thread-safe; one gate per generation stream.
    """

    def __init__(self, block_size: int, *, draft_ms: float = 196.0,
                 verify_base_ms: float = 300.0, verify_slope_ms: float = 109.0,
                 plain_ms: float = 303.0, margin: float = _MARGIN,
                 min_draft: int = 0, decay: float = _DECAY,
                 enabled: bool = True, margin_threshold: float = 0.0):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.draft_ms = draft_ms
        self.verify_base_ms = verify_base_ms
        self.verify_slope_ms = verify_slope_ms
        self.plain_ms = plain_ms
        self.margin = margin
        self.min_draft = min_draft
        self.decay = decay
        self.enabled = enabled
        self.margin_threshold = margin_threshold
        # bucket -> [decayed matches, decayed observations]
        self._stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.rounds = 0
        self.truncated_rounds = 0
        self.skipped_rounds = 0
        self.margin_skipped_rounds = 0

    @classmethod
    def from_env(cls, block_size: int) -> "DSparkGate":
        """Build from DEEPSEEK_DSPARK_GATE* env vars.

        Default off: the gate changes how many tokens are drafted, which changes
        the throughput/accept trade-off, so it should be opted into rather than
        silently altering an existing deployment's behaviour.
        """
        return cls(
            block_size,
            draft_ms=_env_float("DEEPSEEK_DSPARK_GATE_DRAFT_MS", 196.0),
            verify_base_ms=_env_float("DEEPSEEK_DSPARK_GATE_VERIFY_BASE_MS", 300.0),
            verify_slope_ms=_env_float("DEEPSEEK_DSPARK_GATE_VERIFY_SLOPE_MS", 109.0),
            plain_ms=_env_float("DEEPSEEK_DSPARK_GATE_PLAIN_MS", 303.0),
            margin=_env_float("DEEPSEEK_DSPARK_GATE_MARGIN", _MARGIN),
            min_draft=int(_env_float("DEEPSEEK_DSPARK_GATE_MIN_DRAFT", 0)),
            decay=_env_float("DEEPSEEK_DSPARK_GATE_DECAY", _DECAY),
            enabled=_env_flag("DEEPSEEK_DSPARK_GATE"),
            margin_threshold=_env_float("DEEPSEEK_DSPARK_GATE_MARGIN_THRESHOLD", 0.0),
        )

    def match_prob(self, conf: float) -> float:
        """Estimated P(main model accepts this token | its confidence score)."""
        matches, total = self._stats[_bucket_of(conf)]
        return (matches + _PRIOR_N * _PRIOR_P) / (total + _PRIOR_N)

    def should_draft(self, logits: torch.Tensor) -> bool:
        """Whether to attempt drafting based on the committed token's uncertainty.

        If the main model is uncertain (small logit margin between top1 and top2),
        the draft will be poor and speculation will lose to plain decode even
        before verify cost. This check happens before drafting, so skipping here
        avoids the 196ms draft cost.

        `logits` is the last main-model logits [1, vocab_size] for the committed
        token, used to compute the margin as a proxy for draft quality.
        """
        if not self.enabled or self.margin_threshold <= 0:
            return True
        top2 = torch.topk(logits[0], k=2, dim=-1).values
        margin = float(top2[0] - top2[1])
        return margin >= self.margin_threshold

    def choose_draft_len(self, confidence) -> int:
        """How many leading draft tokens to verify. 0 means skip the draft and
        take a plain single-token decode step instead.

        `confidence` is the per-position score sequence from the last DSpark
        stage (any float iterable; a 1-D tensor row works).
        """
        if not self.enabled:
            return self.block_size
        conf = [float(c) for c in confidence]
        if len(conf) < self.block_size:
            # Nothing sensible to say about positions we have no score for.
            return self.block_size

        rates = {0: 1.0 / self.plain_ms}
        run_prob, exp_tokens = 1.0, 1.0   # the bonus token is free on any verify
        for n in range(1, self.block_size + 1):
            run_prob *= self.match_prob(conf[n - 1])
            exp_tokens += run_prob
            cost = self.draft_ms + self.verify_base_ms + self.verify_slope_ms * n
            rates[n] = exp_tokens / cost

        best = max(rates, key=lambda n: rates[n])
        full = self.block_size
        if best != full and rates[best] < rates[full] * self.margin:
            best = full
        if best != 0 and best < self.min_draft:
            best = min(self.min_draft, full)
        return best

    def observe(self, confidence, n_accepted: int, n_drafted: int) -> None:
        """Record a finished round so later rounds see a better estimate.

        Only positions the round resolved are counted: 0..n_accepted-1 matched,
        and position n_accepted did not. Positions beyond that were verified but
        their match status is unknown (the prefix already broke), so they are
        censored rather than counted as misses -- counting them would bias the
        estimate down and make the gate truncate more and more.
        """
        if n_drafted <= 0:
            return
        conf = [float(c) for c in confidence]
        self.rounds += 1
        if n_drafted < self.block_size:
            self.truncated_rounds += 1
        for bucket in self._stats.values():
            bucket[0] *= self.decay
            bucket[1] *= self.decay
        for i in range(min(n_accepted + 1, n_drafted, len(conf))):
            bucket = self._stats[_bucket_of(conf[i])]
            bucket[0] += 1.0 if i < n_accepted else 0.0
            bucket[1] += 1.0

    def note_skipped_round(self) -> None:
        """Record that a round chose plain decode over drafting.

        Such a round observes no draft positions, so it cannot update the
        calibration -- which means a gate that skips forever would never learn
        it should stop. The counter exists so callers can detect that.
        """
        self.rounds += 1
        self.skipped_rounds += 1

    def note_margin_skipped_round(self) -> None:
        """Record that a round was skipped due to low committed-token margin."""
        self.rounds += 1
        self.skipped_rounds += 1
        self.margin_skipped_rounds += 1

    def stats(self) -> dict:
        """Diagnostics; the per-bucket rates are what to look at if the gate
        behaves unexpectedly."""
        return {
            "rounds": self.rounds,
            "truncated_rounds": self.truncated_rounds,
            "skipped_rounds": self.skipped_rounds,
            "margin_skipped_rounds": self.margin_skipped_rounds,
            "bucket_rates": {
                f"[{_BUCKET_EDGES[b]:+.1f},{_BUCKET_EDGES[b + 1]:+.1f})":
                    round(m / n, 3) if n else None
                for b, (m, n) in sorted(self._stats.items())
            },
        }
