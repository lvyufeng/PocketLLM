"""Replay the measured DSpark rounds through the gate and pin the outcome.

tests/data/dspark_rounds_tp4.json holds 31 real rounds (TP=4 FP4, 4x2080Ti, the
0731 checkpoint) with each round's per-position confidence and how many tokens
the main model actually accepted. That is enough to score any draft-length rule
offline: the accept count tells us what a shorter draft would have got, since
accepting a prefix of length a means positions 0..a-1 matched regardless of how
many were submitted.

Replay is causal -- round t is decided using only rounds before it, and each
prompt starts with a cold gate. A real stream keeps gate state across a whole
generation, so this is the pessimistic case.

What it cannot show: whether truncating changes which tokens come out. It cannot,
because the gate only shortens the verified prefix and the main model's greedy
output at each position does not depend on how many positions follow it -- but
that is an argument, not a measurement, so tests/test_dspark_gate_replay.py's
sibling on-device A/B is what actually confirms it.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from src.models.deepseek_v4.dspark_gate import DSparkGate

DATA = Path(__file__).parent / "data" / "dspark_rounds_tp4.json"

# Verify cost shape fitted on the measured multi-token points
# (n, ms) = (1, 296), (2, 432), (3, 494), (6, 850).
VERIFY_BASE, VERIFY_SLOPE = 300.0, 109.0


@pytest.fixture(scope="module")
def rounds():
    with open(DATA) as f:
        return json.load(f)


def replay(prompt_rounds, block_size, draft_ms, plain_ms, **gate_kw):
    """Run one prompt's rounds causally; returns (ms_per_token, draft_lens)."""
    gate = DSparkGate(block_size, draft_ms=draft_ms, plain_ms=plain_ms,
                      verify_base_ms=VERIFY_BASE, verify_slope_ms=VERIFY_SLOPE,
                      **gate_kw)
    ms = tokens = 0.0
    lens = []
    for r in prompt_rounds:
        n = gate.choose_draft_len(r["conf"])
        lens.append(n)
        if n == 0:
            ms += plain_ms
            tokens += 1
            gate.note_skipped_round()
            continue
        ms += draft_ms + VERIFY_BASE + VERIFY_SLOPE * n
        tokens += min(r["accept"], n) + 1
        gate.observe(r["conf"], r["accept"], n)
    return ms / tokens, lens


def cost_params(data):
    allr = [r for rs in data["prompts"].values() for r in rs]
    return (statistics.median(r["draft_ms"] for r in allr),
            statistics.median(r["plain_ms"] for r in allr))


def test_fixture_is_the_measured_data(rounds):
    assert rounds["block_size"] == 5
    assert sum(len(v) for v in rounds["prompts"].values()) == 31
    for rs in rounds["prompts"].values():
        for r in rs:
            assert 0 <= r["accept"] <= r["k"]
            assert len(r["conf"]) == r["k"]


def test_gate_beats_always_drafting_overall(rounds):
    """The headline claim: gating is faster than always drafting all k."""
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    on_ms = off_ms = on_tok = off_tok = 0.0
    for rs in rounds["prompts"].values():
        per_on, _ = replay(rs, bs, draft_ms, plain_ms, enabled=True)
        per_off, _ = replay(rs, bs, draft_ms, plain_ms, enabled=False)
        # Weight by rounds so the overall figure is not a mean of means.
        on_ms += per_on * len(rs)
        off_ms += per_off * len(rs)
        on_tok += len(rs)
        off_tok += len(rs)
    speedup = (off_ms / off_tok) / (on_ms / on_tok)
    # Measured 1.16x on this fixture (3.62 vs 3.12 tok/s round-weighted).
    assert speedup > 1.10, f"gate gained only {speedup:.3f}x over always-k"


def test_gate_does_not_regress_any_prompt(rounds):
    """The fixed-threshold version lost 13% on `code`; this must not.

    A small tolerance is allowed because the gate spends its first rounds
    learning, and on an easy prompt those rounds cannot be better than the
    always-draft baseline -- only equal.
    """
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    for label, rs in rounds["prompts"].items():
        on, _ = replay(rs, bs, draft_ms, plain_ms, enabled=True)
        off, _ = replay(rs, bs, draft_ms, plain_ms, enabled=False)
        assert on <= off * 1.02, (
            f"{label}: gating regressed ({1000/on:.2f} vs {1000/off:.2f} tok/s)")


def test_gain_concentrates_on_the_low_accept_prompt(rounds):
    """Sanity on the mechanism: the win should come from cutting losing drafts.

    `math` accepted 0.88/5 on average and is where always-drafting loses to
    plain decode; if the gate's gain showed up somewhere else, the rule would be
    doing something other than what it claims.
    """
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    gains = {}
    for label, rs in rounds["prompts"].items():
        on, _ = replay(rs, bs, draft_ms, plain_ms, enabled=True)
        off, _ = replay(rs, bs, draft_ms, plain_ms, enabled=False)
        gains[label] = off / on
    assert max(gains, key=lambda k: gains[k]) == "math", gains
    assert gains["math"] > 1.3, gains


def test_easy_prompt_keeps_drafting_full_block(rounds):
    """`repeat` accepted 5/5 every round; the gate must leave it alone."""
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    _, lens = replay(rounds["prompts"]["repeat"], bs, draft_ms, plain_ms, enabled=True)
    assert all(n == bs for n in lens), f"truncated an all-accept prompt: {lens}"


def test_disabled_gate_reproduces_always_k(rounds):
    """Default-off must be bit-identical in decisions to current behaviour."""
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    for rs in rounds["prompts"].values():
        _, lens = replay(rs, bs, draft_ms, plain_ms, enabled=False)
        assert all(n == bs for n in lens)


def test_gate_narrows_but_does_not_close_the_plain_decode_gap(rounds):
    """Documents a real limitation rather than asserting a win that isn't there.

    Always-drafting is slower than no speculation at all on the two hard prompts
    (math 1.80, prose 2.88 vs plain 3.30 tok/s). The gate recovers most of that
    -- math 1.80 -> 2.45 -- but cannot fully reach plain decode, because it only
    ever sees confidence *after* paying for the draft: a round it decides to skip
    still cost draft_ms. Closing the gap would need a decision made before
    drafting, which the confidence score cannot provide.

    The gate is still worth enabling: it never loses to always-drafting, and it
    turns the catastrophic cases into mild ones.
    """
    bs = rounds["block_size"]
    draft_ms, plain_ms = cost_params(rounds)
    recovered = {}
    for label, rs in rounds["prompts"].items():
        on, _ = replay(rs, bs, draft_ms, plain_ms, enabled=True)
        off, _ = replay(rs, bs, draft_ms, plain_ms, enabled=False)
        if off > plain_ms:  # always-drafting loses to plain decode here
            # Fraction of the always-k -> plain gap that gating recovers.
            recovered[label] = (off - on) / (off - plain_ms)
    assert set(recovered) == {"prose", "math"}, recovered
    # Measured: prose 44%, math 72% of the gap recovered. The bound is the
    # measurement, not a target -- it exists to catch the rule getting worse.
    for label, frac in recovered.items():
        assert frac > 0.40, f"{label}: recovered only {frac:.0%} of the gap"
