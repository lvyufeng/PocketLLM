"""Tests for DSpark adaptive draft-length gating.

These are pure-Python: the gate holds no tensors and no model state, so its
behaviour is testable without a checkpoint. The properties worth pinning down
are the ones that made the fixed-threshold version regress -- that an easy
workload converges back to the full draft, and that a hard one truncates.
"""
from __future__ import annotations

import pytest

from src.models.deepseek_v4.dspark_gate import DSparkGate


def make_gate(**kw) -> DSparkGate:
    """Default gate with the measured TP=4 FP4 cost shape, gating on."""
    kw.setdefault("enabled", True)
    return DSparkGate(5, **kw)


def test_disabled_gate_always_drafts_full_block():
    gate = make_gate(enabled=False)
    # Even confidence that screams "reject" must not change the draft length.
    assert gate.choose_draft_len([-9.0] * 5) == 5


def test_cold_start_drafts_full_block():
    """A fresh stream has no evidence, so it must not truncate on a guess."""
    gate = make_gate()
    assert gate.choose_draft_len([1.0] * 5) == 5


def test_high_confidence_stream_keeps_full_draft():
    """The failure mode of a fixed threshold: cutting drafts that would land."""
    gate = make_gate()
    for _ in range(20):
        conf = [9.0, 8.0, 7.0, 6.5, 6.0]
        n = gate.choose_draft_len(conf)
        gate.observe(conf, n_accepted=5, n_drafted=n)
    assert gate.choose_draft_len([9.0, 8.0, 7.0, 6.5, 6.0]) == 5
    assert gate.truncated_rounds == 0


def test_low_confidence_stream_truncates():
    """A workload that keeps rejecting should stop paying for long verifies."""
    gate = make_gate()
    for _ in range(20):
        conf = [-2.0] * 5
        n = gate.choose_draft_len(conf)
        gate.observe(conf, n_accepted=0, n_drafted=n)
    assert gate.choose_draft_len([-2.0] * 5) < 5


def test_truncation_respects_prefix_structure():
    """Confidence collapsing mid-block should cut at the collapse, not before.

    Trained so the high bucket is reliable and the low bucket is not; a draft
    that starts strong and then falls off should keep the strong prefix.
    """
    gate = make_gate()
    for _ in range(30):
        gate.observe([9.0, 9.0, -3.0, -3.0, -3.0], n_accepted=2, n_drafted=5)
    n = gate.choose_draft_len([9.0, 9.0, -3.0, -3.0, -3.0])
    assert 0 < n <= 3, f"expected the strong prefix to survive, got {n}"


def test_min_draft_floor_is_respected():
    gate = make_gate(min_draft=3)
    for _ in range(20):
        gate.observe([-5.0] * 5, n_accepted=0, n_drafted=5)
    n = gate.choose_draft_len([-5.0] * 5)
    assert n == 0 or n >= 3, f"floor violated: {n}"


def test_censored_positions_do_not_count_as_misses():
    """Positions after the first rejection have unknown match status.

    Counting them as misses would drag every bucket's estimate down and make the
    gate truncate progressively harder regardless of the real accept rate.
    """
    gate = make_gate()
    # accept=2 of 5: positions 0,1 matched, 2 missed, 3 and 4 are censored.
    gate.observe([1.0, 1.0, 1.0, 1.0, 1.0], n_accepted=2, n_drafted=5)
    # One bucket holds all five scores; only 3 observations should be recorded.
    (matches, total), = gate._stats.values()
    assert total == pytest.approx(3.0)
    assert matches == pytest.approx(2.0)


def test_full_accept_records_every_position():
    gate = make_gate()
    gate.observe([2.0] * 5, n_accepted=5, n_drafted=5)
    (matches, total), = gate._stats.values()
    assert total == pytest.approx(5.0)
    assert matches == pytest.approx(5.0)


def test_decay_tracks_a_workload_shift():
    """An easy stretch must not permanently mask a later hard one."""
    gate = make_gate()
    for _ in range(15):
        gate.observe([4.0] * 5, n_accepted=5, n_drafted=5)
    easy = gate.match_prob(4.0)
    for _ in range(15):
        gate.observe([4.0] * 5, n_accepted=0, n_drafted=5)
    assert gate.match_prob(4.0) < easy * 0.6, "decay did not track the shift"


def test_short_confidence_sequence_falls_back_to_full_block():
    gate = make_gate()
    assert gate.choose_draft_len([1.0, 1.0]) == 5


def test_observe_ignores_zero_length_draft():
    gate = make_gate()
    gate.observe([1.0] * 5, n_accepted=0, n_drafted=0)
    assert gate.rounds == 0
    assert not gate._stats


def test_skipped_round_is_counted_separately():
    gate = make_gate()
    gate.note_skipped_round()
    assert gate.rounds == 1
    assert gate.skipped_rounds == 1
    assert not gate._stats, "a skipped round observes no draft positions"


def test_rejects_nonpositive_block_size():
    with pytest.raises(ValueError):
        DSparkGate(0)


def test_env_construction_defaults_to_off(monkeypatch):
    """The gate alters draft length, so it must be opt-in."""
    monkeypatch.delenv("DEEPSEEK_DSPARK_GATE", raising=False)
    assert not DSparkGate.from_env(5).enabled


def test_env_construction_reads_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_DSPARK_GATE", "1")
    monkeypatch.setenv("DEEPSEEK_DSPARK_GATE_MARGIN", "1.4")
    monkeypatch.setenv("DEEPSEEK_DSPARK_GATE_VERIFY_SLOPE_MS", "50")
    gate = DSparkGate.from_env(5)
    assert gate.enabled
    assert gate.margin == pytest.approx(1.4)
    assert gate.verify_slope_ms == pytest.approx(50.0)


def test_cheap_verify_slope_never_trims_mid_block():
    """The decision must follow the cost shape, not a hardcoded preference.

    When extra verify positions are nearly free, trimming cannot pay: every
    position after the first is almost pure upside. The gate may still skip the
    round entirely (the fixed draft+verify_base cost is unaffected by the slope
    and can exceed a plain step), but a middle length is never right.
    """
    gate = make_gate(verify_slope_ms=0.5)
    for _ in range(20):
        conf = [3.0, -4.0, -4.0, -4.0, -4.0]
        gate.observe(conf, n_accepted=1, n_drafted=5)
    assert gate.choose_draft_len([3.0, -4.0, -4.0, -4.0, -4.0]) in (0, 5)


def test_cheap_slope_keeps_full_draft_when_drafting_pays():
    """Same cost shape, but a draft worth running: expect the full block."""
    gate = make_gate(verify_slope_ms=0.5)
    for _ in range(20):
        gate.observe([3.0] * 5, n_accepted=5, n_drafted=5)
    assert gate.choose_draft_len([3.0] * 5) == 5


def test_expensive_verify_slope_favours_skipping():
    """When each extra verify token costs more than a whole plain decode step,
    a hopeless draft should be skipped rather than trimmed."""
    gate = make_gate(verify_slope_ms=5000.0)
    for _ in range(20):
        gate.observe([-4.0] * 5, n_accepted=0, n_drafted=5)
    assert gate.choose_draft_len([-4.0] * 5) == 0


def test_stats_reports_bucket_rates():
    gate = make_gate()
    gate.observe([9.0] * 5, n_accepted=5, n_drafted=5)
    st = gate.stats()
    assert st["rounds"] == 1
    assert any(v == 1.0 for v in st["bucket_rates"].values() if v is not None)


def test_choose_never_exceeds_block_size():
    """Guards the caller: a returned length is always sliceable into the draft."""
    gate = make_gate(min_draft=99)
    for _ in range(10):
        gate.observe([0.0] * 5, n_accepted=1, n_drafted=5)
    assert gate.choose_draft_len([0.0] * 5) <= 5
