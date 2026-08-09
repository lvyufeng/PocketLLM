"""Test pre-draft margin gating for DSpark."""
import torch
from src.models.deepseek_v4.dspark_gate import DSparkGate


def test_margin_gate_skips_low_confidence():
    """Low margin (uncertain distribution) should skip drafting."""
    gate = DSparkGate(5, enabled=True, margin_threshold=4.0)
    # Two tokens with similar logits -> low margin
    uncertain_logits = torch.full((1, 128000), -100.0)
    uncertain_logits[0, 42] = 0.0
    uncertain_logits[0, 99] = -0.5  # margin = 0.5 < 4.0
    assert not gate.should_draft(uncertain_logits)


def test_margin_gate_allows_high_confidence():
    """High margin (confident distribution) should allow drafting."""
    gate = DSparkGate(5, enabled=True, margin_threshold=4.0)
    # One dominant token -> high margin
    confident_logits = torch.full((1, 128000), -100.0)
    confident_logits[0, 42] = 0.0
    confident_logits[0, 99] = -10.0  # margin = 10.0 > 4.0
    assert gate.should_draft(confident_logits)


def test_margin_gate_disabled():
    """When margin_threshold <= 0, always draft."""
    gate = DSparkGate(5, enabled=True, margin_threshold=0.0)
    uncertain_logits = torch.full((1, 128000), -100.0)
    uncertain_logits[0, 42] = 0.0
    uncertain_logits[0, 99] = -0.5
    assert gate.should_draft(uncertain_logits)


def test_margin_gate_respects_enabled_flag():
    """When gate is disabled, always draft regardless of margin."""
    gate = DSparkGate(5, enabled=False, margin_threshold=4.0)
    uncertain_logits = torch.full((1, 128000), -100.0)
    uncertain_logits[0, 42] = 0.0
    uncertain_logits[0, 99] = -0.5
    assert gate.should_draft(uncertain_logits)


def test_margin_skipped_counter():
    """margin_skipped_rounds counter tracks pre-draft skips."""
    gate = DSparkGate(5, enabled=True, margin_threshold=4.0)
    assert gate.margin_skipped_rounds == 0
    gate.note_margin_skipped_round()
    assert gate.margin_skipped_rounds == 1
    assert gate.skipped_rounds == 1
    assert gate.rounds == 1
