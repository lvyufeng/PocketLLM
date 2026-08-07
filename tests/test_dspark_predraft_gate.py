"""Test pre-draft entropy gating for DSpark."""
import torch
from src.models.deepseek_v4.dspark_gate import DSparkGate


def test_entropy_gate_skips_high_uncertainty():
    """High entropy (uniform distribution) should skip drafting."""
    gate = DSparkGate(5, enabled=True, entropy_threshold=3.0)
    # Uniform logits -> high entropy (~6.9 for vocab_size=128k)
    uniform_logits = torch.zeros(1, 128000)
    assert not gate.should_draft(uniform_logits)


def test_entropy_gate_allows_low_uncertainty():
    """Low entropy (peaked distribution) should allow drafting."""
    gate = DSparkGate(5, enabled=True, entropy_threshold=3.0)
    # Peaked logits -> low entropy
    peaked_logits = torch.full((1, 128000), -100.0)
    peaked_logits[0, 42] = 0.0  # one very confident token
    assert gate.should_draft(peaked_logits)


def test_entropy_gate_disabled():
    """When entropy_threshold <= 0, always draft."""
    gate = DSparkGate(5, enabled=True, entropy_threshold=0.0)
    uniform_logits = torch.zeros(1, 128000)
    assert gate.should_draft(uniform_logits)


def test_entropy_gate_respects_enabled_flag():
    """When gate is disabled, always draft regardless of entropy."""
    gate = DSparkGate(5, enabled=False, entropy_threshold=3.0)
    uniform_logits = torch.zeros(1, 128000)
    assert gate.should_draft(uniform_logits)


def test_entropy_skipped_counter():
    """entropy_skipped_rounds counter tracks pre-draft skips."""
    gate = DSparkGate(5, enabled=True, entropy_threshold=3.0)
    assert gate.entropy_skipped_rounds == 0
    gate.note_entropy_skipped_round()
    assert gate.entropy_skipped_rounds == 1
    assert gate.skipped_rounds == 1
    assert gate.rounds == 1
