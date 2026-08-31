"""Bit-exactness tests for the fused Qwen4-Exp hyper-connection activations."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.kernels.cuda_loader import load_cuda_kernel


def _extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ext = load_cuda_kernel()
    required = ("qwen4_exp_hc_silu", "qwen4_exp_hc_inject_gate")
    if ext is None or any(not hasattr(ext, name) for name in required):
        pytest.skip("Qwen4-Exp hyper-connection activations are unavailable")
    return ext


@pytest.mark.parametrize("tokens", [1, 7, 8192])
def test_hc_silu_is_bit_exact(tokens: int) -> None:
    ext = _extension()
    torch.manual_seed(20260901 + tokens)
    groups, rank = 4, 320
    raw = torch.randn(1, tokens, rank, device="cuda", dtype=torch.bfloat16) * 8
    want = F.silu(raw / groups)
    got = ext.qwen4_exp_hc_silu(raw.contiguous(), groups)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 8192])
def test_hc_inject_gate_is_bit_exact(tokens: int) -> None:
    ext = _extension()
    torch.manual_seed(20261001 + tokens)
    groups = 4
    raw = torch.randn(1, tokens, groups, device="cuda", dtype=torch.bfloat16) * 8
    want = 2 * torch.sigmoid(raw / groups)
    got = ext.qwen4_exp_hc_inject_gate(raw.contiguous(), groups)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_hc_activations_reject_non_bf16() -> None:
    ext = _extension()
    raw = torch.randn(4, 320, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        ext.qwen4_exp_hc_silu(raw, 4)
    with pytest.raises(RuntimeError):
        ext.qwen4_exp_hc_inject_gate(raw, 4)
