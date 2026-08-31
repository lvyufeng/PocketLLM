"""Correctness tests for Qwen4-Exp hyper-connection CUDA primitives."""

from __future__ import annotations

import pytest
import torch

from src.kernels.cuda_loader import load_cuda_kernel


def _extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ext = load_cuda_kernel()
    required = (
        "qwen4_exp_grouped_rms_norm",
        "qwen4_exp_inject",
    )
    if ext is None or any(not hasattr(ext, name) for name in required):
        pytest.skip("Qwen4-Exp hyper-connection CUDA extension is unavailable")
    return ext


@pytest.mark.parametrize("tokens", [1, 7, 8192])
def test_grouped_rms_norm_bf16_matches_reference(tokens: int) -> None:
    ext = _extension()
    torch.manual_seed(20260830 + tokens)
    groups, group_size = 4, 2560
    hidden = torch.randn(
        1,
        tokens,
        groups * group_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        groups * group_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.05

    h = hidden.float().reshape(1, tokens, groups, group_size)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    want = (
        h.flatten(-2) * (1.0 + weight.float())
    ).to(hidden.dtype)
    got = ext.qwen4_exp_grouped_rms_norm(
        hidden.contiguous(),
        weight.contiguous(),
        group_size,
        1e-6,
    )
    torch.testing.assert_close(got, want, rtol=8e-3, atol=2e-3)


@pytest.mark.parametrize("tokens", [1, 7, 8192])
def test_inject_bf16_matches_reference(tokens: int) -> None:
    ext = _extension()
    torch.manual_seed(20260930 + tokens)
    groups, hidden_size = 4, 2560
    block_output = torch.randn(
        1,
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hyper_input = torch.randn(
        1,
        tokens,
        groups * hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    injection_weights = torch.sigmoid(
        torch.randn(
            1,
            tokens,
            groups,
            device="cuda",
            dtype=torch.bfloat16,
        )
    ) * 2

    injection = (
        block_output.unsqueeze(-2)
        * injection_weights.unsqueeze(-1)
    )
    want = hyper_input + injection.flatten(-2)
    got = ext.qwen4_exp_inject(
        block_output.contiguous(),
        hyper_input.contiguous(),
        injection_weights.contiguous(),
        groups,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=0)
