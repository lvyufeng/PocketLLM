"""Correctness tests for the Qwen4-Exp BF16 gated-delta CUDA scan."""

from __future__ import annotations

import pytest
import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.qwen4_exp.attention import recurrent_gated_delta_rule


def _extension():
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_gated_delta_bf16_forward"):
        pytest.skip("cuda_kernel extension with Qwen4-Exp gated delta is unavailable")
    return ext


def _run_cuda(query, key, value, g, beta, state, groups_per_block=1):
    return _extension().qwen4_exp_gated_delta_bf16_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        state.contiguous(),
        1e-6,
        groups_per_block,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 7, 65])
def test_qwen4_exp_gated_delta_matches_sequential_reference(tokens):
    device = torch.device("cuda")
    torch.manual_seed(41 + tokens)
    key_heads, value_heads = 2, 6
    query = torch.randn(1, tokens, key_heads, 128, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(1, tokens, value_heads, 128, device=device, dtype=torch.bfloat16) * 0.1
    g = -(torch.rand(1, tokens, value_heads, device=device, dtype=torch.float32) * 0.04 + 0.001)
    beta = torch.sigmoid(
        torch.randn(1, tokens, value_heads, device=device, dtype=torch.bfloat16)
    )
    state = torch.randn(1, value_heads, 128, 128, device=device, dtype=torch.float32) * 0.001

    repeat = value_heads // key_heads
    want_output, want_state = recurrent_gated_delta_rule(
        query.repeat_interleave(repeat, dim=2),
        key.repeat_interleave(repeat, dim=2),
        value,
        g,
        beta,
        state.clone(),
    )
    got_output, got_state = _run_cuda(query, key, value, g, beta, state)
    torch.cuda.synchronize()

    torch.testing.assert_close(got_output, want_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(got_state, want_state, rtol=3e-3, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen4_exp_gated_delta_chunk_carry_matches_single_scan():
    device = torch.device("cuda")
    torch.manual_seed(71)
    tokens, split = 17, 6
    key_heads, value_heads = 2, 6
    query = torch.randn(1, tokens, key_heads, 128, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(1, tokens, value_heads, 128, device=device, dtype=torch.bfloat16) * 0.1
    g = -(torch.rand(1, tokens, value_heads, device=device, dtype=torch.float32) * 0.04 + 0.001)
    beta = torch.sigmoid(
        torch.randn(1, tokens, value_heads, device=device, dtype=torch.bfloat16)
    )
    state = torch.zeros(1, value_heads, 128, 128, device=device, dtype=torch.float32)

    whole_output, whole_state = _run_cuda(query, key, value, g, beta, state)
    first_output, first_state = _run_cuda(
        query[:, :split], key[:, :split], value[:, :split], g[:, :split], beta[:, :split], state
    )
    second_output, second_state = _run_cuda(
        query[:, split:],
        key[:, split:],
        value[:, split:],
        g[:, split:],
        beta[:, split:],
        first_state,
    )
    chunked_output = torch.cat([first_output, second_output], dim=1)
    torch.cuda.synchronize()

    torch.testing.assert_close(chunked_output, whole_output, rtol=0, atol=0)
    torch.testing.assert_close(second_state, whole_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen4_exp_gated_delta_launch_packing_is_exact():
    device = torch.device("cuda")
    torch.manual_seed(97)
    query = torch.randn(1, 9, 2, 128, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn(1, 9, 6, 128, device=device, dtype=torch.bfloat16) * 0.1
    g = -(torch.rand(1, 9, 6, device=device, dtype=torch.float32) * 0.04 + 0.001)
    beta = torch.sigmoid(torch.randn(1, 9, 6, device=device, dtype=torch.bfloat16))
    state = torch.zeros(1, 6, 128, 128, device=device, dtype=torch.float32)

    reference_output, reference_state = _run_cuda(query, key, value, g, beta, state, 1)
    for groups in (2, 4, 8):
        output, final_state = _run_cuda(query, key, value, g, beta, state, groups)
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        torch.testing.assert_close(final_state, reference_state, rtol=0, atol=0)
