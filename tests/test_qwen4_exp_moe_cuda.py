"""Correctness tests for the raw-BF16 Qwen4-Exp grouped MoE kernel."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.kernels.cuda_loader import load_cuda_kernel


def _reference(x, route_tokens, route_weights, seg_starts, gate_up, down):
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in range(gate_up.shape[0]):
        begin = int(seg_starts[expert].item())
        end = int(seg_starts[expert + 1].item())
        for route in range(begin, end):
            token = int(route_tokens[route].item())
            gate, up = F.linear(x[token : token + 1], gate_up[expert]).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            value = F.linear(hidden, down[expert]) * route_weights[route]
            out[token] += value[0].float()
    return out.to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen4_exp_bf16_grouped_matches_reference():
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_moe_prefill_bf16_forward"):
        pytest.skip("cuda_kernel extension with Qwen kernel is unavailable")

    device = torch.device("cuda")
    torch.manual_seed(17)
    tokens, hidden, inter, experts = 7, 32, 16, 5
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    gate_up = torch.randn(experts, 2 * inter, hidden, device=device, dtype=torch.bfloat16) * 0.05
    down = torch.randn(experts, hidden, inter, device=device, dtype=torch.bfloat16) * 0.05

    # Expert 2 has no routes; repeated token routes check scatter accumulation.
    route_tokens = torch.tensor([6, 0, 3, 0, 5, 1, 3, 2], device=device, dtype=torch.int64)
    route_weights = torch.tensor(
        [0.2, 0.7, 0.3, 0.1, 0.4, 0.8, 0.6, 0.5], device=device, dtype=torch.float32
    )
    seg_starts = torch.tensor([0, 2, 4, 4, 6, 8], device=device, dtype=torch.int32)

    got = ext.qwen4_exp_moe_prefill_bf16_forward(
        x, route_tokens, route_weights, seg_starts, gate_up, down, 0.0
    )
    want = _reference(x, route_tokens, route_weights, seg_starts, gate_up, down)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, want, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen4_exp_bf16_grouped_empty_routes():
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_moe_prefill_bf16_forward"):
        pytest.skip("cuda_kernel extension with Qwen kernel is unavailable")

    device = torch.device("cuda")
    x = torch.randn(3, 16, device=device, dtype=torch.bfloat16)
    gate_up = torch.randn(2, 8, 16, device=device, dtype=torch.bfloat16)
    down = torch.randn(2, 16, 4, device=device, dtype=torch.bfloat16)
    empty = torch.empty(0, device=device, dtype=torch.int64)
    empty_w = torch.empty(0, device=device, dtype=torch.float32)
    seg_starts = torch.zeros(3, device=device, dtype=torch.int32)
    got = ext.qwen4_exp_moe_prefill_bf16_forward(
        x, empty, empty_w, seg_starts, gate_up, down, 0.0
    )
    assert got.shape == x.shape
    assert got.dtype == x.dtype
    assert torch.count_nonzero(got) == 0
