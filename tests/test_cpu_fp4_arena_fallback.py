"""Regression tests for the raw FP4 CPU-arena representation."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2")
    or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires PyTorch FP4 and UE8M0 dtypes",
)

from src.kernels.ops import Packed4BitWeightAlongK, _dequant_fp4_weight_torch
from src.models.deepseek_v4.runtime import Expert, fp4_block_size


def _packed(raw: torch.Tensor) -> Packed4BitWeightAlongK:
    return Packed4BitWeightAlongK.convert_from(
        raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
    )


def _dequant(raw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return _dequant_fp4_weight_torch(
        _packed(raw),
        scales.view(torch.float8_e8m0fnu).to(torch.float32),
        block_size=fp4_block_size,
    )


def test_forward_cpu_dequantizes_raw_fp4_arena_bytes() -> None:
    """The pinned arena's uint8 views must behave like normal FP4 weights."""
    torch.manual_seed(1234)
    dim = 64
    inter_dim = 96
    expert = Expert(dim, inter_dim, dtype=torch.float4_e2m1fn_x2, swiglu_limit=7.0)

    # This is the representation produced by CPURoutedExpertsBackend's FP4
    # arena: packed weights and UE8M0 scales are all uint8 views.
    w1_raw = torch.randint(0, 256, (inter_dim, dim // 2), dtype=torch.uint8)
    w2_raw = torch.randint(0, 256, (dim, inter_dim // 2), dtype=torch.uint8)
    w3_raw = torch.randint(0, 256, (inter_dim, dim // 2), dtype=torch.uint8)
    s1_raw = torch.randint(124, 131, (inter_dim, dim // fp4_block_size), dtype=torch.uint8)
    s2_raw = torch.randint(124, 131, (dim, inter_dim // fp4_block_size), dtype=torch.uint8)
    s3_raw = torch.randint(124, 131, (inter_dim, dim // fp4_block_size), dtype=torch.uint8)

    expert._cpu_w1 = w1_raw
    expert._cpu_w2 = w2_raw
    expert._cpu_w3 = w3_raw
    expert._cpu_w1_scale = s1_raw
    expert._cpu_w2_scale = s2_raw
    expert._cpu_w3_scale = s3_raw
    expert._cpu_materialized = True

    x = torch.randn(3, dim, dtype=torch.float32)
    route_weight = torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.float32)
    got = expert.forward_cpu(x, route_weight)

    w1 = _dequant(w1_raw, s1_raw)
    w2 = _dequant(w2_raw, s2_raw)
    w3 = _dequant(w3_raw, s3_raw)
    gate = F.linear(x, w1)
    up = F.linear(x, w3)
    up = up.clamp(min=-7.0, max=7.0)
    gate = gate.clamp(max=7.0)
    want = F.linear(torch.nn.functional.silu(gate) * up * route_weight, w2)

    torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
