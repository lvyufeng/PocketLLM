"""Correctness tests for Qwen4-Exp FP8 block dequantization."""

from __future__ import annotations

import os

import pytest
import torch

from src.models.qwen4_exp.layers import dequant_fp8_block
from src.models.qwen4_exp.quant import FP8Tensor
from src.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint

FP8_MODEL = os.environ.get(
    "QWEN4EXP_FP8_MODEL", "/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next-FP8"
)
BF16_MODEL = os.environ.get(
    "QWEN4EXP_MODEL", "/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_block_dequant_whole_tiles(dtype: torch.dtype) -> None:
    codes = torch.tensor(
        [[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]], dtype=torch.float8_e4m3fn
    )
    scale = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16)
    got = dequant_fp8_block(codes, scale, block=(2, 2), out_dtype=dtype)
    want = codes.to(dtype) * scale.to(dtype).repeat_interleave(2, 1)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_block_dequant_ragged_tail() -> None:
    codes = torch.arange(15, dtype=torch.uint8).reshape(3, 5).view(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    got = dequant_fp8_block(codes, scale, block=(2, 3), out_dtype=torch.float32)
    expanded = scale.repeat_interleave(2, 0).repeat_interleave(3, 1)[:3, :5]
    torch.testing.assert_close(got, codes.to(torch.float32) * expanded, rtol=0, atol=0)


def test_block_dequant_rejects_bad_scale_shape() -> None:
    with pytest.raises(ValueError, match="scale shape"):
        dequant_fp8_block(
            torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1, dtype=torch.bfloat16),
            block=(2, 2),
        )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(FP8_MODEL, "model.safetensors.index.json")),
    reason=f"real FP8 checkpoint not found at {FP8_MODEL}",
)
def test_real_fp8_dequant_matches_packed_bf16_reference() -> None:
    if not os.path.exists(os.path.join(BF16_MODEL, "model.safetensors.index.json")):
        pytest.skip(f"BF16 reference checkpoint not found at {BF16_MODEL}")
    fp8 = Qwen4ExpCheckpoint(FP8_MODEL, store=MmapSafetensors(FP8_MODEL))
    bf16_store = MmapSafetensors(BF16_MODEL)
    for layer_idx, expert_id in ((0, 0), (1, 137), (3, 511)):
        gate_up, down = fp8.expert_rows(layer_idx, expert_id)
        assert isinstance(gate_up, FP8Tensor)
        assert isinstance(down, FP8Tensor)
        ref_gate_up = bf16_store.view(
            f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
        )[expert_id].to(torch.float32)
        ref_down = bf16_store.view(
            f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"
        )[expert_id].to(torch.float32)
        got_gate_up = gate_up.dequantize(torch.float32)
        got_down = down.dequantize(torch.float32)
        gate_rel = (got_gate_up - ref_gate_up).abs().max() / ref_gate_up.abs().max()
        down_rel = (got_down - ref_down).abs().max() / ref_down.abs().max()
        # FP8 quantization is intentionally lossy; these are structural checks,
        # not a claim of token-level equality between distinct checkpoints.
        assert gate_rel < 0.04
        assert down_rel < 0.04
