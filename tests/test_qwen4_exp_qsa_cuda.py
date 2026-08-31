from __future__ import annotations

import pytest
import torch

from src.kernels.cuda_loader import load_cuda_kernel


def _extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_qsa_bf16_forward"):
        pytest.skip("Qwen4-Exp QSA CUDA extension is unavailable")
    return ext


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, rows, query_heads, dim = query.shape
    kv_heads = key.shape[1]
    heads_per_kv = query_heads // kv_heads
    safe = selected.clamp_min(0).to(torch.long)
    batch_index = torch.arange(batch, device=query.device).view(batch, 1, 1).expand_as(safe)
    key_tokens = key.transpose(1, 2)
    value_tokens = value.transpose(1, 2)
    gathered_key = key_tokens[batch_index, safe].repeat_interleave(heads_per_kv, dim=3)
    gathered_value = value_tokens[batch_index, safe].repeat_interleave(heads_per_kv, dim=3)
    scores = torch.einsum("bqhd,bqkhd->bqhk", query.float(), gathered_key.float()) * scale
    scores = scores.masked_fill((selected < 0).unsqueeze(2), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bqhk,bqkhd->bqhd", weights, gathered_value.float()).to(query.dtype)


@pytest.mark.parametrize(
    "rows,kv_len,selected_len,query_heads",
    [
        (1, 7, 9, 6),
        (5, 23, 17, 6),
        (19, 64, 33, 6),
        (5, 23, 17, 12),
    ],
)
def test_qsa_bf16_matches_reference(
    rows: int,
    kv_len: int,
    selected_len: int,
    query_heads: int,
) -> None:
    ext = _extension()
    torch.manual_seed(20260830 + rows + query_heads)
    query = torch.randn(
        1,
        rows,
        query_heads,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    key = torch.randn(1, 1, kv_len, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    value = torch.randn(1, 1, kv_len, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    selected = torch.randint(0, kv_len, (1, rows, selected_len), device="cuda", dtype=torch.int32)
    selected[:, :, -2:] = -1
    scale = 256**-0.5

    got = ext.qwen4_exp_qsa_bf16_forward(query, key, value, selected, scale)
    want = _reference(query, key, value, selected, scale)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-3)


def test_qsa_bf16_empty_padding_is_ignored() -> None:
    ext = _extension()
    query = torch.randn(1, 3, 6, 256, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 8, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 1, 8, 256, device="cuda", dtype=torch.bfloat16)
    selected = torch.full((1, 3, 11), -1, device="cuda", dtype=torch.int32)
    selected[:, :, 0] = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)

    got = ext.qwen4_exp_qsa_bf16_forward(query, key, value, selected, 256**-0.5)
    want = value[:, :, [0, 3, 7]].transpose(1, 2).expand(-1, -1, 6, -1)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
