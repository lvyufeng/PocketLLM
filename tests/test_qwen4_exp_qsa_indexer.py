from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.qwen4_exp.attention import QSAIndexer


@pytest.fixture
def config() -> SimpleNamespace:
    return SimpleNamespace(
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rms_norm_eps=1e-6,
    )


def _indexer(config: SimpleNamespace) -> QSAIndexer:
    qk_rows = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim
    weights = {
        "index_qk_proj": torch.randn(qk_rows, 16),
        "q_layernorm": torch.ones(config.indexer_head_dim),
        "k_layernorm": torch.ones(config.indexer_head_dim),
    }
    return QSAIndexer(config, weights)


def _rope(length: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(1, length, dim), torch.zeros(1, length, dim)


def _assert_valid_causal_indices(
    selected: torch.Tensor,
    *,
    past_len: int,
    kv_len: int,
) -> None:
    assert selected.dtype == torch.int32
    for row in range(selected.shape[1]):
        valid = selected[0, row][selected[0, row] >= 0]
        assert torch.all(valid < past_len + row + 1)
        assert torch.all(valid < kv_len)
        assert valid.unique().numel() == valid.numel()


def test_all_visible_context_uses_dynamic_width(config: SimpleNamespace) -> None:
    torch.manual_seed(20260830)
    indexer = _indexer(config)
    hidden = torch.randn(1, 5, 16)
    cos, sin = _rope(5, config.indexer_head_dim)

    selected = indexer(hidden, cos, sin, None, past_len=0)

    assert selected.shape == (1, 5, 5)
    expected = torch.tensor(
        [
            [0, -1, -1, -1, -1],
            [0, 1, -1, -1, -1],
            [0, 1, 2, -1, -1],
            [0, 1, 2, 3, -1],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.int32,
    ).unsqueeze(0)
    torch.testing.assert_close(selected, expected)


def test_sparse_indices_are_causal_unique_and_compact(config: SimpleNamespace) -> None:
    torch.manual_seed(20260831)
    indexer = _indexer(config)
    hidden = torch.randn(1, 19, 16)
    cos, sin = _rope(19, config.indexer_head_dim)

    selected = indexer(hidden, cos, sin, None, past_len=0)

    assert selected.shape == (1, 19, 11)
    _assert_valid_causal_indices(selected, past_len=0, kv_len=19)
    last_valid = selected[0, -1][selected[0, -1] >= 0]
    assert last_valid.numel() <= config.indexer_budget + config.indexer_compress_ratio - 1
    assert 16 in last_valid.tolist()
    assert 17 in last_valid.tolist()
    assert 18 in last_valid.tolist()


def test_cached_non_aligned_tail_stays_causal(config: SimpleNamespace) -> None:
    torch.manual_seed(20260901)
    indexer = _indexer(config)
    hidden = torch.randn(1, 3, 16)
    raw_prefix = torch.randn(1, 10, config.indexer_head_dim)
    cos, sin = _rope(13, config.indexer_head_dim)

    selected = indexer(
        hidden,
        cos,
        sin,
        None,
        past_len=10,
        raw_keys_override=raw_prefix,
    )

    assert selected.shape == (1, 3, 11)
    _assert_valid_causal_indices(selected, past_len=10, kv_len=13)
