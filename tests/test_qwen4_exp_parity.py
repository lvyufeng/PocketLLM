"""Parity of our Qwen4-Exp implementation against upstream transformers.

The golden fixture is produced by `.scratch/mk_tiny_ref.py` under the
`vllm-2080ti-v015` env (transformers 5.16.1 ships `qwen4_exp`; the default
`deepseek` env does not).  It carries a tiny random model's state dict plus its
prefill logits, three cached decode-step logits, and the greedy token chain.

Run: pytest tests/test_qwen4_exp_parity.py
"""

from __future__ import annotations

import os

import pytest
import torch

from src.models.qwen4_exp.builder import build_from_state_dict
from src.models.qwen4_exp.config import Qwen4ExpTextConfig

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".scratch",
    "qwen4exp_tiny.pt",
)


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(FIXTURE):
        pytest.skip(
            f"missing {FIXTURE}; regenerate with "
            "/home/lvyufeng/miniconda3/envs/vllm-2080ti-v015/bin/python .scratch/mk_tiny_ref.py"
        )
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(scope="module")
def built(golden):
    config = Qwen4ExpTextConfig.from_dict(golden["config"])
    model = build_from_state_dict(
        config, golden["state_dict"], device="cpu", dtype=torch.float32, prefix="model"
    )
    return config, model


def test_config_roundtrip(golden):
    config = Qwen4ExpTextConfig.from_dict(golden["config"])
    raw = golden["config"]
    assert config.num_hidden_layers == raw["num_hidden_layers"]
    assert config.hc_count == raw["hc_count"]
    assert config.num_experts == raw["num_experts"]
    # "full_attention" in the checkpoint means the QSA indexer path.
    assert set(config.layer_types) <= {"linear_attention", "qwen_sparse_attention"}
    assert config.layer_types[-1] == "qwen_sparse_attention"
    assert config.ple_layer_ids == raw["ple_layer_ids"]


def test_ngram_vocab_sizes_match_upstream(golden):
    """The hashed table layout must agree or every PLE row id is wrong."""
    config = Qwen4ExpTextConfig.from_dict(golden["config"])
    sizes, offsets, padded = config.ngram_head_vocab_sizes(0)
    assert len(sizes) == config.ngram_heads
    assert offsets[0] == 0
    assert offsets[-1] + sizes[-1] <= padded
    # An in-memory model keeps one table; a saved checkpoint splits it into
    # `split_ngram_parts` equal shards. Either way the row count must match.
    table_keys = [k for k in golden["state_dict"] if "ngram_embedding" in k and k.endswith("weight")]
    table_rows = sum(golden["state_dict"][k].shape[0] for k in table_keys)
    assert table_rows == padded

    # Our per-head primes must be byte-identical to the buffers upstream baked in.
    buf_sizes = next(v for k, v in golden["state_dict"].items() if k.endswith("ngram_heads_vocab_sizes"))
    buf_offsets = next(v for k, v in golden["state_dict"].items() if k.endswith("ngram_heads_offsets"))
    assert sizes == buf_sizes.tolist()
    assert offsets == buf_offsets.tolist()


def test_prefill_logits_match(built, golden):
    config, model = built
    logits = model.forward(golden["input_ids"])
    expected = golden["prefill_logits"].float()
    assert logits.shape == expected.shape
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)


def test_prefill_argmax_matches(built, golden):
    _, model = built
    logits = model.forward(golden["input_ids"])
    assert torch.equal(logits.argmax(-1), golden["prefill_logits"].argmax(-1))


def test_cached_decode_matches(built, golden):
    """Prefill then three cached single-token steps, against upstream goldens."""
    config, model = built
    input_ids = golden["input_ids"]
    total = input_ids.shape[1] + 4
    cache = model.make_cache(batch_size=input_ids.shape[0], max_seq_len=total)

    logits = model.forward(input_ids, cache=cache, past_len=0)
    torch.testing.assert_close(logits, golden["prefill_logits"].float(), rtol=1e-5, atol=1e-5)

    expected_steps = golden["decode_logits"].float()
    greedy = golden["greedy_ids"]
    past = input_ids.shape[1]
    for step in range(expected_steps.shape[1]):
        next_id = greedy[:, past : past + 1]
        step_logits = model.forward(next_id, cache=cache, past_len=past)
        torch.testing.assert_close(
            step_logits[:, -1], expected_steps[:, step], rtol=1e-5, atol=1e-5
        )
        past += 1


def test_greedy_chain_matches(built, golden):
    """End-to-end greedy decode reproduces upstream's token ids exactly."""
    config, model = built
    input_ids = golden["input_ids"]
    expected = golden["greedy_ids"]
    n_new = expected.shape[1] - input_ids.shape[1]

    cache = model.make_cache(batch_size=1, max_seq_len=expected.shape[1] + 1)
    logits = model.forward(input_ids, cache=cache, past_len=0)
    past = input_ids.shape[1]
    produced = []
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(n_new):
        produced.append(int(next_id.item()))
        step = model.forward(next_id, cache=cache, past_len=past)
        past += 1
        next_id = step[:, -1].argmax(-1, keepdim=True)
    assert produced == expected[0, input_ids.shape[1] :].tolist()


def test_chunked_prefill_matches_single_shot(built, golden):
    """Splitting prefill into chunks must not change the final logits.

    This exercises the GatedDeltaNet conv/recurrent carry-over and the QSA
    indexer's block boundaries at non-aligned offsets, which is the failure mode
    chunked prefill introduces.
    """
    config, model = built
    input_ids = golden["input_ids"]
    reference = model.forward(input_ids)

    cache = model.make_cache(batch_size=1, max_seq_len=input_ids.shape[1] + 1)
    past = 0
    last = None
    for chunk in (3, 3, 2):
        piece = input_ids[:, past : past + chunk]
        last = model.forward(piece, cache=cache, past_len=past)
        past += chunk
    torch.testing.assert_close(last[:, -1], reference[:, -1], rtol=1e-5, atol=1e-5)
