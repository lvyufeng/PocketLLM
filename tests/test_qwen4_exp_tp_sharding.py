"""Tensor-parallel sharding correctness for Qwen4-Exp.

Runs `build_heterogeneous` for every rank of a simulated world in one process and
sums the per-rank partial outputs by hand, which is exactly what NCCL does at
runtime.  If the sharding plan is wrong (a head split at the wrong stride, a
row/column mismatch, an unreduced partial), the summed logits diverge from the
single-rank result and from upstream's goldens.

The fixture is a tiny on-disk checkpoint in the real layout (sharded
safetensors, `model.language_model.*` names, split n-gram table); regenerate it
with `.scratch/mk_tp_ref.py` under the vllm env.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.models.qwen4_exp.builder import build_heterogeneous
from src.models.qwen4_exp.config import Qwen4ExpConfig
from src.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".scratch", "tp_tiny"
)


@pytest.fixture(scope="module")
def checkpoint_dir():
    if not os.path.exists(os.path.join(ROOT, "golden.pt")):
        pytest.skip(
            f"missing {ROOT}; regenerate with "
            "/home/lvyufeng/miniconda3/envs/vllm-2080ti-v015/bin/python .scratch/mk_tp_ref.py"
        )
    return ROOT


@pytest.fixture(scope="module")
def golden(checkpoint_dir):
    return torch.load(os.path.join(checkpoint_dir, "golden.pt"), weights_only=False)


def _build(checkpoint_dir: str, rank: int, world_size: int, collector: list | None = None):
    config = Qwen4ExpConfig.from_pretrained(checkpoint_dir)
    checkpoint = Qwen4ExpCheckpoint(checkpoint_dir, store=MmapSafetensors(checkpoint_dir))

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        # Standing in for NCCL: record the partial and hand it straight back, so
        # a single rank's forward stays runnable in isolation.
        if collector is not None:
            collector.append(tensor.detach().clone())
        return tensor

    model = build_heterogeneous(
        config.text_config,
        checkpoint,
        device="cpu",
        dtype=torch.float32,
        rank=rank,
        world_size=world_size,
        all_reduce=all_reduce if world_size > 1 else None,
    )
    return config, model


def test_mmap_views_match_safetensors(checkpoint_dir):
    """The zero-copy reader must agree with the reference safetensors loader."""
    from safetensors import safe_open

    store = MmapSafetensors(checkpoint_dir)
    checked = 0
    for file_name in sorted({e.file_name for e in store.entries.values()}):
        with safe_open(os.path.join(checkpoint_dir, file_name), framework="pt") as f:
            for key in f.keys():
                torch.testing.assert_close(store.view(key), f.get_tensor(key), rtol=0, atol=0)
                checked += 1
    assert checked > 50


def test_single_rank_matches_golden(checkpoint_dir, golden):
    """world_size=1 through the heterogeneous builder reproduces upstream."""
    _, model = _build(checkpoint_dir, rank=0, world_size=1)
    logits = model.forward(golden["input_ids"])
    torch.testing.assert_close(logits, golden["prefill_logits"].float(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_partials_sum_to_golden(checkpoint_dir, golden, world_size):
    """Sharded ranks' partial block outputs must sum to the unsharded result.

    Each rank is run to completion independently while its per-block partials are
    recorded.  The all-reduce is then simulated by checking that the summed
    partials of the first block match the single-rank block output — a full
    lock-step simulation would need the reduce to feed back into the next layer,
    which `test_tp_logits_match_golden` covers by interleaving the ranks.
    """
    _, single = _build(checkpoint_dir, rank=0, world_size=1)
    reference = []
    for layer in single.layers:
        layer.all_reduce = lambda t, sink=reference: (sink.append(t.detach().clone()), t)[1]
    single.forward(golden["input_ids"])

    per_rank = []
    for rank in range(world_size):
        sink: list[torch.Tensor] = []
        _, model = _build(checkpoint_dir, rank=rank, world_size=world_size, collector=sink)
        model.forward(golden["input_ids"])
        per_rank.append(sink)

    assert all(len(sink) == len(reference) for sink in per_rank)
    # The first block sees identical inputs on every rank (the embedding is
    # replicated), so its partials must sum exactly to the unsharded output.
    summed = sum(sink[0] for sink in per_rank)
    torch.testing.assert_close(summed, reference[0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_logits_match_golden(checkpoint_dir, golden, world_size):
    """Full lock-step TP forward: reduce after every block, then gather logits.

    Ranks share one barrier so each block's reduced output feeds the next block
    on every rank, reproducing the real NCCL execution order.
    """
    models = [_build(checkpoint_dir, rank=r, world_size=world_size)[1] for r in range(world_size)]
    logits = _lockstep_forward(models, golden["input_ids"], world_size)
    torch.testing.assert_close(logits, golden["prefill_logits"].float(), rtol=1e-4, atol=1e-4)


def _lockstep_forward(models, input_ids: torch.Tensor, world_size: int) -> torch.Tensor:
    """Drive all ranks block-by-block, reducing between blocks.

    Mirrors `Qwen4ExpModel.forward` but advances every rank one layer at a time
    so a block's reduced output becomes the next block's input everywhere.
    """
    import torch.nn.functional as F

    from src.models.qwen4_exp.layers import inject_into_streams

    driver = models[0]
    config = driver.config
    batch_size, seq_len = input_ids.shape
    cos, sin = driver._rope_cache(seq_len, batch_size)

    token_history = None
    if any(layer.ple is not None for layer in driver.layers):
        pad = torch.full(
            (batch_size, config.ngram_size - 1),
            config.primary_eos_token_id,
            dtype=torch.long,
        )
        token_history = torch.cat([pad, input_ids], dim=1)

    hidden = driver.embed_tokens(input_ids).to(driver.dtype)
    hidden = hidden.repeat(1, 1, config.hc_count)

    for layer_idx in range(config.num_hidden_layers):
        layers = [m.layers[layer_idx] for m in models]
        # PLE and the hyper-connection mixers are replicated, so rank 0's copy
        # is authoritative for the shared parts of the block.
        state = hidden
        if layers[0].ple is not None:
            state = state + layers[0].ple(state, token_history, seq_len, use_cache=False)

        mixed, hyper_input, inject = layers[0].attn_hc(state)
        attn_partials = []
        for layer in layers:
            if layer.is_linear:
                attn_partials.append(layer.attn(mixed, None))
            else:
                attn_partials.append(layer.attn(mixed, cos, sin, None, past_len=0))
        state = inject_into_streams(sum(attn_partials), hyper_input, inject)

        mixed, hyper_input, inject = layers[0].mlp_hc(state)
        flat = mixed.reshape(-1, mixed.shape[-1])
        weights, indices = layers[0].router(flat)
        mlp_partials = []
        for layer in layers:
            routed = layer.moe(layer_idx, flat, indices, weights)
            shared = layer.shared_expert(flat)
            shared = torch.sigmoid(F.linear(flat, layer.shared_expert_gate)) * shared
            mlp_partials.append((routed + shared).reshape(mixed.shape))
        hidden = inject_into_streams(sum(mlp_partials), hyper_input, inject)

    hidden = driver.final_mixer(hidden)
    # lm_head is vocab-sharded; concatenating the shards is the all-gather.
    return torch.cat([F.linear(hidden, m.lm_head) for m in models], dim=-1)


def test_expert_partition_covers_all_experts(checkpoint_dir):
    """Round-robin expert ownership must be a partition: no gaps, no overlap."""
    config = Qwen4ExpConfig.from_pretrained(checkpoint_dir)
    world_size = 4
    owners = {}
    for expert_id in range(config.text_config.num_experts):
        rank = expert_id % world_size
        owners.setdefault(rank, []).append(expert_id)
    flat = sorted(e for ids in owners.values() for e in ids)
    assert flat == list(range(config.text_config.num_experts))
    assert len(owners) == world_size


def test_host_expert_rows_match_dense(checkpoint_dir):
    """`expert_rows` must alias the same values as the packed 3D tensor."""
    checkpoint = Qwen4ExpCheckpoint(checkpoint_dir, store=MmapSafetensors(checkpoint_dir))
    gate_up_all = checkpoint.store.view(checkpoint.expert_key(0, "gate_up_proj"))
    down_all = checkpoint.store.view(checkpoint.expert_key(0, "down_proj"))
    for expert_id in (0, 3, gate_up_all.shape[0] - 1):
        gate_up, down = checkpoint.expert_rows(0, expert_id)
        torch.testing.assert_close(gate_up, gate_up_all[expert_id], rtol=0, atol=0)
        torch.testing.assert_close(down, down_all[expert_id], rtol=0, atol=0)
