"""Checks against the real Qwen3.8-Flash-Next checkpoint.

Skipped unless the 335 GiB checkpoint is present.  These are structural and
sharding checks, not full-model generation: they load a layer slice so they run
in seconds while still touching every weight path (both attention flavours, the
PLE layer, host-resident experts, the TP shard plan).

Point `QWEN4EXP_MODEL` at a different directory to run against another copy.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from src.models.qwen4_exp.builder import build_heterogeneous
from src.models.qwen4_exp.config import Qwen4ExpConfig
from src.models.qwen4_exp.layers import inject_into_streams
from src.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint

MODEL_DIR = os.environ.get("QWEN4EXP_MODEL", "/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next")

# Covers layers 0-7: linear x3, QSA, linear x3, QSA, with PLE on layer 2 (1-indexed).
SLICE_LAYERS = 8

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors.index.json")),
    reason=f"real checkpoint not found at {MODEL_DIR}",
)


@pytest.fixture(scope="module")
def full_config():
    return Qwen4ExpConfig.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def checkpoint():
    return Qwen4ExpCheckpoint(MODEL_DIR, store=MmapSafetensors(MODEL_DIR))


@pytest.fixture(scope="module")
def sliced_config(full_config):
    import copy

    tc = copy.deepcopy(full_config.text_config)
    tc.num_hidden_layers = SLICE_LAYERS
    tc.layer_types = tc.layer_types[:SLICE_LAYERS]
    tc.ple_layer_ids = [i for i in tc.ple_layer_ids if i <= SLICE_LAYERS]
    return tc


def test_config_matches_checkpoint(full_config):
    tc = full_config.text_config
    assert tc.num_hidden_layers == 48
    assert tc.layer_types.count("linear_attention") == 36
    assert tc.layer_types.count("qwen_sparse_attention") == 12
    assert tc.num_experts == 512 and tc.num_experts_per_tok == 10
    assert tc.hc_count == 4 and tc.ple_layer_ids == [2]
    assert tc.vocab_size == 248320


def test_every_expected_tensor_present(full_config, checkpoint):
    """A missing name means a silently wrong layer, so check all 48."""
    tc = full_config.text_config
    store = checkpoint.store
    missing = []
    for layer_idx in range(tc.num_hidden_layers):
        base = f"model.language_model.layers.{layer_idx}"
        expected = [
            f"{base}.mlp.experts.gate_up_proj",
            f"{base}.mlp.experts.down_proj",
            f"{base}.mlp.gate.weight",
            f"{base}.mlp.shared_expert_gate.weight",
            f"{base}.mlp.shared_expert.gate_proj.weight",
            f"{base}.mlp.shared_expert.up_proj.weight",
            f"{base}.mlp.shared_expert.down_proj.weight",
        ]
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            expected += [
                f"{base}.{hc}.hc_norm.weight",
                f"{base}.{hc}.input_mix_weight_down.weight",
                f"{base}.{hc}.input_mix_weight_up.weight",
                f"{base}.{hc}.block_inject_weight.weight",
            ]
        if tc.is_linear_layer(layer_idx):
            expected += [
                f"{base}.linear_attn.in_proj_qkv.weight",
                f"{base}.linear_attn.in_proj_z.weight",
                f"{base}.linear_attn.in_proj_a.weight",
                f"{base}.linear_attn.in_proj_b.weight",
                f"{base}.linear_attn.conv1d.weight",
                f"{base}.linear_attn.out_proj.weight",
                f"{base}.linear_attn.norm.weight",
                f"{base}.linear_attn.A_log",
                f"{base}.linear_attn.dt_bias",
            ]
        else:
            expected += [
                f"{base}.self_attn.q_proj.weight",
                f"{base}.self_attn.k_proj.weight",
                f"{base}.self_attn.v_proj.weight",
                f"{base}.self_attn.o_proj.weight",
                f"{base}.self_attn.q_norm.weight",
                f"{base}.self_attn.k_norm.weight",
                f"{base}.self_attn.indexer.index_qk_proj.weight",
                f"{base}.self_attn.indexer.q_layernorm.weight",
                f"{base}.self_attn.indexer.k_layernorm.weight",
            ]
        missing += [k for k in expected if k not in store]
    assert missing == []


def test_ple_table_layout_matches_config(full_config, checkpoint):
    """Our computed hash-table size must equal the shards actually on disk.

    If these disagree, every PLE row id is wrong and the model degrades silently.
    """
    tc = full_config.text_config
    ple_layer = tc.ple_layer_ids[0] - 1
    shard_keys = checkpoint.ngram_shard_keys(ple_layer)
    assert len(shard_keys) == tc.split_ngram_parts
    rows_on_disk = sum(checkpoint.store.entries[k].shape[0] for k in shard_keys)
    _, _, padded = tc.ngram_head_vocab_sizes(0)
    assert rows_on_disk == padded


def test_tp4_shard_geometry_divides(full_config):
    tc = full_config.text_config
    world = 4
    assert tc.num_attention_heads % world == 0
    assert tc.linear_num_value_heads % world == 0
    assert tc.linear_num_key_heads % world == 0
    assert tc.shared_expert_intermediate_size % world == 0
    assert tc.vocab_size % world == 0
    # Each rank's q heads must map onto a whole number of kv heads.
    q_per_rank = tc.num_attention_heads // world
    groups = tc.num_attention_heads // tc.num_key_value_heads
    for rank in range(world):
        kv_start = (rank * q_per_rank) // groups
        kv_end = ((rank + 1) * q_per_rank - 1) // groups + 1
        assert q_per_rank % (kv_end - kv_start) == 0


def test_expert_rows_alias_packed_tensor(checkpoint):
    gate_up_all = checkpoint.store.view(checkpoint.expert_key(0, "gate_up_proj"))
    down_all = checkpoint.store.view(checkpoint.expert_key(0, "down_proj"))
    assert gate_up_all.shape == (512, 1280, 2560)
    assert down_all.shape == (512, 2560, 640)
    for expert_id in (0, 137, 511):
        gate_up, down = checkpoint.expert_rows(0, expert_id)
        assert torch.equal(gate_up, gate_up_all[expert_id])
        assert torch.equal(down, down_all[expert_id])


def _lockstep_tp_forward(models, input_ids: torch.Tensor) -> torch.Tensor:
    """Run all ranks block-by-block, summing partials between blocks."""
    driver = models[0]
    config = driver.config
    batch, seq = input_ids.shape
    cos, sin = driver._rope_cache(seq, batch)

    history = None
    if any(layer.ple is not None for layer in driver.layers):
        pad = torch.full((batch, config.ngram_size - 1), config.primary_eos_token_id, dtype=torch.long)
        history = torch.cat([pad, input_ids.cpu()], dim=1)

    hidden = driver.embed_tokens(input_ids).to(driver.dtype).repeat(1, 1, config.hc_count)
    for idx in range(config.num_hidden_layers):
        layers = [m.layers[idx] for m in models]
        state = hidden
        if layers[0].ple is not None:
            state = state + layers[0].ple(state, history, seq, use_cache=False)
        mixed, hyper, inject = layers[0].attn_hc(state)
        parts = [
            (l.attn(mixed, None) if l.is_linear else l.attn(mixed, cos, sin, None, past_len=0))
            for l in layers
        ]
        state = inject_into_streams(sum(parts), hyper, inject)

        mixed, hyper, inject = layers[0].mlp_hc(state)
        flat = mixed.reshape(-1, mixed.shape[-1])
        weights, indices = layers[0].router(flat)
        mlp_parts = []
        for layer in layers:
            routed = layer.moe(idx, flat, indices, weights)
            shared = layer.shared_expert(flat)
            shared = torch.sigmoid(F.linear(flat, layer.shared_expert_gate)) * shared
            mlp_parts.append((routed + shared).reshape(mixed.shape))
        hidden = inject_into_streams(sum(mlp_parts), hyper, inject)

    hidden = driver.final_mixer(hidden)
    return torch.cat([F.linear(hidden, m.lm_head) for m in models], dim=-1)


@pytest.mark.slow
def test_tp4_matches_single_rank_on_real_weights(sliced_config, checkpoint):
    """The TP4 shard plan must reproduce single-rank logits on real weights.

    float32 so the comparison measures the sharding plan rather than bf16
    rounding; the residual difference is reduction-order noise.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    tc = sliced_config

    single = build_heterogeneous(tc, checkpoint, device=device, dtype=dtype, rank=0, world_size=1)
    shards = [
        build_heterogeneous(
            tc, checkpoint, device=device, dtype=dtype, rank=r, world_size=4, all_reduce=lambda t: t
        )
        for r in range(4)
    ]

    ids = torch.tensor([[9707, 11, 1879, 0, 1096, 374, 264, 1273, 315, 279, 1614]], dtype=torch.long)
    with torch.no_grad():
        reference = single.forward(ids)
        sharded = _lockstep_tp_forward(shards, ids)

    assert torch.isfinite(reference).all() and torch.isfinite(sharded).all()
    assert torch.equal(reference.argmax(-1), sharded.argmax(-1))
    rel = (reference - sharded).abs().max() / reference.abs().max()
    assert rel < 1e-5, f"TP4 diverged from single rank: max_rel={rel.item():.3e}"


@pytest.mark.slow
def test_sliced_model_decodes_with_cache(sliced_config, checkpoint):
    """Cached decode on real weights must match an uncached re-forward.

    Catches cache bugs the tiny fixture can miss (real head counts, real conv
    dims, a real 2048-token indexer budget).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    model = build_heterogeneous(
        sliced_config, checkpoint, device=device, dtype=dtype, rank=0, world_size=1
    )
    ids = torch.tensor([[9707, 11, 1879, 0, 1096, 374, 264, 1273]], dtype=torch.long)

    with torch.no_grad():
        cache = model.make_cache(batch_size=1, max_seq_len=ids.shape[1] + 2)
        model.forward(ids[:, :-1], cache=cache, past_len=0)
        cached = model.forward(ids[:, -1:], cache=cache, past_len=ids.shape[1] - 1)
        uncached = model.forward(ids)

    torch.testing.assert_close(cached[:, -1], uncached[:, -1], rtol=1e-3, atol=1e-3)
