"""Tests for `src/models/mimo_v2/config.py`.

The checkpoint is a hybrid: 9 global-attention layers and 39 sliding-window ones
inside one 48-layer stack, and the two families differ in KV head count, in RoPE
base and in whether the layer carries an attention sink bias. So `qkv_proj`'s
output is 13568 on one family and 14848 on the other, and the schema exists to
make that hard to get wrong rather than to parse JSON.

Three levels, deliberately:

* The *resolution* contracts, on a hand-written config. These are the tests that
  matter, because every one of them pins a case where the reference's reading of
  an absent field is the opposite of the obvious one -- an absent layer pattern
  means all-global, an absent `moe_layer_freq` means all-dense, an absent
  `attention_projection_layout` means `"split"` against the checkpoint's
  `"fused_qkv"`, and `attention_value_scale = None` means no scaling rather than
  "unstated". A future edit that "simplifies" any of those fails here.
* `verify()`, which has to accept the released file and reject configs the
  reference either raises on or silently ignores.
* The checkpoint's own two config files when they are on this host, which is where
  the shape contract is checked against the real thing. Those tests skip elsewhere
  rather than weakening the claim -- a skip is a skip, not a pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.mimo_v2.config import (
    MimoV2Config,
    MimoV2DraftConfig,
    MimoV2QuantSpec,
    MimoV2TextConfig,
    describe,
    from_dict,
    load_config,
)

REPO = Path(__file__).resolve().parent.parent

CHECKPOINT = Path("/mnt/data3/MiMo-V2.6-Flash-RL")
HF_CONFIG = CHECKPOINT / "config.json"
DRAFT_CONFIG = CHECKPOINT / "dflash" / "config.json"
HAS_CHECKPOINT = HF_CONFIG.is_file()

# What the release states, pinned. These are facts about the checkpoint rather
# than about the schema, so they are written out here and then also read back from
# the file when it is present -- the two together are what make the shape contract
# testable without a 177 GiB download.
RELEASED = {
    "model_type": "mimo_v2",
    "architectures": ["MiMoV2ForCausalLM"],
    "vocab_size": 152576,
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 192,
    "v_head_dim": 128,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 8,
    "swa_head_dim": 192,
    "swa_v_head_dim": 128,
    "rope_theta": 10_000_000.0,
    "swa_rope_theta": 10_000.0,
    "partial_rotary_factor": 0.334,
    "attention_value_scale": 0.707,
    "attention_projection_layout": "fused_qkv",
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "sliding_window": 128,
    "hybrid_layer_pattern": [
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    ],
    "moe_layer_freq": [0] + [1] * 47,
    "intermediate_size": 16384,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": None,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
    "moe_router_dtype": "bfloat16",
    "num_nextn_predict_layers": 3,
    "max_position_embeddings": 1048576,
    "layernorm_epsilon": 1e-6,
    "attention_chunk_size": 128,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
    "image_token_id": 151655,
}

GA_LAYERS = (0, 5, 11, 17, 23, 29, 35, 41, 47)
SWA_LAYERS = tuple(i for i in range(48) if i not in GA_LAYERS)


def released(**overrides) -> MimoV2TextConfig:
    raw = {**RELEASED, **overrides}
    return MimoV2TextConfig.from_dict({k: v for k, v in raw.items() if v is not None})


def released_config(**overrides) -> MimoV2Config:
    raw = {**RELEASED, **overrides}
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "mxfp4_block_size": 32,
        "store_dtype": "mxfp4",
        "ignored_layers": (
            [f"model.layers.{i}.self_attn.o_proj" for i in range(48)]
            + ["model.decoder.self_attn.o_proj"]
        ),
    }
    return from_dict({k: v for k, v in raw.items() if v is not None})


# ---------------------------------------------------------------------------
# The shape contract: the reason the module exists
# ---------------------------------------------------------------------------


def test_ga_and_swa_have_different_fused_widths():
    """The one number a reader is most likely to derive once and reuse."""
    text = released()
    ga = text.attention(0)
    swa = text.attention(1)

    assert ga.qkv_out == 64 * 192 + 4 * 192 + 4 * 128 == 13568
    assert swa.qkv_out == 64 * 192 + 8 * 192 + 8 * 128 == 14848
    assert ga.qkv_out != swa.qkv_out

    # The o_proj input is the same in both: the query count and the value width
    # do not change, only the KV count does.
    assert ga.o_in == swa.o_in == 64 * 128 == 8192


def test_layer_lists_partition_the_stack():
    text = released()
    assert text.global_layer_indices == GA_LAYERS
    assert text.swa_layer_indices == SWA_LAYERS
    assert set(text.global_layer_indices) | set(text.swa_layer_indices) == set(range(48))
    assert not (set(text.global_layer_indices) & set(text.swa_layer_indices))
    assert len(text.swa_layer_indices) == 39
    assert len(text.global_layer_indices) == 9


def test_family_attributes_track_is_swa():
    text = released()
    for idx in range(text.num_hidden_layers):
        shape = text.attention(idx)
        assert shape.layer_idx == idx
        if shape.is_swa:
            assert (shape.num_kv_heads, shape.rope_theta, shape.sliding_window, shape.has_sink) == (
                8, 10_000.0, 128, True,
            )
            assert shape.family == "swa"
        else:
            assert (shape.num_kv_heads, shape.rope_theta, shape.sliding_window, shape.has_sink) == (
                4, 10_000_000.0, None, False,
            )
            assert shape.family == "ga"


def test_value_dim_is_narrower_than_the_query_key_dim():
    """`v_head_dim` 128 against `head_dim` 192, so `o_in` is not `q_size`."""
    text = released()
    shape = text.attention(0)
    assert shape.head_dim == 192
    assert shape.v_head_dim == 128
    assert shape.o_in != shape.q_size
    assert shape.q_size == 64 * 192


def test_partial_rotation_leaves_a_tail():
    text = released()
    shape = text.attention(0)
    assert shape.rope_dim == int(192 * 0.334) == 64
    assert shape.rope_dim % 2 == 0
    assert shape.rope_dim < shape.head_dim
    assert shape.partial_rotary_factor == pytest.approx(0.334)


def test_ffn_kind_partitions_the_stack():
    text = released()
    assert text.dense_layer_indices == (0,)
    assert text.moe_layer_indices == tuple(range(1, 48))
    assert text.ffn_kind(0) == "dense"
    assert text.ffn_kind(1) == "moe"
    assert text.ffn_intermediate_size(0) == 16384
    assert text.ffn_intermediate_size(1) == 2048


def test_attention_rejects_an_out_of_range_layer():
    text = released()
    with pytest.raises(IndexError):
        text.attention(48)
    with pytest.raises(IndexError):
        text.attention(-1)


def test_total_layers_includes_the_draft_heads():
    assert released().total_layers == 48 + 3


# ---------------------------------------------------------------------------
# The "unstated is not false" contracts
# ---------------------------------------------------------------------------


def test_absent_layer_pattern_means_all_global_not_all_swa():
    """The reference resolves an absent pattern to zeros, i.e. no SWA at all.

    The released checkpoint is 39-of-48 sliding-window, so reading an absent
    pattern as the hybrid one is the plausible mistake, and it inverts the model.
    """
    text = released(hybrid_layer_pattern=None)
    assert text.resolved_hybrid_layer_pattern == (0,) * 48
    assert text.swa_layer_indices == ()
    assert text.global_layer_indices == tuple(range(48))
    assert {text.attention(i).qkv_out for i in range(48)} == {13568}


def test_hybrid_block_size_derives_the_pattern_only_when_no_pattern_is_given():
    derived = released(hybrid_layer_pattern=None, hybrid_block_size=5)
    assert derived.resolved_hybrid_layer_pattern == tuple(
        0 if (i + 1) % 5 == 0 else 1 for i in range(48)
    )
    # The pattern wins whenever it is present.
    both = released(hybrid_block_size=5)
    assert both.resolved_hybrid_layer_pattern == tuple(RELEASED["hybrid_layer_pattern"])
    assert any("ignores hybrid_block_size" in p for p in both.verify())


def test_absent_moe_freq_means_all_dense():
    """Omitted, the reference builds a dense model with no error at all."""
    text = released(moe_layer_freq=None)
    assert text.resolved_moe_layer_freq == (False,) * 48
    assert text.moe_layer_indices == ()
    assert text.dense_layer_indices == tuple(range(48))


def test_int_moe_freq_is_a_period_not_a_flag():
    assert released(moe_layer_freq=1).moe_layer_indices == tuple(range(48))
    assert released(moe_layer_freq=2).moe_layer_indices == tuple(range(0, 48, 2))
    assert released(moe_layer_freq=3).moe_layer_indices == tuple(range(0, 48, 3))
    assert released(moe_layer_freq=0).moe_layer_indices == ()


def test_absent_norm_topk_prob_is_true():
    """Absent means unsaid, and the reference's own default is `True`."""
    assert released(norm_topk_prob=None).resolved_norm_topk_prob is True
    assert MimoV2TextConfig().resolved_norm_topk_prob is True
    # An explicit False is a different claim and has to survive.
    assert released(norm_topk_prob=False).resolved_norm_topk_prob is False


def test_absent_projection_layout_is_split_not_the_checkpoint_value():
    """The reference's default is the opposite of what this checkpoint declares.

    This is a guard on the default rather than on the checkpoint: anyone who
    "helpfully" changes it to `fused_qkv` breaks every other MiMo checkpoint.
    """
    assert MimoV2TextConfig().resolved_projection_layout == "split"
    assert released(attention_projection_layout=None).resolved_projection_layout == "split"
    assert released().resolved_projection_layout == "fused_qkv"
    assert released(attention_projection_layout=None).attention(0).projection_layout == "split"


def test_value_scale_none_is_a_value_and_not_unsaid():
    """Unlike `norm_topk_prob`, `None` here means the multiply does not happen."""
    assert released(attention_value_scale=None).attention(0).value_scale is None
    assert released().attention(0).value_scale == pytest.approx(0.707)
    assert MimoV2TextConfig().attention(0).value_scale is None


def test_absent_swa_rope_theta_falls_back_to_the_global_one():
    assert MimoV2TextConfig(rope_theta=1e7).resolved_swa_rope_theta == pytest.approx(1e7)
    # The released file states both, and they differ by three orders of magnitude.
    text = released()
    assert text.resolved_swa_rope_theta == pytest.approx(1e4)
    assert text.resolved_rope_theta == pytest.approx(1e7)
    assert text.attention(1).rope_theta != text.attention(0).rope_theta


def test_rope_parameters_override_the_flat_fields():
    text = released(rope_parameters={"rope_theta": 5e6, "partial_rotary_factor": 0.5})
    assert text.resolved_rope_theta == pytest.approx(5e6)
    assert text.resolved_partial_rotary_factor == pytest.approx(0.5)
    assert text.attention(0).rope_dim == 96


def test_absent_head_dim_falls_back_to_hidden_over_heads():
    """The fallback is 64 against the checkpoint's 192, and nothing raises."""
    text = released(head_dim=None, v_head_dim=None)
    assert text.resolved_head_dim == 4096 // 64 == 64
    assert text.attention(0).qkv_out == 64 * (64 + 4 + 4)
    assert any("head_dim is unstated" in p for p in text.verify())


def test_absent_routed_scaling_factor_is_one():
    assert released(routed_scaling_factor=None).resolved_routed_scaling_factor == 1.0
    assert released(routed_scaling_factor=2.5).resolved_routed_scaling_factor == 2.5


def test_absent_swa_head_fields_defer_to_the_global_ones():
    text = released(swa_num_key_value_heads=None, swa_head_dim=None, swa_v_head_dim=None)
    shape = text.attention(1)
    assert shape.num_kv_heads == text.num_key_value_heads
    assert shape.head_dim == text.resolved_head_dim
    # An absent `swa_v_head_dim` falls back to the SWA *head* dim and not to the
    # global value dim, so it lands on 192 here while the global family's value
    # width is 128. The reference resolves it that way and the difference changes
    # the o_proj input width, so it is worth pinning.
    assert shape.v_head_dim == text.resolved_swa_head_dim == 192
    assert shape.v_head_dim != text.resolved_v_head_dim


def test_window_falls_back_to_sliding_window_size():
    assert released(sliding_window=None, sliding_window_size=64).resolved_window == 64
    assert MimoV2TextConfig().resolved_window is None


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


def test_verify_accepts_the_released_values():
    assert released().verify() == []
    assert released_config().verify() == []


def test_verify_rejects_an_unsupported_scoring_function():
    """The reference's gate implements sigmoid and raises on anything else."""
    problems = released(scoring_func="softmax").verify()
    assert any("scoring_func" in p for p in problems)


def test_verify_rejects_an_unsupported_topk_method():
    assert any("topk_method" in p for p in released(topk_method="greedy").verify())


def test_verify_requires_the_group_fields_when_the_gate_reshapes_by_them():
    assert any("n_group" in p for p in released(n_group=None).verify())
    assert any("topk_group" in p for p in released(topk_group=None).verify())
    assert any("topk_group" in p and "exceeds" in p for p in released(topk_group=2).verify())
    assert any("does not divide" in p for p in released(n_group=7).verify())


def test_verify_requires_a_window_when_a_layer_is_sliding_window():
    problems = released(sliding_window=None).verify()
    assert any("sliding_window" in p for p in problems)
    # ...and does not when no layer is.
    all_global = released(sliding_window=None, hybrid_layer_pattern=None)
    assert not any("sliding_window" in p for p in all_global.verify())


def test_verify_rejects_an_odd_rotary_dimension():
    # int(192 * 0.34) == 65, which the reference rejects; 0.336 would round to 64
    # and pass, so the factor has to be chosen for the floor rather than the value.
    text = released(partial_rotary_factor=0.34)
    assert int(text.resolved_head_dim * 0.34) % 2 == 1
    assert any("even rotary dimension" in p for p in text.verify())


def test_verify_rejects_a_rotary_dimension_wider_than_the_head():
    assert any("exceeds head_dim" in p for p in released(partial_rotary_factor=1.5).verify())


def test_verify_rejects_indivisible_head_counts():
    assert any("divisible" in p for p in released(swa_num_key_value_heads=3).verify())


def test_verify_rejects_an_unknown_projection_layout():
    assert any("attention_projection_layout" in p for p in released(
        attention_projection_layout="everything"
    ).verify())


def test_verify_rejects_a_truncated_layer_pattern():
    problems = released(hybrid_layer_pattern=[0, 1]).verify()
    assert any("hybrid_layer_pattern has 2 entries" in p for p in problems)


def test_verify_rejects_a_non_binary_layer_pattern():
    assert any("must be 0/1" in p for p in released(hybrid_layer_pattern=[0, 2]).verify())


def test_verify_rejects_a_truncated_moe_freq():
    assert any("moe_layer_freq has 3 entries" in p for p in released(moe_layer_freq=[0, 1, 1]).verify())


def test_verify_catches_the_ignored_layer_count_disagreeing_with_the_stack():
    """The 48 layer-scoped o_proj entries are a claim about the weight layout."""
    raw = {**RELEASED}
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": [f"model.layers.{i}.self_attn.o_proj" for i in range(47)],
    }
    problems = from_dict({k: v for k, v in raw.items() if v is not None}).verify()
    assert any("47 layer-scoped attention o_proj" in p for p in problems)


def test_a_dangling_ignored_layer_is_a_note_and_not_a_problem():
    """The released file's 49th entry names a prefix the checkpoint never uses.

    It matches no tensor, so a correct implementation ignores it and the
    checkpoint has to keep loading -- which is why it is reported apart from the
    load-gate problems.
    """
    config = released_config()
    assert config.verify() == []
    assert config.dangling_ignored_layers() == ["model.decoder.self_attn.o_proj"]
    assert any("naming no layer" in note for note in config.runtime_notes())


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def test_quant_spec_reads_all_three_schemes():
    config = released_config()
    quant = config.quant
    assert quant is not None
    assert quant.dense_is_fp8 and quant.experts_are_mxfp4
    assert quant.weight_block_size == (128, 128)
    assert quant.mxfp4_block_size == 32
    assert len(quant.ignored_layer_scoped) == 48
    assert quant.verify() == []


def test_quant_ignores_accepts_weight_and_scale_suffixes():
    quant = MimoV2QuantSpec.from_dict(
        {"quant_method": "fp8", "weight_block_size": [128, 128],
         "ignored_layers": ["model.layers.3.self_attn.o_proj"]}
    )
    assert quant.ignores("model.layers.3.self_attn.o_proj")
    assert quant.ignores("model.layers.3.self_attn.o_proj.weight")
    assert quant.ignores("model.layers.3.self_attn.o_proj.weight_scale_inv")
    assert not quant.ignores("model.layers.3.self_attn.qkv_proj.weight")


def test_quant_verify_rejects_a_foreign_block_geometry():
    quant = MimoV2QuantSpec.from_dict(
        {"quant_method": "fp8", "weight_block_size": [64, 64], "ignored_layers": [
            "model.layers.0.self_attn.o_proj"]}
    )
    assert any("weight_block_size" in p for p in quant.verify())


def test_the_expert_block_is_reported_as_kernel_compatible():
    """Block 32 is what `fp4_e2m1_e8m0_matvec_cuda` indexes, so it is a note."""
    notes = released_config().runtime_notes()
    assert any("no requantization" in note for note in notes)
    other = released_config()
    other = MimoV2Config(
        text=other.text,
        quant=MimoV2QuantSpec.from_dict(
            {"quant_method": "fp8", "store_dtype": "mxfp4", "mxfp4_block_size": 64}
        ),
    )
    assert any("indexes scales at a block of 32" in note for note in other.runtime_notes())


def test_runtime_notes_cover_the_traps_an_implementer_hits():
    notes = " | ".join(released_config().runtime_notes())
    assert "attention_value_scale" in notes          # value scaling is not in any kernel
    assert "no shared expert" in notes               # `n_shared_experts` is inert
    assert "moe_router_dtype" in notes               # the gate upcasts to fp32 anyway
    assert "attention_chunk_size" in notes           # stored and never read
    assert "sink bias" in notes                      # sink is per-family, not global
    assert "fused" in notes and "different output widths" in notes


def test_a_config_without_a_quantization_block_parses():
    config = from_dict({"model_type": "mimo_v2", "hidden_size": 64, "num_hidden_layers": 2})
    assert config.quant is None
    assert "unquantized" in describe(config)
    assert config.runtime_notes() == []


# ---------------------------------------------------------------------------
# The drafter's separate config
# ---------------------------------------------------------------------------

RELEASED_DRAFT = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "v_head_dim": 128,
    "partial_rotary_factor": 0.5,
    "rope_theta": 10000.0,
    "sliding_window": 1024,
    "use_sliding_window": True,
    "layer_types": ["sliding_attention"] * 5,
    "is_causal": False,
    "num_target_layers": 48,
    "dflash_config": {
        "target_layer_ids": [0, 11, 23, 35, 47],
        "mask_token_id": 151675,
        "block_size": 8,
        "attention_value_scale": 0.612,
        "attention_sink_bias": True,
        "num_anchors": 4096,
        "loss_decay_gamma": 7.0,
    },
}


def test_draft_config_merges_the_nested_block():
    draft = MimoV2DraftConfig.from_dict(RELEASED_DRAFT)
    assert draft.block_size == 8
    assert draft.target_layer_ids == (0, 11, 23, 35, 47)
    assert draft.mask_token_id == 151675
    assert draft.attention_value_scale == pytest.approx(0.612)
    assert draft.num_anchors == 4096
    assert draft.verify() == []


def test_the_drafter_is_not_causal():
    """`is_causal` is False: a causal attention path cannot run the drafter."""
    draft = MimoV2DraftConfig.from_dict(RELEASED_DRAFT)
    assert draft.is_causal is False
    assert draft.all_layers_are_swa
    assert draft.verify() == []
    assert any("bidirectionally" in p for p in
               MimoV2DraftConfig.from_dict({**RELEASED_DRAFT, "is_causal": True}).verify())


def test_draft_context_and_projection_widths():
    draft = MimoV2DraftConfig.from_dict(RELEASED_DRAFT)
    # fc: five target-layer hidden states concatenated, down to hidden_size.
    assert draft.context_width == 5 * 4096 == 20480
    # Separate q/k/v, unlike the backbone's fused projection.
    assert draft.qkv_out == 64 * 128 + 2 * 8 * 128 == 10240
    # rope_dim 64 here is a coincidence of 0.5 x 128 matching the backbone's
    # 0.334 x 192, which is exactly why it is derived rather than shared.
    assert draft.rope_dim == 64


def test_draft_verify_requires_the_target_layer_ids():
    raw = {k: v for k, v in RELEASED_DRAFT.items() if k != "dflash_config"}
    assert any("target_layer_ids is unstated" in p for p in MimoV2DraftConfig.from_dict(raw).verify())


def test_draft_verify_rejects_a_target_past_the_model():
    raw = {**RELEASED_DRAFT, "dflash_config": {**RELEASED_DRAFT["dflash_config"],
                                               "target_layer_ids": [0, 48]}}
    assert any("reaches past" in p for p in MimoV2DraftConfig.from_dict(raw).verify())


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def test_load_config_reads_a_file_and_a_directory(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**RELEASED, "architectures": ["MiMoV2ForCausalLM"]}))
    assert load_config(str(path)).text.num_hidden_layers == 48
    assert load_config(str(tmp_path)).text.num_hidden_layers == 48
    assert MimoV2Config.from_pretrained(str(tmp_path)).text.num_hidden_layers == 48


def test_describe_names_both_families_and_the_moe_shape():
    text = describe(released_config())
    assert "GA 9" in text
    assert "SWA 39" in text and "qkv 14848" in text
    assert "256 experts" in text and "top-8" in text
    assert "dense FFN on layer(s) [0]" in text


def test_cli_prints_the_shape_table_and_checks_the_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({**RELEASED, "architectures": ["MiMoV2ForCausalLM"]}))
    result = subprocess.run(
        [sys.executable, "-m", "src.models.mimo_v2.config", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # The table is the deliverable, so the two widths have to show up in it.
    assert "13568" in out and "14848" in out
    assert "qkv_out" in out
    assert "[ok] the config is self-consistent" in out


def test_cli_exits_nonzero_on_a_config_the_reference_cannot_run(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({**RELEASED, "scoring_func": "softmax"}))
    result = subprocess.run(
        [sys.executable, "-m", "src.models.mimo_v2.config", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "scoring_func" in result.stdout


# ---------------------------------------------------------------------------
# The released files, when they are on this host
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no {HF_CONFIG}")
def test_released_config_matches_the_pinned_values():
    """The strong claim: what the file states is what the table above assumes."""
    config = MimoV2Config.from_pretrained(str(CHECKPOINT))
    text = config.text
    assert text.model_type == RELEASED["model_type"]
    assert text.num_hidden_layers == RELEASED["num_hidden_layers"]
    assert text.head_dim == RELEASED["head_dim"]
    assert text.v_head_dim == RELEASED["v_head_dim"]
    assert text.swa_num_key_value_heads == RELEASED["swa_num_key_value_heads"]
    assert text.resolved_projection_layout == RELEASED["attention_projection_layout"]
    assert text.add_swa_attention_sink_bias is True
    assert text.add_full_attention_sink_bias is False
    assert text.global_layer_indices == GA_LAYERS
    assert text.swa_layer_indices == SWA_LAYERS
    assert text.partial_rotary_factor == pytest.approx(RELEASED["partial_rotary_factor"])
    assert text.attention_value_scale == pytest.approx(RELEASED["attention_value_scale"])


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no {HF_CONFIG}")
def test_released_config_verifies_and_reports_its_notes():
    config = MimoV2Config.from_pretrained(str(CHECKPOINT))
    assert config.verify() == []
    assert config.dangling_ignored_layers() == ["model.decoder.self_attn.o_proj"]
    notes = config.runtime_notes()
    assert any("no requantization" in note for note in notes)

    # The two fused widths, read off the real file rather than a fixture.
    assert config.text.attention(0).qkv_out == 13568
    assert config.text.attention(1).qkv_out == 14848
    assert config.has_vision and config.has_audio


@pytest.mark.skipif(not HAS_CHECKPOINT, reason=f"no {HF_CONFIG}")
def test_released_quantization_block_is_the_mixed_scheme():
    config = MimoV2Config.from_pretrained(str(CHECKPOINT))
    quant = config.quant
    assert quant is not None
    assert quant.activation_scheme == "dynamic"
    assert quant.weight_block_size == (128, 128)
    assert quant.mxfp4_block_size == 32
    # Every backbone layer's o_proj is stored, and that count is the layer count.
    assert len(quant.ignored_o_proj_names) == config.text.num_hidden_layers


@pytest.mark.skipif(not DRAFT_CONFIG.is_file(), reason=f"no {DRAFT_CONFIG}")
def test_released_draft_config_parses_and_verifies():
    draft = MimoV2Config.load_draft(str(CHECKPOINT))
    assert draft is not None
    assert draft.verify() == []
    assert draft.num_hidden_layers == 5
    assert draft.is_causal is False
    assert draft.target_layer_ids == (0, 11, 23, 35, 47)
    assert draft.mask_token_id == 151675
    assert draft.context_width == 5 * draft.hidden_size
