"""Tests for `src/models/deepseek_v4_1/config.py`.

The released checkpoint describes one model in two layouts: `config.json`, where
the text hyper-parameters sit under `text_config` and the vision tower under
`vision_config`, and `inference/config.json`, a flat file in the reference
runtime's own key names. Neither is a superset of the other, three of the fields
are list-valued with the *direction* reversed (`kv_source_layer_ids` against
`kv_source_layers`), and almost every other field is renamed. The schema exists
to pay that once, so the tests are about the mapping being complete and being
faithful, not about any single value.

Three levels, deliberately:

* The alias tables are checked structurally against the dataclasses, which is
  what catches the failure mode they were written to prevent -- a field with no
  entry, or an entry naming a key neither file has.
* A small hand-written pair of configs asserts the renames and the round trip.
  Nothing here needs the checkpoint.
* The two files the release actually ships are read when they are present on this
  host, and that is where the strong claims live: they agree field for field, the
  nested one reproduces the flat one exactly, and the header audit runs off the
  nested one. Those tests skip elsewhere rather than weakening the claim -- a
  skip is a skip, not a pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from src.models.deepseek_v4_1.config import (
    _ALIASES,
    _SEQUENCE_FIELDS,
    V41TextConfig,
    V41VisionConfig,
    describe,
    from_dict,
    load_config,
    resolve_config,
)

REPO = Path(__file__).resolve().parent.parent

# The released checkpoint, if it is on this host. The header audit works on a
# header-only prefix tree as well as on complete shards, so what these tests need
# is the two config files, not the 475 GiB of weights.
CHECKPOINT = Path("/mnt/data3/DeepSeek-V4.1-Flash")
HF_CONFIG = CHECKPOINT / "config.json"
FLAT_CONFIG = CHECKPOINT / "inference" / "config.json"
HAS_CHECKPOINT_CONFIGS = HF_CONFIG.is_file() and FLAT_CONFIG.is_file()

# What the release states, pinned. These are facts about the checkpoint rather
# than about the schema, so they are read from the files rather than asserted
# here -- except for the handful below, which are pinned so that a change in the
# alias table or the sequence handling fails loudly instead of silently producing
# a config that still parses.
RELEASED = {
    ("text", "n_layers"): 40,
    ("text", "n_mtp_layers"): 3,
    ("text", "dim"): 5120,
    ("text", "n_heads"): 64,
    ("text", "head_dim"): 512,
    ("text", "q_lora_rank"): 1280,
    ("text", "n_routed_experts"): 384,
    ("text", "n_activated_experts"): 6,
    ("text", "kv_source_layers"): (2, 8, 14, 20),
    ("text", "index_source_layers"): (2, 8, 14, 20, 24, 28, 32, 36),
    ("text", "candidate_source_layer"): 20,
    ("text", "engram_layer_ids"): (1, 14),
    ("text", "engram_num_embeddings"): (384006168, 384016682),
    ("text", "engram_compressed_vocab_size"): 99092,
    ("text", "engram_pad_id"): 2,
    ("text", "dspark_target_layer_ids"): (37, 38, 39),
    ("text", "vocab_size"): 129280,
    ("text", "image_token_id"): 129264,
    ("vision", "n_layers"): 32,
    ("vision", "dim"): 1024,
}

# Fields only the Transformers file names. The flat reference file is silent on
# every one of these, which is why `Differs from` reports them: they are the
# honest statement of what that file does not say, and the test below pins the
# list so a new alias cannot quietly join it.
HF_ONLY = {
    "model_type",
    "architectures",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "param_dtype",
    "hidden_act",
    "max_position_embeddings",
    "num_key_value_heads",
    "tie_word_embeddings",
    "topk_method",
    "norm_topk_prob",
}

# A pair of small configs describing the same toy model, one in each layout. The
# values are arbitrary; what they are chosen to exercise is the mapping -- the
# nine renames, the three direction-reversed layer lists, the rope block the flat
# file hoists, and one field only the nested shape carries.
TOY_FLAT = {
    "vocab_size": 256,
    "image_token_id": 255,
    "dtype": "bf16",
    "expert_dtype": "fp4",
    "dim": 128,
    "moe_inter_dim": 64,
    "n_layers": 4,
    "n_mtp_layers": 1,
    "n_heads": 8,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 0.5,
    "q_lora_rank": 32,
    "head_dim": 64,
    "rope_head_dim": 16,
    "norm_eps": 1e-20,
    "o_groups": 2,
    "o_lora_rank": 16,
    "window_size": 8,
    "compress_ratios": [0, 2, 2, 2, 2],
    "kv_source_layers": [1],
    "index_source_layers": [1, 3],
    "compress_rope_theta": 160000,
    "original_seq_len": 4096,
    "rope_theta": 10000,
    "rope_factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "index_topk": 16,
    "candidate_source_layer": 1,
    "candidate_topk_blocks": 8,
    "candidate_block_size": 2,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [2],
    "engram_num_embeddings": [1000],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 100,
    "engram_n_heads": 2,
    "engram_head_dim": 8,
    "engram_pad_id": 2,
    "engram_compressed_vocab_size": 99092,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 250,
    "dspark_target_layer_ids": [3],
    "dspark_markov_rank": 16,
    "dspark_n_routed_experts": 8,
    "dspark_n_activated_experts": 2,
    "vision_n_layers": 2,
    "vision_dim": 32,
    "vision_n_heads": 4,
    "vision_inter_dim": 16,
    "vision_patch_size": 14,
    "vision_rope_theta": 10000,
    "vision_downsample_ratio": 3,
    "vision_max_n_token": 64,
    "vision_min_pixels": 295936,
    "vision_max_wh_ratio": None,
}


def _toy_nested(flat: dict) -> dict:
    """The same toy model in the Transformers layout, written out by hand.

    Hand-written rather than generated from the alias table on purpose: deriving
    it from the table would make the test agree with the table by construction,
    and the renames are exactly what is under test.
    """
    return {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": flat["engram_pad_id"],
        "image_token_id": flat["image_token_id"],
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "bf16", "expert_dtype": "fp4"},
        "text_config": {
            "vocab_size": flat["vocab_size"],
            "hidden_size": flat["dim"],
            "moe_intermediate_size": flat["moe_inter_dim"],
            "num_hidden_layers": flat["n_layers"],
            "num_nextn_predict_layers": flat["n_mtp_layers"],
            "num_attention_heads": flat["n_heads"],
            "num_key_value_heads": 1,
            "hidden_act": "silu",
            "max_position_embeddings": 1048576,
            "tie_word_embeddings": False,
            "n_routed_experts": flat["n_routed_experts"],
            "n_shared_experts": flat["n_shared_experts"],
            "num_experts_per_tok": flat["n_activated_experts"],
            "scoring_func": flat["score_func"],
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": flat["route_scale"],
            "swiglu_limit": flat["swiglu_limit"],
            "q_lora_rank": flat["q_lora_rank"],
            "head_dim": flat["head_dim"],
            "qk_rope_head_dim": flat["rope_head_dim"],
            "rms_norm_eps": flat["norm_eps"],
            "o_groups": flat["o_groups"],
            "o_lora_rank": flat["o_lora_rank"],
            "sliding_window": flat["window_size"],
            "compress_ratios": flat["compress_ratios"],
            "kv_source_layer_ids": flat["kv_source_layers"],
            "index_source_layer_ids": flat["index_source_layers"],
            "compress_rope_theta": flat["compress_rope_theta"],
            "rope_theta": flat["rope_theta"],
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": flat["rope_factor"],
                "beta_fast": flat["beta_fast"],
                "beta_slow": flat["beta_slow"],
                "original_max_position_embeddings": flat["original_seq_len"],
            },
            "index_n_heads": flat["index_n_heads"],
            "index_head_dim": flat["index_head_dim"],
            "index_topk": flat["index_topk"],
            "candidate_source_layer_id": flat["candidate_source_layer"],
            "candidate_topk_blocks": flat["candidate_topk_blocks"],
            "candidate_block_size": flat["candidate_block_size"],
            "hc_mult": flat["hc_mult"],
            "hc_sinkhorn_iters": flat["hc_sinkhorn_iters"],
            "hc_eps": flat["hc_eps"],
            "engram_layer_ids": flat["engram_layer_ids"],
            "engram_num_embeddings": flat["engram_num_embeddings"],
            "engram_max_ngram_size": flat["engram_max_ngram_size"],
            "engram_vocab_size": flat["engram_vocab_size"],
            "engram_n_heads": flat["engram_n_heads"],
            "engram_head_dim": flat["engram_head_dim"],
            "engram_pad_token_id": flat["engram_pad_id"],
            "engram_compressed_vocab_size": flat["engram_compressed_vocab_size"],
            "dspark_block_size": flat["dspark_block_size"],
            "dspark_noise_token_id": flat["dspark_noise_token_id"],
            "dspark_target_layer_ids": flat["dspark_target_layer_ids"],
            "dspark_markov_rank": flat["dspark_markov_rank"],
            "dspark_n_routed_experts": flat["dspark_n_routed_experts"],
            "dspark_num_experts_per_tok": flat["dspark_n_activated_experts"],
        },
        "vision_config": {
            "num_hidden_layers": flat["vision_n_layers"],
            "hidden_size": flat["vision_dim"],
            "num_attention_heads": flat["vision_n_heads"],
            "intermediate_size": flat["vision_inter_dim"],
            "patch_size": flat["vision_patch_size"],
            "rope_theta": flat["vision_rope_theta"],
            "downsample_ratio": flat["vision_downsample_ratio"],
            "max_image_tokens": flat["vision_max_n_token"],
            "min_pixels": flat["vision_min_pixels"],
            "max_wh_ratio": flat["vision_max_wh_ratio"],
        },
    }


# The dataclasses, so the structural tests can walk them.
HALVES = (("text", V41TextConfig), ("vision", V41VisionConfig))


def test_every_dataclass_field_has_an_alias_row():
    """A field with no row is a KeyError the first time a config is read.

    Which is how the vision fields were first written: the table had them under
    `vision_dim` while the dataclass called the same field `dim`, and the loader
    raised before any test noticed.
    """
    for half, cls in HALVES:
        table = _ALIASES[half]
        declared = {field.name for field in fields(cls)}
        assert declared == set(table), (
            f"{half}: fields without a row {sorted(declared - set(table))}, "
            f"rows without a field {sorted(set(table) - declared)}"
        )


def test_every_alias_row_names_a_key_one_of_the_files_has():
    """Each row is a pair of paths, and neither may name a key nobody ships.

    The nested column is dotted, so its first step has to be a real top-level key
    of `config.json`; the flat column is one level and must stay that way, since
    the whole point of that file is that it is flat.
    """
    top_level = {
        "model_type",
        "architectures",
        "dtype",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "image_token_id",
        "text_config",
        "vision_config",
        "quantization_config",
    }
    for half, table in _ALIASES.items():
        for name, (nested, flat) in table.items():
            assert nested or flat, f"{half}.{name} has no source in either layout"
            if nested is not None:
                assert nested.split(".")[0] in top_level, (
                    f"{half}.{name} reads {nested}, which is not reachable in config.json"
                )
            if flat is not None:
                assert "." not in flat, f"{half}.{name} reads {flat}, which is nested in a flat file"


def test_the_two_layouts_read_the_toy_model_the_same_way():
    """The renames are applied in both directions and nothing is dropped.

    The nested toy file states the twelve fields only the Transformers layout has,
    so `differs_from` reports them; everything the two share has to come back
    identical, which is what comparing the two flat projections checks.
    """
    flat = from_dict(TOY_FLAT)
    nested = from_dict(_toy_nested(TOY_FLAT))

    assert flat.nested is False and nested.nested is True
    assert flat.as_reference_dict() == nested.as_reference_dict() == TOY_FLAT


def test_nothing_from_the_flat_file_is_lost_by_the_nested_one():
    """Every key the flat toy file carries survives the trip through the schema."""
    reference = from_dict(_toy_nested(TOY_FLAT)).as_reference_dict()
    for key, value in TOY_FLAT.items():
        assert key in reference, f"{key} has no counterpart in the Transformers layout"
        assert reference[key] == value, f"{key}: {reference[key]!r} != {value!r}"


def test_the_flat_layout_round_trips_through_the_schema():
    """`as_reference_dict` is the inverse of reading the flat file."""
    assert from_dict(TOY_FLAT).as_reference_dict() == TOY_FLAT


def test_the_reference_key_set_is_the_flat_layouts_key_set():
    """The schema emits exactly the flat file's keys, no more and no fewer.

    This is the property the header audit relies on: it validates presence of
    the keys it needs, so a schema that dropped one or invented one would change
    what the audit reports without changing what the checkpoint says.
    """
    table_keys = {row[1] for table in _ALIASES.values() for row in table.values() if row[1]}
    assert set(from_dict(TOY_FLAT).as_reference_dict()) == table_keys
    assert set(TOY_FLAT) == table_keys


def test_sequence_fields_are_lists_and_become_tuples():
    """Lists are pinned so a config cannot be mutated through the frozen wrapper."""
    config = from_dict(TOY_FLAT)
    for name in _SEQUENCE_FIELDS:
        value = getattr(config.text, name, None)
        assert isinstance(value, tuple), f"{name} is {type(value).__name__}"
    assert config.text.kv_source_layers == (1,)
    assert config.as_reference_dict()["kv_source_layers"] == [1]


def test_verify_accepts_the_toy_model_and_reports_what_it_breaks():
    """The toy model is internally consistent; each break is then reported."""
    assert from_dict(TOY_FLAT).verify() == []
    assert from_dict(_toy_nested(TOY_FLAT)).verify() == []

    # A layer that compresses its KV before the only source that writes the cache.
    broken = dict(TOY_FLAT, kv_source_layers=[3])
    problems = from_dict(broken).verify()
    assert any("no kv source at or before it" in problem for problem in problems), problems

    # A candidate source that is not an index source writes nothing to read.
    broken = dict(TOY_FLAT, candidate_source_layer=2, index_source_layers=[1, 3])
    problems = from_dict(broken).verify()
    assert any("is not an index source" in problem for problem in problems), problems

    # A rope tail wider than the head it lives in.
    broken = dict(TOY_FLAT, rope_head_dim=128)
    problems = from_dict(broken).verify()
    assert any("rope_head_dim 128 exceeds head_dim 64" in problem for problem in problems), problems


def test_a_config_without_engram_layers_is_not_an_error():
    """Engram is optional; the schema says so by leaving the list empty."""
    config = from_dict(dict(TOY_FLAT, engram_layer_ids=[], engram_num_embeddings=[]))
    assert config.verify() == []
    assert config.engram_block()["engram_layer_ids"] == []


def test_two_files_that_disagree_on_one_field_are_reported_as_such():
    """`differs_from` reports disagreements and silences alike, not just one."""
    other = dict(TOY_FLAT, window_size=16)
    differences = from_dict(TOY_FLAT).differs_from(from_dict(other))
    assert differences == [f"text.window_size: 8 != 16"]


def test_an_unknown_key_is_ignored_rather_than_fatal():
    """A repacked config carries keys this schema has no use for; that is fine."""
    config = from_dict(dict(TOY_FLAT, some_repacker_field={"a": 1}))
    assert config.text.dim == 128


def test_the_model_card_wrapper_is_accepted():
    """The card documents `{"model": {...}}`, which the released files do not use."""
    assert from_dict({"model": TOY_FLAT}).as_reference_dict() == TOY_FLAT


def test_describe_names_the_optional_heads_it_finds():
    """`describe` is what the CLI and the logs print, so it has to not lie."""
    line = describe(from_dict(TOY_FLAT))
    assert "4 layers + 1 draft" in line
    assert "Engram on layers [2]" in line
    assert "vision 2 layers" in line
    assert "DSpark block 5" in line
    assert "Engram" not in describe(from_dict(dict(TOY_FLAT, engram_layer_ids=[])))


# ---------------------------------------------------------------------------
# The files the release actually ships. Present on this host; skipped elsewhere.
# ---------------------------------------------------------------------------


requires_checkpoint_configs = pytest.mark.skipif(
    not HAS_CHECKPOINT_CONFIGS,
    reason=f"the released configs are not at {CHECKPOINT}",
)


@requires_checkpoint_configs
def test_the_released_files_are_in_the_layouts_they_are_named_for():
    assert load_config(str(HF_CONFIG)).nested is True
    assert load_config(str(FLAT_CONFIG)).nested is False


@requires_checkpoint_configs
def test_the_released_files_agree_on_every_field_they_both_carry():
    """The strongest claim the schema exists to make, and it holds exactly.

    Twelve fields come back as differences and every one of them is the flat file
    being silent -- there is not a single disagreement on a value, which is the
    test below this one, since a field both files carry cannot differ without
    breaking the round trip. `HF_ONLY` pins which twelve, so a field newly added
    to the alias table cannot join them without this test being updated on
    purpose.
    """
    nested = load_config(str(HF_CONFIG))
    flat = load_config(str(FLAT_CONFIG))

    differences = nested.differs_from(flat)
    assert {difference.split(":")[0] for difference in differences} == {
        f"text.{name}" for name in HF_ONLY
    }


@requires_checkpoint_configs
def test_the_nested_file_reproduces_the_flat_one_exactly():
    """Read the Transformers layout, write it back, and get the other file.

    Byte-for-byte at the value level, which is what makes the schema a mapping
    between the two rather than a third opinion about the model.
    """
    with open(FLAT_CONFIG, encoding="utf-8") as handle:
        shipped = json.load(handle)
    assert load_config(str(HF_CONFIG)).as_reference_dict() == shipped
    assert load_config(str(FLAT_CONFIG)).as_reference_dict() == shipped


@requires_checkpoint_configs
def test_the_released_configs_are_self_consistent():
    for path in (HF_CONFIG, FLAT_CONFIG):
        config = load_config(str(path))
        assert config.verify() == [], f"{path}: {config.verify()}"


@requires_checkpoint_configs
def test_the_released_values_are_what_they_were_measured_to_be():
    """Pinned so a rename that happens to still parse cannot pass unnoticed."""
    config = load_config(str(HF_CONFIG))
    for (half, name), expected in RELEASED.items():
        actual = getattr(getattr(config, half), name)
        assert actual == expected, f"{half}.{name}: {actual!r} != {expected!r}"


@requires_checkpoint_configs
def test_the_engram_derivation_reads_the_nested_config():
    """The consumer that was broken before the schema existed.

    `src/encoding/engram.py` needs the six engram keys and the pad id under the
    flat names. It reads them through `engram_block`, and the block has to be the
    same object whichever file it came from -- that is the whole point, since the
    row counts are 189 GiB of table.
    """
    from src.encoding.engram import EngramLayout

    nested = load_config(str(HF_CONFIG)).engram_block()
    flat = load_config(str(FLAT_CONFIG)).engram_block()
    assert nested == flat

    layout = EngramLayout.from_config(nested)
    assert layout is not None
    assert layout.layer_ids == RELEASED[("text", "engram_layer_ids")]
    assert layout.num_embeddings == RELEASED[("text", "engram_num_embeddings")]


@requires_checkpoint_configs
def test_resolve_config_prefers_the_file_the_release_ships():
    assert resolve_config(str(CHECKPOINT)) == str(HF_CONFIG)
    assert resolve_config(str(CHECKPOINT), explicit=str(FLAT_CONFIG)) == str(FLAT_CONFIG)
    with pytest.raises(FileNotFoundError):
        resolve_config("/nonexistent")


@requires_checkpoint_configs
def test_the_header_audit_reads_the_nested_config(tmp_path):
    """The audit used to fail on the released file with `missing [all 35 keys]`.

    It is run as a subprocess because that is how it is documented, and because
    the point is that it works from a command line rather than from an import.
    The checkpoint itself is not needed: the audit reads headers, and the header
    prefixes the repository already knows how to build are enough.
    """
    header_tree = Path("/tmp/dsv41")
    if not header_tree.is_dir():
        pytest.skip("no header-prefix tree to audit")

    contract = tmp_path / "contract.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "audit_dsv41_headers.py"),
            "--checkpoint-dir",
            str(header_tree),
            "--header-prefix",
            "--config",
            str(HF_CONFIG),
            "--json",
            str(contract),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing [" not in result.stdout

    with open(contract, encoding="utf-8") as handle:
        report = json.load(handle)
    assert [check for check in report["checks"] if not check["passed"]] == []
