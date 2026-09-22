"""The resident expert bank: the layout it derives, the bytes it moves, and who may attach.

The bank is the half of the heterogeneous path that has nothing to do with arithmetic:
it says where every routed layer's experts live in host memory and puts them there. So
these tests are about bytes and identity, not about numbers -- each expert's six tensors
must come back out of the segment exactly as the checkpoint stores them, and an expert
must never be answered with another expert's bytes.

That second property is the one worth a test of its own, because the release makes it
easy to get wrong: a shard stores its four experts in *name* order, so `ep2` holds
`10, 11, 8, 9` and expert 8 is not at `8 * expert_bytes`. Nothing about the shape or the
size of the result would say so. The miniature mirrors it with four experts in a shard
precisely so the test can break on it.

The first half runs on a miniature written to `tmp_path`; the second half runs on the
release when it is on the host, because a layout derived from a synthetic checkpoint is a
statement about this code and a layout derived from the release is a statement about the
checkpoint.
"""

from __future__ import annotations

import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from src.models.mimo_v2 import bank as bank_module  # noqa: E402
from src.models.mimo_v2.config import MimoV2Config  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

# The miniature's backbone tensors are the loader test's: the bank cares only that the
# checkpoint is one this loader will open, and a second copy of that list would be a second
# thing to keep in step with the config.
from tests.test_models_mimo_v2_loader import tiny_layer_tensors  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release = pytest.mark.skipif(
    not HAS_RELEASE, reason=f"MiMo-V2.6 checkpoint not present at {RELEASE}"
)

SAFETENSORS_TO_TORCH = {
    "U8": torch.uint8,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
}
TORCH_TO_SAFETENSORS = {value: key for key, value in SAFETENSORS_TO_TORCH.items()}

#: Two routed layers over three shards of four experts: 12, so that the highest shard
#: holds `8..11` and the release's name-order trap is reproduced exactly -- `10` sorts
#: before `8` as a string, so that shard stores `10, 11, 8, 9`.
MINI = {
    "model_type": "mimo_v2",
    "architectures": ["MiMoV2ForCausalLM"],
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "v_head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 16,
    "swa_v_head_dim": 8,
    "rope_theta": 10_000_000.0,
    "swa_rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_value_scale": 0.707,
    "attention_projection_layout": "fused_qkv",
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "sliding_window": 4,
    "hybrid_layer_pattern": [0, 1, 1],
    "moe_layer_freq": [0, 1, 1],
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "n_routed_experts": 12,
    "num_experts_per_tok": 3,
    "n_shared_experts": None,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 1024,
    "layernorm_epsilon": 1e-6,
    "attention_chunk_size": 4,
    "eos_token_id": 127,
    "pad_token_id": 126,
}
SHARDS = 3
EXPERTS_PER_SHARD = 4


def write_shard(path: str, items: list[tuple[str, torch.Tensor]]) -> None:
    """Write a shard with the tensors in exactly this order, as the bank's source does.

    `safetensors.torch.save_file` orders by name, which is the release's own behaviour --
    and is the trap these tests are about -- but a shard whose order is chosen is what
    lets a test assert the bank follows the file rather than its own idea of the order.
    """
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, tensor in items:
        blob = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": TORCH_TO_SAFETENSORS[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(blob)],
        }
        payload += blob
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def expert_tensors(layer: int, expert: int, config) -> dict[str, torch.Tensor]:
    """One expert's six tensors, each filled with a value that identifies it.

    The values are the test's own oracle: they are what "the bank holds this expert" can be
    checked against byte for byte, and an expert answered with another's bytes fails on the
    first comparison rather than on a logit much later.
    """
    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    tag = layer * 1000 + expert
    out: dict[str, torch.Tensor] = {}
    for proj, rows, cols in (("down_proj", hidden, inter), ("gate_proj", inter, hidden), ("up_proj", inter, hidden)):
        out[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"] = torch.full(
            (rows, cols // 2), tag % 256, dtype=torch.uint8
        )
        out[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight_scale"] = torch.full(
            (rows, cols // 32), (tag // 256) % 256, dtype=torch.uint8
        )
    return out


def write_mini_checkpoint(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w") as handle:
        json.dump(MINI, handle)
    config = MimoV2Config.from_pretrained(root).text

    for index in range(SHARDS):
        tensors: dict[str, torch.Tensor] = {}
        if index == 0:
            seed = torch.Generator().manual_seed(5)
            tensors["model.embed_tokens.weight"] = torch.randn(
                config.vocab_size, config.hidden_size, generator=seed
            ).to(torch.bfloat16)
            tensors["model.norm.weight"] = torch.ones(config.hidden_size)
            tensors["lm_head.weight"] = torch.randn(
                config.vocab_size, config.hidden_size, generator=seed
            ).to(torch.bfloat16)
            for layer in range(config.num_hidden_layers):
                tensors.update(tiny_layer_tensors(layer, config))
        lo = index * EXPERTS_PER_SHARD
        for layer in config.moe_layer_indices:
            for expert in range(lo, lo + EXPERTS_PER_SHARD):
                tensors.update(expert_tensors(layer, expert, config))
        # In ascending expert order here. `save_file` reorders the keys as strings, which
        # is what makes the highest shard store `10, 11, 8, 9` -- the release's own order,
        # arrived at the same way.
        save_file(tensors, os.path.join(root, f"model_pp0_ep{index}_shard0.safetensors"))
    return root


@pytest.fixture(scope="module")
def mini(tmp_path_factory) -> MimoV2Checkpoint:
    return MimoV2Checkpoint(write_mini_checkpoint(str(tmp_path_factory.mktemp("mimo_bank"))))


@pytest.fixture
def bank(mini, tmp_path):
    """A filled bank on the miniature, in a directory of its own, removed afterwards."""
    opened = bank_module.open_expert_bank(mini, root_dir=str(tmp_path / "bank"))
    try:
        yield opened
    finally:
        opened.close(unlink=True)


# ---------------------------------------------------------------------------
# The layout: where each expert's bytes go
# ---------------------------------------------------------------------------


def test_the_layout_covers_every_routed_layer_and_nothing_else(mini):
    layers, size = bank_module._layout(mini)
    assert sorted(layers) == sorted(mini.layout.moe_layers)
    assert size > bank_module._HEADER_BYTES
    for layer in layers.values():
        assert layer.nbytes >= layer.n_experts * layer.expert_bytes
        assert layer.base >= bank_module._HEADER_BYTES
    # Regions are disjoint and ordered, so a byte of one layer is never a byte of another.
    ordered = sorted(layers.values(), key=lambda item: item.base)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.base + earlier.nbytes <= later.base


def test_every_expert_of_a_layer_has_a_position_and_they_are_a_permutation(mini):
    layers, _ = bank_module._layout(mini)
    for layer in layers.values():
        assert sorted(layer.positions) == list(range(layer.n_experts))
        assert sorted(layer.positions.values()) == list(range(layer.n_experts))


def test_a_shard_stores_its_experts_in_name_order_and_the_bank_follows_the_file(mini):
    """`ep2` holds experts 10, 11, 8, 9; the bank must not assume `expert * expert_bytes`.

    The order is read out of the shard itself here rather than hardcoded, so the test says
    "the bank agrees with the file" and not "the bank agrees with this test's idea of the
    release".
    """
    stored = mini.expert_order(2, 2)
    assert sorted(stored) == list(mini.experts_of_shard(2))
    assert stored == (10, 11, 8, 9), "the miniature no longer reproduces the release's name order"

    layers, _ = bank_module._layout(mini)
    layer = layers[2]
    for index, expert in enumerate(stored):
        assert layer.position(expert) == 2 * EXPERTS_PER_SHARD + index
        assert layer.offset(expert, "down_proj", "weight") == (
            layer.base + (2 * EXPERTS_PER_SHARD + index) * layer.expert_bytes
        )


def test_an_expert_that_is_not_in_the_bank_raises_rather_than_answering(bank):
    with pytest.raises(KeyError):
        bank.tensor(99, 0, "down_proj", "weight")
    with pytest.raises(KeyError):
        bank.tensor(1, 0, "sideways_proj", "weight")
    assert not bank.has_expert(99, 0, "down_proj", "weight")
    assert bank.has_expert(1, 0, "down_proj", "weight")


# ---------------------------------------------------------------------------
# The bytes: what comes back out
# ---------------------------------------------------------------------------


def test_every_expert_comes_back_byte_for_byte(mini, bank):
    """The whole point of the bank: the segment's bytes are the checkpoint's bytes."""
    for layer in mini.layout.moe_layers:
        for expert in range(mini.layout.n_experts):
            arrays = mini.expert_arrays(layer, expert)
            views = bank.expert_views(layer, expert)
            assert set(views) == set(arrays)
            for key, expected in arrays.items():
                got = views[key]
                assert got.shape == expected.shape, (layer, expert, key)
                assert torch.equal(got, expected), (layer, expert, key)


def test_the_bank_is_a_view_of_one_segment_and_not_a_copy(mini, bank):
    """Two reads of the same slot are the same memory, which is what the DMA reads."""
    first = bank.tensor(1, 3, "up_proj", "weight")
    second = bank.tensor(1, 3, "up_proj", "weight")
    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() != bank.tensor(1, 4, "up_proj", "weight").data_ptr()


def test_the_payload_is_every_expert_and_not_one_byte_more(mini, bank):
    expected = sum(
        mini.layout.expert_bytes for _ in mini.layout.moe_layers for _ in range(mini.layout.n_experts)
    )
    assert bank.payload_bytes == expected
    assert bank.resident_bytes >= expected
    assert bank_module.resident_bytes(mini) == bank.resident_bytes


def test_a_segment_of_the_wrong_size_is_refused_rather_than_read_past_its_end(mini, bank, tmp_path):
    """A rank that attaches to somebody else's segment must not read it as its own."""
    with pytest.raises(RuntimeError, match="belongs to a different checkpoint"):
        bank_module.MimoV2ExpertBank(
            checkpoint=mini,
            shm_name=bank.shm_name,
            root_dir=str(tmp_path / "bank"),
            size=bank.size + 8,
            n_routed_experts=bank.n_routed_experts,
            layers=bank.layers,
            create=False,
        )


def test_a_segment_written_by_another_layout_version_is_refused(mini, bank, tmp_path):
    struct.pack_into("<Q", bank._buffer, 8, bank_module._HEADER_VERSION + 1)
    try:
        with pytest.raises(RuntimeError, match="layout version"):
            bank_module.MimoV2ExpertBank(
                checkpoint=mini,
                shm_name=bank.shm_name,
                root_dir=str(tmp_path / "bank"),
                size=bank.size,
                n_routed_experts=bank.n_routed_experts,
                layers=bank.layers,
                create=False,
            )
    finally:
        struct.pack_into("<Q", bank._buffer, 8, bank_module._HEADER_VERSION)


# ---------------------------------------------------------------------------
# Opening, filling and attaching
# ---------------------------------------------------------------------------


def test_a_bank_that_is_already_ready_is_attached_and_not_refilled(mini, tmp_path):
    root_dir = str(tmp_path / "bank")
    first = bank_module.open_expert_bank(mini, root_dir=root_dir)
    try:
        assert first.create
        assert os.path.exists(first.ready_path)
        marker = os.path.getmtime(first.ready_path)
        second = bank_module.open_expert_bank(mini, root_dir=root_dir)
        try:
            assert not second.create
            assert os.path.getmtime(second.ready_path) == marker
            assert torch.equal(
                first.tensor(1, 5, "gate_proj", "weight"),
                second.tensor(1, 5, "gate_proj", "weight"),
            )
        finally:
            second.close()
    finally:
        first.close(unlink=True)


def test_a_fill_reports_the_bytes_it_moved(mini, tmp_path):
    bank = bank_module.open_expert_bank(mini, root_dir=str(tmp_path / "bank"))
    try:
        seen: list[str] = []
        moved = bank.fill(progress=seen.append)
        assert moved == bank.payload_bytes
        assert seen
        assert all("resident checkpoint" in line for line in seen)
    finally:
        bank.close(unlink=True)


def test_forcing_a_fill_rewrites_a_ready_bank(mini, tmp_path):
    root_dir = str(tmp_path / "bank")
    first = bank_module.open_expert_bank(mini, root_dir=root_dir)
    first.close()
    rebuilt = bank_module.open_expert_bank(mini, root_dir=root_dir, force_fill=True)
    try:
        assert rebuilt.create
        assert torch.equal(
            rebuilt.tensor(1, 9, "down_proj", "weight"),
            mini.expert_arrays(1, 9)[("down_proj", "weight")],
        )
    finally:
        rebuilt.close(unlink=True)


def test_closing_without_unlinking_leaves_the_segment_for_the_next_run(mini, tmp_path):
    root_dir = str(tmp_path / "bank")
    bank = bank_module.open_expert_bank(mini, root_dir=root_dir)
    name = bank.shm_name
    bank.close()
    assert os.path.exists(os.path.join("/dev/shm", name))
    again = bank_module.open_expert_bank(mini, root_dir=root_dir)
    try:
        assert not again.create
    finally:
        again.close(unlink=True)


def test_the_environment_gates_default_the_way_the_module_documents(mini, monkeypatch):
    monkeypatch.delenv(bank_module.ENABLE_ENV, raising=False)
    monkeypatch.delenv(bank_module.PIN_ENV, raising=False)
    assert not bank_module.enabled()
    assert bank_module.pin_enabled()
    monkeypatch.setenv(bank_module.ENABLE_ENV, "1")
    monkeypatch.setenv(bank_module.PIN_ENV, "0")
    assert bank_module.enabled()
    assert not bank_module.pin_enabled()


def test_pinning_on_demand_honours_the_gate_and_reports_what_the_driver_said(mini, tmp_path, monkeypatch):
    """The device path's one call: pin the source, or be told not to.

    `None` and a failed registration are different answers, which is why this returns the
    `PinResult` rather than a bool -- a driver that refuses the registration leaves a bank
    that is correct and slower, and a caller that could not tell the two apart would either
    abort a good run or ignore a bad one.
    """
    monkeypatch.delenv(bank_module.PIN_ENV, raising=False)
    root_dir = str(tmp_path / "pin")
    bank = bank_module.open_expert_bank(mini, root_dir=root_dir)
    try:
        result = bank.pin_if_enabled()
        if result is None:  # pragma: no cover - the gate defaults on
            pytest.fail("the gate defaults on and `pin_if_enabled` returned None")
        assert result.bytes == bank.resident_bytes
        assert bank.pin_result is result
        # Idempotent: the second call is the first result, not a second registration.
        assert bank.pin_if_enabled() is result
    finally:
        bank.close(unlink=True)

    monkeypatch.setenv(bank_module.PIN_ENV, "0")
    off = bank_module.open_expert_bank(mini, root_dir=root_dir)
    try:
        assert off.pin_if_enabled() is None
        assert off.pin_result is None
    finally:
        off.close(unlink=True)


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


@needs_release
def test_the_released_bank_is_149_81_gib_and_47_layers(release):
    layers, size = bank_module._layout(release)
    assert sorted(layers) == list(range(1, 48))
    payload = sum(layer.nbytes for layer in layers.values())
    assert payload / 2**30 == pytest.approx(149.81, abs=0.01)
    assert size >= payload
    assert layers[1].expert_bytes == 13_369_344
    assert all(layer.n_experts == 256 for layer in layers.values())
    assert all(len(layer.fills) == 64 for layer in layers.values())


@needs_release
def test_the_released_shard_two_stores_its_experts_as_ten_eleven_eight_nine(release):
    """The trap the bank exists downstream of, on the checkpoint that has it."""
    assert release.expert_order(1, 2) == (10, 11, 8, 9)
    assert release.experts_of_shard(2) == (8, 9, 10, 11)
    layers, _ = bank_module._layout(release)
    for index, expert in enumerate(release.expert_order(1, 2)):
        assert layers[1].position(expert) == 2 * release.layout.experts_per_shard + index


@needs_release
def test_the_released_expert_region_is_exactly_the_shards_share(release):
    """One `pread` per (shard, layer), and it is the four experts and not a byte more."""
    per_shard = release.layout.experts_per_shard
    for layer in (1, 24, 47):
        for shard in (0, 2, 63):
            file_name, begin, end = release.expert_region(layer, shard)
            assert file_name == release.layout.files[shard]
            assert end - begin == per_shard * release.layout.expert_bytes
            assert len(release.expert_order(layer, shard)) == per_shard
            assert sorted(release.expert_order(layer, shard)) == list(
                release.experts_of_shard(shard)
            )
