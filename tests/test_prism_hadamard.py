"""The ``prism.hadamard.*`` metadata block, against the released checkpoint.

The fixture is the block as it is stored in
``prism-ml/Ternary-Bonsai-2-27B-gguf``'s ``Ternary-Bonsai-2-27B-PTQ1_0.gguf``, so
these tests run without the 5.5 GiB checkpoint.  A backend test that does read the
checkpoint is included and skips when it is not on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.loader.gguf.prism_hadamard import (
    GDN_GROUPED_WIDTH,
    HadamardSpec,
    HadamardSpecError,
    has_hadamard_block,
    parse_hadamard_spec,
)
from src.loader.gguf.reader import GGUFArraySummary, GGUFReader

FIXTURE_PATH = Path(__file__).parent / "data" / "ternary_bonsai_hadamard.json"
CHECKPOINT_ENV = "POCKETLLM_BONSAI_GGUF"
CHECKPOINT_DEFAULT = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"

#: Tensors the block folds, by role read off the checkpoint's tensor table.  The
#: counts, not the order: the converter's list order is its own and carries no meaning.
FOLDED_ROLE_COUNTS = {
    "attn_qkv.weight": 48,
    "attn_gate.weight": 48,
    "ssm_out.weight": 48,
    "ffn_down.weight": 64,
    "ffn_gate.weight": 64,
    "ffn_up.weight": 64,
    "attn_k.weight": 16,
    "attn_v.weight": 16,
    "attn_q.weight": 16,
    "attn_output.weight": 16,
}


def _load_fixture() -> dict:
    with FIXTURE_PATH.open() as handle:
        return json.load(handle)


def _unpack_signs(blob: str, count: int) -> list[int]:
    """Expand the fixture's bit-packed sign vector: one bit per entry, +1 when set."""
    raw = bytes.fromhex(blob)
    assert len(raw) * 8 >= count
    return [1 if raw[i >> 3] >> (i & 7) & 1 else -1 for i in range(count)]


def _metadata(raw: dict | None = None, **overrides) -> dict:
    fixture = _load_fixture() if raw is None else dict(raw)
    metadata = {
        "prism.hadamard.version": fixture["version"],
        "prism.hadamard.transform": fixture["transform"],
        "prism.hadamard.axis": fixture["axis"],
        "prism.hadamard.block_size": fixture["block_size"],
        "prism.hadamard.sign_mode": fixture["sign_mode"],
        "prism.hadamard.gdn_v_grouped": fixture["gdn_v_grouped"],
        "prism.hadamard.sign_widths": list(fixture["sign_widths"]),
        "prism.hadamard.sign_values": _unpack_signs(
            fixture["sign_values_bits"], sum(fixture["sign_widths"])
        ),
        "prism.hadamard.weight_names": list(fixture["weight_names"]),
        "prism.hadamard.inverse_weight_names": list(fixture["inverse_weight_names"]),
    }
    for key, value in overrides.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


@pytest.fixture(scope="module")
def spec() -> HadamardSpec:
    parsed = parse_hadamard_spec(_metadata())
    assert parsed is not None
    return parsed


def test_the_released_block_parses(spec: HadamardSpec) -> None:
    assert spec.version == 1
    assert spec.transform == "normalized-sylvester-walsh-hadamard"
    assert spec.axis == "input-last-dimension"
    assert spec.block_size == 1024
    assert spec.sign_mode == "explicit"
    assert spec.gdn_v_grouped is True


def test_the_sign_widths_are_the_three_folded_widths(spec: HadamardSpec) -> None:
    assert spec.sign_widths == (5120, 6144, 17408)
    # The activation is folded on the *last* dimension, and the widths are exactly
    # hidden, the gated-DeltaNet value width and the MLP width.
    assert sum(spec.sign_widths) == len(spec.sign_values) == 28672


def test_each_width_slices_its_own_vector_out_of_the_concatenation(spec: HadamardSpec) -> None:
    fixture = _load_fixture()
    packed = _unpack_signs(fixture["sign_values_bits"], sum(fixture["sign_widths"]))
    offset = 0
    for width in spec.sign_widths:
        signs = spec.signs_for_width(width)
        assert len(signs) == width
        assert signs == tuple(packed[offset : offset + width])
        assert set(signs) <= {1, -1}
        offset += width


def test_an_undeclared_width_has_no_sign_vector(spec: HadamardSpec) -> None:
    with pytest.raises(HadamardSpecError, match="no sign vector declared for width 4096"):
        spec.signs_for_width(4096)


def test_the_transposed_reading_would_be_a_different_vector(spec: HadamardSpec) -> None:
    """A sign vector consumed in the wrong order is a plausible, silent bug.

    The three declared widths happen to have different ``+1`` counts, so reading the
    concatenation off by one width does not reproduce the right slice.  Pinning the
    counts is what makes an offset bug in ``signs_for_width`` visible.
    """
    counts = {width: sum(1 for sign in spec.signs_for_width(width) if sign > 0) for width in spec.sign_widths}
    assert counts == {5120: 2481, 6144: 3032, 17408: 8655}
    assert len(set(counts.values())) == 3


def test_the_folded_list_has_one_entry_per_folded_tensor(spec: HadamardSpec) -> None:
    assert len(spec.weight_names) == 401
    assert len(set(spec.weight_names)) == 401
    counts = {role: 0 for role in FOLDED_ROLE_COUNTS}
    for name in spec.weight_names:
        if name == "output.weight":
            continue
        assert name.startswith("blk.")
        role = name.split(".", 2)[2]
        counts[role] = counts.get(role, 0) + 1
    counts = {role: count for role, count in counts.items() if count}
    assert counts == FOLDED_ROLE_COUNTS
    assert "output.weight" in spec.weight_names


def test_the_list_order_is_the_converter_s_not_the_file_s(spec: HadamardSpec) -> None:
    """The names are a set.  Reading order into them is the mistake this guards.

    The converter's list runs ``blk.0, blk.1, blk.10, ... `` -- a string sort of the
    layer index -- while the GGUF tensor table runs its own order.  Only membership
    is meaningful, and :meth:`is_declared` is the accessor that says so.
    """
    import itertools

    layers = [name.split(".")[1] for name in spec.weight_names if name.startswith("blk.")]
    grouped = [key for key, _ in itertools.groupby(layers)]
    assert grouped[:4] == ["0", "1", "10", "11"]
    assert grouped[-1] == "9"
    numeric = sorted(grouped, key=int)
    assert grouped != numeric


def test_the_inverse_is_token_embd_and_only_it(spec: HadamardSpec) -> None:
    assert spec.inverse_weight_names == ("token_embd.weight",)
    assert spec.takes_inverse("token_embd.weight") is True
    # output.weight reads the same table the other way and is a matmul, so it takes
    # the forward transform like everything else.
    assert spec.takes_inverse("output.weight") is False
    assert spec.is_declared("output.weight") is True
    assert spec.is_declared("blk.0.ffn_gate.weight") is True
    assert spec.is_declared("blk.0.ssm_conv1d.weight") is False
    assert len(spec.folded_names) == 402


def test_an_absent_block_is_not_an_error() -> None:
    assert has_hadamard_block({"general.architecture": "qwen35"}) is False
    assert parse_hadamard_spec({"general.architecture": "qwen35"}) is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prism.hadamard.version": 2}, "only version 1"),
        ({"prism.hadamard.transform": "sylvester-walsh-hadamard"}, "expected 'normalized"),
        ({"prism.hadamard.axis": "output-last-dimension"}, "expected 'input-last-dimension'"),
        ({"prism.hadamard.sign_mode": "seeded"}, "expected 'explicit'"),
        ({"prism.hadamard.block_size": 1000}, "not a power of two"),
        ({"prism.hadamard.block_size": 0}, "not a power of two"),
        ({"prism.hadamard.version": None}, "missing metadata key"),
        ({"prism.hadamard.sign_widths": [5120, 6144, 6144]}, "repeats a width"),
    ],
)
def test_a_block_that_is_not_the_one_we_read_raises(overrides: dict, message: str) -> None:
    with pytest.raises(HadamardSpecError, match=message):
        parse_hadamard_spec(_metadata(**overrides))


def test_a_sign_vector_that_does_not_match_its_widths_raises() -> None:
    metadata = _metadata()
    metadata["prism.hadamard.sign_values"] = metadata["prism.hadamard.sign_values"][:-1]
    with pytest.raises(HadamardSpecError, match="sums to 28672"):
        parse_hadamard_spec(metadata)


def test_a_sign_that_is_not_plus_or_minus_one_raises() -> None:
    metadata = _metadata()
    values = list(metadata["prism.hadamard.sign_values"])
    values[17] = 2
    metadata["prism.hadamard.sign_values"] = values
    with pytest.raises(HadamardSpecError, match="values other than"):
        parse_hadamard_spec(metadata)


def test_a_tensor_cannot_take_both_transforms() -> None:
    with pytest.raises(HadamardSpecError, match="both the forward and the inverse"):
        parse_hadamard_spec(_metadata(**{"prism.hadamard.inverse_weight_names": ["output.weight"]}))


def test_the_grouped_gdn_permute_needs_its_width_declared() -> None:
    fixture = _load_fixture()
    signs = _unpack_signs(fixture["sign_values_bits"], sum(fixture["sign_widths"]))
    # Everything else stays consistent, so the only thing wrong is the missing width.
    trimmed = signs[:5120] + signs[5120 + 6144 :]
    metadata = _metadata(**{"prism.hadamard.sign_widths": [5120, 17408], "prism.hadamard.sign_values": trimmed})
    assert sum(metadata["prism.hadamard.sign_widths"]) == len(trimmed)
    with pytest.raises(HadamardSpecError, match=f"does not declare the grouped width {GDN_GROUPED_WIDTH}"):
        parse_hadamard_spec(metadata)


def test_a_summarised_array_says_so_instead_of_misreading_it() -> None:
    """`GGUFReader` skips metadata arrays unless asked, and that is the default."""
    metadata = _metadata()
    metadata["prism.hadamard.sign_values"] = GGUFArraySummary(5, "int32", 28672)
    with pytest.raises(HadamardSpecError, match=r"read_arrays=True"):
        parse_hadamard_spec(metadata)


def test_declared_names_are_checked_against_the_file_when_given() -> None:
    fixture = _load_fixture()
    known = [name for name in fixture["weight_names"] if name != "blk.63.ffn_up.weight"]
    known += fixture["inverse_weight_names"]
    with pytest.raises(HadamardSpecError, match="not in the file"):
        parse_hadamard_spec(_metadata(), known_tensor_names=known)
    everything = fixture["weight_names"] + fixture["inverse_weight_names"]
    assert parse_hadamard_spec(_metadata(), known_tensor_names=everything) is not None


def test_the_block_reads_off_the_checkpoint_when_it_is_present() -> None:
    path = os.environ.get(CHECKPOINT_ENV, CHECKPOINT_DEFAULT)
    if not os.path.exists(path):
        pytest.skip(f"set {CHECKPOINT_ENV} or place the checkpoint at {CHECKPOINT_DEFAULT}")

    without_arrays = GGUFReader(path).read()
    assert has_hadamard_block(without_arrays.metadata)
    assert isinstance(without_arrays.metadata["prism.hadamard.weight_names"], GGUFArraySummary)
    # The default reader describes the arrays rather than materialising them, so the
    # parser cannot see a self-describing block through it.  That is the guard.
    with pytest.raises(HadamardSpecError, match=r"read_arrays=True"):
        parse_hadamard_spec(without_arrays.metadata)

    file = GGUFReader(path, read_arrays=True).read()
    spec = parse_hadamard_spec(file.metadata, known_tensor_names=file.tensors_by_name)
    assert spec is not None
    fixture = _load_fixture()
    assert spec.sign_widths == tuple(fixture["sign_widths"])
    assert spec.weight_names == tuple(fixture["weight_names"])
    assert spec.inverse_weight_names == tuple(fixture["inverse_weight_names"])
    packed = _unpack_signs(fixture["sign_values_bits"], sum(fixture["sign_widths"]))
    assert tuple(spec.sign_values) == tuple(packed)
    # Every folded tensor is a ternary one, and every ternary tensor is folded.
    ternary = {t.name for t in file.tensors if t.type_name == "ptq1_0"}
    assert ternary == set(spec.folded_names)
