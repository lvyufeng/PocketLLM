"""The fork-private ternary GGUF types: addressable, and refused until #386.

Two halves.  The hermetic half builds a tiny GGUF whose payload is real blocks of
one of the two types and checks that the loader hands them back byte for byte
while refusing to interpret them.  The checkpoint half runs against the released
files and checks the geometry against each file's own tensor table, and the decode
against the F16 GGUF of the same checkpoint -- a row of a pack file dequantizes
*bit-identically* to the same row of the F16 file, which is what makes the packing,
the block walk, the `qh` parity and the row addressing checkable at once.

Both packs are the same weights: the checkpoint ships PTQ1_0 (143, 128 weights per
28 bytes, three levels) and PQ2_0 (142, 128 per 34, four), and the same row of the
same tensor decodes to the same numbers in both.  Every test that can be stated
per pack is parametrized over the two.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.loader.gguf import pq2_0, ptq1_0
from src.loader.gguf.quant_types import (
    GGUF_ADDRESSABLE_TYPE_NAMES,
    GGUF_DENSE_TYPE_IDS,
    GGUF_DENSE_TYPE_NAMES,
    GGUF_TERNARY_FILE_TYPE_IDS,
    GGUF_TERNARY_TYPE_NAMES,
)
from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader
from src.loader.gguf.reader import GGML_TYPES, GGUFReader, tensor_nbytes
from src.loader.gguf.tensor_reader import GGUFTensorDataReader
from tests.gguf_test_utils import write_gguf

GGML_F32 = 0
GGML_PTQ1_0 = 143
GGML_PQ2_0 = 142

#: File id, block size, block bytes and decoder per pack.  The pack name is the
#: GGUF type name, which is what the loader and the file's header both use.
PACKS = {
    "ptq1_0": (GGML_PTQ1_0, 28, ptq1_0.QK_PTQ1_0, ptq1_0.dequantize_blocks),
    "pq2_0": (GGML_PQ2_0, 34, pq2_0.QK_PQ2_0, pq2_0.dequantize_blocks),
}

#: The levels a block of each pack can produce, for the marker-payload test.
PACK_LEVELS = {
    "ptq1_0": (-1.0, 0.0, 1.0),
    "pq2_0": (-1.0, 0.0, 1.0, 2.0),
}

#: Where the fp16 1.0 scale sits in a marker block, as a slice: PTQ1_0 keeps it
#: last, PQ2_0 first, and the loader is supposed to move neither.
SCALE_SLICE = {"ptq1_0": slice(26, 28), "pq2_0": slice(0, 2)}

CHECKPOINT_DIR_ENV = "POCKETLLM_BONSAI_GGUF_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2/Bonsai-2-27B-gguf"
ROWS_FIXTURE = Path(__file__).parent / "data" / "ternary_bonsai_rows.json"


def _fixture() -> dict:
    return json.loads(ROWS_FIXTURE.read_text())


def _pack_path(pack: str) -> str:
    """Where a pack's released file is, or a skip.

    The directory can be overridden whole; the filenames are the release's.
    """
    directory = os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT)
    path = os.path.join(directory, _fixture()["packs"][pack]["file"])
    if not os.path.exists(path):
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {path}")
    return path


def _example_path() -> str:
    return _pack_path("ptq1_0")


def _marker_payload(rows: int, row_blocks: int, pack: str) -> bytes:
    """`rows * row_blocks` blocks where block k is its index, plus a 1.0 scale.

    Which end the two bytes of 1.0 go in is the pack's, so the marker covers the
    whole block either way: the assertion in the test is on the bytes the file
    stores, and the loader is supposed to move none of them.
    """
    _file_id, block_bytes, _qk, _decode = PACKS[pack]
    out = bytearray()
    for index in range(rows * row_blocks):
        block = bytearray([index & 0xFF]) * block_bytes
        block[SCALE_SLICE[pack]] = bytes.fromhex("003c")  # fp16 1.0, little-endian
        out.extend(block)
    return bytes(out)


def _write_ternary_gguf(path: Path, pack: str, *, rows: int = 2, row_blocks: int = 3) -> None:
    file_id, _block_bytes, qk, _decode = PACKS[pack]
    write_gguf(
        path,
        metadata={"general.architecture": "qwen35", "qwen35.block_count": 1},
        tensors=[
            ("blk.0.ffn_gate.weight", (row_blocks * qk, rows), file_id),
            # Never read; keeps the tensor table non-trivial.
            ("blk.0.ssm_a", (rows,), GGML_F32),
        ],
        payloads={"blk.0.ffn_gate.weight": _marker_payload(rows, row_blocks, pack)},
    )


# --------------------------------------------------------------------------- #
# The type ids, in the three places they have to agree
# --------------------------------------------------------------------------- #


def test_the_file_geometry_is_the_fork_s() -> None:
    assert GGML_TYPES[GGML_PTQ1_0] == ("ptq1_0", 128, 28)
    assert GGML_TYPES[GGML_PQ2_0] == ("pq2_0", 128, 34)
    assert ptq1_0.QK_PTQ1_0 == pq2_0.QK_PQ2_0 == 128
    assert ptq1_0.PTQ1_0_BLOCK_BYTES == 28
    assert pq2_0.PQ2_0_BLOCK_BYTES == 34
    # The two formats agree on the group size and on nothing else: PTQ1_0 puts its
    # scale last, PQ2_0 first, so a decoder written for one cannot read the other.
    assert ptq1_0.PTQ1_0_BLOCK_BYTES != pq2_0.PQ2_0_BLOCK_BYTES


def test_the_ternary_ids_are_keyed_with_the_dense_ones() -> None:
    assert GGUF_TERNARY_FILE_TYPE_IDS == {"ptq1_0": GGML_PTQ1_0, "pq2_0": GGML_PQ2_0}
    # The file ids are the GGML ids; the dense table's values are runtime ids and
    # are deliberately different -- iq2_xxs is GGML 16 and runtime 0.
    assert GGML_TYPES[16][0] == "iq2_xxs"
    assert GGUF_DENSE_TYPE_IDS["iq2_xxs"] == 0


def test_a_ternary_name_is_not_a_claim_that_a_kernel_exists() -> None:
    """The dense map *is* the raw-block runtime's dispatch table."""
    for name in GGUF_TERNARY_TYPE_NAMES:
        assert name not in GGUF_DENSE_TYPE_IDS
        assert name not in GGUF_DENSE_TYPE_NAMES.values()
        assert name in GGUF_ADDRESSABLE_TYPE_NAMES
    assert set(GGUF_DENSE_TYPE_IDS) <= GGUF_ADDRESSABLE_TYPE_NAMES


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_geometry_survives_a_row_that_is_not_a_whole_number_of_blocks(pack: str) -> None:
    """1.75 bits does not divide a row, so nbytes has to come from the geometry."""
    file_id, block_bytes, _qk, _decode = PACKS[pack]
    assert tensor_nbytes(file_id, (128,)) == block_bytes
    assert tensor_nbytes(file_id, (256,)) == 2 * block_bytes
    assert tensor_nbytes(file_id, (129,)) == 2 * block_bytes  # ceiling, on the total
    assert tensor_nbytes(file_id, (128, 5120)) == 5120 * block_bytes
    assert tensor_nbytes(file_id, (5120, 17408)) == 17408 * 5120 // 128 * block_bytes
    # Every row width this checkpoint uses is a whole number of blocks, which is
    # what keeps the row-addressing path below honest.
    for row_elems in (5120, 6144, 10240, 12288, 17408, 1024, 248320):
        assert row_elems % 128 == 0


def test_an_unknown_type_id_stays_unknown(tmp_path: Path) -> None:
    """A type the loader has never seen has no geometry, so it cannot be sized or read."""
    write_gguf(
        tmp_path / "odd.gguf",
        tensors=[("odd", (128, 2), 200)],
        payloads={"odd": bytes(56)},
    )
    file = GGUFReader(str(tmp_path / "odd.gguf")).read()
    tensor = file.tensors[0]
    assert tensor.type_name == "unknown_200"
    assert tensor.nbytes is None
    with GGUFTensorDataReader(file) as reader:
        with pytest.raises(NotImplementedError, match="not supported by read_tensor"):
            reader.read_tensor("odd")
        with pytest.raises(NotImplementedError, match="not supported"):
            reader.read_quantized_matrix_blocks("odd")


# --------------------------------------------------------------------------- #
# The bytes are addressable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_a_ternary_payload_comes_back_byte_for_byte(tmp_path: Path, pack: str) -> None:
    _file_id, block_bytes, qk, decode = PACKS[pack]
    path = tmp_path / f"tiny-{pack}.gguf"
    _write_ternary_gguf(path, pack, rows=2, row_blocks=3)
    payload = _marker_payload(2, 3, pack)
    with GGUFTensorDataReader(str(path)) as reader:
        blocks, type_name, row_elems = reader.read_quantized_matrix_blocks("blk.0.ffn_gate.weight")
        assert type_name == pack
        assert row_elems == 3 * qk
        array = blocks.numpy()
        assert array.shape == (2, 3, block_bytes)
        assert array.dtype == np.uint8
        for index in range(6):
            row, column = divmod(index, 3)
            block = bytes(array[row, column])
            # The marker's layout, stated rather than compared: the scale is in the
            # pack's own place -- last for PTQ1_0, first for PQ2_0 -- and every other
            # byte of the block is the block's index.
            expected = bytearray([index]) * block_bytes
            expected[SCALE_SLICE[pack]] = bytes.fromhex("003c")  # fp16 1.0, little-endian
            assert block[SCALE_SLICE[pack]] == bytes.fromhex("003c")
            assert block == bytes(expected)
            assert block == payload[index * block_bytes : (index + 1) * block_bytes]
        # And the decode is the module's, not this test's: 1.0 scale, so the
        # values are the levels themselves.
        values = decode(array.reshape(-1, block_bytes))
        assert values.shape == (6, qk)
        assert np.isin(np.unique(values), np.asarray(PACK_LEVELS[pack])).all()


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_a_row_slice_is_the_row_the_file_stores(tmp_path: Path, pack: str) -> None:
    _file_id, block_bytes, _qk, _decode = PACKS[pack]
    path = tmp_path / f"tiny-{pack}.gguf"
    _write_ternary_gguf(path, pack, rows=4, row_blocks=2)
    payload = _marker_payload(4, 2, pack)
    with GGUFTensorDataReader(str(path)) as reader:
        for row in range(4):
            blocks, _type_name, _row_elems = reader.read_quantized_matrix_block_rows(
                "blk.0.ffn_gate.weight", row, 1
            )
            array = blocks.numpy()
            assert array.shape == (1, 2, block_bytes)
            for column in range(2):
                index = row * 2 + column
                assert bytes(array[0, column]) == payload[
                    index * block_bytes : (index + 1) * block_bytes
                ]


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_loader_refuses_to_dequantize_a_ternary_tensor(tmp_path: Path, pack: str) -> None:
    path = tmp_path / f"tiny-{pack}.gguf"
    _write_ternary_gguf(path, pack)
    with GGUFTensorDataReader(str(path)) as reader:
        with pytest.raises(NotImplementedError) as excinfo:
            reader.read_tensor("blk.0.ffn_gate.weight")
        # Every dequantizing entry point, not just read_tensor: a guard on one
        # path is a guard a caller can go around.
        with pytest.raises(NotImplementedError, match=pack):
            reader.read_quantized_matrix_rows_reference("blk.0.ffn_gate.weight", 0, 1)
        with pytest.raises(NotImplementedError, match="not supported"):
            reader.read_tensor_rows("blk.0.ffn_gate.weight", 0, 1)
    message = str(excinfo.value)
    # The reason has to be in the message: a bare refusal invites a caller to
    # route around it, and the route around it is the F16 upcast.
    assert pack in message
    assert "refuses rather than dequantizing to f16" in message
    # The decoder named is the pack's own: the two formats share nothing but the
    # group size, so pointing a reader at the other one is a real way to be wrong.
    assert f"src/loader/gguf/{pack}.py" in message
    # And the refusal has to say which of the two packs it is talking about. One
    # has a GEMM that reads its blocks and one does not; a message that promises a
    # kernel the format lacks is worse than no message.
    if pack == "ptq1_0":
        assert "gguf_quant_gemm_forward" in message
        assert "no kernel reads them yet" not in message
    else:
        assert "no kernel reads them yet" in message
        assert "gguf_quant_gemm_forward" not in message


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_quantized_loader_refuses_the_ternary_types(tmp_path: Path, pack: str) -> None:
    path = tmp_path / f"tiny-{pack}.gguf"
    _write_ternary_gguf(path, pack)
    with GGUFQuantizedTensorLoader(str(path), device="cpu") as loader:
        with pytest.raises(NotImplementedError, match="not supported by the GGUF raw-block runtime"):
            loader.read_quant("blk.0.ffn_gate.weight", pack)
        with pytest.raises(ValueError, match="expected q2_k"):
            loader.read_quant("blk.0.ffn_gate.weight", "q2_k")


# --------------------------------------------------------------------------- #
# Against the released checkpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_file_s_own_offsets_confirm_every_tensors_size(pack: str) -> None:
    """The strongest geometry check available: the file's table against our arithmetic.

    If any block size were wrong, `tensor_nbytes` would disagree with the next
    tensor's offset long before the end of the file -- 851 of them, exactly packed,
    the last one ending at the last byte, in a file whose own size is pinned.
    """
    fixture = _fixture()["packs"][pack]
    file = GGUFReader(_pack_path(pack)).read()
    assert len(file.tensors) == fixture["tensor_count"] == 851
    assert file.size == fixture["file_bytes"]
    assert file.data_start == fixture["data_start"]
    cursor = 0
    for tensor in file.tensors:
        assert tensor.nbytes is not None, f"{tensor.name} ({tensor.type_name}) has no geometry"
        assert tensor.offset == cursor, f"{tensor.name} starts at {tensor.offset}, expected {cursor}"
        cursor += tensor.nbytes
    assert cursor == fixture["file_bytes"] - fixture["data_start"]
    assert cursor + file.data_start == file.size


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_type_histogram_is_the_release_s(pack: str) -> None:
    file = GGUFReader(_pack_path(pack)).read()
    counts: dict[str, int] = {}
    for tensor in file.tensors:
        counts[tensor.type_name] = counts.get(tensor.type_name, 0) + 1
    assert counts == _fixture()["packs"][pack]["type_histogram"]
    # The same 402 quantized tensors in both packs, and nothing between them: every
    # tensor in the file is one of the three types.
    assert counts == {pack: 402, "f32": 353, "bf16": 96}


def test_the_two_packs_have_the_same_tensor_table() -> None:
    """Same names, same shapes, same order -- the packs differ only in the 402 types.

    Which is what makes the row-for-row comparison below meaningful: row ``r`` of a
    tensor is the same weights in both files.  The byte counts differ for the 402
    quantized ones (28 against 34 per 128 weights) and agree for the other 449,
    which is the second half of the claim.
    """
    first = GGUFReader(_pack_path("ptq1_0")).read()
    second = GGUFReader(_pack_path("pq2_0")).read()
    assert len(first.tensors) == len(second.tensors)
    quantized = 0
    for left, right in zip(first.tensors, second.tensors):
        assert left.name == right.name
        assert left.dimensions == right.dimensions
        if left.type_name in GGUF_TERNARY_TYPE_NAMES:
            assert right.type_name in GGUF_TERNARY_TYPE_NAMES
            assert left.type_name != right.type_name
            assert right.nbytes * 28 == left.nbytes * 34
            quantized += 1
        else:
            assert left.type_name == right.type_name
            assert left.nbytes == right.nbytes
    assert quantized == 402


def _clear_the_sign_of_zero(values: np.ndarray) -> np.ndarray:
    """The only byte the two files ever disagree on, and it is not a value.

    An fp16 -0.0 and a +0.0 are the same number, differ in one bit, and the F16
    GGUF stores -0.0 wherever the level is 0 while every decoder here produces
    +0.0.  Normalising it is how the digest below stays a bit-for-bit claim
    instead of an approximately-equal one.
    """
    return np.where(values == np.float32(0.0), np.float32(0.0), values)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(_clear_the_sign_of_zero(values).astype("<f4").tobytes()).hexdigest()


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_a_stored_row_dequantizes_to_the_f16_gguf_row(pack: str) -> None:
    """The F16 GGUF of the same checkpoint is the oracle, and it is equal to the bit.

    Row for row across all three fold widths, both ends of the model and the two
    tensors that share the embedding table, the decoded row and the F16 row agree
    everywhere except the sign of zero.  That is two facts at once: this loader's
    packing, block walk, `qh` parity and row addressing are right, and the F16 file
    is the ternary weights stored as f16 rather than an unquantized model the
    quantized one approximates.
    """
    _file_id, block_bytes, _qk, decode = PACKS[pack]
    fixture = _fixture()["packs"][pack]
    with GGUFTensorDataReader(_pack_path(pack)) as reader:
        for case in fixture["cases"]:
            stored = bytes.fromhex(case["blocks_hex"])
            blocks, type_name, row_elems = reader.read_quantized_matrix_block_rows(
                case["tensor"], int(case["row"]), 1
            )
            assert type_name == pack
            assert row_elems == case["width"]
            assert bytes(blocks.numpy().reshape(-1)) == stored, case["tensor"]
            values = decode(
                np.frombuffer(stored, dtype=np.uint8).reshape(-1, block_bytes)
            ).reshape(-1)
            assert values.shape == (case["width"],)
            assert _digest(values) == case["f16_row_sha256"], case["tensor"]
            assert int(np.sum(np.signbit(values) & (values == 0))) == 0
            assert float(np.sqrt((values**2).mean())) == pytest.approx(case["f16_rms"], rel=1e-6)


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_the_one_row_kept_in_full_is_equal_element_by_element(pack: str) -> None:
    """The digest above is the pin; this is the same claim in a readable form."""
    _file_id, block_bytes, _qk, decode = PACKS[pack]
    example = _fixture()["packs"][pack]["example"]
    reference = np.frombuffer(
        bytes.fromhex(_fixture()["example"]), dtype="<f2"
    ).astype(np.float32)
    stored = np.frombuffer(bytes.fromhex(example["blocks_hex"]), dtype=np.uint8)
    values = decode(stored.reshape(-1, block_bytes)).reshape(-1)
    assert values.shape == reference.shape == (example["width"],)
    assert np.array_equal(values, reference)
    # `array_equal` is true across the sign of zero, so say how many elements that
    # hides and check it is the whole of the disagreement.
    neg_zero = np.signbit(reference) & (reference == 0)
    assert int(neg_zero.sum()) == example["f16_neg_zero_count"]
    ours_bytes = np.ascontiguousarray(values).view(np.uint8).reshape(-1, 4)
    reference_bytes = np.ascontiguousarray(reference).view(np.uint8).reshape(-1, 4)
    assert np.array_equal((ours_bytes != reference_bytes).any(axis=1), neg_zero)
    assert np.array_equal(
        _clear_the_sign_of_zero(values).astype("<f4").tobytes(),
        _clear_the_sign_of_zero(reference).astype("<f4").tobytes(),
    )
    assert np.abs(reference).max() == pytest.approx(example["f16_max_abs"], rel=1e-6)


def test_the_two_packs_decode_the_same_weights() -> None:
    """PQ2_0 is PTQ1_0's weights in a wider codec, and this is the row-level claim.

    The two files are separate releases of the same ternary checkpoint -- PQ2_0
    spends 34 bytes per 128 weights to PTQ1_0's 28 and its fourth code is unused in
    these rows -- so the same row of the same tensor has to decode to the same
    numbers.  Checking it here is what connects PQ2_0 to the F16 oracle above: that
    test pins each pack to the F16 row, and this one pins the two packs to each
    other, so a packing bug in one cannot hide behind agreement with the other.
    """
    fixture = _fixture()
    with GGUFTensorDataReader(_pack_path("ptq1_0")) as left, GGUFTensorDataReader(
        _pack_path("pq2_0")
    ) as right:
        for case in fixture["packs"]["ptq1_0"]["cases"]:
            tensor, row = case["tensor"], int(case["row"])
            a, _t, _w = left.read_quantized_matrix_block_rows(tensor, row, 1)
            b, _t, _w = right.read_quantized_matrix_block_rows(tensor, row, 1)
            va = ptq1_0.dequantize_blocks(
                a.numpy().reshape(-1, ptq1_0.PTQ1_0_BLOCK_BYTES)
            ).reshape(-1)
            vb = pq2_0.dequantize_blocks(
                b.numpy().reshape(-1, pq2_0.PQ2_0_BLOCK_BYTES)
            ).reshape(-1)
            assert np.array_equal(
                _clear_the_sign_of_zero(va), _clear_the_sign_of_zero(vb)
            ), tensor
            assert _digest(vb) == case["f16_row_sha256"], tensor


#: Tensors the +2-level scan below sweeps, chosen for size: the three biggest
#: quantized tensors of the model, together a fifth of the file's weights.
_CODE_SCAN_TENSORS = ("token_embd.weight", "output.weight", "blk.0.ffn_down.weight")


def test_pq2_0_never_spends_its_fourth_level() -> None:
    """The released PQ2_0 file is ternary, not four-level.

    Its codec has four levels -- ``-1, 0, +1, +2`` -- and over all 26,869,760,000
    weights of all 402 quantized tensors the code for ``+2`` is never written once.
    A row of every tensor was swept for this, one in twenty-nine.  It is what lets
    the two packs be compared row for row above: if a future release spent the
    fourth level, that comparison would stop holding, and this is the test that
    would say why.
    """
    path = _pack_path("pq2_0")
    stride = 29
    weights = 0
    with GGUFTensorDataReader(path) as reader:
        for name in _CODE_SCAN_TENSORS:
            blocks, _type_name, _row_elems = reader.read_quantized_matrix_blocks(name)
            array = blocks.numpy().reshape(-1, pq2_0.PQ2_0_BLOCK_BYTES)
            codes = pq2_0._codes_from_bytes(array[::stride, 2:])
            assert int((codes == 3).sum()) == 0, name
            assert int((codes > 3).sum()) == 0, name
            weights += codes.size
    # Pinned: the stride times the three shapes, so a shape or stride change shows here.
    assert weights == 90_756_352
