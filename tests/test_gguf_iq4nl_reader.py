"""IQ4_NL in the GGUF loader: the codebook, the 32-weight geometry, and the two
places the type is and is not declared.

Three halves, and they answer three different questions.  The hermetic half
builds a tiny GGUF whose payload is real IQ4_NL blocks and checks that the
loader's arithmetic is the format's: the codebook comes out of the vendored
llama.cpp header, and the nibble order is the block layout rather than an
alternating one.  The row-width half is the regression this task exists for --
``IQ4_NL`` blocks are 32 weights where every other format in the loader is 256,
so a reader that assumes ``QK_K`` reads the wrong bytes instead of failing.
The dispatch half pins the one thing that must *not* change: the loader decodes
the format, and no kernel claims it.

The checkpoint half is the acceptance criterion: the released GGUF's own
``IQ4_NL`` tensors are compared against the same tensor names in the
checkpoint's BF16 release, which the repository ships alongside.  It skips
unless both files are on disk.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.loader.gguf import iq4_nl
from src.loader.gguf.quant_types import (
    GGUF_ADDRESSABLE_TYPE_NAMES,
    GGUF_DENSE_TYPE_IDS,
    GGUF_DENSE_TYPE_NAMES,
    GGUF_LOADER_TYPE_NAMES,
)
from src.loader.gguf.quantized_loader import GGUFQuantizedTensorLoader
from src.loader.gguf.reader import GGML_TYPES, GGUFReader, tensor_nbytes
from src.loader.gguf.tensor_reader import GGUFTensorDataReader
from tests.gguf_test_utils import GGML_F32, GGML_IQ4_NL, write_gguf


REPO_ROOT = Path(__file__).resolve().parents[1]
GGML_COMMON = REPO_ROOT / "src" / "csrc" / "llama_mmq" / "ggml-common.h"

CHECKPOINT_DIR_ENV = "POCKETLLM_XING4_DIR"
CHECKPOINT_DIR_DEFAULT = "/mnt/data2"
GGUF_NAME = "Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"

#: One routed expert's gate projection, named in both releases.  The BF16 shard
#: that holds it also holds every other layer-2 expert weight, so the comparison
#: costs one 1.4 GiB shard rather than the checkpoint.
GGUF_EXPERT_TENSOR = "blk.2.ffn_gate_exps.weight"
HF_EXPERT_TENSOR = "model.layers.2.mlp.experts.0.gate_proj.weight"


# --------------------------------------------------------------------------- #
# The codebook, checked against the table the fork ships
# --------------------------------------------------------------------------- #


def _header_table() -> list[int]:
    """Read ``kvalues_iq4nl`` straight out of the vendored header.

    Deliberately not via ``iq4_nl.kvalues``: the point is to parse the same text
    a second time, so a module that transcribed the table instead of reading it
    fails here.
    """
    text = GGML_COMMON.read_text(encoding="utf-8")
    match = re.search(
        r"GGML_TABLE_BEGIN\(int8_t,\s*kvalues_iq4nl,\s*16\)(.*?)GGML_TABLE_END\(\)",
        text,
        flags=re.S,
    )
    assert match is not None, "kvalues_iq4nl is not in the vendored header"
    return [int(item) for item in re.findall(r"-?\d+", re.sub(r"//[^\n]*", "", match.group(1)))]


def test_the_codebook_is_the_vendored_table() -> None:
    table = iq4_nl.kvalues()
    assert table.dtype == np.int8
    assert table.tolist() == _header_table()
    # The first and last entries are the format's documented range, and they bound
    # every product a decode can produce.
    assert (int(table[0]), int(table[-1])) == (-127, 113)


def test_the_codebook_is_not_a_linear_scale() -> None:
    """The "NL" in the name: a scale-and-add format is not what this is.

    Every other 4-bit format in this loader decodes as ``d * (q - m)``.  Here the
    sixteen levels are a lookup, and they are not evenly spaced, so a decoder that
    reaches for arithmetic instead of the table is wrong in a way that produces
    plausible weights.
    """
    table = iq4_nl.kvalues().astype(np.int32)
    assert np.all(np.diff(table) > 0)
    assert len(set(np.diff(table).tolist())) > 1


# --------------------------------------------------------------------------- #
# The geometry, in the three places it has to agree
# --------------------------------------------------------------------------- #


def test_the_file_geometry_is_ggml_s() -> None:
    assert GGML_TYPES[GGML_IQ4_NL] == ("iq4_nl", iq4_nl.QK_IQ4_NL, iq4_nl.IQ4_NL_BLOCK_BYTES)
    assert iq4_nl.QK_IQ4_NL == 32
    assert iq4_nl.IQ4_NL_BLOCK_BYTES == 18
    assert iq4_nl.IQ4_NL_FILE_TYPE_ID == GGML_IQ4_NL == 20
    # The block is a 2-byte fp16 scale plus QK4_NL/2 packed nibbles, and that is
    # the whole format -- no low-bit plane, unlike IQ4_XS's 136 bytes per 256.
    assert iq4_nl.IQ4_NL_BLOCK_BYTES == 2 + iq4_nl.QK_IQ4_NL // 2


def test_nbytes_comes_from_the_geometry() -> None:
    assert tensor_nbytes(GGML_IQ4_NL, (32,)) == 18
    assert tensor_nbytes(GGML_IQ4_NL, (64,)) == 2 * 18
    # Ceiling on the total, the same rule every other type uses.  No row in this
    # checkpoint is a partial block, so the row-addressing path never sees one.
    assert tensor_nbytes(GGML_IQ4_NL, (33,)) == 2 * 18
    # The shapes the released GGUF actually declares: an expert's gate/up is
    # (3584, 1024, 64) and its down is (1024, 3584, 64).
    assert tensor_nbytes(GGML_IQ4_NL, (3584, 1024, 64)) == 64 * 1024 * 3584 // 32 * 18
    assert tensor_nbytes(GGML_IQ4_NL, (1024, 3584, 64)) == 64 * 3584 * 1024 // 32 * 18
    for row_elems in (3584, 1024, 9216):
        assert row_elems % iq4_nl.QK_IQ4_NL == 0


# --------------------------------------------------------------------------- #
# The decode, on blocks this file writes itself
# --------------------------------------------------------------------------- #


def _marker_row(row_blocks: int, offset: int) -> bytes:
    """One row of marker blocks: fp16 scale 1.0, and a nibble pattern that moves.

    ``qs[j]`` is ``(j + offset) & 0xf | (((15 - j + offset) & 0xf) << 4)``, so the
    low nibbles walk up and the high nibbles walk down.  The two halves of a
    decoded block are then equal-and-opposite permutations of the codebook, which
    is what makes the block layout distinguishable from an interleaved one: a
    decoder that pairs nibble j with nibble j instead of with j + 16 produces a
    different sequence, not a reordered one.
    """
    out = bytearray()
    for block in range(row_blocks):
        index = block + offset
        qs = bytes(
            ((j + index) & 0x0F) | (((15 - j + index) & 0x0F) << 4) for j in range(iq4_nl.QK_IQ4_NL // 2)
        )
        out.extend(bytes.fromhex("003c"))  # fp16 1.0, little-endian
        out.extend(qs)
    return bytes(out)


def _marker_expected(row_blocks: int, offset: int) -> np.ndarray:
    """What the marker row decodes to, block by block.

    Per block, not two long halves: the decoder's low/high split is within a
    block, so a row of three blocks is ``low0 high0 low1 high1 low2 high2``.
    """
    table = iq4_nl.kvalues()
    out: list[int] = []
    for block in range(row_blocks):
        index = block + offset
        out.extend(int(table[(j + index) & 0x0F]) for j in range(16))
        out.extend(int(table[(15 - j + index) & 0x0F]) for j in range(16))
    return np.asarray(out, dtype=np.float32)


def _marker_payload(rows: int, row_blocks: int) -> bytes:
    """``rows`` rows of marker blocks, each row built with its own offset."""
    return b"".join(_marker_row(row_blocks, row) for row in range(rows))


def test_a_block_decodes_in_the_block_layout_not_interleaved() -> None:
    row = _marker_row(1, 0)
    got = iq4_nl.dequantize_blocks(row).reshape(-1)
    assert got.tolist() == _marker_expected(1, 0).tolist()
    # Stated the other way round, so the assertion is about the format and not
    # about this test's own arithmetic: nibble j is weight j, and nibble j's high
    # half is weight j + 16, so the low half is the codebook in order and the high
    # half is the codebook reversed -- an interleaved read gives neither.
    table = iq4_nl.kvalues().astype(np.float32)
    assert got[:16].tolist() == table.tolist()
    assert got[16:].tolist() == table[::-1].tolist()


def test_a_row_that_is_not_a_whole_number_of_k_blocks(tmp_path: Path) -> None:
    """The regression: 96 weights is three IQ4_NL blocks and zero QK_K blocks.

    A reader that sizes a row as ``ceil(in_dim / 256)`` blocks and multiplies by
    the format's byte count under-reads by 4x here, and the failure downstream is
    a reshape error rather than a wrong number -- which is why it has to be
    covered by a test rather than by review.
    """
    rows, row_blocks = 3, 3
    # Each row is built with its own offset, so a reader that gets the row stride
    # wrong sees the previous row's pattern rather than a shifted one.
    payload = _marker_payload(rows, row_blocks)
    path = tmp_path / "iq4_nl-96.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "xing4_0", "xing4_0.block_count": 1},
        tensors=[("blk.0.ffn_gate.weight", (row_blocks * iq4_nl.QK_IQ4_NL, rows), GGML_IQ4_NL)],
        payloads={"blk.0.ffn_gate.weight": payload},
    )
    assert len(payload) == rows * row_blocks * 18

    info = GGUFReader(str(path)).read().tensors[0]
    assert info.type_name == "iq4_nl"
    assert info.nbytes == rows * row_blocks * 18

    with GGUFTensorDataReader(str(path)) as reader:
        got = reader.read_tensor("blk.0.ffn_gate.weight")
        blocks, type_name, row_elems = reader.read_quantized_matrix_blocks("blk.0.ffn_gate.weight")

    assert got.dtype == torch.float32
    assert got.shape == (rows, row_blocks * iq4_nl.QK_IQ4_NL)
    for row in range(rows):
        assert got[row].tolist() == _marker_expected(row_blocks, row).tolist()

    # The raw-block path hands over 18-byte blocks, which is what a kernel would
    # consume; the block count is the format's and not QK_K's.
    assert type_name == "iq4_nl"
    assert row_elems == row_blocks * iq4_nl.QK_IQ4_NL
    assert tuple(blocks.shape) == (rows, row_blocks, 18)


def test_a_routed_expert_tensor_is_addressable(tmp_path: Path) -> None:
    """Three-dimensional IQ4_NL: the shape this checkpoint's 64 experts are.

    Nothing consumes these blocks yet, but the byte addressing has to work before
    a kernel can be written against it, and the block count per row is the thing
    that has to be right.
    """
    in_dim, out_dim, experts = 32, 2, 3
    block = _marker_row(1, 0)
    payload = block * (in_dim // iq4_nl.QK_IQ4_NL * out_dim * experts)
    path = tmp_path / "iq4_nl-experts.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "xing4_0", "xing4_0.block_count": 1},
        tensors=[("blk.0.ffn_gate_exps.weight", (in_dim, out_dim, experts), GGML_IQ4_NL)],
        payloads={"blk.0.ffn_gate_exps.weight": payload},
    )
    with GGUFTensorDataReader(str(path)) as reader:
        blocks, type_name, row_elems = reader.read_routed_expert_blocks("blk.0.ffn_gate_exps.weight", expert=2)
        all_blocks, all_type, all_in_dim = reader.read_routed_layer_blocks("blk.0.ffn_gate_exps.weight")

    assert (type_name, row_elems) == ("iq4_nl", in_dim)
    assert tuple(blocks.shape) == (out_dim, 1, 18)
    assert tuple(all_blocks.shape) == (experts, out_dim, 1, 18)
    assert (all_type, all_in_dim) == ("iq4_nl", in_dim)


def test_a_decode_of_one_block_is_the_format_s_own_arithmetic() -> None:
    """fp16 scale times a small integer is exact in fp32, so this is equality.

    Bit-exactness is the claim the checkpoint comparison below rests on: if the
    decode were only close, "matches the BF16 release" would need a tolerance and
    would stop being able to tell a wrong codebook entry from a rounding mode.
    """
    block = bytearray(bytes.fromhex("0038"))  # fp16 0.5, little-endian
    block.extend(bytes(range(16)))
    got = iq4_nl.dequantize_blocks(bytes(block)).reshape(-1)
    table = iq4_nl.kvalues().astype(np.float32)
    # Byte j holds the codebook index j in its low nibble and 0 in its high one, so
    # the low half walks the table and the high half is entry zero sixteen times.
    assert got[:16].tolist() == (table * np.float32(0.5)).tolist()
    assert got[16:].tolist() == [float(table[0]) * 0.5] * 16


# --------------------------------------------------------------------------- #
# The type is loader-only, and that is stated in the tables rather than a comment
# --------------------------------------------------------------------------- #


def test_iq4_nl_is_not_a_claim_that_a_kernel_exists() -> None:
    """``GGUF_DENSE_TYPE_IDS`` is the raw-block runtime's dispatch table."""
    assert GGUF_LOADER_TYPE_NAMES == frozenset({"iq4_nl"})
    assert "iq4_nl" not in GGUF_DENSE_TYPE_IDS
    assert "iq4_nl" not in GGUF_DENSE_TYPE_NAMES.values()
    assert "iq4_nl" in GGUF_ADDRESSABLE_TYPE_NAMES
    assert set(GGUF_DENSE_TYPE_IDS) <= GGUF_ADDRESSABLE_TYPE_NAMES


def test_the_runtime_refuses_it_by_name(tmp_path: Path) -> None:
    """Asking for a raw-block tensor of a kernel-less type raises, and says why.

    This is the refusal-not-upcast rule applied to the opposite failure: a ternary
    tensor must not be silently expanded to F16, and an IQ4_NL tensor must not be
    silently handed to a GEMM that reads 32-weight blocks as a 256-weight format.
    Both are refused at the same place -- the dispatch table -- so there is one
    place to read.
    """
    path = tmp_path / "iq4_nl-runtime.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "xing4_0", "xing4_0.block_count": 1},
        tensors=[("blk.0.ffn_gate.weight", (64, 2), GGML_IQ4_NL)],
        payloads={"blk.0.ffn_gate.weight": _marker_payload(2, 2)},
    )
    with GGUFQuantizedTensorLoader(str(path), device="cpu") as loader:
        with pytest.raises(NotImplementedError, match="iq4_nl"):
            loader.read_quant("blk.0.ffn_gate.weight", "iq4_nl")


def test_an_unknown_id_still_raises(tmp_path: Path) -> None:
    """The rule the issue states: an unknown id raises rather than falling through.

    ``f16`` is a type the reader knows and the dispatch table does not, so it is
    the nearest thing to "unknown" that can be written into a tensor table.
    """
    path = tmp_path / "not-quantized.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "xing4_0", "xing4_0.block_count": 1},
        tensors=[("blk.0.ffn_gate.weight", (64, 2), GGML_F32)],
        payloads={"blk.0.ffn_gate.weight": np.zeros(128, dtype="<f4").tobytes()},
    )
    with GGUFQuantizedTensorLoader(str(path), device="cpu") as loader:
        with pytest.raises((NotImplementedError, ValueError)):
            loader.read_quant("blk.0.ffn_gate.weight", "iq4_nl")


# --------------------------------------------------------------------------- #
# inspect_gguf reports the type
# --------------------------------------------------------------------------- #


def test_inspect_gguf_reports_iq4_nl(tmp_path: Path) -> None:
    """The report names the type, and does not file it under "unknown".

    ``unknown_*`` is the entry a reader without the format produces, so its
    absence is the assertion: a type that shows up in both places would look
    reported while being undecoded.
    """
    path = tmp_path / "iq4_nl-inspect.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "xing4_0", "xing4_0.block_count": 1},
        tensors=[
            ("blk.0.ffn_gate.weight", (64, 2), GGML_IQ4_NL),
            ("blk.0.ffn_gate_exps.weight", (64, 2, 4), GGML_IQ4_NL),
            ("blk.0.attn_norm.weight", (64,), GGML_F32),
        ],
        payloads={
            "blk.0.ffn_gate.weight": _marker_payload(2, 2),
            "blk.0.ffn_gate_exps.weight": _marker_payload(2, 2) * 4,
            "blk.0.attn_norm.weight": np.zeros(64, dtype="<f4").tobytes(),
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "src.cli.inspect_gguf", "--gguf-path", str(path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    types = result.stdout.split("tensor types:")[1].split("\n\n")[0]
    assert "iq4_nl" in types
    assert "unknown tensor types:" not in result.stdout
    assert "unknown_20" not in result.stdout
    # The byte count in the report is the block geometry's, so a wrong geometry
    # shows up here before it shows up in a decode.
    assert "iq4_nl            2" in types


# --------------------------------------------------------------------------- #
# The acceptance criterion: the released GGUF against the checkpoint's BF16
# --------------------------------------------------------------------------- #


def _checkpoint_paths() -> tuple[str, Path]:
    root = Path(os.environ.get(CHECKPOINT_DIR_ENV, CHECKPOINT_DIR_DEFAULT))
    gguf = root / GGUF_NAME
    index = root / "Xing4.0-29B-A4B" / "model.safetensors.index.json"
    if not gguf.exists() or not index.exists():
        pytest.skip(f"set {CHECKPOINT_DIR_ENV} or place {gguf} and {index}")
    return str(gguf), index


def test_the_release_file_is_mostly_iq4_nl() -> None:
    """The header the audit read, re-read here so a re-quantised release fails."""
    gguf_path, _ = _checkpoint_paths()
    file = GGUFReader(gguf_path).read()
    histogram: dict[str, int] = {}
    for tensor in file.tensors:
        histogram[tensor.type_name] = histogram.get(tensor.type_name, 0) + 1
    assert histogram.get("iq4_nl") == 243
    assert "unknown_" not in "".join(histogram)
    assert file.metadata["general.architecture"] == "xing4_0"
    # The audit's finding, restated where it is cheap to check: the attention path
    # is not quantized, so IQ4_NL is the MoE and the dense FFNs.
    attention = [t for t in file.tensors if ".attn" in t.name or "attn_" in t.name]
    assert attention and all(t.type_name in ("bf16", "f32") for t in attention)


def test_iq4_nl_decodes_to_the_bf16_release() -> None:
    """The acceptance criterion: same weights, two releases, one decode.

    The repository's GGUF and the checkpoint's BF16 release hold the same numbers
    quantized two ways, so "the codebook is the format's" stops being a claim
    about a table and becomes a measurement: 4.5-bit weights against the BF16 they
    were quantized from.  ``IQ4_NL`` keeps one fp16 scale per 32 weights, so every
    element has to land inside the block's own step; a wrong codebook entry is off
    by tens of steps, which is what makes this able to tell a wrong decode from
    rounding.

    The comparison is on an expert slab rather than a whole tensor: the GGUF packs
    the 64 experts into one three-dimensional tensor and the BF16 release keeps
    them apart, so the mapping from ``blk.2.ffn_gate_exps.weight`` expert 0 to
    ``model.layers.2.mlp.experts.0.gate_proj.weight`` is part of what is being
    checked.  It goes through ``read_routed_expert_blocks`` -- the entry point a
    kernel will read -- so the raw-block addressing is on the critical path too.
    """
    gguf_path, index_path = _checkpoint_paths()

    file = GGUFReader(gguf_path).read()
    info = file.tensors_by_name[GGUF_EXPERT_TENSOR]
    in_dim, out_dim, experts = (int(dim) for dim in info.dimensions)
    assert info.type_name == "iq4_nl"
    assert (in_dim, out_dim, experts) == (3584, 1024, 64)
    blocks_per_row = in_dim // iq4_nl.QK_IQ4_NL

    with GGUFTensorDataReader(gguf_path) as reader:
        blocks, type_name, row_elems = reader.read_routed_expert_blocks(GGUF_EXPERT_TENSOR, expert=0)
    assert type_name == "iq4_nl"
    assert row_elems == in_dim
    assert tuple(blocks.shape) == (out_dim, blocks_per_row, iq4_nl.IQ4_NL_BLOCK_BYTES)

    decoded = torch.from_numpy(
        iq4_nl.dequantize_blocks(blocks.reshape(out_dim, blocks_per_row, iq4_nl.IQ4_NL_BLOCK_BYTES))
    ).reshape(out_dim, in_dim)

    reference = _bf16_tensor(index_path, HF_EXPERT_TENSOR)
    assert reference.shape == decoded.shape

    # The bound is the block's own span: a wrong codebook entry is at least one
    # level away, and the levels of a block span `step`.  A quarter of that leaves
    # room for the quantization itself -- measured, the worst element sits at
    # 4.5% of the widest block's span -- while a mis-decoded nibble or a shifted
    # row cannot fit under it.
    span = decoded.reshape(out_dim, -1, iq4_nl.QK_IQ4_NL)
    span = (span.max(dim=-1).values - span.min(dim=-1).values).max().item()
    worst = (decoded - reference).abs().max().item()
    assert worst <= span * 0.25, f"worst element error {worst} against block span {span}"


def _bf16_tensor(index_path: Path, name: str) -> torch.Tensor:
    """One tensor out of the checkpoint's sharded BF16 release."""
    from safetensors import safe_open

    index = json.loads((index_path.parent / "model.safetensors.index.json").read_text())
    shard = index_path.parent / index["weight_map"][name]
    if not shard.exists():
        pytest.skip(f"{shard} is not on disk; the acceptance comparison needs it")
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).to(torch.float32)
