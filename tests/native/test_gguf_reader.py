"""The C reader must parse a checkpoint exactly as the Python oracle does.

`python/pocketllm/loader/gguf/` is the normative reader; the C one is checked
against it rather than against the format document, because the Python side is
already tested against real llama.cpp output and a second reading of the spec
would only introduce a second way to be wrong.

The comparison runs `pocketllm-gguf-dump`, which links the reader directly and
prints the directory as JSON. That is deliberate: exposing the directory through
the public ABI to make it testable would put a function in `pocketllm.h` that no
host actually calls, and the ABI is meant to be exactly what a host needs.

Everything here skips when either the library is unbuilt or the checkpoint is
absent, because a contributor without a 1.5 GB f16 file must still be able to
run the suite. A skip is not a pass.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from pocketllm import native

#: The checkpoint the reader is checked on. It is the f16 conversion of
#: `Qwen/Qwen3-0.6B`, produced by llama.cpp's `convert_hf_to_gguf.py` -- see
#: `docs/models/`. CI has neither the file nor the engine, so these skip there;
#: they are what a developer runs before touching the reader.
CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3-0.6b-f16.gguf")


def _dump_tool() -> pathlib.Path:
    return native._repository_root() / "build" / "pocketllm-gguf-dump"


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_dump = pytest.mark.skipif(not _dump_tool().is_file(), reason="pocketllm-gguf-dump is not built")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)

pytestmark = [needs_engine, needs_dump, needs_checkpoint]


@pytest.fixture(scope="module")
def dumped() -> dict:
    result = subprocess.run([str(_dump_tool()), str(CHECKPOINT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def oracle() -> object:
    from pocketllm.loader.gguf.reader import GGUFReader

    return GGUFReader(str(CHECKPOINT)).read()


def test_the_header_agrees(dumped: dict, oracle: object) -> None:
    """Version, counts, alignment and where the data starts.

    `data_start` is the one worth pinning: it is `align_up` of the position
    after the tensor directory, and an off-by-one there shifts every tensor into
    the wrong bytes without failing any other check.
    """
    assert dumped["version"] == oracle.version
    assert dumped["tensor_count"] == oracle.tensor_count
    assert dumped["metadata_count"] == oracle.metadata_count
    assert dumped["alignment"] == oracle.alignment
    assert dumped["data_start"] == oracle.data_start
    assert dumped["size"] == oracle.size


def test_the_metadata_map_agrees(dumped: dict, oracle: object) -> None:
    """Every key, every value, both directions.

    A key present on one side only is a reader that misparsed a length and
    desynchronized, so this asserts set equality before comparing values --
    a value-by-value loop over the intersection would silently pass on a reader
    that had lost half the file.
    """
    theirs, ours = dumped["metadata"], oracle.metadata
    assert set(theirs) == set(ours)


def test_scalar_metadata_values_agree(dumped: dict, oracle: object) -> None:
    theirs, ours = dumped["metadata"], oracle.metadata
    for key in sorted(set(theirs) & set(ours)):
        mine, theirs_value = theirs[key], ours[key]
        # Arrays are compared separately: the Python reader summarizes them by
        # default and the dump prints type and length, so there is no value to
        # compare here.
        if isinstance(mine, dict) and mine.get("__array__"):
            continue
        assert mine == pytest.approx(theirs_value) if isinstance(theirs_value, float) else mine == theirs_value, key


def test_metadata_arrays_report_the_same_type_and_length(dumped: dict, oracle: object) -> None:
    """The tokenizer vocabulary and every other array, by shape.

    The C reader materialises arrays and the Python one summarizes them, which
    is a deliberate difference -- the C side has to use the vocabulary. What
    must not differ is the *shape*: a misread item type would desynchronize the
    cursor and corrupt every key after it.
    """
    theirs, ours = dumped["metadata"], oracle.metadata
    for key in sorted(set(theirs) & set(ours)):
        mine = theirs[key]
        if not (isinstance(mine, dict) and mine.get("__array__")):
            continue
        summary = ours[key]
        assert mine["item_type"] == summary.value_type, key
        assert mine["length"] == summary.length, key


def test_the_tensor_directory_agrees(dumped: dict, oracle: object) -> None:
    """Every tensor's name, type, offset and byte size.

    `nbytes` is the check that catches a wrong block geometry: the table here
    and reader.py's `GGML_TYPES` are transcribed from each other, and a
    transposed digit in a block size shows up as a size mismatch on exactly the
    tensors using that type.
    """
    theirs = {t["name"]: t for t in dumped["tensors"]}
    ours = {t.name: t for t in oracle.tensors}
    assert set(theirs) == set(ours)

    mismatches = []
    for name in sorted(theirs):
        mine, reference = theirs[name], ours[name]
        if mine["type_id"] != reference.type_id:
            mismatches.append(f"{name}: type {mine['type_id']} != {reference.type_id}")
        if mine["type"] != reference.type_name:
            mismatches.append(f"{name}: type name {mine['type']} != {reference.type_name}")
        if mine["offset"] != reference.offset:
            mismatches.append(f"{name}: offset {mine['offset']} != {reference.offset}")
        if mine["absolute_offset"] != reference.absolute_offset:
            mismatches.append(f"{name}: absolute_offset {mine['absolute_offset']} != {reference.absolute_offset}")
        if mine["nbytes"] != (reference.nbytes or 0):
            mismatches.append(f"{name}: nbytes {mine['nbytes']} != {reference.nbytes}")
        if tuple(mine["dims"]) != tuple(reference.dimensions):
            mismatches.append(f"{name}: dims {mine['dims']} != {reference.dimensions}")
    assert not mismatches, "\n".join(mismatches)


def test_every_tensor_fits_inside_the_file(dumped: dict) -> None:
    """No tensor's bytes run past the end of the mapping.

    A truncated download parses fine up to the tensor that is short; this is the
    check that turns that into a failure at load time rather than a segfault
    during the first matmul.
    """
    for tensor in dumped["tensors"]:
        assert tensor["absolute_offset"] + tensor["nbytes"] <= dumped["size"], tensor["name"]


def test_the_architecture_is_the_one_expected(dumped: dict) -> None:
    """Qwen3, at the geometry the plan locked from llama.cpp.

    This is not a reader check so much as a guard on the fixtures: if the dump
    test ever runs against a different checkpoint, every other assertion here
    becomes a comparison of two parsers of the wrong file.
    """
    metadata = dumped["metadata"]
    assert metadata["general.architecture"] == "qwen3"
    assert metadata["qwen3.embedding_length"] == 1024
    assert metadata["qwen3.block_count"] == 28
    assert metadata["qwen3.attention.head_count"] == 16
    assert metadata["qwen3.attention.head_count_kv"] == 8
    assert metadata["qwen3.attention.key_length"] == 128
    assert metadata["qwen3.feed_forward_length"] == 3072
    assert metadata["qwen3.rope.freq_base"] == 1_000_000.0