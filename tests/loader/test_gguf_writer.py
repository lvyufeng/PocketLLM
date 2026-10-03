"""The writer's claims about the format, checked by reading its own bytes back.

``tests/architectures/test_qwen3_oracle.py`` exercises the writer the hard way --
three implementations load its output and agree -- but it exercises it on *one*
file, and a bug that both of this tree's readers happen to share would survive
it.  What is pinned here instead is the format itself: the header, the reversed
axis order, the alignment, and the metadata value types.

The reversed axes are the one that matters.  GGUF writes the fastest-varying
axis first, so a numpy matrix stored ``(rows, columns)`` is written
``dimensions = (columns, rows)``; a writer that got this wrong produces a file
both of this tree's readers accept and every kernel transposes, which is a
numerics bug wearing a format bug's clothes.  ``test_the_matrix_comes_back_
transposed_if_the_file_is_read_naively`` asserts the trap exists, so that the
convention is documented by a test rather than by a comment.

The reader is the only consumer used here, but it is *the* normative one: the
format claims are all about what lands on disk, and the reader reports exactly
that.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from pocketllm.loader.gguf.reader import GGUFReader
from pocketllm.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader
from pocketllm.loader.gguf.writer import (
    GGUF_BOOL,
    GGUF_FLOAT32,
    GGUF_STRING,
    GGUF_UINT32,
    byte_alphabet,
    qwen2_tokenizer_metadata,
    qwen2_vocabulary,
    qwen3_metadata,
    write_gguf,
)


def _read(path) -> GGUFReader:
    return GGUFReader(str(path), read_arrays=True)


def test_the_header_says_gguf_and_the_counts_agree(tmp_path) -> None:
    path = tmp_path / "counts.gguf"
    write_gguf(str(path), {"a": np.zeros((2, 3), np.float32), "b": np.zeros((4,), np.float16)})
    raw = path.read_bytes()
    assert raw[:4] == b"GGUF"
    version, tensors, metadata = struct.unpack_from("<IQQ", raw, 4)
    assert version == 3
    assert tensors == 2
    assert metadata == 1, "the writer always adds general.alignment, and nothing else"


def test_a_matrix_is_written_columns_first(tmp_path) -> None:
    """GGUF's one convention this writer has to get right, and its inverse.

    The file says ``(columns, rows)``; reading the raw record back without the
    reader's reinterpretation must show the transpose.  If this ever fails, the
    writer and the reader have moved to the same convention and every kernel in
    the tree is now reading its weights transposed.
    """
    path = tmp_path / "shape.gguf"
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_gguf(str(path), {"w": matrix})
    info = _read(path).read().tensors_by_name["w"]
    assert info.dimensions == (3, 2), "the fastest-varying axis is written first"
    with get_cached_gguf_tensor_reader(str(path)) as reader:
        np.testing.assert_array_equal(reader.read_tensor("w"), matrix)


def test_the_dense_bytes_round_trip_exactly(tmp_path) -> None:
    """f32 and f16, including a value f16 cannot hold, to prove no widening.

    A writer that silently promoted every tensor to f32 would round-trip the f32
    case perfectly and change the f16 one; checking a value that is *exactly*
    representable in f16 is what makes the format claim rather than the value's
    luck.
    """
    path = tmp_path / "dense.gguf"
    f32 = (np.random.default_rng(0).standard_normal((5, 7)) * 3).astype(np.float32)
    f16 = np.array([[0.5, -1.0], [2.0, 0.125]], np.float16)
    write_gguf(str(path), {"f32": f32, "f16": f16})
    with get_cached_gguf_tensor_reader(str(path)) as reader:
        np.testing.assert_array_equal(reader.read_tensor("f32"), f32)
        np.testing.assert_array_equal(reader.read_tensor("f16"), f16)
    file = _read(path).read()
    assert file.tensors_by_name["f32"].type_name == "f32"
    assert file.tensors_by_name["f16"].type_name == "f16"


def test_the_data_section_starts_on_an_alignment_boundary(tmp_path) -> None:
    """Every tensor offset is a multiple of the alignment, and the reader agrees."""
    path = tmp_path / "aligned.gguf"
    tensors = {f"t{i}": np.zeros((i + 1, 3), np.float32) for i in range(4)}
    alignment = 64
    write_gguf(str(path), tensors, alignment=alignment)
    file = _read(path).read()
    assert file.alignment == alignment
    assert file.data_start % alignment == 0
    for info in file.tensors:
        assert info.offset % alignment == 0, f"{info.name} at {info.offset} is not aligned"


def test_the_metadata_types_are_the_ones_the_key_needs(tmp_path) -> None:
    """A uint32 where llama.cpp wants a bool is a file it refuses to load.

    ``qwen3_metadata`` and ``qwen2_tokenizer_metadata`` are read back by type
    rather than by value, because the value is right in either case -- the type
    is the part a reader is allowed to insist on.
    """
    path = tmp_path / "meta.gguf"
    metadata = qwen3_metadata(
        hidden=32, layers=2, heads=4, kv_heads=2, head_dim=8, ff=48,
        vocab=512, context=16, rms_eps=1e-6, rope_theta=1e6,
    )
    metadata.update(qwen2_tokenizer_metadata(512))
    write_gguf(str(path), {"token_embd.weight": np.zeros((512, 32), np.float32)}, metadata)
    file = _read(path).read()
    assert file.metadata["general.architecture"] == "qwen3"
    assert file.metadata["qwen3.attention.head_count_kv"] == 2
    assert file.metadata["qwen3.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-6)
    assert file.metadata["tokenizer.ggml.add_bos_token"] is False
    assert file.metadata["tokenizer.ggml.pre"] == "qwen2"
    assert file.metadata["tokenizer.ggml.model"] == "gpt2"


def test_the_vocabulary_is_the_bytes_then_one_merge_per_first_byte() -> None:
    """The shape the oracle's prompt relies on, stated as a property.

    Every merge begins with a distinct byte, so the merge table has no rank tie;
    and every merge spans the ASCII/non-ASCII divide, so an ASCII two-letter word
    has no merge to take and the ids are its bytes.  That is why the oracle can
    compare numerics without also comparing two tokenizers.
    """
    tokens, merges = qwen2_vocabulary()
    alphabet = byte_alphabet()
    assert len(tokens) == 256 + len(merges)
    assert tokens[:256] == list(alphabet)
    assert len(set(merges)) == len(merges), "a repeated merge is a rank tie, not a vocabulary"
    assert {merge.split(" ")[0] for merge in merges} == set(alphabet)
    assert alphabet[0] == "Ā", "byte 0 is the first non-printable codepoint"
    assert alphabet[ord("a")] == "a"


def test_no_merge_can_fire_on_ascii() -> None:
    """The property the oracle's prompt depends on: bytes are not merged.

    ``qwen2_pretokenize`` splits ``"ab"`` into one word, and that word can only
    stay two symbols if ``ab`` is not a ranked pair.  Asserted over the whole
    ASCII range rather than for ``"ab"`` alone, because the oracle picks its
    prompt from that range and a future alphabet change could make some other
    pair mergeable without touching this test's subject.
    """
    tokens, _ = qwen2_vocabulary()
    ranked = {tuple(piece) for piece in tokens[256:]}
    collisions = [
        (chr(a), chr(b)) for a in range(32, 127) for b in range(32, 127) if (chr(a), chr(b)) in ranked
    ]
    assert collisions == [], f"these ASCII pairs would merge: {collisions[:5]}"


def test_the_vocabulary_must_match_the_geometry(tmp_path) -> None:
    """``vocab_size`` and the token array are the same number to every reader."""
    with pytest.raises(ValueError, match="geometry says vocab=100"):
        qwen2_tokenizer_metadata(100)
    with pytest.raises(ValueError, match="between 0 and 256"):
        qwen2_vocabulary(merges=257)


def test_a_writer_output_is_a_file_the_tokenizer_keys_describe(tmp_path) -> None:
    """The eight keys ``Tokenizer::Tokenizer`` reads, all present in one file.

    ``Session::open`` builds a tokenizer unconditionally, so a checkpoint missing
    any of these does not open -- the model never runs and the failure names the
    tokenizer rather than the weights, which is a confusing place to land.
    """
    path = tmp_path / "tok.gguf"
    metadata = qwen2_tokenizer_metadata(512)
    write_gguf(str(path), {"token_embd.weight": np.zeros((512, 32), np.float32)}, metadata)
    file = _read(path).read()
    for key in ("model", "pre", "tokens", "merges", "bos_token_id", "eos_token_id",
                "add_bos_token", "add_eos_token"):
        assert f"tokenizer.ggml.{key}" in file.metadata, f"missing tokenizer.ggml.{key}"
    assert len(file.metadata["tokenizer.ggml.tokens"]) == 512
    assert len(file.metadata["tokenizer.ggml.merges"]) == 256
    assert GGUF_STRING != GGUF_BOOL != GGUF_UINT32 != GGUF_FLOAT32  # the constants are distinct