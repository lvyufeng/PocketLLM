"""Writing a GGUF, for the tests that need a checkpoint nobody shipped.

Two things in this tree need a checkpoint and cannot have one: the C engine's
Qwen3 forward wants a *file* rather than a dict of tensors, and the Python graph
wants the same file so the two can be compared on identical weights.  A 1.5 GB
f16 Qwen3-0.6B answers neither -- it is too large to commit, it does not exist in
CI, and it cannot be built with hand-checked geometry.

So this writes one.  A tiny Qwen3 -- two layers, four heads, an eight-wide vector
-- laid out exactly as llama.cpp's converter lays out a real one, down to the
metadata key spellings and the ``[columns, rows]`` dimension order.  Both readers
then parse it with the code they use on a real checkpoint, which is the point:
the loader is not stubbed out for the test, it is exercised, and a failure in
either implementation's *reading* shows up here rather than only on the 1.5 GB
file that one machine has.

It also writes a **byte-level BPE vocabulary**, which is not decoration.  The C
engine's ``Session::open`` builds a ``Tokenizer`` unconditionally and that
constructor refuses a file without a ``tokenizer.ggml.tokens`` array, so a
synthetic checkpoint with weights and no vocabulary does not open at all.  The
vocabulary is small and its pieces are byte pairs rather than words, so the
prompt cannot smuggle a tokenizer disagreement into a numerics comparison.

**This writes files, not checkpoints for distribution.**  The writer is faithful
to the format and says nothing about whether a model is good; it exists so a test
can build the input its subject needs.

Everything is little-endian, as GGUF is.
"""

from __future__ import annotations

import struct
from typing import Any, Mapping

import numpy as np

__all__ = [
    "DEFAULT_ALIGNMENT",
    "DEFAULT_MERGES",
    "GGML_F16",
    "GGML_F32",
    "GGUF_ARRAY",
    "GGUF_BOOL",
    "GGUF_FLOAT32",
    "GGUF_INT32",
    "GGUF_STRING",
    "GGUF_UINT32",
    "QWEN2_PRE_TOKENIZER",
    "QWEN3_FLOAT_KEYS",
    "QWEN3_INT_KEYS",
    "TOKENIZER_MODEL",
    "byte_alphabet",
    "qwen2_tokenizer_metadata",
    "qwen2_vocabulary",
    "qwen3_metadata",
    "write_gguf",
]

#: The metadata value types this writer emits.  Named rather than numbered at the
#: call sites, because `4` is not a uint32 to a reader of the calling code.
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9

#: GGML tensor types.  Only the dense ones: a quantized checkpoint is written by
#: quantizing the dense one, which is a different test's job.
GGML_F32 = 0
GGML_F16 = 1

_ITEM_SIZES = {GGUF_UINT32: 4, GGUF_INT32: 4, GGUF_FLOAT32: 4, GGUF_BOOL: 1}
_ITEM_PACKS = {GGUF_UINT32: "<I", GGUF_INT32: "<i", GGUF_FLOAT32: "<f", GGUF_BOOL: "<?"}

#: GGUF's own default.  A tensor's offset is relative to the aligned start of the
#: data section, and every offset must be a multiple of this.
DEFAULT_ALIGNMENT = 32


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _metadata_value(value_type: int, value: Any) -> bytes:
    if value_type == GGUF_STRING:
        return _string(str(value))
    if value_type == GGUF_ARRAY:
        item_type, items = value
        out = struct.pack("<I", item_type) + struct.pack("<Q", len(items))
        if item_type == GGUF_STRING:
            return out + b"".join(_string(str(item)) for item in items)
        pack = _ITEM_PACKS[item_type]
        return out + b"".join(struct.pack(pack, item) for item in items)
    return struct.pack(_ITEM_PACKS[value_type], value)


def write_gguf(
    path: str,
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, tuple[int, Any]] | None = None,
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> str:
    """Write ``tensors`` and ``metadata`` to ``path`` and return it.

    ``metadata`` maps a key to ``(type, value)``; a caller that has only the
    common scalar types is better served by :func:`qwen3_metadata`, which spells
    the keys the way the C engine reads them.

    A tensor's array is written in C order with its **last** axis first, which is
    GGUF's convention: a matrix stored as ``(rows, columns)`` in numpy becomes
    ``dimensions = (columns, rows)`` on disk, because ggml addresses the row axis
    (the one a dot product reduces) first.  Getting this backwards produces a file
    both readers accept and every kernel transposes, so it is done here once.
    """
    metadata = dict(metadata or {})
    metadata.setdefault("general.alignment", (GGUF_UINT32, int(alignment)))
    align = int(alignment)

    entries: list[tuple[str, int, Any]] = []
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        entries.append((name, GGML_F32 if array.dtype == np.float32 else GGML_F16, array))

    # Layout: header, metadata, tensor records, pad to `alignment`, then the data.
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(entries)) + struct.pack("<Q", len(metadata))
    meta = b""
    for key, (value_type, value) in metadata.items():
        meta += _string(key) + struct.pack("<I", value_type) + _metadata_value(value_type, value)

    records = b""
    offset = 0
    for name, type_id, array in entries:
        dimensions = tuple(int(dim) for dim in array.shape[::-1])
        records += _string(name) + struct.pack("<I", len(dimensions))
        records += b"".join(struct.pack("<Q", dim) for dim in dimensions)
        records += struct.pack("<I", type_id) + struct.pack("<Q", offset)
        offset += array.nbytes
        offset = _align_up(offset, align)

    prefix = header + meta + records
    padding = _align_up(len(prefix), align) - len(prefix)

    with open(path, "wb") as handle:
        handle.write(prefix)
        handle.write(b"\x00" * padding)
        for _name, _type_id, array in entries:
            payload = array.tobytes()
            handle.write(payload)
            handle.write(b"\x00" * (_align_up(len(payload), align) - len(payload)))
    return path


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


#: The hyperparameter keys the C engine reads, in the order it reads them.
#:
#: The spellings are not negotiable -- ``Qwen3Model::load`` looks up exactly these
#: -- and they are collected here rather than inline at a call site so a writer
#: and a reader that disagree disagree in one place.
QWEN3_INT_KEYS = (
    "embedding_length",
    "block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "feed_forward_length",
    "vocab_size",
    "context_length",
)
QWEN3_FLOAT_KEYS = ("attention.layer_norm_rms_epsilon", "rope.freq_base")

#: The pre-tokenizer ``Tokenizer::Tokenizer`` insists on, and the BPE model it
#: insists on.  Both are refusals by name rather than fallbacks, so a synthetic
#: checkpoint that spells either differently does not open at all.
QWEN2_PRE_TOKENIZER = "qwen2"
TOKENIZER_MODEL = "gpt2"

#: How many merges :func:`qwen2_vocabulary` generates.  The vocabulary is
#: ``256 + merges`` tokens long, which is the size the geometry has to agree
#: with: ``qwen3.vocab_size`` and ``len(tokenizer.ggml.tokens)`` are two
#: statements about the same number and a checkpoint where they differ is one
#: llama.cpp refuses.
DEFAULT_MERGES = 256

#: The gap between a merge's two halves.  Half the alphabet, so for any 256-byte
#: alphabet one half of every pair is below 128 and the other is at or above it --
#: no merge can ever fire on ASCII, which is what keeps the oracle's prompt from
#: being a tokenizer test.  See :func:`qwen2_vocabulary`.
_MERGE_DISTANCE = 128


def byte_alphabet() -> tuple[str, ...]:
    """GPT-2's byte-to-codepoint assignment, as a table indexed by byte.

    Printable ASCII and a span of Latin-1 map to themselves; every other byte --
    the 32 control codes, the space, the DEL, and the ten Latin-1 gaps -- is
    given a codepoint starting at 256 in increasing byte order.  Written as the
    loop rather than transcribed from a table for the reason
    ``src/tokenizer/unicode.cpp`` gives: a table copied by hand is off by one
    somewhere, and the symptom is a vocabulary mismatch on a handful of bytes.

    This is the *same* assignment the C tokenizer builds, and it has to be: a
    byte-level BPE's pieces are spelled in this alphabet, so a writer that
    disagreed about one byte would write a vocabulary whose ``é`` is the
    tokenizer's ``Ã©``.
    """
    table: list[str] = []
    next_codepoint = 256
    for byte in range(256):
        printable_ascii = 33 <= byte <= 126
        printable_latin = 161 <= byte <= 172 or 174 <= byte <= 255
        if printable_ascii or printable_latin:
            table.append(chr(byte))
        else:
            table.append(chr(next_codepoint))
            next_codepoint += 1
    return tuple(table)


def qwen2_vocabulary(*, merges: int = DEFAULT_MERGES) -> tuple[list[str], list[str]]:
    """A byte-level BPE vocabulary: every byte, then ``merges`` byte pairs.

    Returns ``(tokens, merges)`` -- the two arrays ``tokenizer.ggml.tokens`` and
    ``tokenizer.ggml.merges`` carry -- and the pair is the whole vocabulary.  It
    is a *small* one: the 256 single bytes, then 256 two-byte pieces, in byte
    order, which is enough to exercise the merge path and the byte fallback
    without inviting a fixture nobody can read.

    **The tokens are byte pairs, not words.**  That is what makes the fixture
    safe to compare against another tokenizer: the C tokenizer's pre-tokenizer
    and its merge ranking are both exercised, but no piece depends on a locale,
    a Unicode property or a merge table someone would have to trust.

    **No two ASCII bytes can merge.**  Each pair joins a byte with the one
    :data:`_MERGE_DISTANCE` away, which for a 256-entry alphabet always puts one
    half below 128 and the other at or above it -- so an ASCII-only prompt has no
    merge to take and every implementation tokenizes it to its bytes.  That is a
    property the oracle test relies on, not an accident: it is what keeps a
    numerics comparison from also being a tokenizer comparison.

    ``merges`` must stay at or below 256 -- the number of distinct first bytes --
    and the constructor refuses more rather than repeating a pair, because a
    duplicated merge is a rank tie rather than a vocabulary.
    """
    alphabet = byte_alphabet()
    n = len(alphabet)
    if not 0 <= merges <= n:
        raise ValueError(
            f"merges must be between 0 and {n} (one per first byte of the alphabet), got {merges}: "
            "a second pass over the first bytes would repeat a pair, and a duplicated merge is a "
            "rank tie, not a vocabulary"
        )
    pairs = [alphabet[left] + alphabet[(left + _MERGE_DISTANCE) % n] for left in range(merges)]
    return list(alphabet) + pairs, [f"{pair[0]} {pair[1]}" for pair in pairs]


def qwen2_tokenizer_metadata(vocab_size: int, *, merges: int = DEFAULT_MERGES) -> dict[str, tuple[int, Any]]:
    """The tokenizer keys the C engine needs before it will open a checkpoint.

    ``Session::open`` builds a ``Tokenizer`` unconditionally, and
    ``Tokenizer::Tokenizer`` refuses a file with no ``tokenizer.ggml.tokens``
    array -- so a synthetic checkpoint without a vocabulary is not merely
    untokenizable, it does not open, and the forward pass never runs.  These keys
    are therefore part of "loadable", not decoration.

    No ``tokenizer.ggml.token_type`` array is written, and that is a statement
    rather than an omission: the reader's own comment calls the absent array
    "a vocabulary with no specials at all -- and that is the correct reading,
    not a fallback".  A vocabulary whose every token is NORMAL is exactly what
    this is.

    The two ``add_*_token`` flags are **bools and not uint32s**, which is a
    distinction with a real failure behind it: both this tree's readers accept
    either, but llama.cpp's vocabulary loader refuses a u32 there with "has
    wrong type u32 but expected type bool".  A synthetic checkpoint is worth
    little if it loads everywhere except the one implementation that gives the
    model its second opinion.
    """
    tokens, merge_strings = qwen2_vocabulary(merges=merges)
    if len(tokens) != int(vocab_size):
        raise ValueError(
            f"the byte-level vocabulary is {len(tokens)} tokens (256 bytes + {merges} merges) "
            f"but the geometry says vocab={int(vocab_size)}; the two are the same number to "
            "every reader that opens this file, so they are checked here"
        )
    return {
        "tokenizer.ggml.model": (GGUF_STRING, TOKENIZER_MODEL),
        "tokenizer.ggml.pre": (GGUF_STRING, QWEN2_PRE_TOKENIZER),
        "tokenizer.ggml.tokens": (GGUF_ARRAY, (GGUF_STRING, tokens)),
        "tokenizer.ggml.merges": (GGUF_ARRAY, (GGUF_STRING, merge_strings)),
        "tokenizer.ggml.bos_token_id": (GGUF_UINT32, 0),
        "tokenizer.ggml.eos_token_id": (GGUF_UINT32, 1),
        "tokenizer.ggml.add_bos_token": (GGUF_BOOL, False),
        "tokenizer.ggml.add_eos_token": (GGUF_BOOL, False),
    }


def qwen3_metadata(
    *,
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    ff: int,
    vocab: int,
    context: int,
    rms_eps: float,
    rope_theta: float,
) -> dict[str, tuple[int, Any]]:
    """The metadata a minimal-but-loadable Qwen3 checkpoint needs.

    The keys are namespaced by ``general.architecture``, and the C engine derives
    that prefix from the file rather than assuming one -- so a file that declares
    ``qwen3`` cannot be read with another model's hyperparameters, and a test that
    got the prefix wrong fails in the loader rather than silently running.
    """
    values: dict[str, Any] = {
        "embedding_length": hidden,
        "block_count": layers,
        "attention.head_count": heads,
        "attention.head_count_kv": kv_heads,
        "attention.key_length": head_dim,
        "feed_forward_length": ff,
        "vocab_size": vocab,
        "context_length": context,
    }
    out: dict[str, tuple[int, Any]] = {
        "general.architecture": (GGUF_STRING, "qwen3"),
        "general.name": (GGUF_STRING, "synthetic-qwen3"),
    }
    for key in QWEN3_INT_KEYS:
        out[f"qwen3.{key}"] = (GGUF_UINT32, int(values[key]))
    floats = {
        "attention.layer_norm_rms_epsilon": rms_eps,
        "rope.freq_base": rope_theta,
    }
    for key in QWEN3_FLOAT_KEYS:
        out[f"qwen3.{key}"] = (GGUF_FLOAT32, float(floats[key]))
    return out