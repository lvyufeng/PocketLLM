"""The PQ2_0 two-bit block format.

``PQ2_0`` is the second fork-private GGML type (id 142) from
`PrismML-Eng/llama.cpp <https://github.com/PrismML-Eng/llama.cpp>`_'s ``prism``
branch, the larger of the two artifacts Ternary-Bonsai-2-27B ships.  It is **not**
ternary: the fork's own comment calls it "identical 2-bit codec to Q2_0, one fp16
scale per 128 weights", and its code set is asymmetric.

Block layout (``QK_PQ2_0 = 128`` weights, 34 bytes, 2.125 bits per weight)::

    ggml_half d       one scale for all 128 weights
    uint8_t  qs[32]   four two-bit codes per byte, 128 values, low bits first

The scale comes *first* here, unlike PTQ1_0 where it is the last two bytes.  The
two formats agree on nothing about their layout except the group size, so neither
decoder can be adapted from the other.

The code at weight ``j`` is ``(qs[j / 4] >> (2 * (j % 4))) & 0x03`` and maps to a
level as ``00 = -1, 01 = 0, 10 = +1, 11 = +2``, so the block is ``d`` times a
number in ``{-1, 0, 1, 2}`` — four levels, one of them twice the magnitude of the
others, which is what buys it 0.375 more bits per weight than PTQ1_0.

``tests/test_gguf_ternary_reader.py`` decodes rows of the released file and
compares them against the same rows of the F16 GGUF, where they agree to the bit.
"""

from __future__ import annotations

import numpy as np

QK_PQ2_0 = 128
"""Weights per block."""

PQ2_0_BLOCK_BYTES = 34
"""Bytes per block: 32 + a 2-byte fp16 scale."""

PQ2_0_CODE_LEVELS = (-1, 0, 1, 2)
"""The level each two-bit code means, indexed by the code."""

_QK_BYTES = 32
_SCALE_BYTES = 2
#: Bit offsets of the four codes inside a byte, low bits first.
_CODE_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)


def block_codes(block: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode one 34-byte PQ2_0 block into 128 codes in ``{0, 1, 2, 3}``.

    The scale is not applied; ``dequantize_blocks`` does that.  Kept separate
    because the code string is what a layout regression moves.
    """
    if len(block) != PQ2_0_BLOCK_BYTES:
        raise ValueError(f"PQ2_0 block must be {PQ2_0_BLOCK_BYTES} bytes, got {len(block)}")
    raw = np.frombuffer(memoryview(block).cast("B"), dtype=np.uint8)
    return _codes_from_bytes(raw[_SCALE_BYTES:])


def _codes_from_bytes(qs: np.ndarray) -> np.ndarray:
    """``(..., 32)`` bytes -> ``(..., 128)`` codes."""
    shifted = np.bitwise_and(qs[..., :, None] >> _CODE_SHIFTS, np.uint8(3))
    return shifted.reshape(*qs.shape[:-1], QK_PQ2_0)


def block_scale_fp16(block: bytes | bytearray | memoryview) -> np.uint16:
    """The block's scale as the raw fp16 bit pattern it is stored as."""
    if len(block) != PQ2_0_BLOCK_BYTES:
        raise ValueError(f"PQ2_0 block must be {PQ2_0_BLOCK_BYTES} bytes, got {len(block)}")
    raw = memoryview(block).cast("B")
    return np.uint16(raw[0]) | (np.uint16(raw[1]) << 8)


def dequantize_blocks(blocks: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode whole PQ2_0 blocks into ``float32`` weights.

    ``blocks`` is either a flat byte buffer whose length is a multiple of 34, or an
    array shaped ``(..., 34)``.  The result is ``float32`` shaped ``(..., 128)``.
    Every product is exact for the same reason it is for PTQ1_0: the level is a
    small integer and the scale is a finite fp16.
    """
    if isinstance(blocks, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(blocks, dtype=np.uint8)
    else:
        arr = np.asarray(blocks, dtype=np.uint8)
    if arr.ndim == 0 or arr.shape[-1] != PQ2_0_BLOCK_BYTES:
        if arr.size % PQ2_0_BLOCK_BYTES:
            raise ValueError(
                f"PQ2_0 payload of {arr.size} bytes is not a whole number of "
                f"{PQ2_0_BLOCK_BYTES}-byte blocks"
            )
        arr = arr.reshape(-1, PQ2_0_BLOCK_BYTES)
    flat = arr.reshape(-1, PQ2_0_BLOCK_BYTES)

    scales = flat[:, :_SCALE_BYTES].copy().view(np.float16).reshape(-1).astype(np.float32)
    codes = _codes_from_bytes(flat[:, _SCALE_BYTES:].copy().reshape(-1, _QK_BYTES))
    levels = np.asarray(PQ2_0_CODE_LEVELS, dtype=np.float32)[codes]
    out = levels * scales[:, None]
    return out.reshape(*arr.shape[:-1], QK_PQ2_0)


def row_blocks(row_elems: int) -> int:
    """Blocks in a row of ``row_elems`` weights; PQ2_0 cannot split a block."""
    if row_elems % QK_PQ2_0:
        raise ValueError(f"PQ2_0 rows must be a multiple of {QK_PQ2_0}, got {row_elems}")
    return row_elems // QK_PQ2_0
