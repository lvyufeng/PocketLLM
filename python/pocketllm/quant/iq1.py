"""The IQ1 codebook and the IQ1_M block decoder.

IQ1 is the narrowest end of the width ladder: three levels per weight, encoded
as an index into a shared 2048-entry codebook of eight ternary values.  The
codebook is not arithmetic and cannot be derived -- it is the format.  It is
read from the vendored GGML header (`ggml_tables.iq1s_grid`) rather than stored
in Python, so there is one statement of its bytes across the tree.

Two things make IQ1_M more than "look up eight values":

* The scale is split.  A shared fp16 super-scale `d` is hidden in the four high
  nibbles of four uint16 words, and each word's low twenty-four bits carry four
  3-bit local scales.  The value scale is ``d * (2 * local + 1)``.
* The codebook is signed *per entry*, not per bit: the index's high bit chooses
  a positive or negative quartile, and the block's `qh` bytes carry the level
  offset (``+/- 0.125``) applied after the lookup.

The result of both quirks is that an IQ1_M row is a small integer arithmetic
fused with a gather.  It is cheap to get subtly wrong and cheap to check against
the header, which is why the lookup is table-driven here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .ggml_tables import iq1s_grid

__all__ = ["QK_IQ1", "IQ1_M_BLOCK_BYTES", "iq1_grid_i8", "dequantize_blocks"]

QK_IQ1 = 256
"""Weights per IQ1 super-block."""

IQ1_M_BLOCK_BYTES = 56
"""Bytes per IQ1_M super-block: 32 of indices, 16 of high bits, 8 of scales."""


@lru_cache(maxsize=1)
def iq1_grid_i8() -> np.ndarray:
    """The codebook as ``(2048, 8)`` int8 in ``{-1, 0, 1}``.

    The header packs each entry into a uint64 whose eight bytes are the eight
    ternary values, so expanding is a byte reinterpretation and not a decode.
    """
    packed = iq1s_grid().astype("<u8").tobytes()
    grid = np.frombuffer(packed, dtype=np.int8).reshape(2048, 8).copy()
    grid.flags.writeable = False
    return grid


def dequantize_blocks(blocks: np.ndarray) -> np.ndarray:
    """Decode IQ1_M blocks of 56 bytes into ``float32`` weights.

    ``blocks`` is shaped ``(..., 56)``; the result is ``float32`` shaped
    ``(..., 256)``.  The formula mirrors llama.cpp gguf-py
    ``IQ1_M.dequantize_blocks``.
    """
    arr = np.asarray(blocks, dtype=np.uint8)
    if arr.shape[-1] != IQ1_M_BLOCK_BYTES:
        raise ValueError(
            f"IQ1_M block must be {IQ1_M_BLOCK_BYTES} bytes, got {arr.shape[-1]}"
        )
    lead = arr.shape[:-1]
    flat = arr.reshape(-1, IQ1_M_BLOCK_BYTES)
    n = flat.shape[0]

    qs = flat[:, :32]
    qh = flat[:, 32:48]
    scales = flat[:, 48:56].view(np.uint16)

    # Shared super-scale: the top nibble of each of the four scale words, most
    # significant first, spells one fp16.
    parts = scales.reshape(n, 4, 1) & np.uint16(0xF000)
    parts = parts >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape(1, 4, 1)
    d_words = parts[:, 0, 0] | parts[:, 1, 0] | parts[:, 2, 0] | parts[:, 3, 0]
    d = d_words.astype("<u2").view("<f2").astype(np.float32).reshape(n, 1)

    # Low twelve bits hold four packed 3-bit local scales per word.
    local = scales.reshape(n, 4, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape(1, 1, 4)
    local = (local & 0x07).reshape(n, -1)
    dl = (d * (2 * local + 1)).reshape(n, -1, 2, 1, 1)

    # qh supplies three high index bits per value and one level-sign bit.
    qh_parts = qh.reshape(n, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)
    qidx = qs.astype(np.uint16) | ((qh_parts & 0x07).astype(np.uint16) << 8).reshape(n, -1)

    delta = np.where((qh_parts & 0x08) == 0, np.float32(0.125), np.float32(-0.125))
    delta = delta.reshape(n, -1, 2, 2, 1)

    table = iq1_grid_i8()
    grid = table[qidx.reshape(-1)].astype(np.float32).reshape(n, -1, 2, 2, 8)
    return (dl * (grid + delta)).reshape(*lead, QK_IQ1)