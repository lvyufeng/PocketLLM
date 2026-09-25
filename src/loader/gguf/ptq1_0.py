"""The PTQ1_0 ternary block format.

``PTQ1_0`` is a fork-private GGML type (id 143) that
`PrismML-Eng/llama.cpp <https://github.com/PrismML-Eng/llama.cpp>`_ added on its
``prism`` branch for Ternary-Bonsai-2-27B.  Upstream GGML assigns nothing near
143, so a reader that does not know the type sees ``unknown_143`` and — the
failure mode worth designing against — can fall back to an F16 upcast, which
costs ten times the memory and makes a wrong kernel look right.

The geometry is upstream ``TQ1_0`` (id 34) with the scale group split: the same
base-3 trit packing, but one fp16 scale per 128 weights instead of per 256.

Block layout (``QK_PTQ1_0 = 128`` weights, 28 bytes, 1.75 bits per weight)::

    uint8_t  qs[24]   five trits per byte, 120 values
    uint8_t  qh[2]    four trits per byte, 8 values
    ggml_half d       one scale for all 128 weights, so  d = amax(|w|)

The five trits in a ``qs`` byte are packed most significant first and read back
with the decoder's ``((byte * 3**n) * 3) >> 8`` trick, which is why the stored
byte is a *ceiling* division: ``q = (q * 256 + 242) / 243``.  Arithmetic on the
byte is ``uint8_t``, so the intermediate wraps — that wrap is part of the format,
not an accident, and this module reproduces it by masking to eight bits.

Both ``qs`` and ``qh`` are walked in *stages*: a stage of width ``c`` consumes a
contiguous run of ``5 * c`` weights and writes ``c`` bytes.  ``qs`` is 24 bytes
and the stages are 32, 16 and 8 in that order, so the 32-wide stage never fits
and the whole of ``qs`` is covered by a 16-wide stage (bytes 0..15, weights
0..79) followed by an 8-wide one (bytes 16..23, weights 80..119).  ``qh`` then
carries weights 120..127, four trits per byte, with its two bytes interleaved by
parity: ``qh[0]`` holds 120, 122, 124, 126 and ``qh[1]`` holds 121, 123, 125, 127.

``tests/test_ptq1_0_layout.py`` pins all of this against blocks read out of the
released checkpoint and decoded by the fork's own reference implementation, so a
change here that looks harmless but moves a trit fails the suite.
"""

from __future__ import annotations

import numpy as np

QK_PTQ1_0 = 128
"""Weights per block."""

PTQ1_0_BLOCK_BYTES = 28
"""Bytes per block: 24 + 2 + a 2-byte fp16 scale."""

PTQ1_0_TRIT_STAGES = (32, 16, 8)
"""Stage widths, in the order the encoder and decoder walk them."""

_POW3 = (1, 3, 9, 27, 81, 243)
_QH_BYTES = 2
_SCALE_BYTES = 2


def block_trits(block: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode one 28-byte PTQ1_0 block into 128 trits in ``{-1, 0, 1}``.

    The scale is *not* applied; ``dequantize_blocks`` does that.  Kept separate
    because the trit string is the part a layout regression moves, and it is
    cheaper to compare than a float array.
    """
    if len(block) != PTQ1_0_BLOCK_BYTES:
        raise ValueError(f"PTQ1_0 block must be {PTQ1_0_BLOCK_BYTES} bytes, got {len(block)}")
    raw = memoryview(block).cast("B")
    trits = np.empty(QK_PTQ1_0, dtype=np.int8)

    n_qs = len(raw) - _QH_BYTES - _SCALE_BYTES
    elem = 0
    byte = 0
    for width in PTQ1_0_TRIT_STAGES:
        while byte + width <= n_qs:
            for shift in _POW3[:5]:
                for m in range(width):
                    # uint8_t in the reference: (byte * 3**n) is truncated to 8 bits.
                    q = (int(raw[byte + m]) * shift) & 0xFF
                    trits[elem] = ((q * 3) >> 8) - 1
                    elem += 1
            byte += width

    if elem != QK_PTQ1_0 - 4 * _QH_BYTES:
        raise ValueError(
            f"the qs stages covered {elem} weights, expected {QK_PTQ1_0 - 4 * _QH_BYTES}"
        )

    qh = raw[n_qs:n_qs + _QH_BYTES]
    for shift in _POW3[:4]:
        for h in range(_QH_BYTES):
            q = (int(qh[h]) * shift) & 0xFF
            trits[elem] = ((q * 3) >> 8) - 1
            elem += 1

    return trits


def block_scale_fp16(block: bytes | bytearray | memoryview) -> np.uint16:
    """The block's scale as the raw fp16 bit pattern it is stored as."""
    if len(block) != PTQ1_0_BLOCK_BYTES:
        raise ValueError(f"PTQ1_0 block must be {PTQ1_0_BLOCK_BYTES} bytes, got {len(block)}")
    raw = memoryview(block).cast("B")
    return np.uint16(raw[-2]) | (np.uint16(raw[-1]) << 8)


def dequantize_blocks(blocks: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode whole PTQ1_0 blocks into ``float32`` weights.

    ``blocks`` is either a flat byte buffer whose length is a multiple of 28, or
    an array shaped ``(..., 28)``.  The result is ``float32`` shaped
    ``(..., 128)``.  Every product is exact: a trit is ``-1``, ``0`` or ``1`` and
    the scale is a finite fp16, so the fp32 result reproduces the reference
    bit for bit rather than approximately.
    """
    if isinstance(blocks, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(blocks, dtype=np.uint8)
    else:
        arr = np.asarray(blocks, dtype=np.uint8)
    if arr.ndim == 0 or arr.shape[-1] != PTQ1_0_BLOCK_BYTES:
        if arr.size % PTQ1_0_BLOCK_BYTES:
            raise ValueError(
                f"PTQ1_0 payload of {arr.size} bytes is not a whole number of "
                f"{PTQ1_0_BLOCK_BYTES}-byte blocks"
            )
        arr = arr.reshape(-1, PTQ1_0_BLOCK_BYTES)
    flat = arr.reshape(-1, PTQ1_0_BLOCK_BYTES)

    scales = flat[:, -_SCALE_BYTES:].copy().view(np.float16).reshape(-1)
    trits = np.stack([block_trits(b.tobytes()) for b in flat])
    out = trits.astype(np.float32) * scales.astype(np.float32)[:, None]
    return out.reshape(*arr.shape[:-1], QK_PTQ1_0)


def row_blocks(row_elems: int) -> int:
    """Blocks in a row of ``row_elems`` weights; PTQ1_0 cannot split a block."""
    if row_elems % QK_PTQ1_0:
        raise ValueError(f"PTQ1_0 rows must be a multiple of {QK_PTQ1_0}, got {row_elems}")
    return row_elems // QK_PTQ1_0
