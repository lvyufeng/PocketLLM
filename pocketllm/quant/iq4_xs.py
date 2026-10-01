"""The IQ4_XS k-quant 4-bit format.

``IQ4_XS`` is ``IQ4_NL``'s k-quant sibling: it shares the same 16-entry
non-linear codebook and the same nibble split -- low nibble first, high nibbles
packed sixteen entries later -- but wraps eight 32-weight groups into one
256-weight super-block with its own scale hierarchy.  The scale is split the
same way the codebook is shared: a 6-bit value per group, whose low four bits
live in the ``scales_l`` plane and whose high two bits are interleaved into
``scales_h`` two bits at a time.

Because the split is shared rather than copied, this decoder calls
:func:`pocketllm.quant.iq4_nl.decode_indices` for the values and only does the
scaling itself.  That is the whole reason the two formats live next to each
other: a change to the nibble order must move both, and keeping them apart
invites a fix that silently misses one.
"""

from __future__ import annotations

import numpy as np

from .iq4_nl import decode_indices
from .k_quants import f16_to_f32

__all__ = ["IQ4_XS_BLOCK_BYTES", "dequantize_iq4_xs"]

IQ4_XS_BLOCK_BYTES = 136


def dequantize_iq4_xs(blocks: np.ndarray) -> np.ndarray:
    """Decode IQ4_XS super-blocks of 136 bytes into ``float32`` weights.

    ``blocks`` is shaped ``(..., 136)``; the result is ``(..., 256)``.  The
    layout is ``d[2] + scales_h[2] + scales_l[4] + qs[128]`` for 256 weights.
    """
    arr = np.asarray(blocks, dtype=np.uint8)
    if arr.shape[-1] != IQ4_XS_BLOCK_BYTES:
        raise ValueError(f"IQ4_XS block must be {IQ4_XS_BLOCK_BYTES} bytes, got {arr.shape[-1]}")
    lead = arr.shape[:-1]
    flat = arr.reshape(-1, IQ4_XS_BLOCK_BYTES)
    n = flat.shape[0]

    d = f16_to_f32(flat[:, 0:2])[:, None]
    scales_h = flat[:, 2:4].view("<u2").reshape(n)
    scales_l = flat[:, 4:8]
    qs = flat[:, 8:136]

    out = np.empty((n, 256), dtype=np.float32)
    for group in range(8):
        scale = ((scales_l[:, group // 2] >> (4 if group & 1 else 0)) & 0x0F).astype(np.int16)
        scale |= ((scales_h >> (2 * group)) & 0x03).astype(np.int16) << 4
        scale = (scale - 32).astype(np.float32)[:, None]
        values = decode_indices(qs[:, group * 16:(group + 1) * 16])
        out[:, group * 32:(group + 1) * 32] = d * scale * values
    return out.reshape(*lead, 256)