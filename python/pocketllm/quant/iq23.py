"""The IQ2 and IQ3 super-block formats: IQ2_XXS, IQ2_XS, IQ3_XXS.

All three are gather-and-scale decoders over the sign-expanded grids in
:mod:`pocketllm.quant.ggml_tables`.  They differ in how a 256-weight super-block
is cut into sub-blocks and where the scale comes from:

* ``IQ2_XXS`` -- eight 32-weight sub-blocks; the per-sub-block scale is the top
  nibble of a 32-bit auxiliary word, and the sign index is one of four 7-bit
  fields in that same word.
* ``IQ2_XS`` -- eight 32-weight sub-blocks; scales are four bits each in an
  eight-byte plane, and each grid index carries its own 9-bit sign index.
* ``IQ3_XXS`` -- eight 32-weight sub-blocks of 4-wide grid entries; the grid
  plane and the scale/sign plane are *separate*, not interleaved.

A wrong sign layout here produces plausible weights, so each decoder is pinned
against a hand-built block in the tests.
"""

from __future__ import annotations

import numpy as np

from .ggml_tables import iq2xxs_signed_grid, iq2xs_signed_grid, iq3xxs_signed_grid
from .k_quants import f16_to_f32

__all__ = [
    "IQ2_XXS_BLOCK_BYTES",
    "IQ2_XS_BLOCK_BYTES",
    "IQ3_XXS_BLOCK_BYTES",
    "dequantize_iq2_xxs",
    "dequantize_iq2_xs",
    "dequantize_iq3_xxs",
]

IQ2_XXS_BLOCK_BYTES = 66
IQ2_XS_BLOCK_BYTES = 74
IQ3_XXS_BLOCK_BYTES = 98


def _flat(blocks, block_bytes: int) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(blocks, dtype=np.uint8)
    if arr.shape[-1] != block_bytes:
        raise ValueError(f"block must be {block_bytes} bytes, got {arr.shape[-1]}")
    return arr.reshape(-1, block_bytes), arr.shape[:-1]


def dequantize_iq2_xxs(blocks: np.ndarray) -> np.ndarray:
    flat, lead = _flat(blocks, IQ2_XXS_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    qs = flat[:, 2:66]
    grid = iq2xxs_signed_grid()
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for sub in range(8):
        chunk = qs[:, sub * 8:(sub + 1) * 8]
        aux1 = (
            chunk[:, 4].astype(np.uint32)
            | (chunk[:, 5].astype(np.uint32) << 8)
            | (chunk[:, 6].astype(np.uint32) << 16)
            | (chunk[:, 7].astype(np.uint32) << 24)
        )
        ls = (2 * (aux1 >> 28) + 1).astype(np.float32)[:, None]
        for part in range(4):
            grid_ids = chunk[:, part].astype(np.int64)
            sign_idx = ((aux1 >> (7 * part)) & 127).astype(np.int64)
            values = grid[grid_ids, sign_idx].astype(np.float32)
            start = sub * 32 + part * 8
            out[:, start:start + 8] = 0.125 * d * ls * values
    return out.reshape(*lead, 256)


def dequantize_iq2_xs(blocks: np.ndarray) -> np.ndarray:
    flat, lead = _flat(blocks, IQ2_XS_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    qs = flat[:, 2:66]
    scales = flat[:, 66:74]
    grid = iq2xs_signed_grid()
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for sub in range(8):
        chunk = qs[:, sub * 8:(sub + 1) * 8]
        ls0 = (scales[:, sub] & 0x0F).astype(np.float32)
        ls1 = (scales[:, sub] >> 4).astype(np.float32)
        for part in range(4):
            q = chunk[:, part * 2].astype(np.uint16) | (chunk[:, part * 2 + 1].astype(np.uint16) << 8)
            grid_ids = (q & 0x01FF).astype(np.int64)
            sign_idx = (q >> 9).astype(np.int64)
            values = grid[grid_ids, sign_idx].astype(np.float32)
            start = sub * 32 + part * 8
            scale = ((ls0 if part < 2 else ls1) + 0.5)[:, None]
            out[:, start:start + 8] = 0.25 * d * scale * values
    return out.reshape(*lead, 256)


def dequantize_iq3_xxs(blocks: np.ndarray) -> np.ndarray:
    flat, lead = _flat(blocks, IQ3_XXS_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    qs = flat[:, 2:98]
    # IQ3_XXS block layout (block_iq3_xxs, 98 bytes):
    #   [0:2]   d (fp16)
    #   [2:66]  grid indices: 8 sub-blocks x 8 bytes
    #   [66:98] scales_and_signs: 8 sub-blocks x 4 bytes aux uint32
    # The two planes are separate, not interleaved.
    grid_idx = qs[:, 0:64]
    aux_bytes = qs[:, 64:96]
    grid = iq3xxs_signed_grid()
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for sub in range(8):
        qbytes = grid_idx[:, sub * 8:sub * 8 + 8]
        aux = (
            aux_bytes[:, sub * 4 + 0].astype(np.uint32)
            | (aux_bytes[:, sub * 4 + 1].astype(np.uint32) << 8)
            | (aux_bytes[:, sub * 4 + 2].astype(np.uint32) << 16)
            | (aux_bytes[:, sub * 4 + 3].astype(np.uint32) << 24)
        )
        ls = (aux >> 28).astype(np.float32)[:, None]
        for part in range(8):
            grid_ids = qbytes[:, part].astype(np.int64)
            sign_idx = ((aux >> (7 * (part // 2))) & 127).astype(np.int64)
            values = grid[grid_ids, sign_idx, part % 2 * 4:part % 2 * 4 + 4].astype(np.float32)
            start = sub * 32 + part * 4
            out[:, start:start + 4] = 0.5 * d * (ls + 0.5) * values
    return out.reshape(*lead, 256)