"""The k-quant super-block formats: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0.

These are pure arithmetic: no codebook, no sign table, just bit-fields read out
of a 256-weight super-block and scaled by an fp16 pair.  They are grouped here
because they share that machinery -- :func:`get_scale_min_k4` is used by both
Q4_K and Q5_K, :func:`f16_to_f32` by all of them -- and because a backend that
has one k-quant usually has the whole family.

Each ``dequantize_*`` takes blocks shaped ``(..., block_bytes)`` and returns
``float32`` shaped ``(..., 256)`` (``(..., 32)`` for Q8_0, whose block is
QK8_0 = 32 weights wide).  A partial final row is handled by the *caller*
slicing to the true element count; these functions decode whole blocks, because
a block that is not whole is not addressable.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QK_K",
    "QK8_0",
    "Q2_K_BLOCK_BYTES",
    "Q3_K_BLOCK_BYTES",
    "Q4_K_BLOCK_BYTES",
    "Q5_K_BLOCK_BYTES",
    "Q6_K_BLOCK_BYTES",
    "Q8_0_BLOCK_BYTES",
    "f16_to_f32",
    "get_scale_min_k4",
    "dequantize_q2_k",
    "dequantize_q3_k",
    "dequantize_q4_k",
    "dequantize_q5_k",
    "dequantize_q6_k",
    "dequantize_q8_0",
]

QK_K = 256
"""Weights per k-quant super-block."""

QK8_0 = 32
"""Weights per Q8_0 block."""

Q2_K_BLOCK_BYTES = 84
Q3_K_BLOCK_BYTES = 110
Q4_K_BLOCK_BYTES = 144
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
Q8_0_BLOCK_BYTES = 34


def f16_to_f32(data: np.ndarray) -> np.ndarray:
    """Reinterpret the last axis of an even-width byte array as fp16, widened.

    The bytes are coerced to ``uint8`` first: a caller that hands in a wider
    integer array gets the byte reinterpretation it meant, rather than a view
    over 8-byte elements that silently reads the wrong thing.  The final
    two-byte axis is consumed: an input of shape ``(R, 2)`` returns ``(R,)``.
    """
    raw = np.ascontiguousarray(data).view(np.uint8)
    if raw.shape[-1] % 2:
        raise ValueError(f"an fp16 field needs an even byte width, got {raw.shape[-1]}")
    wide = raw.reshape(*raw.shape[:-1], raw.shape[-1] // 2, 2).view("<f2")
    return wide.reshape(*raw.shape[:-1]).astype(np.float32)


def get_scale_min_k4(scales: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode a GGML k-quant 6-bit scale/min pair for Q4_K/Q5_K.

    Mirrors llama.cpp/ggml ``get_scale_min_k4()`` exactly.  ``scales`` has a
    trailing dimension of 12; the four pairs above index 3 share their bytes
    with the high bits of the four below, which is why the second branch reads
    both ``idx + 4`` and ``idx - 4``.
    """
    if idx < 4:
        return scales[..., idx] & 63, scales[..., idx + 4] & 63
    return (
        (scales[..., idx + 4] & 0x0F) | ((scales[..., idx - 4] >> 6) << 4),
        (scales[..., idx + 4] >> 4) | ((scales[..., idx] >> 6) << 4),
    )


def _flat(blocks, block_bytes: int) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(blocks, dtype=np.uint8)
    if arr.shape[-1] != block_bytes:
        raise ValueError(f"block must be {block_bytes} bytes, got {arr.shape[-1]}")
    return arr.reshape(-1, block_bytes), arr.shape[:-1]


def dequantize_q2_k(blocks: np.ndarray) -> np.ndarray:
    """Q2_K: per-16-group 4-bit scale/min plus 2-bit quads."""
    flat, lead = _flat(blocks, Q2_K_BLOCK_BYTES)
    scales = flat[:, :16]
    qs = flat[:, 16:80]
    d = f16_to_f32(flat[:, 80:82])[:, None]
    dmin = f16_to_f32(flat[:, 82:84])[:, None]
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for group in range(16):
        half_block = group // 8
        group_in_half = group % 8
        shift = (group_in_half // 2) * 2
        byte_start = half_block * 32 + (group_in_half % 2) * 16
        q = ((qs[:, byte_start:byte_start + 16] >> shift) & 0x03).astype(np.float32)
        scale = (scales[:, group] & 0x0F).astype(np.float32)[:, None]
        minv = (scales[:, group] >> 4).astype(np.float32)[:, None]
        out[:, group * 16:(group + 1) * 16] = d * scale * q - dmin * minv
    return out.reshape(*lead, 256)


def dequantize_q3_k(blocks: np.ndarray) -> np.ndarray:
    """Q3_K: 2-bit quads whose high bit lives in a separate mask plane."""
    flat, lead = _flat(blocks, Q3_K_BLOCK_BYTES)
    hmask = flat[:, 0:32]
    qs = flat[:, 32:96]
    scales = flat[:, 96:108]
    d = f16_to_f32(flat[:, 108:110])[:, None]
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for group in range(16):
        if group < 8:
            scale = (scales[:, group] & 0x0F).astype(np.int16)
        else:
            scale = (scales[:, group - 8] >> 4).astype(np.int16)
        scale = (scale - 8).astype(np.float32)[:, None]
        qbytes = qs[:, group * 4:(group + 1) * 4]
        qlow = np.empty((flat.shape[0], 16), dtype=np.uint8)
        qlow[:, 0:4] = qbytes & 0x03
        qlow[:, 4:8] = (qbytes >> 2) & 0x03
        qlow[:, 8:12] = (qbytes >> 4) & 0x03
        qlow[:, 12:16] = (qbytes >> 6) & 0x03
        bits = (
            (hmask[:, group * 2:group * 2 + 2][:, :, None] >> np.arange(8, dtype=np.uint8)) & 1
        ).reshape(flat.shape[0], 16)
        q = qlow.astype(np.int16) - np.where(bits == 0, 4, 0).astype(np.int16)
        out[:, group * 16:(group + 1) * 16] = d * scale * q.astype(np.float32)
    return out.reshape(*lead, 256)


def dequantize_q4_k(blocks: np.ndarray) -> np.ndarray:
    """Q4_K: a 4-bit low nibble set with a 6-bit scale/min per 32-group."""
    flat, lead = _flat(blocks, Q4_K_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    dmin = f16_to_f32(flat[:, 2:4])[:, None]
    scales = flat[:, 4:16]
    qs = flat[:, 16:144]
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    for pair in range(4):
        q = qs[:, pair * 32:(pair + 1) * 32]
        sc, mn = get_scale_min_k4(scales, pair * 2)
        out[:, pair * 64:pair * 64 + 32] = (
            d * sc.astype(np.float32)[:, None] * (q & 0x0F).astype(np.float32)
            - dmin * mn.astype(np.float32)[:, None]
        )
        sc, mn = get_scale_min_k4(scales, pair * 2 + 1)
        out[:, pair * 64 + 32:pair * 64 + 64] = (
            d * sc.astype(np.float32)[:, None] * (q >> 4).astype(np.float32)
            - dmin * mn.astype(np.float32)[:, None]
        )
    return out.reshape(*lead, 256)


def dequantize_q5_k(blocks: np.ndarray) -> np.ndarray:
    """Q5_K: Q4_K plus one high bit per weight, packed in a separate plane."""
    flat, lead = _flat(blocks, Q5_K_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    dmin = f16_to_f32(flat[:, 2:4])[:, None]
    scales = flat[:, 4:16]
    qh = flat[:, 16:48]
    qs = flat[:, 48:176]
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    u1 = 1
    u2 = 2
    for pair in range(4):
        q = qs[:, pair * 32:(pair + 1) * 32]
        high = qh[:, :32]
        sc, mn = get_scale_min_k4(scales, pair * 2)
        q_low = (q & 0x0F).astype(np.float32) + np.where((high & u1) != 0, 16.0, 0.0).astype(np.float32)
        out[:, pair * 64:pair * 64 + 32] = (
            d * sc.astype(np.float32)[:, None] * q_low - dmin * mn.astype(np.float32)[:, None]
        )
        sc, mn = get_scale_min_k4(scales, pair * 2 + 1)
        q_high = (q >> 4).astype(np.float32) + np.where((high & u2) != 0, 16.0, 0.0).astype(np.float32)
        out[:, pair * 64 + 32:pair * 64 + 64] = (
            d * sc.astype(np.float32)[:, None] * q_high - dmin * mn.astype(np.float32)[:, None]
        )
        u1 <<= 2
        u2 <<= 2
    return out.reshape(*lead, 256)


def dequantize_q6_k(blocks: np.ndarray) -> np.ndarray:
    """Q6_K: 6-bit relative values in two 128-wide halves per super-block."""
    flat, lead = _flat(blocks, Q6_K_BLOCK_BYTES)
    ql = flat[:, 0:128]
    qh = flat[:, 128:192]
    scales = flat[:, 192:208].view(np.int8).astype(np.float32)
    d = f16_to_f32(flat[:, 208:210])[:, None]
    out = np.empty((flat.shape[0], 256), dtype=np.float32)
    # Within a half, l in 0..31 gives is = l // 16, and the four sub-lanes use
    # scales sc[is + 0], sc[is + 2], sc[is + 4], sc[is + 6].
    is_idx = np.concatenate([np.zeros(16, dtype=np.int64), np.ones(16, dtype=np.int64)])
    for half in range(2):
        base = half * 128
        ql_h = ql[:, half * 64:(half + 1) * 64]
        qh_h = qh[:, half * 32:(half + 1) * 32]
        sc_h = scales[:, half * 8:(half + 1) * 8]
        ql_l = ql_h[:, 0:32].astype(np.uint8)
        ql_l32 = ql_h[:, 32:64].astype(np.uint8)
        qh_l = qh_h[:, 0:32].astype(np.uint8)
        q1 = ((ql_l & 0x0F) | (((qh_l >> 0) & 0x03) << 4)).astype(np.int16) - 32
        q2 = ((ql_l32 & 0x0F) | (((qh_l >> 2) & 0x03) << 4)).astype(np.int16) - 32
        q3 = ((ql_l >> 4) | (((qh_l >> 4) & 0x03) << 4)).astype(np.int16) - 32
        q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 0x03) << 4)).astype(np.int16) - 32
        sc1 = sc_h[:, is_idx + 0]
        sc2 = sc_h[:, is_idx + 2]
        sc3 = sc_h[:, is_idx + 4]
        sc4 = sc_h[:, is_idx + 6]
        out[:, base + 0:base + 32] = d * sc1 * q1.astype(np.float32)
        out[:, base + 32:base + 64] = d * sc2 * q2.astype(np.float32)
        out[:, base + 64:base + 96] = d * sc3 * q3.astype(np.float32)
        out[:, base + 96:base + 128] = d * sc4 * q4.astype(np.float32)
    return out.reshape(*lead, 256)


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    """Q8_0: 32 signed bytes and one fp16 scale.  No bit unpacking at all."""
    flat, lead = _flat(blocks, Q8_0_BLOCK_BYTES)
    d = f16_to_f32(flat[:, 0:2])[:, None]
    qs = flat[:, 2:34].view(np.int8).astype(np.float32)
    return (qs * d).reshape(*lead, QK8_0)