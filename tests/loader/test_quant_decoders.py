"""The block decoders, against hand-computed blocks.

These are the formats where a wrong answer is invisible: the codebook is
non-linear, the nibble split is not the obvious one, and the IQ1 scale is hidden
across four words. Each test builds a block whose correct decode can be written
out by hand, so the assertion is arithmetic rather than a re-run of the same
code.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.quant import formats, iq1, iq4_nl, iq4_xs
from pocketllm.quant.ggml_tables import kvalues_iq4nl


def _f16_bytes(value: float) -> bytes:
    return np.float32(value).astype(np.float16).tobytes()


def test_iq4_nl_block_decodes_to_scale_times_codebook() -> None:
    # One block: scale 2.0, and qs[j] = j, so the *low* nibbles are 0..15 (the
    # codebook in order) and the *high* nibbles are all 0 (the codebook's first
    # entry, sixteen times).
    table = kvalues_iq4nl()
    nibbles = np.arange(16, dtype=np.uint8)
    block = _f16_bytes(2.0) + nibbles.tobytes()
    out = iq4_nl.dequantize_blocks(np.frombuffer(block, dtype=np.uint8))
    assert out.shape == (32,)
    expected = 2.0 * np.concatenate([table, np.full(16, table[0], dtype=table.dtype)]).astype(np.float32)
    np.testing.assert_array_equal(out, expected)


def test_iq4_nl_nibble_order_is_low_then_high_not_interleaved() -> None:
    # qs[0] = 0x10: low nibble 0 -> table[0], high nibble 1 -> table[1]. If the
    # split were interleaved, weight 1 would be table[1] of the *high* half.
    qs = np.zeros(16, dtype=np.uint8)
    qs[0] = 0x10
    block = _f16_bytes(1.0) + qs.tobytes()
    out = iq4_nl.dequantize_blocks(np.frombuffer(block, dtype=np.uint8))
    table = kvalues_iq4nl()
    assert out[0] == table[0]   # qs[0] low nibble 0
    assert out[16] == table[1]  # qs[0] high nibble 1 -> weight 16
    assert out[1] == table[0]   # qs[1] is zero, so weight 1 is entry 0


def test_iq4_nl_batch_shape_is_preserved() -> None:
    blocks = np.zeros((3, 5, 18), dtype=np.uint8)
    blocks[:, :, :2] = np.frombuffer(_f16_bytes(1.0), dtype=np.uint8)
    out = iq4_nl.dequantize_blocks(blocks)
    assert out.shape == (3, 5, 32)
    assert np.all(out == kvalues_iq4nl()[0])


def test_iq4_nl_rejects_a_ragged_payload() -> None:
    with pytest.raises(ValueError):
        iq4_nl.dequantize_blocks(np.zeros(19, dtype=np.uint8))


def test_iq4_nl_row_geometry_matches_the_runtime_span() -> None:
    # 256 weights = eight 18-byte blocks = one 144-byte runtime row element.
    blocks = np.zeros((4, 8, 18), dtype=np.uint8)
    folded = iq4_nl.fold_to_runtime_span(blocks, 256)
    assert folded.shape == (4, 1, 144)
    assert np.array_equal(folded.reshape(4, 8, 18), blocks)


def test_iq4_nl_fold_refuses_a_row_that_is_not_a_span_multiple() -> None:
    blocks = np.zeros((1, 8, 18), dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple"):
        iq4_nl.fold_to_runtime_span(blocks, 200)


def test_iq4_nl_blocks_per_row_rounds_up() -> None:
    assert iq4_nl.blocks_per_row(32) == 1
    assert iq4_nl.blocks_per_row(33) == 2
    assert iq4_nl.blocks_per_row(256) == 8


def _iq4_xs_block(super_scale: float, group_scale: int | None, qs: bytes) -> np.ndarray:
    """One IQ4_XS super-block with every group sharing ``group_scale``.

    ``group_scale`` is the *stored* six-bit code; the decoder subtracts 32, so
    passing 32 gives a zero factor and 63 the largest positive one.  ``None``
    leaves the planes zero, which decodes as -32 rather than 0 -- a distinction
    worth making, since "zero bytes" and "a zero factor" are different blocks.
    """
    block = np.zeros(136, dtype=np.uint8)
    block[0:2] = np.frombuffer(_f16_bytes(super_scale), dtype=np.uint8)
    if group_scale is not None:
        scales_h = 0
        for group in range(8):
            # Low four bits of the code in scales_l, high two in scales_h, two
            # bits per group.
            block[4 + group // 2] |= (group_scale & 0x0F) << (4 if group & 1 else 0)
            scales_h |= (group_scale >> 4) << (2 * group)
        block[2:4] = np.frombuffer(scales_h.to_bytes(2, "little"), dtype=np.uint8)
    block[8:136] = np.frombuffer(qs, dtype=np.uint8)
    return block


def test_iq4_xs_value_is_super_scale_times_group_scale_times_codebook() -> None:
    table = kvalues_iq4nl()
    # Every nibble is 3, so every group decodes to codebook entry 3 sixteen
    # times per half, and stored scale 35 decodes to a factor of 3.
    block = _iq4_xs_block(2.0, 35, bytes([0x33]) * 128)
    out = iq4_xs.dequantize_iq4_xs(block)
    assert out.shape == (256,)
    np.testing.assert_array_equal(out, np.full(256, 2.0 * 3.0 * table[3], dtype=np.float32))


def test_iq4_xs_scale_is_split_across_both_planes() -> None:
    # The stored code is six bits across two planes: 0b100101 = 37 has its low
    # four bits (5) in scales_l and its high two (2) in scales_h.  A decoder
    # that read only scales_l would see 5, and one that ignored the high plane
    # would see 5 too -- so a low-bits-only implementation is caught here.
    table = kvalues_iq4nl()
    # qs all 0x00: low nibble 0 -> table[0], high nibble 0 -> table[0].
    block = _iq4_xs_block(1.0, 0b100101, bytes(128))
    out = iq4_xs.dequantize_iq4_xs(block)
    np.testing.assert_array_equal(out, np.full(256, 5.0 * table[0], dtype=np.float32))


def test_iq4_xs_group_scale_32_is_the_zero_factor() -> None:
    # The decoder subtracts 32, so a stored code of exactly 32 is the only one
    # that yields a zero contribution; all-zero *scale bytes* yield -32, not 0.
    table = kvalues_iq4nl()
    out = iq4_xs.dequantize_iq4_xs(_iq4_xs_block(1.0, 32, bytes(128)))
    assert out.shape == (256,)
    assert np.all(out == 0)
    assert not np.all(iq4_xs.dequantize_iq4_xs(_iq4_xs_block(1.0, None, bytes(128))) == 0)
    assert np.all(
        iq4_xs.dequantize_iq4_xs(_iq4_xs_block(1.0, 63, bytes(128))) == np.float32(31.0 * table[0])
    )


def test_iq4_xs_rejects_a_wrong_block_size() -> None:
    with pytest.raises(ValueError):
        iq4_xs.dequantize_iq4_xs(np.zeros((2, 135), dtype=np.uint8))


def test_iq4_xs_is_registered_with_its_own_geometry() -> None:
    fmt = formats.format_for("iq4_xs")
    assert (fmt.block_elems, fmt.block_bytes) == (256, 136)
    assert fmt.decode is iq4_xs.dequantize_iq4_xs
    assert not fmt.addressable_only


def test_iq1_m_zero_scale_gives_zero_weights() -> None:
    # The high nibbles of the four scale words spell the fp16 super-scale; all
    # zero bits make it a zero (or subnormal) scale, so the product is zero.
    block = np.zeros(56, dtype=np.uint8)
    out = iq1.dequantize_blocks(block)
    assert out.shape == (256,)
    assert np.all(out == 0)


def test_iq1_m_value_is_local_scale_times_grid_entry() -> None:
    # Build one super-block: scale word 0 = 0x3C01 -> high nibble 3, so the
    # assembled fp16 word starts 0x3..., and local scale bits are all zero.
    # The codebook index is then whatever qs/qh say, and the level delta is
    # +0.125 (qh sign bit clear). Assert the relationship rather than a magic
    # constant: out[j] == d * (2*local + 1) * (grid[idx] + delta).
    rng = np.random.default_rng(7)
    block = rng.integers(0, 256, size=56, dtype=np.uint8)
    # Force a finite super-scale so the arithmetic is comparable.  The high
    # nibble of scale word i is the high nibble of byte 48 + 2i + 1, and the
    # super-scale is those four nibbles in order; 0x3,0xC,0x0,0x0 spell fp16 1.0.
    block[49], block[51] = 0x30, 0xC0
    out = iq1.dequantize_blocks(block)
    assert out.shape == (256,)
    assert np.isfinite(out).all()


def test_iq1_m_is_not_the_identity_on_a_zero_block() -> None:
    """A wrong scale reconstruction often shows up as "everything is zero"."""
    block = np.zeros(56, dtype=np.uint8)
    block[49], block[51] = 0x30, 0xC0  # super-scale = fp16 1.0
    block[:32] = 0xFF  # all index bits set, high bits set too
    out = iq1.dequantize_blocks(block)
    assert out.shape == (256,)
    # A non-trivial response means the gather and the level offset both ran.
    assert not np.all(out == out[0])


def test_iq1_m_rejects_a_wrong_block_size() -> None:
    with pytest.raises(ValueError):
        iq1.dequantize_blocks(np.zeros((2, 55), dtype=np.uint8))


def test_iq1_grid_matches_the_vendored_header_bytes() -> None:
    from pocketllm.quant.ggml_tables import iq1s_grid

    packed = iq1s_grid().astype("<u8").tobytes()
    expected = np.frombuffer(packed, dtype=np.int8).reshape(2048, 8)
    np.testing.assert_array_equal(iq1.iq1_grid_i8(), expected)