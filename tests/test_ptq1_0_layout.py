"""Layout of the PTQ1_0 ternary block, pinned against the released checkpoint.

Every case below is 28 bytes read out of `Ternary-Bonsai-2-27B-PTQ1_0.gguf` at
a stated (row, block) position, together with the scale and the 128 trits the
fork's own reference decoder (ggml/src/ggml-quants.c @ 842b188, unmodified)
produces for them.  See `src/loader/gguf/ptq1_0.py` for the format.)"""

from __future__ import annotations

import numpy as np
import pytest

from src.loader.gguf.ptq1_0 import (
    PTQ1_0_BLOCK_BYTES,
    QK_PTQ1_0,
    block_scale_fp16,
    block_trits,
    dequantize_blocks,
    row_blocks,
)

# (tensor, row, block in row, blocks per row) -> (28 bytes hex, scale hex, trits)
GOLDEN = [
    (
        "blk.0.ffn_gate.weight", 0, 0, 40,
        (
            "f0 e2 b9 3d 08 95 e9 e8 2e 2e 07 58 31 df b8 a6 e5 fd f5 94 13 6f 14 21 e7 cd 1e 85",
            "851e",
            "+++--0++---0-++0+0-+-+++00--00-+0+0-----00--++0+-+00++00+++--000+00-0-+000-+00-0+++0-0--++++---0-+0-+++--++0-+-00-0+---0+++0--00",
        ),
    ),
    (
        "blk.0.ffn_gate.weight", 0, 39, 40,
        (
            "fb 30 1f 58 68 75 fb d7 c8 52 82 76 6d 4a 1b 35 99 ee 23 bf ea 8f 80 21 ae d8 af 56",
            "56af",
            "+--000+++-000---+00--0+00+00-+-0++--0-+0-+0-+0++0---+00+-0+00++00-+++-0--+-0000+0+-++00-++0-++0000-+--0---+-+-000--0--00++-0-00+",
        ),
    ),
    (
        "blk.0.ffn_gate.weight", 17407, 0, 40,
        (
            "ab 41 f8 a2 fb 05 08 6a 11 7a 6d 97 89 34 fd 41 81 ff 27 af de 04 1e 21 64 25 1f d8",
            "d81f",
            "+-+0+--0-0000-+--++++----0-+00++--+++--+0-+-+++--+--00+-++0+00++-00-0000000+00-00+-++---0+0-0-000+0-+---0+-000-0++00--000--00-0+",
        ),
    ),
    (
        "blk.0.ffn_gate.weight", 9000, 17, 40,
        (
            "ad b7 09 7e 91 8e b0 52 41 a9 93 8d 90 f5 96 24 25 6d 57 a4 13 bb 74 21 54 e8 5a 39",
            "395a",
            "++-000+--0000+0----0+0-++++0+++0-0-0-+-+-+-+-0----+--+00++0+-+++++++++++00000000-000-+0-0--+--00-+-++0--+0---+-0+00+--+0-++++-+0",
        ),
    ),
    (
        "blk.63.attn_q.weight", 0, 0, 40,
        (
            "76 df 90 1a ea ff 25 27 af 1c a8 1f 0b d2 ae 72 37 be 10 fb de 72 cc 24 54 c8 65 a8",
            "a865",
            "0+0-++--+-0--++000+-++00--+0-0-0-+-+-+-0-++-00--00-++++-0++---0-000--++00+-+00---+-++0+-0--+0000++0++---+-+00-0+0--0--00-++0+-+-",
        ),
    ),
    (
        "blk.0.attn_qkv.weight", 10239, 39, 40,
        (
            "b8 34 67 68 25 ae 15 57 7b 6d 3e 65 a3 59 c8 fa 9d 07 32 e1 2d ea a9 24 90 f1 73 8f",
            "8f73",
            "+-00-+-000-000++-0--0---0-+-+-0+0+00--+--+-0+--+00+++0--+000-0-0-00++-00+00+0---0--+-+0-+-000++00-++0-+-0+-++++++-+---000+++-0-0",
        ),
    ),
    (
        "blk.31.ffn_down.weight", 0, 135, 136,
        (
            "87 2d c7 47 fb 7d fb 7f 5e 92 39 ce f7 94 97 b7 f8 5d 3f 2f ab 10 99 25 31 07 82 3d",
            "3d82",
            "0-+-+0+000-++00+00-++0+0-++0+++-+0+0+0+0----+--0-++00-00+0-+-0+-+-+0000-+----++++0--+-0-+-+0--+0+--0-00--+0+-+-+00++--0+--0-+--+",
        ),
    ),
    (
        "token_embd.weight", 0, 0, 40,
        (
            "e9 02 6a 18 fe 05 71 18 75 f7 4c 1d 10 4a 20 16 36 9f ab 69 2d 26 8b 2d 1e cb 85 c2",
            "c285",
            "+-0-+-0-0+------+---+---0++0-+0---+++-++-++-00-+0--0+0+00---++0-+00000+0-----0-+-0+0--0-0+--0000+0-+00+0++--+-0+------+--+00---0",
        ),
    ),
    (
        "token_embd.weight", 248319, 39, 40,
        (
            "fa ee 68 cc 41 0d 83 ea e3 90 5e 6e 5d bc ff cf 68 dd af 0e 40 17 f5 1a 05 f2 bd 1b",
            "1bbd",
            "++0+--0++0000+++++-0+-0+0+----+0+00--00-+--+-0+-0-+0+0+++-+0++++--+00-0-+0++00+00++---+--0--+-+-0+-0-+0++-00+0++++00--0--+-+-000",
        ),
    ),
    (
        "output.weight", 12345, 20, 40,
        (
            "f0 dd 48 b9 b7 8c de 39 ae 03 41 9d 53 d4 3a 90 f8 f6 f5 81 bb 26 f6 22 33 51 4a 5c",
            "5c4a",
            "++-++0+-+--0-+-0+0+--00+--+++0++0+000++----0+0----00-+0-0-+0+0--+++0+----+0+--00+++0+-+-+++0-0+0+000000--++0+-+00+0+--++--0+++00",
        ),
    ),
]
GOLDEN_PARAMS = [
    pytest.param(case[0], case[1], case[2], case[3], case[4],
                 id=f"{case[0].split('.')[-2] if 'blk' in case[0] else case[0]}-r{case[1]}-b{case[2]}")
    for case in GOLDEN
]

# cases[i] is (tensor, row, block in row, blocks per row, (bytes hex, scale hex, trits))
_BYTES, _SCALE, _TRITS = 0, 1, 2


@pytest.mark.parametrize(("tensor", "row", "block", "blocks_per_row", "case"), GOLDEN_PARAMS)
def test_a_released_block_decodes_to_the_reference_trits(tensor, row, block, blocks_per_row, case):
    """The decoder agrees with the fork's reference on real checkpoint bytes."""
    raw = bytes.fromhex(case[_BYTES].replace(" ", ""))
    assert len(raw) == PTQ1_0_BLOCK_BYTES
    expected = case[_TRITS]
    assert len(expected) == QK_PTQ1_0
    trits = block_trits(raw)
    got = "".join("-" if t < 0 else "0" if t == 0 else "+" for t in trits)
    assert got == expected, f"{tensor} row {row} block {block} of {blocks_per_row}"
    assert all(t in (-1, 0, 1) for t in trits)


@pytest.mark.parametrize(("tensor", "row", "block", "blocks_per_row", "case"), GOLDEN_PARAMS)
def test_the_scale_is_the_last_two_bytes_and_the_product_is_exact(tensor, row, block,
                                                                 blocks_per_row, case):
    raw = bytes.fromhex(case[_BYTES].replace(" ", ""))
    assert block_scale_fp16(raw) == int(case[_SCALE], 16)

    scale = np.frombuffer(raw[-2:], dtype=np.float16)[0].astype(np.float32)
    dequant = dequantize_blocks(raw)
    assert dequant.shape == (QK_PTQ1_0,)
    assert np.array_equal(dequant, block_trits(raw).astype(np.float32) * scale)


def _naive_trits(raw: bytes) -> np.ndarray:
    """The reading a reader gets if it treats ``qs`` as one continuous run."""
    out = np.empty(QK_PTQ1_0, dtype=np.int8)
    n = 0
    for byte in list(raw[:24]) + list(raw[24:26]):
        for shift in (1, 3, 9, 27, 81):
            q = (byte * shift) & 0xFF
            if n < QK_PTQ1_0:
                out[n] = ((q * 3) >> 8) - 1
                n += 1
    return out


def _stage_trits(raw: bytes, stages: tuple[int, ...]) -> np.ndarray:
    """The same walk over ``qs`` with a caller-supplied stage table."""
    out: list[int] = []
    byte = 0
    for width in stages:
        while byte + width <= 24:
            for shift in (1, 3, 9, 27, 81):
                for m in range(width):
                    q = (raw[byte + m] * shift) & 0xFF
                    out.append(((q * 3) >> 8) - 1)
            byte += width
    return np.array(out, dtype=np.int8)


def test_the_stage_walk_is_not_a_continuous_run():
    """``qs`` is walked in stages, and the 32-wide one never fits.

    24 bytes cannot hold a 32-byte stage, so the format's declared
    ``{32, 16, 8}`` reduces to 16 then 8 and covers byte 0..15 with weights 0..79.
    Reading the 24 bytes as one 120-trit run puts different trits on different
    weights, which is the regression this test exists to catch.
    """
    raw = bytes.fromhex(GOLDEN[0][4][_BYTES].replace(" ", ""))
    full = block_trits(raw)
    assert np.array_equal(_stage_trits(raw, (32, 16, 8)), full[:120])
    assert np.array_equal(_stage_trits(raw, (32, 16, 8)), _stage_trits(raw, (16, 8)))
    assert not np.array_equal(full, _naive_trits(raw))


def test_the_hadamard_block_is_a_whole_number_of_quant_blocks():
    """The format nests: 128 divides the 1024-wide transform block, which divides
    every row the checkpoint declares a sign vector for."""
    assert 1024 % QK_PTQ1_0 == 0
    for width in (5120, 6144, 17408):
        assert width % 1024 == 0
        assert row_blocks(width) == width // QK_PTQ1_0


def test_row_blocks_refuses_a_row_that_would_split_a_block():
    with pytest.raises(ValueError, match="multiple of 128"):
        row_blocks(QK_PTQ1_0 + 1)


def test_dequantize_blocks_accepts_flat_and_shaped_payloads():
    raw = b"".join(bytes.fromhex(case[4][_BYTES].replace(" ", "")) for case in GOLDEN)
    flat = dequantize_blocks(raw)
    assert flat.shape == (len(GOLDEN), QK_PTQ1_0)
    shaped = dequantize_blocks(np.frombuffer(raw, dtype=np.uint8).reshape(-1, PTQ1_0_BLOCK_BYTES))
    assert np.array_equal(flat, shaped)
    with pytest.raises(ValueError, match="not a whole number of"):
        dequantize_blocks(raw[:-1])


def test_a_zero_scale_block_dequantizes_to_zeros():
    raw = bytes(PTQ1_0_BLOCK_BYTES)
    assert np.all(dequantize_blocks(raw) == 0.0)
