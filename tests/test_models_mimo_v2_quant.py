"""The three quantization layouts MiMo-V2.6 ships, checked as layouts.

The parity tests exercise what the model *computes*; these exercise how the
checkpoint *stores* it, which is the other place a port goes silently wrong.
Every one of these has a plausible wrong reading that produces numbers of the
right shape and the wrong value:

* **Nibble order.** One byte holds two E2M1 codes, and which one is the even input
  column is a convention. Swap it and you have transposed pairs of weights inside
  a row -- finite, same shape, wrong.
* **The E2M1 codebook is sign-magnitude.** Code 8 is negative zero, not -0.5, so
  the negative half is not the first half in reverse and a reader that computes
  `-levels[code & 7]` gets -0.5 where the checkpoint means -0.0.
* **E8M0 is `2 ** (byte - 127)`.** It is an exponent with a bias and no mantissa;
  reading it as a float8 or as `2 ** -byte` is a factor of four order of error.
* **The global-attention `qkv_proj` carries two scale rows the weight has no
  blocks for.** A reader that insists the shapes match raises; a reader that
  trusts the count reads two rows of somebody else's scale.

No checkpoint is needed here: these build the stored bytes themselves, which is
the only way to be sure the test is checking the layout rather than checking that
the checkpoint happens to be self-consistent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.quant import (  # noqa: E402
    E2M1_LEVELS,
    E2M1_LEVELS_BY_NIBBLE,
    FP8_BLOCK,
    MXFP4_BLOCK,
    dequant_fp8_block,
    dequant_mxfp4,
    e8m0_to_float,
    unpack_e2m1,
)


def pack_e2m1(codes: list[int]) -> torch.Tensor:
    """The checkpoint's own packing: even column in the low nibble."""
    assert len(codes) % 2 == 0
    bytes_ = [
        (codes[i] & 0x0F) | ((codes[i + 1] & 0x0F) << 4) for i in range(0, len(codes), 2)
    ]
    return torch.tensor(bytes_, dtype=torch.uint8)


def test_e8m0_is_a_biased_exponent_with_127_at_one():
    codes = torch.tensor([0, 126, 127, 128, 130], dtype=torch.uint8)
    got = e8m0_to_float(codes)
    want = torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 8.0], dtype=torch.float32)
    assert torch.equal(got, want)


def test_the_low_nibble_is_the_even_column():
    """Byte 0x21 is `0.5, 1.0`, not `1.0, 0.5`."""
    packed = pack_e2m1([1, 2, 3, 4])
    assert packed.tolist() == [0x21, 0x43]
    assert unpack_e2m1(packed).tolist() == [0.5, 1.0, 1.5, 2.0]


def test_swapping_the_nibbles_gives_a_different_tensor():
    """The falsification for the convention above: it is not a symmetry."""
    packed = pack_e2m1([1, 2])
    flipped = ((packed & 0x0F) << 4) | ((packed >> 4) & 0x0F)
    assert not torch.equal(unpack_e2m1(packed), unpack_e2m1(flipped))


def test_the_codebook_is_sign_magnitude_so_code_eight_is_negative_zero():
    assert E2M1_LEVELS[8] == 0.0
    assert E2M1_LEVELS[15] == -6.0
    # The two halves are mirror images per *code*, not an offset by eight.
    for low in range(8):
        assert E2M1_LEVELS[low + 8] == -E2M1_LEVELS[low]
    # Skip code 0 as well: -0.0 == 0.0, so this would pass on twos complement too.
    assert E2M1_LEVELS[9] == -0.5, "a `code - 8` reading puts -0.5 somewhere else"


def test_the_decoder_matches_the_codes_table_both_ways():
    """Every byte value, both nibble positions, against the table it is defined by.

    `unpack_e2m1` interleaves the two nibbles with `stack` rather than by writing
    into a combined `int64` index, which is a performance decision taken in a module
    whose job is to be obviously right. This is the check that it is not also a
    behavioural one.
    """
    codes = list(range(16))
    packed = pack_e2m1(codes + codes)
    got = unpack_e2m1(packed)
    want = E2M1_LEVELS_BY_NIBBLE[torch.tensor(codes + codes)]
    assert torch.equal(got, want)
    # And as bytes, so a shifted nibble cannot hide behind the ordering above.
    every = torch.arange(256, dtype=torch.uint8)
    decoded = unpack_e2m1(every)
    assert torch.equal(decoded[0::2], E2M1_LEVELS_BY_NIBBLE[(every & 0x0F).long()])
    assert torch.equal(decoded[1::2], E2M1_LEVELS_BY_NIBBLE[((every >> 4) & 0x0F).long()])


def test_dequant_mxfp4_multiplies_each_block_by_its_own_scale():
    # Two rows, one block of 4 columns each: the point is that row 1 does not see
    # row 0's scale and column 3 does not see column 0's. The second row's codes are
    # 8..11, the *negative* half of the codebook, so a sign error shows up here too.
    codes = torch.stack([pack_e2m1([1, 2, 3, 4]), pack_e2m1([8, 9, 10, 11])])
    scales = torch.tensor([[128], [129]], dtype=torch.uint8)  # 2.0 and 4.0
    got = dequant_mxfp4(codes, scales, block=4)
    want = torch.tensor(
        [[0.5 * 2, 1.0 * 2, 1.5 * 2, 2.0 * 2], [-0.0, -0.5 * 4, -1.0 * 4, -1.5 * 4]]
    )
    assert torch.equal(got, want)


def test_dequant_mxfp4_rejects_a_scale_that_does_not_cover_the_columns():
    codes = torch.zeros(2, 4, dtype=torch.uint8)  # 8 columns
    scales = torch.zeros(2, 1, dtype=torch.uint8)  # 4 columns of scale at block 4
    with pytest.raises(ValueError, match="do not cover"):
        dequant_mxfp4(codes, scales, block=4)


def test_dequant_mxfp4_rejects_a_column_count_the_block_does_not_divide():
    codes = torch.zeros(1, 3, dtype=torch.uint8)  # 6 columns
    scales = torch.zeros(1, 2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiple of the block"):
        dequant_mxfp4(codes, scales, block=4)


def test_the_default_mxfp4_block_is_the_released_one():
    assert MXFP4_BLOCK == 32
    assert FP8_BLOCK == (128, 128)


def test_dequant_fp8_block_scales_tile_by_tile():
    codes = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0], [4.0, 8.0]], dtype=torch.float32)
    got = dequant_fp8_block(codes, scale, block=(2, 2))
    want = torch.tensor(
        [
            [0.5, 1.0, 6.0, 8.0],
            [2.5, 3.0, 14.0, 16.0],
            [36.0, 40.0, 88.0, 96.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(got, want)


def test_dequant_fp8_block_pads_a_shape_the_tile_does_not_divide():
    """No released shape needs this, but a pad that forgot to slice back would."""
    codes = torch.ones(3, 3, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2), 2.0)
    got = dequant_fp8_block(codes, scale, block=(2, 2))
    assert tuple(got.shape) == (3, 3)
    assert torch.equal(got, torch.full((3, 3), 2.0))


def test_dequant_fp8_block_rejects_a_scale_with_too_few_rows():
    codes = torch.ones(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.ones(1, 1)
    with pytest.raises(ValueError, match="blocks; .* needed"):
        dequant_fp8_block(codes, scale)


def test_the_extra_rows_of_a_global_qkv_scale_are_never_read():
    """The released anomaly, as a property of the function rather than the file.

    A global-attention `qkv_proj` is 13568 rows -- 106 blocks -- beside a scale with
    108. The scale rows that have no block are filled with NaN here: if the
    dequantizer read them, the output would be NaN rather than merely wrong.
    """
    rows, cols = 13568, 128
    codes = torch.ones(rows, cols, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.ones(108, 1, dtype=torch.float32)
    scale[106:] = float("nan")
    got = dequant_fp8_block(codes, scale, block=(128, 128), out_dtype=torch.float32)
    assert tuple(got.shape) == (rows, cols)
    assert torch.isfinite(got).all()
    assert torch.equal(got, torch.ones(rows, cols))


def test_a_tile_normalised_weight_round_trips_through_the_fp8_layout():
    """The exporter's own normalisation, reproduced: `w = w_fp8 * scale` per tile."""
    torch.manual_seed(1234)
    weight = torch.randn(256, 256) * 0.03
    tiles = weight.view(2, 128, 2, 128)
    per_tile_max = tiles.abs().amax(dim=(1, 3), keepdim=True)
    scale = per_tile_max / 448.0
    stored = (tiles / scale).to(torch.float8_e4m3fn)
    got = dequant_fp8_block(
        stored.reshape(256, 256),
        scale.reshape(2, 2).to(torch.float32),
        out_dtype=torch.float32,
    )
    # E4M3 keeps three mantissa bits, so ~6% per element is the format's own floor.
    assert torch.allclose(got, weight, rtol=0.07, atol=1e-3)
    assert got.abs().max() > 0.05, "a wrong scale would shrink or blow up the result"
