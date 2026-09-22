"""The three quantization layouts MiMo-V2.6 ships, as torch references.

Each of these exists so the CUDA kernel that replaces it has something to be
diffed against that is obviously right. They are written for readability rather
than speed -- the experts these decode are 12.75 MiB each and there are 12,032 of
them, so nothing here is meant to run over the whole checkpoint.

The layouts, and the one detail in each that is easy to get backwards:

* **MXFP4 experts.** One byte holds two E2M1 codes; which nibble is the *even*
  input column is a convention, and getting it wrong transposes pairs of weights
  within a row rather than scrambling anything visibly. Here the low nibble is
  the even column, matching `src/kernels/ops.py`'s `_pack_fp4_codes`. The scale is
  one E8M0 byte per 32 input columns, read as `2 ** (byte - 127)`.
* **FP8 E4M3 dense linears.** 128x128 tiles, one float32 `weight_scale_inv` per
  tile, and the exporter normalises each tile so that `w = w_fp8 * scale`. The
  released global-attention `qkv_proj` carries two more scale rows than the weight
  has row-blocks; see `loader.py`.
* **BF16 everything else.** `o_proj`, the norms, the router, the embedding and the
  head, read at their stored dtype.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "E2M1_LEVELS",
    "E2M1_LEVELS_BY_NIBBLE",
    "dequant_fp8_block",
    "dequant_mxfp4",
    "e8m0_to_float",
    "unpack_e2m1",
]

#: The E2M1 codebook, indexed by the 4-bit code. Sign lives in the top bit, which is
#: why the negative half is not simply the negation of the first eight entries in
#: order: index 8 is -0.0.
E2M1_LEVELS = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

#: The same table as a tensor, which is the definition `unpack_e2m1` decodes to and
#: `test_the_codebook_is_what_the_decoder_computes` compares it against.
E2M1_LEVELS_BY_NIBBLE = torch.tensor(E2M1_LEVELS, dtype=torch.float32)

#: 128x128 is the released `weight_block_size`. Named here so a caller and the loader
#: cannot disagree about which block size a scale was built with.
FP8_BLOCK = (128, 128)

#: MXFP4's block: one E8M0 byte covers this many consecutive input columns.
MXFP4_BLOCK = 32


def e8m0_to_float(codes: torch.Tensor) -> torch.Tensor:
    """E8M0 as the reference reads it: `2 ** (byte - 127)`, with no mantissa.

    A byte of 127 is 1.0. Bytes below 127 are fractions, which is where the
    checkpoint's small weight magnitudes live.
    """
    return torch.exp2(codes.to(torch.float32) - 127.0)


def unpack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """`[..., K/2]` uint8 holding two codes per byte into `[..., K]` floats.

    The low nibble is the even column and the high nibble is the odd one. Pair order
    inside a byte is invisible to any check that only compares sums, so it is stated
    once here and relied on everywhere else.

    The two nibbles are looked up separately and interleaved with `stack`, which is
    about twice as fast as building the combined `int64` index first: at the shapes
    this runs on -- eight million elements per expert projection, twelve thousand
    experts per forward -- the `int64` casts and the strided write into a `long`
    tensor cost more than the gather does.
    """
    raw = codes.contiguous().view(torch.uint8)
    table = E2M1_LEVELS_BY_NIBBLE.to(raw.device)
    low = table[(raw & 0x0F).long()]
    high = table[((raw >> 4) & 0x0F).long()]
    return torch.stack([low, high], dim=-1).reshape(*raw.shape[:-1], -1)


def dequant_mxfp4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    block: int = MXFP4_BLOCK,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`[N, K/2]` codes beside `[N, K/block]` E8M0 scales into a dense `[N, K]`.

    This is the layout `moe_single_token_fp4_cuda` takes directly, so the only
    reason to build the dense tensor is to check that kernel.
    """
    unpacked = unpack_e2m1(codes).to(torch.float32)
    cols = unpacked.shape[-1]
    blocks = -(-cols // block)
    scale = e8m0_to_float(scales[..., :blocks]).to(torch.float32)
    if tuple(scale.shape) != unpacked.shape[:-1] + (blocks,):
        raise ValueError(
            f"scales {tuple(scale.shape)} do not cover {tuple(unpacked.shape)} at block {block}"
        )
    if blocks * block != cols:
        raise ValueError(f"{cols} columns is not a multiple of the block {block}")
    expanded = scale.repeat_interleave(block, dim=-1)
    return (unpacked * expanded).to(out_dtype)


def dequant_fp8_block(
    codes: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = FP8_BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """`w = w_fp8 * scale`, tile by tile, without materialising a broadcast scale.

    Building the full-size scale with `repeat_interleave` is exact but allocates a
    `rows x cols` float32 intermediate; the reshape below touches the same numbers
    in `rows x cols` worth of float32 once.

    `scale` may have more rows than the weight has row-blocks. That is not a
    tolerance, it is the released global-attention `qkv_proj`, whose scale carries
    two extra rows that the reference's own loader never reads. Extra columns are
    treated the same way, since the tile index that would use them does not exist.
    """
    rows, cols = codes.shape
    block_rows, block_cols = block
    row_blocks = -(-rows // block_rows)
    col_blocks = -(-cols // block_cols)
    scale = scale[:row_blocks, :col_blocks]
    expected = (row_blocks, col_blocks)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"block scale has {tuple(scale.shape)} rows/cols for a {tuple(codes.shape)} "
            f"weight with {block} blocks; {expected} needed"
        )

    padded_rows, padded_cols = row_blocks * block_rows, col_blocks * block_cols
    if (padded_rows, padded_cols) != (rows, cols):
        # Only reachable for a shape that is not a multiple of the tile, which none of
        # the released shapes are; the pad keeps the reshape legal if one is not.
        codes = F.pad(codes, (0, padded_cols - cols, 0, padded_rows - rows))
    tiles = codes.to(torch.float32).view(row_blocks, block_rows, col_blocks, block_cols)
    tiles = tiles * scale.to(torch.float32)[:, None, :, None]
    return tiles.view(padded_rows, padded_cols)[:rows, :cols].to(out_dtype)
