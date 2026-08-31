"""FP8 block-scaled weight handling for Qwen4-Exp.

The Flash-Next FP8 checkpoint stores routed experts as `float8_e4m3fn` codes plus
a BF16 scale per 128x128 tile, and the PLE n-gram table as codes plus one scalar
scale.  Both are *multipliers*: ``value = code * scale``.  `torch._scaled_mm` is
unavailable on SM75, so the only route is to dequantize into a tensor-core dtype
and run an ordinary GEMM.

Dequantization multiplies in the destination dtype rather than in float32.  For
float16 that is exact — every `e4m3` code is representable in float16, and a
BF16 scale has fewer mantissa bits than float16 — while avoiding a 4x-wide
intermediate.  For bfloat16 the product rounds to 8 mantissa bits, which is the
same rounding a native BF16 runtime would apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class FP8Tensor:
    """One block-scaled weight: `code` with `scale` over `block` tiles."""

    code: torch.Tensor
    scale: torch.Tensor
    block: tuple[int, int] = (128, 128)

    @property
    def shape(self) -> torch.Size:
        return self.code.shape

    @property
    def device(self) -> torch.device:
        return self.code.device

    def nbytes(self) -> int:
        return (
            self.code.numel() * self.code.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def to(self, device, *, non_blocking: bool = False) -> "FP8Tensor":
        """Move the codes and scales without dequantizing."""
        return FP8Tensor(
            self.code.to(device, non_blocking=non_blocking),
            self.scale.to(device, non_blocking=non_blocking),
            self.block,
        )

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_block_fp8(self.code, self.scale, self.block, dtype)


def dequantize_block_fp8(
    code: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a 128x128-block-scaled FP8 weight to `dtype`.

    `code` is `[out, in]`; `scale` is `[ceil(out/block[0]), ceil(in/block[1])]`.
    """
    if code.dim() != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(code.shape)}")
    rows, cols = code.shape
    block_rows, block_cols = int(block[0]), int(block[1])
    expected = (-(-rows // block_rows), -(-cols // block_cols))
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match a {block_rows}x{block_cols} "
            f"tiling of {(rows, cols)} (expected {expected})"
        )

    values = code.to(dtype)
    if rows % block_rows == 0 and cols % block_cols == 0:
        # Every tile is whole, so the scale broadcasts over a reshape without
        # materializing a full-size scale tensor.
        tiled = values.view(expected[0], block_rows, expected[1], block_cols)
        tiled = tiled * scale.to(dtype).view(expected[0], 1, expected[1], 1)
        return tiled.reshape(rows, cols)
    # Ragged tail: expand and crop.  Not hit by this checkpoint (640, 1280 and
    # 2560 all divide by 128) but kept so the helper is not silently wrong.
    expanded = (
        scale.to(dtype)
        .repeat_interleave(block_rows, dim=0)
        .repeat_interleave(block_cols, dim=1)[:rows, :cols]
    )
    return values * expanded


def fp8_scalar_dequantize(
    code: torch.Tensor,
    scale: torch.Tensor | float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an FP8 tensor that shares one scale (the PLE n-gram table)."""
    factor = scale.to(dtype) if isinstance(scale, torch.Tensor) else scale
    return code.to(dtype) * factor


def pack_gate_up_fp8(
    gate: FP8Tensor,
    up: FP8Tensor,
) -> FP8Tensor:
    """Row-concatenate gate and up into the packed `[2*inter, hidden]` layout.

    The runtime's expert convention is one fused `gate_up` weight with gate rows
    first, which is how the BF16 checkpoint ships it.  `moe_intermediate_size` is
    640 and divides the 128-row block, so concatenating the scale grids keeps
    every tile aligned with its rows.
    """
    if gate.block != up.block:
        raise ValueError(f"gate/up block size mismatch: {gate.block} vs {up.block}")
    rows = gate.code.shape[0]
    block_rows = gate.block[0]
    if rows % block_rows != 0:
        raise ValueError(
            f"gate rows {rows} do not divide the {block_rows}-row scale block, so "
            "concatenating gate and up would misalign the scale grid"
        )
    return FP8Tensor(
        torch.cat([gate.code, up.code], dim=0),
        torch.cat([gate.scale, up.scale], dim=0),
        gate.block,
    )
