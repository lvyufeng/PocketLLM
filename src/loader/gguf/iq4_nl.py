"""The IQ4_NL non-linear 4-bit block format.

``IQ4_NL`` is upstream GGML type 20: 32 weights per block in 18 bytes, i.e. 4.5
bits per weight.  It is ``IQ4_XS``'s sibling and shares its codebook, but not its
block: ``IQ4_XS`` is a k-quant with 256-weight super-blocks, six-bit scales and
eight bytes of low bits per block, while ``IQ4_NL`` has one fp16 scale for the
whole 32 and nothing else.  The 16 codebook entries are *not* evenly spaced --
they are a non-linear set chosen for the distribution of 4-bit weights, which is
the whole point of the format and the one place a decoder can silently drift.

Block layout (``QK4_NL = 32`` weights, 18 bytes)::

    ggml_half d       one scale for all 32 weights
    uint8_t   qs[16]  two nibbles each, low nibble first

The nibbles are *not* interleaved: ``qs[j] & 0x0f`` is weight ``j`` and
``qs[j] >> 4`` is weight ``j + 16``, for ``j`` in 0..15.  ``IQ4_XS`` packs the
same way, which is why both decoders in ``tensor_reader`` write their low halves
into the first 16 slots rather than alternating.  The codebook is read out of the
vendored llama.cpp header rather than transcribed, so there is one statement of
it in this repository: ``kvalues_iq4nl`` in
``src/csrc/llama_mmq/ggml-common.h``.

Decoding blocks and running them are different things, and for one stage they
were separated here on purpose: ``IQ4_NL`` was absent from ``GGUF_DENSE_TYPE_IDS``
-- the raw-block runtime's dispatch table -- so a checkpoint whose tensors were
``IQ4_NL`` raised rather than reaching a kernel that would read 32-weight blocks
as a 256-weight format.  It is in that table now (``#393``), because
``src/csrc/cuda_kernel_impl.cu`` has a ``iq4nl_block_dot_256`` that walks eight
native blocks where the other ten formats unpack one 256-weight header.  What
made that a small kernel rather than a second kernel family is
:func:`fold_to_runtime_span` below: the loader folds eight native blocks into a
144-byte row element so every GEMM above sees the uniform 256-wide geometry the
runtime already had.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

QK_IQ4_NL = 32
"""Weights per block."""

IQ4_NL_BLOCK_BYTES = 18
"""Bytes per block: 16 packed nibbles plus a 2-byte fp16 scale."""

IQ4_NL_FILE_TYPE_ID = 20
"""The GGML file type id, as it appears in a GGUF tensor table."""

_GGML_COMMON = Path(__file__).parents[2] / "csrc" / "llama_mmq" / "ggml-common.h"


@lru_cache(maxsize=1)
def kvalues() -> np.ndarray:
    """The 16-entry codebook, as int8, in codebook-index order.

    Parsed from the vendored llama.cpp header rather than written out here.  The
    upstream table is a list of signed *decimal* literals (unlike the IQ2/IQ3
    grids, which are hex), so this is a separate reader from
    ``tensor_reader._extract_ggml_table``; both read the same file.
    """
    text = _GGML_COMMON.read_text(encoding="utf-8")
    match = re.search(
        r"GGML_TABLE_BEGIN\(int8_t,\s*kvalues_iq4nl,\s*16\)(.*?)GGML_TABLE_END\(\)",
        text,
        flags=re.S,
    )
    if match is None:
        raise RuntimeError(f"failed to locate kvalues_iq4nl in {_GGML_COMMON}")
    body = re.sub(r"//[^\n]*", "", match.group(1))
    values = [int(item) for item in re.findall(r"-?\d+", body)]
    if len(values) != 16:
        raise RuntimeError(f"kvalues_iq4nl expected 16 entries, got {len(values)}")
    table = np.asarray(values, dtype=np.int8)
    if table.min() < -127 or table.max() > 127:
        raise RuntimeError("kvalues_iq4nl is not a signed byte table")
    return table


def dequantize_blocks(blocks: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode whole IQ4_NL blocks into ``float32`` weights.

    ``blocks`` is either a flat byte buffer whose length is a multiple of 18 or
    an array shaped ``(..., 18)``.  The result is ``float32`` shaped ``(..., 32)``,
    so a ``(rows, blocks_per_row, 18)`` slice of a tensor decodes to
    ``(rows, blocks_per_row, 32)`` and reshapes to a row without a transpose.

    Every product is exact: a codebook entry is a small integer and the scale is
    a finite fp16, so the fp32 result is the reference's arithmetic rather than an
    approximation of it.
    """
    if isinstance(blocks, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(blocks, dtype=np.uint8)
    else:
        arr = np.asarray(blocks, dtype=np.uint8)
    if arr.size % IQ4_NL_BLOCK_BYTES:
        raise ValueError(
            f"IQ4_NL payload of {arr.size} bytes is not a whole number of "
            f"{IQ4_NL_BLOCK_BYTES}-byte blocks"
        )
    flat = arr.reshape(-1, IQ4_NL_BLOCK_BYTES)
    scale = _f16_to_f32(flat[:, :2])
    out = decode_indices(flat[:, 2:]) * scale[:, None]
    return out.reshape(*arr.shape[:-1], QK_IQ4_NL)


def decode_indices(qs: np.ndarray) -> np.ndarray:
    """Turn ``(..., 16)`` packed nibbles into ``(..., 32)`` codebook values.

    The low nibble of ``qs[j]`` is weight ``j`` and the high nibble is weight
    ``j + QK4_NL / 2`` -- a block layout, not an alternating one.
    """
    qs = np.asarray(qs, dtype=np.uint8)
    if qs.shape[-1] != QK_IQ4_NL // 2:
        raise ValueError(f"IQ4_NL nibble array must be {QK_IQ4_NL // 2} wide, got {qs.shape[-1]}")
    table = kvalues()
    low = table[(qs & 0x0F).astype(np.intp)]
    high = table[(qs >> 4).astype(np.intp)]
    return np.concatenate([low, high], axis=-1).astype(np.float32)


def blocks_per_row(row_elems: int) -> int:
    """Blocks in a row of ``row_elems`` weights, rounding up."""
    return (int(row_elems) + QK_IQ4_NL - 1) // QK_IQ4_NL


def fold_to_runtime_span(blocks, row_elems: int):
    """Regroup native 18-byte blocks into the runtime's 256-weight row element.

    ``blocks`` arrives as ``(..., row_elems // 32, 18)`` -- the file's own
    geometry, which every decoder in this module is written against.  The
    raw-block kernels want ``(..., row_elems // 256, 144)``, eight native blocks
    end to end, because that is what their block walk assumes of all eleven
    formats.  Both shapes hold the same bytes in the same order, so this
    rearranges nothing; it is a view for a contiguous input and a copy for a
    strided one.  It works on numpy and torch arrays alike and is used on both --
    the decoders here take numpy, the loader hands it a torch tensor.

    ``row_elems`` must be a multiple of 256: a row that is not would leave a
    partial runtime block, and the kernels index the next output row by
    ``blocks_per_row * block_bytes`` rather than by an element cursor, so a
    partial one would silently shift every row after it.  Both of Xing4.0's
    expert widths (3584 and 1024) are multiples, and the check is here rather
    than in a caller because the failure it prevents is a wrong answer.
    """
    span = QK_IQ4_NL * 8
    if int(row_elems) % span:
        raise ValueError(
            f"IQ4_NL row of {row_elems} weights is not a multiple of the runtime's "
            f"{span}-weight span, so its raw blocks cannot be regrouped"
        )
    if blocks.shape[-2] != blocks_per_row(row_elems):
        raise ValueError(
            f"IQ4_NL block grid {(blocks.shape[-1], blocks.shape[-2])} does not match "
            f"a {row_elems}-weight row at {QK_IQ4_NL} weights per block"
        )
    return blocks.reshape(*blocks.shape[:-2], int(row_elems) // span, span // QK_IQ4_NL * IQ4_NL_BLOCK_BYTES)


def _f16_to_f32(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data).view("<f2").astype(np.float32).reshape(data.shape[:-1])
