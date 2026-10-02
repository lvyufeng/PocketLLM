"""The one table from a GGUF type name to its block geometry and decoder.

Both the loader and the reference backend need the same three facts about a
format: how many weights a block holds, how many bytes it occupies, and how to
turn one block into float32.  Stating them once here is what keeps the two from
drifting, and a format with a geometry but no decoder is a visible hole rather
than a silent fall-through to the wrong branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import iq1, iq23, iq4_nl, iq4_xs, k_quants

__all__ = ["BlockFormat", "FORMATS", "format_for", "dequantize_row", "KNOWN_TYPES"]

Decoder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class BlockFormat:
    """Geometry and decoder for one block format."""

    name: str
    block_elems: int
    block_bytes: int
    decode: Decoder | None
    #: ``True`` when the loader addresses the raw blocks and nothing interprets
    #: them yet -- the ternary packs.  A name here is *not* a claim a kernel
    #: consumes it, only that its bytes can be read off the file.
    addressable_only: bool = False


FORMATS: dict[str, BlockFormat] = {
    "q2_k": BlockFormat("q2_k", 256, k_quants.Q2_K_BLOCK_BYTES, k_quants.dequantize_q2_k),
    "q3_k": BlockFormat("q3_k", 256, k_quants.Q3_K_BLOCK_BYTES, k_quants.dequantize_q3_k),
    "q4_k": BlockFormat("q4_k", 256, k_quants.Q4_K_BLOCK_BYTES, k_quants.dequantize_q4_k),
    "q5_k": BlockFormat("q5_k", 256, k_quants.Q5_K_BLOCK_BYTES, k_quants.dequantize_q5_k),
    "q6_k": BlockFormat("q6_k", 256, k_quants.Q6_K_BLOCK_BYTES, k_quants.dequantize_q6_k),
    "q8_0": BlockFormat("q8_0", k_quants.QK8_0, k_quants.Q8_0_BLOCK_BYTES, k_quants.dequantize_q8_0),
    "iq2_xxs": BlockFormat("iq2_xxs", 256, iq23.IQ2_XXS_BLOCK_BYTES, iq23.dequantize_iq2_xxs),
    "iq2_xs": BlockFormat("iq2_xs", 256, iq23.IQ2_XS_BLOCK_BYTES, iq23.dequantize_iq2_xs),
    "iq3_xxs": BlockFormat("iq3_xxs", 256, iq23.IQ3_XXS_BLOCK_BYTES, iq23.dequantize_iq3_xxs),
    "iq1_m": BlockFormat("iq1_m", iq1.QK_IQ1, iq1.IQ1_M_BLOCK_BYTES, iq1.dequantize_blocks),
    "iq4_nl": BlockFormat("iq4_nl", iq4_nl.QK_IQ4_NL, iq4_nl.IQ4_NL_BLOCK_BYTES, iq4_nl.dequantize_blocks),
    "iq4_xs": BlockFormat("iq4_xs", k_quants.QK_K, iq4_xs.IQ4_XS_BLOCK_BYTES, iq4_xs.dequantize_iq4_xs),
    # The fork-private ternary packs: addressable, and deliberately not decoded.
    "ptq1_0": BlockFormat("ptq1_0", 128, 28, None, addressable_only=True),
    "pq2_0": BlockFormat("pq2_0", 128, 34, None, addressable_only=True),
}

#: Every name :data:`FORMATS` knows, which is every format the loader can at
#: least address.
KNOWN_TYPES = frozenset(FORMATS)


def format_for(type_name: str) -> BlockFormat:
    try:
        return FORMATS[type_name]
    except KeyError as exc:
        raise NotImplementedError(f"no block geometry for quant type {type_name}") from exc


def dequantize_row(type_name: str, blocks: np.ndarray, row_elems: int) -> np.ndarray:
    """Decode blocks shaped ``(..., blocks_per_row, block_bytes)`` to ``(..., row_elems)``.

    A GGUF row is stored as whole blocks, so its decoded length is
    ``blocks_per_row * block_elems``; the row's own width may be smaller when it
    is not a multiple of the block size.  Trimming here means callers never see
    the padding.
    """
    fmt = format_for(type_name)
    if fmt.decode is None:
        raise NotImplementedError(f"{type_name} blocks are addressable but have no decoder")
    values = fmt.decode(blocks)
    lead = values.shape[:-1]
    return values.reshape(*lead[:-1], lead[-1] * fmt.block_elems)[..., :row_elems]