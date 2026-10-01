"""The ABI's op vocabulary, declared one family per module.

Importing this module populates :data:`pocketllm.kernels.registry.OPS` and then
freezes it.  A backend reads ``OPS`` to know what it may implement; a caller
reads it to know what it may ask for.  The list is short on purpose -- an op is
added when a real backend needs it, not in anticipation, and adding one means
adding a reference implementation in the same commit.
"""

from __future__ import annotations

from ..registry import OPS
from . import attention, cache, elementwise, embedding, gemm, moe, norm, rope, sample

_FAMILIES = (gemm, attention, moe, rope, norm, elementwise, embedding, sample, cache)


def declare_all() -> None:
    """Declare every family in :data:`OPS` and freeze the vocabulary."""
    for family in _FAMILIES:
        for schema in family.SCHEMAS:
            OPS.declare(schema)
    OPS.freeze()


declare_all()

__all__ = ["OPS", "declare_all"]