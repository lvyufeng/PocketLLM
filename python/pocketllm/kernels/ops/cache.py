"""KV-cache writes and truncation.

Kept as ops rather than hidden inside ``attention`` so a backend that has a
paged or compressed cache can implement them its own way, and so the engine can
see a cache write as one step of a captured region.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

CACHE_APPEND = OpSchema(
    name="cache_append",
    args=(
        ArgSpec("cache", Kind.TENSOR, shape=("capacity", "heads", "d")),
        ArgSpec("values", Kind.TENSOR, shape=("tokens", "heads", "d")),
        ArgSpec("positions", Kind.TENSOR, shape=("tokens",), dtype=DType.I32),
    ),
    returns=(ArgSpec("cache_out", Kind.TENSOR, shape=("capacity", "heads", "d")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["cache"]],
    semantics="cache[positions[i]] = values[i] (an in-place write, returned for composition)",
)

CACHE_TRUNCATE = OpSchema(
    name="cache_truncate",
    args=(ArgSpec("cache", Kind.TENSOR, shape=("capacity", "heads", "d")),),
    returns=(ArgSpec("cache_out", Kind.TENSOR, shape=("capacity", "heads", "d")),),
    dtypes=_FLOAT,
    attrs=("length",),
    shape_rule=lambda shapes, attrs: [shapes["cache"]],
    semantics="forget everything past `length` (a prefix cache's restore)",
)

SCHEMAS = (CACHE_APPEND, CACHE_TRUNCATE)