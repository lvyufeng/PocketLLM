"""Attention over a KV cache.

The cache is passed separately from the query because decode reads the whole
history while writing one row: an attention op is called with ``q`` of shape
``(1, heads, d)`` against a cache of shape ``(capacity, kv_heads, d)``, which is
the shape every backend's fast path is written for.  The cache's layout is
whatever the backend declared; the ABI only fixes that ``cache_append`` writes
it and ``attention`` reads it.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

ATTENTION = OpSchema(
    name="attention",
    args=(
        ArgSpec("q", Kind.TENSOR, shape=("q_len", "heads", "d")),
        ArgSpec("k_cache", Kind.TENSOR, shape=("capacity", "kv_heads", "k_d")),
        ArgSpec("v_cache", Kind.TENSOR, shape=("capacity", "kv_heads", "v_d")),
        ArgSpec("positions", Kind.TENSOR, shape=("q_len",), dtype=DType.I32),
        ArgSpec("mask", Kind.TENSOR, optional=True),
        ArgSpec("window", Kind.INT, optional=True),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("q_len", "heads", "v_d")),),
    dtypes=_FLOAT,
    attrs=("softmax_scale", "causal", "num_kv_heads"),
    shape_rule=lambda shapes, attrs: [(shapes["q"][0], shapes["q"][1], shapes["v_cache"][2])],
    semantics="softmax(q @ k^T * scale + mask) @ v over each query's visible cache",
)

SCHEMAS = (ATTENTION,)