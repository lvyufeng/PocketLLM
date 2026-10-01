"""Rotary position embedding.

Declared as one op with the layout fork left to ``attrs`` rather than two ops,
because the two layouts (interleaved vs. split-halves) are the same computation
on a different permutation, and a backend that knows one usually knows both.

``positions`` is an integer tensor so a decode-with-capture path can pass a
device-resident position without a host round-trip.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

ROPE = OpSchema(
    name="rope",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("tokens", "heads", "d")),
        ArgSpec("positions", Kind.TENSOR, shape=("tokens",), dtype=DType.I32),
        ArgSpec("cos", Kind.TENSOR, shape=("capacity", "d/2")),
        ArgSpec("sin", Kind.TENSOR, shape=("capacity", "d/2")),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("tokens", "heads", "d")),),
    dtypes=_FLOAT,
    attrs=("layout", "theta_base", "scaling"),
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="rotate adjacent (or split-half) pairs of x by positions through cos/sin",
)

SCHEMAS = (ROPE,)