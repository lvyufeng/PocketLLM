"""Elementwise arithmetic and softmax.

Grouped because they share a shape rule -- output shape equals the first input's
-- and because a backend that vectorizes one almost always vectorizes the rest.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

_ADD = OpSchema(
    name="add",
    args=(
        ArgSpec("a", Kind.TENSOR, shape=("m", "n")),
        ArgSpec("b", Kind.TENSOR, shape=("m", "n")),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("m", "n")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["a"]],
    semantics="out = a + b",
)

_MUL = OpSchema(
    name="mul",
    args=(
        ArgSpec("a", Kind.TENSOR, shape=("m", "n")),
        ArgSpec("b", Kind.TENSOR, shape=("m", "n")),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("m", "n")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["a"]],
    semantics="out = a * b",
)

_SOFTMAX = OpSchema(
    name="softmax",
    args=(ArgSpec("x", Kind.TENSOR, shape=("m", "n")),),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("m", "n")),),
    dtypes=_FLOAT,
    attrs=("axis",),
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="normalize exp(x) along the last axis",
)

SCHEMAS = (_ADD, _MUL, _SOFTMAX)