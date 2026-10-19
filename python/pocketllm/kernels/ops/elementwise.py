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

_GELU = OpSchema(
    name="gelu",
    args=(ArgSpec("x", Kind.TENSOR, shape=("m", "n")),),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("m", "n")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
)

#: The exact-erf GELU, `0.5*x*(1+erf(x/sqrt(2)))`.
#:
#: A *separate op* rather than a flag on `gelu`, because the two are different
#: functions and the difference is not a rounding one: they disagree by about 0.01
#: around |x| = 1, which survives a tower's twenty-four blocks.  llama.cpp carries
#: them as separate `ffn_op` values for the same reason -- a vision tower asks for
#: the tanh form and a `qwen3a` audio tower asks for this one -- and a checkpoint
#: that named its activation and got the other would be a different model.
_GELU_ERF = OpSchema(
    name="gelu_erf",
    args=(ArgSpec("x", Kind.TENSOR, shape=("m", "n")),),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("m", "n")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="out = 0.5 * x * (1 + erf(x / sqrt(2)))",
)

SCHEMAS = (_ADD, _MUL, _SOFTMAX, _GELU, _GELU_ERF)
