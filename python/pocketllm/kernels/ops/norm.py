"""Normalizations and the gated activation.

``rms_norm`` and ``layer_norm`` are separate ops because their eps semantics and
their fused forms differ; ``silu_mul`` is the SwiGLU half that a quantized MLP
fuses into its gate/up projection on every backend that can afford to.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

RMS_NORM = OpSchema(
    name="rms_norm",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("tokens", "d")),
        ArgSpec("weight", Kind.TENSOR, shape=("d",)),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("tokens", "d")),),
    dtypes=_FLOAT,
    attrs=("eps",),
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="x / sqrt(mean(x^2) + eps) * weight",
)

LAYER_NORM = OpSchema(
    name="layer_norm",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("tokens", "d")),
        ArgSpec("weight", Kind.TENSOR, shape=("d",)),
        ArgSpec("bias", Kind.TENSOR, shape=("d",), optional=True),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("tokens", "d")),),
    dtypes=_FLOAT,
    attrs=("eps",),
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="(x - mean) / sqrt(var + eps) * weight + bias",
)

SILU_MUL = OpSchema(
    name="silu_mul",
    args=(
        ArgSpec("gate", Kind.TENSOR, shape=("tokens", "d")),
        ArgSpec("up", Kind.TENSOR, shape=("tokens", "d")),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("tokens", "d")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [shapes["gate"]],
    semantics="silu(gate) * up",
)

SCHEMAS = (RMS_NORM, LAYER_NORM, SILU_MUL)