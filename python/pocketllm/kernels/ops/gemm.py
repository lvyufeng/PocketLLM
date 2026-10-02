"""Matrix products, dense and quantized.

The quantized product is where the width ladder is expressed: ``gemm_quant``
takes a packed weight and a float activation and produces a float result, and
the set of ``quants`` it admits is the set of formats a kernel can consume
(Q4 → Q2 → IQ2 → IQ1 → ternary).  A format with no kernel is simply absent from
a backend's capability, and dispatch then refuses the call by name.
"""

from __future__ import annotations

from ..dtypes import DType
from ..dtypes import QUANT_FORMATS
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

#: Every format the ABI names, including the ones no kernel consumes yet -- the
#: schema states what the *op* admits; a backend states what *it* implements, and
#: the gap between the two is exactly what dispatch reports as a refusal.
_ALL_QUANTS = frozenset(QUANT_FORMATS.values())

GEMM = OpSchema(
    name="gemm",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("*", "k")),
        ArgSpec("w", Kind.TENSOR, shape=("n", "k"), role="weight"),
        ArgSpec("bias", Kind.TENSOR, shape=("n",), optional=True),
    ),
    returns=(ArgSpec("y", Kind.TENSOR, shape=("*", "n")),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [(shapes["x"][0], shapes["w"][0])],
    semantics="y[r, j] = sum_k x[r, k] * w[j, k] + bias[j]",
)

GEMM_QUANT = OpSchema(
    name="gemm_quant",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("*", "k")),
        ArgSpec("w_blocks", Kind.TENSOR, shape=("n", "k"), role="quantized weight"),
        ArgSpec("bias", Kind.TENSOR, shape=("n",), optional=True),
    ),
    returns=(ArgSpec("y", Kind.TENSOR, shape=("*", "n")),),
    dtypes=_FLOAT,
    quants=_ALL_QUANTS,
    shape_rule=lambda shapes, attrs: [(shapes["x"][0], shapes["w_blocks"][0])],
    semantics="as gemm, with w_blocks read through its QuantFormat's decoder",
)

SCHEMAS = (GEMM, GEMM_QUANT)