"""Token embedding lookup.

Separate from ``gemm_quant`` because the operation is a gather, not a product --
the whole point is that it reads ``tokens`` rows and multiplies nothing, which is
what makes an embedding cheap on a bandwidth-bound decode even when the table is
quantized.
"""

from __future__ import annotations

from ..dtypes import DType
from ..dtypes import QUANT_FORMATS
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

EMBEDDING = OpSchema(
    name="embedding",
    args=(
        ArgSpec("tokens", Kind.TENSOR, shape=("tokens",), dtype=DType.I32),
        ArgSpec("table", Kind.TENSOR, shape=("vocab", "d"), role="embedding table"),
    ),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("tokens", "d")),),
    dtypes=_FLOAT,
    quants=frozenset(QUANT_FORMATS.values()),
    shape_rule=lambda shapes, attrs: [(shapes["tokens"][0], shapes["table"][1])],
    semantics="out[i, :] = table[tokens[i], :]",
)

SCHEMAS = (EMBEDDING,)