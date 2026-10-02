"""A mixture-of-experts feed-forward block.

One op rather than a scatter/gather pair because every backend that has a fast
MoE path fuses the routing: the expert ids are an *input*, not a control-flow
decision, so a backend can group the rows however its hardware wants.  The three
weight tensors are the gate/up/down projections of the whole expert bank, packed
in whatever quant format the checkpoint shipped.
"""

from __future__ import annotations

from ..dtypes import DType
from ..dtypes import QUANT_FORMATS
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})
_ALL_QUANTS = frozenset(QUANT_FORMATS.values())

MOE_FFN = OpSchema(
    name="moe_ffn",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("tokens", "hidden")),
        ArgSpec("expert_ids", Kind.TENSOR, shape=("tokens", "top_k"), dtype=DType.I32),
        ArgSpec("expert_weights", Kind.TENSOR, shape=("tokens", "top_k")),
        ArgSpec("w1", Kind.TENSOR, role="gate/up weight"),
        ArgSpec("w2", Kind.TENSOR, role="down weight"),
        ArgSpec("shared_w1", Kind.TENSOR, optional=True),
        ArgSpec("shared_w2", Kind.TENSOR, optional=True),
    ),
    returns=(ArgSpec("y", Kind.TENSOR, shape=("tokens", "hidden")),),
    dtypes=_FLOAT,
    quants=_ALL_QUANTS,
    attrs=("top_k", "norm_topk_prob", "swiglu"),
    shape_rule=lambda shapes, attrs: [shapes["x"]],
    semantics="router picks top_k experts per token; their SwiGLU outputs are combined by expert_weights",
)

SCHEMAS = (MOE_FFN,)