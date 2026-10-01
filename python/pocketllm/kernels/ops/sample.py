"""Logit transforms and sampling.

``argmax`` and ``topk_sample`` return token ids, so their return carries an
explicit ``I32`` dtype -- the one place a return's type differs from the op's
input dtype.  Splitting temperature out of the sampler lets a greedy decode skip
the transform entirely, which is the path a captured decode step wants.
"""

from __future__ import annotations

from ..dtypes import DType
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

_TEMPERATURE = OpSchema(
    name="logits_temperature",
    args=(ArgSpec("logits", Kind.TENSOR, shape=("vocab",)),),
    returns=(ArgSpec("out", Kind.TENSOR, shape=("vocab",)),),
    dtypes=_FLOAT,
    attrs=("temperature",),
    shape_rule=lambda shapes, attrs: [shapes["logits"]],
    semantics="logits / temperature",
)

_ARGMAX = OpSchema(
    name="argmax",
    args=(ArgSpec("logits", Kind.TENSOR, shape=("vocab",)),),
    returns=(ArgSpec("token", Kind.TENSOR, shape=(), dtype=DType.I32),),
    dtypes=_FLOAT,
    shape_rule=lambda shapes, attrs: [()],
    semantics="the index of the largest logit",
)

_TOPK_SAMPLE = OpSchema(
    name="topk_sample",
    args=(
        ArgSpec("logits", Kind.TENSOR, shape=("vocab",)),
        ArgSpec("uniform", Kind.TENSOR, shape=(), dtype=DType.F32),
    ),
    returns=(ArgSpec("token", Kind.TENSOR, shape=(), dtype=DType.I32),),
    dtypes=_FLOAT,
    attrs=("top_k", "top_p", "min_p"),
    shape_rule=lambda shapes, attrs: [()],
    semantics="draw one token from the top-k/top-p truncated distribution, given a uniform variate",
)

SCHEMAS = (_TEMPERATURE, _ARGMAX, _TOPK_SAMPLE)