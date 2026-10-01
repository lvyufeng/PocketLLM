"""A deliberately tiny architecture, kept in the tree on purpose.

Most of this package is a scaffold for models that do not exist yet, which makes
it easy for the scaffold to be *shaped* wrongly without anyone noticing: the
builders compile, the specs verify, and nothing has ever been run.

``toy`` is the thing that notices.  It builds an ordinary graph through the same
:class:`~pocketllm.architectures.ir.GraphBuilder` a real model uses --

    tokens -> embedding -> rms_norm -> gemm(gate) -> silu_mul
                                    -> gemm(up)   /
                                    -> gemm(down)

-- and runs it through the same executor, so shape inference, argument
marshalling and the memory planner are exercised end to end.  If it stops
running on ``reference``, something in the executor-visible contract moved,
whatever the unit tests claim.

The gate and up projections are two weights rather than one fused ``(2*ff,
hidden)`` projection, because ``silu_mul`` takes two tensors of one shape and the
ABI has no slice op -- a backend fuses that split into its own GEMM rather than
paying for a copy.  Two weights keeps the graph honest without inventing an op
the ABI does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import TensorDesc

from .cache import CachePlan
from .ir import GraphBuilder, ModelSpec

__all__ = ["ToyConfig", "build"]


@dataclass(frozen=True, slots=True)
class ToyConfig:
    """The dimensions ``toy`` is built at.  The defaults are for a test, not a model."""

    hidden: int = 8
    ff: int = 16
    vocab: int = 32
    dtype: DType = DType.F32


def build(config: ToyConfig | None = None) -> ModelSpec:
    """Build the toy graph, verified, ready to run on any session."""
    config = config or ToyConfig()
    b = GraphBuilder("toy")
    hidden, ff = config.hidden, config.ff

    tokens = b.input("tokens", TensorDesc((1,), DType.I32))
    table = b.weight("embedding", TensorDesc((config.vocab, hidden), config.dtype), role="token embedding")
    norm_w = b.weight("norm.weight", TensorDesc((hidden,), config.dtype), role="rms norm", quantizable=False)
    gate_w = b.weight("ffn.gate", TensorDesc((ff, hidden), config.dtype), role="gate projection")
    up_w = b.weight("ffn.up", TensorDesc((ff, hidden), config.dtype), role="up projection")
    down_w = b.weight("ffn.down", TensorDesc((hidden, ff), config.dtype), role="down projection")

    h = b.one("embedding", tokens, table, outputs="h", tag="embed")
    normed = b.one("rms_norm", h, norm_w, outputs="normed", attrs={"eps": 1e-6}, tag="norm")
    gate = b.one("gemm", normed, gate_w, outputs="gate", tag="ffn.gate")
    up = b.one("gemm", normed, up_w, outputs="up", tag="ffn.up")
    hidden_state = b.one("silu_mul", gate, up, outputs="hidden", tag="ffn.act")
    y = b.one("gemm", hidden_state, down_w, outputs="y", tag="ffn.down")
    b.output(y)

    spec = b.build()
    # The toy has no attention, so its cache plan is empty rather than absent:
    # a caller that sizes memory for it gets zero, not a special case.
    spec.cache = CachePlan()
    return spec