"""The CUDA backend: the sm_75 card path, one card per process.

This is the backend the current tree already half-is, so its declaration is the
most complete one -- and the one with an explicit open question attached.  The
kernels could come from ``relic_core`` (the shared torch operator library) or
this backend could grow its own; if it wraps ``relic_core``, then
``relic_core.kernels.ops``'s own resolver is superseded by
:mod:`pocketllm.kernels.dispatch` and the two must not both exist.  That choice is
recorded in docs rather than guessed at here, so the declaration below is
deliberately at the "what", not the "how".

CUDA is the only backend in this tree that declares a *capture* path: a decode
step is a fixed-shape sequence, which is exactly what a stream capture wants.
The captured region is op-level (``granularity=STEP``) because the host still
owns sampling and the KV-cache position between steps.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import GraphCapability, GraphMode, RegionGranularity

#: torch must be installed *and* a card must be present.  The device node is the
#: load-bearing half: torch is a pip install away on any machine, so a module
#: probe alone reports this backend available on a laptop with no GPU -- and
#: "available, then fails on the first allocation" is exactly the failure the
#: probe exists to prevent.  ``/dev/nvidia0`` is a filesystem question, which is
#: the only kind of question a probe may ask.
_PROBE = RuntimeProbe(modules=("torch",), device_nodes=("/dev/nvidia0",))

#: The ops a captured decode step may contain.  Deliberately excludes the
#: sampling ops: they read a host-provided uniform variate and produce a token
#: id the host must see, so a capture that swallowed them would hang the loop.
_CAPTURABLE = frozenset(
    {
        "gemm",
        "gemm_quant",
        "moe_ffn",
        "attention",
        "rms_norm",
        "layer_norm",
        "rope",
        "silu_mul",
        "add",
        "mul",
        "softmax",
        "embedding",
    }
)


class CudaBackend(StubBackend):
    name = "cuda"
    device_kind = "cuda"
    version = "0.0.0"
    summary = "NVIDIA GPU via torch (sm_75 today; one card, one process)"
    missing_dependency = "torch built for a CUDA toolkit matching the driver, and an NVIDIA device node"
    probe = _PROBE

    op_table = (
        ("gemm", ("f32", "f16", "bf16"), ()),
        (
            "gemm_quant",
            ("f32", "f16", "bf16"),
            ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0", "iq1_m", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq4_nl", "iq4_xs"),
        ),
        ("moe_ffn", ("f32", "f16", "bf16"), ("q4_k", "q5_k", "q6_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("attention", ("f32", "f16", "bf16"), ()),
        ("rms_norm", ("f32", "f16", "bf16"), ()),
        ("layer_norm", ("f32", "f16", "bf16"), ()),
        ("rope", ("f32", "f16", "bf16"), ()),
        ("silu_mul", ("f32", "f16", "bf16"), ()),
        ("add", ("f32", "f16", "bf16"), ()),
        ("mul", ("f32", "f16", "bf16"), ()),
        ("softmax", ("f32", "f16", "bf16"), ()),
        ("embedding", ("f32", "f16", "bf16"), ("q4_k", "q5_k", "q6_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("logits_temperature", ("f32", "f16", "bf16"), ()),
        ("argmax", ("f32", "f16", "bf16"), ()),
        ("topk_sample", ("f32", "f16", "bf16"), ()),
        ("cache_append", ("f32", "f16", "bf16"), ()),
        ("cache_truncate", ("f32", "f16", "bf16"), ()),
    )

    graph_capability = GraphCapability(
        supported=True,
        mode=GraphMode.STREAM_CAPTURE,
        captures=_CAPTURABLE,
        granularity=RegionGranularity.STEP,
        max_nodes=0,
    )


BACKEND = CudaBackend()