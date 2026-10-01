"""The Apple Silicon backend: Metal through PyTorch's MPS backend.

The one device in the v1 list where the *runtime dependency is torch*.  That is
not a compromise of the torch-free rule -- it is the rule working.  The ABI's
core, the loader and the reference backend import numpy and nothing else; a
backend that finds Metal easiest to reach through torch declares torch as *its*
dependency, and an install that never selects this backend never pulls it in.

MPS reports no graph path.  Metal has no stable graph-capture API exposed through
torch, and inventing one would be worse than running op-by-op: the engine's
eager path is correct, and a capture path that silently produced a different
result would not be.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import GraphCapability, RegionGranularity

#: torch *and* a Metal device are both required, and Metal only exists on macOS.
#: The platform gate is what keeps this backend from claiming availability on the
#: x86_64 CUDA host, where torch is installed and MPS could never run.
_PROBE = RuntimeProbe(modules=("torch",), platforms=("darwin",))


class MPSBackend(StubBackend):
    name = "mps"
    device_kind = "mps"
    version = "0.0.0"
    summary = "Apple Silicon GPU (Metal via torch MPS)"
    missing_dependency = "torch>=2.2 with an MPS device"
    probe = _PROBE

    op_table = (
        ("gemm", ("f32", "f16", "bf16"), ()),
        ("gemm_quant", ("f32", "f16", "bf16"), ("q4_k", "q5_k", "q6_k", "q8_0", "iq4_nl", "iq4_xs")),
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

    graph_capability = GraphCapability(supported=False, granularity=RegionGranularity.STEP)


BACKEND = MPSBackend()