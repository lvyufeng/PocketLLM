"""The CPU backend: arm64 and x86, no torch, no CUDA.

This is the one backend that must work everywhere, including the phone and the
edge box this tree exists for.  It is also the one where the *portable* ABI
earns its keep: the same declared op set has to run on Neoverse with SVE2,
Cortex-A with KleidiAI, and a Xeon, and the only thing those have in common is
what the ABI already fixed.

The session is not written yet.  What is here is the declaration, and it is
narrower than the reference backend on purpose: the reference backend decodes
every quant format and runs every op because it is the correctness oracle; a
real CPU backend runs the ops it has a fast path for and lets dispatch fall back
for the rest.  Keeping those two sets different is what stops the reference
backend from being mistaken for a CPU backend.

``available`` is True on any host that has numpy, which means this is the one
stub that is *loadable* in CI.  It still refuses to run anything -- loadable and
implemented are different questions, and conflating them is how a stub gets
mistaken for a backend.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import CompileSpec, GraphCapability, RegionGranularity

#: numpy is the only thing this backend needs to be *loadable*: the kernels
#: themselves will be ctypes calls into a BLAS (or KleidiAI on arm64), but a
#: backend with no BLAS is still a backend -- it is just slow.
_PROBE = RuntimeProbe(modules=("numpy",))

#: The quant formats whose blocks the reference decoders can expand on the host.
#: Listed explicitly rather than as "all of them" because this backend will
#: eventually have *real* kernels for a subset, and the declaration is where that
#: becomes visible.
_QUANTISED_WEIGHTS = (
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "q8_0",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq3_xxs",
    "iq4_nl",
    "iq4_xs",
)


class CPUBackend(StubBackend):
    name = "cpu"
    device_kind = "cpu"
    version = "0.0.0"
    summary = "Host CPU (arm64/x86); numpy, optional BLAS/KleidiAI"
    missing_dependency = "numpy"
    probe = _PROBE

    op_table = (
        ("gemm", ("f32", "f16", "bf16"), ()),
        ("gemm_quant", ("f32", "f16", "bf16"), _QUANTISED_WEIGHTS),
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
        # ``moe_ffn`` is deliberately absent: a CPU MoE is a loop over experts
        # with too small a batch to amortise it, so dispatch falls through to the
        # reference backend rather than to a fast path that would not be one.
    )

    #: AOT is *optional* on CPU and only ever a win for a fixed decode shape: the
    #: offline compile is a plain ahead-of-time codegen step, and a backend that
    #: has not written it simply reports no graph path.
    graph_capability = GraphCapability(supported=False, granularity=RegionGranularity.STEP)

    compile_spec_ = CompileSpec(host_toolchain="cc", target_arch="native", artifact_format="object")


BACKEND = CPUBackend()