"""The Ascend 310B backend: the Orange Pi AIpro / Atlas 200I target.

**310B is not the 910B the host notes in CLAUDE.md describe.** They are different
silicon generations with different CANN support and different capture behaviour,
so the recorded 910B facts do not transfer -- this declaration is written for the
310B and the "which board, which CANN" question is left open rather than guessed.

Unlike the two AOT backends, this one declares *capture* rather than AOT
compilation: CANN's ``aclgraph`` records a stream and replays it, the same shape
as a CUDA graph, which suits a decode loop whose shapes are fixed.  AOT is still
possible through ``atc``/``.om`` -- and is what a 310B deployment often wants --
but it is the second path, not the model the ABI is built around here.

``aclgraph`` differs between 310B and 910B, so the capture declaration is the
part to re-check in place on the actual board.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import CompileSpec, GraphCapability, GraphMode, RegionGranularity

#: CANN's runtime libraries, plus the device nodes the driver creates.  A host
#: with the toolkit installed but no NPU has the libraries and not the node,
#: which is exactly the distinction the probe has to make.
#:
#: ``acl`` is deliberately *not* probed by library name: ``find_library("acl")``
#: resolves to BSD's POSIX ACL library, which is installed on essentially every
#: Linux box, so it would report an Ascend NPU on a machine that has none.  The
#: real gate is ``libascendcl.so`` and the ``/dev`` node.
_PROBE = RuntimeProbe(
    libraries=("ascendcl", "nnopbase"),
    device_nodes=("/dev/davinci0",),
    env=("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"),
)

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


class AscendBackend(StubBackend):
    name = "ascend"
    device_kind = "ascend"
    version = "0.0.0"
    summary = "Ascend 310B (Orange Pi AIpro / Atlas 200I; CANN aclgraph)"
    missing_dependency = "CANN (libascendcl.so) and an Ascend NPU (a /dev/davinci node)"
    probe = _PROBE

    op_table = (
        ("gemm", ("f16", "f32"), ()),
        ("gemm_quant", ("f16", "f32"), ("q4_k", "q5_k", "q6_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("moe_ffn", ("f16", "f32"), ("q4_k", "q5_k", "q6_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("attention", ("f16", "f32"), ()),
        ("rms_norm", ("f16", "f32"), ()),
        ("layer_norm", ("f16", "f32"), ()),
        ("rope", ("f16", "f32"), ()),
        ("silu_mul", ("f16", "f32"), ()),
        ("add", ("f16", "f32"), ()),
        ("mul", ("f16", "f32"), ()),
        ("softmax", ("f16", "f32"), ()),
        ("embedding", ("f16", "f32"), ("q4_k", "q8_0", "iq4_nl", "iq4_xs")),
        # Sampling runs on the host: the token id has to come back for the next
        # step's position anyway, so capturing it would only add a sync.
        ("cache_append", ("f16", "f32"), ()),
        ("cache_truncate", ("f16", "f32"), ()),
    )

    graph_capability = GraphCapability(
        supported=True,
        mode=GraphMode.STREAM_CAPTURE,
        captures=_CAPTURABLE,
        granularity=RegionGranularity.STEP,
        max_nodes=0,
    )

    #: Only used on the ``atc`` path, which is not the declared primary here --
    #: but a backend that *can* be compiled offline should say what that needs
    #: rather than leaving a caller to discover it.
    compile_spec_ = CompileSpec(
        host_toolchain="atc",
        target_arch="ascend310b",
        artifact_format="om",
        options={"soc_version": "Ascend310B1", "precision_mode": "allow_mix_precision"},
    )


BACKEND = AscendBackend()