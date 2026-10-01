"""The Qualcomm phone-NPU backend: Hexagon through the QNN SDK.

This is the primary v1 phone target, and the reason the ABI splits offline
compilation from device execution.  A QNN graph is not built at runtime: a
*host* toolchain (x86) compiles it to a serialised context binary, and the
*device* (arm64, Android) loads that binary through ``libQnnHtp*.so`` and runs
it.  The compile host and the run host are different machines, so an ABI that
assumed they were the same would be unusable here.

That is what :meth:`compile_spec` and :meth:`GraphCapability` describe
separately: ``AOT_COMPILE`` because the artifact is produced ahead of time, and
``GRAPH`` granularity because the QNN runtime owns the whole forward pass once
the context is loaded -- the host is out of the loop between steps.

The declared op set is int8/int4-first because HTP wants quantized weights; f16
exists but is the slower path.  ``available`` checks for the QNN libraries *and*
the ``/dev`` node, and it must not import ``qai_appbuilder``: on a host that has
the SDK installed but no DSP, importing it can hang.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import CompileSpec, GraphCapability, GraphMode, RegionGranularity

#: The HTP backend library is the fast path; the CPU fallback library alone is
#: not enough to call this backend available.  ``find_library`` reads the dynamic
#: loader's cache, so this costs a directory read, not a dlopen.
_PROBE = RuntimeProbe(
    libraries=("QnnHtp", "QnnHtpV73Stub", "QnnHtpV75Stub"),
    device_nodes=("/dev/fastrpc-cdsp", "/dev/adsprpc-smd"),
    env=("QNN_SDK_ROOT", "QNN_LIB_PATH"),
)


class QnnBackend(StubBackend):
    name = "qnn"
    device_kind = "qnn"
    version = "0.0.0"
    summary = "Qualcomm Hexagon NPU (QNN/HTP, AOT context binary)"
    missing_dependency = "the QNN SDK (libQnnHtp*.so) and a Hexagon DSP device node"
    probe = _PROBE

    op_table = (
        # f16 dense product, for the pieces HTP will take in float.
        ("gemm", ("f16", "f32"), ()),
        # The quantized weights: HTP's 4-bit and 8-bit block schemes.
        ("gemm_quant", ("f16", "f32"), ("q4_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("moe_ffn", ("f16", "f32"), ("q4_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("attention", ("f16", "f32"), ()),
        ("rms_norm", ("f16", "f32"), ()),
        ("layer_norm", ("f16", "f32"), ()),
        ("rope", ("f16", "f32"), ()),
        ("silu_mul", ("f16", "f32"), ()),
        ("add", ("f16", "f32"), ()),
        ("mul", ("f16", "f32"), ()),
        ("softmax", ("f16", "f32"), ()),
        ("embedding", ("f16", "f32"), ("q4_k", "q8_0", "iq4_nl", "iq4_xs")),
        ("cache_append", ("f16", "f32"), ()),
        ("cache_truncate", ("f16", "f32"), ()),
        # Sampling stays on the host: a token id is something the host must see,
        # and putting it inside a compiled graph would mean a per-step sync.
    )

    graph_capability = GraphCapability(
        supported=True,
        mode=GraphMode.AOT_COMPILE,
        captures=frozenset(
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
        ),
        granularity=RegionGranularity.GRAPH,
        rebuild_on_shape_change=True,
    )

    compile_spec_ = CompileSpec(
        host_toolchain="qnn-context-binary-generator",
        target_arch="aarch64-android",
        artifact_format="qnn_context_binary",
        options={
            # The HTP version is what the context binary is compiled *for*, and
            # it is a target property, not a host one -- hence a compile option
            # rather than something the probe could discover.
            "htp_arch": "v73",
            "weight_bits": 4,
        },
    )


BACKEND = QnnBackend()