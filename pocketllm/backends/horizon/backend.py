"""The Horizon 地瓜 S600 backend: the D-Robotics BPU.

Structurally the same as the QNN backend -- a host toolchain compiles a model
ahead of time and an arm64 board loads the artifact through ``libhbrt4.so`` --
and it is deliberately the *second* backend of that shape, because one AOT
backend could be a special case and two establish the pattern the ABI has to
support.

The artifact here is a ``.hbm`` model produced by ``hb_mapper`` /
``hb_compile`` from the Horizon OpenExplorer toolchain; at run time the board
loads it into a BPU model and runs a forward pass over it.  The compile stage is
where the quantization actually happens for this device -- BPU wants int8 with
per-channel scales -- which is why its :class:`CompileSpec` carries the
calibration options rather than the run-time ones.
"""

from __future__ import annotations

from pocketllm.backends.base import RuntimeProbe, StubBackend
from pocketllm.kernels.backend import CompileSpec, GraphCapability, GraphMode, RegionGranularity

#: ``libhbrt4.so`` is the run-time library; ``libhbipm.so`` is the model loader.
#: Both are on the board, and neither exists on a development host, which is the
#: point: the probe must answer "no" here without failing.
_PROBE = RuntimeProbe(
    libraries=("hbrt4", "hbipm", "hbtl"),
    device_nodes=("/dev/bpu", "/dev/hbmem"),
    env=("HORIZON_SDK_ROOT", "HB_SDK_ROOT"),
)


class HorizonBackend(StubBackend):
    name = "horizon"
    device_kind = "horizon"
    version = "0.0.0"
    summary = "Horizon 地瓜 S600 BPU (OpenExplorer .hbm, AOT)"
    missing_dependency = "the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device"
    probe = _PROBE

    op_table = (
        ("gemm", ("f16", "f32"), ()),
        # int8 is the BPU's native width; 4-bit block weights are reached by
        # this backend's own decoder rather than by the runtime's.
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
        host_toolchain="hb_compile",
        target_arch="aarch64",
        artifact_format="hbm",
        options={
            "march": "nash",
            "calibration": "max",
            "advice": 0,
        },
    )


BACKEND = HorizonBackend()