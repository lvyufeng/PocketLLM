from __future__ import annotations

import importlib

import pytest


@pytest.fixture(scope="module")
def native_module():
    try:
        return importlib.import_module("pocketllm_cpp")
    except ImportError as exc:
        pytest.skip(f"native pocketllm_cpp module is not built: {exc}")


def test_native_value_types_and_enum(native_module):
    assert native_module.backend in {"cuda", "ascend"}
    options = native_module.QwenEngineOptions()
    assert options.tp_world == 1
    assert options.tp_rank == 0
    assert native_module.qwen_kv_cache_dtype_name(
        native_module.QwenKvCacheDType.Fp16
    ) == "fp16"
    options.kv_cache_dtype = native_module.QwenKvCacheDType.Fp8
    assert options.kv_cache_dtype == native_module.QwenKvCacheDType.Fp8


def test_native_result_and_sampling_defaults(native_module):
    result = native_module.QwenForwardResult()
    assert result.top_token == 0
    assert result.as_dict()["accept_tokens"] == []
    sampling = native_module.SamplingParams()
    assert sampling.greedy is True
    assert sampling.temperature == 1.0
    smoke = native_module.ForwardSmokeOptions()
    assert smoke.as_dict()["skip_fp4_host_prepare"] is False


def test_the_scheduler_is_bound_to_the_engine_interface_not_to_one_engine(native_module):
    """`QwenBatchScheduler` is a library with as many hosts as there are engines, so what it accepts
    is `InferenceEngine`, not the implementation that happened to exist first. The distinction is
    the whole reason the interface exists: a second engine reaches the same scheduler through the
    same call, with no second scheduler and no binding change.

    A signature check rather than a construction: building a real engine needs a checkpoint and a
    card, and what would break here is the type the constructor declares.
    """
    interface = getattr(native_module, "InferenceEngine", None)
    assert interface is not None, (
        "the binding does not expose InferenceEngine, so nothing but the one concrete engine can "
        "be handed to the scheduler"
    )
    assert issubclass(native_module.QwenEngine, interface)

    # The parameter's declared type, read off the docstring the way pybind11 renders it, is the
    # only place the signature is observable without an engine to pass.
    doc = native_module.QwenBatchScheduler.__init__.__doc__ or ""
    assert "InferenceEngine" in doc, doc


def test_the_scheduler_stats_carry_the_paged_block_gauges(native_module):
    """The Python host publishes the scheduler's admission state under the metric names the native
    host already uses, and the paged block gauges are part of that: without them a paged deployment
    has no way to read the pool it is being admitted against from either server."""
    stats = native_module.QwenBatchSchedulerStats()
    for field in ("reserved_blocks", "total_blocks", "free_blocks", "cache_pinned_blocks"):
        assert getattr(stats, field) == 0
