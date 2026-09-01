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
