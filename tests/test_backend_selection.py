from __future__ import annotations

import json

import pytest

from pocketllm.api import EngineArgs, UnsupportedFeatureError
from pocketllm.backends import factory


def _write_config(directory, payload) -> None:
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_explicit_backend_is_never_rewritten():
    assert factory.select_backend(EngineArgs(model="missing", backend="torch")) == "torch"
    assert factory.select_backend(EngineArgs(model="missing", backend="cpp")) == "cpp"


def test_auto_selects_cpp_only_for_qwen35_with_native_module(tmp_path, monkeypatch):
    _write_config(tmp_path, {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "cpp"


def test_auto_falls_back_to_torch_without_native_module(tmp_path, monkeypatch):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: False))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_keeps_torch_for_other_qwen_generations(tmp_path, monkeypatch):
    # The native public adapter implements Qwen3.5 only, so an older Qwen
    # checkpoint must not be routed to it just because the name matches.
    _write_config(tmp_path, {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_keeps_torch_for_gguf_checkpoints(tmp_path, monkeypatch):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_explicit_cpp_rejects_gguf_before_loading_native(tmp_path):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    args = EngineArgs(model=str(tmp_path), backend="cpp", model_format="gguf")
    with pytest.raises(UnsupportedFeatureError, match="GGUF"):
        factory.select_backend(args)


def test_explicit_cpp_rejects_non_qwen35_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v4"})
    args = EngineArgs(model=str(tmp_path), backend="cpp")
    with pytest.raises(UnsupportedFeatureError, match="Qwen3.5"):
        factory.select_backend(args)


def test_auto_detects_nested_text_config(tmp_path, monkeypatch):
    _write_config(tmp_path, {"model_type": "qwen3_5_moe_vl", "text_config": {"model_type": "qwen3_5_text"}})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "cpp"
