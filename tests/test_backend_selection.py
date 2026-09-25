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


def test_explicit_v41_is_never_rewritten():
    assert factory.select_backend(EngineArgs(model="missing", backend="v41")) == "v41"


def test_auto_selects_v41_for_a_v41_checkpoint(tmp_path, monkeypatch):
    # The native adapter has no factory for this architecture, so the auto path must
    # reach the V4.1 adapter even where the native module is available.
    _write_config(tmp_path, {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "v41"


def test_auto_selects_v41_from_the_nested_text_config(tmp_path):
    # The released checkpoint nests model_type at the root and repeats it inside
    # text_config; the root value alone is the wrapper's, and both name V4.1.
    _write_config(
        tmp_path,
        {"model_type": "deepseek_v41", "text_config": {"model_type": "deepseek_v41_text"}},
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "v41"


def test_auto_keeps_torch_for_a_non_v41_deepseek(tmp_path, monkeypatch):
    _write_config(tmp_path, {"model_type": "deepseek_v4"})
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_keeps_torch_for_a_gguf_v41_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_explicit_v41_rejects_gguf_checkpoints(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    args = EngineArgs(model=str(tmp_path), backend="v41", model_format="gguf")
    with pytest.raises(UnsupportedFeatureError, match="GGUF"):
        factory.select_backend(args)


def test_explicit_v41_rejects_a_foreign_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    args = EngineArgs(model=str(tmp_path), backend="v41")
    with pytest.raises(UnsupportedFeatureError, match="DeepSeek-V4.1-Flash"):
        factory.select_backend(args)


def _write_gguf_header(path, architecture: str) -> None:
    """A GGUF with a header and no tensors: enough for the registry to classify.

    ``detect_architecture`` reads one metadata field -- ``general.architecture``
    -- so a file that declares only that is a checkpoint as far as the routing
    question is concerned, and it costs a few bytes instead of 5.9 GiB.
    """
    import struct

    key = b"general.architecture"
    value = architecture.encode("utf-8")
    blob = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    blob += struct.pack("<Q", len(key)) + key
    blob += struct.pack("<I", 8)  # GGUF metadata type: string
    blob += struct.pack("<Q", len(value)) + value
    path.write_bytes(blob)


def test_auto_selects_cpp_for_a_qwen35_gguf(tmp_path, monkeypatch):
    _write_gguf_header(tmp_path / "model.gguf", "qwen35")
    monkeypatch.setattr(factory.CppBackend, "native_available", staticmethod(lambda: True))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "cpp"
    # Named as a file rather than a directory, which is how the native CLI takes
    # it, and the answer does not depend on which spelling the caller used.
    assert (
        factory.select_backend(
            EngineArgs(model=str(tmp_path / "model.gguf"), backend="auto")
        )
        == "cpp"
    )


def test_explicit_cpp_accepts_a_qwen35_gguf(tmp_path):
    _write_gguf_header(tmp_path / "model.gguf", "qwen35")
    args = EngineArgs(model=str(tmp_path), backend="cpp")
    assert factory.select_backend(args) == "cpp"


def test_cpp_refuses_a_gguf_of_another_architecture(tmp_path):
    _write_gguf_header(tmp_path / "model.gguf", "llama")
    args = EngineArgs(model=str(tmp_path), backend="cpp")
    with pytest.raises(UnsupportedFeatureError, match="GGUF"):
        factory.select_backend(args)


def test_cpp_refuses_a_directory_that_names_two_models(tmp_path):
    # Two files, neither a shard of the other: serving either one would be a
    # guess, so the refusal names the format instead of picking by sort order.
    _write_gguf_header(tmp_path / "a.gguf", "qwen35")
    _write_gguf_header(tmp_path / "b.gguf", "qwen35")
    args = EngineArgs(model=str(tmp_path), backend="cpp")
    with pytest.raises(UnsupportedFeatureError, match="GGUF"):
        factory.select_backend(args)


def test_gguf_checkpoint_file_reads_a_header():
    from pocketllm.backends.cpp_backend import gguf_checkpoint_file

    assert gguf_checkpoint_file("/nonexistent/model.gguf") == ""
    assert gguf_checkpoint_file("") == ""
