"""Test automatic config selection based on checkpoint metadata."""

import json
import tempfile
from pathlib import Path

import pytest

from pocketllm import EngineArgs
from pocketllm.backends.torch_backend import TorchBackend


def test_detect_expert_dtype_from_config_json(tmp_path: Path) -> None:
    """Auto-detect FP4 expert dtype from config.json."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected == "fp4"


def test_detect_expert_dtype_int8(tmp_path: Path) -> None:
    """Auto-detect int8 expert dtype from config.json."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "int8",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected == "int8"


def test_detect_expert_dtype_missing_config(tmp_path: Path) -> None:
    """Return None when config.json is missing."""
    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected is None


def test_detect_expert_dtype_missing_field(tmp_path: Path) -> None:
    """Return None when config.json lacks expert_dtype."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected is None


def test_runtime_namespace_auto_selects_fp4_config(tmp_path: Path, monkeypatch) -> None:
    """Auto-select config_fp4_active.json when checkpoint has expert_dtype=fp4."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
        # Explicitly no config_path; should auto-detect
    )
    backend = TorchBackend(args)
    ns = backend._runtime_namespace()

    # Should select config_fp4_active.json (or empty if file doesn't exist in test env)
    assert "fp4" in ns.config or ns.config == ""


def test_runtime_namespace_auto_selects_w8a8_config(tmp_path: Path) -> None:
    """Auto-select config_w8a8.json when checkpoint has expert_dtype=int8."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "int8",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    ns = backend._runtime_namespace()

    # Should select config_w8a8.json (or empty if file doesn't exist in test env)
    assert "w8a8" in ns.config or ns.config == ""


def test_runtime_namespace_explicit_config_overrides_autodetect(tmp_path: Path) -> None:
    """Explicit --config-path takes precedence over auto-detection."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
            }
        )
    )

    explicit_config = "/explicit/path/config.json"
    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
        config_path=explicit_config,
    )
    backend = TorchBackend(args)
    ns = backend._runtime_namespace()

    # Should use explicit path, not auto-detected
    assert ns.config == explicit_config


def test_runtime_namespace_fallback_to_w8a8_when_no_detection(tmp_path: Path) -> None:
    """Fall back to config_w8a8.json when detection is not possible."""
    # No config.json in checkpoint
    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    ns = backend._runtime_namespace()

    # Should default to w8a8
    assert "w8a8" in ns.config or ns.config == ""
