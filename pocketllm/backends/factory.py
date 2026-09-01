"""Backend selection and construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pocketllm.api import BackendUnavailableError, EngineArgs, UnsupportedFeatureError

from .cpp_backend import CppBackend
from .torch_backend import TorchBackend


_QWEN35_TYPES = {"qwen3_5", "qwen3_5_text"}


def _config_path(path: str, explicit: str | None = None) -> Path:
    candidate = Path(explicit) if explicit else Path(path) / "config.json"
    if candidate.is_dir():
        candidate = candidate / "config.json"
    return candidate


def _read_config(path: str, explicit: str | None = None) -> dict[str, Any] | None:
    try:
        config = _config_path(path, explicit)
        if not config.is_file():
            return None
        with config.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _is_qwen35_config(config: dict[str, Any]) -> bool:
    def is_qwen35(value: Any) -> bool:
        return str(value or "").lower() in _QWEN35_TYPES

    if is_qwen35(config.get("model_type")):
        return True
    architectures = config.get("architectures", ())
    if isinstance(architectures, (list, tuple)):
        if any("qwen3_5" in str(item).lower() for item in architectures):
            return True
    nested = config.get("text_config")
    return isinstance(nested, dict) and _is_qwen35_config(nested)


def _looks_like_qwen35(path: str, config_path: str | None = None) -> bool:
    config = _read_config(path, config_path)
    return config is not None and _is_qwen35_config(config)


def _checkpoint_has_gguf(path: str) -> bool:
    """Return whether the requested checkpoint is visibly a GGUF model."""
    try:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.suffix.lower() == ".gguf"
        if not candidate.is_dir():
            return False
        return any(candidate.glob("*.gguf"))
    except OSError:
        return False


def _cpp_model_supported(args: EngineArgs) -> bool:
    if args.model_format == "gguf":
        return False
    if args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir):
        return False
    return _looks_like_qwen35(args.checkpoint_dir, args.config_path)


def _reject_unsupported_cpp_checkpoint(args: EngineArgs) -> None:
    """Fail fast for checkpoints the native adapter provably cannot serve.

    A missing or unreadable path is left alone so injected engines and unusual
    layouts still reach the native loader, which reports the precise error.
    """
    if args.model_format == "gguf" or (
        args.model_format == "auto" and _checkpoint_has_gguf(args.checkpoint_dir)
    ):
        raise UnsupportedFeatureError(
            "the native C++ adapter supports Qwen3.5 safetensors only; "
            "GGUF checkpoints must use backend='torch'"
        )
    config = _read_config(args.checkpoint_dir, args.config_path)
    if config is not None and not _is_qwen35_config(config):
        raise UnsupportedFeatureError(
            "the native C++ adapter supports Qwen3.5 checkpoints only"
        )


def select_backend(args: EngineArgs) -> str:
    """Select a backend without silently changing an explicit user choice."""
    if args.backend != "auto":
        if args.backend == "cpp":
            _reject_unsupported_cpp_checkpoint(args)
        return args.backend
    if CppBackend.native_available() and _cpp_model_supported(args):
        return "cpp"
    return "torch"


def create_backend(args: EngineArgs, **injected: Any):
    """Construct the selected adapter.

    ``injected`` is intentionally useful for tests and embedding applications;
    production callers normally only pass ``EngineArgs``.
    """
    selected = select_backend(args)
    if selected == "torch":
        return TorchBackend(
            args,
            runtime=injected.get("runtime"),
            serving_engine=injected.get("serving_engine"),
            runtime_loader=injected.get("runtime_loader"),
        )
    if selected == "cpp":
        return CppBackend(
            args,
            native_module=injected.get("native_module"),
            engine=injected.get("engine"),
            tokenizer=injected.get("tokenizer"),
        )
    raise BackendUnavailableError(f"unsupported backend {selected!r}")
