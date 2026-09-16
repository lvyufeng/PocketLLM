"""DeepSeek-V4.1-Flash model and runtime.

The configuration schema, a pure-PyTorch op layer, the attention stack, the backbone
above it and the loader that fills that backbone from the released 48-shard
checkpoint. `load_backbone` is the entry point a caller wants: it maps the
checkpoint, fills the 924-parameter text backbone, and returns something that answers
a forward. See `docs/models/deepseek-v4.1-flash.md` for what is and is not validated.

The names below are re-exported lazily rather than imported here. An eager
`from .config import ...` would put `config` in `sys.modules` before `runpy` gets
to it, so the module's own command line -- `python -m src.models.deepseek_v4_1.config`,
which this repository's documentation quotes -- printed a RuntimeWarning on every
run. Resolving them on first attribute access avoids that and keeps
`from src.models.deepseek_v4_1 import load_config` working unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "V41Config",
    "V41TextConfig",
    "V41VisionConfig",
    "from_dict",
    "load_config",
    "V41Checkpoint",
    "LoadedBackbone",
    "build_hasher",
    "load_backbone",
]

_LAZY = {
    "V41Config": "config",
    "V41TextConfig": "config",
    "V41VisionConfig": "config",
    "from_dict": "config",
    "load_config": "config",
    "V41Checkpoint": "loader",
    "LoadedBackbone": "loader",
    "build_hasher": "loader",
    "load_backbone": "loader",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
