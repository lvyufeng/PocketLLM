"""DeepSeek-V4.1-Flash model-shape adapter layer.

The configuration schema and a pure-PyTorch op layer live here so far -- no weight
loading and no execution path. See `config`, `kernels`, and
`docs/models/deepseek-v4.1-flash.md`.

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
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
