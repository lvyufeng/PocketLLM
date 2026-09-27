from __future__ import annotations

from pathlib import Path

from src.loader.gguf.bundle import GGUFBundle, read_gguf_bundle
from src.loader.gguf.prewarm import prewarm_bundle as _prewarm_shards
from src.components.moe.spec import MoEModelSpec


_SPECS: dict[str, MoEModelSpec] | None = None


def _init_specs() -> dict[str, MoEModelSpec]:
    from src.models.deepseek_v4.spec import DeepSeekV4Spec
    from src.models.glm_dsa.spec import GLMDSASpec
    from src.models.minimax_m2.spec import MiniMaxM2Spec

    return {
        DeepSeekV4Spec.architecture: DeepSeekV4Spec(),
        GLMDSASpec.architecture: GLMDSASpec(),
        MiniMaxM2Spec.architecture: MiniMaxM2Spec(),
    }


def _specs() -> dict[str, MoEModelSpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = _init_specs()
    return _SPECS


def known_architectures() -> list[str]:
    return sorted(_specs())


def get_spec(architecture: str) -> MoEModelSpec:
    key = architecture.strip().lower()
    try:
        return _specs()[key]
    except KeyError as exc:
        raise ValueError(f"unsupported MoE architecture {architecture!r}; known: {', '.join(known_architectures())}") from exc


def detect_spec(bundle: GGUFBundle, override: str = "auto") -> MoEModelSpec:
    if override != "auto":
        return get_spec(override)
    arch = bundle.metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("GGUF metadata does not contain general.architecture; pass --architecture explicitly")
    return get_spec(arch)


# ---------------------------------------------------------------------------------------------
# The checkpoint-format seam
# ---------------------------------------------------------------------------------------------
#
# Reading a checkpoint is *format* knowledge, not architecture knowledge, so it cannot live on a
# spec: `detect_spec` needs the bundle before it knows which spec to ask, and a second format would
# have to hand every spec a second reading path. It lives here instead, because this module is the
# one thing above `src.loader` that the generation driver is allowed to import --
# `src/runtime/generation.py` must stay both model- and format-agnostic, which is what
# `tests/test_package_boundaries.py::test_runtime_stays_model_and_checkpoint_format_agnostic`
# asserts.
#
# The effect is that `src/runtime/` names no container format at all. Adding one is a branch here,
# and the generation loop below it does not change.


def load_bundle(path: str | Path) -> GGUFBundle:
    """Read a checkpoint into the bundle the spec layer works on."""
    return read_gguf_bundle(path)


def prewarm_shards(bundle: GGUFBundle) -> dict[str, float]:
    """Pull a bundle's shard files into the OS page cache.

    Exposed here for the same reason as :func:`load_bundle`; the pass itself is
    :mod:`src.loader.gguf.prewarm`'s.
    """
    return _prewarm_shards(bundle)

