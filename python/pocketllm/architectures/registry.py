"""Finding an architecture by name, the same two ways a backend is found.

An architecture is not a backend: a backend is which *device* runs the work, an
architecture is what the work *is*.  They are discovered the same way, though, and
for the same reason -- a checkpoint whose structure shipped in a later release
should not need this tree to be edited, and a third party should be able to ship a
model the core has never heard of.

The in-tree entries are deliberately thin.  ``toy`` is the working example;
``xing4_0`` is the checkpoint this tree was built around and is **not** ported
yet, so its entry is absent rather than present-and-broken.  A registry that
lists an architecture it cannot build is worse than one that lists nothing: the
error arrives as a ``KeyError`` inside a builder instead of at selection time
with a message.

Unlike a backend, an architecture has no ``available()``.  A backend probes for
hardware; an architecture's only dependencies are the ABI and the builder, both
of which are always present.  What can fail is *building* it for a given config,
and that failure belongs at the call, with the config in hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ir import ModelSpec

__all__ = ["ArchitectureEntry", "ARCHITECTURES", "ENTRY_POINT_GROUP", "get", "names", "build"]

#: The entry-point group a third-party architecture publishes under.
ENTRY_POINT_GROUP = "pocketllm.architectures"


@dataclass(frozen=True, slots=True)
class ArchitectureEntry:
    """One buildable architecture."""

    name: str
    #: Called with ``(config, **kwargs)`` to produce a :class:`ModelSpec`.  The
    #: config's type is the architecture's own; the registry does not inspect it,
    #: because "what hyperparameters does this model have" is exactly the
    #: question a shared registry should not try to answer.
    builder: Callable[..., ModelSpec]
    #: One line for ``pocketllm architectures``.
    summary: str = ""
    source: str = "builtin"


def _toy(config=None, **kwargs) -> ModelSpec:
    """The registry's adapter: every architecture builder takes ``(config, **kwargs)``.

    The shape is fixed by the contract rather than by this one architecture, so a
    caller that has a config in hand -- which is the normal case, since a config
    is how a checkpoint's dimensions are passed -- does not need a second
    convention for the built-in.
    """
    from .toy import build as build_toy

    return build_toy(config, **kwargs)


def _qwen3(config=None, **kwargs) -> ModelSpec:
    """The same adapter for :mod:`pocketllm.architectures.qwen3`."""
    from .qwen3 import build as build_qwen3

    return build_qwen3(config, **kwargs)


#: The in-tree architectures, by name.
ARCHITECTURES: dict[str, ArchitectureEntry] = {
    "toy": ArchitectureEntry(
        name="toy",
        builder=_toy,
        summary="A tiny embedding + SwiGLU block; the executor's own smoke test",
    ),
    "qwen3": ArchitectureEntry(
        name="qwen3",
        builder=_qwen3,
        summary="Qwen3 decoder layers: per-head QK-norm, split-half RoPE, SwiGLU MLP",
    ),
}


def names() -> tuple[str, ...]:
    return tuple(ARCHITECTURES) + tuple(entry.name for entry in _discovered())


def _discovered() -> tuple[ArchitectureEntry, ...]:
    from importlib.metadata import entry_points

    found: list[ArchitectureEntry] = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        builder = entry.load()
        dist = getattr(entry, "dist", None)
        source = f"entry-point {dist.name}" if dist is not None else "entry-point"
        found.append(ArchitectureEntry(entry.name, builder, summary=entry.value, source=source))
    return tuple(found)


def get(name: str) -> ArchitectureEntry:
    entry = ARCHITECTURES.get(name)
    if entry is not None:
        return entry
    for candidate in _discovered():
        if candidate.name == name:
            return candidate
    raise KeyError(f"no architecture named {name!r}; known: {sorted(names())}")


def build(name: str, config=None, **kwargs) -> ModelSpec:
    """Build a named architecture, or raise naming what is available."""
    entry = get(name)
    if config is not None:
        return entry.builder(config, **kwargs)
    return entry.builder(**kwargs)