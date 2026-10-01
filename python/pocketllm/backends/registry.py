"""Finding backends: a static table in the tree, and entry points outside it.

There are two ways a backend is discovered, and the split is deliberate.

**In-tree.** :data:`BACKENDS` maps a name to a module and a factory.  The table
holds *strings*: importing ``pocketllm.backends`` loads no backend, and
:func:`get` imports exactly one.

What a listing does import is each backend's *declaration module* -- there is no
way to read an out-of-tree backend's capabilities without it -- so the rule that
makes ``pocketllm devices`` work on a machine with nothing installed is narrower
and has to be stated exactly: **a declaration module imports no runtime, and its
probe is a filesystem question.**  That is why the probes live in
:class:`~pocketllm.backends.base.RuntimeProbe` and why the backend modules import
``numpy`` (if at all) inside their session, not at module level.

**Out-of-tree.** :func:`discover` reads the ``pocketllm.backends`` entry-point
group, so a third party ships a backend as a plugin without this tree knowing it
exists.  This is the same extension point that makes ``DeviceKind`` an open
string rather than an enum: "more devices later" is a design requirement here,
not an aspiration.

Nothing in this module imports a backend's runtime, and nothing raises because a
runtime is absent.  The only failure modes are "that name is not registered" and,
in strict mode, "that backend is registered but not loadable here".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

from pocketllm.kernels.backend import Backend

from .base import BackendUnavailable, DeclaredBackend

__all__ = [
    "BackendEntry",
    "BACKENDS",
    "ENTRY_POINT_GROUP",
    "builtin",
    "discover",
    "get",
    "available_backends",
    "describe",
]

#: The entry-point group a third-party backend publishes under.  The group name
#: is part of this tree's public contract, so it is spelled here and not inline.
ENTRY_POINT_GROUP = "pocketllm.backends"


@dataclass(frozen=True, slots=True)
class BackendEntry:
    """One discoverable backend: what to import, and how to build it."""

    name: str
    module: str
    factory: Callable[[], Backend]
    #: Where the backend comes from -- ``"builtin"`` or ``"entry-point <dist>"``.
    source: str = "builtin"


def _factory(module: str, attribute: str = "BACKEND") -> Callable[[], Backend]:
    def load() -> Backend:
        mod = importlib.import_module(module)
        try:
            return getattr(mod, attribute)
        except AttributeError as exc:  # pragma: no cover - a packaging bug
            raise ImportError(f"{module} does not export {attribute}") from exc

    load.__qualname__ = f"load:{module}.{attribute}"
    return load


#: The in-tree backends, by name.  Order is the order ``pocketllm devices``
#: prints and the default preference among equally-ranked candidates, so it runs
#: from "always present" to "most specialised".
BACKENDS: dict[str, BackendEntry] = {
    "reference": BackendEntry("reference", "pocketllm.backends.reference", _factory("pocketllm.backends.reference")),
    "cpu": BackendEntry("cpu", "pocketllm.backends.cpu", _factory("pocketllm.backends.cpu")),
    "mps": BackendEntry("mps", "pocketllm.backends.mps", _factory("pocketllm.backends.mps")),
    "cuda": BackendEntry("cuda", "pocketllm.backends.cuda", _factory("pocketllm.backends.cuda")),
    "qnn": BackendEntry("qnn", "pocketllm.backends.qnn", _factory("pocketllm.backends.qnn")),
    "horizon": BackendEntry("horizon", "pocketllm.backends.horizon", _factory("pocketllm.backends.horizon")),
    "ascend": BackendEntry("ascend", "pocketllm.backends.ascend", _factory("pocketllm.backends.ascend")),
}


def builtin() -> tuple[str, ...]:
    """The names this tree ships, in listing order."""
    return tuple(BACKENDS)


def discover() -> tuple[BackendEntry, ...]:
    """Entry points published under :data:`ENTRY_POINT_GROUP`, outside this tree.

    A broken entry point is skipped rather than raised: one third-party package
    with a bad metadata line must not make ``pocketllm devices`` unusable, which
    is the command an operator runs *because* something is wrong.
    """
    from importlib.metadata import entry_points

    found: list[BackendEntry] = []
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        def make(entry=entry) -> Backend:
            return entry.load()()

        make.__qualname__ = f"load:{entry.value}"
        dist = getattr(entry, "dist", None)
        source = f"entry-point {dist.name}" if dist is not None else "entry-point"
        found.append(BackendEntry(entry.name, entry.value, make, source=source))
    return tuple(found)


def _entries() -> tuple[BackendEntry, ...]:
    return tuple(BACKENDS.values()) + discover()


def raw(name: str) -> BackendEntry:
    """The entry for a name, in-tree or from an entry point, without loading it."""
    entry = BACKENDS.get(name)
    if entry is not None:
        return entry
    for candidate in discover():
        if candidate.name == name:
            return candidate
    raise KeyError(f"no backend named {name!r}; known: {sorted(set(BACKENDS) | {e.name for e in discover()})}")


def get(name: str, *, available_only: bool = True) -> Backend:
    """Load one backend by name.

    ``available_only`` refuses a backend whose runtime is absent, by name and
    with what it is missing -- which is the message a user needs after a typo or
    on a machine without the device.  Pass ``False`` to load a backend anyway,
    which is what a caller inspecting declared capabilities on a foreign host
    wants.
    """
    backend = raw(name).factory()
    if available_only and not backend.available():
        raise BackendUnavailable(_missing(backend))
    return backend


def _missing(backend: Backend) -> str:
    detail = getattr(backend, "missing_dependency", "") or "its runtime"
    return f"backend {backend.name!r} is not loadable here: needs {detail}"


def available_backends() -> tuple[Backend, ...]:
    """Every loadable backend, in-tree first then entry points.

    A backend whose module fails to import is dropped: this is the listing path,
    and an unusable plugin should not take the listing down with it.
    """
    out: list[Backend] = []
    for entry in _entries():
        try:
            backend = entry.factory()
        except Exception:  # noqa: BLE001 - a bad plugin must not break the listing
            continue
        if backend.available():
            out.append(backend)
    return tuple(out)


def describe(include_unavailable: bool = True) -> tuple[dict, ...]:
    """One row per backend for ``pocketllm devices``.

    Every row carries ``available`` and, when it is False, ``missing`` -- the
    name of the runtime that would fix it.  That pairing is the whole point of
    the command: "no backend for this device" is unhelpful, and "qnn: needs
    libQnnHtp.so" is actionable.
    """
    rows: list[dict] = []
    for entry in _entries():
        try:
            backend = entry.factory()
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "name": entry.name,
                    "source": entry.source,
                    "available": False,
                    "missing": f"import failed: {exc}",
                    "kind": "",
                    "version": "",
                    "ops": 0,
                    "graph": "none",
                    "summary": "",
                }
            )
            continue
        available = backend.available()
        row = {
            "name": backend.name,
            "source": entry.source,
            "available": available,
            "missing": "" if available else (getattr(backend, "missing_dependency", "") or "unknown"),
            "kind": backend.device_kind,
            "version": backend.version,
            "ops": len(backend.capabilities()),
            "graph": backend.graph().mode.value,
            "summary": getattr(backend, "summary", ""),
        }
        if row["available"] or include_unavailable:
            rows.append(row)
    return tuple(rows)


def with_kind(kind: str, *, available_only: bool = True) -> tuple[Backend, ...]:
    """Backends that open devices of ``kind``, for a device-driven selection."""
    if available_only:
        candidates = available_backends()
    else:
        candidates = tuple(entry.factory() for entry in _entries())
    return tuple(backend for backend in candidates if backend.device_kind == kind)