"""The engine session: pick a device, open one backend, own the lifetime.

This is where "one process owns one card" stops being a slogan and becomes a
constructor.  A :class:`EngineSession` is opened with a device and produces
exactly one :class:`~pocketllm.kernels.backend.BackendSession`; it never opens a
second, spawns a worker, or coordinates with another process.  A model that does
not fit is quantized further, which is a decision the caller makes before the
device is chosen, not something this layer does behind its back.

**Device selection is a policy, and the policy differs by workload.**  The
reference backend always *can* run a model -- it is numpy on the host -- but a
29-billion-parameter model on numpy is a ten-minute first token, which is a hang
with better manners rather than graceful degradation.  So the fallback is:

* **on for ``run``**, where a slow answer beats no answer;
* **off for ``serve``**, where a request that will not return in time is worse
  than a refusal naming the device that would have worked.

That asymmetry is the only place the engine has an opinion about performance, and
it is stated here once rather than at every call site.

Nothing in this module imports a device runtime: it asks the registry, which asks
each backend's probe, which is a filesystem question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pocketllm.backends import registry
from pocketllm.backends.base import BackendUnavailable
from pocketllm.kernels.backend import Backend
from pocketllm.kernels.device import Device, KNOWN_DEVICE_KINDS

__all__ = ["EngineSession", "SessionPolicy", "NoUsableBackend"]


class NoUsableBackend(RuntimeError):
    """No backend could be opened for the requested device, and the policy forbade a fallback."""


@dataclass(frozen=True, slots=True)
class SessionPolicy:
    """How a session chooses, and how far it will fall back."""

    #: A backend name, or ``None`` to choose by device kind.
    backend: str | None = None
    #: Whether the reference backend may serve a request no device backend took.
    #: ``run`` allows it; ``serve`` does not.
    allow_reference_fallback: bool = True
    #: Backend names tried before the registry's default order.
    preference: tuple[str, ...] = ()

    @classmethod
    def for_run(cls, **kwargs: Any) -> "SessionPolicy":
        return cls(**kwargs)

    @classmethod
    def for_serve(cls, **kwargs: Any) -> "SessionPolicy":
        return cls(allow_reference_fallback=False, **kwargs)


class EngineSession:
    """One open device, its backend, and the executors that run work on it.

    Construct with :meth:`open` rather than directly, so the selection policy and
    the fallback rules are applied in one place.
    """

    def __init__(
        self,
        backend: Backend,
        session,
        device: Device,
        *,
        policy: SessionPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.session = session
        self.device = device
        self.policy = policy or SessionPolicy()
        self._closed = False

    # -- construction -------------------------------------------------------

    @classmethod
    def open(
        cls,
        device: Device | str | int | None = None,
        *,
        policy: SessionPolicy | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> "EngineSession":
        """Open the best backend for ``device``, applying the fallback policy.

        ``device`` may be a :class:`Device`, a string like ``"cuda:1"``, or
        ``None``/``"auto"`` for "whatever this host has".  The resolution is
        *first match in preference order*, and it is the caller's device kind --
        not the backend's name -- that is matched, so a third-party backend for
        ``cuda`` is picked up without this tree knowing it exists.
        """
        policy = policy or SessionPolicy()
        chosen = _select(device, policy)
        if chosen is None:
            raise NoUsableBackend(_no_backend_message(device, policy))
        resolved_device = _resolve_device(device, chosen)
        try:
            handle = chosen.open(resolved_device, options=options)
        except BackendUnavailable as exc:
            raise NoUsableBackend(str(exc)) from exc
        return cls(chosen, handle, resolved_device, policy=policy)

    # -- use ----------------------------------------------------------------

    @property
    def privileged(self) -> bool:
        """Whether this session may hand a region to the backend's graph path."""
        return self.backend.graph().supported

    def executor(self, **kwargs: Any) -> "EngineExecutor":
        from .executor import Executor

        kwargs.setdefault("allow_reference_fallback", self.policy.allow_reference_fallback)
        return Executor(self.session, **kwargs)

    def describe(self) -> str:
        graph = self.backend.graph()
        where = f"{self.backend.name} on {self.device}"
        if graph.supported:
            return f"{where} (capture: {graph.mode.value}, {len(graph.captures)} ops)"
        return f"{where} (eager only)"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.session.close()

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


#: ``EngineExecutor`` is a re-export of the executor, named here so a caller that
#: has a session never needs to import the module.
from .executor import Executor as EngineExecutor  # noqa: E402


def _select(device: Device | str | int | None, policy: SessionPolicy) -> Backend | None:
    """The first loadable backend matching the policy and device, or ``None``."""
    if policy.backend is not None:
        try:
            backend = registry.get(policy.backend, available_only=True)
        except (KeyError, BackendUnavailable):
            return None
        return backend

    kind = _kind_of(device)
    candidates: list[Backend] = []
    for name in policy.preference:
        try:
            candidates.append(registry.get(name, available_only=True))
        except (KeyError, BackendUnavailable):
            continue
    for backend in registry.available_backends():
        if backend not in candidates:
            candidates.append(backend)

    for backend in candidates:
        if kind is not None and backend.device_kind != kind:
            continue
        if not policy.allow_reference_fallback and getattr(backend, "is_reference", False):
            continue
        return backend
    return None


def _kind_of(device: Device | str | int | None) -> str | None:
    """The device kind to match, or ``None`` when the caller left it to the host."""
    if device is None:
        return None
    text = str(device).strip().lower()
    if text in ("", "auto"):
        return None
    return Device.parse(device).kind


def _resolve_device(device: Device | str | int | None, backend: Backend) -> Device:
    """The concrete device to open, filling in the backend's kind for ``auto``."""
    kind = _kind_of(device)
    if kind is None:
        return Device(backend.device_kind)
    return Device.parse(device)


def _no_backend_message(device: Device | str | int | None, policy: SessionPolicy) -> str:
    kind = _kind_of(device) or "auto"
    lines = [f"no usable backend for {kind!r}:"]
    if policy.backend is not None:
        lines.append(f"  --backend {policy.backend} was requested and is not loadable here")
    lines.append(f"  reference fallback {'allowed' if policy.allow_reference_fallback else 'disabled'}")
    lines.append("  candidates:")
    for row in registry.describe():
        state = "available" if row["available"] else f"missing {row['missing']}"
        lines.append(f"    {row['name']:<9} {row['kind']:<8} {state}")
    lines.append("  the known device kinds are " + ", ".join(KNOWN_DEVICE_KINDS))
    return "\n".join(lines)