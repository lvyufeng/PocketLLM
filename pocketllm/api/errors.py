"""Public exceptions raised by the PocketLLM API.

These are the errors a *client* of the Python API may see and branch on, so they
are named for what the caller did wrong rather than for where in the tree it was
noticed.  The engine and the backends have their own errors -- ``NoUsableBackend``,
``BackendUnavailable``, ``ShapeError`` -- and those are implementation detail by
comparison: an HTTP handler catches :class:`PocketLLMError` and maps it to a
status, and does not need to know which layer produced it.

``TensorParallelSupervisorError`` was here and is gone with the multi-card world
it described: one process owns one card, the second rank this tree might have
supervised does not exist, and a second name for a rank that cannot be spawned
would be a name nothing raises.
"""

from __future__ import annotations

__all__ = [
    "BackendNotImplementedError",
    "BackendUnavailableError",
    "ConfigurationError",
    "PocketLLMError",
    "RequestCancelledError",
    "UnsupportedFeatureError",
]


class PocketLLMError(RuntimeError):
    """Base class for errors that are safe to expose to API clients."""


class ConfigurationError(PocketLLMError, ValueError):
    """The engine or request configuration is invalid."""


class BackendUnavailableError(PocketLLMError):
    """A requested backend is not installed or cannot be initialized."""


class UnsupportedFeatureError(PocketLLMError, NotImplementedError):
    """The selected backend does not implement a requested feature."""


class BackendNotImplementedError(UnsupportedFeatureError):
    """A declared backend exists and is loadable, but the work is not written yet.

    The distinction from :class:`BackendUnavailableError` is the whole reason the
    skeleton can ship: a stub backend is *available* -- its device may well be
    present -- but refuses the call, and says what it is waiting for.  "The
    runtime is missing" and "the kernel is not written" are different facts, and
    only one of them is fixed by installing something.
    """


class RequestCancelledError(PocketLLMError):
    """Generation was cancelled before it completed."""