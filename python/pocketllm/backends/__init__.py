"""Device backends: the implementations of the kernel ABI.

Each subpackage here is one device -- ``cpu``, ``cuda``, ``qnn`` ... -- and
declares what it can run before it can run anything.  The declaration is the
load-bearing part: ``pocketllm devices`` prints it, dispatch resolves against it,
and a backend that is a stub still has to say which op it will implement and
which runtime it is waiting for.  A tree where an unimplemented path is declared
but refused is workable; one where it is silently absent is not.

Importing this package imports **no** backend.  A backend module is loaded only
when it is selected or probed, because each one's import cost is its runtime's
import cost, and on a phone that is the difference between a working install and
a missing shared library.  See :mod:`pocketllm.backends.registry`.
"""

from __future__ import annotations

from .base import BackendUnavailable, DeclaredBackend, StubBackend, UnimplementedSession
from .registry import BACKENDS, available_backends, builtin, describe, discover, get

__all__ = [
    "BACKENDS",
    "BackendUnavailable",
    "DeclaredBackend",
    "UnimplementedSession",
    "available_backends",
    "builtin",
    "describe",
    "discover",
    "get",
]