"""Protocol implemented by execution adapters.

This is the *serving* contract, and it is deliberately not the kernel ABI.  A
backend session runs one op on one device; an engine backend turns a request into
tokens, owns whatever state that takes (a prompt cache, a batch scheduler), and
reports health.  Keeping the two apart is what lets the HTTP layer be tested
against a scripted fake with no device at all, and what lets a real device be
driven without an HTTP server in the way.

``run_worker`` was part of this protocol and is gone.  It existed so a supervised
tensor-parallel rank could enter its own loop after forking; this tree launches no
second rank, so an adapter implementing it would have had a method only a deleted
launcher could call.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .types import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    TokenEvent,
)


class EngineBackend(Protocol):
    """Observable engine contract; physical cache and scheduler stay private."""

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def health(self) -> HealthStatus:
        ...

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        ...

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        ...

    def prepare(self) -> None:
        """Eagerly initialize the backend before one request is served."""
        ...

    def audit_request(self, body: Mapping[str, Any], *, endpoint: str = "chat") -> Any:
        """The first request field this backend cannot serve, or ``None``.

        A field the runtime will not apply, sent with a value that would have changed the answer,
        has to be refused by *name*: answering it with a 200 and text generated as if the field were
        absent is a response the caller cannot tell from the one it asked for.  ``None`` is the
        honest default for a backend that has not audited its fields -- it has made no claim to hold
        a caller to.
        """
        ...

    def metrics(self) -> Mapping[str, float]:
        """Engine-owned values the server exports, as ``name -> value``.

        Request-scoped numbers travel on the result the server already reads; this is for what the
        engine holds *between* requests -- a prompt cache's occupancy, say -- which no one request
        owns.  Empty is the right answer for an engine with nothing to add.
        """
        ...

    def cancel(self, request_id: str) -> bool:
        ...

    def close(self) -> None:
        ...


class BackendFactory(Protocol):
    def __call__(self, args):
        ...


class BackendContext(AbstractContextManager):
    """Typing helper for backend implementations with context-manager support."""

    def __enter__(self) -> EngineBackend:
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


#: Imported for the factory protocol's annotation, which names a callable an
#: adapter supplies; kept so the module's public surface matches its imports.
_ = Callable