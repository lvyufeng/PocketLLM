"""Protocol implemented by Torch and native C++ execution adapters."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Callable, Iterator, Protocol, Sequence

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
        """Eagerly initialize the backend for a supervised rank."""
        ...

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Enter the backend-specific worker loop for a nonzero TP rank."""
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
