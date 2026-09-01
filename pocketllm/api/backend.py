"""Protocol implemented by Torch and native C++ execution adapters."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Iterator, Protocol, Sequence

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
