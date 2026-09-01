"""Public exceptions raised by the PocketLLM API."""

from __future__ import annotations


class PocketLLMError(RuntimeError):
    """Base class for errors that are safe to expose to API clients."""


class ConfigurationError(PocketLLMError, ValueError):
    """The engine or request configuration is invalid."""


class BackendUnavailableError(PocketLLMError):
    """A requested backend is not installed or cannot be initialized."""


class UnsupportedFeatureError(PocketLLMError, NotImplementedError):
    """The selected backend does not implement a requested feature."""


class RequestCancelledError(PocketLLMError):
    """Generation was cancelled before it completed."""
