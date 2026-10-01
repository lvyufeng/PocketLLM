"""The public, backend-neutral PocketLLM API.

Two layers meet here and neither is the kernel ABI.  ``EngineArgs`` is how a
caller says what to build; ``GenerationRequest``/``GenerationResult``/``TokenEvent``
are what a backend emits for the serving layer to render.  Nothing in this package
imports a device runtime, so the types can be imported -- and the HTTP layer
tested -- on a host with no accelerator at all.
"""

from .backend import BackendContext, BackendFactory, EngineBackend
from .errors import (
    BackendNotImplementedError,
    BackendUnavailableError,
    ConfigurationError,
    PocketLLMError,
    RequestCancelledError,
    UnsupportedFeatureError,
)
from .types import (
    BackendCapabilities,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    Usage,
    device_kinds,
)

__all__ = [
    "BackendCapabilities",
    "BackendContext",
    "BackendFactory",
    "BackendNotImplementedError",
    "BackendUnavailableError",
    "ConfigurationError",
    "EngineArgs",
    "EngineBackend",
    "GenerationRequest",
    "GenerationResult",
    "HealthStatus",
    "PocketLLMError",
    "RequestCancelledError",
    "SamplingParams",
    "TimingMetrics",
    "TokenEvent",
    "Usage",
    "device_kinds",
]