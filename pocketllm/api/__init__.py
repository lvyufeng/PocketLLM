"""Public backend-neutral PocketLLM API types."""

from .backend import EngineBackend
from .errors import (
    BackendUnavailableError,
    ConfigurationError,
    PocketLLMError,
    RequestCancelledError,
    TensorParallelSupervisorError,
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
)

__all__ = [
    "BackendCapabilities",
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
    "TensorParallelSupervisorError",
    "TimingMetrics",
    "TokenEvent",
    "UnsupportedFeatureError",
    "Usage",
]
