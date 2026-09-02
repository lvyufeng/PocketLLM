"""PocketLLM: one user-facing API over independent Torch and C++ backends."""

from .api import (
    BackendCapabilities,
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    PocketLLMError,
    RequestCancelledError,
    TensorParallelSupervisorError,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    UnsupportedFeatureError,
    Usage,
)

# Import lazily to keep `import pocketllm` usable on CPU-only hosts without
# importing Torch, CUDA extensions, or the optional native module.
def __getattr__(name: str):
    if name in {"AsyncLLM", "LLM"}:
        from .engine import AsyncLLM, LLM
        return {"LLM": LLM, "AsyncLLM": AsyncLLM}[name]
    raise AttributeError(name)


__all__ = [
    "AsyncLLM",
    "BackendCapabilities",
    "BackendUnavailableError",
    "ConfigurationError",
    "EngineArgs",
    "EngineBackend",
    "GenerationRequest",
    "GenerationResult",
    "HealthStatus",
    "LLM",
    "PocketLLMError",
    "RequestCancelledError",
    "TensorParallelSupervisorError",
    "SamplingParams",
    "TimingMetrics",
    "TokenEvent",
    "UnsupportedFeatureError",
    "Usage",
]
