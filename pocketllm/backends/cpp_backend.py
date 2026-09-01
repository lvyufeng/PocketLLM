"""Optional native C++ backend adapter.

The native module is deliberately optional.  A CUDA/Ascend build can provide
``pocketllm_cpp`` without making the public Python package depend on a vendor
SDK at import time.
"""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    Usage,
    UnsupportedFeatureError,
)

from .base import BackendBase


_NATIVE_MODULE_NAMES = ("pocketllm_cpp", "_pocketllm_cpp", "cpp_engine")


def load_native_module() -> Any:
    errors: list[str] = []
    for name in _NATIVE_MODULE_NAMES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        # The repository's ``cpp_engine`` directory is importable as a Python
        # namespace package even when no extension was built.  Treat that as
        # unavailable instead of returning a module that cannot construct an
        # engine, which would also make ``backend=auto`` select C++ incorrectly.
        if not hasattr(module, "QwenEngine"):
            errors.append(f"{name}: QwenEngine binding is missing")
            continue
        return module
    raise BackendUnavailableError(
        "native C++ backend is unavailable; build cpp_engine with "
        "-DPOCKET_BUILD_PYTHON=ON (" + "; ".join(errors) + ")"
    )


def _native_kv_cache_dtype(value: str) -> str:
    """Resolve the public ``auto`` value to the native Qwen default."""
    normalized = str(value or "auto").lower()
    return "fp16" if normalized == "auto" else normalized


def _native_device_index(value: str | int | None) -> int:
    """Normalize a public device selector for the native single-rank option."""
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ConfigurationError("C++ backend device must be a non-negative device index")
    if isinstance(value, int):
        index = value
    else:
        text = str(value).strip().lower()
        for prefix in ("cuda:", "ascend:", "npu:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        try:
            index = int(text)
        except ValueError as exc:
            raise ConfigurationError(
                "C++ backend device must be an integer or cuda:/ascend:/npu: index"
            ) from exc
    if index < 0:
        raise ConfigurationError("C++ backend device must be a non-negative device index")
    return index


class CppBackend(BackendBase):
    """Serialized compatibility adapter for the stateful native engines.

    The initial bridge intentionally supports Qwen's token-oriented API.  The
    native engine still owns its optimized KV layout and TP protocol; this
    adapter does not turn it into a Torch module or copy its buffers.
    """

    def __init__(
        self,
        args: EngineArgs,
        *,
        native_module: Any | None = None,
        engine: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        if native_module is not None:
            self._native = native_module
        elif engine is None:
            self._native = load_native_module()
        else:
            self._native = None
        self._engine = engine if engine is not None else self._construct_engine()
        # A primitive lock may be released by a different worker thread.  This
        # matters for AsyncLLM, which advances one generator through an executor
        # without guaranteeing that every next() call uses the same thread.
        self._request_lock = threading.Lock()
        self._tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._ready = True

    @staticmethod
    def native_available() -> bool:
        try:
            load_native_module()
            return True
        except BackendUnavailableError:
            return False

    @property
    def capabilities(self) -> BackendCapabilities:
        # The Ascend build rejects the external drafters, so report only what the
        # linked backend actually implements instead of the CUDA superset.
        native_backend = str(getattr(self._native, "backend", "") or "").lower()
        speculative: tuple[str, ...] = ("mtp",) if native_backend == "ascend" else ("mtp", "dspark", "dflash2")
        return BackendCapabilities(
            name="cpp",
            models=("qwen3.5",),
            model_formats=("safetensors",),
            devices=(native_backend,) if native_backend else ("cuda", "ascend"),
            supports_batch=False,
            supports_streaming=True,
            supports_cancellation=True,
            supports_embeddings=False,
            supports_logprobs=False,
            supports_structured_outputs=False,
            supports_prefix_caching=True,
            supports_speculative_decoding=speculative,
            details={
                "execution": "native C++",
                "scheduler": "serialized compatibility session",
                "device_backend": native_backend or "unknown",
                "cancellation": "safe boundary only",
            },
        )

    def _load_tokenizer(self) -> Any | None:
        tokenizer_path = self.args.tokenizer_path or self.args.checkpoint_dir
        if not tokenizer_path:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            return None

    def _construct_engine(self) -> Any:
        kind = str(self.args.backend_options.get("engine_kind", "qwen")).lower()
        if kind != "qwen":
            raise UnsupportedFeatureError(
                "the initial Python C++ adapter supports QwenEngine; "
                "PersistentEngine bindings are available for low-level use"
            )
        cls = getattr(self._native, "QwenEngine", None)
        options_cls = getattr(self._native, "QwenEngineOptions", None)
        if cls is None or options_cls is None:
            raise BackendUnavailableError("native module does not expose QwenEngine bindings")
        options = options_cls()
        mappings = {
            "tp_world": self.args.tensor_parallel_size,
            "tp_rank": self.args.tensor_parallel_rank,
            "device": _native_device_index(self.args.device),
            "prefill_chunk_tokens": self.args.prefill_chunk_tokens or 8192,
            "attention_window": self.args.attention_window,
            "attention_sink_tokens": self.args.attention_sink_tokens,
            "prefix_cache": self.args.enable_prefix_caching,
            "state_snapshot_interval_tokens": int(self.args.backend_options.get("state_snapshot_interval_tokens", 4096)),
            "max_state_snapshots": int(self.args.backend_options.get("max_state_snapshots", 82)),
            "mtp": self.args.speculative_method == "mtp",
            "mtp_speculative_tokens": self.args.speculative_tokens,
            "mtp_adaptive": bool(self.args.backend_options.get("mtp_adaptive", False)),
            "dspark_checkpoint": str(self.args.backend_options.get("dspark_checkpoint", "")),
            "dflash2_checkpoint": str(self.args.backend_options.get("dflash2_checkpoint", "")),
            "nccl_id_path": str(self.args.backend_options.get("nccl_id_path", "")),
        }
        for name, value in mappings.items():
            if hasattr(options, name):
                setattr(options, name, value)
        if hasattr(options, "kv_cache_dtype") and hasattr(self._native, "parse_qwen_kv_cache_dtype"):
            setattr(
                options,
                "kv_cache_dtype",
                self._native.parse_qwen_kv_cache_dtype(_native_kv_cache_dtype(self.args.kv_cache_dtype)),
            )
        for name, value in {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "sampling_seed": 0,
        }.items():
            if hasattr(options, name):
                setattr(options, name, value)
        return cls(self.args.checkpoint_dir, options, 0, self.args.max_model_len or 8192)

    def health(self) -> HealthStatus:
        status = super().health()
        details = dict(status.details)
        details.update({"model": self.args.checkpoint_dir})
        return HealthStatus(status.status, status.backend, status.ready, status.message, details)

    def _prompt_ids(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        if self._tokenizer is None:
            raise ConfigurationError("C++ backend needs tokenizer_path for text prompts")
        encoded = self._tokenizer.encode(request.prompt or "")
        return [int(token) for token in encoded]

    def _check_sampling(self, params: SamplingParams) -> None:
        if not params.greedy:
            raise UnsupportedFeatureError("the initial C++ adapter exposes native greedy generation only")
        if params.top_p is not None or params.top_k is not None or params.min_p is not None:
            raise UnsupportedFeatureError("C++ sampling controls are not exposed by the initial binding")
        if params.n != 1 or params.logprobs or params.stop:
            raise UnsupportedFeatureError("n, logprobs, and stop require the shared scheduler phase")

    @staticmethod
    def _native_token(item: Any) -> int:
        if isinstance(item, int):
            return int(item)
        return int(getattr(item, "top_token", getattr(item, "token", item)))

    def _decode(self, token_ids: list[int]) -> str:
        if self._tokenizer is None:
            return ""
        decode = getattr(self._tokenizer, "decode", None)
        if callable(decode):
            return str(decode(token_ids))
        decode_tokens = getattr(self._tokenizer, "decode_tokens", None)
        if callable(decode_tokens):
            return str(decode_tokens(token_ids))
        return ""

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests serially until a request-aware cache scheduler exists."""
        self._ensure_open()
        outputs: list[GenerationResult] = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_sampling(request.sampling_params)
                self._check_cancelled(request.request_id)
                prompt_ids = self._prompt_ids(request)
                started = time.perf_counter()
                with self._request_lock:
                    self._ensure_open()
                    self._check_cancelled(request.request_id)
                    engine = self._engine
                    if engine is None:
                        raise RuntimeError("native C++ engine is closed")
                    raw = engine.generate(
                        prompt_ids, request.sampling_params.max_tokens
                    )
                    self._ensure_open()
                self._check_cancelled(request.request_id)
                token_ids = [self._native_token(item) for item in raw]
                outputs.append(
                    GenerationResult(
                        request_id=request.request_id,
                        token_ids=token_ids,
                        text=self._decode(token_ids),
                        finish_reason="length",
                        usage=Usage(len(prompt_ids), len(token_ids)),
                        timings=TimingMetrics(
                            total_seconds=time.perf_counter() - started,
                            ttft_seconds=0.0,
                        ),
                    )
                )
            finally:
                self._clear_request(request.request_id)
                if self._closed:
                    self._release_native()
        return outputs

    def _stream_native(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self._check_sampling(request.sampling_params)
        prompt_ids = self._prompt_ids(request)
        max_tokens = request.sampling_params.max_tokens
        engine = self._engine
        if engine is None:
            raise RuntimeError("native C++ engine is closed")
        # Do not reset() here.  QwenEngine::reset() clears the prefix cache, so
        # calling it per request would disable configured prefix reuse.  prefill()
        # already matches the common prefix, restores a snapshot, or zeroes the
        # recurrent state, which is also what native generate() relies on.
        # QwenEngine.prefill predicts the first token without consuming it. The
        # following decode steps consume the previous prediction and predict the
        # next one, matching the native generate() result ordering.
        result = engine.prefill(prompt_ids)
        generated: list[int] = []
        previous_text = ""
        for index in range(max_tokens):
            self._ensure_open()
            self._check_cancelled(request.request_id)
            token = self._native_token(result)
            generated.append(token)
            decoded = self._decode(generated)
            # Decode the complete sequence so BPE/UTF-8 token boundaries are handled
            # by the tokenizer. Emit only the newly visible suffix when possible.
            text = decoded[len(previous_text):] if decoded.startswith(previous_text) else decoded
            previous_text = decoded
            event = TokenEvent(request.request_id, token_id=token, text=text)
            if index + 1 == max_tokens:
                event.finish_reason = "length"
                event.usage = Usage(len(prompt_ids), len(generated))
            yield event
            if index + 1 < max_tokens:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                result = engine.decode_step(token)

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        # A primitive lock can span generator yields even when AsyncLLM resumes
        # the generator on a different executor thread. It serializes all access
        # to the native engine's single mutable KV-cache session.
        self._begin_request(request.request_id)
        with self._request_lock:
            try:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                yield from self._stream_native(request)
            finally:
                self._clear_request(request.request_id)
                if self._closed:
                    self._release_native()

    def _release_native(self) -> None:
        engine = self._engine
        self._engine = None
        self._tokenizer = None
        self._native = None
        self._ready = False
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        if self._closed:
            return
        super().close()
        # Never destroy a native engine while a GIL-released kernel is using it.
        # An active generate/stream call releases it from its own finally block.
        if self._request_lock.acquire(blocking=False):
            try:
                self._release_native()
            finally:
                self._request_lock.release()
