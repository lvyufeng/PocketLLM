"""Adapter for the existing PyTorch/Triton runtimes.

This module intentionally imports the legacy runtime lazily.  Importing
``pocketllm`` on a CPU-only host therefore does not import Torch or CUDA
extensions.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Callable, Mapping

from pocketllm.api import (
    BackendCapabilities,
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


class TorchBackend(BackendBase):
    """Backend adapter over ``src.server`` and model generation functions.

    ``runtime`` and ``serving_engine`` are injectable to keep API tests
    independent of model checkpoints.  The normal constructor loads the
    existing DeepSeek serving runtime and leaves all model-specific kernels and
    performance switches in ``src/``.
    """

    def __init__(
        self,
        args: EngineArgs,
        *,
        runtime: Mapping[str, Any] | None = None,
        serving_engine: Any | None = None,
        runtime_loader: Callable[[argparse.Namespace], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        self._runtime: Mapping[str, Any] | None = runtime
        self._serving_engine = serving_engine
        self._runtime_loader = runtime_loader
        self._request_lock = threading.RLock()
        self._model_id = args.model.rsplit("/", 1)[-1] or "pytorch"
        if runtime is not None or serving_engine is not None:
            self._ready = True

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="torch",
            model_formats=("safetensors", "gguf"),
            devices=("cuda", "cpu"),
            supports_batch=False,
            supports_streaming=True,
            # Cancellation is observed between streamed events and at request
            # boundaries. It never interrupts a running device kernel.
            supports_cancellation=True,
            supports_embeddings=False,
            supports_logprobs=True,
            supports_structured_outputs=False,
            supports_prefix_caching=True,
            details={
                "execution": "existing src/ runtime",
                "scheduler": "legacy serving queue",
                "cancellation": "safe boundary only",
            },
        )

    @property
    def runtime(self) -> Mapping[str, Any] | None:
        return self._runtime

    def _load(self) -> None:
        if self._runtime is not None and self._serving_engine is not None:
            self._ready = True
            return
        if self._runtime is None:
            if self._runtime_loader is not None:
                loader = self._runtime_loader
            else:
                from src.server.openai import _init_runtime

                loader = _init_runtime
            ckpt = self.args.checkpoint_dir
            config = self.args.config_path or ""
            namespace = argparse.Namespace(
                ckpt_format=self.args.model_format,
                partition_policy=str(self.args.backend_options.get("partition_policy", "legacy")),
                pd_mode=str(self.args.backend_options.get("pd_mode", "scheduler")),
                routed_experts_device=str(self.args.backend_options.get("routed_experts_device", "gpu")),
                config=config,
                ckpt_path=ckpt,
                tokenizer_path=self.args.tokenizer_path,
                model=self._model_id,
                max_model_len=self.args.max_model_len or 0,
            )
            self._runtime = loader(namespace)
        if self._serving_engine is None:
            from src.server.engine import DeepSeekServingEngine
            from src.server.openai import _broadcast_payload, _run_payload, _run_payload_stream

            self._serving_engine = DeepSeekServingEngine(
                self._runtime,
                _broadcast_payload,
                _run_payload,
                _run_payload_stream,
            )
        runtime_model_id = self._runtime.get("model_id") if self._runtime else None
        if runtime_model_id:
            self._model_id = str(runtime_model_id)
        self._ready = True

    def _ensure_loaded(self) -> None:
        self._ensure_open()
        if not self._ready:
            with self._request_lock:
                if not self._ready:
                    self._load()

    def health(self) -> HealthStatus:
        status = super().health()
        if self._runtime is not None and isinstance(self._runtime, Mapping):
            details = dict(status.details)
            details.update({"model": self._model_id})
            return HealthStatus(status.status, status.backend, status.ready, status.message, details)
        return status

    def _tokenizer(self) -> Any:
        if self._runtime is None:
            return None
        return self._runtime.get("tokenizer")

    def _messages(self, request: GenerationRequest) -> list[dict[str, Any]] | None:
        """Return the normalized chat messages when the request carries them."""
        messages = request.metadata.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            normalized = [dict(message) for message in messages if isinstance(message, Mapping)]
            if normalized:
                return normalized
        return None

    def _prompt_text(self, request: GenerationRequest) -> str:
        """Render the prompt, preferring the runtime's own chat encoding.

        Chat requests are encoded with the DeepSeek template that the legacy
        server uses, so the unified API produces the same prompt as
        ``src.server.openai``.  Raw prompts and non-DeepSeek encodings fall back
        to the request text unchanged.
        """
        messages = self._messages(request)
        if messages is None:
            return request.prompt or ""
        try:
            from src.encoding.dsv4 import encode_messages
        except Exception:
            return request.prompt or ""
        return encode_messages(
            messages,
            thinking_mode=str(request.metadata.get("thinking_mode", "chat")),
            reasoning_effort=request.metadata.get("reasoning_effort"),
        )

    def _prompt_ids(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        tokenizer = self._tokenizer()
        if tokenizer is None:
            raise ValueError("TorchBackend needs a tokenizer for text prompts")
        encoded = tokenizer.encode(self._prompt_text(request))
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [int(token) for token in encoded]

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        params = request.sampling_params
        prompt_ids = self._prompt_ids(request)
        return {
            "op": "chat_completion",
            "request_id": request.request_id,
            "_prompt_ids": prompt_ids,
            "messages": self._messages(request) or [{"role": "user", "content": request.prompt or ""}],
            "thinking_mode": str(request.metadata.get("thinking_mode", "chat")),
            "reasoning_effort": request.metadata.get("reasoning_effort"),
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "repetition_penalty": params.repetition_penalty,
            "seed": params.seed,
            "stop": list(params.stop) if params.stop else None,
            "logprobs": params.logprobs,
            "top_logprobs": params.top_logprobs,
            "n": params.n,
            "generation_options": params.to_generation_options(),
            "stream": stream,
            "stream_options": request.metadata.get("stream_options", {}),
        }

    @staticmethod
    def _result_from_mapping(request: GenerationRequest, result: Mapping[str, Any]) -> GenerationResult:
        token_ids = [int(token) for token in result.get("token_ids", result.get("completion_ids", []))]
        text = str(result.get("text", result.get("content", "")) or "")
        usage = Usage(
            int(result.get("prompt_tokens", 0) or 0),
            int(result.get("completion_tokens", len(token_ids)) or len(token_ids)),
        )
        return GenerationResult(
            request_id=request.request_id,
            token_ids=token_ids,
            text=text,
            finish_reason=str(result.get("finish_reason", "stop")),
            usage=usage,
            timings=TimingMetrics.from_mapping(result.get("timings")),
            logprobs=result.get("logprobs") or result.get("token_logprobs"),
            metadata={key: value for key, value in result.items() if key not in {
                "token_ids", "completion_ids", "text", "content", "finish_reason",
                "prompt_tokens", "completion_tokens", "timings", "logprobs", "token_logprobs",
            }},
        )

    def _generate_injected(self, request: GenerationRequest) -> GenerationResult | Mapping[str, Any]:
        if self._runtime is None:
            raise RuntimeError("TorchBackend runtime is not loaded")
        callback = self._runtime.get("backend_generate")
        if not callable(callback):
            raise RuntimeError("injected Torch runtime has no backend_generate callback")
        return callback(request)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self._ensure_loaded()
        outputs: list[GenerationResult] = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_cancelled(request.request_id)
                with self._request_lock:
                    self._check_cancelled(request.request_id)
                    if self._runtime and callable(self._runtime.get("backend_generate")):
                        raw = self._generate_injected(request)
                    else:
                        if self._serving_engine is None:
                            raise RuntimeError("Torch serving engine is unavailable")
                        raw = self._serving_engine.submit(self._payload(request, stream=False))
                self._check_cancelled(request.request_id)
                if isinstance(raw, list):
                    if not raw:
                        result = GenerationResult(request.request_id)
                    else:
                        result = self._result_from_mapping(request, raw[0])
                        result.metadata["choices"] = [self._result_from_mapping(request, item) for item in raw]
                elif isinstance(raw, GenerationResult):
                    result = raw
                else:
                    result = self._result_from_mapping(request, raw)
                outputs.append(result)
            finally:
                self._clear_request(request.request_id)
        return outputs

    def _stream_injected(self, request: GenerationRequest) -> Iterator[Any]:
        if self._runtime is None:
            raise RuntimeError("TorchBackend runtime is not loaded")
        callback = self._runtime.get("backend_stream")
        if not callable(callback):
            raise RuntimeError("injected Torch runtime has no backend_stream callback")
        yield from callback(request)

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self._ensure_loaded()
        self._begin_request(request.request_id)
        self._check_cancelled(request.request_id)
        try:
            with self._request_lock:
                if self._runtime and callable(self._runtime.get("backend_stream")):
                    events = self._stream_injected(request)
                else:
                    if self._serving_engine is None:
                        raise RuntimeError("Torch serving engine is unavailable")
                    events = self._serving_engine.submit_stream(self._payload(request, stream=True))
                for event in events:
                    self._check_cancelled(request.request_id)
                    if isinstance(event, TokenEvent):
                        yield event
                        continue
                    if not isinstance(event, Mapping):
                        continue
                    kind = event.get("type")
                    token_ids = [int(token) for token in event.get("token_ids", [])]
                    text = str(event.get("text", "") or "")
                    if not text and token_ids:
                        tokenizer = self._tokenizer()
                        if tokenizer is not None:
                            text = str(tokenizer.decode(token_ids))
                    usage = None
                    if event.get("prompt_tokens") is not None:
                        completion = event.get("completion_tokens") or token_ids
                        if completion and isinstance(completion[0], list):
                            completion = completion[0]
                        usage = Usage(int(event.get("prompt_tokens", 0)), len(completion))
                    yield TokenEvent(
                        request_id=request.request_id,
                        token_id=token_ids[-1] if token_ids else None,
                        text=text,
                        finish_reason=event.get("finish_reason") if kind == "done" else None,
                        usage=usage,
                        metadata={"type": kind} if kind else {},
                    )
        finally:
            self._clear_request(request.request_id)

    def cancel(self, request_id: str) -> bool:
        # The flag is checked at safe request boundaries.  The legacy
        # generation callback is not interruptible inside a device kernel.
        return super().cancel(request_id)

    def close(self) -> None:
        if self._serving_engine is not None and hasattr(self._serving_engine, "close"):
            self._serving_engine.close()
        super().close()
