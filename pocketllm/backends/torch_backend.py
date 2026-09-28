"""Adapter for the existing PyTorch/Triton runtimes.

This module intentionally imports the legacy runtime lazily.  Importing
``pocketllm`` on a CPU-only host therefore does not import Torch or CUDA
extensions.
"""

from __future__ import annotations

import argparse
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Mapping

from pocketllm.api import (
    BackendCapabilities,
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    TimingMetrics,
    TokenEvent,
    Usage,
    UnsupportedFeatureError,
)
from pocketllm.protocol import encode_chat_prompt, render_fallback_prompt

from .base import BackendBase
from .capabilities import declared_capabilities
from .runtime_engine import RuntimeSpec, SchedulerHost, cancel_key


#: What the legacy runtime generates when a request carries no budget of its own
#: (``src/models/deepseek_v4/serving.py`` reads ``payload.get("max_tokens") or 512``).  This adapter has to name
#: the number rather than leave the field absent, because the same field is what the legacy
#: serving queue's admission check counts against its token budget; a missing one would be read
#: there as zero and the request would be admitted on a promise the runtime does not keep.
_LEGACY_DEFAULT_MAX_TOKENS = 512


class TorchBackend(SchedulerHost, BackendBase):
    """Backend adapter over ``src.server`` and model generation functions.

    ``runtime`` and ``serving_engine`` are injectable to keep API tests
    independent of model checkpoints.  The normal constructor loads the
    existing DeepSeek serving runtime and leaves all model-specific kernels and
    performance switches in ``src/``.
    """

    #: Used in the scheduler's error messages and thread names.
    name = "torch"

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
        self._init_batch_scheduler()

    @property
    def capabilities(self) -> BackendCapabilities:
        # The one runtime whose *description* is fully static: it reads every checkpoint the others
        # do not claim, and its cancellation is observed between streamed events and at request
        # boundaries rather than inside a running device kernel. Which path is live is not static,
        # though, so the two fields that depend on it are named as overrides.
        return declared_capabilities(
            "torch",
            details={
                "scheduler": (
                    "BatchScheduler (width 1, continuous batching off)"
                    if self._batching()
                    else "legacy serving queue"
                ),
            },
            # Two different claims, and the honest one depends on the path: driven by the shared
            # scheduler this runtime is submitting to a batch scheduler on which it declares one
            # slot; on the legacy path it serves one request at a time through the serving queue's
            # own bound. Either way the answer to "may this be called concurrently" is this field.
            supports_batch=self._batching(),
        )

    @property
    def runtime(self) -> Mapping[str, Any] | None:
        return self._runtime

    def _detect_expert_dtype(self) -> str | None:
        """Detect expert dtype from checkpoint config.json if available.

        Returns "fp4", "int8", or None if detection is not possible.
        """
        import json
        import pathlib

        checkpoint_dir = pathlib.Path(self.args.checkpoint_dir)
        config_json = checkpoint_dir / "config.json"

        if not config_json.exists():
            return None

        try:
            with open(config_json) as f:
                config = json.load(f)
            expert_dtype = config.get("expert_dtype")
            if expert_dtype in ("fp4", "int8"):
                return expert_dtype
        except (OSError, ValueError, KeyError):
            pass

        return None

    def _runtime_namespace(self) -> argparse.Namespace:
        import pathlib
        config_path = self.args.config_path
        if not config_path:
            # Auto-detect config based on checkpoint metadata when available
            detected_dtype = self._detect_expert_dtype()
            repo_root = pathlib.Path(__file__).resolve().parents[2]

            if detected_dtype == "fp4":
                default_config = repo_root / "configs" / "config_fp4_active.json"
            else:
                # Default to W8A8 for int8 or when detection fails
                default_config = repo_root / "configs" / "config_w8a8.json"

            config_path = str(default_config) if default_config.is_file() else ""
        return argparse.Namespace(
            ckpt_format=self.args.model_format,
            partition_policy=str(self.args.backend_options.get("partition_policy", "legacy")),
            pd_mode=str(self.args.backend_options.get("pd_mode", "scheduler")),
            routed_experts_device=str(self.args.backend_options.get("routed_experts_device", "cpu")),
            config=config_path,
            ckpt_path=self.args.checkpoint_dir,
            tokenizer_path=self.args.tokenizer_path,
            model=self._model_id,
            max_model_len=self.args.max_model_len or 0,
        )

    def _load_runtime(self) -> Mapping[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if self._runtime_loader is not None:
            loader = self._runtime_loader
        else:
            from src.models.deepseek_v4.serving import _init_runtime

            loader = _init_runtime
        self._runtime = loader(self._runtime_namespace())
        return self._runtime

    def _load(self) -> None:
        if self._runtime is not None and self._serving_engine is not None:
            self._ready = True
            return
        runtime = self._load_runtime()
        if self._serving_engine is None:
            from src.server.engine import DeepSeekServingEngine
            from src.models.deepseek_v4.serving import _broadcast_payload, _run_payload, _run_payload_stream

            self._serving_engine = DeepSeekServingEngine(
                runtime,
                _broadcast_payload,
                _run_payload,
                _run_payload_stream,
            )
        runtime_model_id = runtime.get("model_id")
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

    def _tokenize(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        tokenizer = self._tokenizer()
        if tokenizer is None:
            raise ValueError("TorchBackend needs a tokenizer for text prompts")
        messages = self._messages(request)
        if messages is not None:
            encoded = encode_chat_prompt(
                tokenizer,
                messages,
                thinking_mode=str(request.metadata.get("thinking_mode", "chat")),
                reasoning_effort=request.metadata.get("reasoning_effort"),
                tools=request.metadata.get("tools"),
                # The current Torch runtime is the DeepSeek runtime; use its
                # validated encoder when the checkpoint has no HF template.
                deepseek_fallback=True,
            )
            if encoded is not None:
                return encoded
        encoded = tokenizer.encode(self._prompt_text(request))
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [int(token) for token in encoded]

    def _prompt_text(self, request: GenerationRequest) -> str:
        """Return a deterministic text fallback for generic tokenizers."""
        messages = self._messages(request)
        if messages is not None:
            return request.prompt or render_fallback_prompt(messages)
        return request.prompt or ""

    def _budget(self, prompt_ids: Sequence[int], request: GenerationRequest) -> int:
        """Resolve the generation budget this adapter hands the legacy runtime.

        A budget the caller named is theirs.  An absent one is resolved against the context this
        adapter was configured with, the same rule the other backends apply, and only when no
        context was configured does the legacy runtime's own default stand in for it -- reproducing
        that default here rather than omitting the field is what keeps the serving queue's
        admission check counting the same number generation will use.

        The argument order is the shared one -- `(prompt_ids, request)` -- so that this is the same
        call the scheduler's side of the bridge makes. It was `(params, prompt_ids)` until this
        adapter joined the scheduler, and two derivations of one number on two routes is a request
        the engine and the scheduler disagree about the length of.
        """
        params = request.sampling_params
        if params.max_tokens is not None:
            return int(params.max_tokens)
        leftover = (self.args.max_model_len or 0) - len(prompt_ids)
        if leftover > 0:
            return params.token_budget(leftover)
        return _LEGACY_DEFAULT_MAX_TOKENS

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        params = request.sampling_params
        prompt_ids = self._tokenize(request)
        return {
            "op": "chat_completion",
            "request_id": request.request_id,
            "_prompt_ids": prompt_ids,
            "messages": self._messages(request) or [{"role": "user", "content": request.prompt or ""}],
            "thinking_mode": str(request.metadata.get("thinking_mode", "chat")),
            "reasoning_effort": request.metadata.get("reasoning_effort"),
            "max_tokens": self._budget(prompt_ids, request),
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

    # ------------------------------------------------------------- the shared scheduler

    def _runtime_spec(self) -> RuntimeSpec:
        """What the bridge needs to know about this runtime. See :class:`SchedulerHost`.

        `max_context` comes from `--max-model-len`, and has no default here. The legacy runtime's
        own bound is the model's and is not readable until the checkpoint is loaded, and the
        scheduler needs the number before the first request to check a prompt against it -- so a
        launch that leaves the option out is refused here rather than given a number invented for
        it. The refusal is a warning and the serialized path, because this plane serves fine
        without a scheduler; it is the *scheduler* that cannot answer the question.
        """
        context = int(self.args.max_model_len or 0)
        if context <= 0:
            raise ConfigurationError(
                "--enable-batching needs --max-model-len: the scheduler checks a prompt against the "
                "context before the legacy runtime's own bound is known, and that runtime has no "
                "number to give it before it is loaded"
            )
        return RuntimeSpec(
            name=self.name,
            start=self._start_runtime,
            eos_tokens=self._eos_tokens,
            max_context=context,
            # The src/ runtime places its own tensors: `torch.cuda.set_device(runtime["local_rank"])`
            # is the first thing the payload runner does, on the thread that runs it. There is no
            # separate card for this side to bind, and binding one would be a claim about where the
            # weights are that this adapter does not make.
            device=-1,
            wants_request=True,
        )

    def _eos_tokens(self) -> set[int]:
        """The ids that end a turn, from the tokenizer.

        The legacy runtime is handed `tokenizer.eos_token_id` for the same purpose (see
        ``_run_payload``'s call to ``executor.run``), so this reads the same field rather than
        looking for one in the checkpoint config: a scheduler that stopped on a different id from
        the runtime would count a row as finished somewhere the generation did not.
        """
        tokenizer = self._tokenizer()
        ids = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        if isinstance(ids, int):
            return {int(ids)}
        if isinstance(ids, (list, tuple, set)):
            return {int(one) for one in ids}
        return set()

    def _start_runtime(
        self,
        *,
        request_id: int,
        prompt_ids: Sequence[int],
        sampling: Any,
        context: Any,
        on_token: Callable[[int], None],
        on_step: Callable[[], bool],
    ) -> None:
        """One generation, driven a step at a time by the scheduler.

        The legacy runtime has two shapes and this takes the incremental one: `submit` blocks and
        returns a whole completion, `submit_stream` yields an event per decode step and a final
        `done` event. The scheduler wants a token at a time, so the events are what this reads --
        and they are the same events the streaming HTTP route reads, which is where the answer
        parity below comes from.

        The payload is the serial path's own builder, from `context` rather than from a
        `GenerationRequest` the caller supplied, because there is no caller here -- the scheduler
        is one. `context` *is* the request this row was submitted for, so both routes render the
        same prompt, the same sampler and the same thinking mode by construction.

        The step boundary is the scheduler's: one event is one decode step, so `on_step` is asked
        between events and a refusal stops consuming. That is also the cancellation seam -- a row
        the scheduler retired and a client that disconnected both arrive here, and both leave the
        generator unread rather than interrupting a kernel, which is the boundary this runtime has
        always had (see `cancel`).
        """
        self._ensure_loaded()
        payload = self._payload(context, stream=True)
        cancelled_at = cancel_key(context, request_id)
        events = self._serving_engine.submit_stream(payload)

        for event in events:
            if not isinstance(event, Mapping) or event.get("type") != "token":
                # `done` carries the runtime's own completion and its timings, and neither is
                # needed here: the tokens are the events, and the scheduler measures the phases
                # itself. Reaching it also means the run was not stopped at a boundary, which is
                # the ordinary end of a row that the scheduler did not retire.
                continue
            for token in event.get("token_ids") or ():
                on_token(int(token))
            if on_step() or self._is_cancelled(cancelled_at):
                break

    def _batched_result(self, request: GenerationRequest, result: Any) -> GenerationResult:
        """A scheduler result as this backend's own, through the legacy path's own formatter.

        Overridden rather than inherited, because this plane's answer is not `decode(token_ids)`:
        the serial path returns the *parsed* assistant message -- reasoning split out of the
        answer, tool-call markup removed, `stop` strings applied -- and it returns no token ids at
        all. Both are facts about this checkpoint's runtime rather than about serving, so the same
        function that builds the serial answer builds this one, and the two agree by construction
        instead of by inspection.

        One field is deliberately not the same: `timings`. The scheduler measures the two phases
        itself and reports seconds, where the legacy formatter is handed the runtime's own
        prefill/decode counters; the token counts here are the best this side has (`len(prompt)`
        and the tokens the first forward did not produce), so the derived rates are close and not
        identical. The tokens, the text, the reasoning, the tool calls and the finish reason are.
        """
        from src.models.deepseek_v4.serving import _format_completion_result

        token_ids = [int(token) for token in result.generated_tokens]
        payload = self._payload(request, stream=False)
        prompt_ids = self._tokenize(request)
        mapping = _format_completion_result(
            self._tokenizer(),
            str(request.metadata.get("thinking_mode", "chat")),
            prompt_ids,
            token_ids,
            float(result.prefill_seconds),
            float(result.decode_seconds),
            len(prompt_ids),
            max(0, len(token_ids) - 1),
            stop=payload.get("stop"),
            max_tokens=payload.get("max_tokens"),
        )
        return self._result_from_mapping(request, mapping)

    def metrics(self) -> dict[str, float]:
        """The live scheduler's admission state, or nothing when there is no scheduler."""
        return self.scheduler_metrics()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self._ensure_loaded()
        if self._batching():
            return self._generate_batched(requests)
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

    def prepare(self) -> None:
        """Eagerly load the model and serving engine for supervised rank 0."""
        self._ensure_loaded()

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Enter the Torch runtime worker loop for supervised ranks > 0."""
        self._ensure_open()
        runtime = self._load_runtime()
        rank = int(runtime.get("rank", os.getenv("RANK", "0")))
        if rank == 0:
            raise RuntimeError("run_worker must not be called on rank 0")
        if on_ready is not None:
            on_ready()
        # Delegate to the existing Gloo/NCCL broadcast worker protocol.
        from src.models.deepseek_v4.serving import _worker_loop

        _worker_loop(runtime)

    def close(self) -> None:
        # Before the serving engine, because the scheduler's thread runs this runtime's generation:
        # a scheduler stopped after the engine it drives is a thread reading from a closed queue.
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is not None:
            scheduler.stop()
        if self._serving_engine is not None and hasattr(self._serving_engine, "close"):
            self._serving_engine.close()
        super().close()
