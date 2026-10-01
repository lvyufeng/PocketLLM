"""Backend-neutral public types for PocketLLM.

These types describe user intent and observable results only.  They deliberately
contain no Torch, CUDA, NCCL, ACL, or backend-specific buffer types -- and, since
the rebuild, no *card list* either.  The multi-card world left two fields behind
that named a world this tree does not run: ``tensor_parallel_size``/``_rank``,
and ``device_ids``.  All three are gone, and with them the ``from_env`` bridge
and the ``device_hint`` migration message that existed only to explain the
``--device 2`` spelling to callers of the deleted ``--device-ids``.

What replaces them is one field, ``device``, holding one device in the spelling
the engine already parses: ``cpu``, ``cuda:1``, ``qnn:0``, or ``auto``.  The set
of accepted values is no longer a constant in this module -- it is read from the
backend registry, so ``--device`` offers exactly the kinds this install has a
backend for, and a third-party backend's kind appears in ``--help`` without an
edit here.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pocketllm.backends import registry

from .errors import ConfigurationError

#: Backend names, straight from the registry rather than a second list.  A name
#: is what ``--backend`` takes; ``auto`` asks the build.
_BACKENDS = frozenset({"auto"}) | frozenset(registry.BACKENDS)

#: A checkpoint's container format.  ``safetensors`` was listed in the old tree
#: with no code path that read one; it is dropped rather than kept as a value
#: that parses and then fails, which is the worse of the two ways to advertise a
#: format.
_FORMATS = {"auto", "gguf"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def device_kinds() -> tuple[str, ...]:
    """``auto`` plus every device kind a shipped backend declares.

    Read from the registry so that this is one source of truth: a build with no
    ``qnn`` backend does not offer ``--device qnn``, and a third-party backend
    that registers ``s600`` makes it selectable here without this module knowing
    the name.
    """
    kinds = {entry.factory().device_kind for entry in registry._entries()}
    return ("auto",) + tuple(sorted(kinds))


def _coerce_stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        result = tuple(item for item in value if isinstance(item, str) and item)
        if len(result) != len(value):
            raise ConfigurationError("stop must contain only non-empty strings")
        return result
    raise ConfigurationError("stop must be a string, a list of strings, or null")


@dataclass(slots=True)
class EngineArgs:
    """Validated engine construction options.

    One device, one process.  The two places the old tree said otherwise are the
    two this one used to have: a rank index, and a *list* of cards.  Both are
    gone, and :attr:`device` is the single answer.
    """

    model: str = ""
    backend: str = "auto"
    tokenizer_path: str | None = None
    config_path: str | None = None
    model_format: str = "auto"
    #: The device this process runs on: ``auto``, or a kind and optional index in
    #: the engine's own spelling (``cpu``, ``cuda:1``, ``qnn:0``).  A *card* used
    #: to be expressible here; it no longer is, because one process taking one
    #: card is what makes the spelling unambiguous, and there is nothing to
    #: disambiguate it from.
    device: str = "auto"
    max_model_len: int | None = None
    dtype: str | None = None
    kv_cache_dtype: str = "auto"
    prefill_chunk_tokens: int = 0
    enable_prefix_caching: bool = True
    attention_window: int = 0
    attention_sink_tokens: int = 0
    speculative_method: str | None = None
    speculative_tokens: int = 1
    max_batch_size: int = 1
    #: Whether to run the batch scheduler instead of the serialized session.
    #:
    #: ``None`` is "the backend's default" rather than "off": a backend that owns
    #: a scheduler chooses the batch path, and one that does not ignores the flag.
    #: It is deliberately not ``False`` by default, because ``False`` and "nobody
    #: asked" have to be distinguishable -- ``--no-enable-batching`` next to a
    #: batch width is an operator contradicting themselves, and a default of
    #: ``False`` would make that indistinguishable from the ordinary case.
    enable_batching: bool | None = None
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = str(self.backend).lower()
        self.model_format = str(self.model_format).lower()
        self.kv_cache_dtype = str(self.kv_cache_dtype).lower()
        if self.backend not in _BACKENDS:
            raise ConfigurationError(
                f"backend must be one of {sorted(_BACKENDS)}, got {self.backend!r}"
            )
        if self.model_format not in _FORMATS:
            raise ConfigurationError(
                f"model_format must be one of {sorted(_FORMATS)}, got {self.model_format!r}"
            )
        if not self.model and not self.backend_options.get("checkpoint_dir"):
            raise ConfigurationError("model/checkpoint path is required")
        if self.device not in device_kinds():
            # The parse itself is ``Device.parse``; this check only needs the
            # *name* to be one this build can serve, so it does not import the
            # ABI -- it compares against the registry's kinds.
            raise ConfigurationError(
                f"device must be one of {', '.join(device_kinds())}, got {self.device!r}"
            )
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ConfigurationError("max_model_len must be positive")
        if self.prefill_chunk_tokens < 0:
            raise ConfigurationError("prefill_chunk_tokens must be >= 0")
        if self.attention_window < 0 or self.attention_sink_tokens < 0:
            raise ConfigurationError("attention window and sink tokens must not be negative")
        if self.attention_window == 0 and self.attention_sink_tokens:
            raise ConfigurationError("attention_sink_tokens requires attention_window")
        if self.speculative_tokens < 1:
            raise ConfigurationError("speculative_tokens must be >= 1")
        if self.max_batch_size < 1:
            raise ConfigurationError("max_batch_size must be >= 1")
        if self.enable_batching is False and self.max_batch_size > 1:
            # Both fields are the operator's, and they disagree: a width is a
            # request for a scheduler that runs that many rows, and refusing the
            # scheduler is a request for the serialized session, which runs one.
            raise ConfigurationError(
                f"max_batch_size={self.max_batch_size} asks for a batch width and enable_batching is "
                f"False; a serialized session runs one request at a time. Drop one of the two"
            )

    @property
    def checkpoint_dir(self) -> str:
        """Return the checkpoint path under either supported spelling."""
        return self.model or str(self.backend_options.get("checkpoint_dir", ""))

    @classmethod
    def from_env(cls, model: str | None = None, **overrides: Any) -> "EngineArgs":
        """Build options from the environment.

        A compatibility bridge, not the preferred path.  The variables that only
        ever existed for a multi-process world -- ``TENSOR_PARALLEL_SIZE``,
        ``TP_WORLD``, ``TP_RANK``, ``POCKETLLM_DEVICE_IDS`` -- are not read
        here, and reading them anyway would be the worst of both: an operator
        sets one, the engine ignores it, and nothing says so.
        """
        values: dict[str, Any] = {
            "model": model or os.getenv("POCKETLLM_MODEL", os.getenv("CKPT_PATH", "")),
            "backend": os.getenv("POCKETLLM_BACKEND", "auto"),
            "tokenizer_path": os.getenv("TOKENIZER_PATH") or None,
            "config_path": os.getenv("CONFIG_PATH") or os.getenv("CONFIG") or None,
            "model_format": os.getenv("CKPT_FORMAT", "auto"),
            "device": os.getenv("DEVICE") or "auto",
            "max_model_len": _env_int("MAX_MODEL_LEN", 0) or None,
            "dtype": os.getenv("DTYPE") or None,
            "kv_cache_dtype": os.getenv("KV_CACHE_DTYPE", "auto"),
            "prefill_chunk_tokens": _env_int("PREFILL_CHUNK_TOKENS", 0),
            "enable_prefix_caching": _env_bool("ENABLE_PREFIX_CACHING", True),
            "attention_window": _env_int("QWEN_ATTENTION_WINDOW", 0),
            "attention_sink_tokens": _env_int("QWEN_ATTENTION_SINK_TOKENS", 0),
            "speculative_method": os.getenv("SPECULATIVE_METHOD") or None,
            "speculative_tokens": _env_int("SPECULATIVE_TOKENS", 1),
            "max_batch_size": _env_int("MAX_BATCH_SIZE", 1),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(slots=True)
class SamplingParams:
    """Per-request sampling and stopping options.

    ``temperature <= 1e-5`` means greedy generation, matching the established
    behavior of both existing runtimes.
    """

    #: The generation budget, or ``None`` when the client asked for no cap of its
    #: own.  ``None`` is not a default: a number invented here truncates an answer
    #: the model was still writing, and the caller never asked for that length.
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    seed: int | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: tuple[str, ...] = ()
    n: int = 1
    logprobs: bool = False
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stop = _coerce_stop(self.stop)
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ConfigurationError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ConfigurationError("temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigurationError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ConfigurationError("top_k must be >= 1")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ConfigurationError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ConfigurationError("repetition_penalty must be positive")
        if self.n < 1:
            raise ConfigurationError("n must be >= 1")
        if self.top_logprobs is not None and not 0 <= self.top_logprobs <= 20:
            raise ConfigurationError("top_logprobs must be in [0, 20]")
        if self.logprobs and self.top_logprobs is None:
            self.top_logprobs = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 1.0e-5

    def token_budget(self, available: int) -> int:
        """The number of tokens to generate, given the positions the model can still hold.

        ``available`` is the caller's own context minus the prompt, so a caller
        reaches this having already decided whether an explicit ``max_tokens``
        fits: an explicit one is handed back unchanged and the caller's length
        check keeps the last word on it.  An absent one resolves to everything
        that is left, which is how both engines read an absent cap -- vLLM to
        ``max_model_len - input_length``, SGLang to ``max_req_len - input_len -
        1``, one position reserved -- and the answer then ends at EOS or at the
        context limit, whichever comes first.

        The floor of one is for the case where the prompt alone overruns the
        context: there is nothing left to derive from, and asking for one token
        hands the refusal to the caller's own length check instead of reporting a
        generation that produced nothing.
        """
        if self.max_tokens is None:
            return max(1, int(available))
        return int(self.max_tokens)

    @classmethod
    def from_openai(cls, body: Mapping[str, Any]) -> "SamplingParams":
        """Normalize OpenAI-compatible request fields into one typed object."""
        # ``max_completion_tokens`` is the current spelling and wins over the
        # deprecated ``max_tokens``, which is OpenAI's rule for the pair.  A null
        # for either key means the same as leaving it out -- clients do send
        # ``"max_tokens": null`` for "no cap" -- so both spellings of absence
        # have to land on None rather than on a number this function made up.
        cap = body.get("max_completion_tokens")
        if cap is None:
            cap = body.get("max_tokens")
        max_tokens = None if cap is None else int(cap)
        known = {
            "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
            "min_p", "seed", "repetition_penalty", "frequency_penalty",
            "presence_penalty", "stop", "n", "logprobs", "top_logprobs",
            "response_format",
        }
        return cls(
            max_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0) or 0.0),
            top_p=None if body.get("top_p") is None else float(body["top_p"]),
            top_k=None if body.get("top_k") is None else int(body["top_k"]),
            min_p=None if body.get("min_p") is None else float(body["min_p"]),
            seed=None if body.get("seed") is None else int(body["seed"]),
            repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
            frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
            presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
            stop=body.get("stop"),
            n=int(body.get("n", 1) or 1),
            logprobs=bool(body.get("logprobs", False)),
            top_logprobs=None if body.get("top_logprobs") is None else int(body["top_logprobs"]),
            response_format=body.get("response_format"),
            extra={str(k): v for k, v in body.items() if k not in known},
        )

    def to_generation_options(self) -> dict[str, Any]:
        """Return the sampling knobs as one mapping, for a backend to apply."""
        options = {
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }
        options.update(self.extra)
        return options


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: How many of ``prompt_tokens`` a backend answered out of a prompt cache
    #: instead of running.  A subset of ``prompt_tokens`` and not a discount on
    #: it: a client that bills per token still bills the prompt, and one that
    #: wants to know what caching bought it reads this.  Zero means either that
    #: nothing was reused or that this backend does not report reuse, so it is
    #: only ever emitted when nonzero -- a cold response stays byte-identical to
    #: the one the API returned before the field existed.
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens:
            body["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
        return body


@dataclass(slots=True)
class TimingMetrics:
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    total_seconds: float = 0.0
    ttft_seconds: float = 0.0
    tpot_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TimingMetrics":
        values = values or {}
        prefill = float(values.get("prefill_time", values.get("prefill_s", 0.0)) or 0.0)
        decode = float(values.get("decode_time", values.get("decode_s", 0.0)) or 0.0)
        total = float(values.get("total_time", prefill + decode) or (prefill + decode))
        ttft = float(values.get("ttft", prefill) or prefill)
        tpot = float(values.get("tpot", decode) or decode)
        return cls(prefill, decode, total, ttft, tpot)

    def as_dict(self) -> dict[str, float]:
        return {
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "total_seconds": self.total_seconds,
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
        }


@dataclass(slots=True)
class GenerationRequest:
    """A normalized request consumed by a backend."""

    prompt: str | None = None
    prompt_tokens: list[int] | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt is None and self.prompt_tokens is None:
            raise ConfigurationError("either prompt or prompt_tokens is required")
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ConfigurationError("prompt must be a string")
        if self.prompt_tokens is not None:
            self.prompt_tokens = [int(token) for token in self.prompt_tokens]
            if not self.prompt_tokens:
                raise ConfigurationError("prompt_tokens must not be empty")
        if not self.request_id:
            raise ConfigurationError("request_id must not be empty")


@dataclass(slots=True)
class GenerationResult:
    request_id: str
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    timings: TimingMetrics = field(default_factory=TimingMetrics)
    logprobs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenEvent:
    request_id: str
    token_id: int | None = None
    text: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Which of the request's choices this event belongs to, 0 for the only one.
    #:
    #: A backend never sets this: it is streamed one request, and a request is one
    #: choice.  It is the host's fan-out that tags events, because a client that
    #: asked for several choices receives them as one response and has to be told
    #: which choice each chunk belongs to.  Zero is the right default for every
    #: other reader -- a single-choice stream is choice 0.
    choice_index: int = 0


@dataclass(slots=True)
class BackendCapabilities:
    """Feature declaration used for deterministic routing and API errors."""

    name: str
    models: tuple[str, ...] = ()
    model_formats: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    supports_batch: bool = False
    supports_streaming: bool = True
    supports_cancellation: bool = False
    supports_embeddings: bool = False
    supports_logprobs: bool = False
    supports_structured_outputs: bool = False
    supports_prefix_caching: bool = False
    supports_speculative_decoding: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def supports(self, feature: str) -> bool:
        value = getattr(self, feature, False)
        return bool(value)


@dataclass(slots=True)
class HealthStatus:
    status: str
    backend: str
    ready: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.status not in {"dead", "stopped"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "ready": self.ready,
            "alive": self.alive,
            "message": self.message,
            **self.details,
        }