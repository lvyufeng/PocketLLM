"""Optional native C++ backend adapter.

The native module is deliberately optional.  A CUDA/Ascend build can provide
``pocketllm_cpp`` without making the public Python package depend on a vendor
SDK at import time.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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
from pocketllm.choices import SEED_MASK
from pocketllm.protocol import (
    apply_stop_to_text,
    encode_chat_prompt,
    normalize_tool_calls,
    render_fallback_prompt,
)
from pocketllm.protocol.contract import CHAT, FieldRefusal, ServedFields, audit as audit_body
from pocketllm.protocol.contract import structured_output_spec
from pocketllm.protocol.logprobs import complete as ranking_complete
from pocketllm.protocol.logprobs import render as render_logprobs
from pocketllm.protocol.templating import build_templater, detect_architecture

from .base import BackendBase, hold_back, settled_text
from .capabilities import declared_capabilities
from .runtime_engine import card_for_rank, visible_card_count


_NATIVE_MODULE_NAMES = ("pocketllm_cpp", "_pocketllm_cpp", "cpp_engine")

#: The "not built yet" marker for the lazily built answer reader.  ``None`` is a real answer -- no
#: tokenizer, so no reading -- and has to stay distinguishable from it.
_READER_UNSET = object()


def _suffix(text: str, prefix: str) -> str:
    """``text`` with ``prefix`` removed, or all of ``text`` when it does not start that way.

    The running decode grows monotonically, so the fallback is the reading that cannot lose an
    answer: a stream may repeat a piece, which a client can see, but never silently drop one.
    """
    return text[len(prefix):] if text.startswith(prefix) else text


#: Rows the batch scheduler runs when the batch path is on and nobody named a width. Eight is what
#: `enable_batching` has meant since it existed, and the engine sizes its KV cache for this many
#: slots at construction, so it is the number to look at when a checkpoint stops fitting.
DEFAULT_BATCH_SLOTS = 8


def _preload_torch_runtime() -> None:
    """Let torch bind its own NCCL before the native module loads one.

    The extension links its own libnccl.  If it is imported first, that copy
    wins the global symbol lookup and a newer ``libtorch_cuda.so`` can then fail
    on a symbol it does not provide (observed: ``undefined symbol:
    ncclCommResume``).  Importing torch first is enough to fix the order.  Torch
    is optional for this backend, so a missing or broken install is ignored here
    and reported later by whatever actually needs it.
    """
    try:
        importlib.import_module("torch")
    except Exception:
        return


def load_native_module() -> Any:
    _preload_torch_runtime()
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


# Architectures whose GGUF export the native reader can open. The other engine
# reads a safetensors index, so a single-file GGUF reaches Qwen3.5 only; the
# registry canonicalizes the file's own ``general.architecture`` (``qwen35``)
# onto this key, which is why the two spellings never have to meet.
_GGUF_ARCHITECTURES = frozenset({"qwen3_5"})


def gguf_checkpoint_file(path: str) -> str:
    """The one GGUF file a checkpoint path names, or an empty string.

    The native reader holds a whole checkpoint in one file, so a directory of
    shards is not a layout it can open. A directory holding two models is not a
    question a listing order should answer either -- naming the file is the
    caller's decision -- so both answer empty and the caller refuses rather than
    serving whichever file happened to sort first.
    """
    if not path:
        return ""
    try:
        from src.loader.gguf.bundle import resolve_gguf_bundle

        resolved = [str(item) for item in resolve_gguf_bundle(path)]
    except Exception:
        # An unreadable or absent path is not this function's error to report:
        # the loader names it precisely when it opens the checkpoint.
        return ""
    return resolved[0] if len(resolved) == 1 else ""


def gguf_is_servable(path: str) -> bool:
    """Whether the native adapter can open the GGUF this checkpoint path names.

    Both halves are read from the artifact: that there is exactly one file, and
    that its architecture is one the registry routes to a reader that opens a
    GGUF. Asking the registry rather than a name is what keeps this answer equal
    to what ``create_engine`` will do with the same path.
    """
    file = gguf_checkpoint_file(path)
    if not file:
        return False
    try:
        native = load_native_module()
    except BackendUnavailableError:
        return False
    detect = getattr(native, "detect_architecture", None)
    if detect is None:
        return False
    try:
        return str(detect(file)) in _GGUF_ARCHITECTURES
    except Exception:
        return False


def _native_kv_cache_dtype(value: str) -> str:
    """Resolve the public ``auto`` value to the native Qwen default."""
    normalized = str(value or "auto").lower()
    return "fp16" if normalized == "auto" else normalized


def _coerce_eos_ids(value: Any, *, source: str) -> tuple[int, ...]:
    """Normalize a configured EOS value into a tuple of token ids."""
    if value is None:
        return ()
    if isinstance(value, bool):
        raise ConfigurationError(f"{source} eos_token_id must be an integer or list of integers")
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigurationError(f"{source} eos_token_id list must contain integers only")
            ids.append(int(item))
        return tuple(dict.fromkeys(ids))
    raise ConfigurationError(f"{source} eos_token_id must be an integer or list of integers")


def _checkpoint_eos_ids(checkpoint_dir: str) -> tuple[tuple[int, ...], str]:
    """Read the EOS ids a checkpoint declares, preferring generation_config.json."""
    if not checkpoint_dir:
        return (), ""
    for name in ("generation_config.json", "config.json"):
        path = os.path.join(checkpoint_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        ids = _coerce_eos_ids(data.get("eos_token_id"), source=name)
        if ids:
            return ids, name
    return (), ""


def _strip_terminal_stop_token(result: Any) -> list[int]:
    """The answer without the stop token the engine returns as part of its sequence.

    A stop token is the last token of a sequence and the engine counts and returns it, because its KV
    cache has to agree with what it reports. It is not part of the answer. The deleted C++ front end
    dropped it, before detokenizing and before counting `completion_tokens`, in its own
    `strip_stop_token`; the batch scheduler's own streaming path does the same by never emitting it.
    The non-streaming result is the one place it leaked through, so a request answered through
    `pocketllm serve` came back with a visible `<|im_end|>` on the end that the same request through
    the native binary did not have. Caught by the `cpp` served-path fixture, which is recorded from
    the serial path and compares token ids.

    A structured-output terminal token is also reported as "stop" -- it closes the JSON -- so it is
    preserved. `constraint_completed` is what says which of the two this is, and it is read with
    `getattr` because a result object from a build older than that field should not raise here.
    """
    tokens = list(result.generated_tokens)
    if not tokens or result.finish_reason != "stop":
        return tokens
    if bool(getattr(result, "constraint_completed", False)):
        return tokens
    return tokens[:-1]


def _ranking_width(params: SamplingParams) -> int:
    """How wide a ranking this request asks the engine to produce, 0 for none.

    The two endpoints spell the field differently and this is where the two spellings meet. Chat
    takes a boolean and puts the count of ranked alternatives in ``top_logprobs``;
    ``/v1/completions`` takes one count in ``logprobs`` where even 0 asks for the sampled token's own
    probability. Either way a request that asked for a ranking wants at least the sampled token, so
    the width is never 0 once it has asked -- which is exactly what the native front end's
    ``LogprobsSpec.width = max(alternatives, 1)`` says.

    The alternatives *rendered* is a separate number, and smaller: the engines rank a whole batch at
    the widest width any row in the step asked for, so an entry can carry more candidates than this
    request wants. :func:`pocketllm.protocol.logprobs.render` narrows it.
    """
    if not params.logprobs:
        return 0
    return max(int(params.top_logprobs or 0), 1)


#: The sampling values the adapter constructs the engine with. They are the engine's effective
#: values on any path where it cannot be asked -- see :func:`_engine_sampling_refusal`.
_ENGINE_SAMPLING_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 20,
    "sampling_seed": 0,
}


@dataclass(frozen=True, slots=True)
class _EngineSampling:
    """What an engine applies on its own, in the terms the sampling checks ask in.

    Read off the engine's capability object when there is one to ask, which is the only thing that
    knows what this build can vary per request. The serialized session has no scheduler and so no
    capability object to read, and its values are the ones the adapter constructed the engine with
    -- the same dict it hands that engine, so the two cannot drift.
    """

    per_request_sampling: bool = False
    per_request_top_k: bool = False
    fixed_temperature: float = 0.0
    fixed_top_p: float = 1.0
    fixed_top_k: int = 0
    fixed_seed: int = 0

    @classmethod
    def of_engine(cls, engine: Any) -> "_EngineSampling":
        return cls(
            per_request_sampling=bool(getattr(engine, "per_request_sampling", False)),
            per_request_top_k=bool(getattr(engine, "per_request_top_k", False)),
            fixed_temperature=float(getattr(engine, "fixed_temperature", 0.0)),
            fixed_top_p=float(getattr(engine, "fixed_top_p", 1.0)),
            fixed_top_k=int(getattr(engine, "fixed_top_k", 0)),
            fixed_seed=int(getattr(engine, "fixed_seed", 0)),
        )

    @classmethod
    def of_serialized_session(cls) -> "_EngineSampling":
        """The serialized session, whose sampler is fixed at what the engine was built with.

        ``per_request_sampling`` is false because that is what the path *is*: the session hands the
        engine a prompt and a token budget, one request at a time, with no per-row sampler in it to
        carry a temperature to. Stating that here rather than leaving the engine's limits unstated
        is the difference between a request that is refused and one generated as if it had not been
        made.
        """
        return cls(
            fixed_temperature=float(_ENGINE_SAMPLING_DEFAULTS["temperature"]),
            fixed_top_p=float(_ENGINE_SAMPLING_DEFAULTS["top_p"]),
            fixed_top_k=int(_ENGINE_SAMPLING_DEFAULTS["top_k"]),
            fixed_seed=int(_ENGINE_SAMPLING_DEFAULTS["sampling_seed"]),
        )


def _sampling_body(params: SamplingParams) -> dict[str, Any]:
    """A typed request as the body keys the field contract reads.

    Only the fields with a body spelling are emitted, and only when they are set: an absent key is
    "the caller did not ask", which is what the contract's default-value tests are about, so writing
    ``"top_p": None`` here would be the same answer by a longer route. ``n`` and ``logprobs`` are
    emitted unconditionally because their defaults (1, false) are the values the contract accepts
    anyway, and emitting them keeps this mapping total rather than dependent on which defaults
    :class:`SamplingParams` happens to have.

    ``logit_bias`` is the one field with nowhere else to live: it is not a typed field of
    :class:`SamplingParams`, so a caller reaches it through ``extra``, which is where the reader
    below looks.
    """
    body: dict[str, Any] = {
        "n": params.n,
        "logprobs": params.logprobs,
        "frequency_penalty": params.frequency_penalty,
        "presence_penalty": params.presence_penalty,
        "repetition_penalty": params.repetition_penalty,
    }
    logit_bias = params.extra.get("logit_bias")
    if logit_bias is not None:
        body["logit_bias"] = logit_bias
    if params.stop:
        body["stop"] = list(params.stop)
    if params.top_logprobs is not None:
        body["top_logprobs"] = params.top_logprobs
    if params.min_p is not None:
        body["min_p"] = params.min_p
    if params.top_p is not None:
        body["top_p"] = params.top_p
    if params.top_k is not None:
        body["top_k"] = params.top_k
    if params.seed is not None:
        body["seed"] = params.seed
    if params.temperature:
        body["temperature"] = params.temperature
    if params.response_format is not None:
        body["response_format"] = params.response_format
    return body


def _engine_sampling_refusal(body: Mapping[str, Any], sampling: "_EngineSampling") -> FieldRefusal | None:
    """Refuse sampling the engine cannot vary per request, ported from the native server.

    A request naming a temperature, a ``top_p``, a ``top_k`` or a seed the engine will not apply is
    answered with something else, and the caller has no way to see that. Refusing it is the only
    answer that cannot be mistaken for the one that was asked for. A value that *is* what the
    engine does anyway costs nobody anything, and clients send the defaults explicitly all the time,
    so the comparison is against the engine's effective values rather than the presence of a field.

    ``temperature`` is compared whatever the sampling mode, because it decides the mode: a request
    asking for 0.7 against an engine fixed at greedy is not a mismatch of degree. The rest are
    compared only when the request is stochastic, because under greedy decoding none of them can
    change a token -- naming a ``top_p`` the engine does not have is not a disagreement when the
    distribution is never drawn from.

    ``top_k`` and ``seed`` are separate flags from ``per_request_sampling`` because the sampler
    varies temperature, ``top_p`` and seed per request but has no top-k stage at all, so an engine
    can apply most of a request and not all of it.
    """

    def mismatch(field: str, requested: Any, configured: float) -> FieldRefusal:
        return FieldRefusal.build(
            field,
            requested,
            f"this engine's effective {field} is {configured:g} and it cannot apply "
            f"{field}={requested} to this request.",
            "Omit the field to accept the effective value, or run an engine configuration that "
            "supports it.",
        )

    def named(field: str) -> Any | None:
        value = body.get(field)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    temperature = named("temperature")
    if not sampling.per_request_sampling and temperature is not None:
        if abs(float(temperature) - sampling.fixed_temperature) > 1e-5:
            return mismatch("temperature", temperature, sampling.fixed_temperature)

    stochastic = sampling.fixed_temperature > 1e-5 or (
        sampling.per_request_sampling
        and temperature is not None
        and float(temperature) > 1e-5
    )

    top_p = named("top_p")
    if (
        not sampling.per_request_sampling
        and stochastic
        and top_p is not None
        and abs(float(top_p) - sampling.fixed_top_p) > 1e-5
    ):
        return mismatch("top_p", top_p, sampling.fixed_top_p)

    top_k = named("top_k")
    if (
        not sampling.per_request_top_k
        and stochastic
        and top_k is not None
        and int(float(top_k)) != sampling.fixed_top_k
    ):
        return mismatch("top_k", top_k, float(sampling.fixed_top_k))

    seed = named("seed")
    if (
        not sampling.per_request_sampling
        and stochastic
        and seed is not None
        and int(float(seed)) != sampling.fixed_seed
    ):
        return mismatch("seed", seed, float(sampling.fixed_seed))

    # Several choices are several requests, run one level up by the host's dispatch, so the count
    # itself is never this adapter's to refuse. What is left to refuse is a request those runs
    # cannot answer: they differ only in the seed they are given, and an engine that samples at
    # engine-wide values reads no seed it was handed. Every run would then draw from the same
    # distribution at the same seed and the client would receive one text presented as `n`
    # independent samples, with nothing in the response to tell it what happened. Under greedy
    # decoding the same text `n` times is the answer that was asked for, so this refuses only the
    # stochastic case -- the same line the native front end drew.
    n = named("n")
    if n is not None and int(n) > 1 and stochastic and not sampling.per_request_sampling:
        return FieldRefusal.build(
            "n", n,
            f"this engine samples at an engine-wide temperature of "
            f"{sampling.fixed_temperature:g} and a seed it cannot vary per request, so all "
            f"{int(n)} choices would be the same text presented as independent samples.",
            'Remove "n", or run an engine configuration that samples per request.',
        )
    return None


class CppBackend(BackendBase):
    """Native C++ engine adapter with optional scheduler-backed batching.

    The native runtime keeps ownership of its optimized KV layout and tensor-
    parallel protocol. This adapter handles model selection, request conversion,
    lifecycle, and the serial or scheduler result path without turning an engine
    into a Torch module or copying its buffers.
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
        if engine is None and args.tensor_parallel_size > 1:
            nccl_id_path = str(args.backend_options.get("nccl_id_path", ""))
            if not nccl_id_path:
                nccl_id_path = os.environ.get("POCKETLLM_NCCL_ID_PATH", "")
            if not nccl_id_path:
                raise ConfigurationError(
                    "C++ tensor-parallel backend requires nccl_id_path"
                )
        self._tokenizer_error: str | None = None
        self._engine = engine if engine is not None else self._construct_engine()
        # A primitive lock may be released by a different worker thread.  This
        # matters for AsyncLLM, which advances one generator through an executor
        # without guaranteeing that every next() call uses the same thread.
        self._request_lock = threading.Lock()
        self._tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._eos_ids, self._eos_source = self._resolve_eos_ids()
        self._answer_reader_cache: Any = _READER_UNSET
        # The vocabulary constrained decoding masks over, built on the first request that asks for
        # a schema. `_READER_UNSET` rather than `None` because "no request has needed it yet" and
        # "it could not be read" are different answers -- the first is not a failure and must not
        # be reported as one.
        self._constraint_tokenizer_cache: Any = _READER_UNSET
        self._constraint_tokenizer_error: str = ""

        # Phase 3.4: the batch scheduler, on by default and only on rank 0.  Only rank 0 drives
        # scheduling: the scheduler runs a background thread that issues collectives, and on a worker
        # rank that would race run_worker_loop() and deadlock NCCL.  A worker rank therefore reports
        # no scheduling of its own, and the width reaches it through the shared worker options.
        self._scheduler = None
        self._batching_enabled = False
        if self._batching_requested() and self.args.tensor_parallel_rank == 0:
            self._batching_enabled = self._init_batch_scheduler()

        self._ready = True

    def _resolve_eos_ids(self) -> tuple[frozenset[int], str]:
        """Resolve the stop tokens without guessing a vocabulary-specific id.

        ``backend_options["eos_token_id"]`` wins so an operator can override a
        checkpoint.  Otherwise the native engine, the tokenizer, and finally the
        checkpoint's ``generation_config.json``/``config.json`` are consulted.
        An empty result means only the token budget can end generation, which the
        adapter reports as ``finish_reason="length"``.
        """
        override = _coerce_eos_ids(self.args.backend_options.get("eos_token_id"), source="backend_options")
        if override:
            return frozenset(override), "backend_options"
        for holder, attribute, source in (
            (self._engine, "eos_id", "native engine"),
            (self._engine, "eos_token_id", "native engine"),
        ):
            if holder is None:
                continue
            value = getattr(holder, attribute, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            ids = _coerce_eos_ids(value, source=source)
            if ids:
                return frozenset(ids), source
        config = getattr(self._engine, "config", None)
        if isinstance(config, dict):
            ids = _coerce_eos_ids(config.get("eos_token_id"), source="native config")
            if ids:
                return frozenset(ids), "native config"
        # generation_config.json is consulted before the tokenizer because it is
        # what the checkpoint declares for generation. Chat checkpoints commonly
        # stop on a turn-end token that differs from the tokenizer's EOS.
        ids, name = _checkpoint_eos_ids(self.args.checkpoint_dir)
        if ids:
            return frozenset(ids), name
        value = getattr(self._tokenizer, "eos_token_id", None) if self._tokenizer is not None else None
        ids = _coerce_eos_ids(value, source="tokenizer")
        if ids:
            return frozenset(ids), "tokenizer"
        return frozenset(), ""

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_ids

    def _configured_max_batch_size(self) -> int:
        """Batch width for this backend, shared by the engine and the scheduler.

        The engine sizes its KV cache from this at construction, so the scheduler must not ask for
        more slots later.  Batching off means a single slot, which keeps the serial path's memory
        footprint unchanged.

        Three sources, the first one set winning:

        1. `backend_options["max_batch_size"]`, the programmatic spelling. It wins over the flag,
           because every benchmark in `scripts/` sets it and none of them passes a CLI flag --
           reading the flag first would silently reset those to its default of 1.
        2. `--max-batch-size`, which is how an operator asks for a width.
        3. `DEFAULT_BATCH_SLOTS`. A width of 1 is not a batch, so leaving the unspecified width at 1
           would make `--enable-batching` on its own do nothing.
        """
        if not self._batching_requested():
            return 1
        raw = self._requested_batch_width()
        if raw is None:
            return DEFAULT_BATCH_SLOTS
        try:
            requested = int(raw)
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"max_batch_size={raw!r} is not an integer; it is a row count"
            ) from None
        if requested < 1:
            raise ConfigurationError(
                f"max_batch_size={requested} is not a width; use 1 for the serialized session or "
                f"--no-enable-batching, which says the same thing"
            )
        return requested

    def _batching_requested(self) -> bool:
        """Whether the operator, or the caller, asked for the batch scheduler.

        The CLI flag is the operator's spelling and the backend option is the programmatic one; the
        flag wins when it is set, because it is the one a person typed. Neither being set means this
        backend's own default, which is **on**: the batch path is the only one that can honour a
        width above 1, and the only one that accepts a sampling policy at all -- `_check_sampling`
        refuses `top_k`, `top_p`, `min_p`, `n`, `logprobs` and `stop` on the serial path, so
        defaulting to serial would silently refuse options `pocketllm serve` advertises.
        """
        explicit = self.args.backend_options.get("enable_batching")
        if explicit is None:
            explicit = self.args.enable_batching
        return True if explicit is None else bool(explicit)

    def _batching_source(self) -> str:
        """Which place the batch decision came from, for a message about it.

        The width is a decision too, and often the one an operator actually typed:
        ``--max-batch-size`` with no ``--enable-batching`` is how the flag is spelled in every script
        under ``scripts/``, and ``--max-batch-size 8`` alone is a request the backend treats as one.
        A message naming only the backend's default would send that caller looking for a flag they
        never used.
        """
        if self.args.backend_options.get("enable_batching") is not None:
            return "from --backend-option enable_batching"
        if self.args.enable_batching is not None:
            return "from --enable-batching" if self.args.enable_batching else "from --no-enable-batching"
        if "max_batch_size" in self.args.backend_options:
            return "from --backend-option max_batch_size"
        if self.args.max_batch_size != 1:
            return f"from --max-batch-size {self.args.max_batch_size}"
        return "from this backend's default"

    def _batching_was_asked_for(self) -> bool:
        """Whether a *person or caller* asked, as opposed to this backend assuming it.

        The difference decides what an unbuildable scheduler is worth. A request that cannot be
        honoured is refused, because the caller has nowhere else for it to be answered -- that is
        #437, where `--max-batch-size 8` was accepted, clamped and silently dropped to one row. The
        backend's own assumption is not a promise to anybody: it is a claim about the build,
        `capabilities.details['scheduler']` answers it on every construction, and refusing the
        default path would make this backend unusable on a build without a scheduler -- the same
        declaration R2 exists to make uniform.
        """
        if self.args.backend_options.get("enable_batching"):
            return True
        if self.args.enable_batching:
            return True
        # A width can be refused for being unreadable, but this decides only whether somebody asked,
        # so it must not be the thing that raises: a build with no scheduler and a nonsense width
        # would otherwise report the ValueError in place of the ConfigurationError naming the width.
        try:
            return int(self._requested_batch_width() or 1) > 1
        except (TypeError, ValueError):
            return True

    def _requested_batch_width(self) -> Any:
        """The width as asked for, or `None` when neither source named one."""
        if "max_batch_size" in self.args.backend_options:
            return self.args.backend_options["max_batch_size"]
        if self.args.max_batch_size != 1:
            return self.args.max_batch_size
        return None

    def _init_batch_scheduler(self) -> bool:
        """Build the batch scheduler, or refuse the width that needs it.

        A width reaches the engine through the scheduler's own slot allocation
        and not through the engine, so a width that was asked for and a scheduler
        that could not be built are one fact and not two. Answering the request
        with the serialized session would accept the flag and serve one request
        at a time, which is the state :meth:`_unavailable_scheduler` exists to
        remove.
        """
        if not hasattr(self._native, "QwenBatchScheduler"):
            self._unavailable_scheduler(
                "this native module does not expose QwenBatchScheduler",
                "Rebuild cpp_engine with -DPOCKET_BUILD_PYTHON=ON.",
                # A missing class is a fact about the build, and the capability report carries it
                # on every construction; one warning per process for it would be noise.
                warn_when_unasked=False,
            )
            return False

        max_batch_size = self._configured_max_batch_size()
        if max_batch_size <= 1:
            return False

        try:
            self._scheduler = self._native.QwenBatchScheduler(self._engine, max_batch_size)
            return True
        except Exception as e:
            # Whoever asked for the width needs to hear that this engine cannot take it. The
            # failure is not always a type error: a `PersistentEngine` is not an `InferenceEngine`
            # in the binding, so a DeepSeek-V4 checkpoint lands here, and so would any engine the
            # scheduler's constructor rejects for its own reasons.
            self._unavailable_scheduler(
                f"the batch scheduler could not be built over this engine: {e}",
                "Drop --max-batch-size and --enable-batching, or serve a model whose engine the "
                "scheduler takes.",
                warn_when_unasked=True,
            )
            return False

    def _unavailable_scheduler(self, reason: str, remedy: str, *, warn_when_unasked: bool) -> None:
        """Report a scheduler that could not be built: refusal, warning, or silence.

        The line between the three is who asked. A person or a caller who named a
        width gets a refusal, because there is nowhere else for that request to
        be answered. This backend's own default -- which is to batch -- gets the
        answer the capability set already carries, plus one warning when the
        engine actively rejected the scheduler rather than the build lacking one.
        """
        if self._batching_was_asked_for():
            raise UnsupportedFeatureError(
                f"{reason}. The batch path was asked for {self._batching_source()}, and a width is "
                f"delivered by the scheduler rather than by the engine, so refusing is the only "
                f"honest answer. {remedy}"
            )
        if warn_when_unasked:
            import warnings

            warnings.warn(f"{reason}; running the serialized session, one request at a time")

    @staticmethod
    def native_available() -> bool:
        try:
            load_native_module()
            return True
        except BackendUnavailableError:
            return False

    def _registered_architectures(self) -> tuple[str, ...]:
        """Models this build can serve, read from the native engine registry.

        This is the same list the C++ CLI dispatches on, so a build that links a
        second engine reports it here without anyone editing a literal.  The
        fallback covers a native module predating the registry, and a
        capabilities report is not worth failing a backend over.
        """
        listed = getattr(self._native, "registered_architectures", None)
        if listed is None:
            return ("qwen3_5",)
        try:
            return tuple(str(name) for name in listed())
        except Exception:
            return ("qwen3_5",)

    def _engine_capabilities(self) -> Any | None:
        """What the engine itself declares, when there is an engine to ask.

        This is the seam the two declarations meet at. `pocket::Capabilities` is the C++ engine's
        own statement of what it can do -- written because reading `kv_total_blocks() == 0` as "this
        engine has a contiguous arena" was an inference from an accounting field rather than a
        declaration -- and until now the Python adapter re-derived the overlapping half of it by
        hand instead of asking. The scheduler is the only object that exposes it across the binding,
        so a build without one, or a fake without the method, leaves this `None` and the adapter
        falls back to what it resolved at construction.
        """
        scheduler = self._scheduler
        getter = getattr(scheduler, "engine_caps", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    @property
    def capabilities(self) -> BackendCapabilities:
        # The Ascend build rejects the external drafters, so report only what the
        # linked backend actually implements instead of the CUDA superset.
        native_backend = str(getattr(self._native, "backend", "") or "").lower()
        speculative: tuple[str, ...] = ("mtp",) if native_backend == "ascend" else ("mtp", "dspark", "dflash2")
        engine = self._engine_capabilities()
        # Every field the two declarations share is reported from the engine's, so there is one
        # answer rather than two that happen to agree.
        batch_width = (
            int(engine.max_slots) if engine is not None else self._configured_max_batch_size()
        )
        served = self._served_fields()
        return declared_capabilities(
            "cpp",
            models=self._registered_architectures(),
            devices=(native_backend,) if native_backend else ("cuda", "ascend"),
            # Whether this instance really holds more than one request is whether it built a
            # scheduler, not whether the runtime could -- that is the declaration's `supports_batch`.
            # The engine's `continuous_batching` is the same fact from the engine's side, and it is
            # published next to this under `engine_declares` rather than consulted here: the two
            # disagreeing would mean a scheduler built around an engine that cannot run it, which is
            # worth being able to see rather than silently resolving to one of the two.
            supports_batch=self._batching_enabled,
            # From the table `audit_request` refuses from, rather than stated a second time here. A
            # capability a client reads off a model list and a field that same client is refused are
            # one answer, and this is the second one that would come to differ from it: whether this
            # instance serves constrained decoding is its scheduler's answer and its engine's, and
            # neither is a fact about the runtime.
            #
            # `logprobs` is deliberately not here, and it is the one field where the two really are
            # separate questions: the ranking comes off the scheduler's result rather than out of
            # the sampler, so the path that gates it is the scheduler alone and its published
            # capability is left where that decision put it.
            supports_structured_outputs=served.structured_outputs,
            supports_speculative_decoding=speculative,
            # The serialized session resumes a repeated prompt from its per-slot cache. The batch
            # scheduler does not: it prefills through `QwenEngine::batch_prefill`, which never
            # enters the prefix lookup in `QwenEngine::prefill`. Reporting `True` on both paths
            # would be the field answering "does the cache exist" on one and "will it be used" on
            # the other, which is the drift this declaration exists to remove.
            reads_prefix_cache=bool(self.args.enable_prefix_caching) and not self._batching_enabled,
            details={
                "scheduler": "batch scheduler" if self._batching_enabled else "serialized compatibility session",
                "device_backend": native_backend or "unknown",
                "eos_token_ids": sorted(self._eos_ids),
                "eos_source": self._eos_source or "none",
                "max_batch_size": batch_width if self._batching_enabled else 1,
                "kv_paged": (
                    bool(engine.paged_kv) if engine is not None
                    else bool(self.args.backend_options.get("kv_paged", False))
                ),
                # What the engine declares that this adapter does not surface as a capability.
                # Publishing it rather than hiding it is the point of the field: an entry is a path
                # Python would have to deliver end to end before the capability could be reported,
                # and the adapter reporting `False` over an engine saying `True` is a gap to close
                # deliberately rather than a value to copy over. `structured_outputs` was on this
                # list and is not any more; `logprobs` stays for the reason given above.
                "engine_declares": self._engine_declaration(engine),
            },
        )

    @staticmethod
    def _engine_declaration(engine: Any | None) -> dict[str, Any]:
        """The engine's own answer for the facts this adapter does not report as capabilities.

        Which engine can vary sampling per request, and whether it chunks a long prompt, are engine
        facts with no counterpart in the published capability set -- there is no
        ``supports_per_request_sampling`` for a client to read, and there should not be one, because
        what a client acts on is the refusal it gets for a value the engine will not apply. So they
        are published here, beside the adapter's own answers, as the evidence for those refusals.

        ``logprobs`` is here for a third reason: it is an engine fact whose *published* counterpart
        is gated on something else -- the scheduler, not the sampler -- so the two are worth being
        able to compare rather than collapsing into one answer.
        """
        if engine is None:
            return {}
        return {
            "continuous_batching": bool(engine.continuous_batching),
            "chunked_prefill": bool(engine.chunked_prefill),
            "per_request_sampling": bool(engine.per_request_sampling),
            "per_request_top_k": bool(engine.per_request_top_k),
            "logprobs": bool(getattr(engine, "logprobs", False)),
        }

    def _load_tokenizer(self) -> Any | None:
        tokenizer_path = self.args.tokenizer_path or self.args.checkpoint_dir
        if not tokenizer_path:
            self._tokenizer_error = "no tokenizer_path and no checkpoint_dir"
            return None
        # A GGUF carries its vocabulary, its special-token ids and its chat
        # template in its own header, and the released ternary artifact is a GGUF
        # with nothing beside it, so there is no directory for transformers to
        # read. Both paths produce the same class, which is what lets the rest of
        # this adapter stay unaware of which container it is serving. An explicit
        # `tokenizer_path` still wins: a caller who names one wants that one.
        gguf = "" if self.args.tokenizer_path else gguf_checkpoint_file(self.args.checkpoint_dir)
        if gguf:
            try:
                from src.encoding.gguf_tokenizer import build_gguf_hf_tokenizer

                tokenizer, _metadata = build_gguf_hf_tokenizer(gguf)
                return tokenizer
            except Exception as exc:
                self._tokenizer_error = f"{type(exc).__name__}: {exc}"
                return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as exc:
            # Keep the reason. A token-only caller still works without a
            # tokenizer, so this is not fatal here, but swallowing it turns a
            # broken transformers/torch install into a misleading "needs
            # tokenizer_path" error at the first text prompt.
            self._tokenizer_error = f"{type(exc).__name__}: {exc}"
            return None

    def _construct_engine(self) -> Any:
        # Which engine opens a checkpoint is the checkpoint's own declaration,
        # read through the same C++ model registry the native CLI dispatches on.
        # `engine_kind` remains as an explicit override for the cases detection
        # cannot cover -- a directory without a config.json, or forcing one
        # runtime onto a checkpoint for a comparison.
        kind = str(self.args.backend_options.get("engine_kind", "auto")).lower()
        if kind == "auto":
            kind = self._detect_engine_kind()

        self._refuse_unschedulable_engine_before_loading(kind)

        if kind == "persistent":
            return self._construct_persistent_engine()
        elif kind == "qwen":
            return self._construct_qwen_engine()
        else:
            raise UnsupportedFeatureError(
                f"unsupported engine_kind: {kind}; use 'auto', 'persistent' or 'qwen'"
            )

    def _refuse_unschedulable_engine_before_loading(self, kind: str) -> None:
        """Refuse a width this engine could not honour, before the checkpoint is read.

        The same refusal :meth:`_init_batch_scheduler` makes, moved ahead of the
        model load it would otherwise land after: a DeepSeek-V4 checkpoint takes
        minutes to open, and learning at the end of that that the width was never
        going to be honoured is the cost `factory.py` refuses at parse time for
        the runtimes that declare `supports_batch=False`.

        It answers from the binding's own class objects rather than by trying, so
        it is positive evidence only: a module that does not expose both the class
        and `InferenceEngine` is left to the real attempt below. A false negative
        would refuse a working build, which is the direction this must not err in.
        """
        class_name = {"qwen": "QwenEngine", "persistent": "PersistentEngine"}.get(kind)
        if class_name is None:
            return
        cls = getattr(self._native, class_name, None)
        base = getattr(self._native, "InferenceEngine", None)
        if not isinstance(cls, type) or not isinstance(base, type):
            return
        if issubclass(cls, base):
            return
        # Silent when nobody asked: the load is going to happen anyway, and it ends in the late
        # check's warning. This one exists only to move the refusal ahead of the load, so it has
        # nothing to say on the path that does not refuse.
        self._unavailable_scheduler(
            f"{class_name} is not an InferenceEngine in this native module, so the batch "
            f"scheduler cannot be built over the engine this checkpoint selects",
            "Drop --max-batch-size and --enable-batching, or serve a model whose engine the "
            "scheduler takes.",
            warn_when_unasked=False,
        )

    # Architectures the registry knows, mapped to the native engine class this
    # backend constructs for them.  The backend builds the concrete classes
    # rather than going through create_engine() because it needs their full
    # option surface (KV dtype, drafters, prefill chunk), which the registry's
    # model-agnostic EngineOptions deliberately does not carry.
    _ENGINE_KIND_BY_ARCHITECTURE = {
        "qwen3_5": "qwen",
        "deepseek_v4": "persistent",
    }

    def _detect_engine_kind(self) -> str:
        """Ask the native registry which engine this checkpoint wants."""
        detect = getattr(self._native, "detect_architecture", None)
        if detect is None or not self.args.checkpoint_dir:
            # An older native module, or a caller that passed no checkpoint at
            # all (token-only tests).  Keep the previous default rather than
            # failing on a path that never needed detecting.
            return "qwen"
        # The registry reads a directory's config.json, and a GGUF states its
        # architecture in its own header instead -- so a directory holding one
        # names it through the file it holds, which is the same file the engine
        # will open.
        checkpoint = self._native_checkpoint()
        try:
            architecture = str(detect(checkpoint))
        except Exception as exc:
            raise ConfigurationError(
                f"could not detect the model architecture of {checkpoint}: {exc}; "
                "pass --engine-kind qwen or --engine-kind persistent to choose one"
            ) from exc

        kind = self._ENGINE_KIND_BY_ARCHITECTURE.get(architecture)
        if kind is None:
            known = ", ".join(sorted(self._ENGINE_KIND_BY_ARCHITECTURE)) or "none"
            raise UnsupportedFeatureError(
                f"no cpp backend engine for architecture "
                f"'{architecture or '<undeclared>'}' declared by {checkpoint}; "
                f"known architectures: {known}"
            )
        return kind

    def _construct_persistent_engine(self) -> Any:
        """Construct PersistentEngine (supports TP worker loop).

        Deliberately handed ``checkpoint_dir`` rather than ``_native_checkpoint()``: this engine has
        no GGUF forward at all -- it refuses a ``.gguf`` path by name with "GGUF Q2 dense forward is
        not yet wired up" -- so a directory holding one is not a layout it serves, and resolving it
        here would only change which of the two errors a caller sees.
        """
        cls = getattr(self._native, "PersistentEngine", None)
        options_cls = getattr(self._native, "ForwardSmokeOptions", None)
        if cls is None or options_cls is None:
            raise BackendUnavailableError("native module does not expose PersistentEngine bindings")

        options = options_cls()
        options.tp_world = self.args.tensor_parallel_size
        options.tp_rank = self.args.tensor_parallel_rank
        options.device = self._native_rank_device()
        options.skip_fp4_host_prepare = False
        options.nccl_id_path = str(self.args.backend_options.get("nccl_id_path", ""))

        layer_count = 0  # auto-detect from checkpoint
        max_context = self._context_tokens()

        return cls(self.args.checkpoint_dir, options, layer_count, max_context)

    def _construct_qwen_engine(self) -> Any:
        """Construct QwenEngine, which drives TP through its own worker loop."""
        cls = getattr(self._native, "QwenEngine", None)
        options_cls = getattr(self._native, "QwenEngineOptions", None)
        if cls is None or options_cls is None:
            raise BackendUnavailableError("native module does not expose QwenEngine bindings")

        # Get NCCL ID path from backend_options or environment
        nccl_id_path = str(self.args.backend_options.get("nccl_id_path", ""))
        if not nccl_id_path:
            nccl_id_path = os.environ.get("POCKETLLM_NCCL_ID_PATH", "")
        options = options_cls()
        mappings = {
            "tp_world": self.args.tensor_parallel_size,
            "tp_rank": self.args.tensor_parallel_rank,
            "device": self._native_rank_device(),
            # The KV cache is sized at construction, so the batch width must be
            # known here; the scheduler cannot grow it afterwards.
            "max_batch_size": self._configured_max_batch_size(),
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
            "nccl_id_path": nccl_id_path,
            # Paged KV (Phase 3.7). Off by default; the contiguous arena
            # reproduces the max_batch_size * max_context reservation. A zero
            # kv_cache_bytes derives the pool from that same reservation, so
            # turning paging on alone is memory-neutral until the budget is
            # raised deliberately.
            "kv_paged": bool(self.args.backend_options.get("kv_paged", False)),
            "kv_block_size": int(self.args.backend_options.get("kv_block_size", 16)),
            "kv_cache_bytes": int(self.args.backend_options.get("kv_cache_bytes", 0)),
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
        # The engine's own sampler, from the one dict that states it: the same values
        # `_EngineSampling.of_serialized_session` reads when there is no capability object to ask.
        for name, value in _ENGINE_SAMPLING_DEFAULTS.items():
            if hasattr(options, name):
                setattr(options, name, value)
        engine = cls(self._native_checkpoint(), options, 0, self._context_tokens())
        # warmup_tp() builds the command channel and forces the NCCL
        # communicator up.  Both ranks must do it before any rank issues a
        # collective, and a worker cannot enter run_worker_loop() without it.
        if self.args.tensor_parallel_size > 1:
            engine.warmup_tp()
        return engine

    def _native_checkpoint(self) -> str:
        """The checkpoint path the native reader opens, which is not always the one it was named by.

        A checkpoint directory may hold a single GGUF rather than a safetensors index -- the released
        ternary artifact is exactly that, one file and nothing else -- and the C++ reader dispatches
        on the path's own ``.gguf`` suffix. Handing it the directory reads it as an index and fails
        on the ``config.json`` it does not have, which is what ``pocketllm serve`` did on the ternary
        checkpoint until this resolved. Ask this rather than ``checkpoint_dir`` anywhere a path goes
        to the engine, so the file the engine opens is the file every other question was asked about.
        """
        checkpoint = self.args.checkpoint_dir or ""
        return gguf_checkpoint_file(checkpoint) or checkpoint

    def _native_rank_device(self) -> int:
        """Resolve this rank's card index, which the native engine takes as one integer.

        `--device-ids` answers it outright -- rank *r* takes the r-th entry -- and that is exactly
        what the launcher's ``CUDA_VISIBLE_DEVICES=$rank`` beside ``--device 0`` used to say, one
        rank and one process at a time. The pair is still honoured, because it still works: with no
        ids a TP rank claims card ``tensor_parallel_rank`` -- the supervisor gives every rank the
        same visible device list, so leaving the default 0 would stack the whole world onto one GPU
        -- and a launcher that narrowed ``CUDA_VISIBLE_DEVICES`` to one device per rank has already
        renumbered that device to 0, which is why the offset is not applied on top of it.

        Both halves live in `card_for_rank` because two other adapters ask the same question, and
        "which card is mine" answered twice is how the two answers come to differ.
        """
        return card_for_rank(
            self.args.device_ids,
            rank=self.args.tensor_parallel_rank,
            world=self.args.tensor_parallel_size,
            visible=visible_card_count(),
        )

    def health(self) -> HealthStatus:
        status = super().health()
        details = dict(status.details)
        details.update({"model": self.args.checkpoint_dir})
        return HealthStatus(status.status, status.backend, status.ready, status.message, details)

    def metrics(self) -> dict[str, float]:
        """The scheduler's own admission state, named as the native host names it.

        Both hosts of `BatchScheduler` are the same library with the same stats struct, and the
        server prefixes what it publishes, so the two agree when the *suffix* does: the native
        host's `pocket_requests_running` is this server's `pocketllm_requests_running`, and the
        comparison is a prefix substitution rather than a translation table. That is the same
        reason the histogram bucket bounds here are transcribed from vLLM's rather than chosen.

        `requests_running` is the series that matters most: it is how an operator reads that a
        `serve --backend cpp` deployment has one scheduler holding several requests at once rather
        than a lock serializing them, which is a claim about the process that no per-request log
        can make.

        Empty on the serialized path. That path has no scheduler, so it has no running set, and
        publishing zeros would say "the scheduler exists and is idle" about a process that does
        not have one.
        """
        scheduler = self._scheduler
        if not self._batching_enabled or scheduler is None:
            return {}
        try:
            stats = scheduler.get_stats()
        except Exception:
            # A metrics scrape must not fail a request path or a scrape itself: an engine that
            # cannot report is an engine whose series are absent, which a scraper reads as no data.
            return {}
        published = {
            "requests_running": float(stats.running_requests),
            "requests_waiting": float(stats.waiting_requests),
            "slots_free": float(stats.free_slots),
        }
        # The paged block gauges, and only when the engine actually pages: an unpaged engine
        # reports zeros that a scraper would read as a pool of no blocks rather than no pool.
        if bool(getattr(scheduler.engine_caps(), "paged_kv", False)):
            published.update(
                {
                    'kv_blocks{state="total"}': float(stats.total_blocks),
                    'kv_blocks{state="free"}': float(stats.free_blocks),
                    'kv_blocks{state="reserved"}': float(stats.reserved_blocks),
                    'kv_blocks{state="cache_pinned"}': float(stats.cache_pinned_blocks),
                }
            )
        return published

    def _context_tokens(self) -> int:
        """The positions this engine's caches hold, exactly as the native engine was built.

        The native engine is constructed with this same number, so a budget derived from it is a
        budget the engine can actually run.
        """
        return self.args.max_model_len or 8192

    def _prompt_ids(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        if self._tokenizer is None:
            detail = f" (tokenizer load failed: {self._tokenizer_error})" if self._tokenizer_error else ""
            raise ConfigurationError(
                f"C++ backend needs a working tokenizer for text prompts{detail}"
            )
        messages = request.metadata.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            encoded = encode_chat_prompt(
                self._tokenizer,
                messages,
                thinking_mode=str(request.metadata.get("thinking_mode", "chat")),
                reasoning_effort=request.metadata.get("reasoning_effort"),
                tools=request.metadata.get("tools"),
                add_generation_prompt=bool(request.metadata.get("add_generation_prompt", True)),
            )
            if encoded is not None:
                return encoded
            prompt = request.prompt or render_fallback_prompt(messages)
        else:
            prompt = request.prompt or ""
        encoded = self._tokenizer.encode(prompt)
        return [int(token) for token in encoded]

    def _served_fields(self) -> ServedFields:
        """Which request fields this adapter's answers actually apply.

        The capability half of the audit, and it lives beside the adapter rather than in the
        runtime table because it is not one fact about the runtime: ``stop`` is served by code in
        this file, and ``choices`` is served by no adapter at all.

        The ``False`` entries are the fields the native front end refused by name rather than
        dropping: the engine's sampler has no repetition, presence or min-p term, its token
        constraint is not reachable across the binding, and it has no ``logit_bias``. Each of those
        is a field a caller set that would not have changed the answer, which is why the answer is a
        refusal and not a 200.

        ``choices`` is ``True`` although nothing in this file generates a second choice. A request
        for ``n`` choices is ``n`` requests one level up, in the host's dispatch
        (:func:`pocketllm.choices.expanded`), and this adapter's part of serving the field is to
        answer each of them; declaring it ``False`` here would refuse the field on every runtime,
        which is the opposite of where the fan-out lives. What this file must not do is answer ``n``
        choices with one, and it cannot: it is handed one request per choice.

        ``logprobs`` is asked in two parts, and the second one is easy to leave out. The ranking
        arrives on the scheduler's result object, so a build whose scheduler was not created --
        ``batching=false`` selects the serialized compatibility session -- has no ranking anywhere
        in the call, and the field is refused by name there rather than answered with an empty
        array. But the scheduler only carries what the engine produced: the probability itself comes
        off the engine's per-row sampler, and a build without one reports ``caps().logprobs`` false
        and hands the scheduler nothing to carry. Asking the scheduler alone is what turns that into
        a 500 -- the field is declared served, the request is admitted, and the answer comes back
        ranking fewer positions than it has tokens. ``structured_outputs`` below asks the same second
        question, of the same declaration, for the same reason. Streaming logprobs stay ``False``
        for the reason the native front end refused them: a streamed chunk carries the text of its
        token with no ranking beside it, and one ranking per chunk would have to come off the same
        result object the stream is draining.

        ``structured_outputs`` is the same question as ``logprobs`` -- asked of the engine rather
        than of this file -- and it is asked in two parts. The constraint travels on the scheduler
        request, so the scheduler path is necessary; and the mask is applied by the engine's per-row
        device sampler, so the engine's own declaration is necessary too. That second part is not a
        formality: under tensor parallelism the sampler is engine-wide, and an engine that applies
        one temperature to every row applies no per-row mask either. Answering a schema there would
        return unconstrained text with a 200.
        """
        return ServedFields(
            choices=True,
            stop=True,
            logprobs=self._rankings_available(),
            structured_outputs=self._constraints_available(),
        )

    def _rankings_available(self) -> bool:
        """Whether a request for log probabilities can be answered with them.

        See :meth:`_served_fields`. Two things have to hold and neither implies the other: a
        scheduler has to exist to carry the ranking, and the engine behind it has to rank. The
        Ascend build is why the second one is here -- its sampler produces no per-position
        probability, its ``caps()`` says so, and this adapter published ``True`` anyway off the
        scheduler alone, so ``"logprobs": true`` was admitted and answered with a 500 rather than
        refused by name.

        The engine's half is read off the scheduler rather than off ``caps()`` on the engine, for
        the reason :meth:`_constraints_available` reads it there: the scheduler is what will carry
        the value, and the two answers cannot be allowed to differ.

        A scheduler with no binding to ask is read as ``False`` rather than raising, for the reason
        :meth:`_engine_capabilities` swallows the same absence: a capability report and a field
        audit are not the places to propagate "this extension is older than the caller". The
        direction is the safe one -- a runtime that has not said it ranks a position is refused by
        name, which is where every other undeclared capability points.
        """
        if not (self._batching_enabled and self._scheduler is not None):
            return False
        getter = getattr(self._scheduler, "engine_caps", None)
        if not callable(getter):
            return False
        try:
            caps = getter()
        except Exception:
            return False
        return bool(getattr(caps, "logprobs", False))

    def _constraints_available(self) -> bool:
        """Whether a request naming a schema can actually be held to it on this path.

        See :meth:`_served_fields`. The engine's half is read off the scheduler rather than off
        ``caps()`` on the engine, because it is the scheduler that will apply the mask and the two
        answers cannot be allowed to differ.

        A scheduler with no binding to ask is read as ``False`` rather than raising, for the reason
        :meth:`_engine_capabilities` swallows the same absence: a capability report and a field audit
        are not the places to propagate "this extension is older than the caller". The direction is
        the safe one -- a runtime that has not said it applies a field is refused by name, which is
        where every other undeclared capability points.
        """
        if not (self._batching_enabled and self._scheduler is not None):
            return False
        getter = getattr(self._scheduler, "engine_caps", None)
        if not callable(getter):
            return False
        try:
            caps = getter()
        except Exception:
            return False
        return bool(getattr(caps, "structured_outputs", False))

    def _constraint_tokenizer(self) -> Any | None:
        """The engine's own vocabulary, read once, for building constrained-decode masks.

        Read through the native ``Tokenizer`` rather than reused from ``self._tokenizer``, and that
        is the whole reason this is not free. A constraint is a mask computed over the vocabulary
        *piece by piece*, and a piece is what the tokenizer emits rather than what the vocabulary
        file stores: a byte-level BPE vocabulary spells a space ``Ġ``, so a mask built from the raw
        entries would refuse every token that continues a word and the constrained answer would come
        back empty. The pieces have to be the ones the engine samples from, which is a fact about
        the C++ tokenizer and not about the Python one this file already holds for prompt encoding.

        Built on first use and kept, because the vocabulary is megabytes and a decode step never
        touches it: a run that asks for no schema should not pay for the runs that do.

        An explicit ``tokenizer_path`` is deliberately not honoured here, though `_load_tokenizer`
        prefers one. The two are different jobs: prompt encoding may use whichever tokenizer the
        caller named, and a mask has to be indexed the way the *engine* indexes, so the vocabulary
        comes from the checkpoint the engine was built on and from nowhere else.
        """
        if self._constraint_tokenizer_cache is _READER_UNSET:
            tokenizer = None
            path = self._native_checkpoint()
            if path and self._native is not None:
                try:
                    tokenizer = self._native.Tokenizer(path)
                except Exception as exc:
                    self._constraint_tokenizer_error = f"{type(exc).__name__}: {exc}"
            elif not path:
                self._constraint_tokenizer_error = "no checkpoint_dir"
            self._constraint_tokenizer_cache = tokenizer
        return self._constraint_tokenizer_cache

    def _structured_output(self, params: SamplingParams) -> Any | None:
        """The token constraint this request's ``response_format`` asks for, or ``None``.

        ``None`` covers two of the four things this can be, and they are the common ones: an absent
        field, and ``{"type": "text"}`` -- which is what an OpenAI client sends by default. Neither
        is a request to hold the answer to anything, and handing the scheduler a constraint for them
        would generate JSON for a caller who asked for prose. An absent field is also not a malformed
        one, so it does not go through the reading below at all.

        The reading of the field is :func:`structured_output_spec`, which is also what the host
        audited the body with, so a schema that passed the audit is the schema built here. Nothing
        about the schema is re-decided: whether the validator supports it is the validator's answer,
        raised from the C++ factory rather than guessed at from Python, because a keyword the
        validator does not implement is a schema silently half-applied and that is a wrong answer
        rather than a slow one.
        """
        if params.response_format is None:
            return None
        spec = structured_output_spec(params.response_format)
        if isinstance(spec, FieldRefusal):
            # Unreachable through the front end, which audits before it dispatches, and reachable
            # through the library surface, which builds `SamplingParams` by hand. Both have to be
            # answered the same way rather than one of them getting a TypeError later.
            raise UnsupportedFeatureError(spec.message)
        if not spec.constrains:
            return None
        tokenizer = self._constraint_tokenizer()
        if tokenizer is None:
            raise ConfigurationError(
                "structured outputs need the checkpoint's vocabulary to constrain against, and it "
                f"could not be read ({self._constraint_tokenizer_error or 'no native tokenizer'})"
            )
        if spec.kind == "json_object":
            return self._native.make_json_object_constraint(tokenizer)
        return self._native.make_json_schema_constraint(tokenizer, json.dumps(spec.schema))

    def audit_request(
        self, body: Mapping[str, Any], *, endpoint: str = CHAT
    ) -> FieldRefusal | None:
        """The first field in ``body`` this adapter cannot serve, or ``None``.

        Two questions, in this order: does the adapter's answer apply the field at all, and if it
        does, will the *engine* apply the particular value. The first is :meth:`_served_fields`. The
        second is the engine's own capability object, which is the only thing that knows whether
        this build samples per request or holds its sampler fixed -- the same question
        ``cpp_engine``'s ``check_sampling_supported`` asked, ported here so this front end answers a
        request the way the native one did.
        """
        refusal = audit_body(body, endpoint=endpoint, serves=self._served_fields())
        if refusal is not None:
            return refusal
        engine = self._engine_capabilities()
        sampling = (
            _EngineSampling.of_serialized_session()
            if engine is None
            else _EngineSampling.of_engine(engine)
        )
        return _engine_sampling_refusal(body, sampling)

    def _check_sampling(self, params: SamplingParams) -> None:
        """The same audit for a request built from typed parameters rather than a JSON body.

        The library surface (``LLM.chat``, ``AsyncLLM``) never produces a body, so the HTTP audit
        cannot see it -- and a field a caller set on :class:`SamplingParams` is *more* visible than
        one in a JSON key, not less, so it cannot be dropped either. Rather than write the refusals
        a second time, the typed request is mapped back onto the body keys the audit reads: one
        policy, two spellings of the same request, and no way for the two to disagree about what
        this backend serves.

        Shape is not re-checked here, for the reason it is not checked in ``audit_request``:
        :class:`SamplingParams` validates its own fields on construction, so ``min_p=2.0`` never
        reaches a backend.
        """
        refusal = self.audit_request(_sampling_body(params))
        if refusal is not None:
            raise UnsupportedFeatureError(refusal.message)

    @staticmethod
    def _native_token(item: Any) -> int:
        if isinstance(item, int):
            return int(item)
        return int(getattr(item, "top_token", getattr(item, "token", item)))

    # --- TP command fan-out -------------------------------------------------
    # Under TP the worker ranks sit in run_worker_loop() waiting for a command.
    # Every rank has to enter the same collective in the same order, so rank 0
    # announces the op before it runs the op itself.  At world size 1 the
    # worker_command_* calls are native no-ops, so these wrappers stay on the
    # single-rank path unchanged.

    def _engine_or_raise(self) -> Any:
        engine = self._engine
        if engine is None:
            raise RuntimeError("native C++ engine is closed")
        return engine

    def _tp_command(self, name: str, *args: Any) -> None:
        if self.args.tensor_parallel_size <= 1:
            return
        engine = self._engine_or_raise()
        command = getattr(engine, name, None)
        if command is None:
            raise UnsupportedFeatureError(
                f"native engine does not expose {name}; "
                "rebuild cpp_engine with -DPOCKET_BUILD_PYTHON=ON"
            )
        command(*args)

    def _tp_prefill(self, prompt_ids: list[int]) -> Any:
        self._tp_command("worker_command_prefill", prompt_ids)
        return self._engine_or_raise().prefill(prompt_ids)

    def _tp_decode_step(self, token: int) -> Any:
        self._tp_command("worker_command_decode", token)
        return self._engine_or_raise().decode_step(token)

    def _is_eos(self, token: int) -> bool:
        return token in self._eos_ids

    def _generate_until_eos(
        self, request: GenerationRequest, prompt_ids: list[int], started: float
    ) -> tuple[list[int], bool, float]:
        """Step the native session so it never advances past EOS.

        ``QwenEngine.generate`` takes no EOS argument and keeps mutating its
        session for the whole token budget.  Calling it and truncating the result
        would leave the recurrent state and prefix cache positioned after text
        the caller never sees, corrupting reuse for the next request.  Driving
        prefill/decode here keeps the session and the returned tokens in step,
        and matches what the streamed path does.

        Returns the tokens, whether EOS stopped generation, and the time to the
        first token measured from ``started``.  TTFT is observable only here: the
        loop is what learns when prefill produced a token, so reporting 0.0 as
        this path used to made every serial timing comparison degenerate.

        The budget is the caller's, or -- when the caller named none -- everything
        the prompt leaves of the engine's context, so an answer ends at EOS or
        when there is no room left for another token.
        """
        result = self._tp_prefill(prompt_ids)
        ttft = time.perf_counter() - started
        token_ids: list[int] = []
        budget = request.sampling_params.token_budget(self._context_tokens() - len(prompt_ids))
        for index in range(budget):
            self._ensure_open()
            self._check_cancelled(request.request_id)
            token = self._native_token(result)
            if self._is_eos(token):
                return token_ids, True, ttft
            token_ids.append(token)
            if index + 1 < budget:
                result = self._tp_decode_step(token)
        return token_ids, False, ttft

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

    def _answer_reader(self) -> Any | None:
        """The templater that reads this checkpoint's answers, built once.

        The engine hands back decoded text and nothing else, so which of it is the answer, which is
        the reasoning block and which is a tool call has to be read out of the text -- and that
        reading is per architecture. It is the same selection the C++ front end's sidecar made; see
        :mod:`pocketllm.protocol.templating` for why it is shared rather than the C++ side's private
        business.

        ``None`` means no tokenizer, which already makes ``_prompt_ids`` refuse a text prompt: there
        is nothing to read an answer with, and the caller returns the raw decode.
        """
        if self._answer_reader_cache is _READER_UNSET:
            reader = None
            if self._tokenizer is not None:
                reader = build_templater(
                    detect_architecture(self._native_checkpoint()), self._tokenizer
                )
            self._answer_reader_cache = reader
        return self._answer_reader_cache

    def _read_answer(
        self, request: GenerationRequest, text: str, finish_reason: str
    ) -> tuple[str, str, dict[str, Any]]:
        """An answer's visible text, its finish reason, and the split the metadata carries.

        A call is *why* the generation ended, which is OpenAI's own reading of the same answer: a
        client that branches on ``"tool_calls"`` to decide whether to run something and send the
        result back would, under ``"stop"``, read the call as the final answer and stop there. The
        parse is all-or-nothing, so a reported call is a complete one.

        A parse that raises leaves the text exactly as the engine produced it. That is what the C++
        front end did with a sidecar reply that was not ``ok``, and it is the reading that cannot
        lose an answer; the reason is published under ``answer_parse_error`` rather than swallowed,
        because a client seeing markup in its content should be able to find out why.

        Only a chat request is read. ``messages`` in the metadata is what ``_prompt_ids`` already
        uses to tell the two routes apart, and it is the same line the C++ front end drew: it parsed
        the assistant message and handed ``/v1/completions`` the raw decode. A completion is text in
        and text out, and a model that wrote ``</think>`` into one meant it as four characters of
        the answer.
        """
        reader = self._answer_reader()
        if reader is None or "messages" not in request.metadata:
            return text, finish_reason, {}
        try:
            parsed = reader.parse(
                text,
                str(request.metadata.get("thinking_mode", "chat")),
                request.metadata.get("tools"),
            )
        except Exception as exc:  # noqa: BLE001
            return text, finish_reason, {"answer_parse_error": f"{type(exc).__name__}: {exc}"}
        metadata: dict[str, Any] = {}
        reasoning = parsed.get("reasoning_content")
        if reasoning:
            metadata["reasoning_content"] = reasoning
        # Normalized rather than forwarded as parsed: DeepSeek's encoder leaves the ``id`` out of a
        # call, because the calls it reads come out of a *prompt* where the id was the client's to
        # write. A call this backend reports has no such history -- a client that validates against
        # OpenAI's typed models rejects a call without an id, and one that echoes the call back to
        # attribute a tool result to it has nothing to name it by. The Qwen parser already mints
        # one, and this is the same helper the DeepSeek runtime and the V4.1 adapter use.
        tool_calls = normalize_tool_calls(parsed.get("tool_calls"))
        if tool_calls:
            metadata["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        return str(parsed.get("content", "")), finish_reason, metadata

    def _answer_from_tokens(
        self, request: GenerationRequest, token_ids: Sequence[int], finish_reason: str
    ) -> tuple[str, str, dict[str, Any], tuple[int, int]]:
        """A finished run read back, with the client's stop sequences applied first.

        A stop sequence ends the *answer*, and on a chat request the reasoning block is not part of
        it: a sequence is matched against the text after ``</think>`` and not against the text
        before it. That matters for the sequences clients actually use -- ``"\\n\\n"`` is a common
        one and a reasoning block is full of blank lines -- where matching the whole decode would
        end the answer before the model had written any of it.

        The cut comes before the read, not after. A stop sequence that lands inside a tool call has
        to leave no call behind: the parse is all-or-nothing, so handing it the truncated text is
        what makes "a reported call is a complete one" hold. The native server cut the whole decode
        and then parsed, which is the same order with the reasoning block included; this cuts the
        same place minus the block.

        The token ids are not cut. They are what ``usage`` counts and what the engine really
        executed -- the scheduler has no way to see a text-level sequence, so the run goes to its
        budget either way -- which is the same pair the native server reported.

        The fourth element is where the returned text sits in the decoded token stream, in UTF-8
        bytes. It is what lines a per-token ranking up against the text it is reported beside; see
        :func:`pocketllm.protocol.logprobs.render`.

        The start is found by subtraction rather than by asking the reader, because the answer is a
        **suffix** of the text the reader was handed: both cuts remove from an end -- the reasoning
        split from the front, the stop sequence and the parser from the back -- and neither removes
        from the middle, which is the invariant :meth:`_cut_answer` already relies on when it
        reattaches the block it split off. So the answer is the last ``len(text)`` bytes of that
        text, and a trailing marker the parser dropped of its own accord falls outside the span by
        the same arithmetic that puts a stop cut outside it.
        """
        stops = tuple(request.sampling_params.stop)
        decoded = self._decode(list(token_ids))
        cut = decoded
        if stops:
            cut, stopped = self._cut_answer(decoded, stops, request)
            if stopped:
                finish_reason = "stop"
        text, finish_reason, metadata = self._read_answer(request, cut, finish_reason)
        text_bytes = len(text.encode("utf-8"))
        start = len(cut.encode("utf-8")) - text_bytes
        return text, finish_reason, metadata, (start, start + text_bytes)

    def _cut_answer(
        self, decoded: str, stops: Sequence[str], request: GenerationRequest
    ) -> tuple[str, bool]:
        """``decoded`` with its answer cut at a stop sequence, and whether one matched.

        The block to cut is found by splitting the way the answer is read, so the two agree on where
        the answer begins: everything up to the start of the content is handed back untouched, and
        only the content is searched. A request with no chat reader has no block -- its whole decode
        is the answer -- which is also what a chat answer with no ``</think>`` in it looks like.
        """
        reader = self._answer_reader()
        if reader is None or "messages" not in request.metadata:
            return apply_stop_to_text(decoded, stops)
        _, content = reader.split_reasoning(
            decoded, str(request.metadata.get("thinking_mode", "chat"))
        )
        cut, stopped = apply_stop_to_text(content, stops)
        return (decoded[: len(decoded) - len(content)] + cut, True) if stopped else (decoded, False)

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests with optional batch scheduler."""
        self._ensure_open()

        # Phase 3.4: Use batch scheduler if enabled
        if self._batching_enabled and self._scheduler is not None:
            return self._generate_batched(requests)

        # Legacy serial path
        return self._generate_serial(requests)

    def _generate_serial(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests serially (legacy path)."""
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
                    # Native generate() drives its own prefill/decode loop
                    # internally, so rank 0 cannot announce those steps to the
                    # workers. Under TP always take the stepped path, which
                    # broadcasts each command.
                    if self._eos_ids or self.args.tensor_parallel_size > 1:
                        token_ids, hit_eos, ttft = self._generate_until_eos(
                            request, prompt_ids, started
                        )
                    else:
                        raw = self._engine_or_raise().generate(
                            prompt_ids,
                            request.sampling_params.token_budget(
                                self._context_tokens() - len(prompt_ids)
                            ),
                        )
                        token_ids = [self._native_token(item) for item in raw]
                        hit_eos = False
                        # Native generate() drives its own loop and reports no
                        # per-token boundary, so the first token is not separable
                        # from the whole call here.
                        ttft = 0.0
                    self._ensure_open()
                self._check_cancelled(request.request_id)
                # `completion_tokens` counts the EOS step the engine executed, so
                # usage stays comparable with the streamed path.
                completion_tokens = len(token_ids) + (1 if hit_eos else 0)
                # The ranking is not read here: this path has no scheduler to ask for one, which is
                # what `_served_fields` reports and `_check_sampling` has already refused a request
                # for logprobs by. The span is returned all the same -- one reader for both paths is
                # what keeps the two from disagreeing about where an answer starts.
                text, finish_reason, metadata, _span = self._answer_from_tokens(
                    request, token_ids, "stop" if hit_eos else "length"
                )
                outputs.append(
                    GenerationResult(
                        request_id=request.request_id,
                        token_ids=token_ids,
                        text=text,
                        finish_reason=finish_reason,
                        usage=Usage(len(prompt_ids), completion_tokens),
                        timings=TimingMetrics(
                            total_seconds=time.perf_counter() - started,
                            ttft_seconds=ttft,
                        ),
                        metadata=metadata,
                    )
                )
            finally:
                self._clear_request(request.request_id)
                if self._closed:
                    self._release_native()
        return outputs

    def _generate_batched(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate requests using the batch scheduler."""
        if not requests:
            return []

        # Submit all requests to scheduler
        request_map: dict[int, GenerationRequest] = {}
        native_request_ids: list[int] = []

        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_sampling(request.sampling_params)
                prompt_ids = self._prompt_ids(request)

                # Create native sampling params
                sampling = self._native.QwenBatchSamplingParams()
                # A caller who named no budget gets everything the prompt leaves, which is the
                # same rule the stepped path resolves at its own top.
                sampling.max_new_tokens = request.sampling_params.token_budget(
                    self._context_tokens() - len(prompt_ids)
                )
                sampling.temperature = request.sampling_params.temperature or 0.0
                sampling.top_p = request.sampling_params.top_p or 1.0
                sampling.top_k = request.sampling_params.top_k or 20
                # Whether to rank each position at all is a property of the *step* rather than of a
                # row -- the ranking kernels take one width for the whole batch, and under TP every
                # rank has to enter the same collectives at the same width -- so a batch containing
                # one request that asked for a ranking is ranked for all of them and each row is
                # narrowed back to what it asked for when the result is rendered.
                sampling.logprobs_n = _ranking_width(request.sampling_params)
                # A seed the caller named has to reach the sampler, and until now it did not: the
                # field was never assigned, so every batched request ran at the engine's default
                # seed and a client that pinned one was answered as if it had not. That is invisible
                # on a request that asked for one choice and fatal to one that asked for several --
                # the choices of an `n`-choice request differ only in the seed they are given, so
                # dropping it turns the fan-out into the same answer written out `n` times.
                #
                # The engine's seed is unsigned and the mask is how a negative Python int names one,
                # which is the same reinterpretation the native front end made when it assigned a
                # double into a `uint64_t`.
                if request.sampling_params.seed is not None:
                    sampling.seed = int(request.sampling_params.seed) & SEED_MASK
                # The batch scheduler runs the model to max_new_tokens unless
                # told otherwise. EOS truncation is the default for serving, but
                # a benchmark (or any caller that wants the full budget) opts in
                # through extra["ignore_eos"], mirroring the native parity driver
                # which sets req.sampling.ignore_eos = true.
                if bool(request.sampling_params.extra.get("ignore_eos", False)):
                    sampling.ignore_eos = True

                # Submit to scheduler. The constraint is built here rather than up with the other
                # sampling parameters because of what it is: the scheduler owns it for the
                # request's whole life, and `sampling` does not carry it -- the field there is a
                # borrow of the pointer this call hands over.
                native_req_id = self._scheduler.submit_request(
                    prompt_ids,
                    sampling,
                    None,
                    None,
                    self._structured_output(request.sampling_params),
                )
                if native_req_id > 0:
                    native_request_ids.append(native_req_id)
                    request_map[native_req_id] = request
                else:
                    # Submission failed
                    self._clear_request(request.request_id)

            except Exception as e:
                self._clear_request(request.request_id)
                raise

        # Poll for results
        outputs: list[GenerationResult] = []
        native_errors: list[str] = []
        timeout_ms = 60000  # 60 seconds per request

        for native_req_id in native_request_ids:
            request = request_map[native_req_id]
            result = self._scheduler.poll_result(native_req_id, timeout_ms)

            if result is None:
                # Timeout
                self._clear_request(request.request_id)
                raise TimeoutError(f"Request {request.request_id} timed out after {timeout_ms}ms")

            native_error = str(getattr(result, "error", "") or "")
            if native_error:
                # Poll the rest of this submitted batch before raising. A failed
                # native forward normally completes every participating row with
                # an error; abandoning their poll results here would retain one
                # native completed-result entry and one public active request per
                # row after the first.
                native_errors.append(native_error)
                self._clear_request(request.request_id)
                continue

            # Convert native result to GenerationResult
            token_ids = _strip_terminal_stop_token(result)
            text, finish_reason, metadata, span = self._answer_from_tokens(
                request, token_ids, result.finish_reason
            )

            outputs.append(
                GenerationResult(
                    request_id=request.request_id,
                    token_ids=token_ids,
                    text=text,
                    finish_reason=finish_reason,
                    usage=Usage(result.prompt_tokens, result.completion_tokens),
                    timings=TimingMetrics(
                        total_seconds=result.total_seconds,
                        ttft_seconds=result.ttft_seconds,
                    ),
                    logprobs=self._ranking(request, result, token_ids, span),
                    metadata=metadata,
                )
            )

            self._clear_request(request.request_id)

        if native_errors:
            joined = "; ".join(dict.fromkeys(native_errors))
            raise RuntimeError(f"native C++ generation failed: {joined}")

        return outputs

    def _ranking(
        self,
        request: GenerationRequest,
        result: Any,
        token_ids: Sequence[int],
        span: tuple[int, int],
    ) -> dict[str, Any] | None:
        """The OpenAI ``logprobs`` object for one finished choice, or ``None`` if none was asked.

        The engine reports one ranking per generated position, and the vector is parallel to
        ``result.generated_tokens`` -- the tokens *before* the terminal stop token is stripped, which
        is the vector the ranking is indexed against -- while ``token_ids`` is that vector with the
        token the answer does not contain removed. Walking the stripped ids against the unstripped
        ranking is what the native front end did as well, and it is safe for the same reason: the
        stripped vector is a prefix of the other, so index ``i`` is the same position in both.

        A position the engine left unranked fails the request instead of coming back as an array
        shorter than the text. A caller reading a probability per token has no way to notice that the
        two drifted apart, and speculative decoding is exactly where it happens: the engine describes
        the first token a row emitted and no others, so a row that emitted several leaves the rest
        without an entry.
        """
        if not request.sampling_params.logprobs:
            return None
        rankings = list(getattr(result, "logprobs", ()) or ())
        generated = list(getattr(result, "generated_tokens", ()) or ())
        if not ranking_complete(rankings, len(generated)):
            raise RuntimeError(
                "the engine returned no log probabilities for a position it was asked to rank, so "
                f"the ranking for request {request.request_id} describes fewer tokens than its text"
            )
        return render_logprobs(
            token_ids,
            rankings,
            decode=self._decode,
            text_start=span[0],
            text_end=span[1],
            alternatives=int(request.sampling_params.top_logprobs or 0),
        )

    def _stream_native(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self._check_sampling(request.sampling_params)
        prompt_ids = self._prompt_ids(request)
        max_tokens = request.sampling_params.token_budget(self._context_tokens() - len(prompt_ids))
        self._engine_or_raise()
        # Do not reset() here.  QwenEngine::reset() clears the prefix cache, so
        # calling it per request would disable configured prefix reuse.  prefill()
        # already matches the common prefix, restores a snapshot, or zeroes the
        # recurrent state, which is also what native generate() relies on.
        # QwenEngine.prefill predicts the first token without consuming it. The
        # following decode steps consume the previous prediction and predict the
        # next one, matching the native generate() result ordering.
        result = self._tp_prefill(prompt_ids)
        generated: list[int] = []
        # A thinking-mode answer is split as it arrives, the way the V4.1 adapter splits its own:
        # everything before `</think>` goes out as `reasoning_content` and everything after as
        # `content`, so a client watches the reasoning rather than waiting for the answer. The C++
        # front end split on the `</think>` *token id* instead; this reads the running text, which
        # is the same split the unstreamed path takes below, and one reading of the same block
        # across a backend's two paths is worth more here than matching a front end that is on its
        # way out. The one case it leaves is the marker straddling a token boundary -- a client is
        # then sent the marker's first characters as reasoning and cannot have them back, which is
        # a boundary of the format rather than of this diff.
        # A chat request only, for the reason `_read_answer` gives: a completion is text in and text
        # out, and `</think>` inside one is four characters the model wrote.
        reader = self._answer_reader() if "messages" in request.metadata else None
        thinking_mode = str(request.metadata.get("thinking_mode", "chat"))
        stops = tuple(request.sampling_params.stop)
        previous_reasoning = ""
        previous_content = ""
        for index in range(max_tokens):
            self._ensure_open()
            self._check_cancelled(request.request_id)
            token = self._native_token(result)
            if self._is_eos(token):
                # Stop before another decode_step. EOS is counted as a generated
                # step but is not emitted as visible text -- and what the running
                # decode was still holding is emitted here, because nothing is
                # coming that could settle it.
                # `final=True` because nothing is coming that could settle either a half
                # character or a stop sequence's first bytes.
                reasoning, content = self._split_answer(
                    self._decode(generated), reader, thinking_mode
                )
                content, _ = self._visible(content, stops, final=True)
                metadata: dict[str, Any] = {}
                if reasoning != previous_reasoning:
                    metadata["reasoning_content"] = _suffix(reasoning, previous_reasoning)
                yield TokenEvent(
                    request.request_id,
                    text=_suffix(content, previous_content),
                    finish_reason="stop",
                    usage=Usage(len(prompt_ids), len(generated) + 1),
                    metadata=metadata,
                )
                return
            generated.append(token)
            # Decode the complete sequence so BPE token boundaries are handled by the
            # tokenizer, then hold back a tail that is still half a character: the
            # decode renders that as U+FFFD until the next token completes it, and a
            # stream cannot take a character back. Emit only the newly visible suffix
            # when possible. The last token of a bounded generation has nothing after
            # it to settle its tail, and this stream ends where the unstreamed decode
            # ends, so the tail goes out with it rather than being dropped.
            last = index + 1 == max_tokens
            decoded = self._decode(generated)
            if not last:
                decoded = settled_text(decoded)
            reasoning, content = self._split_answer(decoded, reader, thinking_mode)
            content, stopped = self._visible(content, stops, final=last)
            text = _suffix(content, previous_content)
            metadata = (
                {"reasoning_content": _suffix(reasoning, previous_reasoning)}
                if reasoning != previous_reasoning
                else {}
            )
            previous_content = content
            previous_reasoning = reasoning
            event = TokenEvent(request.request_id, token_id=token, text=text, metadata=metadata)
            if stopped:
                # A client stop sequence ended the answer, which is what "stop" means to the
                # caller even though the engine's own reason was the budget. The scheduler
                # still runs to max_tokens -- it has no way to see a text-level sequence --
                # so the run ends here rather than being continued for text nobody reads.
                event.finish_reason = "stop"
                event.usage = Usage(len(prompt_ids), len(generated))
            elif last:
                event.finish_reason = "length"
                event.usage = Usage(len(prompt_ids), len(generated))
            yield event
            if stopped:
                return
            if index + 1 < max_tokens:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                result = self._tp_decode_step(token)

    @staticmethod
    def _visible(
        decoded: str, stops: Sequence[str], *, final: bool
    ) -> tuple[str, bool]:
        """``decoded`` cut at the client's first stop sequence, and whether one was found.

        Applied to the *answer*, not to the decode: a chat request's reasoning block is a separate
        field and a sequence inside it must not end the answer that follows, so the caller passes
        the content and not the whole text. For a completion, and for a chat answer with no block in
        it, the two are the same string.

        Ported from the native server's ``scan_stop_strings``, and the two cases it distinguishes
        are why this is not two lines of ``find``. While more tokens may still arrive, a tail that
        is a *prefix* of a sequence is withheld, because the next token may complete it and a stream
        cannot take a character back; once generation has ended that tail can no longer complete, so
        it is part of the answer rather than a withheld prefix. ``hold_back`` is the first half of
        that and ``final`` is the second, and both callers pass the flag they need -- the running
        loop ``False``, the EOS and budget exits ``True``.
        """
        if not stops:
            return decoded, False
        cut = min((decoded.find(stop) for stop in stops if stop in decoded), default=-1)
        if cut >= 0:
            return decoded[:cut], True
        return (decoded if final else hold_back(decoded, stops)), False

    def _split_answer(
        self, decoded: str, reader: Any | None, thinking_mode: str
    ) -> tuple[str, str]:
        """A running decode as ``(reasoning, content)``, recomputed whole each token.

        Recomputed rather than accumulated because the marker is plain text and its offset moves as
        the byte-level pieces underneath it settle; the caller diffs against what it already sent.
        Without a reader the whole decode is the answer, which is what this backend did before it
        could read one at all -- the reader is ``None`` only when no tokenizer loaded, and that
        already makes ``_prompt_ids`` refuse a text prompt.
        """
        if reader is None:
            return "", decoded
        try:
            return reader.split_reasoning(decoded, thinking_mode)
        except Exception:  # noqa: BLE001
            return "", decoded

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
        # Workers block in read() on the command channel. Without a shutdown
        # they never return from run_worker_loop() and the supervisor has to
        # escalate to SIGKILL.
        if (
            engine is not None
            and self.args.tensor_parallel_size > 1
            and self.args.tensor_parallel_rank == 0
        ):
            shutdown = getattr(engine, "worker_command_shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    # The workers may already be gone; closing is best effort.
                    pass
        self._engine = None
        self._tokenizer = None
        self._answer_reader_cache = _READER_UNSET
        self._constraint_tokenizer_cache = _READER_UNSET
        self._native = None
        self._ready = False
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Enter the native worker loop for a nonzero TP rank.

        This blocks until rank 0 sends a shutdown command through the command
        channel. The engine must already be constructed.
        """
        self._ensure_open()
        if self.args.tensor_parallel_rank == 0:
            raise RuntimeError("run_worker must not be called on rank 0")
        if self._engine is None:
            raise RuntimeError("native engine is not constructed")
        if not hasattr(self._engine, "run_worker_loop"):
            raise UnsupportedFeatureError(
                "native engine does not expose run_worker_loop; "
                "rebuild cpp_engine with -DPOCKET_BUILD_PYTHON=ON"
            )
        if on_ready is not None:
            on_ready()
        # run_worker_loop() blocks until rank 0 sends shutdown
        self._engine.run_worker_loop()

    def close(self) -> None:
        if self._closed:
            return
        # Releases the tensor-parallel supervisor this backend was built with, if any;
        # see BackendBase._release_supervisor.
        super().close()
        # Never destroy a native engine while a GIL-released kernel is using it.
        # An active generate/stream call releases it from its own finally block.
        if self._request_lock.acquire(blocking=False):
            try:
                self._release_native()
            finally:
                self._request_lock.release()
