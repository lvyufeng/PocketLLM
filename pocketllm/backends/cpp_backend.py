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
from collections.abc import Callable, Iterator, Sequence
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
from pocketllm.protocol import encode_chat_prompt, normalize_tool_calls, render_fallback_prompt
from pocketllm.protocol.templating import build_templater, detect_architecture

from .base import BackendBase, settled_text
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
    cache has to agree with what it reports. It is not part of the answer. `openai_server.cpp` drops it
    before detokenizing -- and before counting `completion_tokens` for the client -- in
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
        """Which of the three places the batch decision came from, for a message about it."""
        if self.args.backend_options.get("enable_batching") is not None:
            return "from --backend-option enable_batching"
        if self.args.enable_batching is not None:
            return "from --enable-batching" if self.args.enable_batching else "from --no-enable-batching"
        return "from this backend's default"

    def _batching_was_asked_for(self) -> bool:
        """Whether a *person or caller* asked, as opposed to this backend assuming it.

        The difference decides whether a build without a scheduler is worth a warning. An explicit
        request that cannot be honoured is the case a warning is for. The assumption is not: it is a
        claim about the build, `capabilities.details['scheduler']` answers it on every construction,
        and a warning on the default path would be one line of noise per process for a fact the
        capability report already carries -- the same declaration R2 exists to make uniform.
        """
        if self.args.backend_options.get("enable_batching"):
            return True
        if self.args.enable_batching:
            return True
        # A width can be refused for being unreadable, but this decides whether to *warn* about it, so
        # it must not be the thing that raises: a build with no scheduler and a nonsense width would
        # otherwise report the ValueError in place of the ConfigurationError that names the width.
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
        """Initialize the batch scheduler if available."""
        if not hasattr(self._native, "QwenBatchScheduler"):
            if self._batching_was_asked_for():
                import warnings
                warnings.warn(
                    f"the batch path was asked for ({self._batching_source()}) but this native module "
                    f"does not expose QwenBatchScheduler; falling back to serial execution, which "
                    f"serves one request at a time"
                )
            return False

        max_batch_size = self._configured_max_batch_size()
        if max_batch_size <= 1:
            return False

        try:
            self._scheduler = self._native.QwenBatchScheduler(self._engine, max_batch_size)
            return True
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to create QwenBatchScheduler: {e}; falling back to serial execution")
            return False

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
                # What the engine declares that this adapter does not yet surface as a capability.
                # Publishing them rather than hiding them is the point of the field: each one is a
                # path Python would have to deliver end to end before the capability could be
                # reported, and the adapter reporting `False` over an engine saying `True` is a gap
                # to close deliberately rather than a value to copy over.
                "engine_declares": self._engine_declaration(engine),
            },
        )

    @staticmethod
    def _engine_declaration(engine: Any | None) -> dict[str, Any]:
        """The engine's own answer for the facts this adapter does not yet report as capabilities."""
        if engine is None:
            return {}
        return {
            "continuous_batching": bool(engine.continuous_batching),
            "chunked_prefill": bool(engine.chunked_prefill),
            "per_request_sampling": bool(engine.per_request_sampling),
            "per_request_top_k": bool(engine.per_request_top_k),
            "structured_outputs": bool(getattr(engine, "structured_outputs", False)),
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

        if kind == "persistent":
            return self._construct_persistent_engine()
        elif kind == "qwen":
            return self._construct_qwen_engine()
        else:
            raise UnsupportedFeatureError(
                f"unsupported engine_kind: {kind}; use 'auto', 'persistent' or 'qwen'"
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
        checkpoint = self.args.checkpoint_dir
        if detect is None or not checkpoint:
            # An older native module, or a caller that passed no checkpoint at
            # all (token-only tests).  Keep the previous default rather than
            # failing on a path that never needed detecting.
            return "qwen"
        # The registry reads a directory's config.json, and a GGUF states its
        # architecture in its own header instead -- so a directory holding one
        # names it through the file it holds, which is the same file the engine
        # will open.
        if not checkpoint.endswith(".gguf"):
            checkpoint = gguf_checkpoint_file(checkpoint) or checkpoint
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
        """Construct PersistentEngine (supports TP worker loop)."""
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
        for name, value in {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 20,
            "sampling_seed": 0,
        }.items():
            if hasattr(options, name):
                setattr(options, name, value)
        engine = cls(self.args.checkpoint_dir, options, 0, self._context_tokens())
        # warmup_tp() builds the command channel and forces the NCCL
        # communicator up.  Both ranks must do it before any rank issues a
        # collective, and a worker cannot enter run_worker_loop() without it.
        if self.args.tensor_parallel_size > 1:
            engine.warmup_tp()
        return engine

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
            )
            if encoded is not None:
                return encoded
            prompt = request.prompt or render_fallback_prompt(messages)
        else:
            prompt = request.prompt or ""
        encoded = self._tokenizer.encode(prompt)
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
                    detect_architecture(self.args.checkpoint_dir or ""), self._tokenizer
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
                text, finish_reason, metadata = self._read_answer(
                    request,
                    self._decode(token_ids),
                    "stop" if hit_eos else "length",
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
                # The batch scheduler runs the model to max_new_tokens unless
                # told otherwise. EOS truncation is the default for serving, but
                # a benchmark (or any caller that wants the full budget) opts in
                # through extra["ignore_eos"], mirroring the native parity driver
                # which sets req.sampling.ignore_eos = true.
                if bool(request.sampling_params.extra.get("ignore_eos", False)):
                    sampling.ignore_eos = True

                # Submit to scheduler
                native_req_id = self._scheduler.submit_request(prompt_ids, sampling, None)
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
            text, finish_reason, metadata = self._read_answer(
                request, self._decode(token_ids), result.finish_reason
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
                    metadata=metadata,
                )
            )

            self._clear_request(request.request_id)

        if native_errors:
            joined = "; ".join(dict.fromkeys(native_errors))
            raise RuntimeError(f"native C++ generation failed: {joined}")

        return outputs

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
                reasoning, content = self._split_answer(
                    self._decode(generated), reader, thinking_mode
                )
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
            decoded = self._decode(generated) if last else settled_text(self._decode(generated))
            reasoning, content = self._split_answer(decoded, reader, thinking_mode)
            text = _suffix(content, previous_content)
            metadata = (
                {"reasoning_content": _suffix(reasoning, previous_reasoning)}
                if reasoning != previous_reasoning
                else {}
            )
            previous_content = content
            previous_reasoning = reasoning
            event = TokenEvent(request.request_id, token_id=token, text=text, metadata=metadata)
            if last:
                event.finish_reason = "length"
                event.usage = Usage(len(prompt_ids), len(generated))
            yield event
            if index + 1 < max_tokens:
                self._ensure_open()
                self._check_cancelled(request.request_id)
                result = self._tp_decode_step(token)

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
