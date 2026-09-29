"""Xing4.0-29B-A4B under PocketLLM's backend contract.

A Xing4 checkpoint needs its own adapter rather than a flag on
:mod:`torch_backend`, because it is none of the runtimes already here: this one
serves a 40-block MLA trunk with four residual streams a block and 64 routed
experts a layer out of ``src/models/xing4_0/``, and nothing in that package reads
a Qwen-shaped safetensors tree or the DeepSeek runtime's own plan.

**One card, and that is the interesting number.** The released IQ4_NL export is
17.84 GiB with every expert resident, against 22 GiB a card, so this is the first
checkpoint here that is *smaller* than the card: there is no host bank, no
per-token expert copy and no tensor parallel group, because there is nothing to
spill and no reason to shard. What the 4 GiB that is left buys is context -- the
absorbed cache is 46 KB a token across the forty layers, so 32768 positions is
1.51 GiB and the default is sized there rather than at the checkpoint's own
262144, which would be 11.8 GiB and would not fit.

**What outlives a request is the prefix store.** ``generate`` resets nothing, but
the serving loop builds its answer from the prompt every turn, so a chat loop
that resends its history pays for the history on every turn;
``--enable-prefix-caching`` (on by default) holds each prompt's cache state on
the host keyed by the prompt's own tokens, and a later request restores the
longest prefix it shares with one already served and forwards only the rest. See
:mod:`src.models.xing4_0.prefix_cache` for what is stored and why the whole
latent a layer is enough.

**Requests serialize, and now they can serialize under the shared scheduler.** The trunk's
forward flattens its input to one token axis (``gguf_model.embed`` reshapes to
``[-1]`` and expands a single batch axis), so two sequences handed to it together
would attend to each other, and serving them one at a time is the honest answer.
``--enable-batching`` routes that one-at-a-time serving through the same
``BatchScheduler`` the ``cpp`` backend drives, at the width this runtime declares
(``max_slots = 1``, ``continuous_batching = False``); without it, serialization is the
lock at the backend boundary, where :class:`~pocketllm.backends.base.BackendBase` says
it belongs. ``capabilities.supports_batch`` reports which of the two is live, and it is
the same answer either way.

Stage 5 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
    GenerationResult,
    TokenEvent,
)

from .base import RuntimeAdapter, TokenStreamer
from .capabilities import IGNORED_OPTIONS, declared_capabilities
from .options import BackendOption, Group, Kind, decode_args
from .shared_options import PREFILL_CHUNK, PREFIX_CACHE_BYTES
from .runtime_engine import (
    RuntimeSpec,
    SchedulerHost,
    cancel_key,
    card_for_rank,
    device_index,
    visible_card_count,
)

DEFAULT_MAX_SEQ_LEN = 32768
"""Positions the cache is sized at when ``--max-model-len`` is not given.

46 KB a token across the forty layers, so this is 1.51 GiB of the card and the
checkpoint's own 262144 would be 11.8 GiB against the 3.6 GiB that are free with
the weights resident. The default is the number that fits with room for a second
process on the host, not the number the model was trained at.
"""

DEFAULT_PREFILL_CHUNK = 2048
"""Tokens one prefill forward takes when the room leaves space for that many.

The loop's own default and the widest this adapter will pick.  It is not always
usable, because the absorbed MLA materialises a `heads x chunk x tokens` score
path, so a chunk costs a multiple of its own width of the card -- see
:func:`_chunk_for` for the measurement.
"""

#: The narrowest prefill chunk this loop will run.  Below it a request makes no
#: progress worth the launch, and the score path stops shrinking faster than the
#: context grows, so the floor is also where a context ceiling comes from: the
#: workspace and the KV cache want the same few GiB of the card.
MIN_PREFILL_CHUNK = 128

#: Bytes one element of the prefill's score path costs at its peak, measured.
#: **Not 4.**  The score matrix is fp16 (2 bytes), the fp32 softmax the reference
#: asks for is a second copy of the whole thing (4) and the fp16 probabilities it
#: is cast back into are a third (2); all three are live at once, and the peak of
#: the three is what the card has to have.  A chunk-wide measurement at a fixed
#: 8192-token context gives 281.5, 552.4, 1095.2 and 2183.3 MiB for chunks of
#: 128, 256, 512 and 1024, against this constant's 256.0, 512.0, 1024.0 and
#: 2048.0 -- so the remainder is the two terms below and not a fourth copy.
SCORE_BYTES_PER_ELEMENT = 8

#: What a prefill holds besides the scores, per chunk token: the fp32 residual
#: streams (`[1, chunk, 4, 3584]` is 56 KiB a chunk token on its own, and the
#: hyper-connection rebuilds several of them) and the grouped MoE's per-route
#: buffers.  Measured as the same probe's remainder divided by the chunk.
PREFILL_WORKSPACE_BYTES_PER_CHUNK = 126 << 10

#: The part of that remainder that does not scale with the chunk.
PREFILL_WORKSPACE_FLOOR_BYTES = 11 << 20

#: What to leave free on the card for everything nobody accounted for: the
#: allocator's own fragmentation, the sampler's host-side copy of a 131072-wide
#: logits row, and whatever else the box already has resident.  The default
#: context needs about 300 MiB of it in practice; this is the reserve that makes
#: the derivation conservative rather than tight.
PREFILL_DEVICE_RESERVE = 768 << 20

#: The room a chunk is planned against when nobody measured the device -- a
#: caller with no card in hand, or a test.  Half of what is free with this
#: checkpoint resident.
PREFILL_SCORE_BUDGET = 1 << 30

#: Host memory a rank's prefix store may hold, when prefix caching is on.
DEFAULT_PREFIX_CACHE_BYTES = 2 << 30

#: Every option this runtime reads. Three of them are shared concepts -- the card, the prefill width
#: and the prefix store's budget -- and are taken from
#: :mod:`pocketllm.backends.shared_options`, differing only in what this runtime answers. The rest
#: are Xing4's own.
#:
#: ``prefill_chunk`` is the one whose real default is derived rather than declared: :func:`_chunk_for`
#: picks it from the context and the card's free memory at load, so the declaration carries ``None``
#: and the adapter resolves it -- and says so in ``--help`` via the ``resolution`` sentence.
OPTIONS: tuple[BackendOption, ...] = (
    BackendOption(
        "gguf",
        Kind.STRING,
        None,
        "the -GGUF sibling to read the weights from, when --model does not name it",
        group=Group.MODEL,
    ),
    replace(
        PREFILL_CHUNK,
        resolution="the card's free memory and the context",
    ),
    replace(PREFIX_CACHE_BYTES, default=DEFAULT_PREFIX_CACHE_BYTES),
    BackendOption(
        "tokenizer",
        Kind.STRING,
        None,
        "the directory holding the tokenizer and chat template, when --model does not",
        group=Group.MODEL,
    ),
    BackendOption(
        "use_kernel",
        Kind.FLAG,
        True,
        "run the fused kernels rather than the reference modules",
        group=Group.KERNELS,
    ),
)


def _prefill_peak_bytes(chunk: int, heads: int, context: int) -> int:
    """What one prefill chunk of `chunk` tokens costs the card at its peak.

    Two terms, both measured rather than assumed: the score path, which is
    ``SCORE_BYTES_PER_ELEMENT * heads * chunk * context``, and the per-chunk
    working set, which is linear in the chunk with a small constant under it.
    """
    return (
        SCORE_BYTES_PER_ELEMENT * int(heads) * int(chunk) * int(context)
        + PREFILL_WORKSPACE_BYTES_PER_CHUNK * int(chunk)
        + PREFILL_WORKSPACE_FLOOR_BYTES
    )


def _chunk_for(context: int, heads: int, room_bytes: int | None = None) -> int:
    """The widest prefill chunk whose peak fits `room_bytes` of the card.

    The inverse of :func:`_prefill_peak_bytes`, rounded down to a multiple of 128
    so the chunk boundary is a round number of tiles and floored at 128 because a
    chunk narrower than that is a request that cannot make progress.

    The arithmetic is why this is derived rather than defaulted.  With 3.40 GiB
    free and a 32768-position cache taking 1.41 GiB of it, a 2048-token chunk's
    score path is 2.0 GiB **per chunk** at a 32768-token context and the launch
    dies inside the softmax several minutes into its first long prompt -- which is
    the failure this function exists to prevent and which an earlier version of
    it, budgeting 4 bytes an element instead of 8, did not actually prevent.
    """
    if context <= 0 or heads <= 0:
        return DEFAULT_PREFILL_CHUNK
    room = PREFILL_SCORE_BUDGET if room_bytes is None else max(0, int(room_bytes))
    per_chunk = SCORE_BYTES_PER_ELEMENT * int(heads) * int(context) + PREFILL_WORKSPACE_BYTES_PER_CHUNK
    affordable = (room - PREFILL_WORKSPACE_FLOOR_BYTES) // per_chunk
    chunk = min(DEFAULT_PREFILL_CHUNK, max(MIN_PREFILL_CHUNK, int(affordable)))
    return max(MIN_PREFILL_CHUNK, chunk // MIN_PREFILL_CHUNK * MIN_PREFILL_CHUNK)


@dataclass(slots=True)
class _Options:
    """The launcher's levers, resolved once at construction.

    A bare ``_Options()`` is what :meth:`from_args` produces for a launch that named
    no option at all, which is the invariant ``tests/test_declared_options.py`` holds
    the declarations to: the two cannot say different things about a default.
    """

    gguf: str | None = None
    prefill_chunk: int | None = None
    prefix_cache_bytes: int = DEFAULT_PREFIX_CACHE_BYTES
    tokenizer: str | None = None
    use_kernel: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "_Options":
        values = decode_args(OPTIONS, args, runtime="xing4", ignored=IGNORED_OPTIONS)
        if not bool(getattr(args, "enable_prefix_caching", True)):
            # `--enable-prefix-caching` is the CLI's switch and the budget is the store's shape, so
            # a zero budget is how the rest of this file spells "off" -- one representation, and
            # `capabilities` reads it like any other run's.
            values["prefix_cache_bytes"] = 0
        return cls(**values)


def resolve_paths(model: str, options: _Options, tokenizer_path: str | None) -> tuple[str, str]:
    """`(gguf, tokenizer directory)` from what the launcher named.

    A Xing4 release is two directories -- the safetensors checkpoint that carries
    ``config.json``, the tokenizer and the chat template, and a ``-GGUF`` sibling
    with the quantized weights -- and a service needs both, because the GGUF's own
    header carries the vocabulary but not the merges its BPE needs.  So: the
    weights come from wherever a ``.gguf`` is, and everything else from a
    directory, and neither is guessed silently -- a launch that names a file and
    no tokenizer directory is refused with the path it looked at.
    """
    weight_path = options.gguf or model
    candidates = [Path(weight_path)]
    if Path(model).is_dir():
        candidates.append(Path(model))
    gguf = None
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            gguf = candidate
            break
        if candidate.is_dir():
            found = sorted(candidate.glob("*.gguf"))
            if len(found) == 1:
                gguf = found[0]
                break
            if len(found) > 1:
                raise ConfigurationError(
                    f"{candidate} holds {len(found)} GGUF files; name the one to serve with "
                    f"--backend-option gguf=<path>"
                )
    if gguf is None:
        raise ConfigurationError(
            f"no .gguf file at or under {weight_path}; point --model at the GGUF or name it with "
            f"--backend-option gguf=<path>"
        )

    directories = []
    if options.tokenizer:
        directories.append(Path(options.tokenizer))
    if tokenizer_path:
        directories.append(Path(tokenizer_path))
    if weight_path != model:
        directories.append(Path(model))
    directories.extend(_xing4_directories(gguf))
    for directory in directories:
        if _is_xing4_release(directory):
            return str(gguf), str(directory)
    raise ConfigurationError(
        f"the weights are at {gguf}, but no Xing4 checkpoint directory was found for the "
        f"tokenizer and the chat template; pass --tokenizer-path, or --backend-option "
        f"tokenizer=<dir>, naming the release this GGUF was converted from"
    )


def _xing4_directories(gguf: Path) -> list[Path]:
    """Where the tokenizer might be, most specific first, without guessing."""
    return [gguf.parent, *sorted(path for path in gguf.parent.parent.glob("*") if path.is_dir())]


def _is_xing4_release(directory: Path) -> bool:
    """Whether a directory is the Xing4 release rather than some other checkpoint.

    A ``config.json`` alone is not enough: the model directory here sits beside
    other checkpoints, and a sibling with a config of its own would be picked by
    an alphabetical scan and would tokenize the prompt with the wrong vocabulary.
    The vocabulary is what this needs, so the test is for the vocabulary -- the
    release ships ``tokenizer.model`` and loads it through
    ``tokenization_xing4_0.py``, and neither is present in another model's tree.
    """
    if not directory.is_dir() or not (directory / "config.json").is_file():
        return False
    if (directory / "tokenization_xing4_0.py").is_file():
        return True
    import json

    try:
        with (directory / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return False
    return str(config.get("model_type", "")).lower() == "xing4_0" 


class Xing4Backend(SchedulerHost, RuntimeAdapter):
    """One Xing4.0-29B-A4B checkpoint on one card, one request at a time."""

    #: Read by `RuntimeAdapter._tokenize`, which is the only place a runtime's name is needed.
    _RUNTIME_LABEL = "Xing4"

    name = "xing4"

    def __init__(
        self,
        args: Any,
        *,
        loader: Callable[[str, _Options], Any] | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self._options = _Options.from_args(args)
        self._model_path = str(getattr(args, "model", "") or "")
        self._tokenizer_path = str(getattr(args, "tokenizer_path", "") or "") or None
        self._max_seq_len = int(getattr(args, "max_model_len", 0) or 0) or DEFAULT_MAX_SEQ_LEN
        if self._max_seq_len < 16:
            raise ConfigurationError(
                f"max_model_len of {self._max_seq_len} leaves no room for a prompt and an answer"
            )
        self._loader = loader
        self._tokenizer = tokenizer
        self._model: Any = None
        self._cache: Any = None
        self._prefix_cache: Any = None
        #: The cards the launch named, in rank order; empty when it named none.
        self._device_ids = tuple(getattr(args, "device_ids", ()) or ())
        # The platform, and then the card. Before U3 this was one expression over two names for the
        # same thing -- the option this runtime declared, and the top level's own flag, either of
        # which a launcher might have used. `auto` resolves to `cuda` here rather than to whatever
        # the box has, because this runtime is a CUDA one: `cpu` is how a caller asks for no card at
        # all, which is what the suite's scripted model runs as, and it is the launch's word rather
        # than a fallback this adapter picked.
        self._device = str(getattr(args, "device", "") or "auto")
        if self._device == "auto":
            self._device = "cuda"
        if self._device_ids:
            # This runtime owns one process, so this process's card is the list's first entry; the
            # rank is read from the arguments all the same, so a served-from-one-process launch
            # names its card the same way a sharded one does.
            self._device = "%s:%d" % (
                self._device,
                card_for_rank(
                    self._device_ids,
                    rank=int(getattr(args, "tensor_parallel_rank", 0) or 0),
                    world=int(getattr(args, "tensor_parallel_size", 1) or 1),
                    visible=visible_card_count(),
                ),
            )
        self._heads = 0
        self._request_lock = threading.RLock()
        self._details: dict[str, Any] = {}
        # The one scheduler, when this runtime is driven by it. Built here rather than on the first
        # request because `/capabilities` has to answer whether requests go through it, and a
        # report that said no and then routed them through one would be the same class of lie as a
        # flag accepted and ignored.
        self._scheduler: Any = None
        self._native: Any = None
        self._poll_timeout_ms = int(
            getattr(args, "backend_options", {}).get("scheduler_timeout_ms", 600_000)
        )
        self._init_batch_scheduler()

    def _runtime_spec(self) -> Any:
        return RuntimeSpec(
            name=self.name,
            start=self._start_runtime,
            eos_tokens=self._eos_tokens,
            max_context=self._max_seq_len,
            wants_request=True,
            # A callable because the model is not loaded yet: this spec is built with the backend,
            # and the card the weights ended up on is only known once they are.
            device=self._runtime_device,
        )

    def _runtime_device(self) -> int:
        """The card the runtime bound, read from the model rather than from the option.

        The option is what the launcher named and the model is where the weights are. A runtime on
        the third card that never bound it on the thread running its forwards would allocate on
        the first -- torch's current device is per-thread, like the binding's own.
        """
        device = getattr(self._model, "device", None)
        if device is None:
            # A host stand-in, or a checkpoint whose model names no device. Fall back to what the
            # launcher asked for, which for a CPU run is a device index of -1.
            return device_index(self._device)
        # One reading, not two. This was a local copy of it -- `str(device)`, then the attribute,
        # then the suffix -- and the copy had the bug the shared one had: a model whose `.device`
        # is a *string* would hand `int()` the builtin `str.index` and raise. `device_index` reads
        # every shape a device arrives in, and this is one of them.
        return device_index(device)

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

        The sampling parameters come from the scheduler's request rather than from a
        `GenerationRequest`, so this is the point where the two vocabularies meet. Everything else
        is the same call `_loop` makes, minus its own cancellation flag, which is folded into the
        scheduler's step boundary: a client that disconnects calls `backend.cancel()`, and the
        request has to stop at the same seam whether the cancel came from there or from the
        scheduler retiring the request.
        """
        from src.models.xing4_0.generate import generate

        self._ensure_loaded()
        self._ensure_prefix_cache()
        eos = () if bool(sampling.ignore_eos) else self._eos_tokens()
        try:
            generate(
                self._model,
                list(prompt_ids),
                max_new_tokens=int(sampling.max_new_tokens),
                temperature=float(sampling.temperature),
                top_k=int(sampling.top_k) or None,
                top_p=float(sampling.top_p) or None,
                seed=int(sampling.seed),
                eos_token_id=eos,
                chunk=self._options.prefill_chunk,
                cache=self._cache,
                prefix_cache=self._prefix_cache,
                on_token=lambda token, _logits: on_token(token),
                # Two cancellations meet at the same seam: the runtime's own, which is called with
                # the id the client's request carries, and the scheduler's, which arrives as
                # `on_step` returning true.
                on_step=lambda: self._is_cancelled(cancel_key(context, request_id)) or on_step(),
            )
        finally:
            self._publish_cache_metrics()

    # ------------------------------------------------------------------ lifecycle

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._state_lock:
            if self._model is not None:
                return
            self._load()
            self._ready = True

    def _load(self) -> None:
        if self._loader is not None:
            self._model = self._loader(self._model_path, self._options)
        else:
            from src.models.xing4_0.gguf_model import Xing4_0GGUFModel

            gguf, directory = resolve_paths(self._model_path, self._options, self._tokenizer_path)
            self._say(f"reading {gguf}")
            self._model = Xing4_0GGUFModel(
                gguf,
                device=self._device,
                config_path=os.path.join(directory, "config.json"),
                use_kernel=self._options.use_kernel,
            )
            self._checkpoint_dir = directory
        # One cache for the life of the process, sized to the context the launcher asked for and
        # reset per request. It is 1.51 GiB at the default and allocating it per request would both
        # fragment the card and pay the zeroing every time.
        self._heads = int(getattr(getattr(self._model, "params", None), "n_heads", 0))
        self._cache = self._model.make_cache(self._max_seq_len, batch=1)
        if self._options.prefill_chunk is None:
            # Derived rather than defaulted, and derived from *this* card: the score
            # path's size is a function of the context the launcher asked for, and
            # what is left to spend on it is a function of the card. A constant
            # budget is what made the default launch unable to prefill its own
            # default context -- see `_chunk_for`.
            room = self._prefill_room()
            floor_peak = _prefill_peak_bytes(MIN_PREFILL_CHUNK, self._heads, self._max_seq_len)
            if floor_peak > room:
                # Refused here rather than left to the allocator twenty minutes
                # into the first long prompt: at this context even the narrowest
                # chunk does not fit, and the KV cache and the score workspace are
                # competing for the same few GiB.
                raise ConfigurationError(
                    f"--max-model-len {self._max_seq_len} leaves {room / 2**20:.0f} MiB above the "
                    f"{self._cache_bytes() / 2**20:.0f} MiB cache, and the narrowest prefill chunk "
                    f"({MIN_PREFILL_CHUNK} tokens) needs {floor_peak / 2**20:.0f} MiB of it at that "
                    f"context; lower --max-model-len, or raise it on a card with more room"
                )
            self._options.prefill_chunk = _chunk_for(self._max_seq_len, self._heads, room)
            self._say(
                f"prefill chunk {self._options.prefill_chunk} "
                f"(peak {_prefill_peak_bytes(self._options.prefill_chunk, self._heads, self._max_seq_len) / 2**20:.0f} MiB "
                f"of {room / 2**20:.0f} MiB free above the cache)"
            )
        if self._tokenizer is None:
            self._tokenizer = self._open_tokenizer()
        self._build_details()
        self._say(
            f"resident {self._model.nbytes / 2**30:.2f} GiB, "
            f"cache {self._cache_bytes() / 2**30:.2f} GiB over {self._max_seq_len} positions"
        )

    def _prefill_room(self) -> int:
        """Bytes the prefill may spend above what is already allocated.

        The card's own free memory, minus a reserve, read *after* the model and
        the cache are on it -- so what is left is exactly the working room.  A
        device that cannot be queried (a test, a launcher that names no card)
        falls back to the constant, which is a smaller number and therefore the
        conservative direction.
        """
        device = self._device
        try:
            # Imported here rather than at module scope: this adapter is
            # importable without a CUDA torch, which is what lets the factory and
            # the tests touch it on a box that has no card.
            import torch

            index = torch.device(device).index
            free = int(torch.cuda.mem_get_info(index)[0])
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError):
            return PREFILL_SCORE_BUDGET
        return max(PREFILL_SCORE_BUDGET // 2, free - PREFILL_DEVICE_RESERVE)

    def _ensure_prefix_cache(self) -> None:
        if self._prefix_cache is not None or self._options.prefix_cache_bytes <= 0:
            return
        if self._model is None:
            return
        from src.models.xing4_0.prefix_cache import LatentPrefixCache

        self._prefix_cache = LatentPrefixCache(
            budget_bytes=self._options.prefix_cache_bytes,
            capacity=self._max_seq_len,
            n_layers=len(self._model.blocks),
        )
        # Published here as well as after every request, so a deployment can read
        # the budget it was launched with before it has served anything.
        self._publish_cache_metrics()

    def _build_details(self) -> None:
        self._details = {
            "execution": "src/models/xing4_0 PyTorch runtime, IQ4_NL raw blocks in place",
            "scheduler": "one mutable KV cache, serialized at the backend boundary",
            "experts": "64 routed top-4 plus one shared, every expert resident on the card",
            "prefill": (
                f"one forward a {self._options.prefill_chunk}-token chunk, "
                f"sized so its peak "
                f"{_prefill_peak_bytes(self._options.prefill_chunk, self._heads, self._max_seq_len) / 2**20:.0f} MiB "
                f"fits the room left above the cache"
            ),
            "context": (
                f"{self._max_seq_len} positions, absorbed cache {self._cache_bytes() / 2**30:.2f} GiB"
            ),
            "cancellation": "per-step, at a token boundary; not inside a prefill chunk",
        }

    def _cache_bytes(self) -> int:
        if self._cache is None:
            return 0
        return sum(
            int(layer.latent.numel()) * layer.latent.element_size()
            for layer in (self._cache if isinstance(self._cache, list) else [self._cache])
        )

    def _open_tokenizer(self) -> Any:
        """The released checkpoint's own tokenizer, which is not the GGUF's.

        The GGUF carries the vocabulary, the special-token ids and the chat
        template, but ``tokenizer.ggml.model`` is ``llama`` and there are no
        merges in the header, so the reader that builds a BPE tokenizer from a GGUF
        refuses it -- correctly, because what it would build is a different
        tokenizer. The release ships ``tokenizer.model`` and a
        ``tokenization_xing4_0.py``, and a prompt rendered any other way is a
        prompt the model was not trained on.
        """
        path = getattr(self, "_checkpoint_dir", None) or self._tokenizer_path or self._model_path
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - the repo's requirements carry it
            raise ConfigurationError(
                "serving a Xing4 checkpoint needs `transformers` for its tokenizer"
            ) from exc
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"no tokenizer could be read from {path}: {exc}; pass --tokenizer-path"
            ) from exc

    def _say(self, message: str) -> None:
        print(f"[xing4] {message}", flush=True)

    # ------------------------------------------------------------------ description

    @property
    def capabilities(self) -> BackendCapabilities:
        # The store's existence rather than the option that would build one, for the reason
        # `MimoBackend.capabilities` gives: that is what "a repeated prefix is resumed" means for
        # this configuration.
        return declared_capabilities(
            self.name,
            details={
                **self._details,
                "scheduler": (
                    "BatchScheduler (width 1, continuous batching off)"
                    if self._batching()
                    else "serialized session"
                ),
                "max_batch_size": 1,
                "device": str(self._device),
            },
            reads_prefix_cache=self._prefix_cache is not None,
            # Two different claims, and the honest one depends on the path: a runtime driven by the
            # scheduler *is* submitting to a batch scheduler, on which this one declares one slot;
            # the serialized path serves one request at a time by its own lock. Either way the
            # answer to "may this be called concurrently" is what the field reports.
            supports_batch=self._batching(),
        )

    def metrics(self) -> dict[str, float]:
        """What one Xing4 engine holds between requests, for ``/metrics``.

        The card's own numbers and the store's counters, neither of which a
        request owns a share of. ``kv_cache_bytes`` against the deployment's
        ``vram_bytes`` is the question this checkpoint exists to ask -- 17.84 GiB
        of weights and 1.51 GiB of cache out of 22 -- and the store's counters are
        read from the snapshot the last request's end rebound rather than from the
        store itself, because the HTTP thread is not under the request lock.
        """
        if self._model is None:
            # No model counters yet -- but the scheduler's gauges are not the model's, and a scrape
            # during loading is when an operator most wants to know this process has one.
            return dict(self.scheduler_metrics())
        return {
            "xing4_resident_bytes": float(self._model.nbytes),
            "xing4_kv_cache_bytes": float(self._cache_bytes()),
            "xing4_context_positions": float(self._max_seq_len),
            **self._cache_metrics,
            # The scheduler's own admission state, when this runtime is driven by one. Same
            # series, same `Stats` struct and same names as the `cpp` backend publishes -- which is
            # what makes the two readable as one scheduler rather than as two servers that happen
            # to agree.
            **self.scheduler_metrics(),
        }

    # -------------------------------------------------------------------- requests

    def _eos_tokens(self) -> set[int]:
        """The ids that end a turn, from the config and the tokenizer's own end-of-text.

        Both sources are read because a run that stopped on one and not the other
        would run every answer to its budget.  There is no default: a checkpoint
        that names neither is one whose answers cannot end, and a served request
        against it should say so rather than emit its whole budget.
        """
        found: set[int] = set()
        params = getattr(self._model, "params", None)
        ids = getattr(params, "eos_token_id", None) if params is not None else None
        if isinstance(ids, int):
            found.add(int(ids))
        elif ids:
            found.update(int(token) for token in ids)
        for candidate in (
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._tokenizer, "eos_token_ids", None),
        ):
            if isinstance(candidate, int):
                found.add(int(candidate))
            elif candidate:
                found.update(int(token) for token in candidate)
        if not found:
            raise ConfigurationError(
                "this checkpoint names no end-of-turn token, so a request would run to its "
                "budget; send a prompt_tokens request with an explicit max_tokens"
            )
        return found

    def _loop(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request: GenerationRequest,
        *,
        on_token: Callable[[int], None] | None,
        on_step: Callable[[], bool] | None = None,
    ) -> Any:
        from src.models.xing4_0.generate import generate

        self._ensure_loaded()
        self._ensure_prefix_cache()
        params = request.sampling_params
        try:
            return generate(
                self._model,
                prompt_ids,
                max_new_tokens=budget,
                temperature=float(params.temperature),
                top_k=params.top_k if params.top_k is None else int(params.top_k),
                top_p=params.top_p,
                seed=params.seed,
                eos_token_id=self._eos_tokens(),
                chunk=self._options.prefill_chunk,
                cache=self._cache,
                prefix_cache=self._prefix_cache,
                on_token=None if on_token is None else (lambda token, _logits: on_token(token)),
                # Two reasons to stop, and the earlier one wins: this adapter's own cancel set,
                # which a disconnecting client fills, and whatever the caller's seam says -- the
                # scheduler's step boundary when this loop is driven by one.
                on_step=lambda: self._is_cancelled(request.request_id)
                or (on_step is not None and on_step()),
            )
        finally:
            # Under the request lock, like the run itself, so the counters are the state of the
            # store between requests rather than of one mid-prefill.
            self._publish_cache_metrics()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self._ensure_loaded()
        if self._batching():
            return self._generate_batched(requests)
        results = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                prompt_ids = self._tokenize(request)
                budget = self._budget(prompt_ids, request)
                self._check_cancelled(request.request_id)
                with self._request_lock:
                    self._check_cancelled(request.request_id)
                    marks = {"started": time.perf_counter()}
                    generation = self._loop(prompt_ids, budget, request, on_token=None)
                results.append(self._result(request, prompt_ids, generation, marks))
            finally:
                self._clear_request(request.request_id)
        return results

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield one event a token, off a worker thread, with the stop strings held back.

        The loop has to run somewhere and the events have to be yielded from here,
        and the two cannot be the same frame. A thread a request is also what lets
        a client's disconnect -- which arrives on the HTTP thread as
        :meth:`cancel` -- reach the loop at all.
        """
        self._ensure_loaded()
        self._begin_request(request.request_id)
        events: queue.Queue[TokenEvent | None] = queue.Queue(maxsize=64)
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                prompt_ids = self._tokenize(request)
                budget = self._budget(prompt_ids, request)
                streamer = TokenStreamer(
                    request_id=request.request_id,
                    stops=tuple(request.sampling_params.stop or ()),
                    events=events,
                    decode=self._decode,
                )
                with self._request_lock:
                    marks = {"started": time.perf_counter()}
                    generation = self._loop(
                        prompt_ids, budget, request, on_token=streamer.accept
                    )
                if generation.stopped == "cancel" and not streamer.hit:
                    events.put(TokenEvent(request_id=request.request_id, finish_reason="cancelled"))
                    return
                # Whatever the holdback was holding: the answer is over, so the tail is either text
                # the model really wrote or the start of a stop string that never finished, and only
                # the first of those may be sent.
                streamer.flush(generation.tokens)
                box["result"] = self._result(
                    request, prompt_ids, generation, marks, stopped="stop" if streamer.hit else None
                )
                events.put(
                    TokenEvent(
                        request_id=request.request_id,
                        finish_reason=box["result"].finish_reason,
                        usage=box["result"].usage,
                        metadata={"timings": box["result"].timings.as_dict()},
                    )
                )
            except BaseException as exc:  # a raise here must not leave the client hanging
                box["error"] = exc
            finally:
                self._clear_request(request.request_id)
                events.put(None)

        thread = threading.Thread(target=worker, name=f"xing4-{request.request_id}", daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
        finally:
            thread.join(timeout=1.0)
        if "error" in box:
            raise box["error"]

    def close(self) -> None:
        # Before the rest: the scheduler's thread runs this runtime's generation, so it has to be
        # joined while the model it drives is still resident.
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is not None:
            scheduler.stop()
        super().close()
        self._prefix_cache = None


__all__ = [
    "Xing4Backend",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_PREFILL_CHUNK",
    "PREFILL_SCORE_BUDGET",
    "SCORE_BYTES_PER_ELEMENT",
    "PREFILL_DEVICE_RESERVE",
    "MIN_PREFILL_CHUNK",
    "_chunk_for",
    "_prefill_peak_bytes",
    "resolve_paths",
]
