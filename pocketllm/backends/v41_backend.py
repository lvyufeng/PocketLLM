"""DeepSeek-V4.1-Flash under PocketLLM's backend contract.

A V4.1 checkpoint needs its own adapter rather than a flag on :mod:`torch_backend`, because the two
are different architectures and different runtimes: this one serves the released multimodal wrapper
-- a 40-layer, 5120-hidden text stack with 384 routed experts, two Engram layers and 32 index heads
-- out of ``src/models/deepseek_v4_1/``, while ``torch_backend`` loads the 0731 model through
``src/models/deepseek_v4/serving.py``. Nothing in that runtime can read this checkpoint.

What this adapter adds over ``src/cli/generate_v41.py`` is a process that outlives one prompt: the
checkpoint is read once at startup, the tree is built once, and requests then arrive as
:class:`~pocketllm.api.GenerationRequest` through the OpenAI-compatible server. What it does not add
is concurrency. ``generate.py``'s loop opens with ``front.reset_state(1)`` and then drives one
mutable KV/indexer/Engram state to the end of the generation, so two requests cannot share the tree
and ``capabilities.supports_batch`` is False. Requests serialize on one lock, which is the boundary
:class:`~pocketllm.backends.base.BackendBase` documents as the price of a request-unaware cache.

That same reset is why every request forward-passes its whole prompt: the caches are wiped at the top
of the loop, so a chat loop that resends its history pays for the history again on every turn.
``prefix_cache`` holds each prompt's state on the host keyed by the prompt's own tokens, so a later
request restores the longest prefix it shares with one already served and forwards only the rest --
vLLM's block-hash match and SGLang's longest-prefix rule, over whole-prompt anchors rather than pages.
Every rank builds the same store and reads it the same way, so the reuse is agreed on without a
collective. ``--enable-prefix-caching`` (on by default) is the switch, and it was a silent no-op on
this path until the store existed.

Tensor parallelism is the launcher's: one process a rank, ``dist.init_process_group("nccl")`` off
the standard environment that :class:`~pocketllm.supervisor.TensorParallelSupervisor` sets, and
``load_backbone(world=world, rank=rank)``. A rank computes the same logits as its peers, so the
request is broadcast once and every rank then runs the identical loop; only rank 0 returns tokens.
That is also why cancellation is a per-step collective rather than a local flag: a rank that stopped
on its own would leave its peers inside a collective nobody else enters, which is a hang rather than
an error. The collective is one ``all_reduce`` of a single int32 per decoded token, which at this
runtime's decode rate is free, and it is what makes ``DELETE /v1/requests/{id}`` and a client
disconnect do anything at all. It cannot reach inside the prompt's forward -- there is no hook there
-- so a request cancelled during prefill is observed at the first token after it.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
    GenerationResult,
    RequestCancelledError,
    TimingMetrics,
    TokenEvent,
    UnsupportedFeatureError,
    Usage,
)

from ..work_bell import Bell, BellRinger, WorkerBell, bell_path
from .base import RuntimeAdapter, settled_text
from .capabilities import IGNORED_OPTIONS, declared_capabilities
from .options import BackendOption, Group, Kind, decode_args
from .shared_options import (
    EXPERT_DEAL,
    PREFILL_CHUNK,
    PREFIX_CACHE_BYTES,
    PREFIX_CACHE_HEAD_TOKENS,
)
from .runtime_engine import (
    RuntimeSpec,
    SchedulerHost,
    cancel_key,
    card_for_rank,
    device_index,
    visible_card_count,
)


DEFAULT_MAX_SEQ_LEN = 8192
"""Context the attention caches are sized at when ``--max-model-len`` is not given.

``load_backbone`` registers those caches as buffers rather than growing them, so this is a memory
decision made at startup and not a limit that can be raised later: the default the loader would
otherwise take is the checkpoint's ``max_position_embeddings`` -- 1M, and four cards' worth of 268 MB
``freqs_cis`` tables and 536 MB compress caches per source layer. 8192 keeps a single card's worth of
slack for the prompts a first deployment actually sends, and ``--max-model-len`` is how a 256K
service asks for the shape its own acceptance run measured.
"""

DEFAULT_EXPERT_POOL_ROWS = 288
"""Arena rows a card pools for its drawn experts, the launcher's default at the same lever.

The pool is what makes ``--expert-batched`` usable, and it is a capacity a card pays for at startup.
The 4.0-4.2x prefill and 1.305-1.399x decode this is the default of are in
``docs/performance/deepseek_v4_1_flash_device_experts.md``; a 262144-token service lowers it to 148.
"""

DEFAULT_EXPERT_BUFFERS = 2

DEFAULT_PREFIX_CACHE_BYTES = 4 << 30
"""Host memory a rank's prefix store may hold, when prefix caching is on.

A stored prefix is ~4232 bytes a token plus 5.0 MiB, so 4 GiB is roughly a million tokens a rank: a
262144-token entry is 1.036 GiB, which leaves room for the handful of long conversations a local
service actually holds. It is ordinary pageable memory and deliberately *not* ``/dev/shm``, which the
resident expert bank already holds at 91%; four ranks at this size is 16 GiB.
"""

DEFAULT_PREFIX_CACHE_HEAD_TOKENS = 1024
"""The fixed-length anchor a prefill also stores, or 0 for the prompt's end alone.

The head anchor is for a *different* conversation with the same rendered header -- the same system
message, tools JSON and effort prefix -- which the end anchor cannot serve because the two prompts
diverge before it. It costs the cold prefill one chunk boundary, since a position's ring can only be
observed at a forward boundary, and it is paid on a miss only: a resumed prefill skips it.
"""

#: Every option this runtime reads, declared once. The order is the dataclass's, so the two read
#: as one list; `tests/test_declared_options.py` holds them to each other. The four entries that
#: come from `shared_options` are the concepts another runtime also reads -- one declaration each,
#: with `replace` where this runtime answers differently. `device` used to be a fifth and is not an
#: option any more: U3 moved the platform and the card out to `EngineArgs`, see `shared_options`.
OPTIONS: tuple[BackendOption, ...] = (
    BackendOption(
        "expert_device",
        Kind.STRING,
        None,
        "the card the expert arena lives on; the host when unset",
        group=Group.DEVICE,
    ),
    BackendOption(
        "expert_world",
        Kind.INTEGER,
        None,
        "how many cards the arena is spread over; the network's own width when unset",
        group=Group.EXPERT,
        minimum=1,
    ),
    BackendOption(
        "expert_cache",
        Kind.INTEGER,
        None,
        "experts the loader may keep cached a rank; the loader's own default when unset",
        group=Group.EXPERT,
    ),
    BackendOption(
        "expert_hot_rows",
        Kind.INTEGER,
        0,
        "experts of each routed layer this card keeps resident, refilled once a layer; needs "
        "`expert_device`, since there is no arena to keep anything in otherwise",
        group=Group.EXPERT,
        minimum=0,
    ),
    BackendOption(
        "expert_pool_rows",
        Kind.INTEGER,
        DEFAULT_EXPERT_POOL_ROWS,
        "arena rows a card pools for the experts a step draws",
        group=Group.EXPERT,
        minimum=0,
    ),
    BackendOption(
        "expert_buffers",
        Kind.INTEGER,
        DEFAULT_EXPERT_BUFFERS,
        "arena buffers a rank keeps in flight",
        group=Group.EXPERT,
        minimum=0,
    ),
    BackendOption(
        "expert_batched",
        Kind.FLAG,
        True,
        "one batched expert GEMM a layer rather than one call an expert",
        group=Group.EXPERT,
    ),
    replace(EXPERT_DEAL, resolution="the loader's own deal"),
    BackendOption(
        "resident_engram",
        Kind.FLAG,
        False,
        "copy both Engram tables into RAM at load instead of gathering from the shards",
        group=Group.EXPERT,
    ),
    BackendOption(
        "resident_experts",
        Kind.FLAG,
        None,
        "attach the resident expert bank; `DEEPSEEK_V41_RESIDENT_EXPERTS` answers for it when unset",
        group=Group.EXPERT,
    ),
    replace(
        PREFILL_CHUNK,
        resolution="`--prefill-chunk-tokens`, then the loader's own width",
    ),
    replace(PREFIX_CACHE_BYTES, default=DEFAULT_PREFIX_CACHE_BYTES),
    PREFIX_CACHE_HEAD_TOKENS,
    BackendOption(
        "decode_graphs",
        Kind.FLAG,
        False,
        "capture the decode block into a CUDA graph",
        group=Group.DECODE,
    ),
    BackendOption(
        "cancel_collective",
        Kind.FLAG,
        True,
        "check the cancel flag with a per-step collective, which is how a rank learns in time",
        group=Group.DECODE,
    ),
    BackendOption(
        "threads",
        Kind.INTEGER,
        None,
        "torch's intra-op thread count for this rank's host paths",
        group=Group.EXECUTION,
        minimum=1,
    ),
    BackendOption(
        "skip_special_tokens",
        Kind.FLAG,
        True,
        "drop special tokens when decoding an answer for a client",
        group=Group.EXECUTION,
    ),
    BackendOption(
        "progress",
        Kind.FLAG,
        True,
        "print the load and per-request progress lines to stderr",
        group=Group.EXECUTION,
    ),
)

_DTYPE_ALIASES = {"bf16": "bfloat16", "bfloat16": "bfloat16"}
_STREAM_QUEUE_DEPTH = 64

#: What rank 0 broadcasts to end the workers' loop.
_SHUTDOWN = "shutdown"


def _split_running(text: str, thinking: bool) -> tuple[str, str]:
    """The reasoning and the answer inside a piece of a completion that has not finished yet.

    Recomputed from the whole text rather than tracked across tokens because the text is what the
    marker is found in: ``</think>`` is plain text, so a token can carry the end of the reasoning
    and the start of the answer at once, and nothing else says where the boundary falls.
    """
    if not thinking:
        return "", text
    from src.encoding.deepseek_v4_1 import THINKING_END

    index = text.find(THINKING_END)
    if index < 0:
        return text, ""
    return text[:index], text[index + len(THINKING_END):]


def _identified(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The checkpoint's tool calls, each named with the ``id`` OpenAI's schema requires.

    The checkpoint's ``tool_calls_to_openai_format`` leaves the id out, because the calls it reads
    are read out of a *prompt*, where the id was the client's to write. A call this backend reports
    has no such history: a client that validates the response against OpenAI's typed models rejects
    a call without an id, and a client that echoes the call back -- which is how a tool result is
    attributed to the call it answers -- has nothing to name it by. So one is minted here, once per
    reported call.

    Not the streaming index: a message's calls are identified, a delta's are also *numbered*, and
    only the stream that sends them as deltas knows that numbering.
    """
    return [
        dict(call) if call.get("id") else {"id": f"call_{os.urandom(12).hex()}", **call}
        for call in tool_calls
    ]


class _AbortGeneration(Exception):
    """Stop the token loop at a step every rank agrees on.

    Raised from the per-token hook, so it unwinds all ranks of the group at the same token. The
    alternative -- letting rank 0 leave the loop on its own -- is a peer blocked in a collective, and
    the tree's state does not have to survive this: the next request opens with ``reset_state``.
    """

    def __init__(self, reason: str, text: str = "", token_ids: Sequence[int] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.text = text
        self.token_ids = [int(token) for token in token_ids]


#: What the per-step collective carries.  A stop string is rank 0's decision exactly as a cancel is,
#: so it travels the same way: a rank that unwound on "stop" while its peers continued would leave
#: them in a collective nobody enters.
_STEP_CONTINUE = 0
_STEP_CANCEL = 1
_STEP_STOP = 2


@dataclass(slots=True)
class _Marks:
    """When rank 0's request started and when its first token came back.

    Kept out of ``generate``'s own timings because the prompt's forward is inside neither: TTFT is
    what a client waits for, and it is a different number from the decode rate ``Generation``
    reports.
    """

    started: float
    first_token: float | None = None


@dataclass(slots=True)
class _Options:
    """The ``backend_options`` this adapter reads, resolved once at construction."""

    expert_device: str | None = None
    expert_world: int | None = None
    expert_cache: int | None = None
    expert_hot_rows: int = 0
    expert_pool_rows: int = DEFAULT_EXPERT_POOL_ROWS
    expert_buffers: int = DEFAULT_EXPERT_BUFFERS
    expert_batched: bool = True
    expert_deal: str | None = None
    resident_engram: bool = False
    resident_experts: bool | None = None
    prefill_chunk: int | None = None
    prefix_cache_bytes: int = DEFAULT_PREFIX_CACHE_BYTES
    prefix_cache_head_tokens: int = DEFAULT_PREFIX_CACHE_HEAD_TOKENS
    decode_graphs: bool = False
    cancel_collective: bool = True
    threads: int | None = None
    skip_special_tokens: bool = True
    progress: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "_Options":
        """The options one launch resolves to, decoded from ``OPTIONS``."""
        options = cls(
            **decode_args(OPTIONS, args, runtime="v41", ignored=IGNORED_OPTIONS)
        )
        if not bool(getattr(args, "enable_prefix_caching", True)):
            # ``--enable-prefix-caching`` is the CLI's switch and it was a no-op on this path until
            # the store existed. The two options above are the *shape* of the store, not a second
            # switch, so the CLI is what turns it off and a zero budget is how the rest of this file
            # spells "off" -- one representation, and `capabilities` reads it like any other run's.
            options.prefix_cache_bytes = 0
            options.prefix_cache_head_tokens = 0
        return options


class V41Backend(SchedulerHost, RuntimeAdapter):
    """Serve one DeepSeek-V4.1-Flash checkpoint, one request at a time.

    ``loader`` and ``front`` are injection points for tests: a unit test supplies a callable that
    returns a stand-in for :class:`~src.models.deepseek_v4_1.loader.LoadedBackbone` so the request
    path can be exercised without a 476 GiB checkpoint or four cards.
    """

    #: Read by `RuntimeAdapter._tokenize`, which is the only place a runtime's name is needed.
    _RUNTIME_LABEL = "V4.1"

    name = "v41"

    def _skip_special_tokens(self) -> bool:
        """This run's own answer, because this adapter has a flag for it.

        The base's `True` is the right default for the three runtimes that cannot configure it;
        this one can, and an operator who asked for the control tokens to be visible is asking for
        the text every other part of the answer was built from.
        """
        return bool(self._options.skip_special_tokens)

    def __init__(
        self,
        args: Any,
        *,
        front: Any = None,
        loader: Callable[[], Any] | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self._options = _Options.from_args(args)
        self._backend = str(getattr(args, "backend", "v41")).lower()
        self._checkpoint = str(args.checkpoint_dir)
        self._config_path = getattr(args, "config_path", None)
        self._tokenizer_path = getattr(args, "tokenizer_path", None) or self._checkpoint
        self._max_seq_len = getattr(args, "max_model_len", None) or DEFAULT_MAX_SEQ_LEN
        self._prefill_chunk = self._options.prefill_chunk
        self._world = int(getattr(args, "tensor_parallel_size", 1) or 1)
        self._rank = int(getattr(args, "tensor_parallel_rank", 0) or 0)
        #: The cards this launch named, in rank order -- empty when it named none, which is the
        #: launcher's older spelling and stays correct. U3 moved the card here from `backend_options`,
        #: where it was read three ways; this is the one reading.
        self._device_ids = tuple(getattr(args, "device_ids", ()) or ())
        # This rank's card. `--device-ids` answers it outright, so it is resolved here rather than at
        # the group: the tree, the expert arena's base and the NCCL binding all read this one field,
        # and resolving it once is what keeps them from disagreeing. A launch that named no ids
        # keeps the older rule and takes `LOCAL_RANK` when the group exists -- `_init_distributed`
        # does that half, because outside a group there is no `LOCAL_RANK` to take.
        self._local_rank = card_for_rank(
            self._device_ids, rank=self._rank, world=self._world,
            visible=visible_card_count(),
        )
        self._validate_args(args)

        self._front = front
        self._loader = loader
        self._tokenizer = tokenizer
        self._load_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._expert_device: str | None = None
        self._details: dict[str, Any] = {}
        self._bell: Bell | None = None
        self._prefix_cache: Any = None
        # The one scheduler, when this runtime is driven by it. Built here rather than on the first
        # request because `/capabilities` has to answer whether requests go through it, and a report
        # that said no and then routed them through one is the same class of lie as a flag accepted
        # and ignored. Off unless `enable_batching` asked for it -- see `SchedulerHost`.
        self._scheduler: Any = None
        self._native: Any = None
        self._init_batch_scheduler()

    # ------------------------------------------------------------------ construction

    def _validate_args(self, args: Any) -> None:
        """Refuse the options this runtime has no meaning for, before anything is loaded."""
        model_format = str(getattr(args, "model_format", "auto") or "auto").lower()
        if model_format not in {"auto", "safetensors"}:
            raise UnsupportedFeatureError(
                "backend='v41' reads safetensors shards only; "
                f"model_format={model_format!r} is not supported"
            )
        dtype = getattr(args, "dtype", None)
        if dtype is not None and str(dtype).lower() not in _DTYPE_ALIASES:
            raise UnsupportedFeatureError(
                "DeepSeek-V4.1-Flash chooses its own weight widths -- bf16 for the dense stack "
                "and the checkpoint's own quantized bytes for the routed banks -- so dtype is "
                f"not a knob here; dtype={dtype!r} is not supported"
            )
        for name, value in (
            ("attention_window", getattr(args, "attention_window", 0)),
            ("attention_sink_tokens", getattr(args, "attention_sink_tokens", 0)),
        ):
            if value:
                raise UnsupportedFeatureError(f"backend='v41' has no {name} knob")
        if getattr(args, "speculative_method", None):
            raise UnsupportedFeatureError("backend='v41' has no speculative decoding")
        max_batch_size = int(getattr(args, "max_batch_size", 1) or 1)
        if max_batch_size != 1:
            raise UnsupportedFeatureError(
                "the V4.1 runtime owns one mutable KV state and serves one request at a time; "
                "max_batch_size must be 1"
            )
        kv_cache_dtype = str(getattr(args, "kv_cache_dtype", "auto") or "auto").lower()
        if kv_cache_dtype != "auto":
            raise UnsupportedFeatureError(
                "the V4.1 runtime keeps bf16 attention caches; "
                f"kv_cache_dtype={kv_cache_dtype!r} is not supported"
            )

    def _resolve_expert_device(self) -> str | None:
        """Which card this rank drives its routed experts from, in the loader's own convention.

        ``loader.on_device`` reads ``--expert-device`` as where the split *starts* and adds the rank,
        so what the loader can be told is the *base* of a contiguous run: ``--device-ids 2,3`` gives
        it 2, and its own arithmetic reaches card 3 on rank 1. A list that skips a card is therefore
        refused here -- the loader has one number and cannot honour a gap, and handing it the first
        id anyway would place rank 1's experts on rank 0's card, which is a wrong answer rather than
        a slow one.

        With no ids the older rules stand: a sharded run starts at 0 and lets the rank arithmetic
        name every card, and a single process has no rank to offset, so its experts land where the
        tree does. ``expert_world`` is how a caller says one process should drive several cards.
        """
        if self._options.expert_device is not None:
            return self._options.expert_device
        if self._device_ids:
            base = int(self._device_ids[0])
            if list(self._device_ids) != [base + step for step in range(len(self._device_ids))]:
                raise ConfigurationError(
                    f"expert placement reads --device-ids as a run of adjacent cards and got "
                    f"{list(self._device_ids)}; name a contiguous list, or name the base outright "
                    "with --backend-option expert_device=cuda:N"
                )
            return f"cuda:{base}"
        if self._world > 1:
            return "cuda:0"
        return None

    def _tree_device(self) -> str | None:
        """Where this rank's copy of the dense tree lives.

        The tree follows the process group. ``tp.py`` collects each layer's attention and FFN output
        with NCCL all-reduces, and NCCL has no CPU backend, so a sharded tree left on the host dies
        at the first layer with "No backend type associated with device type cpu". ``_local_rank``
        is that placement -- ``--device-ids``'s answer when the launch named the cards, and
        ``LOCAL_RANK`` when it left them to torchrun.

        A single process has no group and no collective to satisfy, so it keeps the host tree unless
        ``--device-ids`` asks for a card; that is the shape the single-card measurements were taken
        in.
        """
        if self._device_ids or self._world > 1:
            return f"cuda:{self._local_rank}"
        return None

    # ------------------------------------------------------------------ lifecycle

    def _ensure_loaded(self) -> None:
        self._ensure_open()
        with self._load_lock:
            if self._front is None:
                self._load()
            # Here rather than inside `_load`, and under the same lock, for one reason: every path
            # that ends with a loaded `_front` passes through this method, including the two injected
            # ones -- a caller's own `front`, a caller's own `loader` -- and `_load` returns early on
            # the loader's. The lock is what keeps it a single store: the build is idempotent but not
            # atomic, and a second one would replace a store that the first is already serving from.
            self._ensure_prefix_cache()

    def _init_distributed(self) -> None:
        if self._world <= 1:
            return
        import torch
        import torch.distributed as dist

        # This rank's card, resolved before the group exists because the group is bound to it: NCCL
        # guesses the device from the global rank otherwise, and a guess that happens to be right on
        # one node is not the same thing as a placement the tree is known to sit on. An explicit
        # `--device-ids` was resolved at construction and is kept -- the launcher named the cards,
        # so `LOCAL_RANK` is torchrun's numbering and not this launch's.
        if not self._device_ids:
            try:
                self._local_rank = int(os.environ.get("LOCAL_RANK", str(self._rank)))
            except ValueError:
                self._local_rank = self._rank
        if not dist.is_initialized():
            if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
                raise ConfigurationError(
                    "tensor_parallel_size > 1 needs a rendezvous: set MASTER_ADDR/MASTER_PORT/RANK/"
                    "WORLD_SIZE (torchrun does), or launch through `pocketllm serve "
                    "--tensor-parallel-size N`, which supervises the ranks itself"
                )
            # Two hours, which is the launcher's number and not a generous one: the resident bank is
            # registered by rank 0 while the others wait at the barrier below, and that registration
            # is minutes when the segment is cold.
            dist.init_process_group(
                "nccl",
                timeout=timedelta(hours=2),
                device_id=torch.device("cuda", self._local_rank),
            )
        self._rank = int(dist.get_rank())
        self._world = int(dist.get_world_size())
        self._open_bell()

    def _open_bell(self) -> None:
        """Open this rank's end of the idle doorbell.

        Here rather than beside the broadcast it serves, because here is what makes it reliable:
        this runs at the top of ``_load`` on every rank, the barrier that closes the load runs
        after it, and rank 0's first ring runs after that -- so no rank ever rings a socket that
        has not been bound, and neither end has to retry or wait. `work_bell` has the rest.
        """
        if self._bell is not None or self._world <= 1:
            return
        # The group's own rendezvous port names the doorbell, so it is the same string on every
        # rank without a message, and different for a second engine on the same host.
        port = os.environ.get("MASTER_PORT", "")
        if self._rank == 0:
            self._bell = BellRinger([bell_path(port, rank) for rank in range(1, self._world)])
        else:
            bell = WorkerBell(bell_path(port, self._rank))
            bell.bind()
            self._bell = bell

    def _load(self) -> None:
        import torch

        started = time.perf_counter()
        self._init_distributed()
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
        if self._options.threads is not None:
            torch.set_num_threads(self._options.threads)
        if self._loader is not None:
            self._front = self._loader()
            self._ready = True
            return

        from transformers import AutoTokenizer

        from src.models.deepseek_v4_1.config import load_config, resolve_config
        from src.models.deepseek_v4_1.loader import (
            DEFAULT_EXPERT_CACHE,
            V41Checkpoint,
            build_hasher,
            load_backbone,
        )

        # Fail before the 476 GiB read rather than after it: a checkpoint without the encoder this
        # adapter renders prompts with is a configuration error, not a runtime one.
        from src.encoding.deepseek_v4_1 import encoder_path

        if encoder_path(self._checkpoint) is None:
            raise ConfigurationError(
                f"{self._checkpoint} carries no encoding/encoding.py, so backend='v41' cannot "
                "render its prompt format; point --model at a DeepSeek-V4.1-Flash checkpoint"
            )

        config_path = resolve_config(self._checkpoint, self._config_path)
        config = load_config(config_path).text
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        _, hasher = build_hasher(config, self._checkpoint, tokenizer=tokenizer)
        checkpoint = V41Checkpoint(self._checkpoint)
        self._tokenizer = tokenizer

        self._expert_device = self._resolve_expert_device()
        expert_world = self._options.expert_world
        if expert_world is None:
            expert_world = self._world if self._expert_device is not None else 1
        # No arena exists off the card path, so the pool has to become the off state rather than a
        # size the loader would only warn about; this is the launcher's `resolve_pool_rows`.
        pool_rows = int(self._options.expert_pool_rows) if self._expert_device is not None else 0
        device = self._tree_device()

        self._say(
            f"network {self._world} rank{'s' if self._world > 1 else ''}, tree on "
            f"{'the host' if device is None else device}, experts on "
            f"{'the host' if self._expert_device is None else f'{self._expert_device} x{expert_world}'}"
            f", context {self._max_seq_len}"
            + (f", {pool_rows} pooled rows a card" if pool_rows else "")
            + (f", prefill {self._prefill_chunk} tokens a forward" if self._prefill_chunk else "")
            + (", decode graphed a block at a time" if self._options.decode_graphs else "")
        )
        self._front = load_backbone(
            config,
            checkpoint,
            device=device,
            max_seq_len=self._max_seq_len,
            hasher=hasher,
            resident_engram=self._options.resident_engram,
            # Declared, accepted and -- until the declaration was written down -- never passed: the
            # loader's `resident_experts` defaults to asking `DEEPSEEK_V41_RESIDENT_EXPERTS`, so a
            # launch that named this option got the environment's answer and no error. Passing it is
            # one line and changes nothing for a launch that does not name it, because `None` is
            # what the loader falls back on.
            resident_experts=self._options.resident_experts,
            expert_cache=(
                DEFAULT_EXPERT_CACHE
                if self._options.expert_cache is None
                else int(self._options.expert_cache)
            ),
            expert_device=self._expert_device,
            expert_world=expert_world,
            expert_hot_rows=self._options.expert_hot_rows,
            expert_pool_rows=pool_rows,
            expert_buffers=self._options.expert_buffers,
            expert_batched=self._options.expert_batched,
            expert_deal=self._options.expert_deal,
            # A rank is both the tree's rank and the stagger the resident bank wants: four ranks
            # must not read /mnt/data3 at once.
            expert_rank=self._rank,
            world=self._world,
            rank=self._rank,
            progress=self._say,
        )
        self._details.update({
            "world": self._world,
            "rank": self._rank,
            "expert_device": self._expert_device,
            "expert_world": expert_world,
            "expert_pool_rows": pool_rows,
            "max_seq_len": self._max_seq_len,
            "prefill_chunk_tokens": self._prefill_chunk,
        })
        self._say(f"loaded in {time.perf_counter() - started:.1f} s")
        if self._world > 1:
            # Before the first collective rather than inside it: one rank can still be reading a
            # quarter of the tree off /mnt/data3 while another is ready to prefill.
            torch.distributed.barrier()
        self._ready = True

    def _ensure_prefix_cache(self) -> None:
        """Build this rank's prefix store, now that the tree's shapes are known.

        One store per rank and not a shared one: every rank runs the identical loop over the same
        prompt ids, so each has to be able to restore the state the same request left, and a rank
        cannot read another's card memory. The budget is the same on all four, the store is a pure
        function of the request sequence, and so the ranks agree on what a request reuses without a
        collective -- which is what keeps the per-step collectives from desynchronizing.

        A zero budget is the off switch -- ``--no-enable-prefix-caching``, or an explicit
        ``prefix_cache_bytes=0`` -- and it leaves ``_prefix_cache`` at ``None``, which ``generate``
        reads as "forward the prompt". There is no second code path behind that.

        The tag keys the store's hash chain to this run's geometry: a snapshot cannot outlive the
        process, so what it catches is a *different* geometry -- another world size, another
        ``max_seq_len`` -- reading these bytes.
        """
        if self._prefix_cache is not None:
            return
        budget = int(self._options.prefix_cache_bytes)
        if budget <= 0:
            return
        from src.models.deepseek_v4_1.prefix_cache import PrefixCache, geometry_tag

        # `LoadedBackbone` wraps the tree rather than being an `nn.Module`, and the buffers the
        # geometry is read off are the tree's -- the same `getattr` `generate` takes.
        model = getattr(self._front, "model", self._front)
        self._prefix_cache = PrefixCache(
            budget_bytes=budget,
            max_seq_len=self._max_seq_len,
            tag=geometry_tag(model, self._world, self._max_seq_len),
            head_tokens=int(self._options.prefix_cache_head_tokens),
        )
        self._details.update({
            "prefix_cache_bytes": budget,
            "prefix_cache_head_tokens": int(self._options.prefix_cache_head_tokens),
        })
        self._say(
            f"prefix cache {budget} bytes a rank, head anchor "
            f"{int(self._options.prefix_cache_head_tokens)} tokens"
        )

    def _say(self, message: str) -> None:
        if not self._options.progress:
            return
        prefix = f"[rank {self._rank}] " if self._world > 1 else ""
        print(f"{prefix}{message}", file=sys.stderr, flush=True)

    @property
    def capabilities(self) -> BackendCapabilities:
        # The gate is the budget, and a zero budget is how `--no-enable-prefix-caching` is
        # spelled here: three options fold into it, and a positive budget is the one state in
        # which this runtime resumes a prefix rather than forwarding it again.
        return declared_capabilities(
            self.name,
            details={
                "execution": "src/models/deepseek_v4_1 PyTorch runtime",
                "scheduler": (
                    "BatchScheduler (width 1, continuous batching off)"
                    if self._batching()
                    else "one mutable KV state, serialized at the backend boundary"
                ),
                "max_batch_size": 1,
                "cancellation": "per-step collective; not inside the prompt's forward",
                "prompt_format": "the checkpoint's own encoding/encoding.py, loaded by path",
                **self._details,
            },
            reads_prefix_cache=self._options.prefix_cache_bytes > 0,
            # Two different claims, and the honest one depends on the path: a runtime driven by
            # the scheduler *is* submitting to a batch scheduler, on which this one declares a
            # single slot; the serialized path serves one request at a time by its own lock.
            # Either way the answer to "may this be called concurrently" is what the field reports.
            supports_batch=self._batching(),
        )

    # ------------------------------------------------------------------ requests

    # ------------------------------------------------------------------ the shared scheduler

    def _runtime_spec(self) -> RuntimeSpec:
        """What the bridge needs to know about this checkpoint.

        `device` is a callable because the tree's card is a property of the launch rather than of
        the object, and because the run thread binds it *before* it calls this adapter's generation
        entry point -- that is, before `_ensure_loaded` has run. `_tree_device` answers from the
        options and the rank, which are known at construction; reading the card off a loaded buffer
        instead would answer `-1` on the first request and the right card on every one after it.
        """
        return RuntimeSpec(
            name=self.name,
            start=self._start_runtime,
            eos_tokens=self._eos_tokens,
            max_context=self._max_seq_len,
            wants_request=True,
            device=self._runtime_device,
        )

    def _runtime_device(self) -> int:
        return device_index(self._tree_device())

    def _eos_tokens(self) -> set[int]:
        """The id this runtime's own loop stops on, which is the one the scheduler checks here.

        Deliberately the same call `_run_payload` passes as `eos_token_id`: two readings of "the
        end of a sequence" on the two paths would be two different answers, and the one that
        disagreed would run to its budget.
        """
        token = self._eos_token_id()
        return set() if token is None else {int(token)}

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

        Almost the call `_generate_one` makes, and deliberately so. `_run` still broadcasts the
        payload to this rank's peers -- a rank 0 that entered a generation the workers were not told
        about would not be idle, it would be at a different collective -- `_step_hook` is still the
        per-token hook, `_run_payload` is still what drives the model, and the payload is the serial
        path's own builder. The bridge is additive: no model code is involved.

        Two things differ, and both are the scheduler's doing rather than the checkpoint's.

        The payload is built from `context` rather than from a `GenerationRequest` the caller
        supplied, because there is no caller here -- the scheduler is one. `context` *is* the
        request this row was submitted for, so the two routes render the same prompt, the same
        sampler and the same thinking mode by construction rather than by agreement.

        And the step boundary is the scheduler's: `on_step` is where the run parks until the next
        token is asked for, called at the end of the per-token hook -- the last point before the
        loop takes another forward. Either cancellation unwinds there, the scheduler retiring the
        row and a client that disconnected under `backend.cancel` alike, which is the same seam the
        streaming path already uses.
        """
        self._ensure_loaded()
        # The serial path's own payload builder, so the two routes cannot disagree about what they
        # sent: the budget, the sampler, the seed and the thinking mode all come from the request
        # this row was submitted for rather than from a second derivation of them.
        payload = self._payload(context, prompt_ids)
        self._validate_length(prompt_ids, payload)

        # The client's own id for this request, which is what `backend.cancel` is called with. The
        # scheduler's row id is not something a client ever sees, so keying the check on it would
        # mean a disconnect was never noticed.
        cancelled_at = cancel_key(context, request_id)

        def emit(token: int, _logits: Any) -> None:
            on_token(int(token))
            # The step boundary is the scheduler's: this is where the run parks until the next
            # token is asked for, and where either cancellation unwinds.
            if on_step() or self._is_cancelled(cancelled_at):
                raise _AbortGeneration("cancel")

        # `request=None`: the per-step hook reads stop *strings* off it, and this route applies
        # none -- `SchedulerHost._batch_sampling` records why. The engine's own stop check is
        # token-level and is what ends the row.
        with self._request_lock:
            self._run(payload, None, on_token=emit)

    def _batched_result(
        self, request: GenerationRequest, result: Any
    ) -> GenerationResult:
        """A scheduler result as this backend's own, through this runtime's own result builder.

        Overridden rather than inherited because this checkpoint's answer is not the decoded text:
        `_result` splits a finished generation into reasoning, answer and tool calls, and the
        inherited builder would return the same tokens with that reading dropped -- a client would
        see the reasoning arrive as prose on this route and as `reasoning_content` on the other.

        One field is not the same on the two routes: `usage.cached_tokens`, which the prefix store
        reports to the serial path and which the scheduler has no way to carry back. The token ids,
        the text and the finish reason are the same, which is what the two routes are compared on.
        """
        prompt_ids = self._tokenize(request)
        return self._result(
            request.request_id,
            self._payload(request, prompt_ids),
            list(result.generated_tokens),
            self._decode(list(result.generated_tokens)),
            "length" if str(result.finish_reason) == "length" else "eos",
            float(result.decode_seconds) or None,
            None,
        )

    def _encode_chat(
        self, tokenizer: Any, messages: Any, metadata: Mapping[str, Any]
    ) -> list[int]:
        """Render the prompt with the encoder the checkpoint ships.

        Not ``tokenizer.apply_chat_template``: this checkpoint has no ``chat_template`` -- its
        format is a Python module under ``<checkpoint>/encoding/`` -- so the jinja path has nothing
        to apply, and the fallback in :mod:`pocketllm.protocol.chat` would render a generic prompt
        the model was not trained on. The tools the control plane attached to the first system
        message ride along inside it; see :mod:`src.encoding.deepseek_v4_1`.
        """
        from src.encoding.deepseek_v4_1 import EncoderUnavailableError, encode_messages

        try:
            text = encode_messages(
                self._checkpoint,
                messages,
                thinking_mode=str(metadata.get("thinking_mode") or "chat"),
                reasoning_effort_value=metadata.get("reasoning_effort"),
                tools=metadata.get("tools"),
            )
        except EncoderUnavailableError as exc:
            raise ConfigurationError(
                f"{exc}; send /v1/completions with a pre-rendered prompt instead"
            ) from exc
        except ValueError as exc:  # an effort name or budget the checkpoint will not render
            raise ConfigurationError(str(exc)) from exc
        # The rendered prompt opens with its own beginning-of-sentence token, so this must not let
        # the tokenizer add a second one.
        return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]

    def _budget(self, prompt_ids: Sequence[int], request: GenerationRequest) -> int:
        """How many tokens this request may generate, from the request and these caches.

        An absent budget is resolved here and not at parse time, because the number of positions a
        request may fill is a property of these caches: everything the prompt leaves of
        `_max_seq_len`. The answer then ends at EOS or when the caches are full.

        This is the only place the number is derived. `_payload` reads it for the serial path and
        `SchedulerHost._generate_batched` reads it for the scheduler's own length check, and two
        derivations of it would be a request whose budget the engine and the scheduler disagreed
        about.
        """
        params = request.sampling_params
        return int(params.token_budget(self._max_seq_len - len(prompt_ids)))

    def _payload(self, request: GenerationRequest, prompt_ids: Sequence[int]) -> dict[str, Any]:
        params = request.sampling_params
        return {
            "op": "generate",
            "request_id": request.request_id,
            "prompt_ids": [int(token) for token in prompt_ids],
            "max_new_tokens": self._budget(prompt_ids, request),
            "temperature": float(params.temperature),
            "top_k": None if params.top_k is None else int(params.top_k),
            "seed": None if params.seed is None else int(params.seed),
            # Broadcast with the prompt so a worker unwinds on the same reading of the answer as
            # rank 0. Only rank 0 builds a result out of it.
            "thinking_mode": str(request.metadata.get("thinking_mode") or "chat"),
        }

    def _validate_length(self, prompt_ids: Sequence[int], payload: Mapping[str, Any]) -> None:
        """Refuse a request this run's caches cannot hold, before any rank is told about it.

        This is also what refuses a prompt that already fills the caches: the budget above is
        derived from what the prompt leaves, and it is floored, so such a prompt arrives here with
        a request that does not fit rather than with a budget of zero.
        """
        wanted = len(prompt_ids) + int(payload["max_new_tokens"])
        if wanted <= self._max_seq_len:
            return
        raise ConfigurationError(
            f"this request needs {wanted} positions ({len(prompt_ids)} prompt tokens and "
            f"{payload['max_new_tokens']} new), and the attention caches were sized at "
            f"{self._max_seq_len} at startup; raise --max-model-len and restart"
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self._ensure_loaded()
        if self._batching():
            return self._generate_batched(requests)
        return [self._generate_one(request) for request in requests]

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        self._begin_request(request.request_id)
        try:
            prompt_ids = self._tokenize(request)
            payload = self._payload(request, prompt_ids)
            # Before the broadcast, never after: a rank 0 that refused here would leave every peer
            # waiting on a broadcast that is not coming.
            self._check_cancelled(request.request_id)
            self._validate_length(prompt_ids, payload)
            with self._request_lock:
                self._check_cancelled(request.request_id)
                result = self._run(payload, request, on_token=None)
            self._check_cancelled(request.request_id)
            return result
        finally:
            self._clear_request(request.request_id)

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield one event a token, off a worker thread.

        The token loop has to run somewhere and the events have to be yielded from here, and the two
        cannot be the same frame: ``generate``'s hook is a callback, not a coroutine. A thread a
        request is what lets a client disconnect -- which arrives on the HTTP thread as
        ``backend.cancel`` -- reach the loop at all.

        A thinking-mode answer is split as it arrives: everything before ``</think>`` goes out as
        ``reasoning_content`` and everything after as ``content``, so a client sees the reasoning
        stream rather than waiting for the answer. Both halves are recomputed from the running text
        each token and diffed against what was already sent, for the same reason the stop-string
        match is read that way: the marker is plain text and its offset moves as the byte-level
        pieces underneath it settle. The one case that leaves is the marker straddling a token
        boundary -- a client is then sent the marker's first characters as reasoning and cannot have
        them back, which is a boundary of the format rather than of this diff. A client that needs
        the exact split can send the same request without ``stream``.

        What is sent is the *answer*, which is two things the running decode is not. A tail that is
        still half a character is withheld until the character arrives, so no client is sent the
        replacement character a byte-level decode puts there. And the answer stops where a
        tool-call block opens, so the markup of a call is not sent as prose; the call itself cannot
        be read out of a text -- the checkpoint's parser wants the end-of-sentence token, which is
        stripped from everything a client sees -- so it is read out of the finished tokens and
        travels on the last event, as OpenAI's own stream sends it.

        The last event is also where the stream and the answer are reconciled. Holding a fragment
        back means the running text can be shorter than the answer -- for a token of latency, or for
        a whole generation, when the fragment never resolved -- so what the finished reading of the
        generation has past what was sent goes out there. That is nothing in the ordinary case,
        where the loop's tokens and the events are the same tokens read the same way; it is what
        keeps the total a client received equal to the answer, for a generation that settled a
        character the running decode was still holding when the loop stopped.
        """
        self._ensure_loaded()
        self._begin_request(request.request_id)
        try:
            prompt_ids = self._tokenize(request)
            payload = self._payload(request, prompt_ids)
            self._check_cancelled(request.request_id)
            self._validate_length(prompt_ids, payload)

            events: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_DEPTH)
            outcome: list[Any] = []

            def on_token(token: int, _logits: Any) -> None:
                # Timed rather than blocking: a client that stopped reading must not hold the
                # request lock through a queue nobody drains. The cancel below is what ends it.
                while True:
                    try:
                        events.put(token, timeout=0.5)
                        return
                    except queue.Full:
                        if self._is_cancelled(request.request_id):
                            return

            def run() -> None:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.set_device(self._local_rank)
                    with self._request_lock:
                        outcome.append(self._run(payload, request, on_token=on_token))
                except BaseException as exc:  # reported on the HTTP thread, not swallowed
                    outcome.append(exc)
                finally:
                    events.put(None)

            threading.Thread(target=run, name="pocketllm-v41-stream", daemon=True).start()

            from src.encoding.deepseek_v4_1 import cut_tool_calls

            token_ids: list[int] = []
            previous_reasoning = ""
            previous_content = ""
            thinking = str(payload["thinking_mode"]) == "thinking"
            while True:
                token = events.get()
                if token is None:
                    break
                token_ids.append(int(token))
                # Both readings of the running text happen here, and each is the reason a client
                # sees the answer rather than the machinery around it: the decode's unfinished tail
                # is held back rather than sent as a replacement character, and the answer stops
                # where the tool-call block opens instead of carrying its markup.
                reasoning, content = _split_running(settled_text(self._decode(token_ids)), thinking)
                content = cut_tool_calls(content)
                yield TokenEvent(
                    request_id=request.request_id,
                    token_id=int(token),
                    text=content[len(previous_content):],
                    metadata={"reasoning_content": reasoning[len(previous_reasoning):]}
                    if reasoning != previous_reasoning
                    else {},
                )
                previous_reasoning = reasoning
                previous_content = content

            result = outcome[0] if outcome else RuntimeError("the v41 stream ended without a result")
            if isinstance(result, BaseException):
                raise result
            metadata: dict[str, Any] = {}
            # The loop's own reading of the finished generation, which is the only one that can
            # read a tool call back into OpenAI's shape: it wants the end-of-sentence token that
            # `_decode` strips, so it runs on the tokens and not on the text streamed above. The
            # block those tokens hold was already withheld from that text.
            if result.metadata.get("tool_calls"):
                metadata["tool_calls"] = [
                    {"index": index, **call}
                    for index, call in enumerate(result.metadata["tool_calls"])
                ]
            # And what the running text still owed it. The stream decodes as it goes, so anything its
            # decode was holding when the loop stopped was never sent -- and the finished reading of
            # the generation is the authority on whether any of it was the answer. It says nothing is
            # owed when it does not continue what was already sent, which is how a stop string's
            # answer, cut out of a longer running text, stays cut. A fragment of the tag is not owed
            # either: the finished answer is cut at the same place the stream's was, so it does not
            # continue it. What can be owed is a character the loop's last tokens settled, and the
            # ordinary case is that there is nothing -- this is the line that keeps a client's total
            # equal to the answer rather than one character short of it.
            tail = (
                result.text[len(previous_content):]
                if result.text.startswith(previous_content)
                else ""
            )
            yield TokenEvent(
                request_id=request.request_id,
                text=tail,
                finish_reason=result.finish_reason,
                usage=result.usage,
                metadata=metadata,
            )
        finally:
            # Only the loop reads this flag, and it reads it on the thread that is blocked here:
            # cancelling before the request leaves the active set is what lets a discontinued
            # stream's producer unwind instead of waiting on a queue with no consumer.
            self.cancel(request.request_id)
            self._clear_request(request.request_id)

    # ------------------------------------------------------------------ the loop

    def _run(
        self,
        payload: dict[str, Any],
        request: GenerationRequest,
        *,
        on_token: Callable[[int, Any], None] | None,
    ) -> GenerationResult:
        marks = _Marks(time.perf_counter())
        if self._world > 1:
            self._broadcast(payload)
        return self._run_payload(payload, request, on_token, marks)

    def _run_payload(
        self,
        payload: Mapping[str, Any],
        request: GenerationRequest | None,
        on_token: Callable[[int, Any], None] | None,
        marks: _Marks | None = None,
    ) -> GenerationResult:
        from src.models.deepseek_v4_1.generate import generate

        request_id = str(payload["request_id"])
        hook = self._step_hook(request_id, request, on_token, marks)
        try:
            generation = generate(
                self._front,
                payload["prompt_ids"],
                max_new_tokens=int(payload["max_new_tokens"]),
                temperature=float(payload["temperature"]),
                top_k=payload["top_k"],
                eos_token_id=self._eos_token_id(),
                seed=payload["seed"],
                on_token=hook,
                graphs=self._options.decode_graphs,
                prefill_chunk=self._prefill_chunk,
                prefix_cache=self._prefix_cache,
            )
        except _AbortGeneration as abort:
            if abort.reason == "cancel":
                raise RequestCancelledError(
                    f"request {request_id} was cancelled; the loop unwound at a token boundary"
                ) from None
            # No `Generation` came back, so the loop's own decode figure is gone with it; the wall
            # and the first token's timing are not, and `_result` falls back to those. The text is
            # the answer here -- see `_structured` -- so it is read as one. The reuse count went with
            # it too, and this path reports none rather than a guess.
            return self._result(
                request_id, payload, abort.token_ids, abort.text, "stop", None, marks,
                authoritative=True,
            )
        finally:
            # Under the request lock, like the run itself, so the counters are the state of the store
            # between requests rather than of one mid-prefill.
            self._publish_cache_metrics()
        self._release_graphs(generation.driver)
        return self._result(
            request_id,
            payload,
            list(generation.tokens),
            self._decode(list(generation.tokens)),
            generation.stopped,
            generation.decode_seconds,
            marks,
            cached_tokens=generation.cached_tokens,
        )

    def metrics(self) -> dict[str, float]:
        """The backend-owned metric values this rank's exporter has to publish.

        Read from the HTTP thread while a request may be mid-prefill, which is why it hands back the
        snapshot `_publish_cache_metrics` rebound rather than asking the store: ``stats()`` walks the
        store's index, and a walk concurrent with a `store` is a ``dictionary changed size``. The
        price is that a scrape during a request reports the state the *last* request left, which is
        a counter lagging by at most one generation.

        Rank 0 only: the workers hold a store of their own, because each rank restores its own
        caches, and there is no exporter on a worker to read it.
        """
        if self._rank:
            return {}
        # The scheduler's own admission state, when this runtime is driven by one. Same series,
        # same `Stats` struct and same names as the `cpp` backend publishes -- which is what makes
        # the two readable as one scheduler rather than as two servers that happen to agree.
        return {**self._cache_metrics, **self.scheduler_metrics()}

    @staticmethod
    def _release_graphs(driver: Any) -> None:
        """Hand a request's decode graphs back before the next request arrives.

        ``graphs=True`` installs a recording on every block and leaves it there -- ``Generation``'s
        ``driver`` field says so, and hands the caller that wants the eager body back
        ``driver.release()``. The launcher never needs it, because a launcher process serves one
        request and exits. A service serves the next one, and the next one's *prompt* is what pays:
        ``Block.forward`` hands every forward to ``block.decode_graph`` whatever its width, and the
        sink that recording writes into was allocated one row wide by its first real pass, so a
        1364-token prompt dies with ``RuntimeError: output with shape [1, 1, 4, 5120] doesn't match
        the broadcast shape [1, 1364, 4, 5120]`` -- measured on the four-rank service, first request
        fine and every request after it that error.

        Releasing is also what makes a request's recordings its own: a capture is taken from that
        request's prefill activation and at a position in that request's sequence, and
        ``capture_pass`` rewinds the caches it recorded through. A request that unwinds early is the
        model's own to clean up -- there is no driver to hand back here -- and ``_decode_graphs``
        does it.
        """
        if driver is not None:
            driver.release()

    def _step_hook(
        self,
        request_id: str,
        request: GenerationRequest | None,
        on_token: Callable[[int, Any], None] | None,
        marks: _Marks | None = None,
    ) -> Callable[[int, Any], None]:
        """The one point every rank reaches once a token, and so the one place to agree.

        With one rank there is nothing to agree with and this is a cancellation and stop-string
        check. With several, the decision becomes an ``all_reduce`` so that a stop only rank 0 can
        see -- a client disconnect, a stop string in the text -- is a stop every rank takes at the
        same token.
        """
        stops = tuple(getattr(getattr(request, "sampling_params", None), "stop", ()) or ())
        emitted: list[int] = []
        text_so_far = ""
        collective = self._options.cancel_collective and self._world > 1
        flag = None
        if collective:
            import torch
            import torch.distributed as dist

            flag = torch.zeros(1, dtype=torch.int32, device=self._flag_device())

        def hook(token: int, logits: Any) -> None:
            nonlocal text_so_far
            if marks is not None and marks.first_token is None:
                marks.first_token = time.perf_counter()
            code = _STEP_CONTINUE
            text = ""
            if self._rank == 0:
                emitted.append(int(token))
                if self._is_cancelled(request_id):
                    code = _STEP_CANCEL
                elif stops:
                    # The running total is the authority and not the token's own decode: a
                    # byte-level piece can be half a character, so the slice a token added is
                    # whatever this text grew by, which is also what a client is sent.
                    text_so_far = self._decode(emitted)
                    for stop in stops:
                        index = text_so_far.find(stop)
                        if index >= 0:
                            code, text = _STEP_STOP, text_so_far[:index]
                            break
            if collective:
                flag.fill_(code)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                code = int(flag.item())
            if code == _STEP_CANCEL:
                raise _AbortGeneration("cancel")
            if code == _STEP_STOP:
                raise _AbortGeneration("stop", text, emitted)
            if on_token is not None:
                on_token(token, logits)

        return hook

    def _flag_device(self) -> Any:
        """Where the per-step flag lives: the card the attention caches are on.

        Read the same way ``generate._cache_device`` reads it, and for the same reason -- nothing in
        ``load_backbone`` promises that this process's current device is the one the tree is on.
        """
        for name, buffer in self._front.model.named_buffers():
            if name.endswith("window_kv_cache"):
                return buffer.device
        import torch

        return torch.device("cpu")

    def _eos_token_id(self) -> int | None:
        tokenizer = self._tokenizer
        return None if tokenizer is None else getattr(tokenizer, "eos_token_id", None)

    def _structured(
        self,
        token_ids: Sequence[int],
        text: str,
        thinking_mode: str,
        *,
        authoritative: bool = False,
    ) -> dict[str, Any]:
        """Split a finished generation into reasoning, answer and tool calls.

        Two readings of the same tokens, because the checkpoint's parser and a client want
        different ones. The parser is the only thing that reads tool calls back into OpenAI shape
        and it wants the end-of-sentence token present -- strip it first and the parse fails -- while
        a client must not be shown special tokens at all. The parse therefore runs on the raw decode
        and its answer *excludes* that token by construction, and the tolerant split runs on the
        cleaned decode. A generation cut by ``max_tokens`` takes the second path, which is the
        ordinary case a server sees.

        ``authoritative`` is for the generation that unwound on a stop string. There the tokens are
        not the answer -- the answer is the text up to the marker, which only rank 0 has, and the
        tokens past it are ones the loop had already produced while the text it was matching had
        not caught up. Re-parsing them would report reasoning and tool calls the client is not
        being sent, so that text is read tolerantly and nothing else is consulted.

        Neither reading returns the text it was given. V4.1's tool-call block is plain text and a
        *generation*, not a message: it can stop inside the block, on a fragment of its tag, in which
        case the tag's first characters are all the text says about what was coming. So the cut is
        applied here rather than by the caller that knows about streaming, and the character a
        decode ended in the middle of is dropped here rather than by the stream that would have to
        hold it back. Both are what a client is entitled to: the answer, and only the answer.
        """
        from src.encoding.deepseek_v4_1 import cut_tool_calls, parse_strict, split_completion

        structured = None
        if not authoritative and token_ids:
            structured = parse_strict(
                self._checkpoint,
                self._decode(token_ids, skip_special_tokens=False),
                thinking_mode=thinking_mode,
            )
        if structured is None:
            # Never fall back to a fresh decode of the tokens. An empty answer is a real answer -- a
            # stop string can sit at the very first character -- and decoding them again would put
            # back exactly the text the marker had already cut off.
            structured = split_completion(text, thinking_mode=thinking_mode)
        # The block is markup in *every* reading of the text, so the cut belongs to the reading
        # rather than to the stream. The parser keeps its own content clear of it, but the tolerant
        # split does not know to stop where the block opens -- and that is the path a generation cut
        # by ``max_tokens`` takes, so without this a truncated call's half-written tag would be
        # reported as the answer's last characters. It also keeps every reading's content equal to
        # what a stream sends, which is what the stream's own held-back tail is diffed against.
        structured["content"] = settled_text(cut_tool_calls(structured["content"]))
        # And the same for the character the generation stopped in the middle of: a decode that ends
        # inside a character ends in the replacement character, which is not something the model
        # wrote and not something a client should be handed -- and a stream, which holds that tail
        # back as it arrives, has already promised as much.
        reasoning = structured.get("reasoning_content")
        if isinstance(reasoning, str):
            structured["reasoning_content"] = settled_text(reasoning)
        return structured

    def _result(
        self,
        request_id: str,
        payload: Mapping[str, Any],
        token_ids: list[int],
        text: str,
        stopped: str,
        decode_seconds: float | None,
        marks: _Marks | None,
        *,
        authoritative: bool = False,
        cached_tokens: int = 0,
    ) -> GenerationResult:
        structured = self._structured(
            token_ids, text, str(payload.get("thinking_mode") or "chat"), authoritative=authoritative
        )
        elapsed = 0.0 if marks is None else time.perf_counter() - marks.started
        # The first token comes off the prompt's forward, so when the loop reported nothing -- it
        # unwound on a stop string instead of returning -- the wall minus that forward is the
        # decode. `generate.decode_seconds` is preferred when it exists because on the graph path
        # it excludes the capture pass, which is a step that is not one of the tokens.
        ttft = (
            None
            if marks is None or marks.first_token is None
            else max(0.0, marks.first_token - marks.started)
        )
        if decode_seconds is None:
            decode_seconds = 0.0 if ttft is None else max(0.0, elapsed - ttft)
        metadata: dict[str, Any] = {
            "stopped": stopped,
            "rank0_only": True,
            "reasoning_content": structured["reasoning_content"],
        }
        tool_calls = structured["tool_calls"]
        if tool_calls:
            metadata["tool_calls"] = _identified(tool_calls)
        # A call is *why* this generation ended, which is OpenAI's own reading of the same answer: a
        # client that keys on the reason rather than reading the field would otherwise see "stop" on
        # a response that carries a call. Only "stop" is rewritten -- a generation cut by
        # ``max_tokens`` never reached its end-of-sentence token for the parser to read a call from.
        finish_reason = {"eos": "stop", "length": "length", "max_seq_len": "length"}.get(
            stopped, "stop"
        )
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        return GenerationResult(
            request_id=request_id,
            token_ids=token_ids,
            text=structured["content"],
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=len(payload["prompt_ids"]),
                completion_tokens=len(token_ids),
                # How much of that prompt the store answered instead of the cards. It is a subset of
                # `prompt_tokens`, not a discount on it: OpenAI's own `cached_tokens` counts the
                # prompt tokens served from cache, which is exactly this.
                cached_tokens=int(cached_tokens),
            ),
            timings=TimingMetrics(
                # The loop's decode figure excludes the prompt's forward but not the bookkeeping
                # around the request, so this is the prefill plus that; `ttft_seconds` is the one
                # that is only the wait.
                prefill_seconds=max(0.0, elapsed - decode_seconds),
                decode_seconds=decode_seconds,
                total_seconds=elapsed,
                ttft_seconds=ttft if ttft is not None else max(0.0, elapsed - decode_seconds),
                tpot_seconds=(decode_seconds / len(token_ids)) if token_ids else 0.0,
            ),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ multi-rank

    def _broadcast(self, payload: Any) -> Any:
        """Rank 0 hands every other rank the next thing to do, and blocks until they have it.

        Both messages this backend sends travel this way -- a request and the shutdown below -- so
        the workers are one loop reading one shape of message rather than a loop with a side channel.

        The doorbell ahead of the collective is what keeps that loop off the device. A non-root
        ``broadcast_object_list`` waits *inside* a device-side NCCL poll kernel, which holds a core
        and pins the card at 100% utilization for as long as no request is in flight -- a worker
        parked there is not idle, it is busy waiting for work. Ringing first means the wait happens
        on the host instead: an idle service costs a sleeping ``recv`` per rank and nothing on the
        cards, and the collective is entered only once rank 0 has something to put in it. The ring
        is counted, not latched, so a worker that has not yet reached this call for one message
        still consumes the rings in order.
        """
        import torch.distributed as dist

        bell = self._bell
        if bell is not None:
            if self._rank == 0:
                bell.ring()
            else:
                bell.wait()
        box = [payload]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Load, announce, then serve rank 0's requests until it broadcasts a shutdown."""
        self._ensure_open()
        # The rank has to come from the group and not from the arguments: a supervised launch sets
        # both, but a torchrun one only sets the environment, and rank 0 must not enter this loop.
        self._init_distributed()
        if self._world <= 1:
            raise UnsupportedFeatureError(
                "run_worker serves rank 0's requests over a process group; a single-process v41 run "
                "serves them through the HTTP server instead"
            )
        if self._rank == 0:
            raise UnsupportedFeatureError("run_worker must not be called on rank 0")
        self._ensure_loaded()
        if on_ready is not None:
            on_ready()
        while not self._closed:
            payload = self._broadcast(None)
            if not isinstance(payload, Mapping) or payload.get("op") == _SHUTDOWN:
                break
            if payload.get("op") != "generate":
                continue
            try:
                self._run_payload(payload, None, None)
            except (RequestCancelledError, _AbortGeneration):
                # Every rank unwinds together, so a cancelled or stopped request is not a
                # desynchronized group.
                pass

    def close(self) -> None:
        already_closed = self._closed
        # Rank 0 is the only rank that broadcasts, and the shutdown has to be sent before
        # the base class reaps the group: once the ranks this process spawned are gone
        # there is nobody left to receive it, and once this rank leaves the group a
        # broadcast is a collective without a peer.  A worker rank that reaches the end of
        # run_worker is already out of the message it was sent, so it has nothing to send.
        if not already_closed and self._world > 1 and self._rank == 0:
            try:
                self._broadcast({"op": _SHUTDOWN})
            except Exception:
                # A peer that already left is not this rank's problem, and close() must not raise.
                pass
        # After the shutdown, never before: a worker parked on its bell reads that socket closing
        # as the end of the group, and it has to be handed the shutdown it was sent first.
        if self._bell is not None:
            self._bell.close()
        # Before the rest of the teardown, and after the shutdown above: the scheduler's thread
        # runs this runtime's generation, so it has to be joined while the model it drives is still
        # resident -- and a worker parked on its bell has to have been sent the shutdown first.
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is not None:
            scheduler.stop()
        super().close()


__all__ = ["V41Backend", "DEFAULT_MAX_SEQ_LEN"]
