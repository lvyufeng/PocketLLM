"""MiMo-V2.6-Flash under PocketLLM's backend contract.

A MiMo checkpoint needs its own adapter rather than a flag on :mod:`torch_backend`, because it is
neither of the runtimes already here: this one serves Xiaomi's 48-layer text backbone -- nine
global-attention layers and thirty-nine sliding-window ones, 256 routed experts a layer, MXFP4
weights out of a host-resident bank -- through ``src/models/mimo_v2/``, and nothing in that package
reads a GGUF file or a Qwen-shaped safetensors tree.

What this adapter adds over ``tests/bench_mimo_v2_prefill.py`` is a process that outlives one prompt.
The checkpoint is read once at startup, the expert bank is attached (or filled, once, if the host
has none yet), the tree is built once, and requests then arrive as
:class:`~pocketllm.api.GenerationRequest` from the OpenAI-compatible server. What it does not add is
concurrency: ``capabilities.supports_batch`` is False, because one mutable KV cache serves one
sequence here and two requests cannot share it. Requests serialize on one lock, which is the
boundary :class:`~pocketllm.backends.base.BackendBase` documents as the price of a request-unaware
cache.

**What outlives a request is the prefix store, and it is per rank.** ``generate.py``'s loop resets the
cache at the top, so a chat loop that resends its history pays for the history on every turn;
``prefix_cache`` holds each prompt's state on the host keyed by the prompt's own tokens, so a later
request restores the longest prefix it shares with one already served and forwards only the rest.
Every rank builds the same store and reads it the same way -- the key is the tokens, the budget and
the anchor lengths are launcher options, and each rank's own buffers are a quarter of the heads -- so
the reuse is agreed on *without a collective*, which is the thing that matters here: a rank that
resumed where its peers did not would enter a layer's all-reduce alone and hang. The one thing that
makes this work at four ranks and not at one is that the decision has to be a function of the request
and the options and of nothing local, and it is: the store's index is keyed by tokens, and the byte
accounting that drives eviction is the same number on every rank because the split is even.
``--enable-prefix-caching`` (on by default) is the switch.

**The experts are dealt, and the deal is a property of the layer's shape.** A decode step's draw is
eight experts of 256, so the ``sorted`` deal -- sorted position to rank -- gives each of four ranks
exactly two of them and one all-reduce of a ``[1, 4096]`` fp32 partial closes the layer. A prompt is
not a draw: a chunk of two thousand tokens reaches nearly every expert, so the prefill's deal is
``id``, which partitions the *experts* and gives a rank a quarter of them to stage instead of all of
them. Neither can stand in for the other -- a chunked request under ``sorted`` stages every expert
on every rank, four times the copy -- so a served model holds **both arenas**, and the row count of
the call is the dispatch: one row is a step through the deal's own module, more than one is a chunk
through the ``id`` one. The second arena is two rows a slot, which is 51 MiB.
``backend_options['expert_deal']`` still names the step's deal, and ``chunk_rows`` the band.

**Tensor parallelism is the launcher's**, exactly as for V4.1: one process a rank,
``dist.init_process_group("nccl")`` off the environment the supervisor sets, and
``EpGroup.from_env()``. Every rank computes the same logits -- the collective returns the same sum
everywhere -- so a greedy loop is run identically on all four without a message between them, and
only rank 0 returns tokens. Cancellation is the one thing that cannot be local: a rank that stopped
on its own would leave its peers inside the collective nobody else enters, so a cancel is a per-step
``broadcast`` of one int32 from rank 0, which is what makes ``DELETE /v1/requests/{id}`` and a client
disconnect do anything at all.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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

from .base import BackendBase, TokenStreamer, settled_text
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
    card_for_rank,
    device_index,
    visible_card_count,
)

DEFAULT_MAX_SEQ_LEN = 32768
"""Positions the KV cache is sized at when ``--max-model-len`` is not given.

The cache is allocated once at startup and reset per request, so this is a memory decision rather
than a limit that can be raised later: 23 KB a token across the nine global layers and the
thirty-nine rings, which is 1.43 GiB at 64k and **5.65 GiB at 262144** -- the size a 256k service
asks for with ``--max-model-len 262144``, measured in
``docs/models/mimo-v2.6-flash.md``. The default is the smaller number because a first deployment
sends prose, and the rings make the difference between the two prices smaller than it looks.
"""

DEFAULT_PREFILL_CHUNK = 2048
"""Tokens a prefill call, which is the width the prompt's rate was measured at.

A chunk's expert bytes are a cost per *call* and not a token, so a wider chunk amortises the copy
and a narrower one costs calls. The width that wins is not monotone in the context: at a 64k prompt
a 4096-token chunk is the faster one, 104.4 tokens a second against 95.5, and at 262144 the same
width did not finish in four hours where 2048 finished in under two -- the attention's cost per
prefill token is 19.4 ms at that depth against 3.6 at 32k, and a wider chunk attends to more keys
per token in exactly the way that number describes. 2048 is therefore the default, because a
service has to answer at whatever `--max-model-len` its launcher named rather than at the one width
its prompt happens to favour; a deployment that only ever sends short prompts can take the 4096.
"""

DEFAULT_EXPERT_ROWS = 16
"""Experts one expert-kernel call stages, which is the arena's width in bands.

The arena has to be as wide as the experts one call holds, so this is card memory against call
count: 16 experts is 408 MiB of arena a slot and four calls a layer, 64 is 1632 MiB and one call.
The measured cost of the narrower band is 17-21% of the prompt's rate, and what it buys is 1.2 GiB
of a 22 GiB card -- which is the difference between a 256k context that fits and one that does not,
because the cache is 5.65 GiB of it.
"""

DEFAULT_EXPERT_SLOTS = 2
"""Expert arena slots: one call in flight while the next one's copy lands."""

DEFAULT_RESIDENT_ROWS = 0
"""Experts of each routed layer the card keeps, which is the one lever the decode step has on bytes.

A decode token stages ``top_k / world`` experts a layer -- 94 copies, 1198.5 MiB a rank, at a world
of four -- and the layers are a chain, so the copy chain is a floor under the step that no host work
can hide. Holding a layer's hottest ``resident_rows`` experts on the card removes their copies, and
the set is learned from the draws as they arrive rather than calibrated, because a resident row holds
the bytes a staging row would have held: the answer is identical under *any* policy, which is what
makes the policy a performance question with no correctness constraint on it.

It is off by default because it is card memory and the amount that fits is a function of the
context. A routed layer costs ``resident_rows`` experts -- 12.75 MiB each at the released
dimensions -- so eight rows is 4.78 GiB of a 22 GiB card: free at a short context and impossible at
262144, where the cache alone is 5.65 GiB. A deployment that serves short prompts should turn it on
and pay for the rows; see ``docs/models/mimo_v2_6_flash.md`` for what it measured.
"""

DEFAULT_PREFIX_CACHE_BYTES = 4 << 30
"""Host memory a rank's prefix store may hold, when prefix caching is on.

A stored prefix is 5760 bytes a token plus 6.1 MiB of rings, so 4 GiB is roughly seven hundred
thousand tokens a rank: a 262144-token entry is 1.41 GiB, which leaves room for the handful of long
conversations a local service actually holds. It is ordinary pageable memory and deliberately *not*
``/dev/shm``, which the 149.81 GiB expert bank already owns at 91%; four ranks at this size is 16 GiB.

Both figures are the served shape's: four ranks, with the attention split along the checkpoint's own
partition, which is a quarter of the key heads a rank and therefore a quarter of these bytes. A rank
that held the whole attention would pay 23040 bytes a token.

The rings are why the constant per entry is not zero: all thirty-nine of them together are 6.1 MiB
whatever the prompt is, because a windowed layer's buffer is `min(window, capacity)` slots and the
prompt never changes that. It is also why the store's floor is a whole hash block -- an entry of
eight tokens would spend the rings' six megabytes on eight tokens of prefix.
"""

DEFAULT_PREFIX_CACHE_HEAD_TOKENS = 1024
"""The fixed-length anchor a prefill also stores, or 0 for the prompt's end alone.

The head anchor is for a *different* conversation with the same rendered header -- the same system
message, the same tools JSON -- which the end anchor cannot serve because the two prompts diverge
before it. It costs the cold prefill one chunk boundary, since a position can only be observed at a
forward boundary, and it is paid on a miss only: a resumed prefill skips it.
"""

#: Every option this runtime reads, declared once. The five concepts this shares with another
#: runtime -- the card, the prefill width, the two prefix-store numbers and the deal -- are taken
#: from :mod:`pocketllm.backends.shared_options` and differ from the shared shape only in what this
#: runtime answers, which is the one argument passed to :func:`dataclasses.replace`. ``chunk_rows``
#: keeps its second name: the adapter answered to ``expert_rows`` before it was ``MimoBackend``, and
#: a launch script that still says it keeps working.
OPTIONS: tuple[BackendOption, ...] = (
    BackendOption(
        "chunk_rows",
        Kind.INTEGER,
        DEFAULT_EXPERT_ROWS,
        "experts one expert-kernel call stages, which is the arena's width in bands; `null` leaves "
        "this a decode model, feeding a prompt one token at a time",
        group=Group.EXPERT,
        aliases=("expert_rows",),
        minimum=0,
    ),
    replace(
        EXPERT_DEAL,
        default="sorted",
        # This runtime's own resolution: unset is not "no deal", it is the process's environment.
        help=EXPERT_DEAL.help + "; `null` takes the process's POCKETLLM_MIMO_EXPERT_DEAL",
    ),
    BackendOption(
        "pin",
        Kind.FLAG,
        True,
        "page-lock the staging buffers, which is what makes the copies asynchronous",
        group=Group.EXPERT,
    ),
    replace(PREFILL_CHUNK, default=DEFAULT_PREFILL_CHUNK),
    replace(PREFIX_CACHE_BYTES, default=DEFAULT_PREFIX_CACHE_BYTES),
    PREFIX_CACHE_HEAD_TOKENS,
    BackendOption(
        "resident_rows",
        Kind.INTEGER,
        DEFAULT_RESIDENT_ROWS,
        "experts of each routed layer the card keeps instead of staging a token",
        group=Group.EXPERT,
        minimum=0,
    ),
    BackendOption(
        "slots",
        Kind.INTEGER,
        DEFAULT_EXPERT_SLOTS,
        "expert arena slots: one call in flight while the next one's copy lands",
        group=Group.EXPERT,
        minimum=1,
    ),
)


@dataclass(slots=True)
class _Options:
    """The launcher's levers, resolved once at construction.

    Every one of these changes what the run does, which is why an unknown key is refused rather
    than ignored: ``chunk_rows`` is the arena a card pays for, ``expert_deal`` is which deal the
    experts are divided by, ``prefill_chunk`` is the width a prompt goes through at, ``slots`` is how
    many calls the pipeline keeps in flight, ``resident_rows`` is how many of a routed layer's
    experts the card keeps instead of copying a token, and the two ``prefix_cache_*`` keys are the
    store's budget and the length of the head anchor.

    The field names are the declarations' canonical names, which is why the deal is ``expert_deal``
    here and was ``deal``: the flag's own name is the one the two runtimes that read it agree on.
    """

    chunk_rows: int | None = DEFAULT_EXPERT_ROWS
    expert_deal: str | None = "sorted"
    pin: bool = True
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    prefix_cache_bytes: int = DEFAULT_PREFIX_CACHE_BYTES
    prefix_cache_head_tokens: int = DEFAULT_PREFIX_CACHE_HEAD_TOKENS
    resident_rows: int = DEFAULT_RESIDENT_ROWS
    slots: int = DEFAULT_EXPERT_SLOTS

    @classmethod
    def from_args(cls, args: Any) -> "_Options":
        values = decode_args(OPTIONS, args, runtime="mimo", ignored=IGNORED_OPTIONS)
        budget = values["prefix_cache_bytes"]
        head = values["prefix_cache_head_tokens"]
        if not bool(getattr(args, "enable_prefix_caching", True)):
            # ``--enable-prefix-caching`` is the CLI's switch. The two options above are the *shape*
            # of the store and not a second switch, so the CLI is what turns it off and a zero budget
            # is how the rest of this file spells "off" -- one representation, and `capabilities`
            # reads it like any other run's. This is the same pair of lines the V4.1 adapter has, and
            # for the same reason: a launch's two spellings of "no prefix cache" have to agree.
            budget = 0
            head = 0
        return cls(
            chunk_rows=values["chunk_rows"],
            expert_deal=values["expert_deal"],
            pin=values["pin"],
            prefill_chunk=values["prefill_chunk"],
            prefix_cache_bytes=budget,
            prefix_cache_head_tokens=head,
            resident_rows=values["resident_rows"],
            slots=values["slots"],
        )


class MimoBackend(SchedulerHost, BackendBase):
    """One MiMo-V2.6-Flash checkpoint, one process a rank, one request at a time."""

    name = "mimo"

    def __init__(
        self,
        args: Any,
        *,
        loader: Callable[[Any, _Options], Any] | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self._options = _Options.from_args(args)
        #: The cards the launch named, in rank order; empty when it named none. U3 moved the card
        #: here from `backend_options`, where this runtime read it as its own device and the top
        #: level read the same name as a different one.
        self._device_ids = tuple(getattr(args, "device_ids", ()) or ())
        self._checkpoint_dir = str(getattr(args, "model", "") or "")
        self._tokenizer_path = str(getattr(args, "tokenizer_path", "") or "")
        self._max_seq_len = int(getattr(args, "max_model_len", 0) or 0) or DEFAULT_MAX_SEQ_LEN
        if self._max_seq_len < 16:
            raise ConfigurationError(
                f"max_model_len of {self._max_seq_len} leaves no room for a prompt and an answer"
            )
        self._loader = loader
        self._tokenizer = tokenizer
        self._checkpoint: Any = None
        self._bank: Any = None
        self._model: Any = None
        self._cache: Any = None
        self._prefix_cache: Any = None
        self._ep: Any = None
        self._device: Any = None
        self._world = 1
        self._rank = 0
        self._request_lock = threading.RLock()
        self._distributed = False
        self._details: dict[str, Any] = {}
        # The one scheduler, when this runtime is driven by it. Built here rather than on the first
        # request because `/capabilities` has to answer whether requests go through it, and a report
        # that said no and then routed them through one is the same class of lie as a flag accepted
        # and ignored. Off unless `enable_batching` asked for it -- see `SchedulerHost`.
        self._scheduler: Any = None
        self._native: Any = None
        self._init_batch_scheduler()

    # ------------------------------------------------------------------ lifecycle

    def prepare(self) -> None:
        self._ensure_open()
        self._ensure_loaded()
        # Here rather than inside `_load`, and for the reason the V4.1 adapter gives: every path that
        # ends with a loaded model passes through `_ensure_loaded`, including the injected ones, and
        # a store built in only one of them would be a switch that quietly does nothing on the others.
        self._ensure_prefix_cache()

    def _init_distributed(self) -> None:
        """Join the group the launcher set, if there is one.

        Read from the environment and not from the arguments, because a ``torchrun`` launch sets
        only the environment and a supervised one sets both: two sources that could disagree are
        one source too many, and the group is the thing that has to be right.
        """
        if self._distributed:
            return
        from src.models.mimo_v2.ep import EpGroup

        # Which card this rank drives. `--device-ids` is the launch's answer and `torchrun`'s
        # `LOCAL_RANK` is the fallback, and the group is asked before either is used: a world of one
        # has no rank to offset and takes the named card as it stands, and a group of more asks for
        # it through the same `device` field the model then reads. Unnamed and unsharded stays `None`,
        # which is what `EpGroup` reads as the current device.
        device = self._card()
        self._ep = EpGroup.from_env(device=device)
        self._world, self._rank = self._ep.world, self._ep.rank
        self._device = self._ep.device
        self._distributed = True

    def _card(self) -> int | None:
        """The card this launch named for this rank, or ``None`` when it named none.

        The world the ids are indexed by comes from the arguments and not from the group, because
        this runs *before* the group exists -- the group is the thing being told which card to bind.
        They agree wherever both are set, and the argument is the one that exists in both cases.
        """
        if not self._device_ids:
            return None
        return card_for_rank(
            self._device_ids,
            rank=int(getattr(self.args, "tensor_parallel_rank", 0) or 0),
            world=int(getattr(self.args, "tensor_parallel_size", 1) or 1),
            visible=visible_card_count(),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._state_lock:
            if self._model is not None:
                return
            self._init_distributed()
            if self._loader is not None:
                # An injected loader is a test's or an embedding application's model, and it comes
                # with its own tokenizer: nothing here reaches for the checkpoint, so a run with
                # one never needs a checkpoint directory on disk.
                self._model = self._loader(self.args, self._options)
                if self._cache is None:
                    self._cache = self._model.cache(self._max_seq_len)
                self._build_details()
                self._ready = True
                return
            self._load()

    def _load(self) -> None:
        """Read the checkpoint, attach the expert bank, build the tree and the cache.

        The bank is 149.81 GiB of ``/dev/shm`` shared by every rank, and attaching it is a
        ``cudaHostRegister`` of the whole mapping rather than a copy: the first run on a host that
        has never built one pays a twelve-minute fill from the release, and every run after it
        pays the pin. Both are startup work, and both are why the supervisor's rendezvous timeout
        is measured in hours rather than in minutes.
        """
        from src.models.mimo_v2.bank import open_expert_bank
        from src.models.mimo_v2.device_model import MimoV2DeviceModel
        from src.models.mimo_v2.loader import MimoV2Checkpoint

        options = self._options
        if not os.path.isdir(self._checkpoint_dir):
            raise ConfigurationError(f"no checkpoint at {self._checkpoint_dir}")
        self._checkpoint = MimoV2Checkpoint(self._checkpoint_dir)
        self._bank = open_expert_bank(self._checkpoint, progress=self._say)
        self._model = MimoV2DeviceModel(
            self._checkpoint,
            device=self._device,
            expert_source=self._bank,
            ep=self._ep,
            slots=options.slots,
            deal=options.expert_deal,
            chunk_rows=options.chunk_rows,
            pin=options.pin,
            resident_rows=options.resident_rows,
        )
        # One cache for the life of the process, sized to the context the launcher asked for and
        # reset per request. It is 5.65 GiB at 262144 and allocating it per request would both
        # fragment the card and pay the zeroing every time.
        self._cache = self._model.cache(self._max_seq_len)
        self._build_details()
        if self._tokenizer is None:
            self._tokenizer = self._open_tokenizer()
        self._publish_ready()

    def _build_details(self) -> None:
        experts = getattr(self._model, "experts", None)
        chunks = getattr(self._model, "chunk_experts", None)
        cache = getattr(self._cache, "memory_bytes", None)

        def describe(module) -> str:
            kept = (
                ""
                if not module.resident_rows
                else (
                    f", the hottest {module.resident_rows} experts a routed layer resident "
                    f"({module.resident_bytes / 2**30:.2f} GiB)"
                )
            )
            if module.chunk_rows is None:
                return (
                    f"one draw a step under the `{module.deal}` deal, "
                    f"{module.arena_bytes / 2**20:.0f} MiB arena a slot{kept}"
                )
            return (
                f"{module.n_local} experts a rank in bands of {module.chunk_rows}, "
                f"`{module.deal}` deal, {module.arena_bytes / 2**20:.0f} MiB arena a slot"
            )

        self._details = {
            "execution": "src/models/mimo_v2 PyTorch runtime",
            "scheduler": "one mutable KV cache, serialized at the backend boundary",
            "experts": (
                "host-resident bank over PCIe; one draw a step"
                if experts is None
                else describe(experts)
            ),
            "prefill": (
                f"grouped multi-token kernel, {self._options.prefill_chunk} tokens a call"
                + (
                    f"; a chunk deals by `id` through a second arena, {describe(chunks)}"
                    if chunks is not None
                    else ""
                )
            ),
            "context": (
                f"{self._max_seq_len} positions, cache {cache / 2**30:.2f} GiB"
                if cache is not None
                else f"{self._max_seq_len} positions"
            ),
            "cancellation": "per-step broadcast; not inside the prompt's forward",
        }

    def _publish_ready(self) -> None:
        if self._world > 1:
            import torch.distributed as dist

            dist.barrier()
        self._ready = True

    def _open_tokenizer(self) -> Any:
        """The checkpoint's own tokenizer, out of its ``tokenizer.json`` and chat template.

        Not a fallback and not a generic renderer: MiMo's chat markup is its own (``<|im_start|>``,
        ``<|im_end|>``), it ships the template as ``chat_template.jinja`` next to the weights, and a
        prompt rendered any other way is a prompt the model was not trained on.
        """
        path = self._tokenizer_path or self._checkpoint_dir
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - the repo's requirements carry it
            raise ConfigurationError(
                "serving a MiMo checkpoint needs `transformers` for its tokenizer"
            ) from exc
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"no tokenizer could be read from {path}: {exc}; pass --tokenizer-path"
            ) from exc

    def _say(self, message: str) -> None:
        if self._world <= 1 or self._rank == 0:
            print(f"[mimo] {message}", flush=True)

    def _ensure_prefix_cache(self) -> None:
        """Build this rank's store, once, or leave it off and say why.

        Idempotent and called from every path that is about to serve: ``prepare`` so the store exists
        before the first request and ``capabilities`` can report it, and the two request funnels so a
        caller that never prepared -- a test, an embedding application, the worker loop -- gets the
        store rather than a silent `None`. A second build would replace a store the first is already
        serving from, so the check-and-build is under the state lock.

        **Every rank builds one, and that is not a choice.** A rank that resumed a prompt its peers
        forward-passed would enter a layer's ``all_reduce`` alone, which NCCL answers by hanging.
        Nothing is broadcast to keep them agreeing because nothing has to be: the store is keyed by
        the prompt's own tokens, and the budget, the anchor lengths and the geometry tag are the same
        on every rank -- the split attention is even, so a snapshot's byte count is the same number
        everywhere and the eviction decisions are the same sequence.

        A cache that does not describe its layers is a stand-in rather than a cache, and the store is
        left off with the reason recorded in ``capabilities.details``: the payload is the buffers a
        snapshot walks, and a store over a cache nobody can walk would report misses forever while
        holding nothing, which is a run whose numbers mean something different from the ones a
        deployment gets.
        """
        with self._state_lock:
            if self._prefix_cache is not None:
                return
            budget = int(self._options.prefix_cache_bytes)
            if budget <= 0 or self._cache is None:
                return
            if not hasattr(self._cache, "layers"):
                self._details["prefix_cache"] = (
                    "off: this run's cache describes no state to snapshot"
                )
                return
            from src.models.mimo_v2.prefix_cache import PrefixCache, geometry_tag

            head = int(self._options.prefix_cache_head_tokens)
            self._prefix_cache = PrefixCache(
                budget_bytes=budget,
                max_seq_len=self._max_seq_len,
                tag=geometry_tag(self._cache, self._world, self._max_seq_len),
                head_tokens=head,
            )
            self._details.update({
                "prefix_cache_bytes": budget,
                "prefix_cache_head_tokens": head,
            })
            self._say(f"prefix cache {budget} bytes a rank, head anchor {head} tokens")

    # -------------------------------------------------------------------- contract

    @property
    def capabilities(self) -> BackendCapabilities:
        # The store's existence rather than the option that would build one: a budget the launcher
        # set on a cache that cannot be snapshotted is a run with no store, and a capability that
        # reported it anyway is how a deployment sizes a budget it never gets. That is the gate,
        # and it is the same question the declaration asks -- whether a repeated prefix is
        # resumed -- answered for this configuration.
        return declared_capabilities(
            self.name,
            details={
                **self._details,
                # After `_details`, because that one is built when the model is and says what the
                # serialized path does. This one is read live, before anything is loaded.
                "scheduler": (
                    "BatchScheduler (width 1, continuous batching off)"
                    if self._batching()
                    else "one mutable KV cache, serialized at the backend boundary"
                ),
                "max_batch_size": 1,
            },
            reads_prefix_cache=self._prefix_cache is not None,
            # Two different claims, and the honest one depends on the path: a runtime driven by
            # the scheduler *is* submitting to a batch scheduler, on which this one declares a
            # single slot; the serialized path serves one request at a time by its own lock.
            # Either way the answer to "may this be called concurrently" is what the field reports.
            supports_batch=self._batching(),
        )

    def metrics(self) -> dict[str, float]:
        """What one MiMo engine holds between requests, for ``/metrics``.

        The expert bank's own counters and the draw's per-token weight are properties of the run
        rather than of a request: what a step costs is a function of the deal and the arena, and a
        client reading a rate wants to know which one it was measured under. The prefix store's
        counters ride along for the same reason -- no request owns a share of a store that outlives
        it -- and they are read from the snapshot the last request's end rebound rather than from the
        store itself: the HTTP thread is not under the request lock, and a `stats()` walk concurrent
        with a `store` is a `dictionary changed size`.
        """
        experts = getattr(self._model, "experts", None)
        if experts is None:
            # Nothing loaded, so there are no model counters to add. The scheduler's gauges are not
            # the model's, though, and a scrape during loading is exactly when an operator wants to
            # know whether this process has a scheduler at all -- so they are published either way.
            return dict(self.scheduler_metrics())
        # Two arenas when there are two, and the counts are reported separately because they
        # answer different questions: a step's share is a property of the deal over the drawing
        # and a chunk's is a share of the expert set. The step module of a `sorted` run holds no
        # band at all, so a single number would read as zero experts on a card staging two.
        chunks = getattr(self._model, "chunk_experts", None)
        held = experts if chunks is None else chunks
        return {
            "mimo_experts_staged_total": float(experts.staged_experts),
            "mimo_expert_bytes_staged_total": float(experts.staged_bytes),
            "mimo_arena_bytes": float(experts.arena_bytes),
            "mimo_experts_local": float(held.n_local),
            "mimo_experts_per_call": float(held.chunk_rows or 0),
            "mimo_kv_cache_bytes": float(self._cache.memory_bytes) if self._cache else 0.0,
            "mimo_world": float(self._world),
            **self._cache_metrics,
            # The scheduler's own admission state, when this runtime is driven by one. Same series,
            # same `Stats` struct and same names as the `cpp` backend publishes -- which is what
            # makes the two readable as one scheduler rather than as two servers that agree.
            **self.scheduler_metrics(),
        }

    # -------------------------------------------------------------------- requests

    def _tokenize(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            return list(request.prompt_tokens)
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("the MiMo tokenizer is not loaded")
        messages = request.metadata.get("messages")
        if messages:
            return self._encode_chat(tokenizer, messages, request.metadata)
        # A raw completion prompt is not chat and gets no header.
        return [int(token) for token in tokenizer(request.prompt)["input_ids"]]

    def _encode_chat(
        self, tokenizer: Any, messages: Any, metadata: Mapping[str, Any]
    ) -> list[int]:
        """Render a chat with the template the checkpoint ships."""
        tools = metadata.get("tools")
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        try:
            text = tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, **kwargs
            )
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"the chat template refused this conversation: {exc}") from exc
        # The rendered prompt opens with its own control tokens, so nothing may be added here.
        return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]

    def _eos_tokens(self) -> set[int]:
        """The ids that end a turn, from the config and the tokenizer's own end-of-text.

        The config's ``eos_token_id`` is a tuple on this checkpoint because the release ends a
        turn with either of two control tokens, and the tokenizer's ``eos_token_id`` is the same
        id again; both are read because a run that stopped on one of them and not the other would
        run every answer to its budget. There is no default and no fallback: a checkpoint that
        names neither is one whose answers cannot end.
        """
        found: set[int] = set()
        layer = getattr(self._checkpoint, "layer", None)
        ids = getattr(layer, "eos_token_id", None) if layer is not None else None
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

    def _budget(self, prompt_ids: Sequence[int], request: GenerationRequest) -> int:
        params = request.sampling_params
        room = self._max_seq_len - len(prompt_ids)
        if room < 1:
            raise ConfigurationError(
                f"a prompt of {len(prompt_ids)} tokens fills the {self._max_seq_len} positions "
                f"this run's cache was sized at; raise --max-model-len and restart"
            )
        return int(params.token_budget(room))

    def _step_sync(
        self, request_id: str, local: Callable[[], bool] | None = None
    ) -> Callable[[], bool]:
        """A per-step "should this loop stop", agreed on by every rank.

        Rank 0 reads its own cancel flag and broadcasts it; every other rank reads what it was
        sent. That message is what makes a client's disconnect reach a loop whose every other line
        is deterministic and local -- and it is *this* collective rather than a local flag because
        a rank that left on its own would leave three peers inside a layer's all-reduce.

        `local` is a condition only rank 0 can evaluate -- the stream's stop-string check -- and it
        is folded into the same flag for exactly that reason. A stop string is found on one rank
        and nowhere else, so a rank 0 that unwound on its own to report it would leave the other
        three in a layer; asking here instead makes the answer stop one step later, at a boundary
        every rank arrives at together.
        """
        if self._world <= 1:
            def alone() -> bool:
                return self._is_cancelled(request_id) or (local is not None and local())

            return alone

        import torch
        import torch.distributed as dist

        flag = torch.zeros(1, dtype=torch.int32, device=self._device)

        def agreed() -> bool:
            if self._rank == 0:
                stop = self._is_cancelled(request_id) or (local is not None and local())
                flag.fill_(1 if stop else 0)
            dist.broadcast(flag, src=0)
            return bool(int(flag.item()))

        return agreed

    # ------------------------------------------------------------------ the shared scheduler

    def _runtime_spec(self) -> RuntimeSpec:
        return RuntimeSpec(
            name=self.name,
            start=self._start_runtime,
            eos_tokens=self._eos_tokens,
            max_context=self._max_seq_len,
            wants_request=True,
            device=self._runtime_device,
        )

    def _runtime_device(self) -> int:
        """The card this rank drives, read the way the model's own card is chosen.

        `_init_distributed` is what picks it, and it runs on this thread inside `_ensure_loaded` --
        the same call that builds the model and the cache -- so by the time a forward is about to
        happen this answers. Before that it falls back to the launcher's option, and the bridge's
        bind is a no-op rather than a wrong card: the ep group binds its own rank's card itself,
        which is also why nothing here has to.
        """
        if self._device is not None:
            return device_index(self._device)
        card = self._card()
        return card if card is not None else -1

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

        This is `_loop` with two of its arguments supplied differently, and it is `_loop` on purpose
        rather than a second call to `generate`: every routed layer closes with an `all_reduce`, so
        a rank 0 that ran a request the workers were not told about would not be idle, it would be
        at a different collective. Going through the same method is what keeps the payload, the
        broadcast and the per-step agreement in one place instead of two that agree by inspection.

        The budget is the scheduler's, which derived it from this request with the same rule the
        serial path uses. Everything else -- the sampler, the seed, the prompt -- is read off
        `context`, which *is* the request this row was submitted for, so the two routes render the
        same thing by construction.

        And the step boundary is the scheduler's. `_step_sync` is what tells rank 0's park to the
        other ranks, so they take that step together or none of them does, and a client that
        disconnected -- which arrives on the HTTP thread under the request's own id, not the
        scheduler's -- reaches the loop at the same seam.
        """
        self._loop(
            list(prompt_ids),
            int(sampling.max_new_tokens),
            context,
            on_token=on_token,
            marks={"started": time.perf_counter()},
            stop=on_step,
        )

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        self._begin_request(request.request_id)
        try:
            prompt_ids = self._tokenize(request)
            budget = self._budget(prompt_ids, request)
            self._check_cancelled(request.request_id)
            with self._request_lock:
                self._check_cancelled(request.request_id)
                marks = {"started": time.perf_counter()}
                generation = self._loop(
                    prompt_ids, budget, request, on_token=None, marks=marks
                )
            return self._result(request, prompt_ids, generation, marks)
        finally:
            self._clear_request(request.request_id)

    def _loop(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request: GenerationRequest,
        *,
        on_token: Callable[[int], None] | None,
        marks: dict[str, float],
        stop: Callable[[], bool] | None = None,
    ) -> Any:
        from src.models.mimo_v2.generate import generate

        self._ensure_loaded()
        self._ensure_prefix_cache()
        params = request.sampling_params
        self._dispatch(request, prompt_ids, budget)
        try:
            generation = generate(
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
                on_step=self._step_sync(request.request_id, stop),
            )
        finally:
            # Under the request lock, like the run itself, so the counters are the state of the store
            # between requests rather than of one mid-prefill.
            self._publish_cache_metrics()
        return generation

    def _dispatch(
        self, request: GenerationRequest, prompt_ids: Sequence[int], budget: int
    ) -> None:
        """Hand every other rank the request, so that the work below is mirrored and not soloed.

        **The collective is what makes this mandatory rather than tidy.** Every routed layer closes
        with an ``all_reduce``, so a rank that is not running the same request is not idle -- it is
        at a *different* collective, and NCCL answers a mismatch by hanging both sides. Rank 0
        therefore may not enter a generation the workers have not been told about, and the
        broadcast below is the only thing that tells them: the worker loop blocks on this exact
        call, one payload a request, and a request that never arrives is a rank 0 that deadlocks
        on its own first expert layer rather than a rank 0 that quietly returns a wrong answer.

        The payload is the whole request because the workers have to reproduce it exactly: the
        same prompt ids, the same budget, the same sampler and the same seed. They are not given
        the prompt *text* -- tokenization is rank 0's, and a rank that tokenized it itself would be
        a second renderer that could disagree. Greedy is deterministic and the logits are the same
        sum on every rank, so the tokens each rank draws are the same tokens; only rank 0's leave.
        """
        if self._world <= 1 or self._rank != 0:
            return
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        params = request.sampling_params
        payload = {
            "op": "generate",
            "request_id": request.request_id,
            "prompt_ids": [int(token) for token in prompt_ids],
            "max_new_tokens": int(budget),
            "temperature": float(params.temperature),
            "top_k": None if params.top_k is None else int(params.top_k),
            "top_p": params.top_p,
            "seed": params.seed,
        }
        torch.distributed.broadcast_object_list([payload], src=0)

    def _decode(self, token_ids: Sequence[int], skip_special_tokens: bool | None = None) -> str:
        if self._tokenizer is None:
            return ""
        skip = True if skip_special_tokens is None else bool(skip_special_tokens)
        try:
            return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip)
        except Exception:  # pragma: no cover - a tokenizer that cannot decode one token
            return ""

    def _result(
        self,
        request: GenerationRequest,
        prompt_ids: Sequence[int],
        generation: Any,
        marks: Mapping[str, float],
        *,
        stopped: str | None = None,
    ) -> GenerationResult:
        text = self._decode(generation.tokens)
        finish = {
            "eos": "stop",
            "length": "length",
            "stop": "stop",
            "cancel": "cancelled",
        }.get(generation.stopped if stopped is None else stopped, "stop")
        now = time.perf_counter()
        return GenerationResult(
            request_id=request.request_id,
            token_ids=list(generation.tokens),
            text=text,
            finish_reason=finish,
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(generation.tokens),
                cached_tokens=int(getattr(generation, "cached_tokens", 0)),
            ),
            timings=TimingMetrics(
                prefill_seconds=generation.prefill_seconds,
                decode_seconds=generation.decode_seconds,
                total_seconds=now - marks.get("started", now),
                ttft_seconds=generation.ttft_seconds,
                tpot_seconds=generation.step_seconds,
            ),
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        self._ensure_loaded()
        if self._batching():
            return self._generate_batched(requests)
        return [self._generate_one(request) for request in requests]

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield one event a token, off a worker thread, with the stop strings held back.

        The loop has to run somewhere and the events have to be yielded from here, and the two
        cannot be the same frame. A thread a request is also what lets a client's disconnect --
        which arrives on the HTTP thread as :meth:`cancel` -- reach the loop at all.
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
                        prompt_ids,
                        budget,
                        request,
                        on_token=streamer.accept,
                        marks=marks,
                        stop=streamer.reached,
                    )
                if generation.stopped == "cancel" and not streamer.hit:
                    events.put(
                        TokenEvent(request_id=request.request_id, finish_reason="cancelled")
                    )
                    return
                # Whatever the holdback was holding: the answer is over, so the tail is either
                # text the model really wrote or the start of a stop string that never finished,
                # and only the first of those may be sent.
                streamer.flush(generation.tokens)
                box["result"] = self._result(
                    request,
                    prompt_ids,
                    generation,
                    marks,
                    stopped="stop" if streamer.hit else None,
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

        thread = threading.Thread(target=worker, name=f"mimo-{request.request_id}", daemon=True)
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

    # -------------------------------------------------------------------- ranks

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Load, announce, then serve rank 0's requests until it says to stop."""
        self._ensure_open()
        self._init_distributed()
        if self._world <= 1:
            raise UnsupportedFeatureError(
                "run_worker serves rank 0's requests over a process group; a single-rank MiMo run "
                "serves them through the HTTP server instead"
            )
        if self._rank == 0:
            raise UnsupportedFeatureError("run_worker must not be called on rank 0")
        self._ensure_loaded()
        if on_ready is not None:
            on_ready()
        import torch.distributed as dist

        while not self._closed:
            payload: list[Any] = [None]
            dist.broadcast_object_list(payload, src=0)
            if not isinstance(payload[0], Mapping) or payload[0].get("op") == "shutdown":
                break
            if payload[0].get("op") != "generate":
                continue
            try:
                self._run_payload(payload[0])
            except RequestCancelledError:
                # Every rank unwinds together, so a cancelled request is not a desynchronized group.
                pass
        dist.barrier()

    def _run_payload(self, payload: Mapping[str, Any]) -> None:
        """Run the same loop rank 0 is running, for the collective's sake and not for its answer.

        The store is this rank's own, built the same way from the same options and fed the same
        requests, which is what makes the two loops agree about *how much* of the prompt to forward.
        They have to: a rank that resumed a prompt its peers prefilled from zero would be inside a
        different set of layers' collectives, and the two sides of an `all_reduce` that disagree
        about when they enter it is not a wrong answer but a hang. Nothing is broadcast to enforce
        it, because the store is a function of the request and the options and of nothing local.
        """
        from src.models.mimo_v2.generate import generate

        self._ensure_prefix_cache()
        prompt_ids = [int(token) for token in payload["prompt_ids"]]
        try:
            generate(
                self._model,
                prompt_ids,
                max_new_tokens=int(payload["max_new_tokens"]),
                temperature=float(payload["temperature"]),
                top_k=payload["top_k"],
                top_p=payload["top_p"],
                seed=payload["seed"],
                eos_token_id=self._eos_tokens(),
                chunk=self._options.prefill_chunk,
                cache=self._cache,
                prefix_cache=self._prefix_cache,
                on_step=self._step_sync(str(payload["request_id"])),
            )
        finally:
            self._publish_cache_metrics()

    def close(self) -> None:
        already_closed = self._closed
        if not already_closed and self._world > 1 and self._rank == 0:
            try:
                import torch.distributed as dist

                dist.broadcast_object_list([{"op": "shutdown"}], src=0)
            except Exception:
                # A peer that already left is not this rank's problem, and close() must not raise.
                pass
        # Before the base class tears the rank down: the scheduler's thread runs this runtime's
        # generation, and a step in flight is a collective the other ranks are already inside.
        scheduler = self._scheduler
        self._scheduler = None
        if scheduler is not None:
            scheduler.stop()
        super().close()


__all__ = [
    "MimoBackend",
    "DEFAULT_EXPERT_ROWS",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_PREFILL_CHUNK",
    "DEFAULT_PREFIX_CACHE_BYTES",
    "DEFAULT_PREFIX_CACHE_HEAD_TOKENS",
]
