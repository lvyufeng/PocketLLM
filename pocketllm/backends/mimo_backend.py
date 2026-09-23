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
from dataclasses import dataclass
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

from .base import BackendBase, settled_text

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

_KNOWN_OPTIONS = frozenset({
    "chunk_rows",
    "deal",
    "device",
    "expert_deal",
    "expert_rows",
    "pin",
    "prefill_chunk",
    "slots",
})

_IGNORED_OPTIONS = frozenset({"engine_kind", "routed_experts_device", "pd_mode", "nccl_id_path"})
"""``backend_options`` keys a launch always carries that this backend has no use for.

The CLI fills in ``engine_kind``, ``routed_experts_device`` and ``pd_mode`` on every serve command
and the supervisor adds ``nccl_id_path`` for a sharded one, so all four arrive whether or not the
selected adapter reads them; accepting them is what makes a MiMo launch the same command line as a
native one. Every *other* unknown key is a refusal, because a tuning option that silently does
nothing is how a run ends up measured on the wrong lever.
"""


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} is a flag and {value!r} is not one")


@dataclass(slots=True)
class _Options:
    """The launcher's levers, resolved once at construction.

    Every one of these changes what the run does, which is why an unknown key is refused rather
    than ignored: ``chunk_rows`` is the arena a card pays for, ``deal`` is which deal the experts
    are divided by, ``prefill_chunk`` is the width a prompt goes through at, and ``slots`` is how
    many calls the pipeline keeps in flight.
    """

    chunk_rows: int | None = DEFAULT_EXPERT_ROWS
    deal: str | None = "sorted"
    device: str | None = None
    pin: bool = True
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    slots: int = DEFAULT_EXPERT_SLOTS

    @classmethod
    def from_args(cls, args: Any) -> "_Options":
        values = dict(getattr(args, "backend_options", None) or {})
        for name in _IGNORED_OPTIONS:
            values.pop(name, None)
        unknown = sorted(set(values) - _KNOWN_OPTIONS)
        if unknown:
            raise ConfigurationError(
                f"backend='mimo' has no option {unknown[0]!r}; it knows "
                f"{', '.join(sorted(_KNOWN_OPTIONS))}"
            )
        rows = values.pop("chunk_rows", values.pop("expert_rows", DEFAULT_EXPERT_ROWS))
        deal = values.pop("deal", values.pop("expert_deal", "sorted"))
        prefill = int(values.pop("prefill_chunk", DEFAULT_PREFILL_CHUNK))
        slots = int(values.pop("slots", DEFAULT_EXPERT_SLOTS))
        pin = _flag(values.pop("pin", True), "pin")
        if rows is not None and int(rows) < 0:
            raise ConfigurationError(f"chunk_rows is an expert count and {rows!r} is not one")
        if prefill < 1:
            raise ConfigurationError(f"prefill_chunk is a token count and {prefill} is not one")
        if slots < 1:
            raise ConfigurationError(f"slots is a slot count and {slots} is not one")
        if deal is not None and str(deal) not in {"id", "sorted"}:
            raise ConfigurationError(f"deal is `id` or `sorted`, got {deal!r}")
        return cls(
            chunk_rows=None if rows is None else int(rows),
            deal=None if deal is None else str(deal),
            device=values.pop("device", None),
            pin=pin,
            prefill_chunk=prefill,
            slots=slots,
        )


class MimoBackend(BackendBase):
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
        self._ep: Any = None
        self._device: Any = None
        self._world = 1
        self._rank = 0
        self._request_lock = threading.RLock()
        self._distributed = False
        self._details: dict[str, Any] = {}

    # ------------------------------------------------------------------ lifecycle

    def prepare(self) -> None:
        self._ensure_open()
        self._ensure_loaded()

    def _init_distributed(self) -> None:
        """Join the group the launcher set, if there is one.

        Read from the environment and not from the arguments, because a ``torchrun`` launch sets
        only the environment and a supervised one sets both: two sources that could disagree are
        one source too many, and the group is the thing that has to be right.
        """
        if self._distributed:
            return
        from src.models.mimo_v2.ep import EpGroup

        self._ep = EpGroup.from_env(device=self._options.device)
        self._world, self._rank = self._ep.world, self._ep.rank
        self._device = self._ep.device
        self._distributed = True

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
            deal=options.deal,
            chunk_rows=options.chunk_rows,
            pin=options.pin,
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
            if module.chunk_rows is None:
                return (
                    f"one draw a step under the `{module.deal}` deal, "
                    f"{module.arena_bytes / 2**20:.0f} MiB arena a slot"
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

    # -------------------------------------------------------------------- contract

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            models=("mimo_v2", "mimo_v2_6"),
            model_formats=("safetensors",),
            devices=("cuda",),
            supports_batch=False,
            supports_streaming=True,
            supports_cancellation=True,
            supports_logprobs=False,
            supports_prefix_caching=False,
            details=dict(self._details),
        )

    def metrics(self) -> dict[str, float]:
        """What one MiMo engine holds between requests, for ``/metrics``.

        The expert bank's own counters and the draw's per-token weight are properties of the run
        rather than of a request: what a step costs is a function of the deal and the arena, and a
        client reading a rate wants to know which one it was measured under.
        """
        experts = getattr(self._model, "experts", None)
        if experts is None:
            return {}
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
        params = request.sampling_params
        self._dispatch(request, prompt_ids, budget)
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
            on_token=None if on_token is None else (lambda token, _logits: on_token(token)),
            on_step=self._step_sync(request.request_id, stop),
        )

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
                prompt_tokens=len(prompt_ids), completion_tokens=len(generation.tokens)
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
                streamer = _Streamer(
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
        """Run the same loop rank 0 is running, for the collective's sake and not for its answer."""
        from src.models.mimo_v2.generate import generate

        prompt_ids = [int(token) for token in payload["prompt_ids"]]
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
            on_step=self._step_sync(str(payload["request_id"])),
        )

    def close(self) -> None:
        already_closed = self._closed
        if not already_closed and self._world > 1 and self._rank == 0:
            try:
                import torch.distributed as dist

                dist.broadcast_object_list([{"op": "shutdown"}], src=0)
            except Exception:
                # A peer that already left is not this rank's problem, and close() must not raise.
                pass
        super().close()


class _Streamer:
    """The streaming half of a request: what has been sent, and what may be sent next.

    Held in an object rather than in a closure because two of these fields (`sent` and the id list)
    are written from the token callback and read from it again, and a closure that assigns to a name
    it also reads is a local variable with an unbound first read.

    The text is decoded from the run's tokens every token rather than incrementally, because a
    byte-level tokenizer cannot decode a token in isolation: what it has is a byte stream, and one
    token's bytes may end in the middle of a character. The whole decode is the only thing that
    knows; ``settled_text`` is what keeps the half-character from being sent.

    A stop string is *recorded* here and not raised. The loop it interrupts is a four-rank lockstep,
    and only rank 0 has a stop string to find, so leaving on the spot would leave three peers inside
    a layer's all-reduce. ``reached`` is handed to the per-step sync instead, which is a collective:
    every rank then leaves at the same token boundary, one step after the marker was found. What the
    client sees is the same text either way -- the cut has already been emitted by then.
    """

    def __init__(self, *, request_id: str, stops: Sequence[str], events: Any, decode: Any) -> None:
        self.request_id = request_id
        self.stops = tuple(stops)
        self.events = events
        self.decode = decode
        self.ids: list[int] = []
        self.text = ""
        self.sent = ""
        self.token = 0
        self.hit = False

    def reached(self) -> bool:
        """Whether a stop string has been found, which is a local fact and only rank 0's."""
        return self.hit

    def accept(self, token: int) -> None:
        """One token from the loop: send everything that is now settled, or end on a stop string."""
        if self.hit:
            return
        self.ids.append(int(token))
        self.token = int(token)
        self.text = settled_text(self.decode(self.ids))
        cut = self._cut(self.text)
        if cut >= 0:
            self._emit(self.text[:cut], token=self.token)
            self.hit = True
            return
        self._emit(_hold_back(self.text, self.stops), token=self.token)

    def flush(self, tokens: Sequence[int]) -> None:
        """The answer is over: send the tail unless it is the start of a stop string.

        Nothing was sent for it until now precisely because it could still have turned out to be a
        marker. A stop string that never completed is text the model really wrote and goes out; one
        that did complete was already cut at its first character.
        """
        text = settled_text(self.decode(list(tokens)))
        cut = self._cut(text)
        self._emit(text[:cut] if cut >= 0 else text)

    def _cut(self, text: str) -> int:
        return min((text.find(stop) for stop in self.stops if stop in text), default=-1)

    def _emit(self, target: str, *, token: int | None = None) -> None:
        """Send what `target` adds to what has already gone out.

        The id travels with the text because the server's own counters and its inter-token
        latency are keyed off an event that carries one: a stream of text-only events is a
        response a client renders and a metrics scrape reads as zero tokens and no TTFT.
        """
        if len(target) > len(self.sent):
            self.events.put(
                TokenEvent(
                    request_id=self.request_id,
                    token_id=token,
                    text=target[len(self.sent) :],
                )
            )
            self.sent = target


def _hold_back(text: str, stops: Sequence[str]) -> str:
    """``text`` without a tail that is a *partial* match of a stop string.

    A stream cannot take a character back, so a tail that could still turn out to be the start of a
    marker waits for the token that decides it. A whole stop string at the end is not held: the
    caller has already cut the answer at it, and holding one here would delay text that is not
    going to be sent again.
    """
    keep = 0
    for stop in stops:
        for size in range(1, min(len(stop) - 1, len(text)) + 1):
            if text.endswith(stop[:size]):
                keep = max(keep, size)
    return text[: len(text) - keep] if keep else text


__all__ = ["MimoBackend", "DEFAULT_EXPERT_ROWS", "DEFAULT_MAX_SEQ_LEN", "DEFAULT_PREFILL_CHUNK"]
