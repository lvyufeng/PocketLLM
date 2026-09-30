"""Shared helpers for backend adapters."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    RequestCancelledError,
    TensorParallelSupervisorError,
    TimingMetrics,
    TokenEvent,
    Usage,
)
from pocketllm.choices import CHOICE_MARK
from pocketllm.protocol.contract import CHAT, FieldRefusal


#: What a byte-level tokenizer's decode puts where a token ended inside a character.
_REPLACEMENT = "�"

#: How each vocabulary this tree produces spells "why the loop ended", folded onto the API's.
#:
#: Six words for four outcomes, and the two groups do not overlap. A runtime's own loop reports
#: ``eos`` / ``length`` / ``cancel`` (``src/models/mimo_v2/generate.py:160``) or ``eos`` /
#: ``length`` / ``max_seq_len`` (``src/models/deepseek_v4_1/generate.py:273``), and
#: ``BatchScheduler`` reports ``stop`` / ``length`` / ``cancelled``
#: (``batch_scheduler.cpp:854``). So a mapping written for one family is silently wrong for the
#: other's spelling -- which is what this table exists to make impossible: it covers every word
#: either route emits, and its answer does not depend on which one produced the word.
#:
#: ``error`` is deliberately absent. It is not a stop at all: the scheduler keeps it in a separate
#: field precisely so that callers report it as a failure rather than as a finish reason
#: (``batch_scheduler.hpp:111``), and the scheduler host raises on that field before any result is
#: built (``pocketllm/backends/runtime_engine.py:789``). Mapping it here would give it a place to be
#: quietly absorbed.
_STOPPED_FINISH_REASONS = {
    "eos": "stop",
    "stop": "stop",
    "length": "length",
    "max_seq_len": "length",
    "cancel": "cancelled",
    "cancelled": "cancelled",
}


def settled_text(decoded: str) -> str:
    r"""``decoded`` without the trailing run of replacement characters.

    A byte-level tokenizer decodes ids to bytes and then decodes *those* to text with
    ``errors="replace"``, so a token that ends in the middle of a multi-byte character renders as
    U+FFFD until the token that finishes it arrives: the emoji in ``你好！😊`` is one token and the
    character before it is not, and ``decode([30594, 1175, 28927])`` is ``你好！�``. A stream
    that sent that has sent a character the model never produced, and a stream cannot take a
    character back. Holding the tail until it settles costs at most one token of latency and sends
    what the unstreamed decode sends.

    Only a *trailing* run is held back. A replacement character anywhere else is real -- the bytes
    there really were invalid -- and the unstreamed decode has it in the same place, so a stream
    that dropped it would disagree with the answer it is a stream of.
    """
    return decoded.rstrip(_REPLACEMENT)


class BackendBase:
    """Small common implementation for lifecycle and cancellation bookkeeping.

    The lock is intentionally at the backend boundary.  Current native engines
    own one mutable KV-cache transaction, so concurrent calls must serialize
    until a request-aware cache scheduler is implemented.
    """

    def __init__(self) -> None:
        self._closed = False
        self._ready = False
        self._state_lock = threading.RLock()
        self._cancelled: set[str] = set()
        self._active_requests: set[str] = set()
        #: What :meth:`_publish_cache_metrics` last read, in the exporter's spelling, and what
        #: :meth:`metrics` merges into its answer. Rebound whole rather than mutated: the HTTP thread
        #: reads it while a request may be mid-prefill, and a `stats()` walk running concurrently with
        #: a store is a `dictionary changed size`. A scrape therefore reads one request's worth of
        #: state or the request before it, never a half-updated dict. Empty until a runtime with a
        #: prefix store publishes one, and a runtime without a store never does -- which
        #: :meth:`metrics` reads as "no backend-owned series".
        self._cache_metrics: dict[str, float] = {}

    @property
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    def health(self) -> HealthStatus:
        with self._state_lock:
            closed = self._closed
            ready = self._ready and not closed
        return HealthStatus(
            status="stopped" if closed else "ready" if ready else "loading",
            backend=self.capabilities.name,
            ready=ready,
        )

    def cancel(self, request_id: str) -> bool:
        """Cancel ``request_id``, and every choice of it when it is a fan-out.

        A request asking for several choices is one request to the client and n engine requests
        internally, and the client only ever learns the outer id. So a cancellation named for the
        outer id has to reach the choices as well -- otherwise ``DELETE /v1/.../{id}`` returns
        success while n-1 generations keep the engine busy, which is the failure mode the endpoint
        exists to prevent. The shape of a choice's id is what makes that possible; see
        :data:`pocketllm.choices.CHOICE_MARK`.

        Nothing here requires the fan-out to have happened: at n == 1 the child set is just
        ``{request_id}``, so this is the same call it always was.
        """
        with self._state_lock:
            request_id = str(request_id)
            if self._closed:
                return False
            targets = {
                name
                for name in self._active_requests
                if name == request_id or name.startswith(request_id + CHOICE_MARK)
            }
            if not targets:
                return False
            self._cancelled |= targets
            return True

    def _begin_request(self, request_id: str) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("backend is closed")
            self._active_requests.add(str(request_id))

    def _is_cancelled(self, request_id: str) -> bool:
        with self._state_lock:
            return request_id in self._cancelled

    def _clear_request(self, request_id: str) -> None:
        with self._state_lock:
            request_id = str(request_id)
            self._active_requests.discard(request_id)
            self._cancelled.discard(request_id)

    def active_request_count(self) -> int:
        with self._state_lock:
            return len(self._active_requests)

    def audit_request(self, body: Mapping[str, Any], *, endpoint: str = CHAT) -> FieldRefusal | None:
        """The first field in ``body`` this backend cannot serve, or ``None``.

        A field the runtime will not apply, sent with a value that would have changed the answer,
        has to be refused by name: answering it with a 200 and text generated as if the field were
        absent is a response the caller has no way to tell from the one it asked for. The refusal
        says what arrived, what the server does instead, and what the caller can do -- all four
        parts, because a caller who cannot tell "dropped" from "ignored" from "wrong spelling" has
        to read the source to find out.

        Shape is not this method's business. It is checked for every request before this runs,
        because a ``stop`` of ``5`` is wrong on every runtime; what is left here is the question
        only the backend can answer.

        ``None`` by default, and deliberately: a runtime that has not stated which fields it applies
        has no claim to hold a request to, and refusing every field in the table for every adapter
        that has not been audited would replace a silent ignore with a blanket refusal. The C++
        adapter is the one that declares -- it is the runtime whose own front end is being retired,
        so its contract is the one being carried over.
        """
        return None

    def metrics(self) -> dict[str, float]:
        """Metric values the engine owns, as ``name -> value``, for the server's exporter.

        Request-scoped numbers reach ``/metrics`` through :class:`~pocketllm.api.GenerationResult`,
        which the HTTP layer already reads. What does not is anything the engine holds *between*
        requests -- a prompt cache's occupancy and its running hit count, say -- because no request
        owns a share of it. This is that channel, and it is deliberately a plain mapping: an adapter
        publishes values, the server decides how to export them, and no backend imports the
        exporter. Absolute values, not deltas: the backend is the one keeping the total.

        Empty by default, which is what an engine with nothing to add returns and what the server
        reads as "no backend-owned series". Names must be valid Prometheus metric suffixes and are
        flat, matching the rest of this server's spelling.
        """
        return {}

    def _check_cancelled(self, request_id: str) -> None:
        if self._is_cancelled(request_id):
            raise RequestCancelledError(f"request {request_id} was cancelled")

    def _publish_cache_metrics(self) -> None:
        """Hand the prefix store's counters to the exporter, in the names ``/metrics`` reads.

        The store owns these numbers and no request owns a share of them -- hits and misses are
        cumulative over the process -- so they travel as whole values rather than as per-request
        deltas the HTTP layer would have to accumulate, which is what :meth:`metrics` is for.

        The names are fixed here rather than per adapter. They were three copies of the same nine
        keys, and a series that a scrape sees only when one runtime happens to be serving is worse
        than one it sees always: a dashboard cannot compare two adapters whose metric names were
        typed three times. A runtime with no prefix store publishes nothing, which
        :meth:`metrics` reads as "no backend-owned series" rather than as zero.
        """
        cache = getattr(self, "_prefix_cache", None)
        if cache is None:
            return
        stats = cache.stats()
        self._cache_metrics = {
            "prefix_cache_hits_total": float(stats["hits"]),
            "prefix_cache_misses_total": float(stats["misses"]),
            "prefix_cache_reused_tokens_total": float(stats["reused_tokens"]),
            "prefix_cache_entries": float(stats["entries"]),
            "prefix_cache_bytes": float(stats["bytes"]),
            "prefix_cache_budget_bytes": float(stats["budget_bytes"]),
        }

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._cancelled.clear()
        self._release_supervisor()

    def _release_supervisor(self) -> None:
        """Stop the tensor-parallel ranks this backend's construction started.

        :func:`~pocketllm.backends.factory.create_backend` attaches the supervisor that
        launched ranks 1..N-1 to the rank-0 backend it returns, and nothing else holds a
        reference to those processes, so the backend that owns the engine owns the ranks.
        ``cleanup`` is the supervisor's only teardown; there is no ``stop``.  A backend
        built without the factory -- every single-rank run, and every injected test
        double -- simply has no such attribute, and that is not an error.

        The reference is dropped before the call so a second ``close`` cannot clean up a
        supervisor that has already been torn down, and a failure to tear down must not
        turn a close into a raise.
        """
        supervisor = getattr(self, "_supervisor", None)
        if supervisor is None:
            return
        self._supervisor = None
        try:
            supervisor.cleanup()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def prepare(self) -> None:
        """Eagerly initialize a backend before a supervised rank announces readiness."""
        self._ensure_open()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        raise NotImplementedError

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        raise NotImplementedError

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Enter a backend-specific worker loop for supervised TP ranks > 0.

        This method is only called on nonzero ranks when the CLI uses its
        built-in process supervisor.  Backends without native TP worker support
        must raise UnsupportedFeatureError rather than silently returning.

        ``on_ready`` is called exactly once after the worker has initialized and
        is ready to participate in collectives, immediately before entering the
        blocking worker loop.

        The Python runtimes do not implement this; ``RankedWorker`` in
        :mod:`pocketllm.backends.runtime_engine` does, and this is the body every adapter
        outside that family inherits -- ``torch`` and ``cpp`` have their own worker entries and
        a backend that has none has to say so rather than return.
        """
        raise TensorParallelSupervisorError(
            "backend does not implement a supervised TP worker entry point"
        )

    @staticmethod
    def _metadata_copy(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}


class RuntimeAdapter(BackendBase):
    """The request-lifecycle half that the torch-runtime adapters share.

    The adapters in this family -- ``v41``, ``mimo``, ``xing4``, ``torch`` -- serve a request the
    same way: turn a prompt into ids, resolve a budget against the context this run was configured
    with, loop a runtime, and read the finished generation back into a :class:`GenerationResult`.
    They differ in what the runtime *is*, not in that shape.

    What is here is the part of the shape whose body says nothing about its runtime once the
    runtime has a name.  Between ``mimo`` and ``xing4``, ``_encode_chat`` and ``_result`` were
    byte-identical and ``prepare``, ``_tokenize``, ``_budget`` and ``_decode`` differed by the
    runtime's name inside one error message -- and two copies of a body like that are two places
    for the next fix to miss.

    Which adapters are on this base is a fact about how far the convergence has got, not about
    which ones the body fits: ``v41`` and ``torch`` keep their own copies until the differences
    they carry are resolved one at a time.

    A subclass supplies its runtime's name through :attr:`_RUNTIME_LABEL`, and overrides whatever
    it does differently: ``_encode_chat`` where the checkpoint's format is not a jinja template,
    ``_budget`` and ``_tokenize`` where the arithmetic is not this one, ``_result`` where the answer
    has a structure to read, ``_loop`` for the runtime's own generation call, ``_batch_dispatch``
    where a scheduler serves the request instead.  What a subclass is left keeping is exactly that
    list, which is the point of the base -- the parts that are left are the parts that differ.

    The only state read here is ``_tokenizer`` and ``_max_seq_len``, which every adapter in this
    family has.
    """

    #: How this adapter's own messages spell its runtime, as this family spells it in "the MiMo
    #: tokenizer is not loaded".  Read by :meth:`_tokenize` alone: nothing else here needs to know
    #: which runtime it is.
    _RUNTIME_LABEL = ""

    def prepare(self) -> None:
        """Open and load, which is every path that has to have happened before a request.

        Deliberately *not* here: building a prefix store.  Three of the adapters have one and they
        do not agree on where it belongs -- ``v41`` builds it inside ``_ensure_loaded``, under the
        load lock, so that the injected loader paths reach it too; ``mimo`` and ``xing4`` build it
        from this method.  That is a decision about the store, not about preparing, so each adapter
        that has one says so itself.
        """
        self._ensure_open()
        self._ensure_loaded()

    def _batch_dispatch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[GenerationResult] | None:
        """Serve these requests through a scheduler, or ``None`` when there is not one to serve them.

        ``None`` is the default because the scheduler is a property of a runtime that has one, and
        the answer arrives as a value rather than as a second predicate so that the two decisions --
        *is there a scheduler* and *serve the batch* -- cannot be read apart. An adapter whose
        scheduler lives on another base answers by building it there: ``SchedulerHost`` returns
        ``self._generate_batched(requests)`` on exactly the condition ``generate`` would have tested,
        and the search for the first of two superclasses that has one stays there instead of here.
        """
        return None

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Answer every request, through a scheduler if one serves this runtime and serially if not.

        One body rather than a copy per adapter, because the choice and not the answering is what
        this method is: the loop below is the same loop that stood in ``v41``, ``mimo`` and ``xing4``
        -- a scheduler first, then one request at a time -- and the differences those three carried
        are in :meth:`_run_serial` instead.
        """
        self._ensure_loaded()
        batched = self._batch_dispatch(requests)
        if batched is not None:
            return batched
        return [self._generate_one(request) for request in requests]

    def _generate_one(self, request: GenerationRequest) -> GenerationResult:
        """One request through this adapter's own loop, with its lifetime bracket around it.

        The bracket is the whole of this method and is the same for every adapter that has run a
        request: the request becomes visible to :meth:`cancel` before anything slow happens to it,
        the cancellation is checked both before and *inside* the request lock -- a client that
        disconnected while this thread waited for the lock is otherwise not noticed until it has
        been let in -- and the request's id comes out of the cancelled set when it is over, so a
        later request reusing the id is not born cancelled.

        What sits between the two checks is the runtime's, and it is one call: whatever a request
        has to become for that runtime, derived once, followed by the loop under the lock. Reading the
        loop's answer is the same call's other half and moves with it -- see :meth:`_serial_result`.
        """
        self._begin_request(request.request_id)
        try:
            prompt_ids = self._tokenize(request)
            prepared = self._prepare_serial(request, prompt_ids)
            self._check_cancelled(request.request_id)
            with self._request_lock:
                self._check_cancelled(request.request_id)
                generation = self._run_serial(prepared, request)
            return self._serial_result(request, prompt_ids, generation)
        finally:
            self._clear_request(request.request_id)

    def _serial_result(
        self, request: GenerationRequest, prompt_ids: Sequence[int], generation: Any
    ) -> GenerationResult:
        """The loop's answer as the API's result, on the serial path.

        The shared :meth:`_result` by default, which is where every adapter but one wants to end up.
        The exception is the adapter whose loop hands back something that is not this family's
        generation at all -- ``v41``, whose answer splits into reasoning, prose and tool calls and
        whose builder therefore takes the pieces rather than the object -- and it is a hook rather
        than an override of :meth:`_generate_one` because *reading the answer* is what differs there,
        while the bracket around it is not.
        """
        return self._result(request, prompt_ids, generation, {"started": time.perf_counter()})

    def _prepare_serial(self, request: GenerationRequest, prompt_ids: Sequence[int]) -> Any:
        """What the loop is handed for a serial request, derived before the request lock is taken.

        The prompt ids, by default: an adapter whose loop takes ids passes :meth:`_budget` next to
        them, and ``v41`` -- whose loop takes a broadcast payload -- builds that here instead. Both
        have to happen before the lock for the same reason: the budget refuses a request that does
        not fit, and a refusal that arrived after the lock was taken would have held it for a request
        that was never going to run.
        """
        return prompt_ids

    def _run_serial(self, prepared: Any, request: GenerationRequest) -> Any:
        """Run this runtime's loop for one serial request, and hand back what it produced.

        The call this family makes for a request that has the runtime to itself, and the difference
        between this and :meth:`_run_loop` is the whole of the streamed path: that one hands the loop
        a streamer, this one hands it nothing to talk to. An adapter whose loop takes a budget beside
        its prompt derives it here, and ``v41`` -- whose loop takes a broadcast payload -- writes
        this out because it is the one runtime whose loop is reached that way.
        """
        return self._loop(
            prepared,
            self._budget(prepared, request),
            request,
            on_token=None,
            stop=lambda: self._is_cancelled(request.request_id),
        )

    def _skip_special_tokens(self) -> bool:
        """Whether a decode leaves the checkpoint's control tokens out, when the caller did not say.

        ``True``, which is what every adapter in this family wanted before ``v41`` grew a CLI option
        for it.  Overridden by an adapter that lets a run configure it.
        """
        return True

    def _tokenize(self, request: GenerationRequest) -> list[int]:
        if request.prompt_tokens is not None:
            # A copy, and only a copy: the request already coerced these in its own
            # ``__post_init__``, so the only thing left to decide here is that the ids this
            # adapter reports back are not the caller's own list.
            return list(request.prompt_tokens)
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(f"the {self._RUNTIME_LABEL} tokenizer is not loaded")
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

    def _budget(self, prompt_ids: Sequence[int], request: GenerationRequest) -> int:
        """How many tokens this request may generate, from the request and this run's context.

        The number is derived and then checked, and the check is the second half of
        :meth:`SamplingParams.token_budget`'s contract: an explicit ``max_tokens`` is handed back
        unchanged by that method *because* the caller is the one that gets to refuse it. This is
        that refusal, and it is here rather than at the three call sites because a budget that came
        back from a context this run cannot hold is wrong wherever it was asked for.

        This is also what refuses a prompt that already fills the context: the derived budget is
        floored at one, so such a prompt arrives at the check as a request that does not fit rather
        than as a budget of zero.
        """
        params = request.sampling_params
        budget = int(params.token_budget(self._max_seq_len - len(prompt_ids)))
        wanted = len(prompt_ids) + budget
        if wanted <= self._max_seq_len:
            return budget
        raise ConfigurationError(
            f"this request needs {wanted} positions ({len(prompt_ids)} prompt tokens and "
            f"{budget} new), and the attention caches were sized at "
            f"{self._max_seq_len} at startup; raise --max-model-len and restart"
        )

    def _decode(self, token_ids: Sequence[int], skip_special_tokens: bool | None = None) -> str:
        """The text of ``token_ids``, or ``""`` when there is nothing that can read them.

        An absent argument means :meth:`_skip_special_tokens`, which is the usual answer and is a
        run-configurable one on the adapter that has a flag for it.

        Two things are absorbed rather than raised, and neither is this method's to report. A
        tokenizer with no ``skip_special_tokens`` keyword is a duck-typed one, and it is asked
        again without it: this repository hands ``_decode`` a real HF tokenizer in production and a
        three-line stand-in in tests, and the stand-in is the case that has no keyword. Anything
        else a tokenizer raises is dropped, because a request that answered with no text is still
        an answer, and raising here would lose the ids that did decode.
        """
        if self._tokenizer is None or not token_ids:
            return ""
        skip = self._skip_special_tokens() if skip_special_tokens is None else bool(skip_special_tokens)
        try:
            return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip)
        except TypeError:  # a tokenizer whose decode has no such keyword
            pass
        except Exception:  # pragma: no cover - a tokenizer that cannot decode these ids
            return ""
        try:
            return self._tokenizer.decode(list(token_ids))
        except Exception:  # pragma: no cover - the same, without the keyword to blame
            return ""

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield one event a token, off a worker thread, with the stop strings held back.

        The loop has to run somewhere and the events have to be yielded from here, and the two cannot
        be the same frame: the runtime's loop is a call that ends with the whole generation, while a
        stream is read out of it a token at a time. A thread a request is also what lets a client's
        disconnect -- which arrives on the HTTP thread as :meth:`cancel` -- reach the loop at all.
        The thread, the queue it hands events over on and the join are `_StreamRun`; what the thread
        runs is :meth:`_produce`, and the one thing left to the runtime is :meth:`_run_loop`.

        The body is deferred to the first ``next()``, as it has been for as long as there has been a
        body to defer: a caller may build the iterator and never read it -- a client that is gone
        before the response headers are written is exactly that -- and starting a generation for a
        stream nobody drains would hold the request lock until the queue's depth ran out.
        """
        self._ensure_loaded()
        self._begin_request(request.request_id)
        run = _StreamRun(self, request, self._RUNTIME_LABEL)
        run.start()
        yield from run.drain()

    def _produce(
        self, request: GenerationRequest, events: Any, outcome: dict[str, Any]
    ) -> None:
        """One streamed generation, on `_StreamRun`'s thread.

        The shape is the same for both runtimes that get here and is written once: tokenize, resolve
        the budget, hand the loop a streamer that both forwards tokens and decides where the client's
        stop string ends the run, then read the ending. What an ending *means* is where this family
        could diverge, so it is spelled out rather than inferred: a cancellation under a stop string
        is a stop string (the client asked for the cut and got it), a cancellation on its own is a
        ``cancelled`` event with no result at all, and anything else is a result whose completion
        count and usage are the loop's own.

        A subclass that streams something that is not this shape overrides this rather than
        :meth:`_run_loop`; ``v41`` does, because an answer there is a running decode to be diffed
        rather than a token to be forwarded.
        """
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
            generation = self._run_loop(prompt_ids, budget, request, streamer)
        if generation.stopped == "cancel" and not streamer.hit:
            events.put(TokenEvent(request_id=request.request_id, finish_reason="cancelled"))
            return
        # Whatever the holdback was holding: the answer is over, so the tail is either text the
        # model really wrote or the start of a stop string that never finished, and only the first
        # of those may be sent.
        streamer.flush(generation.tokens)
        outcome["result"] = self._result(
            request, prompt_ids, generation, marks, stopped="stop" if streamer.hit else None
        )
        events.put(
            TokenEvent(
                request_id=request.request_id,
                finish_reason=outcome["result"].finish_reason,
                usage=outcome["result"].usage,
                metadata={"timings": outcome["result"].timings.as_dict()},
            )
        )

    def _run_loop(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request: GenerationRequest,
        streamer: TokenStreamer,
    ) -> Any:
        """Run this runtime's loop for one streamed request, and hand back what it produced.

        The one part of a streamed request that is a call rather than a shape: the loop, with the
        streamer's two hooks passed under the names it takes them by. It is the same call a serial
        request makes, with the stop string's check standing where the cancel flag stands -- which is
        why the two ``_run_loop`` bodies in this family were byte-identical and neither survives.

        What a stream adds is `stop`: without it a run whose marker was found keeps generating tokens
        nobody reads, up to the whole budget the prompt left. The streamer also answers the step
        boundary a runtime needs, through :attr:`TokenStreamer.reached`.
        """
        return self._loop(
            prompt_ids, budget, request, on_token=streamer.accept, stop=streamer.reached
        )

    def _loop(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request: GenerationRequest,
        *,
        on_token: Callable[[int], None] | None,
        stop: Callable[[], bool] | None = None,
    ) -> Any:
        """Run this runtime's generation loop. Not implemented here.

        The whole of a runtime is this call, so the base cannot supply it: whose loop it is, what its
        parameters are called, and whether it reads the request or a payload built from it are the
        adapter's. What every runtime in this family shares is the *name* and the two hooks -- a
        token at a time, and a predicate asked before each step -- and that is what the two methods
        above are written against.
        """
        raise NotImplementedError

    def _finish_reason(self, stopped: str) -> str:
        """``stopped``, in the vocabulary the API reports.

        One table for the whole tree (``_STOPPED_FINISH_REASONS``) rather than one per route, because
        the words a route produces are a property of the loop or the scheduler that produced them and
        not of the adapter that read them.
        """
        return _STOPPED_FINISH_REASONS.get(str(stopped), "stop")

    def _result(
        self,
        request: GenerationRequest,
        prompt_ids: Sequence[int],
        generation: Any,
        marks: Mapping[str, float],
        *,
        stopped: str | None = None,
    ) -> GenerationResult:
        """The finished generation as the API's result, with this run's timings on it."""
        text = self._decode(generation.tokens)
        finish = self._finish_reason(generation.stopped if stopped is None else stopped)
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


#: How many events a stream may have in flight before its producer starts waiting.
#:
#: A slow reader is back-pressure on the producer rather than unbounded growth, and this is how much
#: is handed over before the waiting starts. The producer is a loop holding the request lock while it
#: emits, so a reader that never comes back parks it here rather than releasing it -- which is why
#: the reader's own join in `_StreamRun.drain` is bounded, and why a producer's blocking ``put`` is
#: not something this depth can be asked to fix. Both of the adapters that run a ``_loop`` under
#: `_StreamRun` had the same 64 written out; V4.1 writes its own, because its producer puts with a
#: timeout and reads a full queue as a cancellation.
_STREAM_QUEUE_DEPTH = 64


class _StreamRun:
    """One streamed request's shape: the thread that loops, and the queue it hands tokens over on.

    Written once because the two adapters that run a ``_loop`` are otherwise the same fifteen lines
    of plumbing around the same call with the same two hooks: a queue, a worker thread that runs the
    generation and puts its outcome in a box, a generator that yields what the thread put there, and
    a join that re-raises the thread's failure on the client's own thread. Every part of that is
    about a thread and a queue and nothing about a model.

    What stays with the adapter is the loop itself -- *how* it is called, with the streamer's two
    hooks under the names that loop uses -- and the adapter writes it in :meth:`_run_loop`. V4.1's
    stream is a third shape again: it diffs a running decode rather than forwarding tokens, so it
    has neither the streamer nor the budget here and keeps its own thread.
    """

    def __init__(self, adapter: Any, request: GenerationRequest, name: str) -> None:
        self.adapter = adapter
        self.request = request
        self.events: queue.Queue[Any] = queue.Queue(maxsize=_STREAM_QUEUE_DEPTH)
        #: What the loop produced, or the exception it raised; read once the thread has joined.
        self.outcome: dict[str, Any] = {}
        self.thread = threading.Thread(
            target=self._run, name=f"{name}-{request.request_id}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            self.adapter._produce(self.request, self.events, self.outcome)
        except BaseException as exc:  # a raise here must not leave the client hanging
            self.outcome["error"] = exc
        finally:
            self.adapter._clear_request(self.request.request_id)
            self.events.put(None)

    def drain(self) -> Iterator[TokenEvent]:
        """The events the producer put there, until it says it is done.

        The join is in a ``finally`` because a client that stops reading mid-stream leaves the loop
        running: the generator is being closed, and returning without joining would leave that
        thread to finish on its own, which under the request lock is a closed connection holding the
        next request out. It is bounded, because a producer parked on a full queue -- the one case
        the join cannot resolve, since nothing is draining it any more -- must not also hang a
        shutdown; a producer still running after the bound is one that is waiting for a reader.
        """
        try:
            while True:
                event = self.events.get()
                if event is None:
                    break
                yield event
        finally:
            self.thread.join(timeout=1.0)
        if "error" in self.outcome:
            raise self.outcome["error"]


class TokenStreamer:
    """The streaming half of a request: what has been sent, and what may be sent next.

    Held in an object rather than in a closure because two of these fields (`sent` and the id list)
    are written from the token callback and read from it again, and a closure that assigns to a name
    it also reads is a local variable with an unbound first read.

    The text is decoded from the run's tokens every token rather than incrementally, because a
    byte-level tokenizer cannot decode a token in isolation: what it has is a byte stream, and one
    token's bytes may end in the middle of a character. The whole decode is the only thing that
    knows; ``settled_text`` is what keeps the half-character from being sent.

    A stop string is *recorded* here and not raised. The loop it interrupts may be several ranks in
    lockstep, and only rank 0 has a stop string to find, so leaving on the spot would leave peers
    inside a collective nobody else enters. ``reached`` is handed to the loop's own per-step
    predicate instead, so the loop ends at a token boundary; what the client sees is the same text
    either way, because the cut has already been emitted by then.
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
        self._emit(hold_back(self.text, self.stops), token=self.token)

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


def hold_back(text: str, stops: Sequence[str]) -> str:
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
