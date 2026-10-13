"""The ``EngineBackend`` adapter over the C engine.

`pocketllm serve` had no implementation because nothing in this tree implemented
`EngineBackend`: the HTTP surface in :mod:`pocketllm.server.openai` is finished and
tested, and :mod:`pocketllm.native` already drives the engine token by token, but the
object between them did not exist.  This is that object.

**The engine is driven by one thread, and that is what makes concurrency safe.**
The server is a ``ThreadingHTTPServer`` -- one thread per request -- while a C
``Session`` holds one KV cache and one set of per-sequence lengths with no locking
anywhere in ``src/`` or ``native.py``.  Two threads sharing one session do not crash;
they interleave one request's tokens into another's context and return a 200 with the
wrong text.  So exactly one thread drives the engine: the scheduler below.  Requests
from every handler thread are handed to it as jobs, and it runs them in groups, each
request in its own KV slot -- the *sequence axis* Stage 1 put on the slab.  That
structural isolation is what :attr:`capabilities` reports as ``supports_batch`` (at a
width above one); it is not a promise the handler could be free to break.

The class lives under ``server/`` rather than ``backends/`` because of what it imports,
and the import table in ``tests/test_package_boundaries.py`` is what settles it:
``backends`` may reach ``kernels``, ``quant`` and itself, while this needs
:mod:`pocketllm.protocol` for the chat template and :mod:`pocketllm.native` for the
engine.  A device backend implements the kernel ABI; this implements the *serving*
contract, which is a different interface over a different layer.

``supports_cancellation`` is ``False`` and that is a fact about the C ABI, not a gap to
be filled later: ``Session::forward`` runs to completion and nothing in the exported
surface observes a flag mid-call, so a request cannot be abandoned once its forward has
started.  Declaring it true would turn ``DELETE /v1/requests/<id>`` into a promise that
silently does nothing.
"""

from __future__ import annotations

import os
import pathlib
import queue
import random
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    RequestCancelledError,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    UnsupportedFeatureError,
    Usage,
)
from pocketllm.choices import RequestState
from pocketllm.native import ContextLengthExceeded, Engine, EngineUnavailable, is_available
from pocketllm.protocol.contract import CHAT, FieldRefusal, ServedFields, audit
from pocketllm.protocol.templating import split_reasoning

__all__ = ["NativeBackend"]

#: The replacement character.  A decode that ends mid-character emits this, and it is how the
#: incremental decode below tells "these bytes are not a character yet" from "this is the text".
_REPLACEMENT = "�"

#: The fields this runtime applies, as the audit reads them.
#:
#: Every one of these defaults to ``False`` and is opted in to here, which is the safe direction:
#: a runtime that has not claimed a field gets a refusal rather than a request generated as if the
#: field were absent.  ``logprobs``, the penalties, ``logit_bias``, ``structured_outputs``,
#: ``echo``, ``suffix``, ``best_of`` and ``parallel_tool_calls`` are absent deliberately -- the C
#: ABI has no op for any of them, and refusing them by name is the difference between a 400 a
#: client can act on and a 200 that ignored what it asked for.
_SERVES = ServedFields(
    # `n` is honoured by the *host*: `choices.expanded` runs one generation per choice.  Under the
    # serializing lock that is n times the engine time for one request, which is the cost of
    # keeping a field every OpenAI client assumes.
    choices=True,
    # Matched against the running decode between tokens, which is host code.
    stop=True,
    min_p=True,
)


# -- the batch scheduler ----------------------------------------------------
#
# One request at a time was the whole concurrency story: a single C `Session`
# holds one KV cache and one set of per-sequence lengths, and two threads
# sharing it interleave one request's tokens into another's context and return
# a 200 with the wrong text.  Stage 1 gave the engine a *sequence axis* on the
# KV slab, so one forward can carry several sequences; this turns that into
# serving.
#
# The model is **grouped and static**.  A scheduler thread forms a group of up
# to `max_batch_size` requests, admits them together, and runs the whole group
# to completion: each step is one `next_tokens` call whose result is fanned
# back to the live members.  A member that finishes early keeps its slot -- fed
# a dummy token, its output ignored -- until the slowest member finishes,
# because the engine addresses a sequence by its position in the call and there
# is no way to close a hole.  New requests wait for the next group.
#
# The one hard rule: **the KV slabs are reserved once and never grow**.  The
# engine's grow path releases and reallocates the slabs without copying, so a
# sequence crossing a capacity boundary would wipe every other slot's history
# and return fluent-wrong text with no error.  `_run_group` reserves the window
# before admitting, and every member stops at the window with
# `finish_reason="length"` rather than let a forward grow past it.


#: How long the scheduler waits for a partial group to fill before running it
#: anyway.  Small enough not to be felt on a single request, large enough that
#: requests already in flight land in the same group.  A full group skips it.
_BATCH_WAIT = 0.02

#: What a finished member is fed to keep its slot alive while the group runs
#: on.  The id reaches a slot whose output is discarded, so its value does not
#: matter; what matters is that every slot contributes one token per step.
_FINISHED_TOKEN = 0


@dataclass(slots=True)
class _Job:
    """One request in the scheduler, and the channel its events come back on.

    A structural record -- the stepping lives on the backend, which owns the
    engine, the eos id and the decoder.  A job is either collected
    (``stream=False``): its terminal `GenerationResult` is put on `events`; or
    streamed: `TokenEvent`s are put on `events` as they are produced and a
    terminal ``None`` closes the stream.  A failure is put on `events` in place
    of the result and raised by whoever drains it.
    """

    request: GenerationRequest
    stream: bool
    events: queue.Queue = field(default_factory=queue.Queue)

    # -- per-request decode state, advanced by the group loop ---------------
    prompt_tokens: list[int] = field(default_factory=list)
    budget: int = 0
    emitted_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "length"
    completion_tokens: int = 0
    started: float = 0.0
    first_at: float | None = None
    decode_started: float = 0.0
    splitter: "_ReasoningSplitter | None" = None
    done: bool = False
    failure: BaseException | None = None
    #: The slot this job holds for the life of its group, and the one token
    #: that slot contributes to the next `next_tokens` call.
    slot: int = 0
    next_token: int = _FINISHED_TOKEN

    @property
    def greedy(self) -> bool:
        return self.request.sampling_params.greedy


def _as_request_error(exc: BaseException) -> BaseException:
    """The engine's context-length refusal is a bad request (400), not a 500."""
    if isinstance(exc, ContextLengthExceeded):
        return ConfigurationError(str(exc))
    return exc


class NativeBackend:
    """One C session behind the serving contract, driven by a batch scheduler.

    Constructed with :class:`~pocketllm.api.EngineArgs` and an already-resolved device
    backend name, because the two vocabularies differ: ``--device`` is a *kind* the
    registry validates and the C core is asked for a *backend* -- ``cpu`` in every
    build, ``cuda`` or ``ascend`` in a build that has one, and the 310B board serves
    on ``ascend``.  :func:`pocketllm.cli._run_device` is the same resolution the ``run``
    path applies, and this takes its answer rather than repeating it.
    """

    def __init__(self, args: EngineArgs, device: str = "cpu", *, lib: Any = None) -> None:
        self._args = args
        self._device = device
        self._path = _checkpoint_file(args.checkpoint_dir)
        self._lib = lib
        self._state = RequestState()
        #: Guards the queue depth.  It is not held across a generation -- the
        #: scheduler owns the engine for the whole of a group -- so a `metrics`
        #: scrape from an HTTP thread never waits on a decode.
        self._queue_lock = threading.Lock()
        #: How many requests are waiting for the engine, the one number that is genuinely the
        #: engine's rather than a request's -- see `metrics`.
        self._queued = 0
        self._closed = False

        if not is_available() and lib is None:
            raise BackendUnavailableError(
                "the C engine is not built on this host, so `serve` has nothing to run. "
                "Build it with `cmake -B build -S src && cmake --build build`, or point "
                "POCKETLLM_CORE_LIB at an existing libpocketllm.so."
            )
        try:
            self._engine = Engine.open(self._path, device, lib=lib)
        except EngineUnavailable as exc:
            raise BackendUnavailableError(f"the engine refused {self._path!r}: {exc}") from exc

        self._eos = _eos_id(self._path)
        self._context_length = _context_length(self._path)
        self._template = _chat_template(self._path)
        self._encoder = _build_encoder(self._template)

        # -- the batch scheduler -------------------------------------------
        # One thread owns the engine.  At width 1 it runs groups of one through
        # the unchanged single-sequence loop; above that it runs the batched
        # decode.  Either way the engine is touched by one thread only, and that
        # isolation is what `supports_batch` now reports.
        self._max_batch = max(1, int(args.max_batch_size))
        self._window = int(args.batch_kv_window)
        self._cv = threading.Condition()
        self._pending: deque[_Job] = deque()
        self._stopping = False
        #: The widths the scheduler has run, so a caller can prove the batched
        #: path was reached rather than a serialization that only looks batched.
        self._group_sizes: list[int] = []
        #: The widths the *batched decode* actually ran at, appended only inside
        #: `_run_group`.  `_group_sizes` records how many requests were
        #: co-submitted -- a group of two that ran `_run_serial` twice is still
        #: width two there -- so a gate must assert on this, not on that.
        self._batched_widths: list[int] = []
        self._scheduler = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="pocketllm-scheduler"
        )
        self._scheduler.start()

    # -- lifecycle ----------------------------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        """What this path actually has, with the two ``False``s load-bearing.

        ``supports_batch`` is true at a width above one because the scheduler gives each
        request its own KV slot on one thread -- the isolation is real, not a promise -- and
        false at width one, where there is no batch to run.  ``supports_cancellation`` is
        false because a forward cannot be interrupted.  Reporting either as true would let
        the handler's fan-out or its cancellation endpoint act on a capability that is not
        there.
        """
        return BackendCapabilities(
            name="native",
            models=(os.path.basename(self._path),),
            model_formats=("gguf",),
            devices=(self._device,),
            supports_batch=self._max_batch > 1,
            supports_streaming=True,
            supports_cancellation=False,
            supports_logprobs=False,
            supports_structured_outputs=False,
            supports_prefix_caching=False,
            details={"context_length": self._context_length, "eos_token_id": self._eos},
        )

    def prepare(self) -> None:
        """The session is opened in ``__init__``, so there is nothing left to do eagerly.

        A failure to load is raised at construction rather than at the first request, which
        is what makes ``serve`` fail at startup with a message instead of accepting a
        connection and then answering 503.
        """
        return None

    def health(self) -> HealthStatus:
        if self._closed:
            return HealthStatus(status="stopped", backend="native", ready=False)
        return HealthStatus(
            status="ready",
            backend="native",
            ready=True,
            message=f"{os.path.basename(self._path)} on {self._device}",
            details={"device": self._device, "context_length": self._context_length},
        )

    def metrics(self) -> Mapping[str, float]:
        """The queue depth, which is the engine's own and not a request's.

        The request path already reports its own counters to the server; what the HTTP layer
        cannot see is how many callers are waiting for the scheduler, and how wide the groups
        it forms actually are.  A `last_group_size` above one is the scheduler's own proof that
        the batched path ran rather than a serialization that only looks batched.  Nothing else
        is added here -- an absent series is honest, and an invented one is not.
        """
        with self._queue_lock:
            return {
                "pocketllm_waiting_requests": float(self._queued),
                "pocketllm_max_batch_size": float(self._max_batch),
                "pocketllm_last_group_size": float(self._group_sizes[-1] if self._group_sizes else 0),
            }

    def close(self) -> None:
        # Stop the scheduler and wait for it: it is mid-group on the engine, and
        # closing the engine under it would be a use-after-free.  Only after it
        # has returned is nothing else touching the session.
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._stopping = True
            self._cv.notify_all()
        self._scheduler.join(timeout=60)
        self._state.close()
        # The engine is freed only if the thread actually exited.  A group still
        # mid-forward after the timeout would be freed under: `pocketllm_close`
        # deletes the session (and its KV slab) with no sync.  Refuse rather than
        # corrupt -- an abort at shutdown is recoverable, a use-after-free is not.
        if self._scheduler.is_alive():
            raise BackendUnavailableError(
                "the scheduler thread did not stop within 60s; refusing to close the "
                "engine under a live forward (that would be a use-after-free)"
            )
        self._engine.close()

    # -- the field audit ----------------------------------------------------

    def audit_request(self, body: Mapping[str, Any], *, endpoint: str = CHAT) -> FieldRefusal | None:
        """The first field this runtime will not apply, or ``None``.

        Delegates to :func:`pocketllm.protocol.contract.audit`, which is the table written for
        exactly this call: the host has already checked the field's *shape*, and what is left
        is whether this runtime applies it.  Doing it by hand here would be a second copy of
        the field list, and the two would drift.
        """
        return audit(body, endpoint=endpoint, serves=_SERVES)

    def cancel(self, request_id: str) -> bool:
        """Answer honestly: this path cannot stop a generation that has started.

        Returning ``True`` would make ``DELETE /v1/requests/<id>`` report a cancellation
        that never happens.  ``False`` maps to a 404, and the capability above is what tells
        a client not to ask in the first place.
        """
        del request_id
        return False

    # -- generation ---------------------------------------------------------

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """One result per request, in the order given.

        The order is the contract: the server pairs results with the choices it
        dispatched, so a request answered out of order would be attached to the
        wrong choice.  Every request is submitted before any is collected, so
        they land in the same group and share a weight pass.
        """
        jobs = [self._submit(request, stream=False) for request in requests]
        return [self._collect(job) for job in jobs]

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """One request's events, yielded as they decode.

        The scheduler owns the engine, so this submits the request and drains
        the job's channel; the per-token decode, the eos/stop checks and the
        timing split all happen in the group loop -- the same code the collected
        path runs -- so streaming and collecting cannot drift.
        """
        job = self._submit(request, stream=True)
        while True:
            event = job.events.get()
            if event is None:
                break
            if isinstance(event, BaseException):
                raise event
            yield event

    def _submit(self, request: GenerationRequest, *, stream: bool) -> "_Job":
        """Enqueue the request.  Tokenizing happens later, on the scheduler thread.

        Nothing engine-backed runs here: one thread -- the scheduler -- is the
        only user of the session, which is what replaces the old lock and what
        makes ``supports_batch`` true against a real isolation rather than a
        promise.
        """
        if self._closed:
            raise BackendUnavailableError("the backend is closed")
        job = _Job(request=request, stream=stream)
        self._state.begin(request.request_id)
        self._enter_queue()
        with self._cv:
            if self._closed:
                self._state.clear(request.request_id)
                self._leave_queue()
                raise BackendUnavailableError("the backend is closed")
            self._pending.append(job)
            self._cv.notify()
        return job

    def _collect(self, job: "_Job") -> GenerationResult:
        result = job.events.get()
        if isinstance(result, BaseException):
            raise result
        return result

    # -- the scheduler thread -----------------------------------------------

    def _scheduler_loop(self) -> None:
        """Form a group, run it, repeat -- until `close` sets `_stopping`.

        The thread is the only user of the engine, so the isolation the old lock
        bought is now structural: one thread, one call at a time, and each
        request written into its own KV slot.
        """
        while True:
            with self._cv:
                while not self._pending and not self._stopping:
                    self._cv.wait()
                if self._stopping and not self._pending:
                    return
                jobs = self._take_group_locked()
            self._group_sizes.append(len(jobs))
            for job in jobs:
                try:
                    self._start_job(job)
                except BaseException as exc:  # noqa: BLE001 - reported to the job
                    job.failure = _as_request_error(exc)
            live = [job for job in jobs if job.failure is None]
            batched = self._max_batch > 1 and len(live) > 1 and all(job.greedy for job in live)
            try:
                if batched:
                    # One group failure is every member's: the engine state is
                    # shared, so a mid-group raise means the batch is lost.
                    try:
                        self._run_group(live)
                    except BaseException as exc:  # noqa: BLE001
                        for job in live:
                            if not job.done:
                                job.failure = _as_request_error(exc)
                else:
                    # Width 1, and sampling (which the greedy batched ABI cannot
                    # serve), go one at a time through the original loop.
                    for job in live:
                        try:
                            self._run_serial(job)
                        except BaseException as exc:  # noqa: BLE001
                            job.failure = _as_request_error(exc)
            finally:
                for job in jobs:
                    self._finish_job(job)

    def _take_group_locked(self) -> list["_Job"]:
        """Form one group under `self._cv`.

        A partial group waits a short grace for in-flight requests to arrive, so
        a burst lands in one group; a full group, or `close`, skips the wait.
        """
        take = min(self._max_batch, len(self._pending))
        group = [self._pending.popleft() for _ in range(take)]
        if len(group) < self._max_batch and not self._stopping:
            deadline = time.perf_counter() + _BATCH_WAIT
            while len(group) < self._max_batch and not self._stopping:
                while self._pending and len(group) < self._max_batch:
                    group.append(self._pending.popleft())
                if len(group) >= self._max_batch:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
        for slot, job in enumerate(group):
            job.slot = slot
        return group

    def _start_job(self, job: "_Job") -> None:
        """Tokenize and time the request -- on the scheduler thread, so the
        engine's tokenizer is touched by one thread only."""
        if self._state.is_cancelled(job.request.request_id):
            raise RequestCancelledError(f"request {job.request.request_id} was cancelled")
        job.prompt_tokens = self._prompt_tokens(job.request)
        job.budget = job.request.sampling_params.token_budget(
            max(1, self._context_length - len(job.prompt_tokens))
        )
        job.splitter = _ReasoningSplitter(_thinking_mode(job.request))
        job.started = time.perf_counter()

    def _run_serial(self, job: "_Job") -> None:
        """The single-sequence loop, unchanged -- width 1 and sampling.

        This is the loop the backend has always run: one `reset`, one `_advance`
        per token, the greedy fast path or the sampler behind it.  Keeping it
        verbatim is what makes a single request's streamed output byte-identical
        to the pre-scheduler backend.
        """
        params = job.request.sampling_params
        self._engine.reset()
        if len(job.prompt_tokens) >= self._context_length:
            # The single-sequence path's own refusal, kept identical to the
            # pre-scheduler `_run`: an over-length prompt is a bad request (400),
            # and the message names the prompt length and the window.  Letting the
            # engine's generic ContextLengthError surface instead would be a
            # behavior regression on the B=1 path, which must not move.
            raise ConfigurationError(
                f"the prompt is {len(job.prompt_tokens)} tokens, past this model's "
                f"context window of {self._context_length}"
            )
        token = self._advance(job.prompt_tokens, params, len(job.prompt_tokens))
        job.decode_started = time.perf_counter()
        while True:
            if not self._advance_job(job, token):
                return
            token = self._advance([token], params, len(job.prompt_tokens) + job.completion_tokens)

    def _run_group(self, jobs: list["_Job"]) -> None:
        """Prefill the group ragged, then one `next_tokens` per decode step.

        `jobs` is slot-ordered and every job is live and greedy.  The KV slabs
        are reserved once, to the window, before the first forward -- after that
        they never grow, which is what makes a mid-batch wipe impossible.
        """
        self._engine.reset()
        self._engine.reserve(self._window, self._max_batch)
        self._batched_widths.append(len(jobs))
        # The group's window is ONE number, and a padded finished slot keeps
        # advancing on it just like a live one (the engine has no way to close a
        # hole), so a bound on a job's own generation does not bound its slot's
        # position.  Bound the whole group instead: `cap` is the longest prompt,
        # every slot stops `self._window - cap` steps past its own prompt, and no
        # slot's final position (`len(prompt_tokens) + steps <= cap + (window - cap)`)
        # can reach the reservation.  Without this a short job finished early and
        # padded while a long one ran would walk its slot past `cache_capacity_`
        # and `ensure_kv` would reallocate the slab without copying, wiping the
        # long job's live KV -- fluent-wrong text with no error.
        cap = 0
        for job in jobs:
            if len(job.prompt_tokens) + 1 > self._window:
                raise ConfigurationError(
                    f"the prompt is {len(job.prompt_tokens)} tokens, past the batch KV "
                    f"window of {self._window}; raise batch_kv_window or lower max_batch_size"
                )
            cap = max(cap, len(job.prompt_tokens))
        group_steps = self._window - cap
        assert group_steps >= 1, "the reservation always leaves at least one step"
        for job in jobs:
            job.budget = max(1, min(job.budget, group_steps))
        first_tokens = self._engine.next_tokens([job.prompt_tokens for job in jobs])
        prefill_done = time.perf_counter()
        for job in jobs:
            job.decode_started = prefill_done
        for job, token in zip(jobs, first_tokens):
            self._advance_job(job, int(token))
        steps = 0
        while any(not job.done for job in jobs):
            assert steps < group_steps, (
                f"the group ran {steps} steps, past its own bound of {group_steps}; "
                "a padded slot would have grown the slab")
            batch = [[job.next_token] if not job.done else [_FINISHED_TOKEN] for job in jobs]
            ids = self._engine.next_tokens(batch)
            steps += 1
            for job, token_id in zip(jobs, ids):
                if not job.done:
                    self._advance_job(job, int(token_id))

    def _advance_job(self, job: "_Job", token: int) -> bool:
        """Fold one decoded token into a job.  Returns False when the job ends.

        The per-request body of the loop, shared by `_run_serial` and
        `_run_group` so the two cannot drift.  `job.next_token` is queued for the
        next batched step; a job that ends is fed `_FINISHED_TOKEN` until the
        group does.
        """
        if self._stopping:
            # `close` is waiting to join us; stop at a token boundary rather than
            # run the group out, so shutdown cannot outlast the join and leave the
            # engine freed under a live call.  In `_advance_job` and not in the
            # two loops, so the serial path (width 1 and sampling) is covered too.
            raise BackendUnavailableError("the backend is closing")
        job.emitted_ids.append(token)
        job.completion_tokens += 1
        job.next_token = token
        if token == self._eos:
            job.finish_reason = "stop"
            job.done = True
            return False
        text = self._decoded_text(job.emitted_ids)
        job.text = text
        if job.stream:
            assert job.splitter is not None
            reasoning, content = job.splitter.feed(text)
            if job.first_at is None:
                job.first_at = time.perf_counter()
            if reasoning or content:
                job.events.put(
                    TokenEvent(
                        request_id=job.request.request_id,
                        token_id=token,
                        text=content,
                        metadata={"reasoning_content": reasoning} if reasoning else {},
                    )
                )
        if _stop_match(text, job.request.sampling_params.stop) is not None:
            job.finish_reason = "stop"
            job.done = True
            return False
        if job.completion_tokens >= job.budget:
            job.finish_reason = "length"
            job.done = True
            return False
        return True

    def _finish_job(self, job: "_Job") -> None:
        """Deliver the terminal event (or the failure), once, and free the slot."""
        ended = time.perf_counter()
        if job.failure is not None:
            job.events.put(job.failure)
        elif job.stream:
            job.events.put(
                TokenEvent(
                    request_id=job.request.request_id,
                    finish_reason=job.finish_reason,
                    usage=Usage(
                        prompt_tokens=len(job.prompt_tokens),
                        completion_tokens=job.completion_tokens,
                    ),
                    metadata={
                        "timings": TimingMetrics(
                            prefill_seconds=(job.first_at or job.started) - job.started,
                            decode_seconds=ended - job.decode_started,
                            total_seconds=ended - job.started,
                            ttft_seconds=(job.first_at or job.started) - job.started,
                        ).as_dict()
                    },
                )
            )
            job.events.put(None)
        else:
            text = _trim_stop(job.text, job.request.sampling_params.stop)
            reasoning, content = split_reasoning(text, _thinking_mode(job.request))
            metadata: dict[str, Any] = {}
            if reasoning:
                metadata["reasoning_content"] = reasoning
            job.events.put(
                GenerationResult(
                    request_id=job.request.request_id,
                    token_ids=[t for t in job.emitted_ids if t != self._eos],
                    text=content,
                    finish_reason=job.finish_reason,
                    usage=Usage(
                        prompt_tokens=len(job.prompt_tokens),
                        completion_tokens=job.completion_tokens,
                    ),
                    timings=TimingMetrics(
                        # The collected path has no first emitted token to time
                        # from, so the prefill boundary is the end of the
                        # prefill forward -- which is what the pre-scheduler
                        # `_run` reported.
                        prefill_seconds=job.decode_started - job.started,
                        decode_seconds=ended - job.decode_started,
                        total_seconds=ended - job.started,
                    ),
                    metadata=metadata,
                )
            )
        self._state.clear(job.request.request_id)
        self._leave_queue()

    # -- helpers ------------------------------------------------------------

    def _decoded_text(self, emitted_ids: list[int]) -> str:
        """The text of everything generated so far, with no half-decoded tail.

        **Decoding one token at a time is wrong, and the engine cannot tell you so.**  A BPE
        split lands mid-character on any script whose glyphs are more than one token wide, and
        ``decode`` renders an incomplete UTF-8 sequence as a replacement character that the next
        token cannot repair: on a real checkpoint ``decode([i])`` per token turns ``한국어 텍스트``
        into ``한국어 ���스트``, and ``Ünïcödé ñ`` into ``Ünïcödé ��``.  Nothing raises -- the
        server simply returns mangled text for a whole class of scripts, which is why the loop
        decodes the grown list instead.

        Re-decoding the whole list per token is O(n^2) in tokens, and that is the right trade
        here: the list is a few hundred ids, ``decode`` is a table lookup and a join rather than a
        forward pass, and the alternative is an incremental tokenizer state machine this layer
        would have to own.  What must not be skipped is stripping the trailing replacement
        character -- a partial sequence can only ever sit at the end of the decode, so a trailing
        one is not text, while a replacement character in the middle is a genuinely unrepresentable
        byte and is kept.
        """
        text = self._engine.decode(emitted_ids)
        return text[: -len(_REPLACEMENT)] if text.endswith(_REPLACEMENT) else text

    def _enter_queue(self) -> None:
        with self._queue_lock:
            self._queued += 1

    def _leave_queue(self) -> None:
        with self._queue_lock:
            self._queued = max(0, self._queued - 1)

    def _prompt_tokens(self, request: GenerationRequest) -> list[int]:
        """The prompt as token ids, using the caller's own ids when it supplied them."""
        if request.prompt_tokens:
            return list(request.prompt_tokens)
        text = self._render(request)
        tokens = self._engine.encode(text, add_special=False, parse_special=True)
        if not tokens:
            raise ConfigurationError("the prompt tokenized to nothing")
        return tokens

    def _render(self, request: GenerationRequest) -> str:
        """The prompt text, through the checkpoint's own chat template when there is one.

        ``request.prompt`` is the protocol layer's fallback rendering -- ``role: content`` lines --
        which is a deterministic stand-in and not what the model was trained on.  A chat request
        carries its ``messages`` in the metadata, so when a template exists the real one is
        preferred; a completions request has no messages and its prompt *is* the text.
        """
        messages = request.metadata.get("messages")
        if self._encoder is not None and messages:
            return self._encoder(
                messages,
                request.metadata.get("tools"),
                bool(request.metadata.get("add_generation_prompt", True)),
                _thinking_mode(request) == "thinking",
            )
        return request.prompt or ""

    def _advance(self, last: list[int], params: SamplingParams, position: int) -> int:
        """The next token from ``last``, greedy or sampled.

        Greedy takes the ABI's argmax fast path (``Engine.next_token``) and never
        materializes the vocabulary -- on this board the whole-vocabulary round
        trip is the largest host-side cost of a decode step, so the greedy
        default must not pay it.  Sampling needs the distribution, so the
        sampling branch is the only caller of :meth:`_draw`.  Both loops (stream
        and run) drive this one method, so the two cannot drift and there is one
        place to read the routing.

        ``last`` is always a token list -- the prompt on the first call, a
        one-element list after -- because that is what both engine calls take,
        and a one-token list costs nothing against the vocabulary it replaces.
        ``position`` is the draw position, consulted only when sampling.
        """
        if params.greedy:
            return self._engine.next_token(last)
        return self._draw(self._engine.forward(last), params, position)

    def _draw(self, logits: list[float], params: SamplingParams, position: int) -> int:
        """One token, greedy or sampled, from the host's generator.

        The draw is rebuilt from ``(seed, position)`` rather than carried across the loop, and
        that is what makes the streaming and non-streaming paths produce identical text: a
        stream is a generator a client can abandon halfway, so a generator held across the loop
        would make the answer depend on how much of it was consumed.  Deriving from the position
        makes the loop a pure function of ``(logits, seed)`` in either direction -- the same
        property the CLI has, reached a different way.

        Greedy is the default and never touches the sampler, which is what keeps a request with
        no flags on the exact path `pocketllm run` takes.
        """
        if params.greedy:
            return Engine.argmax(logits)
        draw = random.Random((params.seed or 0) * 1_000_003 + position).random()
        return Engine.sample(
            Engine.temperature(logits, params.temperature),
            draw,
            top_k=params.top_k or 0,
            top_p=params.top_p if params.top_p is not None else 1.0,
            min_p=params.min_p if params.min_p is not None else 0.0,
        )


# -- module helpers ----------------------------------------------------------


def _checkpoint_file(model: str) -> str:
    """The ``.gguf`` the engine opens, from either a file path or a directory holding one.

    A directory is accepted because that is how the repository's own model paths are written,
    and a sole candidate is required: with two artifacts in one directory there is no way to
    say which the caller meant, and guessing one would be silent.
    """
    path = pathlib.Path(model)
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = sorted(p for p in path.glob("*.gguf"))
        if len(candidates) == 1:
            return str(candidates[0])
        if not candidates:
            raise ConfigurationError(f"{model!r} holds no .gguf checkpoint")
        raise ConfigurationError(
            f"{model!r} holds {len(candidates)} .gguf files; name the one to serve"
        )
    raise ConfigurationError(f"{model!r} is neither a .gguf file nor a directory")


def _metadata(path: str) -> Mapping[str, Any]:
    from pocketllm.loader.gguf.bundle import read_gguf_bundle

    try:
        return read_gguf_bundle(path).metadata
    except Exception:
        # A checkpoint the loader cannot parse still opens in C, and the engine's own error is
        # the better message -- so a missing metadata read degrades to defaults rather than
        # failing a path the C side would have accepted.
        return {}


def _eos_id(path: str) -> int:
    """The checkpoint's end-of-text id, or ``-1``.

    ``-1`` matches the C tokenizer's own default for an absent key (``bpe.cpp``), so the
    comparison below is against a value no real token can take rather than against a guess.
    """
    return int(_metadata(path).get("tokenizer.ggml.eos_token_id", -1))


def _context_length(path: str) -> int:
    """The receptive field the engine will grow its KV cache to."""
    return int(_metadata(path).get("qwen3.context_length", 0)) or 4096


def _chat_template(path: str) -> str:
    template = _metadata(path).get("tokenizer.chat_template")
    return template if isinstance(template, str) else ""


def _build_encoder(template: str):
    """A renderer for the checkpoint's Jinja chat template, or ``None`` when there is none.

    ``jinja2`` is imported lazily and its absence is not an error.  The package declares no
    dependency on it and this must not become one: a phone install has no use for a template
    engine, and ``pip install pocketllm`` must keep working without it.  Without Jinja a
    templated chat request falls back to the protocol layer's plain rendering -- which the
    model was not trained on and which produces fluent but unformatted output -- so the
    situation is reported at startup rather than left for a client to notice.

    ``enable_thinking`` is always passed, including when it is false, and that is not cosmetic.
    Qwen3's template reads it as ``enable_thinking is defined and enable_thinking is false``:
    an *undefined* variable is neither, so the prompt ends at ``<|im_start|>assistant\\n`` and
    the model opens its own `` thinking`` block -- thinking mode, in which the whole answer
    arrives wrapped in reasoning.  Passing it explicitly is the difference between the two
    modes, and leaving it to whether this function happens to supply the name would make the
    serving default depend on a renderer detail.  The default matches the protocol layer's
    (``thinking_mode == "chat"``).
    """
    if not template:
        return None
    try:
        import jinja2
    except ImportError:
        return None

    environment = jinja2.Environment(
        trim_blocks=False,
        lstrip_blocks=False,
        # The checkpoint's templates are written by model authors and are not HTML: autoescape
        # would turn a `&` in a user message into `&amp;` and change the prompt.
        autoescape=False,
        keep_trailing_newline=True,
    )
    compiled = environment.from_string(template)

    def render(
        messages: Any, tools: Any, add_generation_prompt: bool, enable_thinking: bool
    ) -> str:
        return compiled.render(
            messages=list(messages),
            tools=tools or None,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    return render


def _thinking_mode(request: GenerationRequest) -> str:
    """The request's reasoning mode, defaulting to the protocol layer's ``"chat"``."""
    mode = request.metadata.get("thinking_mode")
    return mode if isinstance(mode, str) and mode else "chat"


class _ReasoningSplitter:
    """Splits a growing decode into a reasoning delta and a content delta.

    The split itself is :func:`pocketllm.protocol.templating.split_reasoning`, so the streaming
    and non-streaming answers cannot disagree about where the thinking block ends.  What is
    added here is the diffing: a stream has already sent everything it has seen, so each step
    reports only what is new.

    The one case that cannot be recovered is a ``</think>`` marker straddling a token boundary.
    The characters before the boundary were sent as reasoning and are not retractable, so the
    splitter emits nothing further for reasoning (the prefix no longer grows) and sends the
    remainder as content -- which is the cost ``split_reasoning`` documents, paid here rather
    than at a retraction the wire format has no room for.
    """

    def __init__(self, thinking_mode: str) -> None:
        self._mode = thinking_mode
        self._reasoning = ""
        self._content = ""

    def feed(self, text: str) -> tuple[str, str]:
        """The new ``(reasoning, content)`` since the last call."""
        reasoning, content = split_reasoning(text, self._mode)
        delta_reasoning = (
            reasoning[len(self._reasoning):] if reasoning.startswith(self._reasoning) else ""
        )
        delta_content = (
            content[len(self._content):] if content.startswith(self._content) else content
        )
        self._reasoning, self._content = reasoning, content
        return delta_reasoning, delta_content


def _stop_match(text: str, stop: Sequence[str]) -> int | None:
    """The index a stop sequence first appears at, or ``None``."""
    for sequence in stop:
        if sequence and sequence in text:
            return text.index(sequence)
    return None


def _trim_stop(text: str, stop: Sequence[str]) -> str:
    """``text`` up to the earliest stop sequence, which is what the response should carry."""
    found = _stop_match(text, stop)
    return text if found is None else text[:found]