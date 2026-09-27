"""Present a Python model runtime to the C++ scheduler as an engine.

`BatchScheduler` is one library with as many hosts as there are engines. `cpp` is a host because a
native engine implements `pocket::InferenceEngine` in C++, and `QwenBatchScheduler` takes that
interface rather than the one concrete engine that happened to exist first. This module is what
lets the Python runtimes be hosts too: it implements the same interface *in Python*, so `v41`,
`mimo` and `xing4` reach the same scheduler -- the same admission, the same slot lifecycle, the
same cancellation, the same per-request timings and the same gauges -- instead of each adapter
running its own request loop beside one.

Two things make that possible, and the file is mostly about them.

**A whole-generation loop, advanced a token at a time.** The runtimes in this repository generate
by running to the end: `src/models/<model>/generate.py` takes a prompt and a budget, hands each
token to an `on_token` callback, and asks an `on_step` callback before every decode step whether to
keep going. The scheduler wants the other shape -- one token per call, so it can interleave, admit
and cancel between tokens. [`RuntimeRun`] reconciles the two by running the loop on its own thread
and meeting it at the callbacks it already has: `on_step` is where it waits for permission to take
the next step and `on_token` is where the token comes back. No model code changes, and the seam is
one the runtime was already using to observe cancellation.

**Width 1, declared rather than assumed.** Every runtime here currently advances one request at a
time, so each declares `continuous_batching=False` and `max_slots=1` -- the declaration the
scheduler reads to decide how many rows it may hand over. That is not a placeholder: the scheduler
takes the smaller of the requested width and the declaration, so an honest `False` is what makes
it correct to drive such an engine at all. Raising a runtime's width is a later change, and it
happens by changing its declaration, not by changing the scheduler.

The cost of the handshake is one thread and one semaphore per in-flight request, which at width 1
is one of each. The step itself is a `Semaphore.release` and a queue read; against a decode step
that is tens of milliseconds of device work, that is not measurable.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pocketllm.api import ConfigurationError, GenerationResult, TimingMetrics, Usage

#: How long the scheduler's side waits for a runtime to hand back a token before failing the
#: request. It is a backstop against a wedged runtime, not a deadline: a 64K prefill on one of
#: these models takes minutes, and a timeout that fired on a slow but working run would turn a
#: long request into a failure.
DEFAULT_STEP_TIMEOUT = 900.0

#: Put on the token queue when the run is over. An `int` cannot be confused with it.
_DONE = object()

#: How long a run thread waits for the adapter to publish which of its requests a row is for.
#:
#: The publish happens microseconds after `submit_request` returns, on the thread that made the
#: call, so anything but an immediate answer means the adapter is not publishing at all -- a
#: programming error rather than a slow path. Short for that reason: it bounds a mistake rather
#: than a wait.
DEFAULT_ROW_TIMEOUT = 5.0


@dataclass(frozen=True)
class RuntimeSpec:
    """What the bridge needs to know about one runtime that it cannot work out for itself.

    Everything here is a fact about the runtime, not about serving: what its generation entry point
    looks like, which tokens end a sequence, how wide it can run. The lifecycle around those facts
    -- slots, admission, cancellation, timings, metrics -- is the scheduler's and is not
    parameterised.
    """

    #: Used in error messages and thread names, so a wedged run says which runtime wedged.
    name: str
    #: Runs one whole generation. Called with `request_id`, `prompt_ids`, `sampling`, `context`,
    #: `on_token` and `on_step` as keyword arguments; returns when the run is over. It must call
    #: `on_token(token)` for every generated token -- the first one included -- and `on_step()`
    #: before every decode step, stopping when it returns true.
    #:
    #: The signature of the runtime's own callbacks is deliberately *not* this one. `xing4` and
    #: `mimo` pass `(token, logits)` to `on_token` and the callers in their adapters unwrap it;
    #: adapting that is the spec's job, and doing it here would mean the bridge knew about models.
    #:
    #: `request_id` is the scheduler's id for the row. `context` is the adapter's *own* request
    #: object for it, which the adapter publishes when it submits and which carries what the
    #: scheduler has no field for -- `thinking_mode` is the one today, and it decides how a
    #: finished generation is split. It is the request's own id, not the scheduler's, that a
    #: client's disconnect arrives under, so cancellation is keyed on `context.request_id`. A spec
    #: that does not ask for it (`wants_request`) is called with `context=None`.
    start: Callable[..., None]
    #: Tokens that end a sequence. A callable rather than a set because a runtime that has not
    #: loaded its checkpoint cannot answer yet.
    eos_tokens: Callable[[], set[int]]
    max_context: int
    #: The card the *run thread* binds before it drives the runtime, or a callable answering that,
    #: or -1 for a runtime with no accelerator context.
    #:
    #: It is not the engine's `device()`: the scheduler thread this engine is called on touches no
    #: device memory, and binding a card there would bind it for the wrong thread. The forwards
    #: happen on the run thread, so that is where the runtime's own card has to be selected --
    #: torch reads the same per-thread current device the binding's `device_set` writes, so a
    #: runtime on the third card that never bound it would allocate on the first.
    device: int | Callable[[], int] = -1
    #: Concurrent request slots. One means one, and the scheduler serializes whatever width it was
    #: constructed with.
    max_slots: int = 1
    continuous_batching: bool = False
    chunked_prefill: bool = False
    paged_kv: bool = False
    per_request_sampling: bool = True
    per_request_top_k: bool = True
    step_timeout: float = DEFAULT_STEP_TIMEOUT
    #: Whether `start` wants the adapter's own request object as `context`.
    #:
    #: False by default because resolving it is not free: it is a publish-and-wait between two
    #: threads, and a runtime that renders entirely from `sampling` should not pay for it. True for
    #: a runtime that needs something the scheduler has no field for -- `thinking_mode` -- or that
    #: has to key its own cancellation on the id its client knows the request by.
    wants_request: bool = False


class RuntimeRun:
    """One runtime generation, running on its own thread, advanced one token per call.

    The thread runs the runtime's loop; this side holds the two ends of the handshake. A step is
    `take`: release the gate so the loop takes one more step, then wait for the token it produces.
    The loop's own end -- its budget, its EOS -- arrives as the `_DONE` sentinel.

    Only [`RuntimeEngine`] uses this, but it is plain Python with no native module in it, so the
    handshake is tested without one.
    """

    def __init__(
        self,
        start: Callable[..., None],
        *,
        name: str,
        timeout: float,
        device: int | Callable[[], int] = -1,
    ) -> None:
        self._start = start
        self._timeout = timeout
        self._device = device
        self._gate = threading.Semaphore(0)
        self._tokens: queue.Queue[Any] = queue.Queue()
        self._cancelled = False
        self._over = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._main, name=f"{name}-run", daemon=True)
        self._thread.start()

    # -- the runtime's side, called on the run thread ------------------------------------------

    def _permission(self) -> bool:
        """Before every decode step: block until the scheduler asks for one, or for a stop."""
        self._gate.acquire()
        return self._cancelled

    def _emit(self, token: int) -> None:
        self._tokens.put(int(token))

    def _main(self) -> None:
        try:
            self._bind_device()
            self._start(on_token=self._emit, on_step=self._permission)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the scheduler's thread
            # Held rather than raised: this is not the thread the failure belongs to. The
            # scheduler is waiting on the queue, and an exception that ended this thread silently
            # would leave it waiting until the timeout, with no idea what went wrong.
            self._error = exc
        finally:
            self._tokens.put(_DONE)

    def _bind_device(self) -> None:
        """Select the runtime's card on this thread, which is the thread the forwards run on."""
        device = self._device() if callable(self._device) else self._device
        if device is None or int(device) < 0:
            return
        import torch

        torch.cuda.set_device(int(device))

    # -- the scheduler's side -------------------------------------------------------------------

    def take(self) -> int | None:
        """Let the run take one step and return the token it produced, or None if it ended."""
        if self._over:
            return None
        self._gate.release()
        try:
            item = self._tokens.get(timeout=self._timeout)
        except queue.Empty:
            raise RuntimeError(
                f"the runtime produced no token within {self._timeout:g}s; it is either wedged or "
                "its generation entry point is not calling on_token"
            ) from None
        if item is _DONE:
            self._over = True
            if self._error is not None:
                raise RuntimeError(f"the runtime failed: {self._error}") from self._error
            return None
        return int(item)

    def cancel(self) -> None:
        """Ask the run to unwind at its next step boundary.

        A cancel cannot interrupt a forward -- there is no seam inside one -- so a run inside a long
        prefill unwinds after it, exactly as it does when a client disconnects on the existing
        streaming path. The gate is released because the loop may be parked on it, waiting for a
        step that will now never be asked for.
        """
        self._cancelled = True
        self._gate.release()

    def close(self) -> None:
        self.cancel()
        self._thread.join(timeout=1.0)


#: Built classes, keyed by `id(native)`. Held at module scope so the type is created once: pybind11
#: refuses a second registration of the same C++ base, and a per-call class would also defeat the
#: `isinstance` checks a caller might make.
_ENGINE_CLASSES: dict[int, type] = {}


def engine_class(native: Any) -> type:
    """The `InferenceEngine` subclass for `native`, defined once per module.

    Defined here rather than at import time because the native module is optional: a checkout
    without a built extension must still be able to import this file, and a class cannot inherit
    from a type that is not there. The cache is keyed on the module object so a test that hands in
    a fake native module gets a class built from that one.
    """
    cached = _ENGINE_CLASSES.get(id(native))
    if cached is not None:
        return cached

    class RuntimeEngine(native.InferenceEngine):
        """A Python runtime behind the scheduler's engine interface.

        One row per call. The scheduler hands over what it has admitted and this drives the
        runtime for exactly that, which at width 1 is one request.
        """

        def __init__(self, spec: RuntimeSpec) -> None:
            super().__init__()
            self.spec = spec
            self._native = native
            self._slots: dict[int, int] = {}
            self._free_slots: list[int] = list(range(spec.max_slots))[::-1]
            self._runs: dict[int, RuntimeRun] = {}
            # The adapter's own request for each row it has submitted, and the condition a run
            # thread waits on when it gets to a row before the adapter has published it. See
            # `row_request`.
            self._rows: dict[int, Any] = {}
            self._rows_changed = threading.Condition(threading.Lock())

        # -- the adapter's own request, per row ----------------------------------------------

        def publish_row(self, request_id: int, request: Any) -> None:
            """Record which of the adapter's requests a scheduler row is for.

            Called on the submitting thread the moment the row's id is known, which is the earliest
            it can be: the scheduler assigns it inside `submit_request`.
            """
            with self._rows_changed:
                self._rows[int(request_id)] = request
                self._rows_changed.notify_all()

        def forget_row(self, request_id: int) -> None:
            with self._rows_changed:
                self._rows.pop(int(request_id), None)

        def row_request(self, request_id: int, timeout: float = DEFAULT_ROW_TIMEOUT) -> Any:
            """The adapter's request for a row, waiting briefly if it has not been published yet.

            The wait is the whole reason this lives on the engine rather than in the adapter. A
            scheduler is free to admit a row the instant `submit_request` returns, which can be
            before the submitting thread next gets the GIL -- so a run thread that read the mapping
            without waiting would sometimes find nothing there, and a runtime that renders from it
            would render the default for that request and the requested value for the next.

            The wait releases the GIL, which is what makes it make progress rather than deadlock:
            the thread being waited for has to acquire the GIL to publish. It is bounded, and a
            timeout returns `None` -- an adapter that has nothing to say about a row is a normal
            case, not an error.
            """
            deadline = time.monotonic() + timeout
            with self._rows_changed:
                while int(request_id) not in self._rows:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._rows_changed.wait(remaining)
                return self._rows[int(request_id)]

        # -- declaration ---------------------------------------------------------------------

        def caps(self):
            caps = native.Capabilities()
            caps.max_slots = int(self.spec.max_slots)
            caps.continuous_batching = bool(self.spec.continuous_batching)
            caps.chunked_prefill = bool(self.spec.chunked_prefill)
            caps.paged_kv = bool(self.spec.paged_kv)
            caps.per_request_sampling = bool(self.spec.per_request_sampling)
            caps.per_request_top_k = bool(self.spec.per_request_top_k)
            return caps

        def max_context(self) -> int:
            return int(self.spec.max_context)

        def device(self) -> int:
            # -1, and not the spec's card: the scheduler thread this method is read from runs no
            # forward and touches no device memory, and the runtime binds its own card on the
            # thread that does. See `RuntimeSpec.device`.
            return -1

        # -- slots ---------------------------------------------------------------------------

        def allocate_batch_slots(self, max_batch_size: int) -> None:
            if max_batch_size > self.spec.max_slots:
                raise ValueError(
                    f"the {self.spec.name} runtime declares {self.spec.max_slots} slot(s); "
                    f"the scheduler asked for {max_batch_size}"
                )

        def allocate_slot(self, request_id: int) -> int:
            if request_id in self._slots:
                return self._slots[request_id]
            if not self._free_slots:
                return -1
            slot = self._free_slots.pop()
            self._slots[request_id] = slot
            return slot

        def free_slot(self, request_id: int) -> None:
            slot = self._slots.pop(request_id, None)
            if slot is not None:
                self._free_slots.append(slot)
            run = self._runs.pop(request_id, None)
            if run is not None:
                run.close()

        # -- the cache is whatever the runtime has, and it is not a block pool ----------------

        def kv_paged(self) -> bool:
            return False

        def kv_free_blocks(self) -> int:
            return 0

        def kv_total_blocks(self) -> int:
            return 0

        def kv_blocks_for_tokens(self, tokens: int) -> int:
            return 0

        # -- the two batched forwards --------------------------------------------------------

        def batch_prefill(self, requests, token_budget: int):
            out = native.BatchPrefillResult()
            rows = []
            flags = []
            consumed = 0
            for request in requests:
                if int(request.seq_len) != 0:
                    # The runtime declares chunked_prefill False, so its prompt always completes
                    # in one call and the scheduler never resumes one. Arriving here means it did,
                    # and replaying the prompt into a cache that already holds part of it would
                    # duplicate positions.
                    raise ValueError(
                        f"the {self.spec.name} runtime cannot resume a partially prefilled "
                        "prompt; it does not declare chunked_prefill"
                    )
                run = RuntimeRun(
                    self._start_for(request),
                    name=self.spec.name,
                    timeout=self.spec.step_timeout,
                    device=self.spec.device,
                )
                self._runs[int(request.request_id)] = run
                try:
                    token = run.take()
                except BaseException:
                    # The run's thread is still parked on its gate. Without this it would stay
                    # there for the life of the process, one per failed prefill, holding whatever
                    # the runtime had loaded.
                    run.close()
                    self._runs.pop(int(request.request_id), None)
                    raise
                if token is None:
                    raise RuntimeError(
                        "the runtime ended before producing a token for a prompt it was given"
                    )
                request.seq_len = len(request.prompt_tokens)
                request.last_token = token
                # A first token that ends the sequence never reaches the decode batch, so the
                # scheduler is told here rather than one step later.
                request.finished = self._is_stop(request.sampling, token)

                row = native.QwenForwardResult()
                row.token = token
                row.top_token = token
                row.position = int(request.seq_len)
                rows.append(row)
                flags.append(False)
                consumed += len(request.prompt_tokens)

            out.results = rows
            out.incomplete = flags
            out.total_tokens = consumed
            return out

        def batch_decode_step(self, requests):
            out = native.BatchDecodeResult()
            next_tokens = []
            finished = []
            hit_stop = []
            for request in requests:
                run = self._runs.get(int(request.request_id))
                if run is None:
                    raise RuntimeError(
                        f"request {request.request_id} has no run: it was never prefilled"
                    )
                token = run.take()
                if token is None:
                    # The run is over and the engine was not told by a token. Its own loop ends
                    # only on its budget or its EOS, and this engine tracks both -- the budget
                    # because it is the same number on both sides, the EOS through `_is_stop` on
                    # the token that ended it. Reaching here means the loop stopped for a reason
                    # this engine cannot see, and inventing a token for it would put a token in
                    # the answer that the model never produced.
                    raise RuntimeError(
                        f"the {self.spec.name} runtime ended a request without producing a token"
                    )
                stopped = self._is_stop(request.sampling, token)
                request.last_token = token
                next_tokens.append(token)
                finished.append(stopped)
                hit_stop.append(stopped)

            out.next_tokens = next_tokens
            out.finished = finished
            out.hit_stop_token = hit_stop
            return out

        # -- internals -----------------------------------------------------------------------

        def _start_for(self, request) -> Callable[..., None]:
            """Bind one row to the spec's entry point.

            The sampling parameters cross here rather than being applied afterwards, because the
            runtime samples its own tokens: the scheduler's copy is what the request asked for and
            the runtime is what decides what it gets, so a runtime that cannot honour a parameter
            has to say so in its declaration instead of being corrected in flight.

            `context` crosses too -- the adapter's own request object for this row, which the
            scheduler has no field for and the adapter still needs. It is resolved here, on the run
            thread, rather than in the adapter, so that the resolution can wait: the row's id exists
            only once `submit_request` returns, and the adapter publishes the mapping immediately
            after, on a thread the scheduler may have run ahead of.
            """
            sampling = request.sampling
            prompt_ids = [int(token) for token in request.prompt_tokens]
            request_id = int(request.request_id)

            context = None
            if self.spec.wants_request:
                context = self.row_request(request_id)

            return lambda *, on_token, on_step: self.spec.start(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling=sampling,
                context=context,
                on_token=on_token,
                on_step=on_step,
            )

        def _is_stop(self, sampling, token: int) -> bool:
            """Whether `token` ends this request.

            The order is the scheduler's own: `ignore_eos` first, so a benchmark that wants a fixed
            token count gets one; then the request's stop ids, which the runtime does not know
            about and which must be recognised here; and only then the checkpoint's, which is what
            the runtime's own loop also stops on.
            """
            if bool(sampling.ignore_eos):
                return False
            stop_ids = list(sampling.stop_token_ids)
            if stop_ids:
                return token in stop_ids
            return token in self.spec.eos_tokens()

    _ENGINE_CLASSES[id(native)] = RuntimeEngine
    return RuntimeEngine


def make_runtime_engine(spec: RuntimeSpec, native: Any):
    """An engine the scheduler can be constructed from, for `spec`."""
    return engine_class(native)(spec)


#: How long an adapter waits for its own submitted request to come back out of the scheduler.
#: Generous for the same reason `DEFAULT_STEP_TIMEOUT` is: a 64K prefill on one of these models takes
#: minutes, and this is a backstop against a wedged request rather than a deadline.
DEFAULT_POLL_TIMEOUT_MS = 600_000


def cancel_key(context: Any, request_id: int) -> str:
    """The id a client's `cancel` for this row arrives under.

    The bridge publishes the adapter's own request as `context`, and the request's id is what the
    HTTP layer knows that request by; the scheduler's row id is not something a client ever sees.
    Falling back to the row id keeps a runtime whose context was never published cancellable at
    all, rather than silently uncancellable.
    """
    return str(getattr(context, "request_id", None) or request_id)


def device_index(device: Any) -> int:
    """The card index a device names, or -1 for a host device.

    Accepts what a launcher option carries (``"cpu"``, ``"cuda"``, ``"cuda:2"``) and what torch
    reports (``torch.device`` and ``torch.device("cpu")``), because an adapter that reads one and
    not the other is a runtime that binds the wrong card on the one path nobody exercised.
    """
    if device is None:
        return -1
    text = str(device)
    if text.startswith("cpu"):
        return -1
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    _, _, suffix = text.partition(":")
    try:
        return int(suffix)
    except ValueError:
        return 0


class SchedulerHost:
    """The serving half of a Python runtime adapter: join the scheduler, submit to it, publish it.

    A runtime adapter is two things stacked. One is about the model -- which entry point generates,
    which tokens end a sequence, which card the weights are on -- and that cannot be shared because
    it is different for every checkpoint. The other is about serving: where the scheduler comes from,
    what it is told when it is not available, how a request is submitted and how its result is read
    back. That half is the same for every Python runtime, and it lives here so that there is one
    answer to "was `--enable-batching` asked for" rather than three.

    Keeping it here rather than in each adapter is not tidiness. The failure this avoids is specific:
    three adapters that each build their own scheduler from a slightly different reading of
    `enable_batching` are three answers to the question `--enable-batching` asks, and only one of
    them is the backend's.

    A subclass provides `_runtime_spec` and the pieces that method reaches for, and calls
    `_init_batch_scheduler()` at the end of its `__init__`. Everything else it inherits: `self.args`,
    `self.name`, `self._tokenize`, `self._budget`, `self._begin_request`, `self._clear_request` and
    `self._decode` all come from the backend base class.
    """

    #: The scheduler, when this runtime is driven by one, and the native module it came from. Both
    #: are also set by `_init_batch_scheduler`; declared here so an adapter that never calls it has
    #: the attribute rather than an `AttributeError` on the first `/capabilities`.
    _scheduler: Any = None
    _engine: Any = None
    _native: Any = None
    _poll_timeout_ms: int = DEFAULT_POLL_TIMEOUT_MS

    # -- construction ----------------------------------------------------------------------------

    def _batching_requested(self) -> bool:
        """Whether this runtime was asked to join the shared scheduler.

        Two spellings, and the backend option wins because it is the programmatic one -- the same
        precedence `cpp` uses. Unlike `cpp`, neither being set means *off*: there the batch path is
        the only one that can honour a width above one, and here it is a queue.
        """
        explicit = self.args.backend_options.get("enable_batching")
        if explicit is None:
            explicit = getattr(self.args, "enable_batching", None)
        return False if explicit is None else bool(explicit)

    def _init_batch_scheduler(self) -> None:
        """Join the shared `BatchScheduler`, when the operator asked for it and it is available.

        Off by default for these runtimes. The scheduler it joins is the same library the `cpp`
        backend drives, and at a declared width of one it is a queue rather than a batcher -- which
        is what these adapters already do under their own request lock. Asking for it is therefore
        asking for *the same code path as `cpp`*, not for concurrency, and it stays opt-in until
        that path has been measured rather than merely works.

        Three ways it does not happen, all of them reported rather than silent: no native module
        (these runtimes do not need one to serve), no scheduler binding in the native module, or
        nothing asks for it. `capabilities.details["scheduler"]` says which path is live.
        """
        self._poll_timeout_ms = self._configured_timeout()
        if not self._batching_requested():
            return
        try:
            from .cpp_backend import load_native_module

            self._native = load_native_module()
        except Exception as exc:
            self._warn_scheduler(
                f"--enable-batching needs the native pocketllm_cpp module for its scheduler "
                f"({exc})"
            )
            return
        if not hasattr(self._native, "QwenBatchScheduler"):
            self._warn_scheduler(
                "--enable-batching was asked for but this native module does not expose "
                "QwenBatchScheduler"
            )
            return
        try:
            engine = make_runtime_engine(self._runtime_spec(), self._native)
            # Held because the row-to-request mapping lives on the engine, and this is the only
            # object that knows which of its requests a row is for.
            self._engine = engine
            # The width is the scheduler's to clamp: these runtimes declare one slot, so a command
            # line naming eight is answered with one rather than refused. Refusing would make the
            # width a property of the launch script instead of of the runtime.
            self._scheduler = self._native.QwenBatchScheduler(
                engine, max(1, int(self.args.max_batch_size or 1))
            )
        except Exception as exc:
            self._scheduler = None
            self._warn_scheduler(f"could not build the batch scheduler: {exc}")

    @staticmethod
    def _warn_scheduler(reason: str) -> None:
        import warnings

        warnings.warn(f"{reason}; serving this runtime serialized, one request at a time")

    def _configured_timeout(self) -> int:
        """How long this adapter waits for one of its own requests to come back out of the queue.

        Generous by default for the reason `DEFAULT_STEP_TIMEOUT` is: a 64K prefill on one of these
        models takes minutes, and a timeout that fired on a slow but working request would turn it
        into a failure. The option is here rather than on `RuntimeSpec` because it bounds *this*
        side's wait, not the runtime's step.
        """
        raw = self.args.backend_options.get("scheduler_timeout_ms", DEFAULT_POLL_TIMEOUT_MS)
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"backend option 'scheduler_timeout_ms' is a number of milliseconds and {raw!r} "
                "is not one"
            ) from None

    def _runtime_spec(self) -> RuntimeSpec:
        """What the bridge needs to know about this model. The one method a subclass must write."""
        raise NotImplementedError

    # -- the scheduler-facing half of a request --------------------------------------------------

    def scheduler_metrics(self) -> dict[str, float]:
        """The live scheduler's admission state, or nothing when there is no scheduler."""
        return scheduler_gauges(self._scheduler)

    def _batching(self) -> bool:
        return self._scheduler is not None

    def _generate_batched(self, requests: Sequence[Any]) -> list[Any]:
        """The same generation, submitted to the shared `BatchScheduler`.

        Submission and polling are the whole of this adapter's part: which request runs when, how
        slots are handed out, when a request is cancelled and what the timings are is the
        scheduler's, exactly as it is for the `cpp` backend. That is the point of routing these
        runtimes through it -- one request lifecycle for both languages rather than two that agree
        by inspection.
        """
        scheduler = self._scheduler
        submitted: list[tuple[int, Any, list[int]]] = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                prompt_ids = self._tokenize(request)
                budget = self._budget(prompt_ids, request)
                sampling = self._batch_sampling(request, budget)
                native_id = scheduler.submit_request(prompt_ids, sampling, None)
                if native_id > 0:
                    # Which of this adapter's requests the row is for, published on the engine so
                    # the run thread can wait for it -- see `RuntimeEngine.row_request`. Without
                    # it a runtime that renders part of its answer from the request (v41's thinking
                    # mode) would render the default whenever the scheduler admitted the row before
                    # this thread got the GIL back.
                    self._engine.publish_row(int(native_id), request)
            except Exception:
                self._clear_request(request.request_id)
                raise
            if native_id <= 0:
                self._clear_request(request.request_id)
                raise RuntimeError(f"the scheduler refused request {request.request_id}")
            submitted.append((native_id, request, prompt_ids))

        outputs: list[Any] = []
        errors: list[str] = []
        for native_id, request, prompt_ids in submitted:
            try:
                result = scheduler.poll_result(native_id, self._poll_timeout_ms)
                if result is None:
                    raise TimeoutError(
                        f"request {request.request_id} did not finish within "
                        f"{self._poll_timeout_ms} ms"
                    )
                error = str(getattr(result, "error", "") or "")
                if error:
                    # Poll the rest of the batch before raising. The scheduler fails every row of
                    # a wave it could not run, and abandoning their results here would leave that
                    # many completed entries in it.
                    errors.append(error)
                    continue
                outputs.append(self._batched_result(request, result))
            finally:
                self._engine.forget_row(int(native_id))
                self._clear_request(request.request_id)

        if errors:
            raise RuntimeError("; ".join(dict.fromkeys(errors)))
        return outputs

    def _batch_sampling(self, request: Any, budget: int) -> Any:
        """One request's sampling parameters, in the scheduler's vocabulary.

        `stop` strings are deliberately not translated: the serial path applies them by holding
        text back as it streams, and a token-level stop is a different thing that these runtimes
        have no way to derive from a string. Leaving them out means both paths stop on the same
        tokens -- the checkpoint's -- instead of one of them stopping somewhere the other cannot.
        """
        params = request.sampling_params
        sampling = self._native.QwenBatchSamplingParams()
        sampling.max_new_tokens = int(budget)
        sampling.temperature = float(params.temperature or 0.0)
        sampling.top_p = float(params.top_p if params.top_p is not None else 1.0)
        sampling.top_k = int(params.top_k) if params.top_k else 0
        sampling.seed = int(params.seed or 0)
        sampling.ignore_eos = bool(params.extra.get("ignore_eos", False))
        return sampling

    def _batched_result(self, request: Any, result: Any) -> Any:
        """A scheduler result as this backend's own.

        The token list is the scheduler's, unstripped: these runtimes' serial paths keep the stop
        token in `token_ids` -- their loops append before they check -- and the two paths have to
        return the same answer, not the same answer with one token removed to look tidier.
        """
        token_ids = list(result.generated_tokens)
        return GenerationResult(
            request_id=request.request_id,
            token_ids=token_ids,
            text=self._decode(token_ids),
            finish_reason=result.finish_reason,
            usage=Usage(
                prompt_tokens=int(result.prompt_tokens),
                completion_tokens=int(result.completion_tokens),
            ),
            timings=TimingMetrics(
                prefill_seconds=max(0.0, float(result.prefill_seconds)),
                decode_seconds=max(0.0, float(result.decode_seconds)),
                total_seconds=float(result.total_seconds),
                ttft_seconds=max(0.0, float(result.ttft_seconds)),
            ),
        )


def scheduler_gauges(scheduler: Any) -> dict[str, float]:
    """The live scheduler's admission state, under the names the native host uses.

    `requests_running` is the one that matters and the only one that can separate a scheduler from
    a lock: two concurrent clients reaching either one produce identical tokens and identical
    responses, and the difference is visible only from inside the process. The same `Stats` struct,
    the same field, read here and published under the same suffix the native server uses -- so the
    two hosts are compared by substituting `pocketllm_` for `pocket_` rather than through a
    translation table.

    Nothing is published on a serialized path -- not zeros. There is no scheduler there, and a zero
    would read as "the scheduler is here and idle" about a process that does not have one.
    """
    if scheduler is None:
        return {}
    try:
        stats = scheduler.get_stats()
        caps = scheduler.engine_caps()
    except Exception:
        # A scrape is not worth failing a process's metrics over: a scheduler that cannot answer is
        # a scheduler whose gauges are absent, which is the same reading as a serialized path.
        return {}
    published = {
        "requests_running": float(stats.running_requests),
        "requests_waiting": float(stats.waiting_requests),
        "slots_free": float(stats.free_slots),
    }
    if bool(getattr(caps, "paged_kv", False)):
        published.update(
            {
                'kv_blocks{state="total"}': float(stats.total_blocks),
                'kv_blocks{state="free"}': float(stats.free_blocks),
                'kv_blocks{state="reserved"}': float(stats.reserved_blocks),
                'kv_blocks{state="cache_pinned"}': float(stats.cache_pinned_blocks),
            }
        )
    return published


__all__ = [
    "DEFAULT_POLL_TIMEOUT_MS",
    "DEFAULT_ROW_TIMEOUT",
    "DEFAULT_STEP_TIMEOUT",
    "RuntimeRun",
    "RuntimeSpec",
    "SchedulerHost",
    "cancel_key",
    "device_index",
    "engine_class",
    "make_runtime_engine",
    "scheduler_gauges",
]
