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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: How long the scheduler's side waits for a runtime to hand back a token before failing the
#: request. It is a backstop against a wedged runtime, not a deadline: a 64K prefill on one of
#: these models takes minutes, and a timeout that fired on a slow but working run would turn a
#: long request into a failure.
DEFAULT_STEP_TIMEOUT = 900.0

#: Put on the token queue when the run is over. An `int` cannot be confused with it.
_DONE = object()


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
    #: Runs one whole generation. Called with `request_id`, `prompt_ids`, `sampling`, `on_token`
    #: and `on_step` as keyword arguments; returns when the run is over. It must call
    #: `on_token(token)` for every generated token -- the first one included -- and `on_step()`
    #: before every decode step, stopping when it returns true.
    #:
    #: The signature of the runtime's own callbacks is deliberately *not* this one. `xing4` and
    #: `mimo` pass `(token, logits)` to `on_token` and the callers in their adapters unwrap it;
    #: adapting that is the spec's job, and doing it here would mean the bridge knew about models.
    #:
    #: `request_id` is the scheduler's id for the row, and it is here because an adapter's own
    #: cancellation is keyed on it: a client that disconnects calls `backend.cancel(request_id)`,
    #: and a runtime driven by the scheduler has to fold that into the same step boundary the
    #: scheduler's own cancel arrives at.
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
            """
            sampling = request.sampling
            prompt_ids = [int(token) for token in request.prompt_tokens]
            request_id = int(request.request_id)

            return lambda *, on_token, on_step: self.spec.start(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling=sampling,
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


__all__ = [
    "DEFAULT_STEP_TIMEOUT",
    "RuntimeRun",
    "RuntimeSpec",
    "engine_class",
    "make_runtime_engine",
]
