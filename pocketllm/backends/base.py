"""Shared helpers for backend adapters."""

from __future__ import annotations

import threading
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
    TokenEvent,
)
from pocketllm.choices import CHOICE_MARK
from pocketllm.protocol.contract import CHAT, FieldRefusal


#: What a byte-level tokenizer's decode puts where a token ended inside a character.
_REPLACEMENT = "�"


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
        """
        raise TensorParallelSupervisorError(
            "backend does not implement a supervised TP worker entry point"
        )

    @staticmethod
    def _metadata_copy(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}


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
