"""One request, several choices: the fan-out both front ends share.

``n`` is an OpenAI request field and the engine underneath has no notion of it. What an engine
schedules is one generation, so ``n`` choices are ``n`` generations, and the request field is
therefore a *dispatch* rule rather than a model rule. That is why the fan-out lives here and not in
an adapter: implemented per backend it would be the rule written once per runtime, and it is not the
runtime's rule to own -- the two backends that exist would answer it the same way, and the third
would have to be told.

Both hosts call in from here: ``python/pocketllm/server/openai.py`` for HTTP and ``python/pocketllm/engine.py``
for the library surface. One implementation, so ``LLM.chat(n=3)`` and an HTTP ``"n": 3`` cannot
disagree about how many generations that is.

Three rules, and all three are ported from the C++ front end that served ``n`` before this module
did:

* **A choice is its own request.** Each is admitted, scheduled and sampled on its own, which is what
  makes a cancellation or a deadline land per choice rather than per group.
* **Choice 0 keeps the caller's seed; the rest derive from it.** Varying the seed is what makes the
  choices different answers instead of one answer repeated, and it only means anything once sampling
  is stochastic -- under greedy decoding the seed is never read, so every choice is given the same
  value and the answer is the greedy text ``n`` times, which is what a greedy request for ``n``
  choices asks for.
* **Usage is OpenAI's**, not the engine's: the prompt is counted once for the request and
  ``completion_tokens`` is the sum over the choices. See :func:`folded_usage`.

The request id of a choice is its parent's with :data:`CHOICE_MARK` and the index appended, and
that shape is load-bearing rather than cosmetic: it is how ``BackendBase.cancel`` reaches every
choice of a request from the one id the client was handed (see :data:`CHOICE_MARK`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace

from pocketllm.api import (
    ConfigurationError,
    GenerationRequest,
    GenerationResult,
    RequestCancelledError,
    TokenEvent,
    Usage,
)
from pocketllm.protocol.contract import MAX_CHOICES

__all__ = [
    "CHOICE_MARK",
    "SEED_MASK",
    "RequestState",
    "choice_seed",
    "expanded",
    "folded_usage",
    "streamed",
]

#: What separates a choice's request id from the request it belongs to: ``req-1#2`` is the third
#: choice of ``req-1``. Read by ``BackendBase.cancel``, which cancels every id that is the given
#: one or begins with it and this mark -- a client cancels the request it was handed, and the
#: choices are engine requests it has never seen.
CHOICE_MARK = "#"

#: The width of the seed every engine here takes. It is unsigned, so a Python ``int`` -- which does
#: not wrap -- has to be masked into it, and the mask is stated once because it is stated in three
#: places: the splitmix64 below, the adapter that assigns the field, and the engine's own
#: ``uint64_t``. A second spelling of it is a seed that means two different things.
SEED_MASK = 0xFFFFFFFFFFFFFFFF

#: The temperature above which the sampler reads its seed at all. The same threshold the sampling
#: check uses for "is this request stochastic", and the same one ``SamplingParams.greedy`` applies.
_STOCHASTIC = 1.0e-5


def choice_seed(base: int, choice: int, *, vary: bool) -> int:
    """The seed choice ``choice`` runs at, given the one the caller named.

    Splitmix64, ported from the C++ front end, and plain 64-bit arithmetic rather than anything
    from ``random`` or ``hash`` -- a seed that depended on the interpreter version or on a hash
    randomisation seed would change the generated text from one run to the next, and the whole
    point of the field is that a caller can ask for the same choices again.

    ``vary`` is whether the request samples at all. It is passed rather than derived from ``base``
    because the base is not the engine's effective seed: a greedy request that named no seed still
    gets the derived ones, and none of them can change a token.
    """
    base &= SEED_MASK
    if choice == 0 or not vary:
        return base
    z = (base + 0x9E3779B97F4A7C15 * (choice + 1)) & SEED_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & SEED_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & SEED_MASK
    return (z ^ (z >> 31)) & SEED_MASK


def expanded(request: GenerationRequest) -> list[GenerationRequest]:
    """``request`` as its choices: itself when it asks for one, ``n`` requests when it asks for more.

    A single-choice request is handed back **unchanged and uncopied**, so the ordinary path keeps
    the id and the parameters the caller built -- the fan-out must not be observable on a request
    that did not ask for it.

    One choice is one request to the runtime, so ``n`` is what bounds how much of the queue a single
    caller can occupy; the count is held to :data:`~pocketllm.protocol.contract.MAX_CHOICES` here as
    well as in the field contract, because the library surface never builds a body and would
    otherwise reach the queue without the contract's ceiling ever being applied.
    """
    params = request.sampling_params
    count = int(params.n)
    if count <= 1:
        return [request]
    if count > MAX_CHOICES:
        raise ConfigurationError(
            f"n must be {MAX_CHOICES} or less; a request for {count} choices is {count} "
            "requests to the runtime"
        )
    vary = float(params.temperature) > _STOCHASTIC
    base = 0 if params.seed is None else int(params.seed)
    return [
        replace(
            request,
            request_id=f"{request.request_id}{CHOICE_MARK}{choice}",
            sampling_params=replace(
                params, n=1, seed=choice_seed(base, choice, vary=vary)
            ),
        )
        for choice in range(count)
    ]


def streamed(
    backend: object, request: GenerationRequest
) -> Iterator[TokenEvent]:
    """The choices of ``request`` as one stream of events, each tagged with its choice.

    One choice's stream at a time, in index order. A client watching two choices receives choice 0's
    events and then choice 1's rather than the two interleaved: interleaving would need the engine to
    be driving both at once, and a stream is serialized against the runtime's one mutable KV
    session. The chunk format is what makes that invisible to a client -- an OpenAI stream is
    accumulated per ``index``, so a chunk's position in the response carries no meaning beyond the
    choice it names.

    The event's ``request_id`` is put back to the caller's. A choice's id is how the cancel reaches
    it, and it is the one piece of fan-out bookkeeping that would otherwise leak into a response the
    client can see: ``req-1#2`` names an engine request the client never made.

    ``backend`` is the object to call ``stream`` on; it is not typed as
    :class:`~pocketllm.api.EngineBackend` because this module is imported by the server before the
    backends are, and the protocol would be a cycle for the sake of one call.
    """
    for index, choice in enumerate(expanded(request)):
        for event in backend.stream(choice):  # type: ignore[attr-defined]
            event.choice_index = index
            event.request_id = request.request_id
            yield event


def folded_usage(results: Sequence[GenerationResult]) -> Usage:
    """The usage of a group of choices, counted the way OpenAI counts it.

    ``prompt_tokens`` once for the request rather than once per choice -- the prompt was tokenized
    and prefilled per choice, but it was *asked for* once, and a client that bills per token must
    not be billed ``n`` times for the same conversation. ``completion_tokens`` is the sum, because
    that text was generated ``n`` times and really was paid for ``n`` times. ``cached_tokens`` is a
    subset of the prompt and is counted the same way the prompt is.

    The first result is the one the prompt count is read from, which assumes the choices agree
    about it -- they are the same prompt through the same tokenizer, so they do.
    """
    if not results:
        return Usage()
    return Usage(
        prompt_tokens=int(results[0].usage.prompt_tokens),
        completion_tokens=sum(int(result.usage.completion_tokens) for result in results),
        cached_tokens=int(results[0].usage.cached_tokens),
    )


class RequestState:
    """Which requests an engine backend has in flight, and which were cancelled.

    The fan-out produces ``n`` engine requests from one client request, and the client only ever
    knows the outer id. A cancellation named for that id therefore has to reach every choice of it;
    :data:`CHOICE_MARK` is what makes the choice ids recognisable, and this class is what holds the
    two sets. It is here rather than in an engine adapter because the rule it implements -- "a
    cancellation reaches a whole group" -- is the same fan-out rule the rest of this module is, and
    the two have to agree about the id shape or one of them silently stops working.

    It lived on a ``BackendBase`` that every adapter inherited. That base also carried per-runtime
    machinery -- tokenization, streaming, the serial/batch split -- none of which is shared here, so
    what survives is the lifecycle half, which is genuinely device-neutral: an adapter composes one
    of these rather than subclassing a runtime base.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._cancelled: set[str] = set()

    def cancel(self, request_id: str) -> bool:
        """Cancel ``request_id`` and every choice of it.  ``False`` if nothing was running.

        At ``n == 1`` the choice set is just ``{request_id}``, so this is the same call it always
        was; the mark only widens it. The ``False`` matters: the endpoint's answer has to mean
        something, and "no request by that id is running" is not the same as a successful cancel.
        """
        with self._lock:
            request_id = str(request_id)
            targets = {
                name
                for name in self._active
                if name == request_id or name.startswith(request_id + CHOICE_MARK)
            }
            if not targets:
                return False
            self._cancelled |= targets
            return True

    def begin(self, request_id: str) -> None:
        with self._lock:
            self._active.add(str(request_id))

    def clear(self, request_id: str) -> None:
        with self._lock:
            request_id = str(request_id)
            self._active.discard(request_id)
            self._cancelled.discard(request_id)

    def is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancelled

    def check(self, request_id: str) -> None:
        """Raise if ``request_id`` was cancelled -- the read the generation loop makes."""
        if self.is_cancelled(request_id):
            raise RequestCancelledError(f"request {request_id} was cancelled")

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def close(self) -> None:
        with self._lock:
            self._active.clear()
            self._cancelled.clear()
