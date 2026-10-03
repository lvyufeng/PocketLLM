"""The ``EngineBackend`` adapter over the C engine.

`pocketllm serve` had no implementation because nothing in this tree implemented
`EngineBackend`: the HTTP surface in :mod:`pocketllm.server.openai` is finished and
tested, and :mod:`pocketllm.native` already drives the engine token by token, but the
object between them did not exist.  This is that object.

**One request at a time, and the lock is not an inconvenience.**  The server is a
``ThreadingHTTPServer`` -- one thread per request -- while a C ``Session`` holds one
``position_`` and one KV cache with no locking anywhere in ``src/`` or ``native.py``.
Two threads sharing one session do not crash; they interleave one request's tokens into
another's context and return a 200 with the wrong text, which is the worst failure a
server can have and the reason :attr:`capabilities` declares ``supports_batch = False``
rather than leaving the handler free to fan out against a promise nothing can keep.

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
import random
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
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
from pocketllm.native import Engine, EngineUnavailable, is_available
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


class NativeBackend:
    """One C session behind the serving contract, serialized by a lock.

    Constructed with :class:`~pocketllm.api.EngineArgs` and an already-resolved device
    backend name, because the two vocabularies differ: ``--device`` is a *kind* the
    registry validates and the C core is asked for a *backend*, which is ``cpu`` or
    ``cuda`` today.  :func:`pocketllm.cli._run_device` is the same resolution the ``run``
    path applies, and this takes its answer rather than repeating it.
    """

    def __init__(self, args: EngineArgs, device: str = "cpu", *, lib: Any = None) -> None:
        self._args = args
        self._device = device
        self._path = _checkpoint_file(args.checkpoint_dir)
        self._lib = lib
        self._state = RequestState()
        #: Guards the session *and* the decode loop.  A plain `Lock` and not a `RLock`: nothing
        #: inside the critical section re-enters, and an `RLock` would hide it if something did.
        self._lock = threading.Lock()
        #: Guards the two fields below, and is deliberately *not* `self._lock`: the queue depth is
        #: read by `metrics` from the HTTP thread while a generation holds the session lock, so a
        #: metrics scrape would block for as long as the generation runs.  Two locks, because they
        #: protect two unrelated things.
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

    # -- lifecycle ----------------------------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        """What this path actually has, with the two ``False``s load-bearing.

        ``supports_batch`` is false because one session cannot be shared; read the module
        docstring for what going the other way would cost.  ``supports_cancellation`` is
        false because a forward cannot be interrupted.  Reporting either as true would let
        the handler's fan-out or its cancellation endpoint act on a capability that is not
        there.
        """
        return BackendCapabilities(
            name="native",
            models=(os.path.basename(self._path),),
            model_formats=("gguf",),
            devices=(self._device,),
            supports_batch=False,
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
        cannot see is how many callers are waiting behind the lock, and under serialization
        that is the number an operator needs.  Nothing else is added here -- an absent series
        is honest, and an invented one is not.
        """
        with self._queue_lock:
            return {"pocketllm_waiting_requests": float(self._queued)}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state.close()
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
        """One run per request, in the order given, each holding the lock for its whole life.

        The order is the contract: the server pairs results with the choices it dispatched, so
        a request answered out of order would be attached to the wrong choice.
        """
        return [self._run(request) for request in requests]

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """One run, yielded as it decodes.

        The lock is held across the whole generator, not per token: between two ``forward``
        calls the session's position is one token further on, and releasing it there would let
        a second request write into the same KV cache.  The cost is that a slow client holds
        the engine, which is the same serialization the non-streaming path pays.
        """
        self._state.begin(request.request_id)
        self._enter_queue()
        self._lock.acquire()
        try:
            if self._state.is_cancelled(request.request_id):
                raise RequestCancelledError(f"request {request.request_id} was cancelled")
            self._engine.reset()
            prompt_tokens = self._prompt_tokens(request)
            started = time.perf_counter()
            logits = self._engine.forward(prompt_tokens)
            first_at: float | None = None
            budget = request.sampling_params.token_budget(
                max(1, self._context_length - len(prompt_tokens))
            )
            decode_started = time.perf_counter()

            # The decoded text is carried whole and diffed, rather than assembled from
            # per-token decodes.  A BPE split lands mid-character on any script whose glyphs
            # are more than one token wide -- `decode([id])` on a partial UTF-8 sequence emits
            # a replacement character that the next token cannot undo, so '한국어' arrives as
            # '한국어 �' and never repairs.  Decoding the grown list and stripping a
            # trailing replacement is correct because a partial sequence can only ever sit at
            # the end: everything before it is complete, and the next token completes it.
            emitted_ids: list[int] = []
            splitter = _ReasoningSplitter(_thinking_mode(request))
            finish_reason = "length"
            completion_tokens = 0

            for _ in range(budget):
                token = self._draw(logits, request.sampling_params, len(prompt_tokens) + completion_tokens)
                emitted_ids.append(token)
                completion_tokens += 1
                # Before the decode, not after: the end-of-text token decodes to its literal
                # spelling -- Qwen3's is `<|im_end|>` -- and yielding first would put that in the
                # response text.  The collected path breaks here too, so the two agree.
                if token == self._eos:
                    finish_reason = "stop"
                    break
                # See `_decoded_text` for why the whole list is decoded rather than one id.
                text = self._decoded_text(emitted_ids)
                reasoning, content = splitter.feed(text)

                if first_at is None:
                    first_at = time.perf_counter()
                if reasoning or content:
                    yield TokenEvent(
                        request_id=request.request_id,
                        token_id=token,
                        text=content,
                        metadata={"reasoning_content": reasoning} if reasoning else {},
                    )

                if _stop_match(text, request.sampling_params.stop) is not None:
                    # The stop sequence was matched against a text whose content half has already
                    # been sent, which is the one place a stream is not byte-exact with the
                    # collected answer: a client sees the sequence it asked to stop at.  Holding
                    # the tokens back until the sequence is known to be absent would mean a
                    # lookahead of its length, and `_run` -- which has the whole text -- trims
                    # instead.  The difference is documented rather than hidden.
                    finish_reason = "stop"
                    break
                logits = self._engine.forward([token])

            ended = time.perf_counter()
            usage = Usage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
            )
            yield TokenEvent(
                request_id=request.request_id,
                finish_reason=finish_reason,
                usage=usage,
                metadata={
                    "timings": TimingMetrics(
                        prefill_seconds=(first_at or started) - started,
                        decode_seconds=ended - decode_started,
                        total_seconds=ended - started,
                        ttft_seconds=(first_at or started) - started,
                    ).as_dict()
                },
            )
        finally:
            self._lock.release()
            self._leave_queue()
            self._state.clear(request.request_id)

    def _run(self, request: GenerationRequest) -> GenerationResult:
        """The whole generation, collected.  The non-streaming twin of :meth:`stream`."""
        self._state.begin(request.request_id)
        self._enter_queue()
        self._lock.acquire()
        try:
            if self._state.is_cancelled(request.request_id):
                raise RequestCancelledError(f"request {request.request_id} was cancelled")
            self._engine.reset()
            prompt_tokens = self._prompt_tokens(request)
            started = time.perf_counter()
            logits = self._engine.forward(prompt_tokens)
            prefill_done = time.perf_counter()
            budget = request.sampling_params.token_budget(
                max(1, self._context_length - len(prompt_tokens))
            )

            emitted_ids: list[int] = []
            text = ""
            finish_reason = "length"
            for _ in range(budget):
                token = self._draw(
                    logits, request.sampling_params, len(prompt_tokens) + len(emitted_ids)
                )
                emitted_ids.append(token)
                if token == self._eos:
                    finish_reason = "stop"
                    break
                text = self._decoded_text(emitted_ids)
                if _stop_match(text, request.sampling_params.stop) is not None:
                    finish_reason = "stop"
                    break
                logits = self._engine.forward([token])

            completed = time.perf_counter()
            text = _trim_stop(text, request.sampling_params.stop)
            # The same split the streaming path applies, computed once over the finished text,
            # so the two answers agree about where the reasoning ends.  `_ReasoningSplitter`
            # exists for the streaming case; here the whole text is in hand.
            reasoning, content = split_reasoning(text, _thinking_mode(request))
            metadata: dict[str, Any] = {}
            if reasoning:
                metadata["reasoning_content"] = reasoning
            return GenerationResult(
                request_id=request.request_id,
                token_ids=[t for t in emitted_ids if t != self._eos],
                text=content,
                finish_reason=finish_reason,
                usage=Usage(prompt_tokens=len(prompt_tokens), completion_tokens=len(emitted_ids)),
                timings=TimingMetrics(
                    prefill_seconds=prefill_done - started,
                    decode_seconds=completed - prefill_done,
                    total_seconds=completed - started,
                ),
                metadata=metadata,
            )
        finally:
            self._lock.release()
            self._leave_queue()
            self._state.clear(request.request_id)

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