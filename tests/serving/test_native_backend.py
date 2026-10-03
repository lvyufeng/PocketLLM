"""The serving adapter over the real C engine.

Every test here needs two things a bare checkout does not have -- ``libpocketllm.so`` and a
Qwen3-0.6B GGUF -- so the module skips itself without them, and a skip is not a pass.

What is worth stating is *why* these are not covered by ``test_server.py``.  That file drives the
HTTP layer against a scripted fake, which is the right way to test the protocol and says nothing
about whether a real engine can be driven through it.  Three of the tests below exist because the
adapter made a mistake that no fake would have caught:

* the per-token decode, which mangles any script whose glyphs span more than one token;
* the end-of-text token, which decodes to its literal spelling and was streamed as text;
* the session lock, without which concurrent requests return *wrong answers and no error*.

The last of those is the module's reason to exist, and ``test_concurrent_requests_stay_separate``
is the one that would fail loudest if it were removed.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from urllib import error, request as urlrequest

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.protocol import build_chat_request, build_completion_request
from pocketllm.protocol.contract import COMPLETIONS

pytestmark = pytest.mark.skipif(
    not pathlib.Path("/mnt/data1/models/qwen3-0.6b-f16.gguf").exists(),
    reason="the Qwen3-0.6B checkpoint is not on this host",
)

CHECKPOINT = "/mnt/data1/models/qwen3-0.6b-f16.gguf"

#: The checkpoint's end-of-text id, read here rather than imported so a change to how the adapter
#: finds it cannot hide behind the same lookup on both sides.
EOS = 151645


def _native_available() -> bool:
    from pocketllm import native

    return native.is_available()


pytestmark = [
    pytestmark,
    pytest.mark.skipif(
        not _native_available(),
        reason="libpocketllm.so is not built (cmake -B build -S src && cmake --build build)",
    ),
]


@pytest.fixture(scope="module")
def backend():
    from pocketllm.server.native_backend import NativeBackend

    instance = NativeBackend(EngineArgs(model=CHECKPOINT), "cpu")
    yield instance
    instance.close()


@pytest.fixture()
def served(backend):
    """The adapter behind the real HTTP server, on an ephemeral port.

    A fresh ``ThreadingHTTPServer`` per test rather than a module-scoped one, because the tests
    that matter here are the ones that send *concurrent* requests and a shared server would let
    one test's sockets outlive it.
    """
    from pocketllm.server.openai import PocketLLMHTTPServer, OpenAIHandler, serve

    server = PocketLLMHTTPServer(("127.0.0.1", 0), OpenAIHandler, backend, "qwen3")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _post(base: str, path: str, body: dict, *, timeout: float = 180.0):
    req = urlrequest.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _chat(backend, text: str, **fields):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 16, **fields}
    return backend.generate([build_chat_request(body)])[0]


# -- what the backend declares ----------------------------------------------


def test_capabilities_declare_what_the_engine_cannot_do(backend) -> None:
    """The two ``False``s are the whole concurrency story and are asserted, not assumed.

    ``supports_batch`` false is what stops the handler expecting overlap it cannot get;
    ``supports_cancellation`` false is what stops ``DELETE /v1/requests/<id>`` reporting a
    cancellation that never happens.  Either one flipped without the engine changing underneath
    would be a lie the HTTP layer acts on.
    """
    caps = backend.capabilities
    assert caps.supports_batch is False
    assert caps.supports_cancellation is False
    assert caps.supports_streaming is True
    assert caps.supports_logprobs is False
    assert caps.details["eos_token_id"] == EOS


def test_cancel_reports_that_it_cannot(backend) -> None:
    """A forward cannot be interrupted, so the endpoint must not claim otherwise."""
    assert backend.cancel("anything") is False


def test_a_missing_checkpoint_fails_at_construction() -> None:
    """``serve`` must not advertise a port it cannot answer on."""
    from pocketllm.server.native_backend import NativeBackend

    with pytest.raises(ConfigurationError):
        NativeBackend(EngineArgs(model="/nonexistent/model.gguf"), "cpu")


# -- the field audit ---------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"frequency_penalty": 0.5},
        {"presence_penalty": 0.5},
        {"repetition_penalty": 1.2},
        {"logit_bias": {"1": 2.0}},
        {"response_format": {"type": "json_object"}},
        {"logprobs": True},
        {"parallel_tool_calls": False},
    ],
)
def test_fields_the_engine_cannot_apply_are_refused_by_name(backend, body) -> None:
    """A field with no C op behind it is refused rather than silently ignored.

    ``param`` is what a client acts on, so the refusal has to carry the *name* and not just a
    sentence -- which is why the assertion is on the field rather than the message.
    """
    refusal = backend.audit_request(body)
    assert refusal is not None, f"{body} should have been refused"
    assert refusal.field in body


@pytest.mark.parametrize(
    "body",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 40},
        {"min_p": 0.05},
        {"stop": ["x"]},
        {"n": 2},
        {"seed": 7},
        {"max_tokens": 8},
        {},
    ],
)
def test_fields_the_engine_applies_are_accepted(backend, body) -> None:
    assert backend.audit_request(body) is None


@pytest.mark.parametrize("field", ["echo", "suffix", "best_of"])
def test_completions_only_fields_are_refused_on_that_endpoint(backend, field) -> None:
    value = {"echo": True, "suffix": "x", "best_of": 2}[field]
    assert backend.audit_request({field: value}, endpoint=COMPLETIONS).field == field


# -- generation --------------------------------------------------------------


def test_greedy_generation_is_coherent(backend) -> None:
    """The default path, end to end, against a question with a stable answer."""
    result = _chat(backend, "What is the capital of France? Answer in one word.")
    assert "Paris" in result.text
    assert result.finish_reason == "stop"
    assert result.usage.completion_tokens > 0


def test_the_end_of_text_token_is_not_part_of_the_answer(backend) -> None:
    """The EOS token decodes to ``<|im_end|>``, and that string must never reach a client.

    This is a real bug the first draft had, and it is worth a test rather than a comment because
    the failure is silent: the response is valid JSON, the tokens are counted, and the text simply
    has a control marker glued to the end of it.
    """
    result = _chat(backend, "Say exactly: hello")
    assert "<|im_end|>" not in result.text
    assert "<|im_start|>" not in result.text
    assert result.finish_reason == "stop"


def test_multibyte_text_survives_the_decode(backend) -> None:
    """The regression test for decoding one token at a time.

    A BPE split lands mid-character in any script whose glyphs are wider than a token, and the
    incomplete sequence decodes to U+FFFD that the next token cannot repair.  The model is asked
    to repeat Korean because a 0.6B model will echo given text far more reliably than it will
    compose a sentence in a script it may not know -- and it is the *decode*, not the model, that
    is under test.
    """
    result = _chat(backend, "Repeat this exactly, nothing else: 한국어 텍스트", max_tokens=24)
    assert "�" not in result.text, f"mangled text: {result.text!r}"


def test_a_stop_sequence_ends_the_generation_and_is_trimmed(backend) -> None:
    """A stop sequence is matched against the decoded text and the response excludes it.

    The prompt is one whose continuation is predictable on this checkpoint -- a model this small
    is asked to continue a sequence it has certainly seen rather than to obey an instruction to
    emit a particular word, which it does not reliably do.  The first prompt tried here ("Count:
    one, two, three...") ran to the length cap without ever producing the stop sequence, so the
    test passed nothing: it asserted a stop that never happened.
    """
    result = backend.generate(
        [
            build_completion_request(
                {
                    "prompt": "The sequence is a, b, c, d, e, f. So next is",
                    "max_tokens": 40,
                    "stop": ["e"],
                }
            )
        ]
    )[0]
    assert result.finish_reason == "stop"
    assert "e" not in result.text


def test_the_length_cap_is_honoured(backend) -> None:
    """``finish_reason`` distinguishes a cap from a stop, which a client uses to decide to ask for
    more.  A generation that reaches the cap must say ``length`` rather than ``stop``."""
    result = backend.generate(
        [build_completion_request({"prompt": "Write an essay about the sea.", "max_tokens": 4})]
    )[0]
    assert result.finish_reason == "length"
    assert result.usage.completion_tokens == 4


def test_thinking_mode_separates_reasoning_from_content(backend) -> None:
    """With thinking on, the reasoning is split out rather than left in ``content``.

    The adapter passes ``enable_thinking`` explicitly; leaving the variable undefined is not the
    same as passing it false, because the template tests ``is defined and ... is false`` and an
    undefined name makes the model open its own think block.
    """
    body = {
        "messages": [{"role": "user", "content": "What is 17 times 23?"}],
        "max_tokens": 48,
    }
    request = build_chat_request(body)
    request.metadata["thinking_mode"] = "thinking"
    result = backend.generate([request])[0]
    assert result.metadata.get("reasoning_content"), "thinking mode produced no reasoning"
    assert "Okay" not in result.text[:20]

    plain = build_chat_request(body)
    plain.metadata["thinking_mode"] = "chat"
    assert not backend.generate([plain])[0].metadata.get("reasoning_content")


# -- streaming ---------------------------------------------------------------


def test_streaming_and_collecting_produce_the_same_text(backend) -> None:
    """The two paths must not be two answers.

    A stream is assembled from deltas and the collected path from one decode, so a difference in
    where either splits -- or an end-of-text token yielded before it is checked -- shows up here
    and nowhere else.
    """
    request = build_chat_request(
        {"messages": [{"role": "user", "content": "Name three colours."}], "max_tokens": 24}
    )
    streamed = "".join(event.text for event in backend.stream(request))
    collected = backend.generate([request])[0].text
    assert streamed == collected
    assert streamed


def test_the_stream_ends_with_a_finish_reason(backend) -> None:
    """The last event carries the reason and the usage, which the SSE writer reads."""
    request = build_chat_request(
        {"messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 8}
    )
    events = list(backend.stream(request))
    assert events[-1].finish_reason in {"stop", "length"}
    assert events[-1].usage is not None
    assert events[-1].token_id is None


# -- concurrency, which is why the lock is there -----------------------------


def test_concurrent_requests_stay_separate(backend) -> None:
    """The test the whole module exists for: six threads, six prompts, six right answers.

    Without the session lock this does not fail -- it returns six *wrong* answers and no
    exception, because the requests share one ``position_`` and one KV cache.  Measured on this
    host: 6/6 correct with the lock, 0/6 without, zero errors either way.  So the assertion has to
    be on the text and not on whether anything raised.
    """
    count = 6
    results: dict[int, str] = {}
    failures: dict[int, str] = {}

    def worker(index: int) -> None:
        request = build_chat_request(
            {
                "messages": [{"role": "user", "content": f"Reply with only the word: item{index}"}],
                "max_tokens": 8,
            },
            request_id=f"req-{index}",
        )
        try:
            results[index] = backend.generate([request])[0].text.strip()
        except Exception as exc:  # noqa: BLE001 - reported as a failure, not swallowed
            failures[index] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)

    assert not failures, f"{len(failures)} of {count} raised: {failures}"
    wrong = {i: text for i, text in results.items() if f"item{i}" not in text}
    assert not wrong, f"{len(wrong)} of {count} answers belong to another request: {wrong}"


def test_the_session_lock_serializes_the_engine(backend) -> None:
    """Mutual exclusion, observed rather than assumed.

    The test above proves the *symptom* is gone; this proves the mechanism is what removed it, by
    recording whether two generations were ever inside the engine at the same time.  Without it a
    future refactor could widen the lock into a no-op that still happened to pass under a
    favourable interleaving.
    """
    depth = 0
    peak = 0
    guard = threading.Lock()
    original = backend._engine

    class Counting:
        def __getattr__(self, name):
            attr = getattr(original, name)
            if name != "forward":
                return attr

            def wrapped(*args, **kwargs):
                nonlocal depth, peak
                with guard:
                    depth += 1
                    peak = max(peak, depth)
                try:
                    time.sleep(0.01)
                    return attr(*args, **kwargs)
                finally:
                    with guard:
                        depth -= 1

            return wrapped

    backend._engine = Counting()
    try:
        threads = [
            threading.Thread(
                target=backend.generate,
                args=(
                    [
                        build_chat_request(
                            {"messages": [{"role": "user", "content": f"Say {i}"}], "max_tokens": 2},
                            request_id=f"serial-{i}",
                        )
                    ],
                ),
            )
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
    finally:
        backend._engine = original

    assert peak == 1, f"{peak} generations were inside the engine at once; the lock is not holding"


# -- over HTTP ---------------------------------------------------------------


def test_chat_completion_over_http(served) -> None:
    status, body = _post(
        served,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}], "max_tokens": 12},
    )
    assert status == 200
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_a_refused_field_is_a_400_naming_it(served) -> None:
    """The refusal reaches the wire as ``param``, which is what a client can act on."""
    status, body = _post(
        served,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "frequency_penalty": 0.5},
    )
    assert status == 400
    assert body["error"]["param"] == "frequency_penalty"
    assert body["error"]["code"] == "unsupported_feature"


def test_n_choices_over_http(served) -> None:
    """``n`` is honoured by the host fan-out, which is what ``ServedFields.choices`` claims."""
    status, body = _post(
        served,
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "Name a colour."}],
            "max_tokens": 8,
            "n": 2,
            "temperature": 0.9,
        },
    )
    assert status == 200
    assert [choice["index"] for choice in body["choices"]] == [0, 1]


def test_streaming_over_http_is_sse(served) -> None:
    req = urlrequest.Request(
        served + "/v1/chat/completions",
        data=json.dumps(
            {"messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 8, "stream": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    deltas = []
    with urlrequest.urlopen(req, timeout=180) as response:
        for line in response:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                deltas.append(json.loads(line[6:]))
    assert deltas[0]["choices"][0]["delta"].get("role") == "assistant"
    assert "".join(d["choices"][0]["delta"].get("content") or "" for d in deltas)
    assert deltas[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


def test_health_and_metrics(served) -> None:
    with urlrequest.urlopen(served + "/health", timeout=10) as response:
        health = json.loads(response.read())
    assert health["status"] == "ready"
    assert health["backend"] == "native"

    with urlrequest.urlopen(served + "/metrics", timeout=10) as response:
        text = response.read().decode()
    assert "pocketllm_waiting_requests" in text