from __future__ import annotations

import json
import re
import threading
from urllib import error, request

import pytest

from pocketllm.api import (
    BackendCapabilities,
    GenerationResult,
    TokenEvent,
    UnsupportedFeatureError,
    Usage,
)
from pocketllm.backends.base import BackendBase, settled_text
from pocketllm.server.metrics import HISTOGRAMS, Metrics
from pocketllm.server.openai import OpenAIHandler, PocketLLMHTTPServer


class ContractBackend(BackendBase):
    def __init__(self, fail: bool = False):
        super().__init__()
        self._ready = True
        self._fail = fail
        self.seen: list = []

    @property
    def capabilities(self):
        return BackendCapabilities(name="fake", supports_streaming=True, supports_cancellation=True)

    def generate(self, requests):
        self.seen.extend(requests)
        if self._fail:
            raise UnsupportedFeatureError("logprobs are not exposed by this backend")
        return [GenerationResult(
            request_id=req.request_id,
            token_ids=[11],
            text="ok",
            usage=Usage(2, 1),
        ) for req in requests]

    def stream(self, req):
        self._begin_request(req.request_id)
        try:
            if self._fail:
                yield TokenEvent(req.request_id, text="o", token_id=11)
                raise UnsupportedFeatureError("stop strings are not exposed by this backend")
            yield TokenEvent(req.request_id, text="o", token_id=11)
            yield TokenEvent(req.request_id, text="k", token_id=12, finish_reason="stop", usage=Usage(2, 2))
        finally:
            self._clear_request(req.request_id)


def _server(fail: bool = False, backend: ContractBackend | None = None):
    backend = backend or ContractBackend(fail=fail)
    server = PocketLLMHTTPServer(("127.0.0.1", 0), OpenAIHandler, backend, "fake-model")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(base, path, body):
    req = request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode())


def _post_raw(base, path, body):
    req = request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return response.read().decode()


def _metric_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(f"pocketllm_{name} "):
            return float(line.rsplit(" ", 1)[-1])
    raise AssertionError(f"metric {name} not exported:\n{text}")


_BUCKET_LINE = re.compile(
    r'^pocketllm_(?P<family>\w+)_bucket\{le="(?P<bound>[^"]+)"\} (?P<count>\S+)$'
)


def _buckets(text: str, family: str) -> list[tuple[str, float]]:
    """The `_bucket` series of one family, in exposition order."""
    found = []
    for line in text.splitlines():
        match = _BUCKET_LINE.match(line)
        if match and match.group("family") == family:
            found.append((match.group("bound"), float(match.group("count"))))
    return found


def _metrics(base: str) -> str:
    with request.urlopen(base + "/metrics", timeout=10) as response:
        return response.read().decode()


class TokenCountBackend(ContractBackend):
    """Streams a fixed number of token-bearing events and nothing else."""

    def __init__(self, tokens: int):
        super().__init__()
        self._tokens = tokens

    def stream(self, req):
        self._begin_request(req.request_id)
        try:
            for index in range(self._tokens):
                yield TokenEvent(
                    req.request_id,
                    text="x",
                    token_id=100 + index,
                    finish_reason="stop" if index == self._tokens - 1 else None,
                )
        finally:
            self._clear_request(req.request_id)


def test_shared_server_routes_chat_and_completions():
    server, base = _server()
    try:
        chat = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["content"] == "ok"
        completion = _post(base, "/v1/completions", {"prompt": "hi"})
        assert completion["object"] == "text_completion"
        assert completion["choices"][0]["text"] == "ok"
        with request.urlopen(base + "/ready", timeout=10) as response:
            assert response.status == 200
        metrics = _metrics(base)
        assert "pocketllm_requests_total" in metrics
        # The active-request gauge must return to zero once requests finish.
        assert _metric_value(metrics, "requests_active") == 0.0
        assert _metric_value(metrics, "prompt_tokens_total") == 4.0
    finally:
        server.shutdown()
        server.server_close()


def test_completion_streaming_uses_text_completion_chunks():
    server, base = _server()
    try:
        raw = _post_raw(base, "/v1/completions", {"prompt": "hi", "stream": True})
        payloads = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(item) for item in payloads if item != "[DONE]"]
        assert all(chunk["object"] == "text_completion" for chunk in chunks)
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "ok"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["completion_tokens"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_chat_streaming_keeps_chat_chunk_schema():
    server, base = _server()
    try:
        raw = _post_raw(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        payloads = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        chunks = [json.loads(item) for item in payloads if item != "[DONE]"]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "ok"
        assert chunks[-1]["object"] == "chat.completion.chunk"
    finally:
        server.shutdown()
        server.server_close()


def test_typed_backend_errors_map_to_http_status():
    server, base = _server(fail=True)
    try:
        try:
            _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
            raise AssertionError("expected an HTTP error")
        except error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode())
            assert body["error"]["type"] == "unsupported_feature"
        metrics = _metrics(base)
        assert _metric_value(metrics, "request_errors_total") == 1.0
    finally:
        server.shutdown()
        server.server_close()


def test_stream_backend_failure_is_reported_in_band():
    server, base = _server(fail=True)
    try:
        raw = _post_raw(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        payloads = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        assert payloads[-1] == "[DONE]"
        errors = [json.loads(item) for item in payloads if item != "[DONE]" and "error" in item]
        assert errors and errors[-1]["error"]["type"] == "unsupported_feature"
    finally:
        server.shutdown()
        server.server_close()


def test_chat_requests_carry_normalized_messages_to_the_backend():
    backend = ContractBackend()
    server, base = _server(backend=backend)
    try:
        _post(base, "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "tool_choice": "required",
            "reasoning_effort": "high",
        })
        request = backend.seen[-1]
        # The backend receives the normalized messages so it can apply its own
        # chat template rather than a flattened "role: content" string.
        assert request.metadata["messages"][0]["role"] == "system"
        assert request.metadata["messages"][0]["tools"] == [
            {"type": "function", "function": {"name": "weather"}}
        ]
        assert request.metadata["thinking_mode"] == "thinking"
        assert request.metadata["reasoning_effort"] == "high"
        assert "must call at least one available tool" in request.metadata["messages"][-1]["content"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_chat_uses_shared_request_builder():
    backend = ContractBackend()
    server, base = _server(backend=backend)
    try:
        _post(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "request_id": "http-chat-1",
            "max_completion_tokens": 9,
        })
        request = backend.seen[-1]
        assert request.request_id == "http-chat-1"
        assert request.prompt == "user: hi"
        assert request.sampling_params.max_tokens == 9
    finally:
        server.shutdown()
        server.server_close()



def test_completion_requests_keep_raw_prompt_path():
    backend = ContractBackend()
    server, base = _server(backend=backend)
    try:
        _post(base, "/v1/completions", {
            "prompt": "raw completion",
            "request_id": "completion-1",
            "max_tokens": 5,
        })
        request = backend.seen[-1]
        assert request.request_id == "completion-1"
        assert request.prompt == "raw completion"
        assert "messages" not in request.metadata
    finally:
        server.shutdown()
        server.server_close()



def test_http_and_protocol_builder_construct_equivalent_chat_requests():
    from pocketllm.api import SamplingParams
    from pocketllm.protocol import build_chat_request

    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": "required",
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
        "max_tokens": 9,
    }
    expected = build_chat_request(body, SamplingParams.from_openai(body), request_id="parity")
    backend = ContractBackend()
    server, base = _server(backend=backend)
    try:
        _post(base, "/v1/chat/completions", body)
        actual = backend.seen[-1]
        assert actual.prompt == expected.prompt
        assert actual.metadata == expected.metadata
        assert actual.sampling_params == expected.sampling_params
    finally:
        server.shutdown()
        server.server_close()




def test_invalid_chat_and_completion_bodies_are_rejected():
    server, base = _server()
    try:
        for path, body in (
            ("/v1/chat/completions", {"messages": []}),
            ("/v1/chat/completions", {"messages": ["hi"]}),
            ("/v1/completions", {"prompt": ""}),
            ("/v1/completions", {"prompt": [1]}),
        ):
            try:
                _post(base, path, body)
                raise AssertionError(f"expected an HTTP error for {path} {body}")
            except error.HTTPError as exc:
                assert exc.code == 400
                assert json.loads(exc.read().decode())["error"]["type"] == "invalid_request_error"
    finally:
        server.shutdown()
        server.server_close()


def test_reasoning_and_tool_calls_are_forwarded_in_responses():
    class ReasoningBackend(ContractBackend):
        def generate(self, requests):
            return [GenerationResult(
                request_id=req.request_id,
                token_ids=[11],
                text="answer",
                usage=Usage(1, 1),
                metadata={
                    "reasoning_content": "because",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "weather", "arguments": "{}"}}],
                },
            ) for req in requests]

        def stream(self, req):
            self._begin_request(req.request_id)
            try:
                yield TokenEvent(req.request_id, text="", metadata={"reasoning_content": "because"})
                yield TokenEvent(req.request_id, text="answer", token_id=11, finish_reason="stop",
                                 usage=Usage(1, 1))
            finally:
                self._clear_request(req.request_id)

    server, base = _server(backend=ReasoningBackend())
    try:
        chat = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        message = chat["choices"][0]["message"]
        assert message["reasoning_content"] == "because"
        assert message["tool_calls"][0]["function"]["name"] == "weather"

        raw = _post_raw(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}], "stream": True,
        })
        chunks = [json.loads(item) for item in
                  (line[len("data: "):] for line in raw.splitlines() if line.startswith("data: "))
                  if item != "[DONE]"]
        assert any(chunk["choices"][0]["delta"].get("reasoning_content") == "because" for chunk in chunks)
    finally:
        server.shutdown()
        server.server_close()


def test_cancelling_unknown_request_returns_404():
    server, base = _server()
    try:
        try:
            req = request.Request(base + "/v1/requests/does-not-exist", method="DELETE")
            request.urlopen(req, timeout=10)
            raise AssertionError("expected an HTTP error")
        except error.HTTPError as exc:
            assert exc.code == 404
            assert json.loads(exc.read().decode())["cancelled"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_metrics_exposition_is_a_prometheus_histogram():
    """Every declared family exposes `_bucket` series, not just `_sum`/`_count`.

    A `_count` and a `_sum` alone are not a histogram -- a scraper cannot take a
    quantile from them -- so the shape is asserted before any sample exists, the
    way the families are exported from process start.
    """
    text = Metrics().render()
    for family, (bounds, help_text) in HISTOGRAMS.items():
        assert f"# HELP pocketllm_{family} {help_text}\n" in text
        assert f"# TYPE pocketllm_{family} histogram\n" in text
        buckets = _buckets(text, family)
        # One series per finite bound plus the trailing +Inf slot.
        assert len(buckets) == len(bounds) + 1, family
        assert buckets[-1][0] == "+Inf"
        finite = [float(bound) for bound, _ in buckets[:-1]]
        assert finite == sorted(finite) == list(bounds), family
        counts = [count for _, count in buckets]
        assert all(a <= b for a, b in zip(counts, counts[1:])), family
        assert counts[-1] == _metric_value(text, f"{family}_count") == 0.0
        assert _metric_value(text, f"{family}_sum") == 0.0


def test_a_byte_gauge_is_exported_as_the_number_it_is():
    """A 4 GiB budget is nine digits, and the default six significant digits round it.

    The exposition is parsed by a float reader either way, so this is not a syntax question: it is
    that `4294967296` came out `4.29497e+09`, and a reader comparing occupancy against the budget
    would be comparing a rounded number with a rounded number.
    """
    metrics = Metrics()
    metrics.set("prefix_cache_budget_bytes", 4 << 30)
    metrics.set_counter("prefix_cache_reused_tokens_total", 5354)
    text = metrics.render()
    assert _metric_value(text, "prefix_cache_budget_bytes") == 4294967296.0
    assert "pocketllm_prefix_cache_budget_bytes 4294967296" in text
    # An integral value still prints without a decimal point, as it did before.
    assert "pocketllm_prefix_cache_reused_tokens_total 5354" in text


def test_metrics_counters_and_gauges_carry_their_prometheus_type():
    metrics = Metrics()
    metrics.inc("requests_total")
    metrics.set("build_info", 1)
    text = metrics.render()
    assert "# TYPE pocketllm_requests_total counter\n" in text
    assert "# TYPE pocketllm_build_info gauge\n" in text


def test_observed_samples_land_in_inclusive_cumulative_buckets():
    metrics = Metrics()
    for value in (0.05, 0.4, 3.0, 9000.0):
        metrics.observe("inter_token_latency_seconds", value)
    text = metrics.render()
    buckets = dict(_buckets(text, "inter_token_latency_seconds"))

    # A sample exactly on a bound belongs to that bound's bucket (`le` is
    # inclusive), and each series counts everything at or below it.
    assert buckets["0.05"] == 1
    assert buckets["0.1"] == 1
    assert buckets["0.5"] == 2
    assert buckets["5.0"] == 3
    # The 9000 s sample is above every finite bound, so only +Inf sees it.
    assert buckets["80.0"] == 3
    assert buckets["+Inf"] == 4
    assert _metric_value(text, "inter_token_latency_seconds_count") == 4.0
    assert _metric_value(text, "inter_token_latency_seconds_sum") == pytest.approx(9003.45)


def test_undeclared_histogram_is_rejected():
    """A histogram cannot be created on first use the way a counter can."""
    with pytest.raises(KeyError, match="undeclared histogram"):
        Metrics().observe("decoded_tokens_per_second", 1.0)


def test_streaming_records_ttft_itl_and_tpot():
    server, base = _server(backend=TokenCountBackend(4))
    try:
        _post_raw(base, "/v1/completions", {"prompt": "hi", "stream": True})
        text = _metrics(base)
        # The role delta the chat path writes first is not a token, so TTFT is
        # latched on the first event that carries one.
        assert _metric_value(text, "ttft_seconds_count") == 1.0
        assert _metric_value(text, "inter_token_latency_seconds_count") == 3.0
        assert _metric_value(text, "request_time_per_output_token_seconds_count") == 1.0
        assert _metric_value(text, "request_duration_seconds_count") == 1.0
        # The per-request mean times the number of intervals is the pooled
        # interval sum: the two families are derived from the same gaps.
        itl = _metric_value(text, "inter_token_latency_seconds_sum")
        tpot = _metric_value(text, "request_time_per_output_token_seconds_sum")
        assert tpot * 3 == pytest.approx(itl, rel=1e-6)
    finally:
        server.shutdown()
        server.server_close()


def test_single_token_stream_has_no_interval_to_average():
    server, base = _server(backend=TokenCountBackend(1))
    try:
        _post_raw(base, "/v1/completions", {"prompt": "hi", "stream": True})
        text = _metrics(base)
        assert _metric_value(text, "ttft_seconds_count") == 1.0
        assert _metric_value(text, "inter_token_latency_seconds_count") == 0.0
        # vLLM excludes `output_len <= 1` for the same reason: there is no
        # interval, and a zero would read as an instantaneous one.
        assert _metric_value(text, "request_time_per_output_token_seconds_count") == 0.0
    finally:
        server.shutdown()
        server.server_close()


def test_non_streaming_records_no_per_token_latency():
    server, base = _server()
    try:
        _post(base, "/v1/completions", {"prompt": "hi"})
        text = _metrics(base)
        assert _metric_value(text, "request_duration_seconds_count") == 1.0
        # The non-streaming path has no per-token boundary to observe; it
        # reports the end-to-end latency and invents nothing else.
        assert _metric_value(text, "ttft_seconds_count") == 0.0
        assert _metric_value(text, "inter_token_latency_seconds_count") == 0.0
        assert _metric_value(text, "request_time_per_output_token_seconds_count") == 0.0
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------- engine metrics


class MetricsBackend(ContractBackend):
    """An engine that holds numbers *between* requests and reports their absolute values.

    A prompt cache's occupancy and running hit count are the case: no request owns a share of them,
    so they cannot ride on a ``GenerationResult`` the way ``usage`` does.
    """

    def __init__(self, values=None, boom: bool = False):
        super().__init__()
        self.values = dict(values or {})
        self._boom = boom

    def metrics(self):
        if self._boom:
            raise RuntimeError("the engine cannot report")
        return dict(self.values)


def test_the_engines_own_values_reach_the_exposition():
    backend = MetricsBackend({
        "prefix_cache_reused_tokens_total": 64,
        "prefix_cache_entries": 2,
        "prefix_cache_bytes": 4096,
    })
    server, base = _server(backend=backend)
    try:
        text = _metrics(base)
        assert _metric_value(text, "prefix_cache_reused_tokens_total") == 64.0
        assert _metric_value(text, "prefix_cache_entries") == 2.0
        assert _metric_value(text, "prefix_cache_bytes") == 4096.0
        # The type follows the name: `_total` is Prometheus's suffix for a counter, and the engine
        # spells its counters that way so that nothing else has to be declared here.
        assert "# TYPE pocketllm_prefix_cache_reused_tokens_total counter\n" in text
        assert "# TYPE pocketllm_prefix_cache_entries gauge\n" in text
    finally:
        server.shutdown()
        server.server_close()


def test_a_backend_owned_counter_is_set_and_not_added():
    """The engine already keeps the running total; adding it again would square it per scrape."""
    backend = MetricsBackend({"requests_answered_total": 7})
    server, base = _server(backend=backend)
    try:
        assert _metric_value(_metrics(base), "requests_answered_total") == 7.0
        assert _metric_value(_metrics(base), "requests_answered_total") == 7.0

        # ... and a value that moves is read again on the next scrape, which is the whole reason
        # this is a pull at scrape time rather than a push from the request path.
        backend.values["requests_answered_total"] = 9
        assert _metric_value(_metrics(base), "requests_answered_total") == 9.0
    finally:
        server.shutdown()
        server.server_close()


def test_an_engine_that_cannot_report_leaves_the_scrape_standing():
    """A missing series is how a scraper reads "no data"; a 500 on /metrics is how it reads "down"."""
    server, base = _server(backend=MetricsBackend(boom=True))
    try:
        # Counted by the server rather than the engine, so the assertion is about the scrape standing
        # and not about which families exist yet -- a counter is created on first use.
        _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert _metric_value(_metrics(base), "requests_total") == 1.0
    finally:
        server.shutdown()
        server.server_close()


def test_a_backend_with_nothing_to_add_contributes_no_series():
    """``BackendBase.metrics`` is empty, and an empty mapping adds nothing to the exposition."""
    server, base = _server()
    try:
        assert "pocketllm_prefix_cache_entries" not in _metrics(base)
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------- reused prompt tokens


def test_a_reused_prompt_reports_its_cached_tokens():
    class CachingBackend(ContractBackend):
        def generate(self, requests):
            return [GenerationResult(
                request_id=req.request_id,
                token_ids=[11],
                text="ok",
                usage=Usage(6, 1, cached_tokens=5),
            ) for req in requests]

    server, base = _server(backend=CachingBackend())
    try:
        chat = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        # A subset of `prompt_tokens`, in OpenAI's own spelling and not a discount on it.
        assert chat["usage"]["prompt_tokens"] == 6
        assert chat["usage"]["prompt_tokens_details"] == {"cached_tokens": 5}
    finally:
        server.shutdown()
        server.server_close()


def test_a_cold_response_carries_no_prompt_token_details():
    """Emitted only when nonzero, so a backend that reuses nothing -- or does not report it -- keeps
    the response body byte-identical to the one this server returned before the field existed."""
    server, base = _server()
    try:
        chat = _post(base, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        assert "prompt_tokens_details" not in chat["usage"]
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------- decode tails


def test_a_half_character_at_the_end_of_a_decode_is_held_back():
    """A byte-level decode renders a character a token ended inside as U+FFFD until the next token
    finishes it. What a stream must not send is the replacement character, so the tail waits."""
    assert settled_text("你好！�") == "你好！"
    assert settled_text("你好！😊") == "你好！😊"
    assert settled_text("�") == ""
    assert settled_text("你好！��") == "你好！"


def test_a_replacement_character_inside_a_decode_is_left_where_it_is():
    """Only the tail can be a character still arriving. One anywhere else is a byte sequence that
    really was invalid, and the unstreamed decode has it in the same place -- dropping it would
    make the stream disagree with the answer it is a stream of."""
    assert settled_text("你�好") == "你�好"
    assert settled_text("a�b�") == "a�b"


def test_a_decode_with_nothing_to_settle_is_returned_unchanged():
    assert settled_text("") == ""
    assert settled_text("answer") == "answer"
    assert settled_text("line\n") == "line\n"
