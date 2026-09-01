from __future__ import annotations

import json
import threading
from urllib import error, request

from pocketllm.api import (
    BackendCapabilities,
    GenerationResult,
    TokenEvent,
    UnsupportedFeatureError,
    Usage,
)
from pocketllm.backends.base import BackendBase
from pocketllm.server.openai import OpenAIHandler, PocketLLMHTTPServer


class ContractBackend(BackendBase):
    def __init__(self, fail: bool = False):
        super().__init__()
        self._ready = True
        self._fail = fail

    @property
    def capabilities(self):
        return BackendCapabilities(name="fake", supports_streaming=True, supports_cancellation=True)

    def generate(self, requests):
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


def _server(fail: bool = False):
    backend = ContractBackend(fail=fail)
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


def _metrics(base: str) -> str:
    with request.urlopen(base + "/metrics", timeout=10) as response:
        return response.read().decode()


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
