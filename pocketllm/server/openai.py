"""Backend-neutral OpenAI-compatible HTTP server.

The protocol layer accepts any ``EngineBackend``.  Model loading remains the
responsibility of the chosen adapter, so this module can be tested with a fake
backend and can serve Torch or native C++ without duplicating JSON handling.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from pocketllm.api import (
    BackendUnavailableError,
    ConfigurationError,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    RequestCancelledError,
    TokenEvent,
    UnsupportedFeatureError,
)
from pocketllm.protocol import build_chat_request, build_completion_request

from .metrics import Metrics


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _openai_error(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type}}


def _result_response(result: GenerationResult, model: str, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    reasoning = result.metadata.get("reasoning_content")
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_calls = result.metadata.get("tool_calls")
    if tool_calls:
        message["tool_calls"] = tool_calls
    choice = {
        "index": 0,
        "message": message,
        "finish_reason": result.finish_reason,
    }
    if result.logprobs is not None:
        choice["logprobs"] = result.logprobs
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
        "usage": result.usage.as_dict(),
    }


def _completion_response(result: GenerationResult, model: str, request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": result.text,
            "index": 0,
            "logprobs": result.logprobs,
            "finish_reason": result.finish_reason,
        }],
        "usage": result.usage.as_dict(),
    }


def _event_json(event: TokenEvent, model: str, *, completion: bool = False) -> dict[str, Any]:
    if completion:
        # Text completions use their own chunk schema; a chat delta here would
        # break OpenAI-compatible clients that read `choices[].text`.
        item: dict[str, Any] = {
            "id": event.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "text": event.text,
                "logprobs": None,
                "finish_reason": event.finish_reason,
            }],
        }
        if event.usage is not None:
            item["usage"] = event.usage.as_dict()
        return item
    delta: dict[str, Any] = {}
    if event.text:
        delta["content"] = event.text
    if event.metadata.get("role"):
        delta["role"] = event.metadata["role"]
    # A backend that separates reasoning from content forwards both; a backend
    # that does not simply leaves these keys absent.
    if event.metadata.get("reasoning_content"):
        delta["reasoning_content"] = event.metadata["reasoning_content"]
    if event.metadata.get("tool_calls"):
        delta["tool_calls"] = event.metadata["tool_calls"]
    item = {
        "id": event.request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": event.finish_reason}],
    }
    if event.usage is not None:
        item["usage"] = event.usage.as_dict()
    return item


def _error_status(exc: BaseException) -> tuple[int, str]:
    """Map public exception types to HTTP status and OpenAI error type."""
    if isinstance(exc, RequestCancelledError):
        return 499, "request_cancelled"
    if isinstance(exc, UnsupportedFeatureError):
        return 400, "unsupported_feature"
    if isinstance(exc, ConfigurationError):
        return 400, "invalid_request_error"
    if isinstance(exc, BackendUnavailableError):
        return 503, "backend_unavailable"
    if isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
        return 400, "invalid_request_error"
    return 500, "server_error"


class PocketLLMHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler_class, backend: EngineBackend, model: str, metrics: Metrics | None = None):
        super().__init__(address, handler_class)
        self.backend = backend
        self.model = model
        self.metrics = metrics or Metrics()
        self.started_at = time.time()


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "PocketLLM/0.1"

    @property
    def pocket_server(self) -> PocketLLMHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, obj: Any) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        server = self.pocket_server
        if path in {"/health", "/alive"}:
            health = server.backend.health()
            status = 200 if health.alive else 503
            body = health.as_dict()
            if path == "/alive":
                body = {"alive": health.alive, "backend": health.backend}
            self._send_json(status, body)
            return
        if path == "/ready":
            health = server.backend.health()
            self._send_json(200 if health.ready else 503, health.as_dict())
            return
        if path == "/metrics":
            data = server.metrics.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": server.model, "object": "model", "owned_by": "local"}]})
            return
        self._send_json(404, _openai_error("not found"))

    def _request(self, body: dict[str, Any], *, completion: bool = False) -> GenerationRequest:
        request_id = str(body.get("request_id") or f"chatcmpl-{uuid.uuid4().hex}")
        if completion:
            return build_completion_request(body, request_id=request_id)
        return build_chat_request(body, request_id=request_id)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/v1/requests/"):
            self._send_json(404, _openai_error("not found"))
            return
        request_id = path.rsplit("/", 1)[-1]
        cancelled = self.pocket_server.backend.cancel(request_id)
        self._send_json(200 if cancelled else 404, {"id": request_id, "cancelled": cancelled})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/v1/chat/completions", "/v1/completions"}:
            self._send_json(404, _openai_error("not found"))
            return
        server = self.pocket_server
        started = time.perf_counter()
        server.metrics.inc("requests_total")
        server.metrics.add("requests_active", 1)
        try:
            completion = path == "/v1/completions"
            body = self._read_body()
            request = self._request(body, completion=completion)
            if bool(body.get("stream", False)):
                self._stream(request, completion=completion)
                return
            result = server.backend.generate([request])[0]
            server.metrics.inc("generation_tokens_total", result.usage.completion_tokens)
            server.metrics.inc("prompt_tokens_total", result.usage.prompt_tokens)
            response = (_completion_response if completion else _result_response)(
                result, server.model, request.request_id
            )
            self._send_json(200, response)
        except Exception as exc:
            status, error_type = _error_status(exc)
            server.metrics.inc("request_errors_total")
            self._send_json(status, _openai_error(str(exc), error_type))
        finally:
            server.metrics.add("requests_active", -1)
            server.metrics.observe("request_duration_seconds", time.perf_counter() - started)

    def _stream(self, request: GenerationRequest, *, completion: bool = False) -> None:
        server = self.pocket_server
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if not completion:
            role = TokenEvent(request.request_id, metadata={"role": "assistant"})
            self.wfile.write(b"data: " + _json_bytes(_event_json(role, server.model)) + b"\n\n")
            self.wfile.flush()
        prompt_counted = False
        try:
            for event in server.backend.stream(request):
                if event.token_id is not None:
                    server.metrics.inc("generation_tokens_total")
                if event.usage is not None and not prompt_counted:
                    server.metrics.inc("prompt_tokens_total", event.usage.prompt_tokens)
                    prompt_counted = True
                payload = _event_json(event, server.model, completion=completion)
                self.wfile.write(b"data: " + _json_bytes(payload) + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The client is gone; cancel at the next safe generation boundary.
            server.metrics.inc("cancellations_total")
            server.backend.cancel(request.request_id)
        except Exception as exc:
            # Headers are already sent, so report backend failures in-band.
            _, error_type = _error_status(exc)
            server.metrics.inc("request_errors_total")
            try:
                self.wfile.write(b"data: " + _json_bytes(_openai_error(str(exc), error_type)) + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
        finally:
            self.close_connection = True


def serve(backend: EngineBackend, *, host: str = "0.0.0.0", port: int = 8000, model: str = "local", metrics: Metrics | None = None) -> None:
    """Run the unified HTTP server until interrupted."""
    server = PocketLLMHTTPServer((host, port), OpenAIHandler, backend, model, metrics)
    try:
        server.serve_forever()
    finally:
        backend.close()
        server.server_close()
