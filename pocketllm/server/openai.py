"""Backend-neutral OpenAI-compatible HTTP server.

The protocol layer accepts any ``EngineBackend``.  Model loading remains the
responsibility of the chosen adapter, so this module can be tested with a fake
backend and can serve Torch or native C++ without duplicating JSON handling.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
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
from pocketllm.choices import expanded, folded_usage, streamed
from pocketllm.protocol import build_chat_request, build_completion_request
from pocketllm.protocol.contract import CHAT, COMPLETIONS, FieldRefusal, audit_shape

from .metrics import Metrics


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _openai_error(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type}}


def _field_refusal_error(refusal: FieldRefusal) -> dict[str, Any]:
    """The 400 body for a field this server will not serve.

    ``param`` is what makes the refusal machine-readable, and it is the reason the field's name
    travels beside its message rather than being parsed back out of it: a client that can see which
    parameter to change does not have to read English to act. The same spelling OpenAI uses for the
    errors its own validators raise.
    """
    return {
        "error": {
            "message": refusal.message,
            "type": "invalid_request_error",
            "param": refusal.field,
            "code": "unsupported_feature",
        }
    }


def _chat_choice(index: int, result: GenerationResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    reasoning = result.metadata.get("reasoning_content")
    if reasoning:
        message["reasoning_content"] = reasoning
    tool_calls = result.metadata.get("tool_calls")
    if tool_calls:
        message["tool_calls"] = tool_calls
    choice = {
        "index": index,
        "message": message,
        "finish_reason": result.finish_reason,
    }
    if result.logprobs is not None:
        choice["logprobs"] = result.logprobs
    return choice


def _completion_choice(index: int, result: GenerationResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "index": index,
        # Always present on this endpoint, and null when none was asked for, which is the shape
        # OpenAI's own completions response has: a client reading the key unconditionally is not
        # making a mistake. Chat instead omits it, because its spec says so.
        "logprobs": result.logprobs,
        "finish_reason": result.finish_reason,
    }


def _result_response(
    results: Sequence[GenerationResult],
    model: str,
    request_id: str,
    *,
    completion: bool = False,
) -> dict[str, Any]:
    """One response for a request, from the one or more runs that answered it.

    A request for ``n`` choices is ``n`` runs, so this takes them all and writes them out in index
    order. The count is whatever the dispatch produced rather than the ``n`` that was asked for:
    where a runtime answers fewer, the response says so by having fewer choices, which is the one
    way a client can notice -- an error would lose the choices that did succeed.

    ``usage`` is the group's, not a choice's. See :func:`pocketllm.choices.folded_usage` for why the
    prompt is counted once and the completion as the sum.
    """
    build = _completion_choice if completion else _chat_choice
    return {
        "id": request_id,
        "object": "text_completion" if completion else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [build(index, result) for index, result in enumerate(results)],
        "usage": folded_usage(results).as_dict(),
    }


def _event_json(
    event: TokenEvent, model: str, *, completion: bool = False
) -> dict[str, Any]:
    if completion:
        # Text completions use their own chunk schema; a chat delta here would
        # break OpenAI-compatible clients that read `choices[].text`.
        item: dict[str, Any] = {
            "id": event.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": event.choice_index,
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
        "choices": [
            {"index": event.choice_index, "delta": delta, "finish_reason": event.finish_reason}
        ],
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


def _publish_backend_metrics(server: "PocketLLMHTTPServer") -> None:
    """Fold the backend's own metric values into the exporter, before rendering.

    Some numbers belong to the engine rather than to the request path -- a prompt cache's occupancy,
    its hit count -- and the HTTP layer never sees them: a request reports what it *used*, not what
    the engine is holding. ``BackendBase.metrics`` is the channel, and it hands over absolute values
    from the engine's own running totals, so they are set rather than added.

    The type comes from the name: ``_total`` is Prometheus's suffix for a counter and everything else
    is a gauge. That convention is the whole dispatch, and it is why the engine spells its counters
    with the suffix.
    """
    collector = getattr(server.backend, "metrics", None)
    if collector is None:
        return
    try:
        values = collector()
    except Exception:
        # A metrics scrape must not fail a request path or a scrape itself: an engine that cannot
        # report is an engine whose series are absent, which a scraper reads as no data.
        return
    for name, value in (values or {}).items():
        if str(name).endswith("_total"):
            server.metrics.set_counter(str(name), float(value))
        else:
            server.metrics.set(str(name), float(value))


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

    def _write_event(
        self, event: TokenEvent, model: str, *, completion: bool = False
    ) -> None:
        """One server-sent event, flushed.

        Written and flushed per event rather than buffered, because a stream a client cannot read
        until it ends is not a stream: the point of the transport is that the first token arrives
        when the model produced it.
        """
        payload = _event_json(event, model, completion=completion)
        self.wfile.write(b"data: " + _json_bytes(payload) + b"\n\n")
        self.wfile.flush()

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
            _publish_backend_metrics(server)
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
            endpoint = COMPLETIONS if completion else CHAT
            # Two audits, and the split is deliberate. Shape is the host's: `"n": 2.5` is not a
            # number of choices on any runtime, and the check is the same cheap JSON inspection
            # wherever the request lands. Capability is the backend's, because whether the answer
            # applies a field depends on the runtime and, for the C++ one, on the engine under it.
            # Both run before dispatch, so a request this server will not serve is refused with a
            # 400 naming the field rather than streamed halfway and then abandoned.
            refusal = audit_shape(body, endpoint=endpoint)
            if refusal is None:
                refusal = server.backend.audit_request(body, endpoint=endpoint)
            if refusal is not None:
                server.metrics.inc("request_errors_total")
                self._send_json(400, _field_refusal_error(refusal))
                return
            request = self._request(body, completion=completion)
            if bool(body.get("stream", False)):
                self._stream(request, started=started, completion=completion)
                return
            # A request for `n` choices is `n` requests to the runtime, run here rather than in an
            # adapter: it is the same fan-out whatever is underneath, and the response is what needs
            # to know about it. See `pocketllm.choices`.
            results = server.backend.generate(expanded(request))
            for result in results:
                server.metrics.inc("generation_tokens_total", result.usage.completion_tokens)
            server.metrics.inc("prompt_tokens_total", folded_usage(results).prompt_tokens)
            response = _result_response(
                results, server.model, request.request_id, completion=completion
            )
            self._send_json(200, response)
        except Exception as exc:
            status, error_type = _error_status(exc)
            server.metrics.inc("request_errors_total")
            self._send_json(status, _openai_error(str(exc), error_type))
        finally:
            server.metrics.add("requests_active", -1)
            server.metrics.observe("request_duration_seconds", time.perf_counter() - started)

    def _stream(self, request: GenerationRequest, *, started: float, completion: bool = False) -> None:
        server = self.pocket_server
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        prompt_counted = False
        # Latency sampling. A role delta is not a token, so TTFT is latched on the
        # first event that actually carries one -- the same signal the
        # generation-token counter already keys off.
        tokens_seen = 0
        first_token_at: float | None = None
        last_token_at = 0.0
        inter_token_total = 0.0
        #: Which choices have been opened with their role delta. A set rather than a boolean, because
        #: a request for several choices is several streams in one response and each of them opens
        #: the way a single-choice stream does.
        greeted: set[int] = set()
        if not completion:
            # Choice 0's opener goes out before generation starts, which is what this server has
            # always done and is worth keeping: a client sees the assistant turn begin as soon as the
            # response headers are on the wire, rather than at the first token, which on a long
            # prefill is seconds later. The choices after it open at their own first event, because
            # they do not begin until the one before them has finished.
            greeted.add(0)
            self._write_event(
                TokenEvent(request.request_id, metadata={"role": "assistant"}), server.model
            )
        try:
            for event in streamed(server.backend, request):
                # A chat stream opens each choice with its role: a client accumulating per `index`
                # sees an assistant delta arrive before that choice's first content, and does not
                # have to infer from a content-only delta that a new choice started.
                if not completion and event.choice_index not in greeted:
                    greeted.add(event.choice_index)
                    self._write_event(
                        TokenEvent(
                            request.request_id,
                            choice_index=event.choice_index,
                            metadata={"role": "assistant"},
                        ),
                        server.model,
                    )
                if event.token_id is not None:
                    server.metrics.inc("generation_tokens_total")
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                        server.metrics.observe("ttft_seconds", now - started)
                    else:
                        gap = now - last_token_at
                        inter_token_total += gap
                        server.metrics.observe("inter_token_latency_seconds", gap)
                    last_token_at = now
                    tokens_seen += 1
                if event.usage is not None and not prompt_counted:
                    # Once for the request, not once per choice: the prompt was sent once and the
                    # engine prefilled it `n` times, which is the engine's cost to pay and not the
                    # client's to be billed for. `folded_usage` is the same rule on the response.
                    server.metrics.inc("prompt_tokens_total", event.usage.prompt_tokens)
                    prompt_counted = True
                self._write_event(event, server.model, completion=completion)
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
            # Recorded on every terminal path, including a truncated stream: the
            # tokens that were generated took the time they took. A single-token
            # response has no interval to average and contributes nothing, the
            # same way vLLM excludes `output_len <= 1`.
            if tokens_seen >= 2:
                server.metrics.observe(
                    "request_time_per_output_token_seconds",
                    inter_token_total / (tokens_seen - 1),
                )
            self.close_connection = True


def serve(
    backend: EngineBackend,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str = "local",
    metrics: Metrics | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Run the unified HTTP server until interrupted.

    ``on_ready`` runs after the listening socket has been bound and before the
    request loop starts. It is used by the local TP supervisor; ordinary callers
    can leave it unset.
    """
    server = PocketLLMHTTPServer((host, port), OpenAIHandler, backend, model, metrics)
    try:
        if on_ready is not None:
            on_ready()
        server.serve_forever()
    finally:
        backend.close()
        server.server_close()
