#!/usr/bin/env python3
"""Verify the native Qwen C++ OpenAI-compatible server with real weights.

This is an opt-in integration check rather than a unit test. It starts one native
process per tensor-parallel rank, exercises the HTTP surface, and tears the group
down on every exit path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


class HttpResult:
    def __init__(self, status: int, body: bytes, headers: Any = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> dict[str, Any]:
        value = json.loads(self.text)
        if not isinstance(value, dict):
            raise AssertionError(f"expected JSON object, got {type(value).__name__}")
        return value


def http_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
    stream: bool = False,
) -> HttpResult:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Reading the complete body also makes the server observe the client
            # reaching a safe completion boundary before the next request.
            return HttpResult(response.status, response.read(), response.headers)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), exc.headers)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")
    return devices


def wait_for_health(base_url: str, processes: list[subprocess.Popen[bytes]], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            exited = [process.returncode for process in processes]
            raise RuntimeError(f"a TP rank exited before readiness: {exited}")
        try:
            result = http_request(base_url, "/health", timeout=2.0)
            if result.status == 200 and result.json().get("status") == "ok":
                return
            last_error = f"HTTP {result.status}: {result.text}"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def read_logs(log_dir: pathlib.Path) -> str:
    chunks: list[str] = []
    for path in sorted(log_dir.glob("rank*.log")):
        chunks.append(f"--- {path.name}\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(chunks)


def validate_nonstream(result: HttpResult, expected_model: str) -> dict[str, Any]:
    require(result.status == 200, f"non-stream request failed: HTTP {result.status}: {result.text}")
    body = result.json()
    require(body.get("object") == "chat.completion", "wrong non-stream object")
    require(body.get("model") == expected_model, "wrong non-stream model")
    choices = body.get("choices")
    require(isinstance(choices, list) and choices, "non-stream response has no choices")
    choice = choices[0]
    require(isinstance(choice, dict), "non-stream choice is not an object")
    require(choice.get("message", {}).get("role") == "assistant", "missing assistant role")
    require(isinstance(choice.get("message", {}).get("content"), str), "missing assistant content")
    require(choice["message"]["content"] != "", "assistant content is empty")
    require(choice.get("finish_reason") in {"stop", "length"}, "invalid non-stream finish reason")
    usage = body.get("usage")
    require(isinstance(usage, dict), "non-stream response has no usage")
    require(usage.get("prompt_tokens", 0) > 0, "prompt token count is not positive")
    require(usage.get("completion_tokens", 0) > 0, "completion token count is not positive")
    require(
        usage.get("total_tokens") == usage.get("prompt_tokens") + usage.get("completion_tokens"),
        "usage total does not equal prompt plus completion",
    )
    return body


def validate_stream(result: HttpResult, expected_model: str) -> list[dict[str, Any]]:
    require(result.status == 200, f"stream request failed: HTTP {result.status}: {result.text}")
    events: list[dict[str, Any]] = []
    saw_done = False
    for line in result.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            saw_done = True
            continue
        event = json.loads(payload)
        require(event.get("object") == "chat.completion.chunk", "wrong stream object")
        require(event.get("model") == expected_model, "wrong stream model")
        choices = event.get("choices")
        require(isinstance(choices, list) and choices, "stream event has no choices")
        events.append(event)
    require(saw_done, "stream did not terminate with [DONE]")
    require(events, "stream contained no JSON events")
    require(events[0]["choices"][0]["delta"].get("role") == "assistant", "stream lacks role event")
    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if isinstance(event.get("choices"), list)
    )
    require(content != "", "stream produced no content delta")
    terminal = events[-1]["choices"][0]
    require(terminal.get("finish_reason") in {"stop", "length"}, "stream lacks terminal finish reason")
    return events


def chat_payload(text: str, max_tokens: int, *, stream: bool = False) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "stream": stream,
    }


# Request fields the server does not implement. The audit refuses each one with a
# 400 naming the field in OpenAI's `param` slot when the value would have changed
# the output, so a caller is never handed a response generated as if the field
# were at its default. Every case here is refused before tokenization, so the
# whole check costs no generation.
#
# `stop` is implemented, so it is here only at a value of the wrong shape: a
# number is not a stop sequence, and accepting it would leave a request that
# never stops looking configured.
REFUSED_CHAT_FIELDS: tuple[tuple[str, Any], ...] = (
    ("n", 3),
    ("stop", 5),
    ("logprobs", True),
    ("top_logprobs", 5),
    ("frequency_penalty", 1.5),
    ("presence_penalty", -1.0),
    ("logit_bias", {"100": -100}),
    ("tool_choice", "required"),
    ("parallel_tool_calls", False),
)

# The same fields at the value that names what the server already does. These
# must be accepted: an SDK that sends the documented default explicitly is not
# asking for anything.
ACCEPTED_CHAT_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("n", 1),
    ("stop", []),
    ("logprobs", False),
    ("top_logprobs", 0),
    ("frequency_penalty", 0),
    ("presence_penalty", 0),
    ("logit_bias", {}),
    ("tool_choice", "auto"),
    ("parallel_tool_calls", True),
)


def refusal_error(result: HttpResult, label: str) -> dict[str, Any]:
    """Asserts the 400 carries the OpenAI error shape with a populated `param`."""
    require(result.status == 400, f"{label} was not refused: HTTP {result.status}: {result.text}")
    error = result.json().get("error")
    require(isinstance(error, dict), f"{label} refusal is not in the OpenAI error shape")
    require(error.get("type") == "invalid_request_error", f"{label} refusal has the wrong type")
    require(error.get("message"), f"{label} refusal has no message")
    return error


def validate_request_field_refusals(base_url: str, model_name: str, timeout: float) -> None:
    """Checks the request-field contract on both OpenAI endpoints."""
    messages = [{"role": "user", "content": "This request is inspected, not generated."}]
    base = {"messages": messages, "max_tokens": 1}

    for field, value in REFUSED_CHAT_FIELDS:
        result = http_request(
            base_url, "/v1/chat/completions", {**base, field: value}, timeout=timeout
        )
        error = refusal_error(result, f"chat {field}={value!r}")
        require(
            error.get("param") == field,
            f"{field} refusal named {error.get('param')!r} instead of {field!r}",
        )

    for field, value in ACCEPTED_CHAT_DEFAULTS:
        result = http_request(
            base_url, "/v1/chat/completions", {**base, field: value}, timeout=timeout
        )
        require(
            result.status == 200,
            f"chat {field}={value!r} should be accepted, got HTTP {result.status}: {result.text}",
        )

    streaming = http_request(
        base_url,
        "/v1/chat/completions",
        {**base, "stream": True, "stream_options": {"include_usage": True}},
        timeout=timeout,
    )
    error = refusal_error(streaming, "chat stream_options.include_usage")
    require(
        error.get("param") == "stream_options.include_usage",
        f"include_usage refusal named {error.get('param')!r}",
    )

    # Fields that only exist on /v1/completions are refused there, and ignored as
    # unknown parameters on chat rather than being read as something else.
    for field, value in (("echo", True), ("best_of", 2), ("suffix", " END")):
        result = http_request(
            base_url,
            "/v1/completions",
            {"prompt": "This prompt is inspected, not generated.", "max_tokens": 1, field: value},
            timeout=timeout,
        )
        error = refusal_error(result, f"completions {field}={value!r}")
        require(error.get("param") == field, f"{field} refusal named {error.get('param')!r}")

    # "max_completion_tokens" supersedes the deprecated "max_tokens": a request
    # carrying 32 and 1 must generate one token, not 32.
    precedence = http_request(
        base_url,
        "/v1/chat/completions",
        {"messages": messages, "max_tokens": 32, "max_completion_tokens": 1,
         "temperature": 0.0},
        timeout=timeout,
    )
    body = validate_nonstream(precedence, model_name)
    require(
        body["usage"]["completion_tokens"] == 1,
        "max_completion_tokens did not take precedence over max_tokens: "
        f"{body['usage']['completion_tokens']} tokens generated",
    )

    print("[PASS] request field refusals: all undocumented fields rejected with a named param")


def sse_events(result: HttpResult) -> list[dict[str, Any]]:
    """The JSON events of an SSE response, `[DONE]` excluded."""
    require(result.status == 200, f"stream request failed: HTTP {result.status}: {result.text}")
    events: list[dict[str, Any]] = []
    saw_done = False
    for line in result.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            saw_done = True
            continue
        events.append(json.loads(payload))
    require(saw_done, "stream did not terminate with [DONE]")
    return events


def answer_and_finish(
    result: HttpResult, chat: bool, stream: bool, expected_model: str
) -> tuple[str, str]:
    """The answer text and finish reason of a response, in any of its four shapes."""
    require(result.status == 200, f"request failed: HTTP {result.status}: {result.text}")
    if not stream:
        body = result.json()
        require(body.get("model") == expected_model, "wrong non-stream model")
        choice = body["choices"][0]
        text = choice["message"]["content"] if chat else choice["text"]
        return text, choice["finish_reason"]
    text = ""
    finish_reason = ""
    for event in sse_events(result):
        require(event.get("model") == expected_model, "wrong stream model")
        choice = event["choices"][0]
        # The two endpoints do not share a chunk shape. A chat chunk wraps the
        # incremental text in `delta`; a completions chunk carries `text` on the
        # choice itself, which is what OpenAI's own /v1/completions stream does.
        delta = choice.get("delta") or {}
        text += delta.get("content", "") if chat else choice.get("text", "")
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
    return text, finish_reason


def validate_stop_sequences(base_url: str, model_name: str, timeout: float) -> None:
    """Checks that a client stop sequence really truncates the answer.

    The sequence is lifted out of a first, unconstrained answer to the same
    greedy request, so the model has every reason to produce it again and the
    expected truncation is derived from the reference rather than guessed. The
    server cuts at the *first* occurrence, which is what `str.index` reports.
    """
    prompt = "List the numbers from 1 to 40, separated by commas."
    max_tokens = 64
    unmatched = "<|not-a-stop-sequence|>"

    for chat in (True, False):
        endpoint = "/v1/chat/completions" if chat else "/v1/completions"
        base: dict[str, Any] = (
            chat_payload(prompt, max_tokens)
            if chat
            else {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
            }
        )

        reference, _ = answer_and_finish(
            http_request(base_url, endpoint, base, timeout=timeout), chat, False, model_name
        )
        require(
            len(reference) >= 16,
            f"reference answer is too short to truncate: {reference!r}",
        )
        middle = len(reference) // 2
        sequence = reference[middle : middle + 4]
        expected = reference[: reference.index(sequence)]
        require(expected != "", f"stop sequence chosen at the start of the answer: {sequence!r}")

        for stream in (False, True):
            mode = "stream" if stream else "non-stream"
            label = f"{'chat' if chat else 'completions'} {mode}"

            # The documented spellings: a list of sequences, and the bare string
            # that means a list of one.
            spellings: tuple[tuple[str, Any], ...] = (
                ("stop", [sequence]),
                ("stop", [unmatched, sequence]),
            )
            if not stream:
                spellings += (("stop", sequence),)

            for field, value in spellings:
                result = http_request(
                    base_url, endpoint, {**base, "stream": stream, field: value}, timeout=timeout
                )
                text, finish_reason = answer_and_finish(result, chat, stream, model_name)
                require(
                    text == expected,
                    f"{label} {field}={value!r} returned {text!r}, expected the answer cut "
                    f"before {sequence!r}: {expected!r}",
                )
                require(sequence not in text, f"{label} leaked the stop sequence into the answer")
                require(
                    finish_reason == "stop",
                    f"{label} {field}={value!r} reported finish_reason {finish_reason!r}",
                )

            # A sequence that never occurs must not truncate anything. On a
            # stream this is also the regression check for the holdback: the
            # scan withholds a trailing partial sequence, and it must give every
            # byte back rather than dropping one.
            result = http_request(
                base_url, endpoint, {**base, "stream": stream, "stop": [unmatched]}, timeout=timeout
            )
            text, _ = answer_and_finish(result, chat, stream, model_name)
            require(
                text == reference,
                f"{label} with an unmatched stop sequence returned {text!r}, expected {reference!r}",
            )

            # An empty list is what an SDK sends when nothing is configured, and
            # the empty strings some clients pad it with must not match at
            # position 0 and truncate the answer to nothing.
            result = http_request(
                base_url, endpoint, {**base, "stream": stream, "stop": []}, timeout=timeout
            )
            text, _ = answer_and_finish(result, chat, stream, model_name)
            require(
                text == reference,
                f"{label} with an empty stop list returned {text!r}, expected {reference!r}",
            )

    print("[PASS] stop sequences: the answer is cut at the first matching sequence")


def run(args: argparse.Namespace) -> int:
    checkpoint = pathlib.Path(args.ckpt).resolve()
    binary = pathlib.Path(args.binary).resolve()
    sidecar = pathlib.Path(args.sidecar).resolve()
    python_bin = pathlib.Path(args.python).resolve()
    devices = parse_devices(args.devices)
    tp_world = len(devices)

    for path, label in (
        (checkpoint, "checkpoint"),
        (binary, "native binary"),
        (sidecar, "sidecar script"),
        (python_bin, "Python executable"),
    ):
        require(path.exists(), f"{label} does not exist: {path}")
    require((checkpoint / "config.json").exists(), "checkpoint has no config.json")
    require((checkpoint / "tokenizer.json").exists(), "checkpoint has no tokenizer.json")

    log_dir: pathlib.Path
    temporary_log_dir: str | None = None
    if args.log_dir:
        log_dir = pathlib.Path(args.log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary_log_dir = tempfile.mkdtemp(prefix="pocketllm-qwen-openai-")
        log_dir = pathlib.Path(temporary_log_dir)

    rendezvous = log_dir / f"nccl-{uuid.uuid4().hex}.id"
    processes: list[subprocess.Popen[bytes]] = []
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        common = [
            str(binary),
            "--serve",
            "--ckpt",
            str(checkpoint),
            "--tp-world",
            str(tp_world),
            "--nccl-id-path",
            str(rendezvous),
            "--smoke-layers",
            str(args.layers),
            "--max-context",
            str(args.max_context),
            "--max-batch-size",
            str(args.max_batch_size),
            "--prefill-token-budget",
            str(args.prefill_token_budget),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--python",
            str(python_bin),
            "--sidecar",
            str(sidecar),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--kv-block-size",
            str(args.kv_block_size),
        ]
        if args.prefill_chunk_tokens > 0:
            common.extend(["--prefill-chunk-tokens", str(args.prefill_chunk_tokens)])
        if args.kv_paged:
            common.append("--kv-paged")

        for rank, visible_device in enumerate(devices):
            log = (log_dir / f"rank{rank}.log").open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = visible_device
            process = subprocess.Popen(
                common + ["--tp-rank", str(rank), "--device", "0"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append(process)
            log.close()

        wait_for_health(base_url, processes, args.startup_timeout)

        health = http_request(base_url, "/health", timeout=5.0)
        require(health.status == 200 and health.json().get("status") == "ok", "health check failed")

        models = http_request(base_url, "/v1/models", timeout=5.0)
        require(models.status == 200, f"models endpoint failed: {models.text}")
        model_body = models.json()
        model_data = model_body.get("data")
        require(isinstance(model_data, list) and model_data, "models endpoint has no data")
        model_name = model_data[0].get("id")
        require(model_name == "qwen3_5", f"unexpected served model: {model_name!r}")

        nonstream = http_request(
            base_url,
            "/v1/chat/completions",
            chat_payload("Reply with exactly OK.", args.max_tokens),
            timeout=args.request_timeout_seconds + 30,
        )
        validate_nonstream(nonstream, model_name)

        stream = http_request(
            base_url,
            "/v1/chat/completions",
            chat_payload("Reply with exactly OK.", args.max_tokens, stream=True),
            timeout=args.request_timeout_seconds + 30,
        )
        validate_stream(stream, model_name)

        validate_request_field_refusals(base_url, model_name, timeout=10.0)

        validate_stop_sequences(base_url, model_name, timeout=args.request_timeout_seconds + 30)

        incompatible = http_request(
            base_url,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "This must be rejected."}],
                "max_tokens": 1,
                "temperature": 0.5,
            },
            timeout=10.0,
        )
        require(incompatible.status == 400, "fixed-sampling mismatch was not rejected")
        require("effective temperature" in incompatible.text, "sampling error did not explain the fixed temperature")

        def concurrent_request(index: int) -> dict[str, Any]:
            result = http_request(
                base_url,
                "/v1/chat/completions",
                chat_payload(f"Reply briefly to request {index}.", min(args.max_tokens, 2)),
                timeout=args.request_timeout_seconds + 30,
            )
            return validate_nonstream(result, model_name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_results = list(pool.map(concurrent_request, (0, 1)))
        require(len(concurrent_results) == 2, "concurrent request count mismatch")

        logs = read_logs(log_dir)
        expected_width = min(args.max_batch_size, 2)
        require(
            f"[server] batch width {expected_width}" in logs,
            "rank-0 log did not report the requested scheduler width",
        )
        require(
            f"prefill budget {args.prefill_token_budget}" in logs,
            "rank-0 log did not report the configured prefill budget",
        )
        print(
            f"[PASS] native Qwen OpenAI serving: tp={tp_world} model={model_name} "
            f"log_dir={log_dir} concurrent_requests={len(concurrent_results)}"
        )
        return 0
    except Exception:
        print("[FAIL] native Qwen OpenAI serving", file=sys.stderr)
        print(read_logs(log_dir), file=sys.stderr)
        raise
    finally:
        terminate_processes(processes)
        try:
            rendezvous.unlink()
        except FileNotFoundError:
            pass
        if temporary_log_dir is not None:
            shutil.rmtree(temporary_log_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Qwen Safetensors checkpoint directory")
    parser.add_argument("--binary", default="build/cpp_engine/pocketllm_engine")
    parser.add_argument("--python", default=sys.executable, help="Python with transformers installed")
    parser.add_argument("--sidecar", default="src/server/cpp_sidecar.py")
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA device IDs")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--layers", type=int, default=0, help="0 means complete Qwen depth")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--prefill-token-budget", type=int, default=4096)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-paged", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--log-dir", default=None, help="Keep rank logs in this directory")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
