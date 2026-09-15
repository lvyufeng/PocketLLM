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
import math
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
# `n` and `stop` are implemented, so they are here only at a value that cannot be
# served: a count below one or past the server's ceiling is not a number of
# choices, a number is not a stop sequence, and accepting either would leave a
# request that looks configured and is not.
REFUSED_CHAT_FIELDS: tuple[tuple[str, Any], ...] = (
    ("n", 0),
    ("n", 129),
    ("stop", 5),
    ("logprobs", 1),
    ("logprobs", "true"),
    ("top_logprobs", 5),
    ("top_logprobs", "5"),
    ("frequency_penalty", 1.5),
    ("presence_penalty", -1.0),
    ("logit_bias", {"100": -100}),
    ("tool_choice", "required"),
    ("tool_choice", "none"),
    ("tool_choice", {"type": "function", "function": {"name": "get_weather"}}),
    ("parallel_tool_calls", False),
)

# The same fields at the value that names what the server already does. These
# must be accepted: an SDK that sends the documented default explicitly is not
# asking for anything.
#
# `logprobs = true` and a non-zero `top_logprobs` are accepted too, but they are
# checked by validate_logprobs instead of here: accepting them is only half the
# contract, and the other half is that the response actually carries a ranking.
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

# /v1/completions spells the same fields differently: "logprobs" is a count, where
# even 0 asks for the sampled token's own probability, and "top_logprobs" does not
# exist. A boolean is the chat spelling of the field and a count past the server's
# ceiling is a request it cannot serve.
REFUSED_COMPLETIONS_FIELDS: tuple[tuple[str, Any], ...] = (
    ("logprobs", True),
    ("logprobs", -1),
    ("logprobs", 2.5),
    ("logprobs", 21),
    ("best_of", 2),
    ("echo", True),
    ("suffix", " END"),
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

    # Fields that only exist on /v1/completions, plus that endpoint's spelling of
    # the log-probability fields, are refused there.
    for field, value in REFUSED_COMPLETIONS_FIELDS:
        result = http_request(
            base_url,
            "/v1/completions",
            {"prompt": "This prompt is inspected, not generated.", "max_tokens": 1, field: value},
            timeout=timeout,
        )
        error = refusal_error(result, f"completions {field}={value!r}")
        require(error.get("param") == field, f"{field} refusal named {error.get('param')!r}")

    # A streamed chunk carries the text of its token and no ranking beside it, so
    # asking for log probabilities on the streaming path is refused rather than
    # answered with a stream that looks the same as one that asked for none. On
    # /v1/completions 0 is a real request for the sampled token's probability, so
    # only chat has an inert spelling of the field here.
    for path, payload in (
        ("/v1/chat/completions", {**base, "logprobs": True}),
        ("/v1/chat/completions", {**base, "logprobs": True, "top_logprobs": 3}),
        (
            "/v1/completions",
            {"prompt": "This prompt is inspected, not generated.", "max_tokens": 1, "logprobs": 5},
        ),
        (
            "/v1/completions",
            {"prompt": "This prompt is inspected, not generated.", "max_tokens": 1, "logprobs": 0},
        ),
    ):
        result = http_request(base_url, path, {**payload, "stream": True}, timeout=timeout)
        error = refusal_error(result, f"{path} stream with logprobs")
        require(
            error.get("param") == "logprobs",
            f"streaming logprobs refusal named {error.get('param')!r}",
        )
        require(
            "stream" in error.get("message", ""),
            f"streaming logprobs refusal does not mention streaming: {error.get('message')!r}",
        )

    for path, payload in (
        ("/v1/chat/completions", {**base, "logprobs": False}),
        ("/v1/completions", {"prompt": "x", "max_tokens": 1}),
    ):
        result = http_request(base_url, path, {**payload, "stream": True}, timeout=timeout)
        require(
            result.status == 200,
            f"{path} stream without logprobs should be accepted, "
            f"got HTTP {result.status}: {result.text}",
        )

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


def choice_breakdown(
    result: HttpResult, chat: bool, stream: bool
) -> tuple[dict[int, str], dict[int, str], dict[str, Any] | None]:
    """Per-choice text and finish reason, plus the usage block when there is one.

    Unlike answer_and_finish this reads *every* entry of `choices`, which is what
    a multi-choice response is made of, and keys the result by the index the
    server reported rather than by position.
    """
    texts: dict[int, str] = {}
    finishes: dict[int, str] = {}
    if not stream:
        body = result.json()
        for choice in body["choices"]:
            index = choice["index"]
            texts[index] = choice["message"]["content"] if chat else choice["text"]
            finishes[index] = choice["finish_reason"]
        usage = body.get("usage")
        return texts, finishes, usage if isinstance(usage, dict) else None
    for event in sse_events(result):
        for choice in event["choices"]:
            index = choice["index"]
            delta = choice.get("delta") or {}
            piece = delta.get("content", "") if chat else choice.get("text", "")
            texts[index] = texts.get(index, "") + piece
            if choice.get("finish_reason") is not None:
                finishes[index] = choice["finish_reason"]
    return texts, finishes, None


def validate_n_choices(base_url: str, model_name: str, timeout: float) -> None:
    """Checks that "n" really produces n choices on both endpoints.

    The acceptance engine samples greedily at engine-wide values, which is the
    one configuration where n > 1 is served rather than refused, and that is what
    makes the expected answer knowable: every choice has to repeat the greedy
    answer a plain n = 1 request gives. Three choices that each match a separate
    reference generation is the difference between three real generations and
    one response with its index relabelled, and it is why the usage block is
    checked against a number derived from the reference rather than a constant.
    """
    prompt = "List the numbers from 1 to 20, separated by commas."
    max_tokens = 24
    n = 3
    indices = list(range(n))

    for chat in (True, False):
        endpoint = "/v1/chat/completions" if chat else "/v1/completions"
        greedy: dict[str, Any] = (
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

        reference = http_request(base_url, endpoint, greedy, timeout=timeout)
        reference_body = reference.json()
        reference_text, reference_finish = answer_and_finish(reference, chat, False, model_name)
        reference_usage = reference_body["usage"]
        require(len(reference_text) >= 8, f"reference answer is too short: {reference_text!r}")

        for stream in (False, True):
            label = f"{'chat' if chat else 'completions'} {'stream' if stream else 'non-stream'} n={n}"
            result = http_request(
                base_url, endpoint, {**greedy, "n": n, "stream": stream}, timeout=timeout
            )
            require(result.status == 200, f"{label} failed: HTTP {result.status}: {result.text}")
            texts, finishes, usage = choice_breakdown(result, chat, stream)
            require(
                sorted(texts) == indices,
                f"{label} returned choices {sorted(texts)}, expected {n} indexed 0..{n - 1}",
            )
            for index in indices:
                require(
                    texts[index] == reference_text,
                    f"{label} choice {index} returned {texts[index]!r}, expected the greedy "
                    f"answer {reference_text!r}",
                )
                require(
                    finishes.get(index) == reference_finish,
                    f"{label} choice {index} reported finish_reason {finishes.get(index)!r}, "
                    f"expected {reference_finish!r}",
                )
            if usage is None:
                continue
            # OpenAI counts the prompt once and the completion as the sum over
            # choices, which for n greedy copies of the same answer is n times
            # the single-choice count.
            require(
                usage.get("prompt_tokens") == reference_usage["prompt_tokens"],
                f"{label} counted {usage.get('prompt_tokens')} prompt tokens, expected the "
                f"reference's {reference_usage['prompt_tokens']} counted once",
            )
            expected_completion = reference_usage["completion_tokens"] * n
            require(
                usage.get("completion_tokens") == expected_completion,
                f"{label} reported {usage.get('completion_tokens')} completion tokens, "
                f"expected {n} x {reference_usage['completion_tokens']}",
            )
            require(
                usage.get("total_tokens")
                == usage.get("prompt_tokens") + usage.get("completion_tokens"),
                f"{label} usage total does not equal prompt plus completion",
            )

    # A count the server cannot serve is refused before generation, naming the
    # field, on both endpoints.
    for endpoint, extra in (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "inspected"}]}),
        ("/v1/completions", {"prompt": "inspected"}),
    ):
        for value in (0, 129, 2.5):
            result = http_request(
                base_url, endpoint, {**extra, "max_tokens": 1, "n": value}, timeout=timeout
            )
            error = refusal_error(result, f"{endpoint} n={value!r}")
            require(error.get("param") == "n", f"n={value!r} refusal named {error.get('param')!r}")

    print(
        f"[PASS] n choices: {n} indexed choices on both endpoints, greedy text and "
        "summed usage"
    )


def logprob_content(result: HttpResult, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The per-token entries of a response's only choice, shape-checked.

    Returns the "content" array and the choice it came from. Every invariant that
    does not need two requests is checked here, so the callers below only have to
    compare positions against something.
    """
    require(result.status == 200, f"{label} failed: HTTP {result.status}: {result.text}")
    body = result.json()
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{label} has {len(choices or [])} choices")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    require(isinstance(logprobs, dict), f"{label} returned no logprobs object: {choice}")
    content = logprobs.get("content")
    require(
        isinstance(content, list) and content,
        f"{label} reported no per-token log probabilities: {logprobs}",
    )
    for position, entry in enumerate(content):
        where = f"{label} position {position}"
        require(isinstance(entry, dict), f"{where} is not an object")
        token = entry.get("token")
        require(isinstance(token, str), f"{where} has no token string")
        value = entry.get("logprob")
        require(
            isinstance(value, (int, float)) and math.isfinite(value) and value <= 0.0,
            f"{where} reported logprob {value!r}, which is not a log probability",
        )
        raw = entry.get("bytes")
        require(
            isinstance(raw, list) and raw and all(isinstance(b, int) and 0 <= b <= 255 for b in raw),
            f"{where} bytes is not a non-empty list of octets: {raw!r}",
        )
        # The generated text is ASCII here, so no token is a partial UTF-8
        # sequence and the byte array must be exactly the token's encoding.
        require(
            bytes(raw) == token.encode("utf-8"),
            f"{where} bytes {raw!r} do not encode its token {token!r}",
        )
        alternatives = entry.get("top_logprobs")
        if alternatives is None:
            continue
        require(isinstance(alternatives, list), f"{where} top_logprobs is not a list")
        previous = 0.0
        for rank, candidate in enumerate(alternatives):
            spot = f"{where} alternative {rank}"
            require(isinstance(candidate, dict), f"{spot} is not an object")
            require(isinstance(candidate.get("token"), str), f"{spot} has no token string")
            logged = candidate.get("logprob")
            require(
                isinstance(logged, (int, float)) and math.isfinite(logged) and logged <= 0.0,
                f"{spot} reported logprob {logged!r}, which is not a log probability",
            )
            require(
                rank == 0 or logged <= previous,
                f"{spot} is ranked above the one before it ({logged!r} > {previous!r})",
            )
            previous = logged
            require(
                bytes(candidate.get("bytes") or []) == candidate["token"].encode("utf-8"),
                f"{spot} bytes do not encode its token {candidate['token']!r}",
            )
    return content, choice


def validate_logprobs(base_url: str, model_name: str, timeout: float) -> None:
    """Checks that log probabilities are reported and describe the same run.

    The engine ranks the model's own next-token distribution from the raw logits,
    so the strongest check available without those logits is self-consistency:
    the ranking at a position must agree with the token the engine generated
    there, the alternatives must be ordered, and the array must cover the text
    the caller was handed. The text is read from /v1/completions, where it is the
    decoded token stream and nothing else -- on chat the sidecar has already split
    reasoning out of it, so a token count no longer lines up with `content`.
    """
    prompt = "List the numbers from 1 to 20, separated by commas."
    max_tokens = 24
    greedy = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 20}

    reference = http_request(base_url, "/v1/completions", greedy, timeout=timeout)
    reference_text, reference_finish = answer_and_finish(reference, False, False, model_name)
    require(len(reference_text) >= 8, f"reference answer is too short: {reference_text!r}")

    alternatives_wanted = 3
    result = http_request(
        base_url, "/v1/completions", {**greedy, "logprobs": alternatives_wanted}, timeout=timeout
    )
    content, choice = logprob_content(result, f"completions logprobs={alternatives_wanted}")
    body = result.json()
    text = choice["text"]

    # Asking for a ranking must not change what is generated: the same greedy
    # request has to return the same tokens with and without it.
    require(
        text == reference_text,
        f"the ranked answer {text!r} differs from the unranked one {reference_text!r}",
    )
    require(
        choice["finish_reason"] == reference_finish,
        f"the ranked answer finished with {choice['finish_reason']!r}, "
        f"the unranked one with {reference_finish!r}",
    )

    # The array is per token of the answer, in order, and rejoins into it.
    require(
        "".join(entry["token"] for entry in content) == text,
        f"the ranked tokens do not rejoin into the text {text!r}",
    )
    require(
        len(content) <= body["usage"]["completion_tokens"],
        f"{len(content)} ranked positions for {body['usage']['completion_tokens']} tokens",
    )

    for position, entry in enumerate(content):
        candidates = entry.get("top_logprobs")
        require(
            isinstance(candidates, list) and len(candidates) == alternatives_wanted,
            f"position {position} reported {len(candidates or [])} alternatives, "
            f"expected {alternatives_wanted}",
        )
        # Greedy decoding takes the argmax of the distribution the sampler draws
        # from, and the ranking is that same distribution from the raw logits, so
        # the top alternative must be the token that was generated -- with the
        # probability reported twice, once beside the token and once in the
        # ranking. This is what ties the ranking to the run instead of letting a
        # plausible-looking distribution be reported for tokens it did not
        # produce.
        require(
            candidates[0]["token"] == entry["token"],
            f"position {position} generated {entry['token']!r} but ranked "
            f"{candidates[0]['token']!r} first",
        )
        require(
            candidates[0]["logprob"] == entry["logprob"],
            f"position {position} reported {entry['logprob']!r} for the sampled token "
            f"and {candidates[0]['logprob']!r} for the same token in the ranking",
        )

    # A count of 0 asks for the sampled token's own probability and no
    # alternatives, which is not the same as asking for nothing.
    bare = http_request(base_url, "/v1/completions", {**greedy, "logprobs": 0}, timeout=timeout)
    bare_content, bare_choice = logprob_content(bare, "completions logprobs=0")
    require(bare_choice["text"] == text, "logprobs=0 changed the generated text")
    require(
        "".join(entry["token"] for entry in bare_content) == text,
        "the logprobs=0 array does not rejoin into the text",
    )
    require(
        all("top_logprobs" not in entry for entry in bare_content),
        "logprobs=0 reported alternatives",
    )

    # Chat spells the same request as a boolean plus a count, and the count is
    # the only thing that decides whether alternatives are reported. The ranked
    # tokens are not compared against the text here: the sidecar splits a chat
    # completion into reasoning and content, so the token stream is no longer the
    # answer string. What still holds is per position -- the top alternative is
    # the token generated there -- and that is what is checked.
    for top_logprobs, expected in ((4, 4), (0, 0)):
        chat = http_request(
            base_url,
            "/v1/chat/completions",
            {**chat_payload(prompt, max_tokens), "logprobs": True, "top_logprobs": top_logprobs},
            timeout=timeout,
        )
        label = f"chat logprobs=true top_logprobs={top_logprobs}"
        chat_content, _ = logprob_content(chat, label)
        for position, entry in enumerate(chat_content):
            candidates = entry.get("top_logprobs")
            if expected == 0:
                require(
                    not candidates,
                    f"{label} reported {len(candidates or [])} alternatives at position "
                    f"{position}",
                )
                continue
            require(
                isinstance(candidates, list) and len(candidates) == expected,
                f"{label} position {position} reported {len(candidates or [])} "
                f"alternatives, expected {expected}",
            )
            require(
                candidates[0]["token"] == entry["token"],
                f"{label} position {position} ranked {candidates[0]['token']!r} above "
                f"the generated {entry['token']!r}",
            )

    # A client stop sequence ends the answer inside the token stream, and the
    # ranking has to end with it: reporting a probability for a position the
    # caller never received would put the array out of step with the text.
    middle = len(text) // 2
    sequence = text[middle : middle + 4]
    expected_text = text[: text.index(sequence)]
    require(expected_text != "", f"stop sequence chosen at the start of the answer: {sequence!r}")
    stopped = http_request(
        base_url, "/v1/completions", {**greedy, "logprobs": 2, "stop": [sequence]}, timeout=timeout
    )
    stopped_content, stopped_choice = logprob_content(stopped, "completions logprobs=2 with stop")
    require(
        stopped_choice["text"] == expected_text,
        f"the stopped answer {stopped_choice['text']!r} is not {expected_text!r}",
    )
    # The cut can land inside a token, so the array ends at the last position that
    # fits entirely before it rather than exactly on it. Both halves of that are
    # checked: nothing past the cut is reported, and the array does not stop early
    # -- the shortfall is less than one token of the run being ranked.
    covered = "".join(entry["token"] for entry in stopped_content)
    require(
        expected_text.startswith(covered),
        f"the ranking covers {covered!r}, which runs past the stopped text {expected_text!r}",
    )
    longest = max(len(entry["token"]) for entry in content)
    require(
        len(expected_text) - len(covered) < longest,
        f"the ranking stops {len(expected_text) - len(covered)} characters before the "
        f"stopped text {expected_text!r}, further than one token",
    )

    print("[PASS] logprobs: per-token probabilities that agree with the tokens generated")


# The tool the tool-call check offers. `city` and `days` are described as
# required and `days` is an integer on purpose: the XML the chat template emits
# marks no type of its own, so the only way "3" comes back as the number 3 rather
# than the string "3" is if the schema reached the parser.
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city to look up."},
                "days": {"type": "integer", "description": "How many days ahead."},
            },
            "required": ["city", "days"],
        },
    },
}


def tool_call_arguments(tool_call: dict[str, Any], label: str) -> dict[str, Any]:
    function = tool_call.get("function")
    require(isinstance(function, dict), f"{label} has no function object")
    require(
        isinstance(function.get("arguments"), str),
        f"{label} arguments are not a JSON string: {function.get('arguments')!r}",
    )
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} arguments are not JSON: {function['arguments']!r}") from exc
    require(isinstance(arguments, dict), f"{label} arguments are not an object")
    return arguments


def validate_tool_calls(base_url: str, model_name: str, timeout: float) -> None:
    """Checks that a tool call comes back as a tool call and not as prose.

    The offered tool is the one the prompt asks for by name, so a model that can
    follow the template at all emits the call; the phrasings are tried in order
    because this is the one part of the surface whose output the server does not
    control, and a single miss would otherwise read as a parser bug.
    """
    attempts = (
        "What is the weather in Paris for the next 3 days? Use the get_weather tool.",
        "Call get_weather with city=Paris and days=3.",
        "Use the tool named get_weather for Paris, 3 days out.",
    )
    declared = {WEATHER_TOOL["function"]["name"]}
    properties = WEATHER_TOOL["function"]["parameters"]["properties"]
    missed: list[str] = []

    for prompt in attempts:
        result = http_request(
            base_url,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "tools": [WEATHER_TOOL],
                "max_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
            },
            timeout=timeout,
        )
        require(
            result.status == 200,
            f"a request carrying tools was not served: HTTP {result.status}: {result.text}",
        )
        body = result.json()
        require(body.get("model") == model_name, "wrong model on the tool-call response")
        choice = body["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls")
        if calls is None:
            # No call this time: the call syntax, if any, has to be in the content
            # and the completion has to be a normal one. Retried below.
            missed.append(
                f"{prompt!r} -> finish_reason={choice.get('finish_reason')!r} "
                f"content={message.get('content', '')!r}"
            )
            continue

        require(isinstance(calls, list) and calls, "tool_calls is present but empty")
        require(
            choice.get("finish_reason") == "tool_calls",
            "a response carrying tool calls finished with "
            f"{choice.get('finish_reason')!r}",
        )
        # The call was moved out of the text, so nothing of its own syntax is left
        # behind for a client to trip over.
        require(
            "<tool_call>" not in message.get("content", "")
            and "</tool_call>" not in message.get("content", ""),
            f"the call syntax is still in the content: {message.get('content')!r}",
        )

        for index, call in enumerate(calls):
            label = f"tool call {index}"
            require(call.get("type") == "function", f"{label} is not a function call")
            identifier = call.get("id")
            require(
                isinstance(identifier, str)
                and len(identifier) == 29
                and identifier.startswith("call_")
                and all(c in "0123456789abcdef" for c in identifier[5:]),
                f"{label} has a malformed id: {identifier!r}",
            )
            name = call["function"].get("name")
            require(name in declared, f"{label} calls {name!r}, which was not offered")
            arguments = tool_call_arguments(call, label)
            # Nothing the model was not offered: a typed schema has no room for an
            # argument the tool does not declare.
            unknown = set(arguments) - set(properties)
            require(not unknown, f"{label} invented arguments: {sorted(unknown)}")

        # The declared types are what make the arguments usable without the caller
        # reparsing text, so at least one declared parameter has to come back with
        # the type the schema gave it -- a string here, an integer there.
        arguments = tool_call_arguments(calls[0], "the first tool call")
        require(arguments, "the tool call carried no arguments at all")
        for key, value in arguments.items():
            declared_type = properties[key]["type"]
            if declared_type == "string":
                require(isinstance(value, str), f"{key} came back as {value!r}, not a string")
            elif declared_type == "integer":
                require(
                    isinstance(value, int) and not isinstance(value, bool),
                    f"{key} came back as {value!r}, not an integer",
                )
        print(
            f"[PASS] tool calls: {len(calls)} call(s) parsed out of the completion "
            f"with arguments {json.dumps(arguments, ensure_ascii=False)}"
        )
        return

    raise AssertionError(
        "no attempt produced a tool call from a model asked to use one; the parser "
        "or the checkpoint's template is at fault, not the request.\n"
        + "\n".join(missed)
    )


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

        validate_n_choices(base_url, model_name, timeout=args.request_timeout_seconds + 30)

        validate_logprobs(base_url, model_name, timeout=args.request_timeout_seconds + 30)

        validate_tool_calls(base_url, model_name, timeout=args.request_timeout_seconds + 30)

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
