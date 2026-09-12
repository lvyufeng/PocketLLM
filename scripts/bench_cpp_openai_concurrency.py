#!/usr/bin/env python3
"""End-to-end HTTP concurrency acceptance test for the native TP OpenAI server.

Each invocation owns one four-rank server lifetime and runs a single scheduler
configuration. Run serial and batch configurations in separate processes so model
construction, NCCL state, and GPU clocks do not contaminate the comparison.

Example:
    python scripts/bench_cpp_openai_concurrency.py \
        --ckpt /mnt/data2/Qwen3.8-27B-FP8 \
        --binary cpp_engine/build-python/pocketllm_engine \
        --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
        --mode batch --max-batch-size 8 --json-out batch.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class HttpResult:
    status: int
    body: bytes
    elapsed_seconds: float
    first_event_seconds: float | None = None

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> dict[str, Any]:
        value = json.loads(self.text)
        if not isinstance(value, dict):
            raise AssertionError(f"expected JSON object, got {type(value).__name__}")
        return value


@dataclass
class ServerGroup:
    processes: list[subprocess.Popen[bytes]]
    log_handles: list[Any]
    log_dir: pathlib.Path
    rendezvous: pathlib.Path
    base_url: str

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 20.0
        for process in self.processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
        for process in self.processes:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        for handle in self.log_handles:
            handle.close()
        self.rendezvous.unlink(missing_ok=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    require(devices, "--devices must contain at least one device")
    return devices


def http_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
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
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return HttpResult(response.status, body, time.perf_counter() - started)
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), time.perf_counter() - started)


def stream_request(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    disconnect_after_events: int | None = None,
) -> HttpResult:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    chunks: list[bytes] = []
    first_event: float | None = None
    event_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                chunks.append(line)
                if line.startswith(b"data: "):
                    event_count += 1
                    if first_event is None:
                        first_event = time.perf_counter() - started
                    if disconnect_after_events is not None and event_count >= disconnect_after_events:
                        # Closing the response is the client-disconnect path. The
                        # server must not be allowed to poison the next request.
                        break
            status = response.status
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), time.perf_counter() - started, first_event)
    return HttpResult(status, b"".join(chunks), time.perf_counter() - started, first_event)


def wait_for_health(group: ServerGroup, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in group.processes):
            codes = [process.returncode for process in group.processes]
            raise RuntimeError(f"a TP rank exited before readiness: {codes}\n{read_logs(group.log_dir)}")
        try:
            result = http_request(group.base_url, "/health", timeout=2.0)
            if result.status == 200 and result.json().get("status") == "ok":
                return
            last_error = f"HTTP {result.status}: {result.text}"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy: {last_error}\n{read_logs(group.log_dir)}")


def read_logs(log_dir: pathlib.Path) -> str:
    chunks: list[str] = []
    for path in sorted(log_dir.glob("rank*.log")):
        chunks.append(f"--- {path.name}\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n".join(chunks)


def prompt_text(index: int, words: int) -> str:
    sentence = (
        "Explain the following benchmark request in one concise paragraph. "
        "The quick brown fox jumps over the lazy dog while the deployment team "
        "checks scheduler fairness, KV cache isolation, and tensor parallel safety. "
    )
    text = (f"Request {index}: " + sentence) * max(1, words // 35 + 1)
    return text[: max(64, words * 5)]


def payload(index: int, prompt_words: int, max_tokens: int, *, stream: bool = False) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt_text(index, prompt_words)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 20,
        "stream": stream,
    }


def validate_completion(result: HttpResult, expected_model: str) -> dict[str, Any]:
    require(result.status == 200, f"HTTP {result.status}: {result.text[:500]}")
    body = result.json()
    require(body.get("object") == "chat.completion", "wrong completion object")
    require(body.get("model") == expected_model, f"wrong model: {body.get('model')!r}")
    choices = body.get("choices")
    require(isinstance(choices, list) and choices, "completion has no choices")
    choice = choices[0]
    require(isinstance(choice, dict), "completion choice is not an object")
    require(choice.get("message", {}).get("role") == "assistant", "missing assistant role")
    require(choice.get("finish_reason") in {"stop", "length"}, "invalid finish reason")
    usage = body.get("usage")
    require(isinstance(usage, dict), "completion has no usage")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    require(prompt_tokens > 0 and completion_tokens > 0, "completion token counts must be positive")
    require(usage.get("total_tokens") == prompt_tokens + completion_tokens, "invalid total token count")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": choice["finish_reason"],
    }


def validate_stream(result: HttpResult, expected_model: str) -> dict[str, Any]:
    require(result.status == 200, f"stream HTTP {result.status}: {result.text[:500]}")
    events: list[dict[str, Any]] = []
    saw_done = False
    for line in result.text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            saw_done = True
            continue
        event = json.loads(raw)
        require(event.get("object") == "chat.completion.chunk", "wrong stream object")
        require(event.get("model") == expected_model, "wrong stream model")
        require(isinstance(event.get("choices"), list) and event["choices"], "stream has no choices")
        events.append(event)
    require(saw_done, "stream did not end with [DONE]")
    require(events, "stream had no JSON events")
    require(events[0]["choices"][0]["delta"].get("role") == "assistant", "stream lacks role event")
    finish = events[-1]["choices"][0].get("finish_reason")
    require(finish in {"stop", "length"}, f"stream terminal reason is {finish!r}")
    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
    )
    require(content, "stream had no content")
    return {"events": len(events), "first_event_seconds": result.first_event_seconds, "content_chars": len(content)}


def start_server(args: argparse.Namespace, log_dir: pathlib.Path) -> ServerGroup:
    devices = parse_devices(args.devices)
    rendezvous = log_dir / f"nccl-{uuid.uuid4().hex}.id"
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[Any] = []
    common = [
        str(pathlib.Path(args.binary).resolve()), "--serve", "--ckpt", str(pathlib.Path(args.ckpt).resolve()),
        "--tp-world", str(len(devices)), "--nccl-id-path", str(rendezvous),
        "--smoke-layers", str(args.layers), "--max-context", str(args.max_context),
        "--max-batch-size", str(args.max_batch_size), "--prefill-token-budget", str(args.prefill_token_budget),
        "--request-timeout-seconds", str(args.request_timeout_seconds), "--python", str(pathlib.Path(args.python).resolve()),
        "--sidecar", str(pathlib.Path(args.sidecar).resolve()), "--host", "127.0.0.1", "--port", str(args.port),
        "--kv-block-size", str(args.kv_block_size),
    ]
    if args.kv_paged:
        common.append("--kv-paged")
    for rank, visible_device in enumerate(devices):
        handle = (log_dir / f"rank{rank}.log").open("wb")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_device
        process = subprocess.Popen(
            common + ["--tp-rank", str(rank), "--device", "0"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append(process)
        handles.append(handle)
    group = ServerGroup(processes, handles, log_dir, rendezvous, f"http://127.0.0.1:{args.port}")
    try:
        wait_for_health(group, args.startup_timeout)
    except Exception:
        group.stop()
        raise
    return group


def run_concurrent(group: ServerGroup, expected_model: str, count: int, prompt_words: int, max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()

    def one(index: int) -> dict[str, Any]:
        request_started = time.perf_counter()
        result = http_request(
            group.base_url, "/v1/chat/completions", payload(index, prompt_words, max_tokens),
            timeout=group_timeout(group),
        )
        parsed = validate_completion(result, expected_model)
        parsed.update({"request_id": index, "latency_seconds": time.perf_counter() - request_started})
        return parsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(one, range(count)))
    wall = time.perf_counter() - started
    tokens = sum(item["completion_tokens"] for item in results)
    return {
        "count": count,
        "prompt_words": prompt_words,
        "max_tokens": max_tokens,
        "wall_seconds": wall,
        "requests_per_second": count / wall,
        "output_tokens_per_second": tokens / wall,
        "avg_latency_seconds": sum(item["latency_seconds"] for item in results) / count,
        "results": results,
    }


def group_timeout(group: ServerGroup) -> float:
    # The process is configured with a generous request timeout; HTTP gets an
    # extra margin for queueing and response serialization.
    return 1200.0


def run_interleave(group: ServerGroup, expected_model: str, max_tokens: int) -> dict[str, Any]:
    long_result: dict[str, Any] = {}
    long_error: list[BaseException] = []

    def long_request() -> None:
        try:
            result = http_request(
                group.base_url, "/v1/chat/completions", payload(900, 1600, max_tokens),
                timeout=group_timeout(group),
            )
            long_result.update(validate_completion(result, expected_model))
        except BaseException as exc:  # propagate after the short request
            long_error.append(exc)

    thread = threading.Thread(target=long_request)
    started = time.perf_counter()
    thread.start()
    time.sleep(0.5)
    short_started = time.perf_counter()
    short_http = http_request(
        group.base_url, "/v1/chat/completions", payload(901, 32, max_tokens),
        timeout=group_timeout(group),
    )
    short = validate_completion(short_http, expected_model)
    short_elapsed = time.perf_counter() - short_started
    thread.join(timeout=group_timeout(group))
    require(not thread.is_alive(), "long request did not finish")
    if long_error:
        raise long_error[0]
    return {
        "short_latency_seconds": short_elapsed,
        "short_completion_tokens": short["completion_tokens"],
        "long_completion_tokens": long_result["completion_tokens"],
        "started_long_before_short": True,
        "wall_seconds": time.perf_counter() - started,
    }


def run_stream_cases(group: ServerGroup, expected_model: str, max_tokens: int) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(stream_request, group.base_url, payload(i, 64, max_tokens, stream=True), timeout=group_timeout(group))
            for i in (100, 101)
        ]
        streams = [validate_stream(future.result(), expected_model) for future in futures]

    disconnected = stream_request(
        group.base_url, payload(102, 512, max_tokens, stream=True),
        timeout=group_timeout(group), disconnect_after_events=2,
    )
    require(disconnected.status == 200, f"disconnect stream failed: HTTP {disconnected.status}")
    recovery = http_request(
        group.base_url, "/v1/chat/completions", payload(103, 32, max_tokens),
        timeout=group_timeout(group),
    )
    recovery_body = validate_completion(recovery, expected_model)
    return {"concurrent_streams": streams, "disconnect_events_read": disconnected.text.count("data: "), "recovery": recovery_body}


def run(args: argparse.Namespace) -> dict[str, Any]:
    log_dir = pathlib.Path(args.log_dir or tempfile.mkdtemp(prefix="pocketllm-http-concurrency-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    group: ServerGroup | None = None
    started = time.perf_counter()
    record: dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": str(pathlib.Path(args.ckpt).resolve()),
        "tp_world": len(parse_devices(args.devices)),
        "devices": args.devices,
        "max_batch_size": args.max_batch_size,
        "prefill_token_budget": args.prefill_token_budget,
        "kv_paged": args.kv_paged,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "log_dir": str(log_dir),
    }
    try:
        group = start_server(args, log_dir)
        models = http_request(group.base_url, "/v1/models", timeout=10.0).json()
        model = models["data"][0]["id"]
        record["model"] = model

        single = run_concurrent(group, model, 1, args.short_prompt_words, args.max_tokens)
        concurrency = [run_concurrent(group, model, n, args.short_prompt_words, args.max_tokens) for n in args.concurrency]
        interleave = run_interleave(group, model, args.max_tokens)
        streams = run_stream_cases(group, model, args.max_tokens)
        record.update({"single": single, "concurrency": concurrency, "interleave": interleave, "streaming": streams, "status": "pass"})
    except Exception as exc:
        record.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}", "logs": read_logs(log_dir)})
        raise
    finally:
        if group is not None:
            group.stop()
        record["elapsed_seconds"] = time.perf_counter() - started
        if args.json_out:
            pathlib.Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--binary", default="cpp_engine/build-python/pocketllm_engine")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sidecar", default="src/server/cpp_sidecar.py")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=18280)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--prefill-token-budget", type=int, default=4096)
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-paged", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--short-prompt-words", type=int, default=128)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--log-dir")
    parser.add_argument("--json-out")
    parser.add_argument("--mode", choices=("serial", "batch"), default="batch")
    args = parser.parse_args()
    if args.mode == "serial":
        args.max_batch_size = 1
        args.prefill_token_budget = 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
