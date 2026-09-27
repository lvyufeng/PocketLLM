#!/usr/bin/env python3
"""Does the Python server's own ``/metrics`` show one scheduler holding several requests?

`bench_pocketllm_serve_concurrency.py` measures what concurrency buys. This measures the thing
underneath it: that two concurrent arrivals land in **one** scheduler with a non-empty running set,
rather than in a lock that serializes them. The distinction matters because a lock produces the same
token counts and the same answers -- a concurrency benchmark that only reads responses cannot tell
the two apart, and an implementation that quietly serialized would look like a slower scheduler
rather than like no scheduler.

The evidence is the server's own exposition. `pocketllm_requests_running` is a gauge the server
reads off the live `BatchScheduler`, and it is the same gauge, from the same stats struct, that the
native host publishes as `pocket_requests_running` -- one scheduler library, two hosts, so the two
servers agree up to the metric prefix. This script samples it while a group of requests is in
flight and reports the peak, which is the number a serialized server cannot produce.

The requests are non-streamed. The cpp adapter's streaming path still holds a lock across the whole
generation, so a streamed run would measure that lock rather than the scheduler; that is a known gap
and it is recorded as such in the guide, not hidden by this script's choice.

Run both modes to see the contrast -- the serial arm is the null:

    python scripts/bench_cpp_scheduler_metrics.py --port 8123
    python scripts/bench_cpp_scheduler_metrics.py --port 8124 --no-enable-batching

Example:
    python scripts/bench_cpp_scheduler_metrics.py \\
        --port 8123 --concurrent 4 --max-tokens 64 --json-out /tmp/scheduler_metrics.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CKPT = "/mnt/data2/Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"

# Prompts that do not stop early, so the group's wall time is a decode measurement rather than a
# race between different EOS points. Distinct on purpose: identical prompts would share a prefix
# and the second request would pay only for its decode.
PROMPTS = (
    "List every integer from 1 to 60 in words, separated by commas, with no other text.",
    "Name the planets of the solar system in order from the sun, one per line, and add one fact "
    "about each planet.",
    "Write the first twenty elements of the periodic table with their symbols, one per line.",
    "Count backwards from 100 to 1 in steps of seven, one number per line.",
)

#: The scheduler gauges this script reads, as the Python server exports them. The native host's
#: names are these with `pocketllm_` replaced by `pocket_`; the suffixes are the same on purpose so
#: a comparison of the two hosts is a prefix substitution rather than a translation table.
GAUGES = ("requests_running", "requests_waiting", "slots_free")


def http(url: str, body: dict | None = None, timeout: float = 900.0) -> tuple[int, str]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def scrape(url: str) -> dict[str, float]:
    """One exposition, parsed into `{name: value}` for the gauge suffixes this script watches."""
    status, text = http(url)
    if status != 200:
        return {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.rpartition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        leaf = name.strip().split("{", 1)[0].removeprefix("pocketllm_")
        if leaf in GAUGES:
            values[leaf] = value
    return values


def wait_ready(port: int, process: subprocess.Popen, deadline: float) -> None:
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the server exited with {process.returncode}")
        try:
            status, _ = http(f"http://127.0.0.1:{port}/health", timeout=5.0)
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(2.0)
    raise SystemExit("the server did not become ready")


class _Sampler(threading.Thread):
    """Polls the exposition while the group runs, keeping the peak of each gauge."""

    def __init__(self, port: int, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.url = f"http://127.0.0.1:{port}/metrics"
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        #: Gauges seen at least once. The peak alone cannot tell "the scheduler is here and was
        #: never busy" from "there is no scheduler to ask" -- both read as zero -- and those are
        #: the two arms this script exists to separate.
        self.seen: set[str] = set()
        # Not `self._stop`: `threading.Thread` has a private `_stop` of its own that `join()` calls
        # on the way to reaping the thread, so an attribute of that name shadows a method and
        # `join()` dies with "'Event' object is not callable".
        self._done = threading.Event()

    def run(self) -> None:
        while not self._done.is_set():
            try:
                sample = scrape(self.url)
                self.seen.update(sample)
                self.samples.append(sample)
            except Exception:
                pass
            self._done.wait(self.interval)

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=5.0)

    def peak(self, name: str) -> float:
        return max((sample.get(name, 0.0) for sample in self.samples), default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--concurrent", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--backend",
        default="cpp",
        help="the runtime to serve; a Python runtime joins the same scheduler behind "
        "--enable-batching and reports the same gauges at its own width",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        help="name=value, repeatable; passed through to the server as --backend-option",
    )
    parser.add_argument("--no-enable-batching", action="store_true")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to allow for readiness, given to the server's TP supervisor as well as "
        "used here; a runtime that fills a resident expert bank on startup needs far more "
        "than the supervisor's 300 s default, and the supervisor's is the one that fires "
        "first -- it reports the ranks as missing while the fill is still running",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    arm = "serial" if args.no_enable_batching else "batch"
    command = [
        sys.executable, "-m", "pocketllm", "serve",
        "--backend", args.backend,
        "--model", args.checkpoint,
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--port", str(args.port),
        "--tensor-parallel-startup-timeout", str(args.startup_timeout),
    ]
    for option in args.backend_option:
        command += ["--backend-option", option]
    if args.no_enable_batching:
        command.append("--no-enable-batching")
    else:
        command.append("--enable-batching")

    print(f"=== arm: {arm} ===", flush=True)
    started = time.perf_counter()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        wait_ready(args.port, process, time.perf_counter() + args.startup_timeout)
        print(f"ready after {time.perf_counter() - started:.1f}s", flush=True)

        # One request first, so the model's lazy one-time work lands outside the measured group.
        status, text = http(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": PROMPTS[0]}], "max_tokens": 8,
             "temperature": 0.0},
        )
        print(f"warm-up: status {status}", flush=True)

        results: list[tuple[float, int]] = []
        lock = threading.Lock()

        def one(index: int) -> None:
            payload = {
                "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            }
            begin = time.perf_counter()
            status, body = http(f"http://127.0.0.1:{args.port}/v1/chat/completions", payload)
            elapsed = time.perf_counter() - begin
            tokens = 0
            if status == 200:
                tokens = int(json.loads(body)["usage"]["completion_tokens"])
            with lock:
                results.append((elapsed, tokens))

        sampler = _Sampler(args.port)
        sampler.start()
        group_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            list(pool.map(one, range(args.concurrent)))
        group_seconds = time.perf_counter() - group_started
        sampler.stop()

        total_tokens = sum(tokens for _, tokens in results)
        peak_running = sampler.peak("requests_running")
        peak_waiting = sampler.peak("requests_waiting")
        latencies = sorted(elapsed for elapsed, _ in results)

        print(f"peak pocketllm_requests_running  {peak_running:.0f}")
        print(f"peak pocketllm_requests_waiting  {peak_waiting:.0f}")
        print(f"samples                          {len(sampler.samples)}")
        print(f"aggregate                        {total_tokens} tokens in {group_seconds:.2f}s "
              f"= {total_tokens / max(group_seconds, 1e-9):.2f} tok/s")
        print(f"per-request latency              min {latencies[0]:.2f}s  "
              f"max {latencies[-1]:.2f}s")
        # The verdict the two arms are read against. A serialized server cannot report more than
        # one running request, however many clients are connected to it -- but a *width-1* runtime
        # under the scheduler cannot either, and it is not the same thing: the scheduler says so by
        # holding the other client, which a lock has no way to report. So the series' presence is
        # read before its peak, and one running with one waiting is the scheduler at width 1 rather
        # than the scheduler absent.
        if not sampler.seen:
            verdict = (
                "no scheduler published these gauges — this backend serializes its requests "
                "without one"
            )
        elif peak_running > 1:
            verdict = (
                f"one scheduler, {peak_running:.0f} requests running at once of "
                f"{args.concurrent} concurrent clients"
            )
        else:
            verdict = (
                f"one scheduler at the runtime's declared width — peak running "
                f"{peak_running:.0f}, peak waiting {peak_waiting:.0f} of {args.concurrent} "
                "concurrent clients"
            )
        print(f"verdict: {verdict}")

        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(
                    {
                        "arm": arm,
                        "concurrent": args.concurrent,
                        "max_tokens": args.max_tokens,
                        "peak_requests_running": peak_running,
                        "peak_requests_waiting": peak_waiting,
                        "samples": len(sampler.samples),
                        "aggregate_tokens": total_tokens,
                        "group_seconds": group_seconds,
                        "aggregate_tokens_per_second": total_tokens / max(group_seconds, 1e-9),
                        "latency_seconds": latencies,
                    },
                    indent=2,
                )
            )
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
