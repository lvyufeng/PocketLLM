#!/usr/bin/env python3
"""Split single-request latency into prefill and decode for the native server.

`bench_cpp_openai_concurrency.py` reports throughput and single-request latency
for a whole request, which is what the acceptance criteria ask for but is not
enough to compare configurations: a single `output_tokens_per_second` mixes a
prefill term that scales with the prompt against a decode term that does not.

The native server emits no per-token timing on the wire, and its stream cannot
be used to delimit the phases: `handle_stream` writes the role chunk *before*
`sched.submit_request`, so the client sees a first event within milliseconds of
the request regardless of prompt length. `deepseek_timings` is a field of the
*Python* server (`src/server/openai.py`) and never appears on this path.
(Equally, `handle_nonstream` passes a null token callback, so the engine has no
first-token instant there either and its TTFT equals its full duration —
measured, not assumed: 8.8588 s against 8.8587 s.)

What the native server does export is exact per-request accounting in
`cpp_engine/core/metrics.cpp`, as Prometheus counters and histograms at
`/metrics`, and only on the streaming path does TTFT mean anything:

    pocket_ttft_seconds_sum / _count            time to the first generated token
    pocket_request_duration_seconds_sum / _count  whole request
    pocket_tokens_total{type="prompt"}          prompt tokens
    pocket_tokens_total{type="generation"}      generated tokens

Reading those before and after a single streamed request therefore gives that
request's phase split from the engine's own clock, with no reliance on chunk
arrival times or on how the client schedules reads. The deltas are attributed
to one request by asserting that all three count deltas are exactly 1.

Timing convention (`docs/guides/benchmarking.md`): the first generated token
belongs to prefill. Server-side TTFT is measured when that token is dequeued,
so

    prefill_seconds = ttft
    decode_tokens   = completion_tokens - 1
    decode_seconds  = duration - ttft

`duration` runs to the end of the response, which includes response assembly
after the last token, so `decode_seconds` is an upper bound on the decode
intervals and the decode rate here is a floor.

Two prompt lengths are measured, because a short prompt is not a substitute for
a long one in either phase. Repeats are reported individually: this host shows
several percent of run-to-run spread, and a single sample would hide it.

Example:
    python scripts/bench_cpp_openai_phases.py \
        --ckpt /mnt/data3/DeepSeek-V4-Flash-0731 \
        --binary cpp_engine/build/pocketllm_engine \
        --python /home/lvyufeng/miniconda3/envs/deepseek/bin/python \
        --max-batch-size 1 --prefill-token-budget 0 \
        --json-out phases.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bench_cpp_openai_concurrency as harness  # noqa: E402  (path set above)

# Exact metric line prefixes read from /metrics. Names, not values, are pinned:
# a renamed or dropped metric must fail loudly rather than silently become 0.
METRICS = {
    "ttft_seconds": "pocket_ttft_seconds_sum",
    "ttft_count": "pocket_ttft_seconds_count",
    "duration_seconds": "pocket_request_duration_seconds_sum",
    "duration_count": "pocket_request_duration_seconds_count",
    "prompt_tokens": 'pocket_tokens_total{type="prompt"}',
    "generation_tokens": 'pocket_tokens_total{type="generation"}',
    "successes": 'pocket_requests_total{status="success"}',
}

_METRIC_LINE = re.compile(r"^(pocket_[a-z_]+(?:\{[^}]*\})?)\s+(\S+)$", re.MULTILINE)


def metrics_snapshot(base_url: str, *, timeout: float) -> dict[str, float]:
    """Read the engine's cumulative counters; every key in METRICS must be present."""
    result = harness.http_request(base_url, "/metrics", timeout=timeout)
    harness.require(result.status == 200, f"/metrics returned HTTP {result.status}")
    found = {name: float(value) for name, value in _METRIC_LINE.findall(result.text)}
    missing = [name for name in METRICS.values() if name not in found]
    harness.require(not missing, f"/metrics is missing {missing}; the split cannot be derived")
    return {key: found[name] for key, name in METRICS.items()}


def measure_one(group: harness.ServerGroup, model: str, body: dict[str, Any]) -> dict[str, Any]:
    """One streamed request, its phase split taken from the counter deltas.

    Streamed rather than buffered on purpose: only the streaming handler
    records TTFT at the first token.
    """
    before = metrics_snapshot(group.base_url, timeout=30.0)
    result = harness.stream_request(group.base_url, body, timeout=harness.group_timeout(group))
    streamed = harness.validate_stream(result, model)
    after = metrics_snapshot(group.base_url, timeout=30.0)
    delta = {key: after[key] - before[key] for key in METRICS}

    # One request, one observation: without this the deltas could belong to a
    # neighbour, and every number below would be silently wrong.
    for key in ("ttft_count", "duration_count", "successes"):
        harness.require(delta[key] == 1, f"{key} moved by {delta[key]}, expected exactly 1")

    prompt_tokens = int(delta["prompt_tokens"])
    completion_tokens = int(delta["generation_tokens"])
    harness.require(prompt_tokens > 0, "engine counted no prompt tokens")
    harness.require(completion_tokens > 1, "need more than one generated token to measure decode")

    prefill_seconds = delta["ttft_seconds"]
    total_seconds = delta["duration_seconds"]
    decode_seconds = total_seconds - prefill_seconds
    decode_tokens = completion_tokens - 1
    harness.require(prefill_seconds > 0.0, f"ttft was {prefill_seconds}")
    harness.require(decode_seconds > 0.0, f"duration {total_seconds} does not exceed ttft {prefill_seconds}")
    # The server times its own request from before admission; the client cannot
    # finish any earlier than that, and the role chunk is written before the
    # request is even submitted, so both orderings must hold.
    harness.require(
        streamed["first_event_seconds"] < prefill_seconds,
        f"first stream event at {streamed['first_event_seconds']:.3f}s does not precede "
        f"ttft {prefill_seconds:.3f}s",
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "content_chars": streamed["content_chars"],
        # From the engine's own counters.
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt_tokens / prefill_seconds,
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "total_seconds": total_seconds,
        # From the client's socket, as a cross-check on the engine's clock.
        "observed_total_seconds": result.elapsed_seconds,
    }


def measure_length(
    group: harness.ServerGroup,
    model: str,
    prompt_words: int,
    max_tokens: int,
    repeats: int,
    warmup_rounds: int,
) -> dict[str, Any]:
    body = harness.payload(1, prompt_words, max_tokens, stream=True)
    for _ in range(warmup_rounds):
        measure_one(group, model, body)
    samples = [measure_one(group, model, body) for _ in range(repeats)]
    scalar_keys = (
        "prefill_seconds",
        "prefill_tokens_per_second",
        "decode_seconds",
        "decode_tokens_per_second",
        "total_seconds",
        "observed_total_seconds",
    )
    return {
        "prompt_words": prompt_words,
        "prompt_tokens": samples[0]["prompt_tokens"],
        "completion_tokens": samples[0]["completion_tokens"],
        "samples": samples,
        "median": {key: statistics.median([s[key] for s in samples]) for key in scalar_keys},
        "range": {
            key: [min(s[key] for s in samples), max(s[key] for s in samples)]
            for key in ("prefill_tokens_per_second", "decode_tokens_per_second")
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    log_dir = pathlib.Path(args.log_dir or tempfile.mkdtemp(prefix="pocketllm-http-phases-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    group: harness.ServerGroup | None = None
    started = time.perf_counter()
    record: dict[str, Any] = {
        "checkpoint": str(pathlib.Path(args.ckpt).resolve()),
        "tp_world": len(harness.parse_devices(args.devices)),
        "devices": args.devices,
        "max_batch_size": args.max_batch_size,
        "prefill_token_budget": args.prefill_token_budget,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "log_dir": str(log_dir),
    }
    try:
        group = harness.start_server(args, log_dir)
        model = harness.http_request(group.base_url, "/v1/models", timeout=10.0).json()["data"][0]["id"]
        record["model"] = model
        record["short"] = measure_length(
            group, model, args.short_prompt_words, args.max_tokens, args.repeats, args.warmup_rounds
        )
        record["long"] = measure_length(
            group, model, args.long_prompt_words, args.max_tokens, args.repeats, args.warmup_rounds
        )
        record["status"] = "pass"
    except Exception as exc:
        record.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}", "logs": harness.read_logs(log_dir)})
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--binary", default="cpp_engine/build-python/pocketllm_engine")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sidecar", default="src/server/cpp_sidecar.py")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=18281,
                        help="kept off the concurrency harness default so an accidental overlap fails loudly")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--prefill-token-budget", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-paged", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--short-prompt-words", type=int, default=128)
    parser.add_argument("--long-prompt-words", type=int, default=1600,
                        help="matches the long request in the concurrency harness interleave case")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--log-dir")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    harness.require(args.repeats > 0, "--repeats must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
