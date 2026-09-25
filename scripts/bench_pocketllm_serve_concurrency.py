#!/usr/bin/env python3
"""Aggregate throughput at several concurrency levels against ``pocketllm serve``.

The third of the server benchmarks: `bench_cpp_openai_phases.py` splits one
request into phases for the *native* server, `bench_pocketllm_serve_phases.py`
does the same for the Python one, and this measures what several requests at once
cost and buy on that same Python server.  It talks to a server that is already
running, because which scheduler the server started with is the thing under test
and only the command line that started it knows.

The requests are non-streamed.  That is not a convenience: the streaming path in
the cpp adapter holds one lock for the whole generation, because the native
engine has a single mutable KV session, so a streamed run would measure the lock
rather than the scheduler.  The non-streamed path goes through
``generate([request])``, which is the path the batch scheduler is on.

The metric is aggregate tokens a second -- generated tokens over the wall time
of the whole simultaneous group -- reported against the same figure at
concurrency 1.  Per-request latency is reported beside it because the two move in
opposite directions and only the pair says whether concurrency was worth having.

Rate the numbers as a client sees them: this is a throughput measurement, not a
`prefill_tps`/`decode_tps` split, and the two conventions are not interchangeable
(`docs/guides/latency_metrics.md`).

Example:
    python scripts/bench_pocketllm_serve_concurrency.py \
        --url http://127.0.0.1:8123 --concurrency 1 2 4 --max-tokens 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request


def one_request(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    usage = payload.get("usage") or {}
    return {
        "seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "text": ((payload.get("choices") or [{}])[0].get("message") or {}).get("content", ""),
    }


def measure(url: str, model: str, prompt: str, max_tokens: int, concurrency: int) -> dict:
    """One group of ``concurrency`` simultaneous requests, timed as a group."""

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, url, model, prompt, max_tokens) for _ in range(concurrency)]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - started

    generated = sum(item["completion_tokens"] for item in results)
    latencies = sorted(item["seconds"] for item in results)
    return {
        "concurrency": concurrency,
        "wall_seconds": wall,
        "generated_tokens": generated,
        "aggregate_tps": generated / wall if wall > 0 else float("nan"),
        "median_request_seconds": latencies[len(latencies) // 2],
        "slowest_request_seconds": latencies[-1],
        # Every row of one concurrency level decodes the same prompt greedily, so
        # the rows must agree. One text is kept as the reference for that check.
        "reference_text": results[0]["text"],
        "distinct_texts": len({item["text"] for item in results}),
        "prompt_tokens": results[0]["prompt_tokens"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8123", help="base URL of a running `pocketllm serve`")
    parser.add_argument("--model", default=None, help="model id; defaults to the only entry in /v1/models")
    parser.add_argument("--prompt-file", default="", help="file whose text is the prompt")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    model = args.model
    if not model:
        with urllib.request.urlopen(f"{args.url}/v1/models", timeout=30.0) as response:
            entries = json.loads(response.read().decode("utf-8")).get("data") or []
        if len(entries) != 1:
            parser.error(f"pass --model: /v1/models offers {len(entries)} entries")
        model = entries[0]["id"]

    prompt = open(args.prompt_file, encoding="utf-8").read() if args.prompt_file else args.prompt

    # One warmup request, discarded: it is what pays for the first call's
    # allocations, and at concurrency 1 it would otherwise be the only row in the
    # table with a cold cost in it.
    one_request(args.url, model, prompt, 8)

    rows = []
    baseline = None
    for level in args.concurrency:
        row = measure(args.url, model, prompt, args.max_tokens, level)
        if baseline is None:
            baseline = row["aggregate_tps"]
        row["speedup"] = row["aggregate_tps"] / baseline if baseline else float("nan")
        rows.append(row)
        print(
            f"concurrency={level:>2}  prompt={row['prompt_tokens']:>5}  "
            f"group={row['wall_seconds']:7.3f}s  aggregate={row['aggregate_tps']:7.2f} tok/s  "
            f"speedup={row['speedup']:5.2f}x  median request={row['median_request_seconds']:7.3f}s  "
            f"distinct texts={row['distinct_texts']}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"model": model, "url": args.url, "rows": rows}, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
