#!/usr/bin/env python3
"""Split single-request latency into prefill and decode for ``pocketllm serve``.

``bench_cpp_openai_phases.py`` does this for the *native* server
(``cpp_engine/.../pocketllm_engine``), whose `/metrics` carries the
``pocket_*`` families.  This is its counterpart for the Python server that
``pocketllm serve`` starts, whatever backend it selected, whose families are
``pocketllm_*``.  The two are separate scripts because they are separate
servers, not because the measurement differs.

The phase split is read from the engine's own clock rather than from chunk
arrival times, for the reason that script documents at length: client-side
timestamps measure the client's read loop as much as the model's.  Only the
streamed path is used, because `handle_stream` latches TTFT on the first event
that carries a token -- a non-streamed response produces no TTFT at all.

Per request, from the deltas of one streamed call:

    pocketllm_ttft_seconds_sum/_count          time to the first generated token
    pocketllm_request_duration_seconds_sum     whole request, arrival to last token
    pocketllm_prompt_tokens_total              prompt tokens
    pocketllm_generation_tokens_total          generated tokens

and then, by the repository's timing convention
(`docs/guides/benchmarking.md`): the first generated token is produced by the
prompt forward and belongs to prefill, so

    prefill_seconds = ttft
    prefill_tps     = prompt_tokens / ttft
    decode_tokens   = generation_tokens - 1
    decode_seconds  = duration - ttft
    decode_tps      = decode_tokens / decode_seconds

`duration` runs to the end of the response, after the last token, so
`decode_seconds` is an upper bound on the intervals and `decode_tps` a floor.

Repeats of one prompt are reported individually rather than averaged.  They are
also the prefix-cache probe: an identical prompt sent twice should reuse the
stored prefix, which shows up as a lower TTFT on the second call with the same
`prompt_tokens` -- so a repeat whose TTFT did not move says the reuse did not
happen, and averaging would hide exactly that.

Example:
    python scripts/bench_pocketllm_serve_phases.py \
        --url http://127.0.0.1:8123 \
        --prompt-file docs/guides/benchmarking.md \
        --max-tokens 64 --repeats 2 --json-out bonsai.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

METRIC_NAMES = (
    "ttft_seconds",
    "request_duration_seconds",
    "prompt_tokens_total",
    "generation_tokens_total",
    "requests_total",
    "cancellations_total",
)


@dataclass
class Scrape:
    """One `/metrics` reading, flattened to the numbers the deltas need."""

    values: dict[str, float] = field(default_factory=dict)

    def __sub__(self, other: "Scrape") -> dict[str, float]:
        return {name: self.values.get(name, 0.0) - other.values.get(name, 0.0) for name in self.values}


def scrape(url: str) -> Scrape:
    """Read `/metrics` and keep the `_sum` of each histogram plus the counters.

    Only the `_sum` is kept and not the buckets: a single request's phase split
    is a sum and a count, and bucketing it would put a 4-second prefill in the
    same bucket as a 2-second one.
    """

    with urllib.request.urlopen(f"{url}/metrics", timeout=30.0) as response:
        text = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, raw = line.partition(" ")
        # `pocketllm_ttft_seconds_sum` -> the family's sum; a bare counter name
        # (no `_bucket`, `_sum`, `_count` suffix) is itself the value.
        for family in METRIC_NAMES:
            for suffix in ("_sum", "_count", ""):
                if name == f"pocketllm_{family}{suffix}":
                    try:
                        values[f"{family}{suffix}"] = float(raw)
                    except ValueError:
                        pass
    return Scrape(values)


@dataclass
class Sample:
    label: str
    prompt_tokens: int
    generation_tokens: int
    ttft: float
    duration: float
    client_ttft: float
    client_total: float

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.ttft if self.ttft > 0 else float("nan")

    @property
    def decode_tps(self) -> float:
        tokens = self.generation_tokens - 1
        seconds = self.duration - self.ttft
        return tokens / seconds if tokens > 0 and seconds > 0 else float("nan")

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "prefill_seconds": self.ttft,
            "prefill_tps": self.prefill_tps,
            "decode_seconds": self.duration - self.ttft,
            "decode_tps": self.decode_tps,
            "request_seconds": self.duration,
            "client_ttft_seconds": self.client_ttft,
            "client_total_seconds": self.client_total,
        }


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, float, str, int]:
    """One streamed request; returns client TTFT, client total, text, token count."""

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_at = 0.0
    pieces: list[str] = []
    tokens = 0
    with urllib.request.urlopen(request, timeout=3600.0) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    tokens += 1
                    pieces.append(content)
                    if not first_at:
                        first_at = time.perf_counter()
    return first_at - started, time.perf_counter() - started, "".join(pieces), tokens


def measure(url: str, model: str, label: str, prompt: str, max_tokens: int) -> Sample:
    before = scrape(url)
    client_ttft, client_total, _text, client_tokens = stream_once(url, model, prompt, max_tokens)
    after = scrape(url)
    delta = after - before

    requests = int(round(delta.get("requests_total", 0.0)))
    ttft_count = int(round(delta.get("ttft_seconds_count", 0.0)))
    if requests != 1 or ttft_count != 1:
        raise RuntimeError(
            f"the metrics deltas are not one request's: requests={requests} ttft_observations={ttft_count}; "
            "nothing else may talk to this server while it is measured"
        )
    return Sample(
        label=label,
        prompt_tokens=int(round(delta.get("prompt_tokens_total", 0.0))),
        generation_tokens=int(round(delta.get("generation_tokens_total", 0.0))),
        ttft=delta.get("ttft_seconds_sum", 0.0),
        duration=delta.get("request_duration_seconds_sum", 0.0),
        client_ttft=client_ttft,
        client_total=client_total,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8123", help="base URL of a running `pocketllm serve`")
    parser.add_argument("--model", default=None, help="model id; defaults to the only entry in /v1/models")
    parser.add_argument("--prompt-file", action="append", default=[], help="file whose text is the prompt; repeatable")
    parser.add_argument("--prompt", action="append", default=[], help="literal prompt; repeatable")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not args.prompt_file and not args.prompt:
        parser.error("pass at least one --prompt-file or --prompt")

    model = args.model
    if not model:
        with urllib.request.urlopen(f"{args.url}/v1/models", timeout=30.0) as response:
            listing = json.loads(response.read().decode("utf-8"))
        entries = listing.get("data") or []
        if len(entries) != 1:
            parser.error(f"pass --model: /v1/models offers {len(entries)} entries")
        model = entries[0]["id"]

    # A prompt is a whole file or a whole literal, never a truncated one: the
    # engine's `prompt_tokens` is what gets reported, so a prompt cut here to an
    # approximate length would make the reported context an approximation of a
    # number somebody chose. Cutting to an exact token count is a property of how
    # the file was prepared, not of the measurement.
    prompts: list[tuple[str, str]] = []
    for path in args.prompt_file:
        with open(path, encoding="utf-8") as handle:
            prompts.append((path, handle.read()))
    for index, text in enumerate(args.prompt):
        prompts.append((f"literal[{index}]", text))

    samples: list[Sample] = []
    for label, text in prompts:
        for repeat in range(args.repeats):
            name = label if args.repeats == 1 else f"{label}#{repeat + 1}"
            sample = measure(args.url, model, name, text, args.max_tokens)
            samples.append(sample)
            print(
                f"{name:>34}  prompt={sample.prompt_tokens:>6}  gen={sample.generation_tokens:>4}  "
                f"ttft={sample.ttft:7.3f}s  prefill={sample.prefill_tps:8.1f} tok/s  "
                f"decode={sample.decode_tps:7.2f} tok/s  request={sample.duration:7.3f}s"
            )
            sys.stdout.flush()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"model": model, "url": args.url, "samples": [item.as_dict() for item in samples]}, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
