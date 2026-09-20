#!/usr/bin/env python3
"""Tables for a serving sweep, read from the artifacts `run_serving_sweep.sh` writes.

Three views, selected by a flag. Each one reads a file the sweep produced and
prints a table; none of them recomputes anything the bench already reported.

  --client     (default) one row per `<tag>.json`. Latency percentiles and
               throughput as `bench_serving.py` recorded them.
  --server     one row per `<tag>.metrics`, the last `/metrics` scrape. This is
               the only place queue time, prefill time and decode time are
               separable, because the engine exports them as histograms.
  --occupancy  time-weighted batch width from the client's own token timestamps.
               Two runs can report the same aggregate throughput while holding
               the server at very different widths, and this is what shows it.
               Needs the record's per-request `start_seconds`, which is the send
               time offset from the first send: a refilled batch cannot be placed
               on a shared axis from per-request relative times. A refilled record
               predating that field is flagged and its widths are a lower bound.

usage: summarize_serving_sweep.py [--client|--server|--occupancy] <file> [file ...]

Percentiles are nested one level deeper than the metric itself --
`metrics["ttft"]["percentiles"]["99.0"]`, with a string key -- and the console
table these replace has no Std column at all.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

# Server-side histograms. Every one is exported without labels, so the match has
# to be anchored to the start of a line and run with re.M: a substring search
# would also hit `pocket_ttft_seconds_sum` while looking for a longer name.
SERVER_HISTOGRAMS = [
    ("pocket_request_queue_time_seconds", "queue"),
    ("pocket_request_prefill_time_seconds", "prefill"),
    ("pocket_request_decode_time_seconds", "decode"),
    ("pocket_inter_token_latency_seconds", "itl"),
    ("pocket_request_duration_seconds", "e2el"),
    ("pocket_ttft_seconds", "ttft"),
]


def _percentile(metric: dict, key: str = "99.0") -> float:
    return metric.get("percentiles", {}).get(key, 0.0)


def _tag(path: str) -> str:
    return pathlib.Path(path).stem


def client(paths: list[str]) -> None:
    hdr = (f"{'run':>12} {'slots':>5} {'conc':>4} {'prompts':>7} {'ok':>4} {'fail':>4} "
           f"{'dur_s':>7} {'tok/s':>7} {'req/s':>6} {'good':>6} {'peak':>4} "
           f"{'TTFT':>8} {'TTFTp99':>9} {'TPOT':>8} {'TPOTsd':>8} {'ITL':>8} "
           f"{'E2EL_s':>7} {'outtok':>7}")
    print(hdr)
    print("-" * len(hdr))
    for path in paths:
        d = json.loads(pathlib.Path(path).read_text())
        m = d["metrics"]
        # The launch knobs live under `server`; a record written before that
        # block existed has no slot count at all, and a bench pointed at an
        # already-running server (`--base-url`) cannot know one either.
        slots = (d.get("server") or {}).get("max_batch_size", d.get("max_batch_size"))
        slots_cell = "--" if slots is None else str(slots)
        ok = sum(1 for r in d["requests"] if r.get("success"))
        # `prompt_tokens` is null and there is no `output_len`; the generated
        # count per request is `output_tokens`.
        lens = [r["output_tokens"] for r in d["requests"] if r.get("success")]
        span = f"{min(lens)}/{max(lens)}" if lens else "--"
        print(f"{_tag(path):>12} {slots_cell:>5} {d.get('max_concurrency', 0):>4} "
              f"{d.get('num_prompts', 0):>7} {ok:>4} {len(d['requests']) - ok:>4} "
              f"{m.get('duration_seconds', 0):>7.1f} {m.get('output_throughput', 0):>7.2f} "
              f"{m.get('request_throughput', 0):>6.3f} {m.get('request_goodput', 0):>6.3f} "
              f"{m.get('max_concurrent_requests', 0):>4} "
              f"{m['ttft']['mean'] * 1000:>8.1f} {_percentile(m['ttft']) * 1000:>9.1f} "
              f"{m['tpot']['mean'] * 1000:>8.2f} {m['tpot']['std'] * 1000:>8.2f} "
              f"{m['itl']['mean'] * 1000:>8.2f} {m['e2el']['mean']:>7.3f} "
              f"{m.get('total_output_tokens', 0):>7}"
              f"   out_len {span}")


def server(paths: list[str]) -> None:
    hdr = f"{'run':>12} " + " ".join(f"{name:>20}" for _, name in SERVER_HISTOGRAMS)
    print(hdr)
    print("-" * len(hdr))
    for path in paths:
        text = pathlib.Path(path).read_text()
        cells = []
        for metric, _ in SERVER_HISTOGRAMS:
            total = re.findall(rf"^{re.escape(metric)}_sum ([0-9.eE+-]+)$", text, re.M)
            count = re.findall(rf"^{re.escape(metric)}_count ([0-9.eE+-]+)$", text, re.M)
            if total and count and float(count[0]) > 0:
                mean = float(total[0]) / float(count[0])
                cells.append(f"{float(total[0]):>8.1f}s/{mean * 1000:>7.1f}ms")
            else:
                # A rank that died before serving leaves the counters at zero,
                # and a run whose server never came up leaves no scrape at all.
                cells.append(f"{'--':>20}")
        print(f"{_tag(path):>12} " + " ".join(f"{c:>20}" for c in cells))


def occupancy(paths: list[str], bucket_seconds: float = 1.0) -> None:
    for path in paths:
        d = json.loads(pathlib.Path(path).read_text())
        events: list[tuple[float, int]] = []   # (t, +1 open / -1 close)
        tokens: list[float] = []
        # A record with no more prompts than the client's in-flight cap sends
        # every request at once, so a zero offset is exact and the field is not
        # needed. Only a refilled run has to be told where each request started.
        refilled = d.get("num_prompts", 0) > (d.get("max_concurrency") or 0)
        placed = True
        for r in d["requests"]:
            if not r.get("success"):
                continue
            # `ttft_seconds` and `itl_seconds` are relative to the request, so a
            # run that refills its batch needs `start_seconds` -- the client's own
            # send time as an offset from the first send -- to put the requests
            # on one axis. Without it every request would be assumed sent at
            # once, which reads as a wave wider than the slot count allows.
            send = r.get("start_seconds")
            if send is None:
                placed = placed and not refilled
                send = 0.0
            opens = send + r["ttft_seconds"]
            tokens.append(opens)
            t = opens
            for itl in r["itl_seconds"]:
                t += itl
                tokens.append(t)
            events.append((opens, +1))
            events.append((t, -1))
        if not events:
            print(f"--- {path}  no successful requests")
            continue
        events.sort()
        span = max(t for t, _ in events)
        if span <= 0:
            print(f"--- {_tag(path)}  all requests start and end together")
            continue
        buckets = int(span / bucket_seconds) + 1
        # Width-seconds per bucket, integrated across the intervals between
        # events rather than sampled at the boundaries: at a width of 1 a
        # request that opens and closes inside one bucket would otherwise be
        # counted only when the bucket happened to end inside its lifetime.
        held = [0.0] * buckets
        live = 0
        peak = 0
        previous = events[0][0]
        for t, delta in events:
            if t > previous:
                first = int(previous / bucket_seconds)
                last = int(t / bucket_seconds)
                if first == last:
                    held[first] += live * (t - previous)
                else:
                    held[first] += live * ((first + 1) * bucket_seconds - previous)
                    for b in range(first + 1, last):
                        held[b] += live * bucket_seconds
                    held[last] += live * (t - last * bucket_seconds)
            live += delta
            peak = max(peak, live)
            previous = t
        per_bucket = collections.Counter(int(t / bucket_seconds) for t in tokens)
        note = "" if placed else "  [!] no start_seconds on a refilled run: widths are a lower bound"
        print(f"--- {_tag(path)}  span={span:.1f}s  tokens={len(tokens)}  "
              f"mean_width={sum(held) / span:.2f}  max_width={peak}{note}")
        print("   t(s)  width  tok/s")
        shown = None
        for b in range(buckets):
            value = held[b] / bucket_seconds
            rounded = round(value, 2)
            if b % 2 == 0 or rounded != shown:
                print(f"  {b * bucket_seconds:5.0f} {rounded:6.2f} {per_bucket.get(b, 0):6d}")
            shown = rounded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--client", action="store_true", help="Per-run table from the bench JSONs (default).")
    view.add_argument("--server", action="store_true", help="Per-run table from the /metrics scrapes.")
    view.add_argument("--occupancy", action="store_true", help="Batch width over time from client timestamps.")
    parser.add_argument("--bucket-seconds", type=float, default=1.0, help="Occupancy bucket width.")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    if args.server:
        server(args.files)
    elif args.occupancy:
        occupancy(args.files, args.bucket_seconds)
    else:
        client(args.files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
