#!/usr/bin/env python3
"""Run a bench while polling the engine's `/metrics`, keeping the last scrape.

`bench_serving.py`'s JSON has no server-side series, so the prefill / queue /
decode split has to come from a scrape taken while the server is alive. The
counters are cumulative, so the last successful scrape before teardown is the
run's total minus at most one poll interval of tail.

That interval is not a rounding detail. The bench stops the server as soon as
the last request returns, which is often sooner than the next poll, so a scrape
routinely lands while requests are still in flight and reports fewer of them
than were served -- measured on the width ladder, L4 recorded 3 of its 4
requests and L1 0 of its 1. `--server-drain-seconds` on the bench is what makes
the artifact complete; pass it here, and the run record says whether it was
passed. The poller itself cannot fix this: the server is a child of the bench
and goes away with it.

usage: _serving_metrics_scrape.py <port> <out.metrics> -- <command...>
"""
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

POLL_SECONDS = 0.1


def main() -> int:
    port = sys.argv[1]
    out_path = pathlib.Path(sys.argv[2])
    sep = sys.argv.index("--")
    args = sys.argv[sep + 1:]

    url = f"http://127.0.0.1:{port}/metrics"
    scrapes = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    scrapes.append((time.monotonic(), r.read().decode()))
            except Exception:
                # The server is not up yet, or has already gone. Both are
                # expected at the ends of the run; a later scrape supersedes.
                pass
            stop.wait(POLL_SECONDS)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    proc = subprocess.run(args, capture_output=True, text=True)
    stop.set()
    t.join(timeout=2)

    if scrapes:
        out_path.write_text(scrapes[-1][1], encoding="utf-8")
        print(f"[scrape] {len(scrapes)} scrapes; wrote {out_path}", file=sys.stderr)
    else:
        print("[scrape] no scrape succeeded", file=sys.stderr)

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
