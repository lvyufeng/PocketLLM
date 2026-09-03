#!/usr/bin/env python3
"""Compare the per-process Phase 3.5 measurements and apply the targets.

run_phase35_benchmarks.sh writes one JSON per (mode, concurrency) because both
modes cannot share a process without the second engine paying a ~10% penalty.
This reads those files and does the comparison the bench scripts used to do
in-process.
"""

import argparse
import json
import sys
from pathlib import Path

LATENCY_TARGET = 1.05
THROUGHPUT_TARGETS = {2: 1.7, 4: 3.0, 8: 4.5}


def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    d = args.results_dir
    failures = []
    missing = []

    print("=" * 64)
    print("Phase 3.5 Summary")
    print("=" * 64)

    serial = load(d / "single_serial.json")
    batch = load(d / "single_batch.json")
    print("\nSingle-request latency (target: batch <= %.2fx serial)" % LATENCY_TARGET)
    if not serial or not batch:
        missing.append("single-request latency")
        print("  missing measurement; cannot compare")
    else:
        ratio = batch["avg_latency"] / serial["avg_latency"]
        print(f"  serial {serial['avg_latency']:.4f}s   batch {batch['avg_latency']:.4f}s"
              f"   ratio {ratio:.3f}x")
        if ratio <= LATENCY_TARGET:
            print("  PASS")
        else:
            print(f"  FAIL (over by {(ratio - LATENCY_TARGET) * 100:.1f} points)")
            failures.append(f"single-request latency {ratio:.3f}x")

    print("\nConcurrent throughput")
    for n in args.concurrency:
        s = load(d / f"concurrent_{n}_serial.json")
        b = load(d / f"concurrent_{n}_batch.json")
        target = THROUGHPUT_TARGETS.get(n, 1.0)
        if not s or not b:
            missing.append(f"concurrency {n}")
            print(f"  {n:>2} concurrent: missing measurement")
            continue
        ratio = b["throughput"] / s["throughput"]
        verdict = "PASS" if ratio >= target else "FAIL"
        print(f"  {n:>2} concurrent: serial {s['throughput']:.3f} req/s   "
              f"batch {b['throughput']:.3f} req/s   {ratio:.2f}x "
              f"(target >={target}x)  {verdict}")
        if ratio < target:
            failures.append(f"concurrency {n} {ratio:.2f}x < {target}x")

    print("\n" + "=" * 64)
    if missing:
        print("INCOMPLETE: " + "; ".join(missing))
    if failures:
        print("FAIL: " + "; ".join(failures))
    elif not missing:
        print("PASS: all Phase 3.5 targets met")
    return 1 if (failures or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
