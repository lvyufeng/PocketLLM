#!/usr/bin/env python3
"""Attribute batch-mode single-request overhead to a specific layer.

The batch path is ~10% slower than the serial path for one request.  Wall time
alone cannot say whether that sits in the Python submit/poll plumbing or inside
the scheduler loop, so print both clocks for the same runs:

  * Python wall  - generate() entry to return, everything included.
  * scheduler    - submit_time to completion_time, measured in C++.

A gap between them is plumbing (thread handoff, poll wakeup, result copy).  A
scheduler time that already exceeds the serial baseline puts the cost in the
loop, i.e. in the per-step work around batch_decode_step.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketllm import EngineArgs, LLM, SamplingParams


def run(checkpoint, *, enable_batching, max_batch_size, tp, max_model_len,
        prompt_length, max_tokens, warmup, runs):
    args = EngineArgs(
        model=checkpoint,
        backend="cpp",
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        backend_options={
            "enable_batching": enable_batching,
            "max_batch_size": max_batch_size if enable_batching else 1,
        },
    )
    llm = LLM(args)
    label = "batch" if enable_batching else "serial"
    actual = bool(getattr(llm.backend, "_batching_enabled", False))
    if actual != enable_batching:
        llm.close()
        raise SystemExit(f"{label}: backend reports _batching_enabled={actual}")

    prompt = list(range(1, prompt_length + 1))
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    walls, engine_times, ttfts = [], [], []
    try:
        for _ in range(warmup):
            llm.generate([prompt], sampling_params=sampling)
        for _ in range(runs):
            start = time.perf_counter()
            result = llm.generate([prompt], sampling_params=sampling)[0]
            walls.append(time.perf_counter() - start)
            engine_times.append(result.timings.total_seconds)
            ttfts.append(result.timings.ttft_seconds)
            assert len(result.token_ids) == max_tokens, len(result.token_ids)
    finally:
        llm.close()

    avg = lambda xs: sum(xs) / len(xs)
    mode = f"{label}(slots={max_batch_size if enable_batching else 1})"
    # Per-run values, because the averages moved between sweeps by more than the
    # effect being measured; a spread this visible has to be reported, not hidden.
    print(f"  {mode} walls: " + " ".join(f"{w:.4f}" for w in walls), flush=True)
    return {
        "mode": mode,
        "wall": avg(walls),
        "engine": avg(engine_times),
        "ttft": avg(ttfts),
        "min_wall": min(walls),
        "max_wall": max(walls),
        "per_step_wall": avg(walls) / max_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--test-runs", type=int, default=5)
    parser.add_argument("--slots", type=int, nargs="+", default=[2, 8])
    args = parser.parse_args()

    common = dict(
        tp=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        warmup=args.warmup_runs,
        runs=args.test_runs,
    )

    # Repeat the serial baseline at the end as well.  The first sweep put 8 slots
    # at 1.005x and the second at 1.117x, so the baseline itself has to be shown
    # to be stable before any slot-count claim means anything.
    rows = [run(args.checkpoint, enable_batching=False, max_batch_size=1, **common)]
    for slots in args.slots:
        rows.append(run(args.checkpoint, enable_batching=True, max_batch_size=slots, **common))
    rows.append(run(args.checkpoint, enable_batching=False, max_batch_size=1, **common))

    baseline = rows[0]["wall"]
    print()
    print(f"{'mode':<20} {'wall':>9} {'engine':>9} {'min':>9} {'max':>9} {'ms/step':>9} {'vs serial':>10}")
    for row in rows:
        print(
            f"{row['mode']:<20} {row['wall']:>8.4f}s {row['engine']:>8.4f}s "
            f"{row['min_wall']:>8.4f}s {row['max_wall']:>8.4f}s "
            f"{row['per_step_wall']*1000:>8.2f} "
            f"{row['wall']/baseline:>9.3f}x"
        )
    print()
    for row in rows[1:]:
        gap = row["wall"] - row["engine"]
        print(
            f"{row['mode']}: plumbing (wall-engine) {gap*1000:.2f}ms, "
            f"loop excess over serial {(row['engine']-baseline)*1000:.2f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
