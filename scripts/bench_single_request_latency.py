#!/usr/bin/env python3
"""
Benchmark single-request latency for batch scheduler regression testing.

This script tests that enabling batching does not introduce significant latency
overhead for single requests. Target: ≤1.05× baseline (serial mode).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketllm import EngineArgs, LLM, SamplingParams


def benchmark_single_request(
    checkpoint: str,
    enable_batching: bool,
    max_tokens: int = 128,
    warmup_runs: int = 3,
    test_runs: int = 10,
    prompt_length: int = 32,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    max_batch_size: int = 8,
):
    """Benchmark single request latency."""

    mode = "batch" if enable_batching else "serial"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {mode.upper()} mode")
    print(f"{'='*60}")

    # Create LLM with specified mode
    args = EngineArgs(
        model=checkpoint,
        backend="cpp",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        backend_options={
            "enable_batching": enable_batching,
            # Several slots even though this test sends one request at a time: the
            # point is to measure what a batch-configured engine costs a lone
            # request.  Sweeping this separates scheduler overhead, which is
            # slot-count independent, from any per-slot KV layout cost.
            "max_batch_size": max_batch_size if enable_batching else 1,
        }
    )

    llm = LLM(args)

    # Check capabilities
    caps = llm.backend.capabilities
    print(f"Backend: {caps.name}")
    print(f"Scheduler: {caps.details.get('scheduler')}")
    print(f"Supports batch: {caps.supports_batch}")

    # A silent fall back to the serial path would be reported as a batch
    # measurement, which is exactly the comparison this script exists to make.
    actually_batching = bool(getattr(llm.backend, "_batching_enabled", False))
    if actually_batching != enable_batching:
        llm.close()
        raise SystemExit(
            f"requested enable_batching={enable_batching} but backend reports "
            f"{actually_batching}; refusing to report this as a {mode} measurement"
        )

    # Create test prompt (dummy token IDs)
    prompt_tokens = list(range(1, prompt_length + 1))
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    print(f"\nPrompt length: {prompt_length} tokens")
    print(f"Max new tokens: {max_tokens}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Test runs: {test_runs}")

    latencies = []
    ttfts = []
    try:
        # Warmup
        print(f"\nWarmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            result = llm.generate([prompt_tokens], sampling_params=sampling)[0]
            print(f"  Run {i+1}: {result.timings.total_seconds:.3f}s")

        # Benchmark
        print(f"\nBenchmark ({test_runs} runs)...")

        for i in range(test_runs):
            start = time.perf_counter()
            result = llm.generate([prompt_tokens], sampling_params=sampling)[0]
            end = time.perf_counter()

            latency = end - start
            ttft = result.timings.ttft_seconds

            latencies.append(latency)
            ttfts.append(ttft)

            print(f"  Run {i+1}: {latency:.3f}s (TTFT: {ttft:.3f}s, generated: {len(result.token_ids)} tokens)")
    finally:
        # Serial and batch modes each build their own LLM.  Closing releases the
        # engine, which is what tells rank 0 to send the TP workers a shutdown;
        # otherwise the serial run's workers stay resident through the batch run.
        llm.close()

    # Statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    avg_ttft = sum(ttfts) / len(ttfts)
    min_ttft = min(ttfts)
    max_ttft = max(ttfts)

    print(f"\n{'='*60}")
    print(f"Results Summary ({mode.upper()} mode)")
    print(f"{'='*60}")
    print(f"Total Latency:")
    print(f"  Average: {avg_latency:.3f}s")
    print(f"  Min:     {min_latency:.3f}s")
    print(f"  Max:     {max_latency:.3f}s")
    print(f"\nTime to First Token (TTFT):")
    print(f"  Average: {avg_ttft:.3f}s")
    print(f"  Min:     {min_ttft:.3f}s")
    print(f"  Max:     {max_ttft:.3f}s")

    return {
        "mode": mode,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "avg_ttft": avg_ttft,
        "min_ttft": min_ttft,
        "max_ttft": max_ttft,
        "latencies": latencies,
        "ttfts": ttfts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark single-request latency (regression test)"
    )
    parser.add_argument(
        "checkpoint",
        help="Path to Qwen3.5 checkpoint directory"
    )
    # Building a second engine in one process costs the *second* one about 10%,
    # whichever mode it is: four serial engines in a row gave 0.672 / 0.735 /
    # 0.675 / 0.674s at identical GPU clocks.  Measuring both modes in one
    # process therefore charges that to whichever ran second, which is why the
    # driver gives each mode its own process and collects these JSON files.
    parser.add_argument(
        "--single",
        choices=["serial", "batch"],
        help="Measure only this mode, then write --json-out and exit"
    )
    parser.add_argument(
        "--json-out",
        help="Write the --single measurement here as JSON"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per request (default: 128)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=10,
        help="Number of test runs (default: 10)"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=32,
        help="Prompt length in tokens (default: 32)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max sequence length; bounds per-slot KV cache (default: 4096)"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Slots to configure in batch mode (default: 8)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Single-Request Latency Benchmark")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Batch slots: {args.max_batch_size}")

    if args.single:
        result = benchmark_single_request(
            args.checkpoint,
            enable_batching=args.single == "batch",
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            prompt_length=args.prompt_length,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_batch_size=args.max_batch_size,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2))
        return 0

    # Benchmark serial mode (baseline)
    serial_result = benchmark_single_request(
        args.checkpoint,
        enable_batching=False,
        max_tokens=args.max_tokens,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        prompt_length=args.prompt_length,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    # Benchmark batch mode
    batch_result = benchmark_single_request(
        args.checkpoint,
        enable_batching=True,
        max_tokens=args.max_tokens,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
        prompt_length=args.prompt_length,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_batch_size=args.max_batch_size,
    )

    # Compare results
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")

    latency_ratio = batch_result["avg_latency"] / serial_result["avg_latency"]

    print(f"Average Latency:")
    print(f"  Serial: {serial_result['avg_latency']:.3f}s")
    print(f"  Batch:  {batch_result['avg_latency']:.3f}s")
    print(f"  Ratio:  {latency_ratio:.3f}× ({'+' if latency_ratio > 1 else ''}{(latency_ratio-1)*100:.1f}%)")

    print(f"\nAverage TTFT:")
    print(f"  Serial: {serial_result['avg_ttft']:.3f}s")
    print(f"  Batch:  {batch_result['avg_ttft']:.3f}s")
    # A path that cannot separate the first token reports 0.0, and the latency
    # ratio above is the verdict either way, so say so instead of dividing.
    if serial_result["avg_ttft"] > 0.0:
        ttft_ratio = batch_result["avg_ttft"] / serial_result["avg_ttft"]
        print(f"  Ratio:  {ttft_ratio:.3f}× ({'+' if ttft_ratio > 1 else ''}{(ttft_ratio-1)*100:.1f}%)")
    else:
        print("  Ratio:  n/a (serial path does not report TTFT)")

    # Verdict
    print(f"\n{'='*60}")
    print("Verdict")
    print(f"{'='*60}")

    target = 1.05
    if latency_ratio <= target:
        print(f"✅ PASS: Batch mode latency is {latency_ratio:.3f}× baseline (≤{target}×)")
    else:
        print(f"❌ FAIL: Batch mode latency is {latency_ratio:.3f}× baseline (>{target}×)")
        print(f"   Overhead: {(latency_ratio-1)*100:.1f}% (target: ≤5%)")

    return 0 if latency_ratio <= target else 1


if __name__ == "__main__":
    sys.exit(main())
