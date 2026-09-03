#!/usr/bin/env python3
"""Simple concurrent throughput benchmark."""

import argparse
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketllm import EngineArgs, LLM, SamplingParams


def benchmark_serial(llm, prompt, sampling, num_requests):
    """Benchmark serial execution."""
    start = time.perf_counter()
    for _ in range(num_requests):
        llm.generate([prompt], sampling_params=sampling)
    end = time.perf_counter()
    return end - start


def benchmark_concurrent(llm, prompt, sampling, num_requests):
    """Benchmark concurrent execution."""
    results = []

    def worker(idx):
        start = time.perf_counter()
        result = llm.generate([prompt], sampling_params=sampling)
        end = time.perf_counter()
        results.append((idx, end - start, len(result[0].token_ids)))

    threads = []
    start = time.perf_counter()
    for i in range(num_requests):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    end = time.perf_counter()

    return end - start, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Model checkpoint path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-concurrent", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--test-runs", type=int, default=3)
    args = parser.parse_args()

    print("="*60)
    print("Concurrent Throughput Benchmark")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tensor parallel: {args.tensor_parallel_size}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Create LLM
    engine_args = EngineArgs(
        model=args.checkpoint,
        backend="cpp",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=2048,
        backend_options={"enable_batching": False}
    )

    print("Initializing LLM...", flush=True)
    llm = LLM(engine_args)
    print("✓ LLM ready\n", flush=True)

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompt = list(range(1, args.prompt_length + 1))

    all_results = {}

    for num_concurrent in args.num_concurrent:
        print(f"{'='*60}")
        print(f"Testing {num_concurrent} concurrent requests")
        print(f"{'='*60}")

        # Serial baseline
        print(f"Serial baseline ({num_concurrent} requests)...")
        serial_times = []
        for run in range(args.test_runs):
            t = benchmark_serial(llm, prompt, sampling, num_concurrent)
            serial_times.append(t)
            print(f"  Run {run+1}: {t:.3f}s")
        avg_serial = sum(serial_times) / len(serial_times)

        # Concurrent
        print(f"\nConcurrent ({num_concurrent} requests)...")
        concurrent_times = []
        for run in range(args.test_runs):
            t, _ = benchmark_concurrent(llm, prompt, sampling, num_concurrent)
            concurrent_times.append(t)
            print(f"  Run {run+1}: {t:.3f}s")
        avg_concurrent = sum(concurrent_times) / len(concurrent_times)

        speedup = avg_serial / avg_concurrent
        print(f"\nAverage serial: {avg_serial:.3f}s")
        print(f"Average concurrent: {avg_concurrent:.3f}s")
        print(f"Speedup: {speedup:.2f}×\n")

        all_results[num_concurrent] = {
            "serial": avg_serial,
            "concurrent": avg_concurrent,
            "speedup": speedup
        }

    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    for n, r in all_results.items():
        print(f"{n} concurrent: {r['speedup']:.2f}× speedup")

    llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
