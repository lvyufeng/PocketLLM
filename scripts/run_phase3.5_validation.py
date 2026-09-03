#!/usr/bin/env python3
"""
Comprehensive performance validation suite for Phase 3.5.

Runs all benchmarks and generates a summary report.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_benchmark(script: str, checkpoint: str, extra_args: list = None) -> dict:
    """Run a benchmark script and capture results."""
    cmd = [sys.executable, script, checkpoint]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive Phase 3.5 performance validation"
    )
    parser.add_argument(
        "checkpoint",
        help="Path to Qwen3.5 checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        default="phase3.5_results",
        help="Directory to save results (default: phase3.5_results)"
    )
    parser.add_argument(
        "--skip-single-latency",
        action="store_true",
        help="Skip single-request latency test"
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="Skip concurrent throughput test"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per request (default: 128)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print("Phase 3.5 Performance Validation Suite")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")

    results = {
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "max_tokens": args.max_tokens,
        "benchmarks": {}
    }

    scripts_dir = Path(__file__).parent

    # Test 1: Single-request latency
    if not args.skip_single_latency:
        print("\n" + "="*70)
        print("Test 1: Single-Request Latency (Regression Test)")
        print("="*70)

        result = run_benchmark(
            str(scripts_dir / "bench_single_request_latency.py"),
            args.checkpoint,
            ["--max-tokens", str(args.max_tokens)]
        )

        results["benchmarks"]["single_request_latency"] = result

        # Save output
        output_file = output_dir / f"single_latency_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(result["stdout"])
            if result["stderr"]:
                f.write("\n\n=== STDERR ===\n")
                f.write(result["stderr"])

        print(f"\n{'='*70}")
        if result["returncode"] == 0:
            print("✅ Single-request latency test PASSED")
        else:
            print("❌ Single-request latency test FAILED")
        print(f"Output saved to: {output_file}")

    # Test 2: Concurrent throughput
    if not args.skip_concurrent:
        print("\n" + "="*70)
        print("Test 2: Concurrent Request Throughput")
        print("="*70)

        result = run_benchmark(
            str(scripts_dir / "bench_concurrent_throughput.py"),
            args.checkpoint,
            [
                "--num-concurrent", "2", "4", "8",
                "--max-tokens", str(args.max_tokens)
            ]
        )

        results["benchmarks"]["concurrent_throughput"] = result

        # Save output
        output_file = output_dir / f"concurrent_throughput_{timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(result["stdout"])
            if result["stderr"]:
                f.write("\n\n=== STDERR ===\n")
                f.write(result["stderr"])

        print(f"\n{'='*70}")
        if result["returncode"] == 0:
            print("✅ Concurrent throughput test PASSED")
        else:
            print("❌ Concurrent throughput test FAILED")
        print(f"Output saved to: {output_file}")

    # Save JSON summary
    json_file = output_dir / f"results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("Phase 3.5 Validation Summary")
    print("="*70)

    all_passed = True
    for test_name, test_result in results["benchmarks"].items():
        status = "✅ PASS" if test_result["returncode"] == 0 else "❌ FAIL"
        print(f"{test_name}: {status}")
        if test_result["returncode"] != 0:
            all_passed = False

    print(f"\nResults directory: {output_dir}")
    print(f"JSON summary: {json_file}")

    if all_passed:
        print("\n🎉 All Phase 3.5 validation tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some Phase 3.5 validation tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
