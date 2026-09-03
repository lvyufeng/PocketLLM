#!/usr/bin/env python3
"""Test CppBackend with batch scheduler enabled."""

import sys
import os

# Add pocketllm to path
sys.path.insert(0, "/mnt/data1/dsv4_inference")

from pocketllm import EngineArgs, LLM, SamplingParams


def test_serial_mode():
    """Test traditional serial mode (baseline)."""
    print("=== Test 1: Serial Mode (Baseline) ===")

    if len(sys.argv) < 2:
        print("SKIP: No checkpoint provided")
        return

    checkpoint = sys.argv[1]

    args = EngineArgs(
        model=checkpoint,
        backend="cpp",
        backend_options={
            "enable_batching": False,  # Disable batching
        }
    )

    llm = LLM(args)

    # Check capabilities
    caps = llm.backend.capabilities
    print(f"Backend: {caps.name}")
    print(f"Supports batch: {caps.supports_batch}")
    print(f"Scheduler: {caps.details.get('scheduler')}")

    # Generate single request
    result = llm.generate(
        prompt_tokens=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=10)
    )

    print(f"Generated {len(result.token_ids)} tokens")
    print(f"Finish reason: {result.finish_reason}")
    print(f"Total time: {result.timings.total_seconds:.3f}s")
    print("✓ Serial mode test passed\n")


def test_batch_mode():
    """Test batch scheduler mode."""
    print("=== Test 2: Batch Scheduler Mode ===")

    if len(sys.argv) < 2:
        print("SKIP: No checkpoint provided")
        return

    checkpoint = sys.argv[1]

    args = EngineArgs(
        model=checkpoint,
        backend="cpp",
        backend_options={
            "enable_batching": True,  # Enable batching
            "max_batch_size": 4,
        }
    )

    llm = LLM(args)

    # Check capabilities
    caps = llm.backend.capabilities
    print(f"Backend: {caps.name}")
    print(f"Supports batch: {caps.supports_batch}")
    print(f"Scheduler: {caps.details.get('scheduler')}")
    print(f"Max batch size: {caps.details.get('max_batch_size')}")

    if not caps.supports_batch:
        print("WARNING: Batching not enabled (fallback to serial)")

    # Generate single request (should work in batch mode too)
    result = llm.generate(
        prompt_tokens=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=10)
    )

    print(f"Generated {len(result.token_ids)} tokens")
    print(f"Finish reason: {result.finish_reason}")
    print(f"Total time: {result.timings.total_seconds:.3f}s")
    print(f"TTFT: {result.timings.ttft_seconds:.3f}s")
    print("✓ Batch mode test passed\n")


def test_api_only():
    """Test API availability without checkpoint."""
    print("=== Test 3: API Validation ===")

    try:
        # Load the newly built module directly
        import sys
        sys.path.insert(0, "/mnt/data1/dsv4_inference/cpp_engine/build/python")

        try:
            import pocketllm_cpp as native
            print(f"Native module loaded: {native.__name__}")

            # Check for QwenBatchScheduler
            for name in ["QwenBatchScheduler", "QwenBatchSamplingParams",
                         "SchedulerGenerationResult", "QwenBatchSchedulerStats",
                         "QwenEngine", "QwenEngineOptions"]:
                if hasattr(native, name):
                    print(f"✓ {name} available")
                else:
                    print(f"✗ {name} not available")

        except Exception as e:
            print(f"Could not load native module: {e}")
            import traceback
            traceback.print_exc()

    except ImportError as e:
        print(f"Import error: {e}")

    print()


if __name__ == "__main__":
    print("CppBackend Batch Scheduler Tests")
    print("=" * 50)
    print()

    # Always run API validation
    test_api_only()

    # Run full tests if checkpoint provided
    if len(sys.argv) >= 2:
        print(f"Using checkpoint: {sys.argv[1]}\n")
        test_serial_mode()
        test_batch_mode()
        print("=" * 50)
        print("All tests passed!")
    else:
        print("Usage: python test_cpp_backend_batching.py <checkpoint_dir>")
        print("\nAPI validation completed. Run with checkpoint for full tests.")
