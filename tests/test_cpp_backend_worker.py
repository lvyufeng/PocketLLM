"""Test C++ backend worker loop support for TP supervision."""

import pytest

from pocketllm import EngineArgs
from pocketllm.backends.cpp_backend import CppBackend


def test_cpp_backend_run_worker_requires_nonzero_rank():
    """run_worker should reject rank 0."""
    try:
        from pocketllm.backends.cpp_backend import load_native_module
        load_native_module()
    except Exception:
        pytest.skip("C++ backend not available")

    args = EngineArgs(
        model="/fake/path",
        backend="cpp",
        tensor_parallel_size=2,
        tensor_parallel_rank=0,
    )

    # Mock engine to avoid actual construction
    backend = CppBackend(args, engine=object())

    with pytest.raises(RuntimeError, match="run_worker must not be called on rank 0"):
        backend.run_worker()


def test_cpp_backend_run_worker_requires_constructed_engine():
    """run_worker should reject None engine."""
    try:
        from pocketllm.backends.cpp_backend import load_native_module
        load_native_module()
    except Exception:
        pytest.skip("C++ backend not available")

    args = EngineArgs(
        model="/fake/path",
        backend="cpp",
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
    )

    # Create a minimal mock engine that has run_worker_loop attribute
    class MockEngine:
        def run_worker_loop(self):
            pass

    # Pass mock engine to avoid construction, then set to None
    backend = CppBackend(args, engine=MockEngine())
    backend._engine = None

    with pytest.raises(RuntimeError, match="native engine is not constructed"):
        backend.run_worker()


def test_cpp_backend_exposes_run_worker_loop():
    """Verify native module exposes run_worker_loop method."""
    try:
        from pocketllm.backends.cpp_backend import load_native_module
        native = load_native_module()
    except Exception:
        pytest.skip("C++ backend not available")

    # Check that PersistentEngine has run_worker_loop
    assert hasattr(native, "PersistentEngine"), "Native module should expose PersistentEngine"

    # We can't instantiate without a real checkpoint, but we can check the binding exists
    # by inspecting the class's methods
    engine_class = native.PersistentEngine
    assert hasattr(engine_class, "run_worker_loop"), "PersistentEngine should expose run_worker_loop"
