"""Preflight checks for the native C++ engine build.

`POCKETLLM_BUILD_CPP` defaults to on, so a plain `pip install pocketllm` expects to
build the native module. When a prerequisite is missing the build has to stop -
silently continuing without `pocketllm_cpp` would let `backend="auto"` fall back to
Torch and hand the caller different kernels than requested.

Where it stops matters as much as that it stops. These tests pin the check ahead of
the Torch CUDA extensions, which compile for minutes; reporting a missing cmake
afterwards makes the user wait through a compile that could never succeed.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("torch", reason="setup.py only defines the native build path when torch is importable")

SETUP_PY = Path(__file__).resolve().parents[1] / "setup.py"
PYTORCH_ONLY_HINT = "POCKETLLM_BUILD_CPP=0"


def _load_setup():
    """Execute setup.py for its definitions without invoking setup().

    Loaded as a real module so monkeypatch can replace the globals that the
    methods under test resolve by name.
    """
    spec = importlib.util.spec_from_file_location("setup_under_test", SETUP_PY)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("setuptools.setup"):
        spec.loader.exec_module(module)
    return module


def _without_cmake(monkeypatch):
    real_which = shutil.which

    def which(name, *args, **kwargs):
        if name == "cmake":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)


def _without_pybind11(monkeypatch):
    # A None entry in sys.modules makes `import pybind11` raise ImportError, which
    # is how a machine without pybind11 installed behaves.
    monkeypatch.setitem(sys.modules, "pybind11", None)


def test_satisfied_prerequisites_report_nothing():
    setup = _load_setup()
    assert setup._missing_native_build_prerequisites() == []
    setup._require_native_build_prerequisites()


def test_missing_cmake_names_cmake_and_the_pytorch_only_alternative(monkeypatch):
    setup = _load_setup()
    _without_cmake(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        setup._require_native_build_prerequisites()

    message = str(excinfo.value)
    assert "cmake" in message
    assert PYTORCH_ONLY_HINT in message
    assert "--no-build-isolation" in message


def test_missing_pybind11_is_named_and_offers_the_same_way_out(monkeypatch):
    setup = _load_setup()
    _without_pybind11(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        setup._require_native_build_prerequisites()

    message = str(excinfo.value)
    assert "pybind11" in message
    assert PYTORCH_ONLY_HINT in message


def test_every_missing_prerequisite_is_listed_together(monkeypatch):
    """One failure should name everything the user has to install, not just the first."""
    setup = _load_setup()
    _without_cmake(monkeypatch)
    _without_pybind11(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        setup._require_native_build_prerequisites()

    message = str(excinfo.value)
    assert "cmake" in message
    assert "pybind11" in message


def _run_command(setup, monkeypatch, tmp_path):
    from setuptools.dist import Distribution

    monkeypatch.setattr(setup, "EXTENSIONS_DIR", tmp_path / "extensions")
    command = setup.BuildExtensions(Distribution({"name": "pocketllm", "ext_modules": []}))
    command.extensions = []
    return command


def test_preflight_failure_happens_before_the_cuda_extensions_compile(monkeypatch, tmp_path):
    """The regression this guards: a long compile followed by "cmake not found"."""
    setup = _load_setup()
    _without_cmake(monkeypatch)

    started = []
    monkeypatch.setattr(setup.BuildExtension, "run", lambda self: started.append("compiled"))
    monkeypatch.setattr(
        setup.BuildExtensions, "build_native_module", lambda self: started.append("native")
    )

    command = _run_command(setup, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="cmake"):
        command.run()

    assert started == [], f"compilation started despite a missing prerequisite: {started}"


def test_pytorch_only_install_is_not_blocked_by_the_preflight(monkeypatch, tmp_path):
    """POCKETLLM_BUILD_CPP=0 must keep working on a machine with no cmake."""
    setup = _load_setup()
    _without_cmake(monkeypatch)
    monkeypatch.setattr(setup, "_build_native_requested", lambda: False)

    steps = []
    monkeypatch.setattr(setup.BuildExtension, "run", lambda self: steps.append("compiled"))
    monkeypatch.setattr(
        setup.BuildExtensions, "build_native_module", lambda self: steps.append("native")
    )

    command = _run_command(setup, monkeypatch, tmp_path)
    command.run()

    assert steps == ["compiled"], "POCKETLLM_BUILD_CPP=0 still attempted a native build"
