"""Importing the package must not drag in a device runtime.

This is the test that makes "torch-free core" a fact rather than an intention.
Each case runs in a **fresh subprocess**, because `sys.modules` is global to a
pytest session and some other test importing numpy would mask the leak.

What is being protected: the wheel installs on a phone and an edge board as
well as on a CUDA host. numpy is allowed in the base install (the loader and the
reference backend need it), but torch and relic_core are optional dependencies
of specific backends and must not appear merely from importing the core.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Runtimes that are optional: a base install must work without them.
OPTIONAL_RUNTIMES = ("torch", "relic_core")


def _run(source: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


@pytest.mark.parametrize("runtime", OPTIONAL_RUNTIMES)
def test_core_import_does_not_load_runtime(runtime: str) -> None:
    leaked = _run(
        f"""
        import sys
        import pocketllm
        leaked = [m for m in sys.modules if m == {runtime!r} or m.startswith({runtime!r} + ".")]
        print(",".join(leaked))
        """
    )
    assert leaked == "", f"import pocketllm pulled in {leaked}"


@pytest.mark.parametrize("runtime", OPTIONAL_RUNTIMES)
def test_loader_import_does_not_load_runtime(runtime: str) -> None:
    leaked = _run(
        f"""
        import sys
        import pocketllm.loader.gguf.quantized_loader
        leaked = [m for m in sys.modules if m == {runtime!r} or m.startswith({runtime!r} + ".")]
        print(",".join(leaked))
        """
    )
    assert leaked == "", f"importing the loader pulled in {leaked}"


def test_numpy_is_not_loaded_by_the_abi_alone() -> None:
    """``pocketllm.kernels`` is descriptors only; numpy belongs to the layers above."""
    leaked = _run(
        """
        import sys
        import pocketllm.kernels
        print("numpy" if "numpy" in sys.modules else "")
        """
    )
    assert leaked == "", "the kernel ABI imported numpy; it is meant to be stdlib-only"


def test_package_version_is_exposed() -> None:
    version = _run("import pocketllm; print(pocketllm.__version__)")
    assert version == "0.2.0.dev0"


#: Backends whose names must not be imported merely by listing the registry.
#: ``reference`` is deliberately absent: it is the oracle and is expected to be
#: loadable everywhere, so importing it is not a leak.
_DEVICE_BACKENDS = ("cpu", "mps", "cuda", "qnn", "horizon", "ascend")


@pytest.mark.parametrize("runtime", OPTIONAL_RUNTIMES)
def test_backend_package_import_does_not_load_runtime(runtime: str) -> None:
    """Listing the backends must not import any backend, or its runtime."""
    leaked = _run(
        f"""
        import sys
        import pocketllm.backends
        leaked = [m for m in sys.modules if m == {runtime!r} or m.startswith({runtime!r} + ".")]
        print(",".join(leaked))
        """
    )
    assert leaked == "", f"importing pocketllm.backends pulled in {leaked}"


@pytest.mark.parametrize("runtime", OPTIONAL_RUNTIMES)
def test_devices_listing_does_not_import_a_runtime(runtime: str) -> None:
    """``pocketllm devices`` reads declarations, and declarations import no runtime.

    Listing does import each backend's *declaration module* -- that is the only
    way to read an out-of-tree backend's capabilities, so it is unavoidable --
    but a declaration module is pure tables plus a probe, and the probe is
    required not to import the runtime it is checking for.  That distinction is
    the whole reason a listing works on a machine where nothing is installed.
    """
    leaked = _run(
        f"""
        import sys
        from pocketllm.backends import registry
        registry.describe()
        leaked = [m for m in sys.modules if m == {runtime!r} or m.startswith({runtime!r} + ".")]
        print(",".join(leaked))
        """
    )
    assert leaked == "", f"listing the backends pulled in {leaked}"


def test_probing_availability_imports_no_optional_runtime() -> None:
    """Asking "is this backend loadable?" is a filesystem question, not an import.

    numpy *is* expected to appear: it is a base dependency and the reference
    backend -- which is always available and always listed -- imports it to run.
    What must not appear is a runtime that a base install does not have.  A probe
    that imported torch to ask about torch would make ``pocketllm devices`` cost
    as much as a model load on the machines that have it, and would defeat the
    point of the probe on the machines that do not.
    """
    leaked = _run(
        """
        import sys
        from pocketllm.backends import registry
        for entry in registry.BACKENDS.values():
            registry.raw(entry.name).factory().available()
        leaked = [m for m in sys.modules if m.split(".")[0] in ("torch", "relic_core", "qai_appbuilder")]
        print(",".join(sorted(set(leaked))))
        """
    )
    assert leaked == "", f"probing availability imported {leaked}"


def test_listing_the_backends_needs_no_device_runtime() -> None:
    """The listing works on a host where nothing is installed -- which is the point."""
    output = _run(
        """
        from pocketllm.backends import registry
        rows = registry.describe()
        print(",".join(r["name"] for r in rows))
        """
    )
    listed = set(output.split(","))
    assert {"reference", "cpu", "cuda", "qnn", "horizon", "ascend", "mps"} <= listed


def test_backend_base_import_does_not_load_a_backend_runtime() -> None:
    """``pocketllm.backends.base`` is the declaration machinery and nothing else."""
    leaked = _run(
        """
        import sys
        import pocketllm.backends.base
        leaked = [m for m in sys.modules if m.split(".")[0] in ("torch", "numpy")]
        print(",".join(sorted(leaked)))
        """
    )
    assert leaked == "", f"the backend declaration machinery imported {leaked}"


def test_reference_backend_needs_nothing_but_numpy() -> None:
    leaked = _run(
        """
        import sys
        import pocketllm.backends.reference
        leaked = [m for m in sys.modules if m == "torch" or m.startswith("torch.")]
        print(",".join(leaked))
        """
    )
    assert leaked == "", f"the reference backend pulled in {leaked}"