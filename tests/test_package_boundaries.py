"""The dependency directions the tree's layering rests on.

Two rules do the work here, and both are easy to break by accident:

* **The ABI imports nothing.**  ``pocketllm.kernels`` is descriptors and
  declarations; if it ever imports numpy or torch, every backend has to have that
  runtime to read a shape.
* **A backend does not import another backend, and nothing below the engine
  imports a backend package.**  A shared kernel would otherwise arrive by one
  backend importing another's module, and the two would stop being separable --
  which is what makes "one install, several devices" work.

The check is a static one: it parses each module's imports rather than importing
it, so a violation is reported by file and line, and a module that cannot be
imported on this host is still checked.  That matters more here than usual --
the whole point is to police backends whose runtimes are absent.

Allowed direction, from the design:

    kernels/       -> stdlib only
    quant/         -> stdlib + numpy (a leaf; loader and reference both use it)
    backends/      -> kernels, quant, numpy (plus its own runtime, lazily)
    loader/        -> kernels, quant, numpy
    engine/        -> kernels, backends, loader, ...
    cli, __init__  -> anything
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: The Python tree.  It sits under ``python/`` so the C++ engine's ``src/`` can be
#: the repository's other top-level tree.  It is also the base a path is made
#: relative to, which is what turns ``python/pocketllm/quant/formats.py`` back into
#: the dotted name ``pocketllm.quant.formats``.
PYTHON = ROOT / "python"
PACKAGE = PYTHON / "pocketllm"

#: package prefix -> the ``pocketllm`` prefixes it may import.  ``None`` means
#: "anything", used for the façade and the CLI, which are the assembly point.
_ALLOWED: dict[str, set[str] | None] = {
    "pocketllm.kernels": set(),  # stdlib only
    "pocketllm.quant": {"pocketllm.quant"},
    "pocketllm.loader": {"pocketllm.kernels", "pocketllm.quant", "pocketllm.loader"},
    "pocketllm.backends": {"pocketllm.kernels", "pocketllm.quant", "pocketllm.backends"},
    "pocketllm.engine": {
        "pocketllm.kernels",
        "pocketllm.quant",
        "pocketllm.backends",
        "pocketllm.loader",
        "pocketllm.architectures",
        "pocketllm.engine",
    },
    # The model IR describes a graph and names weights; it runs nothing and
    # allocates nothing, so a backend is not its business.
    "pocketllm.architectures": {"pocketllm.kernels", "pocketllm.architectures"},
    # The public API types: intent and results, no device.  It reaches the backend
    # registry so ``--device``'s choices come from the declarations rather than a
    # second list, which is the one place it knows backends exist.
    "pocketllm.api": {"pocketllm.api", "pocketllm.backends"},
    # The HTTP and request layer speaks text, token ids and the api types; its one
    # exception is `templating`, which reads a checkpoint's declared architecture
    # out of its GGUF metadata to pick the right prompt format.  That reach into
    # the loader is deliberate and is the loader only -- numpy, no device runtime --
    # so the layer still imports no accelerator and is still testable without one.
    "pocketllm.protocol": {"pocketllm.api", "pocketllm.loader", "pocketllm.protocol"},
    "pocketllm.server": {"pocketllm.api", "pocketllm.choices", "pocketllm.protocol", "pocketllm.server"},
    # The one module under `server/` that drives a device, and the only reason it is not in
    # `backends/`: `backends/` implements the *kernel ABI* and may import nothing above it, while
    # this implements the *serving* contract and needs the ctypes bridge (`native`) to run a
    # generation and the loader (via `protocol.templating`) to read the checkpoint's chat template
    # out of its GGUF metadata.  That is still no accelerator: `native` is a `dlopen` and the loader
    # is numpy-only, so the HTTP layer stays testable on a host with no card.  Scoped to this module
    # rather than widening `pocketllm.server`, which would let any future module in the package reach
    # a device without saying so here.
    "pocketllm.server.native_backend": {
        "pocketllm.api",
        "pocketllm.choices",
        "pocketllm.loader",
        "pocketllm.native",
        "pocketllm.protocol",
        "pocketllm.server",
    },
    "pocketllm.choices": {"pocketllm.api", "pocketllm.protocol"},
    # Skeleton: a vocabulary and a merge table.  It reads the GGUF metadata the
    # loader exposes, and nothing else.
    "pocketllm.tokenizer": {"pocketllm.api", "pocketllm.loader", "pocketllm.tokenizer"},
    # The ctypes bridge to the native engine.  It imports nothing from
    # `pocketllm` at all -- it is the seam, and everything above it (the CLI,
    # the server) calls *it* rather than `ctypes`, so there is exactly one place
    # that knows a shared library exists.
    "pocketllm.native": {"pocketllm.native"},
    # The ctypes bridge to the S600 `libxlm.so` delegate.  The second seam, and
    # like the first it imports nothing from `pocketllm`: it is the only place
    # that knows the delegate exists, and a serving adapter above it calls it
    # rather than `ctypes` directly.  Stdlib only -- `ctypes`, `glob`, `os` --
    # so a host without the S600 SDK imports it and gets an `XlmUnavailable`
    # from `load()` rather than an import error.
    "pocketllm.xlm": {"pocketllm.xlm"},
}

#: Runtimes that a base install does not have, so only a backend may import one.
_OPTIONAL_RUNTIMES = ("torch", "relic_core", "qai_appbuilder")


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _dotted(path: pathlib.Path) -> str:
    relative = path.relative_to(PYTHON).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every module name ``path`` imports, with the line it does so on."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import stays inside this package.
                continue
            if node.module:
                found.append((node.module, node.lineno))
    return found


def _governing(dotted: str) -> tuple[str, set[str] | None] | None:
    """The rule that applies to a module, most specific first."""
    for prefix in sorted(_ALLOWED, key=len, reverse=True):
        if dotted == prefix or dotted.startswith(prefix + "."):
            return prefix, _ALLOWED[prefix]
    return None


@pytest.mark.parametrize("path", _modules(), ids=lambda p: _dotted(p))
def test_imports_follow_the_layering(path: pathlib.Path) -> None:
    dotted = _dotted(path)
    rule = _governing(dotted)
    if rule is None:
        return  # the façade, the CLI: the assembly point, allowed anything
    prefix, allowed = rule
    if allowed is None:
        return

    violations = []
    for name, lineno in _imports(path):
        if not name.startswith("pocketllm"):
            continue
        # Importing yourself or your own subtree is always fine.
        if name == prefix or name.startswith(prefix + "."):
            continue
        if not any(name == a or name.startswith(a + ".") for a in allowed):
            violations.append(f"{path.relative_to(ROOT)}:{lineno} imports {name}")
    assert not violations, (
        f"{prefix} may import only {sorted(allowed)}; found:\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: _dotted(p))
def test_only_the_kernel_abi_is_stdlib_only(path: pathlib.Path) -> None:
    """``pocketllm.kernels`` imports no third-party module, at any level.

    A schema is data and a device is a name; neither needs numpy.  This is
    checked by name because the ABI's own import of numpy would only fail on the
    installs that do not have it, which are the ones this rule protects.
    """
    dotted = _dotted(path)
    if not dotted.startswith("pocketllm.kernels"):
        return
    third_party = {
        "numpy",
        "torch",
        "relic_core",
    }
    violations = [
        f"{path.relative_to(ROOT)}:{lineno} imports {name.split('.')[0]}"
        for name, lineno in _imports(path)
        if name.split(".")[0] in third_party
    ]
    assert not violations, "the kernel ABI must be stdlib-only; found:\n  " + "\n  ".join(violations)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: _dotted(p))
def test_optional_runtimes_are_imported_only_by_backends(path: pathlib.Path) -> None:
    """torch, relic_core and the QNN bindings are a backend's business, not the core's.

    A core module that imported one would make the whole wheel uninstallable on a
    phone, which is the opposite of what this tree is for.  A *backend* module is
    allowed to, and is expected to do it lazily -- ``available`` must not import
    it, and the ABI test in ``tests/abi/test_no_runtime_import.py`` checks that
    behaviourally.
    """
    dotted = _dotted(path)
    if dotted.startswith("pocketllm.backends"):
        return
    violations = [
        f"{path.relative_to(ROOT)}:{lineno} imports {name.split('.')[0]}"
        for name, lineno in _imports(path)
        if name.split(".")[0] in _OPTIONAL_RUNTIMES
    ]
    assert not violations, (
        f"{dotted} imports an optional runtime; only a backend may:\n  " + "\n  ".join(violations)
    )


def test_importing_the_package_loads_no_runtime() -> None:
    """``import pocketllm`` must put neither torch nor numpy into the process.

    This is the behavioural half of the rule above, and it is the one that
    enforces the decision the ABI test can only check structurally: a wheel that
    imports numpy at ``import pocketllm`` is a wheel that cannot be trimmed for a
    device with no numpy, and one that imports torch cannot install on a phone at
    all.  Run in a subprocess so the assertion is about a *fresh* interpreter
    rather than whatever the test session already imported.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import pocketllm; "
        "leaked = [m for m in ('numpy', 'torch', 'relic_core') if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing pocketllm pulled a runtime into sys.modules:\n" + result.stderr
    )


def test_the_cli_imports_with_no_backend_runtime() -> None:
    """``pocketllm devices`` is the command run *because* something is broken.

    It therefore imports no backend and no runtime: every check behind it is a
    filesystem probe.  Checked in a subprocess for the same reason as above.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import pocketllm.cli; "
        "leaked = [m for m in ('numpy', 'torch', 'relic_core') if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_package_has_no_second_root() -> None:
    """``pocketllm`` is the only top-level package ``python/`` holds.

    A second root that another wheel also claims has no defined owner, and
    install order silently decides which tree wins.  The check is scoped to
    ``python/`` rather than the repository root because ``src/`` -- the C++
    engine -- is deliberately a second top-level tree, just not a Python one.
    """
    roots = [p.name for p in PYTHON.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    assert roots == ["pocketllm"], f"unexpected top-level packages: {sorted(roots)}"


def test_the_package_does_not_vendor_a_compiled_artifact() -> None:
    """The *Python package* is pure Python: no extension module, no shared library.

    The engine is native code, but not this package's: it is ``src/``, built
    out of tree into a ``libpocketllm.so`` the host shell loads through
    ``ctypes``.  An artifact *inside* the package would instead be something
    ``pip install`` has to produce, which is the install-time compile step this
    tree does not have.
    """
    offenders = [
        p.relative_to(ROOT)
        for pattern in ("*.so", "*.pyd", "*.cu", "*.cuh", "*.c", "*.cpp")
        for p in PACKAGE.rglob(pattern)
    ]
    assert not offenders, (
        "the package vendors a native artifact; native sources belong in the "
        f"tree-level src/ directory: {sorted(str(o) for o in offenders)}"
    )


def test_the_vendored_header_is_the_only_non_python_payload() -> None:
    """What ships beside the .py files is the GGML header, and it is deliberate."""
    payload = {
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*")
        if p.is_file() and p.name != "py.typed" and p.suffix not in {".py", ".pyc"} and "__pycache__" not in p.parts
    }
    allowed = {
        "loader/gguf/vendor/ggml-common.h",
        "loader/gguf/vendor/README.md",
        "py.typed",
        "backends/reference/README.md",
        "backends/cpu/README.md",
        "backends/cuda/README.md",
        "backends/mps/README.md",
        "backends/qnn/README.md",
        "backends/horizon/README.md",
        "backends/ascend/README.md",
    }
    unexpected = payload - allowed
    assert not unexpected, (
        "a non-Python file appeared in the package; if it is meant to ship, add it to "
        f"MANIFEST.in and to this list: {sorted(unexpected)}"
    )