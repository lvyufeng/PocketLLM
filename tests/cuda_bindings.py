"""Getting the built CUDA extension, with a skip instead of a failure when it cannot be had.

`pytest.importorskip` and `load_cuda_kernel() -> None` guard the *extension loading*. They do not
guard an individual op: a test that loads the extension successfully and then calls a binding that
does not exist gets an `AttributeError`, which pytest reports as a **failure**, not a skip. Four
tests were in exactly that state -- the GQA decode kernels whose sources are on disk and not
compiled, and whose bindings therefore resolve nowhere -- and each of them read as a broken test
rather than as an unbuilt feature.

`extension()` closes that hole: ask for the bindings the test needs and get either the module or a
skip that names the missing ones. A skip here is a statement about the build, not about the test's
correctness, which is the same claim `pytest.importorskip` makes one level up.

`docs/guides/cuda_extension_builds.md` is where the missing-source situation is recorded; pass
`why` when the skip needs to point at it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The trap this module exists for. Imported lazily so that a machine with no torch can still
#: collect the suite.
_SKIP_PREFIX = "CUDA extension not usable"


def extension(*, requires: Sequence[str] = (), why: str = "") -> Any:
    """Return the built CUDA extension, or skip.

    `requires` is the list of bindings the caller is about to call. Naming them here is what turns
    "this tree does not build that kernel" into a skip rather than an `AttributeError` raised
    halfway through a test body, where the traceback suggests the test is wrong.
    """
    import pytest

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the kernels
        pytest.skip(f"{_SKIP_PREFIX}: torch is not importable")

    if not torch.cuda.is_available():
        pytest.skip(f"{_SKIP_PREFIX}: no visible CUDA device")

    from src.kernels.cuda_loader import load_cuda_kernel

    module = load_cuda_kernel()
    if module is None:
        pytest.skip(f"{_SKIP_PREFIX}: no built cuda_kernel extension was found")

    missing = [name for name in requires if not hasattr(module, name)]
    if missing:
        detail = f"the build does not export {', '.join(missing)}"
        if why:
            detail = f"{detail} ({why})"
        pytest.skip(f"{_SKIP_PREFIX}: {detail}")

    return module
