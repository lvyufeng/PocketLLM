"""The kernel ABI's exception vocabulary.

Every failure this layer can raise is named, and the names are the point: the
ABI's contract is that an unsupported request is *refused by name* rather than
silently approximated.  The loader made the same bargain for ternary tensors
(see ``pocketllm.loader.gguf.tensor_reader``), and the reason is the same -- a
silent fallback (an F16 upcast there, a slow-but-working op here) produces a
wrong answer that looks right.
"""

from __future__ import annotations

__all__ = [
    "KernelError",
    "OpNotDeclaredError",
    "NoBackendError",
    "ShapeError",
    "BackendNotImplementedError",
]


class KernelError(Exception):
    """Base class for every error the kernel ABI raises."""


class OpNotDeclaredError(KernelError):
    """An op name is not in the registry.

    Raised before dispatch, because a name no schema declares is a programming
    error rather than a coverage gap: the op vocabulary is fixed by
    ``pocketllm.kernels.ops``.
    """


class NoBackendError(KernelError):
    """No registered backend can run this op over these dtypes on this device.

    The message carries the resolution trace (which backends were considered and
    why each was rejected), so the operator sees *why* rather than only that the
    call failed.  See :meth:`pocketllm.kernels.dispatch.Dispatcher.explain`.
    """


class ShapeError(KernelError):
    """An argument or output does not match the op's declared shape rule."""


class BackendNotImplementedError(KernelError):
    """A backend is declared and discoverable but its session is a stub.

    Every device backend ships as a stub in the skeleton, and each stub raises
    this with the native dependency it is waiting for, so an unimplemented path
    is visibly pending rather than apparently done.
    """