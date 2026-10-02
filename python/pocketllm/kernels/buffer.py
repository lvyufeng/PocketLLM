"""Buffers: the ABI's unit of device memory.

A :class:`Buffer` is a handle to some bytes on some device.  The ABI never
allocates one itself -- allocation is a backend's job (``BackendSession.alloc``)
-- because "how do I get memory on a Hexagon DSP" is precisely the question this
layer exists to keep out of the shared vocabulary.

The one operation worth calling out is :meth:`Buffer.host_view`.  A CPU buffer
and a unified-memory NPU buffer can hand back a ``memoryview`` over their own
bytes; a discrete card cannot, and returns ``None``.  Callers must handle the
``None`` rather than assuming a copy is free -- the honest answer for a discrete
card is that reading its memory costs a transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .device import Device

__all__ = ["Buffer", "DeviceBuffer", "HostBuffer"]


@runtime_checkable
class Buffer(Protocol):
    """A handle to ``nbytes`` of device memory.  Never owns a lifetime by itself."""

    @property
    def nbytes(self) -> int:
        ...

    @property
    def device(self) -> Device:
        ...

    @property
    def alignment(self) -> int:
        """Byte alignment of the allocation, for a kernel that requires one."""

    def address(self) -> int:
        """An opaque device address for a native call.  ``0`` means 'no address'.

        A backend whose kernels take pointers returns a real address here; one
        that passes a Python object down (numpy, a ctypes wrapper) returns 0 and
        keeps the object on the buffer instead.
        """
        ...

    def host_view(self) -> memoryview | None:
        """The bytes as host memory, or ``None`` when the device cannot map them."""
        ...

    def subview(self, offset: int, nbytes: int) -> "Buffer":
        """A view over ``[offset, offset + nbytes)`` of this buffer."""
        ...


@dataclass(slots=True)
class DeviceBuffer:
    """A concrete :class:`Buffer` a backend allocates.

    ``owner`` holds whatever native handle must outlive the buffer -- a numpy
    array, a torch tensor, a ctypes allocation, an NPU heap reference -- so the
    buffer keeps its storage alive without the ABI knowing what the storage is.
    """

    device: Device
    nbytes: int
    alignment: int = 64
    _address: int = 0
    owner: Any = None
    _view: memoryview | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.nbytes < 0:
            raise ValueError(f"buffer size must be non-negative, got {self.nbytes}")

    def address(self) -> int:
        return int(self._address)

    def host_view(self) -> memoryview | None:
        return self._view

    def subview(self, offset: int, nbytes: int) -> "DeviceBuffer":
        offset = int(offset)
        nbytes = int(nbytes)
        if offset < 0 or nbytes < 0 or offset + nbytes > self.nbytes:
            raise ValueError(f"subview [{offset}, {offset + nbytes}) is outside a {self.nbytes}-byte buffer")
        view = None
        if self._view is not None:
            view = self._view[offset:offset + nbytes]
        address = self._address + offset if self._address else 0
        # The subview shares the owner, so the parent buffer must outlive it;
        # that is the caller's contract, and holding `owner` here makes it hold.
        return DeviceBuffer(
            device=self.device,
            nbytes=nbytes,
            alignment=self.alignment,
            _address=address,
            owner=self.owner,
            _view=view,
        )


def HostBuffer(array: Any, *, device: Device | None = None) -> DeviceBuffer:
    """Wrap a host-side object (numpy array or anything buffer-protocol) as a buffer.

    This is the bridge the loader uses to hand host bytes to a session, so it is
    here rather than in the loader: it is a statement about the ABI's buffer
    contract, not about GGUF.
    """
    view = memoryview(array)
    return DeviceBuffer(
        device=device or Device("cpu"),
        nbytes=view.nbytes,
        alignment=1,
        owner=array,
        _view=view,
    )