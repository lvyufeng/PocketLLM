"""Devices, named by an open string rather than a closed enum.

A device is ``<kind>:<index>`` -- ``cpu``, ``cuda:2``, ``qnn:0``.  The *kind* is
the extension point: a backend registers whatever kind name it runs on, so a
third-party engine can add ``s600`` or ``rknn`` without this module changing.
The well-known kinds below are constants for readability, not a closed set.

This is deliberately not an ``enum.Enum``.  A closed enum would mean a new
device needed a change to the ABI, which is the coupling the rebuild exists to
remove.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Device", "KNOWN_DEVICE_KINDS", "parse_device"]

#: The kinds the tree ships backends for.  A backend may still register any
#: other name; this tuple is documentation and default-ordering, not a whitelist.
KNOWN_DEVICE_KINDS: tuple[str, ...] = (
    "cpu",
    "mps",
    "cuda",
    "qnn",
    "horizon",
    "ascend",
)


@dataclass(frozen=True, slots=True, order=True)
class Device:
    """One device: a kind, and an index within it.

    ``Device("cpu")`` and ``Device("cpu", 0)`` compare equal, so callers that do
    not care about the index never have to spell one.
    """

    kind: str
    index: int = 0

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("device kind must be a non-empty string")
        if self.index < 0:
            raise ValueError(f"device index must be non-negative, got {self.index}")

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}" if self.index else self.kind

    @property
    def is_host(self) -> bool:
        """Whether this device's memory is host memory.

        The one place the ABI reasons about a device's nature: a host device can
        hand out a ``memoryview`` over its own bytes, which is what lets the
        loader upload without a copy on a phone or CPU target.
        """
        return self.kind == "cpu"

    @classmethod
    def parse(cls, text: str | int) -> "Device":
        return parse_device(text)


def parse_device(value: str | int | Device) -> Device:
    """Parse ``"cuda:2"``, ``"cpu"``, ``"qnn"``, or an index, into a :class:`Device`.

    An integer is a CPU index, because a bare number names a device on the
    default kind and this tree's default is the host.  A string with no colon is
    kind 0 of that kind.
    """
    if isinstance(value, Device):
        return value
    if isinstance(value, bool):
        raise TypeError("a bool is not a device")
    if isinstance(value, int):
        return Device("cpu", value)
    text = str(value).strip().lower()
    if not text or text == "auto":
        raise ValueError(f"{value!r} does not name a device; 'auto' is resolved by the engine")
    if ":" in text:
        kind, _, index = text.partition(":")
        try:
            return Device(kind, int(index))
        except ValueError as exc:
            raise ValueError(f"{value!r} has a non-integer device index") from exc
    if text.isdigit():
        return Device("cpu", int(text))
    return Device(text)