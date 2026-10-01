"""The bridge between ABI dtypes and numpy, including the one numpy lacks.

``bfloat16`` is not a numpy dtype.  The reference backend carries it as
``uint16`` in memory and widens to ``float32`` to compute, which is the same
choice the loader makes and for the same reason: bfloat16 is the *top half* of a
float32, so the widening is a bit shift rather than a rounding, and it is exact.
Narrowing back after a write drops the low mantissa bits, which is what storing
an f32 result into an f16 or bf16 tensor means everywhere else too.

The mapping is a module-level table rather than a function with a branch, so the
unsupported-dtype path is a lookup miss and not a fall-through.
"""

from __future__ import annotations

import numpy as np

from pocketllm.kernels.dtypes import DType

__all__ = ["NP_OF", "read", "write", "is_bf16"]

#: ABI dtype -> the numpy dtype that stores it.  ``BF16`` is absent on purpose:
#: numpy has no such dtype, and pretending ``float16`` is close enough would be
#: wrong by more than one ulp.
NP_OF: dict[DType, np.dtype] = {
    DType.F32: np.dtype("<f4"),
    DType.F16: np.dtype("<f2"),
    DType.I8: np.dtype("<i1"),
    DType.I16: np.dtype("<i2"),
    DType.I32: np.dtype("<i4"),
    DType.I64: np.dtype("<i8"),
    DType.U8: np.dtype("<u1"),
    DType.BOOL: np.dtype("?"),
}

#: The storage width of every dtype, including the ones with no numpy equivalent.
ITEMSIZE: dict[DType, int] = {**{d: np.dtype(t).itemsize for d, t in NP_OF.items()}, DType.BF16: 2}


def is_bf16(dtype: DType | None) -> bool:
    return dtype is DType.BF16


def read(raw: np.ndarray, dtype: DType, shape: tuple[int, ...]) -> np.ndarray:
    """Interpret raw bytes as ``dtype`` with ``shape``."""
    if dtype is DType.BF16:
        # Widen to float32 by placing the 16 bits in the high half of a uint32.
        # This is exact: bfloat16 *is* the top half of a float32.
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    try:
        return raw.view(NP_OF[dtype]).reshape(shape)
    except KeyError as exc:
        raise NotImplementedError(f"the reference backend has no numpy type for {dtype}") from exc


def write(target: np.ndarray, values: np.ndarray, dtype: DType) -> None:
    """Store ``values`` into the raw bytes at ``target``, casting to ``dtype``.

    ``target`` is a flat ``uint8`` view of the storage; the values' shape is
    restored on it so a broadcast is a shape error here rather than a write into
    the wrong elements.
    """
    flat = target.view(np.uint8).reshape(-1) if target.dtype != np.uint8 else target.reshape(-1)
    shape = values.shape
    if values.size != flat.size // ITEMSIZE[dtype]:
        raise ValueError(
            f"cannot store {values.size} {dtype.value} values into {flat.size} bytes"
        )
    if dtype is DType.BF16:
        # Round-to-nearest on the way down, by adding the bits below the kept
        # half before truncating -- the same rounding the hardware uses, so a
        # round-trip through the reference matches a device's.
        wide = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        rounded = (wide + np.uint32(0x7FFF) + ((wide >> 16) & 1)) >> 16
        flat.view("<u2").reshape(shape)[...] = rounded.astype(np.uint16)
        return
    try:
        cast = NP_OF[dtype]
    except KeyError as exc:
        raise NotImplementedError(f"the reference backend has no numpy type for {dtype}") from exc
    flat.view(cast).reshape(shape)[...] = values.astype(cast, copy=False)