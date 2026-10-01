"""The one boundary between loader arrays and device buffers.

Every decoder in this package produces a numpy array on the host.  Turning one
into a device-resident :class:`~pocketllm.kernels.tensor.Tensor` is the only
place the loader touches a backend, and it touches it through the session
protocol -- ``alloc``, ``to_device`` -- so the loader names no device runtime.

Keeping this in one function matters for a second reason: it is the only spot
where a numpy dtype has to be matched to an ABI :class:`~pocketllm.kernels.dtypes.DType`,
and the mapping is small enough to state once.  A dtype the ABI does not have
(a bfloat16 carrier, for instance) is refused by name rather than silently
widened, because the caller that asked for it can choose a policy and this
function cannot.
"""

from __future__ import annotations

import numpy as np

from pocketllm.kernels.backend import BackendSession
from pocketllm.kernels.dtypes import DType, QuantFormat
from pocketllm.kernels.tensor import Tensor, TensorDesc

__all__ = ["dtype_for", "upload", "quantized_desc", "NumpyToDType"]

#: numpy kind+itemsize -> the ABI element type.  byte order is normalised to
#: little-endian here; a big-endian array is byte-swapped rather than refused,
#: because GGUF itself is defined little-endian and a big-endian host is a
#: property of the machine, not of the data.
NumpyToDType: dict[tuple[str, int], DType] = {
    ("f", 4): DType.F32,
    ("f", 2): DType.F16,
    ("i", 1): DType.I8,
    ("i", 2): DType.I16,
    ("i", 4): DType.I32,
    ("i", 8): DType.I64,
    ("u", 1): DType.U8,
    ("b", 1): DType.BOOL,
}


def dtype_for(array: np.ndarray) -> DType:
    """The ABI element type of a numpy array, or a refusal naming the dtype."""
    key = (array.dtype.kind, array.dtype.itemsize)
    try:
        return NumpyToDType[key]
    except KeyError as exc:
        raise TypeError(
            f"numpy dtype {array.dtype!r} has no kernel-ABI element type; "
            f"the ABI knows {sorted(name.value for name in DType)}"
        ) from exc


def upload(session: BackendSession, array: np.ndarray, desc: TensorDesc | None = None) -> Tensor:
    """Copy a host array into device memory and return a tensor over it.

    ``desc`` overrides the shape/type that would be inferred from ``array`` --
    a quantized operand's descriptor carries its :class:`QuantFormat`, which the
    numpy dtype cannot express.  The array is made C-contiguous first, so a
    strided view is copied once here rather than failing at the allocation; a
    0-d array is already contiguous and is left alone, because
    ``ascontiguousarray`` would promote it to shape ``(1,)`` and contradict the
    descriptor.
    """
    if desc is None:
        desc = TensorDesc(shape=tuple(int(d) for d in array.shape), dtype=dtype_for(array))
    elif desc.dtype is None and desc.quant is None:
        raise ValueError("a tensor descriptor must carry either a dtype or a quant format")
    contiguous = array if array.flags["C_CONTIGUOUS"] else np.ascontiguousarray(array)
    return session.to_device(memoryview(contiguous).cast("B"), desc)


def quantized_desc(shape: tuple[int, ...], fmt: QuantFormat) -> TensorDesc:
    """A descriptor for raw quantized blocks: a byte payload of known geometry."""
    return TensorDesc(shape=shape, dtype=None, quant=fmt)