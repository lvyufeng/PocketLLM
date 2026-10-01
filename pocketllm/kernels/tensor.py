"""Tensors: a descriptor plus the bytes it reads.

The ABI's tensor is deliberately poorer than a framework tensor.  It has no
dtype promotion, no device transfer, no autograd, and no ownership of its
storage: it is a *shape and a type* bound to a :class:`~pocketllm.kernels.buffer.Buffer`
that someone else owns.  Everything a kernel needs to be called, and nothing it
does not.

A tensor's element type is exactly one of a :class:`DType` or a
:class:`QuantFormat`.  A quantized tensor keeps its packed blocks; ``nbytes``
walks the block geometry rather than assuming one element is one element wide,
which is what makes a 1.75-bit row addressable at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

from .buffer import Buffer
from .dtypes import DType, QuantFormat
from .errors import ShapeError

__all__ = ["TensorDesc", "Tensor", "elem_count"]


def elem_count(shape: tuple[int, ...]) -> int:
    """Number of unpacked elements a shape describes (empty shape == 1, a scalar)."""
    return reduce(mul, (int(dim) for dim in shape), 1)


@dataclass(frozen=True, slots=True)
class TensorDesc:
    """The shape and element type of a tensor.  Exactly one of dtype/quant is set."""

    shape: tuple[int, ...]
    dtype: DType | None = None
    quant: QuantFormat | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if (self.dtype is None) == (self.quant is None):
            raise ShapeError("a tensor desc carries exactly one of dtype or quant")
        if any(dim < 0 for dim in self.shape):
            raise ShapeError(f"negative dimension in shape {self.shape}")

    @property
    def is_quantized(self) -> bool:
        return self.quant is not None

    @property
    def elem_count(self) -> int:
        return elem_count(self.shape)

    @property
    def packed_row_elems(self) -> int:
        """Elements along the packed axis (the last one) for a quantized desc."""
        if self.quant is None:
            raise ShapeError("packed_row_elems is only defined for a quantized desc")
        if not self.shape:
            raise ShapeError("a quantized desc needs at least one dimension")
        return self.shape[-1]

    @property
    def rows(self) -> int:
        """The number of packed rows: every dimension but the last, multiplied."""
        if not self.shape:
            return 1
        return elem_count(self.shape[:-1])

    @property
    def nbytes(self) -> int:
        if self.quant is not None:
            blocks_per_row = math.ceil(self.packed_row_elems / self.quant.block_elems)
            return self.rows * blocks_per_row * self.quant.block_bytes
        assert self.dtype is not None
        return self.elem_count * self.dtype.itemsize

    def with_shape(self, shape: tuple[int, ...]) -> "TensorDesc":
        return TensorDesc(shape, dtype=self.dtype, quant=self.quant)


@dataclass(slots=True)
class Tensor:
    """A descriptor bound to the buffer it reads.  Never owns the buffer."""

    desc: TensorDesc
    buffer: Buffer

    def __post_init__(self) -> None:
        if self.buffer.nbytes < self.desc.nbytes:
            raise ShapeError(
                f"buffer holds {self.buffer.nbytes} bytes but {self.desc.shape} "
                f"({self.desc.dtype or self.desc.quant}) needs {self.desc.nbytes}"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.desc.shape

    @property
    def dtype(self) -> DType | None:
        return self.desc.dtype

    @property
    def quant(self) -> QuantFormat | None:
        return self.desc.quant

    @property
    def device(self):
        return self.buffer.device

    @property
    def nbytes(self) -> int:
        return self.desc.nbytes