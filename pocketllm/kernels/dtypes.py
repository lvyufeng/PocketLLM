"""Element and block types for the kernel ABI.

Two different things live here and they are deliberately separate:

* :class:`DType` is the type of an *unpacked* element -- what a tensor's values
  are once a kernel has them.
* :class:`QuantFormat` is a *packed block* format -- GGUF's storage types, where
  a run of weights shares a scale and the values are recovered by a decoder.

A tensor carries exactly one of the two (see
:class:`pocketllm.kernels.tensor.TensorDesc`), which is what lets a quantized
weight stay packed all the way to a kernel instead of being expanded to a dense
copy on the way in.

The format table is stated here, in the ABI, rather than reusing the loader's
``quant_types`` table.  They answer different questions: the loader's table is
"which GGML *file* id is this tensor", and this one is "what does a kernel need
to know to read the blocks".  ``file_type_id`` is the file id; ``runtime_id`` is
the compact id a raw-block kernel switches on, and it is *not* the same number --
``iq2_xxs`` is file id 16 and runtime id 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "DType",
    "QuantFormat",
    "QUANT_FORMATS",
    "quant_format",
    "quant_format_by_file_id",
]


class DType(Enum):
    """The type of one unpacked element."""

    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    BOOL = "bool"

    @property
    def itemsize(self) -> int:
        return _DTYPE_ITEMSIZE[self]

    @property
    def is_float(self) -> bool:
        return self in (DType.F32, DType.F16, DType.BF16)


_DTYPE_ITEMSIZE = {
    DType.F32: 4,
    DType.F16: 2,
    DType.BF16: 2,
    DType.I8: 1,
    DType.I16: 2,
    DType.I32: 4,
    DType.I64: 8,
    DType.U8: 1,
    DType.BOOL: 1,
}


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """A packed block format: geometry a kernel needs, and the ids that name it.

    ``runtime_id`` is ``None`` for a format the loader can decode but no
    raw-block kernel consumes yet.  That is an honest "no consumer", not a
    placeholder -- the ternary packs are in exactly this state, and the same
    category ``iq4_nl`` was in before it graduated.
    """

    name: str
    file_type_id: int
    block_elems: int
    block_bytes: int
    runtime_id: int | None = None

    def __post_init__(self) -> None:
        if self.block_elems <= 0 or self.block_bytes <= 0:
            raise ValueError(f"{self.name}: block geometry must be positive")

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.block_bytes / self.block_elems


#: The raw-block formats a kernel can switch on, keyed by ABI name.  ``runtime_id``
#: values match the loader's historical dispatch table so a CUDA kernel that
#: already switches on them needs no renumbering.
QUANT_FORMATS: dict[str, QuantFormat] = {
    fmt.name: fmt
    for fmt in (
        QuantFormat("q2_k", 10, 256, 84, runtime_id=1),
        QuantFormat("q3_k", 11, 256, 110),
        QuantFormat("q4_k", 12, 256, 144, runtime_id=3),
        QuantFormat("q5_k", 13, 256, 176, runtime_id=4),
        QuantFormat("q6_k", 14, 256, 210, runtime_id=8),
        QuantFormat("q8_0", 8, 32, 34, runtime_id=21),
        QuantFormat("iq2_xxs", 16, 256, 66, runtime_id=0),
        QuantFormat("iq2_xs", 17, 256, 74, runtime_id=5),
        QuantFormat("iq3_xxs", 18, 256, 98, runtime_id=6),
        QuantFormat("iq1_s", 19, 256, 50),
        QuantFormat("iq4_nl", 20, 32, 18, runtime_id=20),
        QuantFormat("iq4_xs", 23, 256, 136, runtime_id=7),
        QuantFormat("iq1_m", 29, 256, 56, runtime_id=2),
        # Fork-private ternary packs from PrismML-Eng/llama.cpp's `prism` branch.
        # Addressed by the loader, consumed by nothing yet: runtime_id is None.
        QuantFormat("pq2_0", 142, 128, 34),
        QuantFormat("ptq1_0", 143, 128, 28),
    )
}

_BY_FILE_ID = {fmt.file_type_id: fmt for fmt in QUANT_FORMATS.values()}


def quant_format(name: str) -> QuantFormat:
    """Look a format up by its ABI name, refusing an unknown one by name."""
    try:
        return QUANT_FORMATS[name]
    except KeyError as exc:
        raise KeyError(f"unknown quant format {name!r}; known: {sorted(QUANT_FORMATS)}") from exc


def quant_format_by_file_id(file_type_id: int) -> QuantFormat | None:
    """The format whose GGML file id this is, or ``None`` if the ABI has none."""
    return _BY_FILE_ID.get(int(file_type_id))