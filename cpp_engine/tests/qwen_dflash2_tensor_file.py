"""Versioned tensor container shared by DFlash2 parity scripts.

The format is intentionally flat and self-describing: each record has a stable
name, dtype, shape, and little-endian payload. Large hidden/logit tensors stay in
binary files instead of stdout.
"""

from __future__ import annotations

import dataclasses
import math
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x32464451  # "QDF2"
VERSION = 1
DTYPE_TO_CODE = {np.dtype("<f2"): 1, np.dtype("<f4"): 2, np.dtype("<i4"): 3}
CODE_TO_DTYPE = {value: key for key, value in DTYPE_TO_CODE.items()}
FILE_HEADER = struct.Struct("<IIiiI")
RECORD_HEADER = struct.Struct("<IIIIQ")


@dataclasses.dataclass
class TensorFile:
    position_offset: int
    anchor_token: int
    tensors: dict[str, np.ndarray]


def _canonical(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind == "f" and array.dtype.itemsize == 2:
        dtype = np.dtype("<f2")
    elif array.dtype.kind == "f" and array.dtype.itemsize == 4:
        dtype = np.dtype("<f4")
    elif array.dtype.kind in "iu" and array.dtype.itemsize == 4:
        dtype = np.dtype("<i4")
    else:
        raise ValueError(f"unsupported DFlash2 tensor dtype: {array.dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


def write_tensor_file(path: str | Path, file: TensorFile) -> None:
    records = [(name, _canonical(array)) for name, array in file.tensors.items()]
    with open(path, "wb") as output:
        output.write(
            FILE_HEADER.pack(
                MAGIC,
                VERSION,
                int(file.position_offset),
                int(file.anchor_token),
                len(records),
            )
        )
        for name, array in records:
            encoded = name.encode("utf-8")
            if not encoded or len(encoded) > 4096 or not array.shape or array.ndim > 8:
                raise ValueError(f"invalid DFlash2 tensor metadata: {name}")
            output.write(
                RECORD_HEADER.pack(
                    len(encoded),
                    DTYPE_TO_CODE[array.dtype],
                    array.ndim,
                    0,
                    array.nbytes,
                )
            )
            output.write(encoded)
            output.write(struct.pack(f"<{array.ndim}Q", *array.shape))
            output.write(array.tobytes(order="C"))


def read_tensor_file(path: str | Path) -> TensorFile:
    with open(path, "rb") as input_file:
        raw = input_file.read(FILE_HEADER.size)
        if len(raw) != FILE_HEADER.size:
            raise ValueError(f"truncated DFlash2 tensor header: {path}")
        magic, version, position_offset, anchor_token, count = FILE_HEADER.unpack(raw)
        if magic != MAGIC or version != VERSION or count > 512:
            raise ValueError(f"unsupported DFlash2 tensor file: {path}")
        tensors: dict[str, np.ndarray] = {}
        for _ in range(count):
            raw = input_file.read(RECORD_HEADER.size)
            if len(raw) != RECORD_HEADER.size:
                raise ValueError(f"truncated DFlash2 tensor record: {path}")
            name_size, dtype_code, rank, _, byte_size = RECORD_HEADER.unpack(raw)
            if not 0 < name_size <= 4096 or not 0 < rank <= 8 or byte_size > 1 << 34:
                raise ValueError(f"invalid DFlash2 tensor record: {path}")
            dtype = CODE_TO_DTYPE.get(dtype_code)
            if dtype is None:
                raise ValueError(f"unknown DFlash2 dtype code {dtype_code}")
            name = input_file.read(name_size).decode("utf-8")
            shape_raw = input_file.read(8 * rank)
            if len(shape_raw) != 8 * rank:
                raise ValueError(f"truncated DFlash2 tensor shape: {name}")
            shape = struct.unpack(f"<{rank}Q", shape_raw)
            expected = math.prod(shape) * dtype.itemsize
            if byte_size != expected:
                raise ValueError(f"DFlash2 tensor extent mismatch: {name}")
            payload = input_file.read(byte_size)
            if len(payload) != byte_size:
                raise ValueError(f"truncated DFlash2 tensor payload: {name}")
            if name in tensors:
                raise ValueError(f"duplicate DFlash2 tensor: {name}")
            tensors[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        if input_file.read(1):
            raise ValueError(f"trailing bytes in DFlash2 tensor file: {path}")
    return TensorFile(position_offset, anchor_token, tensors)
