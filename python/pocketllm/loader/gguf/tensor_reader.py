from __future__ import annotations

import math
import mmap
import os
import re
import time
from functools import lru_cache
from typing import Iterable

import numpy as np

from pocketllm.loader.gguf.quant_types import GGUF_TERNARY_TYPE_NAMES
from pocketllm.loader.gguf.reader import GGUFFile, GGUFReader, GGUFTensorInfo
from pocketllm.quant import formats


_GGUF_READER_PROFILE = os.getenv("DEEPSEEK_GGUF_READER_PROFILE", "0").lower() in {"1", "true", "yes"}
_GGUF_READER_PROFILE_LIMIT = int(os.getenv("DEEPSEEK_GGUF_READER_PROFILE_LIMIT", "64"))
_GGUF_READER_PROFILE_COUNT = 0


_DENSE_DTYPES = {
    "f32": ("<f4", 4),
    "f16": ("<f2", 2),
    "i32": ("<i4", 4),
}

# The block formats this loader can address at all.  The geometry itself is
# *not* restated here: it lives in `pocketllm.quant.formats`, which is also what
# the reference backend dequantizes through, so a format has one statement
# rather than two that can drift apart.  This name exists only to say which
# formats the readers below accept.
#
# The two ternary formats are in the set because their *geometry* is what makes
# their bytes addressable at all -- a 1.75-bit packing does not divide a row, so
# nothing downstream can infer nbytes from a shape.  Their presence is not a
# claim that a kernel consumes them: `read_tensor` refuses them by name below,
# and `read_quantized_matrix_blocks` hands back raw blocks with no decode.
_QUANT_BLOCK_META = formats.KNOWN_TYPES

#: Block formats this loader will not dequantize.  Dequantizing one of these is
#: the F16-upcast failure mode this loader is written to avoid, so it is refused
#: by name rather than left to fall off the end of a dispatch chain.  The set
#: comes from `quant_types` so that "which types are ternary" is answered in one
#: place; the geometry above is a second statement of the same fact and the two
#: are pinned against `reader.GGML_TYPES` by `tests/test_gguf_ternary_reader.py`.
#:
#: The refusal is about *this* entry point, not about the format: these two packs
#: no longer stand in the same place.  `PTQ1_0` has a GEMM that reads its blocks
#: out of `read_quantized_matrix_block_rows`, so its message says where the blocks
#: go instead; `PQ2_0` has nothing, and its message says so.  Both still refuse
#: here, because a caller that asked for a dense tensor wants one this format
#: cannot honestly give.
_TERNARY_BLOCK_META = GGUF_TERNARY_TYPE_NAMES

#: Ternary packs a kernel consumes, and the name it is reached by.  `None` means
#: the blocks are addressable and nothing reads them.
_TERNARY_CONSUMERS: dict[str, str | None] = {
    "ptq1_0": "gguf_quant_gemm_forward / gguf_quant_gemm_prefill_forward, file type id 143",
    "pq2_0": None,
}


def _quant_block_meta(type_name: str) -> tuple[int, int]:
    """``(block_elems, block_bytes)`` for a format, from the shared table."""
    fmt = formats.format_for(type_name)
    return fmt.block_elems, fmt.block_bytes


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _storage_shape(dimensions: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(tuple(int(dim) for dim in dimensions)))

def _f16_bytes_to_f32(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data).view("<f2").astype(np.float32).reshape(data.shape[:-1])


def _get_scale_min_k4(scales: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode GGML K-quant 6-bit scale/min pair for Q4_K/Q5_K.

    Mirrors llama.cpp/ggml `get_scale_min_k4()` exactly.  `scales` has
    trailing dimension 12 and returns arrays broadcast over the leading dims.
    """
    if idx < 4:
        return scales[..., idx] & 63, scales[..., idx + 4] & 63
    return (
        (scales[..., idx + 4] & 0x0F) | ((scales[..., idx - 4] >> 6) << 4),
        (scales[..., idx + 4] >> 4) | ((scales[..., idx] >> 6) << 4),
    )


@lru_cache(maxsize=4)
def get_cached_gguf_tensor_reader(path: str) -> GGUFTensorDataReader:
    return GGUFTensorDataReader(path)


class GGUFTensorDataReader:
    def __init__(self, gguf: GGUFFile | str):
        self.gguf = GGUFReader(gguf).read() if isinstance(gguf, str) else gguf
        self._fd = os.open(self.gguf.path, os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_COPY)

    def close(self) -> None:
        mapped = getattr(self, "_mmap", None)
        if mapped is not None:
            try:
                mapped.close()
            except BufferError:
                pass
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "GGUFTensorDataReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _tensor(self, name: str | GGUFTensorInfo) -> GGUFTensorInfo:
        if isinstance(name, GGUFTensorInfo):
            return name
        try:
            return self.gguf.tensors_by_name[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor not found: {name}") from exc

    def _read_at(self, offset: int, nbytes: int) -> bytes:
        data = os.pread(self._fd, int(nbytes), int(offset))
        if len(data) != int(nbytes):
            raise EOFError(f"short GGUF read at offset {offset}: got {len(data)}, expected {nbytes}")
        return data

    def read_tensor(self, name: str | GGUFTensorInfo) -> np.ndarray:
        tensor = self._tensor(name)
        if tensor.type_name in _TERNARY_BLOCK_META:
            raise NotImplementedError(self._ternary_refusal(tensor))
        if tensor.type_name in _DENSE_DTYPES or tensor.type_name == "bf16":
            return self._read_dense_tensor(tensor)
        if tensor.type_name == "q8_0":
            return self._read_q8_0_tensor(tensor)
        if tensor.type_name in _QUANT_BLOCK_META and len(tensor.dimensions) == 2:
            return self._read_quantized_matrix(tensor, tensor.absolute_offset, int(tensor.dimensions[0]), int(tensor.dimensions[1]), tensor.type_name)
        raise NotImplementedError(f"payload decode for {tensor.name} ({tensor.type_name}) is not supported by read_tensor")

    @staticmethod
    def _ternary_refusal(tensor: GGUFTensorInfo) -> str:
        consumer = _TERNARY_CONSUMERS.get(tensor.type_name)
        if consumer is None:
            kernel = "and no kernel reads them yet"
        else:
            kernel = f"and the sm_75 GEMM consumes them there ({consumer})"
        return (
            f"{tensor.name} is {tensor.type_name}: the raw blocks are addressable "
            f"(read_quantized_matrix_block_rows) {kernel} -- but this entry point returns a "
            "dense tensor, so it refuses rather than dequantizing to f16: a silent upcast "
            "costs ten times the memory and makes a wrong kernel look right. The decoder "
            f"lives in python/pocketllm/quant/{tensor.type_name}.py"
        )

    def read_tensor_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> np.ndarray:
        tensor = self._tensor(name)
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        if tensor.type_name in _DENSE_DTYPES or tensor.type_name == "bf16":
            return self._read_dense_rows(tensor, row_start, row_count)
        if tensor.type_name == "q8_0":
            return self._read_q8_0_rows(tensor, row_start, row_count)
        raise NotImplementedError(f"row decode for {tensor.name} ({tensor.type_name}) is not supported")

    def read_routed_expert(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> np.ndarray:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if expert < 0 or expert >= n_experts:
            raise ValueError(f"expert {expert} is outside {tensor.name} expert count {n_experts}")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert decode for {tensor.type_name} is not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        blocks_per_row = math.ceil(in_dim / block_elems)
        row_bytes = blocks_per_row * block_bytes
        row_count = out_dim - row_start if row_count is None else int(row_count)
        if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} out_dim={out_dim}")
        expert_bytes = out_dim * row_bytes
        offset = tensor.absolute_offset + expert * expert_bytes + row_start * row_bytes
        return self._read_quantized_matrix(tensor, offset, in_dim, row_count, tensor.type_name)

    def _routed_expert_block_meta(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[GGUFTensorInfo, int, int, int, int, int, int, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if expert < 0 or expert >= n_experts:
            raise ValueError(f"expert {expert} is outside {tensor.name} expert count {n_experts}")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert raw blocks for {tensor.type_name} are not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        if in_dim % block_elems != 0:
            raise ValueError(f"{tensor.name} in_dim={in_dim} is not divisible by {block_elems}")
        blocks_per_row = in_dim // block_elems
        row_bytes = blocks_per_row * block_bytes
        row_count = out_dim - row_start if row_count is None else int(row_count)
        if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} out_dim={out_dim}")
        expert_bytes = out_dim * row_bytes
        offset = tensor.absolute_offset + expert * expert_bytes + row_start * row_bytes
        nbytes = row_count * row_bytes
        return tensor, in_dim, out_dim, blocks_per_row, block_bytes, offset, nbytes, row_count

    def routed_expert_blocks_ptr(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[int, str, int, int, int, memoryview]:
        tensor, in_dim, _out_dim, blocks_per_row, block_bytes, offset, nbytes, _row_count = self._routed_expert_block_meta(
            name,
            expert,
            row_start,
            row_count,
        )
        view = memoryview(self._mmap)[offset:offset + nbytes]
        return int(offset), tensor.type_name, in_dim, blocks_per_row, block_bytes, view

    def read_routed_layer_blocks(
        self,
        name: str | GGUFTensorInfo,
        expert_start: int = 0,
        expert_count: int | None = None,
    ) -> tuple[np.ndarray, str, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 3:
            raise ValueError(f"{tensor.name} is not a routed expert tensor")
        in_dim, out_dim, n_experts = (int(dim) for dim in tensor.dimensions)
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"routed expert raw blocks for {tensor.type_name} are not supported")
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        if in_dim % block_elems != 0:
            raise ValueError(f"{tensor.name} in_dim={in_dim} is not divisible by {block_elems}")
        blocks_per_row = in_dim // block_elems
        row_bytes = blocks_per_row * block_bytes
        expert_start = int(expert_start)
        expert_count = n_experts - expert_start if expert_count is None else int(expert_count)
        if expert_start < 0 or expert_count < 0 or expert_start + expert_count > n_experts:
            raise ValueError(f"expert range [{expert_start}, {expert_start + expert_count}) is outside {tensor.name} experts={n_experts}")
        expert_bytes = out_dim * row_bytes
        nbytes = expert_count * expert_bytes
        offset = tensor.absolute_offset + expert_start * expert_bytes
        view = memoryview(self._mmap)[offset:offset + nbytes]
        blocks = np.frombuffer(view, dtype=np.uint8, count=nbytes).reshape(expert_count, out_dim, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def read_routed_expert_blocks(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[np.ndarray, str, int]:
        tensor, in_dim, _out_dim, blocks_per_row, block_bytes, offset, nbytes, row_count = self._routed_expert_block_meta(
            name,
            expert,
            row_start,
            row_count,
        )
        view = memoryview(self._mmap)[offset:offset + nbytes]
        blocks = np.frombuffer(view, dtype=np.uint8, count=nbytes).reshape(row_count, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def _read_dense_tensor(self, tensor: GGUFTensorInfo) -> np.ndarray:
        data = self._read_at(tensor.absolute_offset, tensor.nbytes or 0)
        return self._dense_from_bytes(data, tensor.type_name, _storage_shape(tensor.dimensions))

    def _read_dense_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> np.ndarray:
        row_elems = int(tensor.dimensions[0])
        if tensor.type_name == "bf16":
            elem_size = 2
        else:
            elem_size = _DENSE_DTYPES[tensor.type_name][2]
        row_bytes = row_elems * elem_size
        data = self._read_at(tensor.absolute_offset + row_start * row_bytes, row_count * row_bytes)
        return self._dense_from_bytes(data, tensor.type_name, (row_count, row_elems))

    def _dense_from_bytes(self, data: bytes, type_name: str, shape: tuple[int, ...]) -> np.ndarray:
        if type_name == "bf16":
            # Upcast bfloat16 to float32 by shifting the raw 16 bits into the
            # high half of a uint32 and reinterpreting.  numpy has no bfloat16
            # dtype, so float32 is the loader's carrier for it; a caller that
            # wants the narrow form gets it at upload time.
            raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            return (raw << 16).view(np.float32).reshape(shape).copy()
        dtype, _elem_size = _DENSE_DTYPES[type_name]
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def _read_q8_0_tensor(self, tensor: GGUFTensorInfo) -> np.ndarray:
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        values = self._read_q8_0_rows_array(tensor.absolute_offset, row_elems, rows)
        return values.reshape(_storage_shape(tensor.dimensions)).copy()

    def read_q8_0_blocks(self, name: str | GGUFTensorInfo) -> np.ndarray:
        tensor = self._tensor(name)
        if tensor.type_name != "q8_0":
            raise NotImplementedError(f"raw q8_0 blocks for {tensor.name} ({tensor.type_name}) are not supported")
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        return self._read_q8_0_block_rows(tensor.absolute_offset, row_elems, rows)

    def read_q8_0_block_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> np.ndarray:
        tensor = self._tensor(name)
        if tensor.type_name != "q8_0":
            raise NotImplementedError(f"raw q8_0 block rows for {tensor.name} ({tensor.type_name}) are not supported")
        rows = _product(tensor.dimensions[1:])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        row_elems = int(tensor.dimensions[0])
        blocks_per_row = math.ceil(row_elems / 32)
        row_bytes = blocks_per_row * 34
        return self._read_q8_0_block_rows(tensor.absolute_offset + row_start * row_bytes, row_elems, row_count)

    def read_quantized_matrix_blocks(self, name: str | GGUFTensorInfo) -> tuple[np.ndarray, str, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 2:
            raise ValueError(f"{tensor.name} is not a 2D quantized matrix tensor")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"raw quantized matrix blocks for {tensor.name} ({tensor.type_name}) are not supported")
        row_elems = int(tensor.dimensions[0])
        rows = int(tensor.dimensions[1])
        return self._read_quantized_matrix_block_rows(tensor.absolute_offset, row_elems, rows, tensor.type_name), tensor.type_name, row_elems

    def read_quantized_matrix_block_rows(
        self,
        name: str | GGUFTensorInfo,
        row_start: int,
        row_count: int,
    ) -> tuple[np.ndarray, str, int]:
        tensor = self._tensor(name)
        if len(tensor.dimensions) != 2:
            raise ValueError(f"{tensor.name} is not a 2D quantized matrix tensor")
        if tensor.type_name not in _QUANT_BLOCK_META:
            raise NotImplementedError(f"raw quantized matrix blocks for {tensor.name} ({tensor.type_name}) are not supported")
        rows = int(tensor.dimensions[1])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        row_elems = int(tensor.dimensions[0])
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        blocks_per_row = math.ceil(row_elems / block_elems)
        row_bytes = blocks_per_row * block_bytes
        offset = tensor.absolute_offset + int(row_start) * row_bytes
        return self._read_quantized_matrix_block_rows(offset, row_elems, row_count, tensor.type_name), tensor.type_name, row_elems

    def read_quantized_matrix_rows_reference(
        self,
        name: str | GGUFTensorInfo,
        row_start: int,
        row_count: int,
    ) -> np.ndarray:
        """Decode a small quantized matrix row slice for correctness tests.

        This is intentionally a reference path.  Runtime hot paths must keep
        GGUF q4_k/q5_k weights in raw block form and use CUDA kernels instead
        of resident fp32/bf16 expansion.
        """
        tensor = self._tensor(name)
        if tensor.type_name in _TERNARY_BLOCK_META:
            raise NotImplementedError(self._ternary_refusal(tensor))
        if len(tensor.dimensions) != 2:
            raise ValueError(f"{tensor.name} is not a 2D quantized matrix tensor")
        rows = int(tensor.dimensions[1])
        if row_start < 0 or row_count < 0 or row_start + row_count > rows:
            raise ValueError(f"row range [{row_start}, {row_start + row_count}) is outside {tensor.name} rows={rows}")
        row_elems = int(tensor.dimensions[0])
        block_elems, block_bytes = _quant_block_meta(tensor.type_name)
        blocks_per_row = math.ceil(row_elems / block_elems)
        offset = tensor.absolute_offset + int(row_start) * blocks_per_row * block_bytes
        return self._read_quantized_matrix(tensor, offset, row_elems, int(row_count), tensor.type_name)

    def _read_q8_0_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> np.ndarray:
        row_elems = int(tensor.dimensions[0])
        blocks_per_row = math.ceil(row_elems / 32)
        row_bytes = blocks_per_row * 34
        values = self._read_q8_0_rows_array(tensor.absolute_offset + row_start * row_bytes, row_elems, row_count)
        return values.copy()

    def _read_q8_0_rows_array(self, offset: int, row_elems: int, rows: int) -> np.ndarray:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:34].view(np.int8).astype(np.float32)
        values = qs * d[:, :, None]
        return values.reshape(rows, blocks_per_row * 32)[:, :row_elems]

    def _read_q8_0_block_rows(self, offset: int, row_elems: int, rows: int) -> np.ndarray:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34).copy()
        return blocks

    def _read_quantized_matrix_block_rows(self, offset: int, row_elems: int, rows: int, type_name: str) -> np.ndarray:
        block_elems, block_bytes = _quant_block_meta(type_name)
        blocks_per_row = math.ceil(int(row_elems) / block_elems)
        nbytes = int(rows) * blocks_per_row * block_bytes
        data = self._read_at(offset, nbytes)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(int(rows), blocks_per_row, block_bytes).copy()
        return blocks

    def _read_quantized_matrix(self, tensor: GGUFTensorInfo, offset: int, in_dim: int, out_dim: int, type_name: str) -> np.ndarray:
        """Decode a quantized matrix's rows through the shared block decoders.

        The block geometry and the decoder come from :mod:`pocketllm.quant.formats`,
        so this reader never restates a format.  ``iq4_xs`` is read here too -- it
        shares IQ4_NL's nibble split through that module rather than through a
        copy of it.
        """
        fmt = formats.format_for(type_name)
        blocks_per_row = math.ceil(in_dim / fmt.block_elems)
        nbytes = out_dim * blocks_per_row * fmt.block_bytes
        data = self._read_at(offset, nbytes)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(out_dim, blocks_per_row, fmt.block_bytes)
        values = fmt.decode(blocks)
        return values.reshape(out_dim, blocks_per_row * fmt.block_elems)[:, :in_dim].copy()
