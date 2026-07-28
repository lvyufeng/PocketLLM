from __future__ import annotations

import math
import mmap
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.loader.gguf.reader import GGUFFile, GGUFReader, GGUFTensorInfo


_GGUF_READER_PROFILE = os.getenv("DEEPSEEK_GGUF_READER_PROFILE", "0").lower() in {"1", "true", "yes"}
_GGUF_READER_PROFILE_LIMIT = int(os.getenv("DEEPSEEK_GGUF_READER_PROFILE_LIMIT", "64"))
_GGUF_READER_PROFILE_COUNT = 0


_IQ2XXS_GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
)

_DENSE_DTYPES = {
    "f32": ("<f4", torch.float32, 4),
    "f16": ("<f2", torch.float16, 2),
    "i32": ("<i4", torch.int32, 4),
}

# Quantized routed/matrix block geometry: type_name -> (block_elems, block_bytes).
# Mirrors GGML_QUANT_SIZES for the raw GGUF block formats supported by
# runtime readers.  Q4_K/Q5_K are used by MiniMax dense tensors; keep them as
# raw blocks in runtime and use reference dequant only in tests.
_QUANT_BLOCK_META = {
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "iq2_xxs": (256, 66),
    "iq2_xs": (256, 74),
    "iq3_xxs": (256, 98),
    "iq1_m": (256, 56),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "iq4_xs": (256, 136),
}


def _quant_block_meta(type_name: str) -> tuple[int, int]:
    try:
        return _QUANT_BLOCK_META[type_name]
    except KeyError as exc:
        raise NotImplementedError(f"no block geometry for quant type {type_name}") from exc


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


def _extract_ggml_table(table_name: str, *, dtype: str, expected: int) -> np.ndarray:
    """Parse a vendored llama.cpp IQ lookup table without duplicating constants."""
    header = Path(__file__).parents[2] / "csrc" / "llama_mmq" / "ggml-common.h"
    text = header.read_text(encoding="utf-8")
    match = re.search(
        rf"GGML_TABLE_BEGIN\([^,]+,\s*{re.escape(table_name)},\s*{expected}\)(.*?)GGML_TABLE_END\(\)",
        text,
        flags=re.S,
    )
    if match is None:
        raise RuntimeError(f"failed to locate {table_name} in {header}")
    values = [int(item, 16) for item in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    if len(values) != int(expected):
        raise RuntimeError(f"{table_name} expected {expected} entries, got {len(values)}")
    return np.asarray(values, dtype=np.dtype(dtype))


@lru_cache(maxsize=1)
def _iq2xs_grid() -> np.ndarray:
    return _extract_ggml_table("iq2xs_grid", dtype="<u8", expected=512)


@lru_cache(maxsize=1)
def _iq3xxs_grid() -> np.ndarray:
    return _extract_ggml_table("iq3xxs_grid", dtype="<u4", expected=256)


@lru_cache(maxsize=1)
def _iq2xs_signed_grid() -> np.ndarray:
    base = np.empty((512, 8), dtype=np.int8)
    for idx, value in enumerate(_iq2xs_grid()):
        base[idx] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8).astype(np.int8)
    signs = np.empty((128, 8), dtype=np.int8)
    masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    for idx in range(128):
        sign_mask = idx | ((int(idx).bit_count() & 1) << 7)
        signs[idx] = np.where((sign_mask & masks) != 0, -1, 1).astype(np.int8)
    return (base[:, None, :].astype(np.int16) * signs[None, :, :].astype(np.int16)).astype(np.int8)


@lru_cache(maxsize=1)
def _iq3xxs_signed_grid() -> np.ndarray:
    base = np.empty((256, 4), dtype=np.int8)
    for idx, value in enumerate(_iq3xxs_grid()):
        base[idx] = np.frombuffer(int(value).to_bytes(4, "little"), dtype=np.uint8).astype(np.int8)
    signs = np.empty((128, 8), dtype=np.int8)
    masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    for idx in range(128):
        sign_mask = idx | ((int(idx).bit_count() & 1) << 7)
        signs[idx] = np.where((sign_mask & masks) != 0, -1, 1).astype(np.int8)
    expanded = np.empty((256, 128, 8), dtype=np.int8)
    expanded[:, :, 0:4] = (base[:, None, :].astype(np.int16) * signs[None, :, 0:4].astype(np.int16)).astype(np.int8)
    expanded[:, :, 4:8] = (base[:, None, :].astype(np.int16) * signs[None, :, 4:8].astype(np.int16)).astype(np.int8)
    return expanded


@lru_cache(maxsize=1)
def _iq2xxs_signed_grid() -> np.ndarray:
    base = np.empty((256, 8), dtype=np.int8)
    for idx, value in enumerate(_IQ2XXS_GRID):
        base[idx] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8).astype(np.int8)
    signs = np.empty((128, 8), dtype=np.int8)
    masks = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    for idx in range(128):
        sign_mask = idx | ((int(idx).bit_count() & 1) << 7)
        signs[idx] = np.where((sign_mask & masks) != 0, -1, 1).astype(np.int8)
    return (base[:, None, :].astype(np.int16) * signs[None, :, :].astype(np.int16)).astype(np.int8)


@lru_cache(maxsize=1)
def get_iq2xxs_signed_grid_tensor() -> torch.Tensor:
    return torch.from_numpy(_iq2xxs_signed_grid().copy()).contiguous().to(device="cpu")


@lru_cache(maxsize=1)
def get_iq2xs_signed_grid_tensor() -> torch.Tensor:
    return torch.from_numpy(_iq2xs_signed_grid().copy()).contiguous().to(device="cpu")


@lru_cache(maxsize=1)
def get_iq3xxs_signed_grid_tensor() -> torch.Tensor:
    return torch.from_numpy(_iq3xxs_signed_grid().copy()).contiguous().to(device="cpu")


@lru_cache(maxsize=1)
def get_iq2xs_iq3xxs_signed_grid_tensor() -> torch.Tensor:
    # GLM MoE uses IQ2_XS for routed w1/w3 and IQ3_XXS for routed w2.  Keep a
    # single packed tensor for that runtime, but preserve per-format helpers
    # above so other callers can fetch the exact table they need.
    return torch.cat(
        [
            get_iq2xs_signed_grid_tensor().reshape(-1),
            get_iq3xxs_signed_grid_tensor().reshape(-1),
        ]
    ).contiguous()


@lru_cache(maxsize=1)
def get_iq1_grid_tensor() -> torch.Tensor:
    from src.loader.gguf.iq1_grid import iq1_grid_i8

    return torch.from_numpy(iq1_grid_i8().copy()).contiguous().to(device="cpu")


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

    def read_tensor(self, name: str | GGUFTensorInfo) -> torch.Tensor:
        tensor = self._tensor(name)
        if tensor.type_name in _DENSE_DTYPES or tensor.type_name == "bf16":
            return self._read_dense_tensor(tensor)
        if tensor.type_name == "q8_0":
            return self._read_q8_0_tensor(tensor)
        if tensor.type_name in _QUANT_BLOCK_META and len(tensor.dimensions) == 2:
            return self._read_quantized_matrix(tensor, tensor.absolute_offset, int(tensor.dimensions[0]), int(tensor.dimensions[1]), tensor.type_name)
        raise NotImplementedError(f"payload decode for {tensor.name} ({tensor.type_name}) is not supported by read_tensor")

    def read_tensor_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
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
    ) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, str, int]:
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
        blocks = torch.frombuffer(view, dtype=torch.uint8, count=nbytes).view(expert_count, out_dim, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def read_routed_expert_blocks(
        self,
        name: str | GGUFTensorInfo,
        expert: int,
        row_start: int = 0,
        row_count: int | None = None,
    ) -> tuple[torch.Tensor, str, int]:
        tensor, in_dim, _out_dim, blocks_per_row, block_bytes, offset, nbytes, row_count = self._routed_expert_block_meta(
            name,
            expert,
            row_start,
            row_count,
        )
        view = memoryview(self._mmap)[offset:offset + nbytes]
        blocks = torch.frombuffer(view, dtype=torch.uint8, count=nbytes).view(row_count, blocks_per_row, block_bytes)
        return blocks, tensor.type_name, in_dim

    def _read_dense_tensor(self, tensor: GGUFTensorInfo) -> torch.Tensor:
        data = self._read_at(tensor.absolute_offset, tensor.nbytes or 0)
        return self._dense_from_bytes(data, tensor.type_name, _storage_shape(tensor.dimensions))

    def _read_dense_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        if tensor.type_name == "bf16":
            elem_size = 2
        else:
            elem_size = _DENSE_DTYPES[tensor.type_name][2]
        row_bytes = row_elems * elem_size
        data = self._read_at(tensor.absolute_offset + row_start * row_bytes, row_count * row_bytes)
        return self._dense_from_bytes(data, tensor.type_name, (row_count, row_elems))

    def _dense_from_bytes(self, data: bytes, type_name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if type_name == "bf16":
            raw = np.frombuffer(data, dtype="<u2").astype(np.uint32)
            array = (raw << 16).view(np.float32).reshape(shape).copy()
            return torch.from_numpy(array).to(torch.bfloat16)
        dtype, _torch_dtype, _elem_size = _DENSE_DTYPES[type_name]
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return torch.from_numpy(array)

    def _read_q8_0_tensor(self, tensor: GGUFTensorInfo) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        values = self._read_q8_0_rows_array(tensor.absolute_offset, row_elems, rows)
        return torch.from_numpy(values.reshape(_storage_shape(tensor.dimensions)).copy())

    def read_q8_0_blocks(self, name: str | GGUFTensorInfo) -> torch.Tensor:
        tensor = self._tensor(name)
        if tensor.type_name != "q8_0":
            raise NotImplementedError(f"raw q8_0 blocks for {tensor.name} ({tensor.type_name}) are not supported")
        row_elems = int(tensor.dimensions[0])
        rows = _product(tensor.dimensions[1:])
        return self._read_q8_0_block_rows(tensor.absolute_offset, row_elems, rows)

    def read_q8_0_block_rows(self, name: str | GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
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

    def read_quantized_matrix_blocks(self, name: str | GGUFTensorInfo) -> tuple[torch.Tensor, str, int]:
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
    ) -> tuple[torch.Tensor, str, int]:
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
    ) -> torch.Tensor:
        """Decode a small quantized matrix row slice for correctness tests.

        This is intentionally a reference path.  Runtime hot paths must keep
        GGUF q4_k/q5_k weights in raw block form and use CUDA kernels instead
        of resident fp32/bf16 expansion.
        """
        tensor = self._tensor(name)
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

    def _read_q8_0_rows(self, tensor: GGUFTensorInfo, row_start: int, row_count: int) -> torch.Tensor:
        row_elems = int(tensor.dimensions[0])
        blocks_per_row = math.ceil(row_elems / 32)
        row_bytes = blocks_per_row * 34
        values = self._read_q8_0_rows_array(tensor.absolute_offset + row_start * row_bytes, row_elems, row_count)
        return torch.from_numpy(values.copy())

    def _read_q8_0_rows_array(self, offset: int, row_elems: int, rows: int) -> np.ndarray:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:34].view(np.int8).astype(np.float32)
        values = qs * d[:, :, None]
        return values.reshape(rows, blocks_per_row * 32)[:, :row_elems]

    def _read_q8_0_block_rows(self, offset: int, row_elems: int, rows: int) -> torch.Tensor:
        blocks_per_row = math.ceil(row_elems / 32)
        data = self._read_at(offset, rows * blocks_per_row * 34)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 34).copy()
        return torch.from_numpy(blocks).to(device="cpu")

    def _read_quantized_matrix_block_rows(self, offset: int, row_elems: int, rows: int, type_name: str) -> torch.Tensor:
        block_elems, block_bytes = _quant_block_meta(type_name)
        blocks_per_row = math.ceil(int(row_elems) / block_elems)
        nbytes = int(rows) * blocks_per_row * block_bytes
        data = self._read_at(offset, nbytes)
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(int(rows), blocks_per_row, block_bytes).copy()
        return torch.from_numpy(blocks).to(device="cpu")

    def _read_quantized_matrix(self, tensor: GGUFTensorInfo, offset: int, in_dim: int, out_dim: int, type_name: str) -> torch.Tensor:
        blocks_per_row = math.ceil(in_dim / 256)
        if type_name == "q2_k":
            values = self._read_q2_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "q3_k":
            values = self._read_q3_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq2_xxs":
            values = self._read_iq2_xxs_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq2_xs":
            values = self._read_iq2_xs_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq3_xxs":
            values = self._read_iq3_xxs_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq1_m":
            values = self._read_iq1_m_rows(offset, out_dim, blocks_per_row)
        elif type_name == "q4_k":
            values = self._read_q4_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "q5_k":
            values = self._read_q5_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "q6_k":
            values = self._read_q6_k_rows(offset, out_dim, blocks_per_row)
        elif type_name == "iq4_xs":
            values = self._read_iq4_xs_rows(offset, out_dim, blocks_per_row)
        else:
            raise NotImplementedError(type_name)
        return torch.from_numpy(values.reshape(out_dim, blocks_per_row * 256)[:, :in_dim].copy())

    def _read_q2_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 84
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 84)
        scales = blocks[:, :, :16]
        qs = blocks[:, :, 16:80]
        d = _f16_bytes_to_f32(blocks[:, :, 80:82])
        dmin = _f16_bytes_to_f32(blocks[:, :, 82:84])
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for group in range(16):
            half_block = group // 8
            group_in_half = group % 8
            shift = (group_in_half // 2) * 2
            byte_start = half_block * 32 + (group_in_half % 2) * 16
            q = ((qs[:, :, byte_start:byte_start + 16] >> shift) & 0x03).astype(np.float32)
            scale = (scales[:, :, group] & 0x0F).astype(np.float32)
            minv = (scales[:, :, group] >> 4).astype(np.float32)
            out[:, :, group * 16:(group + 1) * 16] = d[:, :, None] * scale[:, :, None] * q - dmin[:, :, None] * minv[:, :, None]
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q2_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq2_xxs_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 66
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 66)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:66]
        signed_grid = _iq2xxs_signed_grid()
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for sub in range(8):
            chunk = qs[:, :, sub * 8:(sub + 1) * 8]
            aux1 = (
                chunk[:, :, 4].astype(np.uint32)
                | (chunk[:, :, 5].astype(np.uint32) << 8)
                | (chunk[:, :, 6].astype(np.uint32) << 16)
                | (chunk[:, :, 7].astype(np.uint32) << 24)
            )
            ls = (2 * (aux1 >> 28) + 1).astype(np.float32)
            for part in range(4):
                grid_ids = chunk[:, :, part].astype(np.int64)
                sign_idx = ((aux1 >> (7 * part)) & 127).astype(np.int64)
                values = signed_grid[grid_ids, sign_idx].astype(np.float32)
                start = sub * 32 + part * 8
                out[:, :, start:start + 8] = 0.125 * d[:, :, None] * ls[:, :, None] * values
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq2_xxs rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq2_xs_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode IQ2_XS rows using llama.cpp grid/sign layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 74
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 74)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:66]
        scales = blocks[:, :, 66:74]
        signed_grid = _iq2xs_signed_grid()
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for sub in range(8):
            chunk = qs[:, :, sub * 8:(sub + 1) * 8]
            ls0 = (scales[:, :, sub] & 0x0F).astype(np.float32)
            ls1 = (scales[:, :, sub] >> 4).astype(np.float32)
            for part in range(4):
                q = chunk[:, :, part * 2].astype(np.uint16) | (chunk[:, :, part * 2 + 1].astype(np.uint16) << 8)
                grid_ids = (q & 0x01FF).astype(np.int64)
                sign_idx = (q >> 9).astype(np.int64)
                values = signed_grid[grid_ids, sign_idx].astype(np.float32)
                start = sub * 32 + part * 8
                scale = (ls0 if part < 2 else ls1) + 0.5
                out[:, :, start:start + 8] = 0.25 * d[:, :, None] * scale[:, :, None] * values
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq2_xs rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq3_xxs_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode IQ3_XXS rows using llama.cpp grid/sign layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 98
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 98)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        qs = blocks[:, :, 2:98]
        # IQ3_XXS block layout (block_iq3_xxs, 98 bytes):
        #   [0:2]   d (fp16)
        #   [2:66]  grid indices: 8 sub-blocks x 8 bytes  (QK_K/4 = 64 bytes)
        #   [66:98] scales_and_signs: 8 sub-blocks x 4 bytes aux uint32 (32 bytes)
        # The grid-index region and the aux region are SEPARATE, not interleaved.
        grid_idx = qs[:, :, 0:64]                              # [R, B, 64]
        aux_bytes = qs[:, :, 64:96]                            # [R, B, 32]
        signed_grid = _iq3xxs_signed_grid()
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for sub in range(8):
            qbytes = grid_idx[:, :, sub * 8:sub * 8 + 8]       # [R, B, 8] grid indices
            aux = (
                aux_bytes[:, :, sub * 4 + 0].astype(np.uint32)
                | (aux_bytes[:, :, sub * 4 + 1].astype(np.uint32) << 8)
                | (aux_bytes[:, :, sub * 4 + 2].astype(np.uint32) << 16)
                | (aux_bytes[:, :, sub * 4 + 3].astype(np.uint32) << 24)
            )
            ls = (aux >> 28).astype(np.float32)
            for part in range(8):
                grid_ids = qbytes[:, :, part].astype(np.int64)
                sign_idx = ((aux >> (7 * (part // 2))) & 127).astype(np.int64)
                values = signed_grid[grid_ids, sign_idx, part % 2 * 4:part % 2 * 4 + 4].astype(np.float32)
                start = sub * 32 + part * 4
                out[:, :, start:start + 4] = 0.5 * d[:, :, None] * (ls[:, :, None] + 0.5) * values
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq3_xxs rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_q3_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode Q3_K rows to float32 using llama.cpp/ggml layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 110
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 110)
        hmask = blocks[:, :, 0:32]
        qs = blocks[:, :, 32:96]
        scales = blocks[:, :, 96:108]
        d = _f16_bytes_to_f32(blocks[:, :, 108:110])
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for group in range(16):
            if group < 8:
                scale = (scales[:, :, group] & 0x0F).astype(np.int16)
            else:
                scale = (scales[:, :, group - 8] >> 4).astype(np.int16)
            scale = (scale - 8).astype(np.float32)
            qbytes = qs[:, :, group * 4:(group + 1) * 4]
            qlow = np.empty((rows, blocks_per_row, 16), dtype=np.uint8)
            qlow[:, :, 0:4] = qbytes & 0x03
            qlow[:, :, 4:8] = (qbytes >> 2) & 0x03
            qlow[:, :, 8:12] = (qbytes >> 4) & 0x03
            qlow[:, :, 12:16] = (qbytes >> 6) & 0x03
            bits = ((hmask[:, :, group * 2:group * 2 + 2][:, :, :, None] >> np.arange(8, dtype=np.uint8)) & 1).reshape(rows, blocks_per_row, 16)
            q = qlow.astype(np.int16) - np.where(bits == 0, 4, 0).astype(np.int16)
            out[:, :, group * 16:(group + 1) * 16] = d[:, :, None] * scale[:, :, None] * q.astype(np.float32)
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q3_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_q4_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode Q4_K rows to float32 using llama.cpp/ggml layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 144
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 144)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        dmin = _f16_bytes_to_f32(blocks[:, :, 2:4])
        scales = blocks[:, :, 4:16]
        qs = blocks[:, :, 16:144]
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for pair in range(4):
            q = qs[:, :, pair * 32:(pair + 1) * 32]
            sc, mn = _get_scale_min_k4(scales, pair * 2)
            out[:, :, pair * 64:pair * 64 + 32] = (
                d[:, :, None] * sc.astype(np.float32)[:, :, None] * (q & 0x0F).astype(np.float32)
                - dmin[:, :, None] * mn.astype(np.float32)[:, :, None]
            )
            sc, mn = _get_scale_min_k4(scales, pair * 2 + 1)
            out[:, :, pair * 64 + 32:pair * 64 + 64] = (
                d[:, :, None] * sc.astype(np.float32)[:, :, None] * (q >> 4).astype(np.float32)
                - dmin[:, :, None] * mn.astype(np.float32)[:, :, None]
            )
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q4_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_q5_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode Q5_K rows to float32 using llama.cpp/ggml layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 176
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 176)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        dmin = _f16_bytes_to_f32(blocks[:, :, 2:4])
        scales = blocks[:, :, 4:16]
        qh = blocks[:, :, 16:48]
        qs = blocks[:, :, 48:176]
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        u1 = 1
        u2 = 2
        for pair in range(4):
            q = qs[:, :, pair * 32:(pair + 1) * 32]
            high = qh[:, :, :32]
            sc, mn = _get_scale_min_k4(scales, pair * 2)
            q_low = (q & 0x0F).astype(np.float32) + np.where((high & u1) != 0, 16.0, 0.0).astype(np.float32)
            out[:, :, pair * 64:pair * 64 + 32] = (
                d[:, :, None] * sc.astype(np.float32)[:, :, None] * q_low
                - dmin[:, :, None] * mn.astype(np.float32)[:, :, None]
            )
            sc, mn = _get_scale_min_k4(scales, pair * 2 + 1)
            q_high = (q >> 4).astype(np.float32) + np.where((high & u2) != 0, 16.0, 0.0).astype(np.float32)
            out[:, :, pair * 64 + 32:pair * 64 + 64] = (
                d[:, :, None] * sc.astype(np.float32)[:, :, None] * q_high
                - dmin[:, :, None] * mn.astype(np.float32)[:, :, None]
            )
            u1 <<= 2
            u2 <<= 2
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q5_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_q6_k_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode Q6_K rows to float32 using llama.cpp/ggml layout."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 210
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 210)
        ql = blocks[:, :, 0:128]
        qh = blocks[:, :, 128:192]
        scales = blocks[:, :, 192:208].view(np.int8).astype(np.float32)
        d = _f16_bytes_to_f32(blocks[:, :, 208:210])
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        d_ = d[:, :, None]
        # ggml dequantize_row_q6_K layout: two 128-wide halves per super-block.
        # For l in 0..31 within a half: is = l // 16, and the four sub-lanes use
        # scales sc[is+0], sc[is+2], sc[is+4], sc[is+6].
        is_idx = np.concatenate([np.zeros(16, dtype=np.int64), np.ones(16, dtype=np.int64)])
        for half in range(2):
            base = half * 128
            ql_h = ql[:, :, half * 64:(half + 1) * 64]
            qh_h = qh[:, :, half * 32:(half + 1) * 32]
            sc_h = scales[:, :, half * 8:(half + 1) * 8]
            ql_l = ql_h[:, :, 0:32].astype(np.uint8)
            ql_l32 = ql_h[:, :, 32:64].astype(np.uint8)
            qh_l = qh_h[:, :, 0:32].astype(np.uint8)
            q1 = ((ql_l & 0x0F) | (((qh_l >> 0) & 0x03) << 4)).astype(np.int16) - 32
            q2 = ((ql_l32 & 0x0F) | (((qh_l >> 2) & 0x03) << 4)).astype(np.int16) - 32
            q3 = ((ql_l >> 4) | (((qh_l >> 4) & 0x03) << 4)).astype(np.int16) - 32
            q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 0x03) << 4)).astype(np.int16) - 32
            sc1 = sc_h[:, :, is_idx + 0]
            sc2 = sc_h[:, :, is_idx + 2]
            sc3 = sc_h[:, :, is_idx + 4]
            sc4 = sc_h[:, :, is_idx + 6]
            out[:, :, base + 0:base + 32] = d_ * sc1 * q1.astype(np.float32)
            out[:, :, base + 32:base + 64] = d_ * sc2 * q2.astype(np.float32)
            out[:, :, base + 64:base + 96] = d_ * sc3 * q3.astype(np.float32)
            out[:, :, base + 96:base + 128] = d_ * sc4 * q4.astype(np.float32)
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=q6_k rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq4_xs_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Reference-decode IQ4_XS rows to float32."""
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 136
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, blocks_per_row, 136)
        d = _f16_bytes_to_f32(blocks[:, :, 0:2])
        scales_h = blocks[:, :, 2:4].view("<u2").reshape(rows, blocks_per_row)
        scales_l = blocks[:, :, 4:8]
        qs = blocks[:, :, 8:136]
        kvalues = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
        out = np.empty((rows, blocks_per_row, 256), dtype=np.float32)
        for group in range(8):
            scale = ((scales_l[:, :, group // 2] >> (4 if group & 1 else 0)) & 0x0F).astype(np.int16)
            scale |= (((scales_h >> (2 * group)) & 0x03).astype(np.int16) << 4)
            scale = (scale - 32).astype(np.float32)
            q = qs[:, :, group * 16:(group + 1) * 16]
            lo = kvalues[q & 0x0F]
            hi = kvalues[q >> 4]
            values = np.empty((rows, blocks_per_row, 32), dtype=np.float32)
            # ggml iq4_xs packs each 16-byte group as 16 low nibbles then 16 high
            # nibbles (block layout), not interleaved.
            values[:, :, 0:16] = lo
            values[:, :, 16:32] = hi
            out[:, :, group * 32:(group + 1) * 32] = d[:, :, None] * scale[:, :, None] * values
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq4_xs rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out

    def _read_iq1_m_rows(self, offset: int, rows: int, blocks_per_row: int) -> np.ndarray:
        """Decode IQ1_M rows to float32.

        IQ1_M block layout is 56 bytes for 256 values:
        ``qs[32] + qh[16] + scales[8]``.  The formula mirrors llama.cpp
        gguf-py ``IQ1_M.dequantize_blocks``; imatrix is only needed while
        producing IQ1_M, not while decoding it.
        """
        global _GGUF_READER_PROFILE_COUNT
        profile = _GGUF_READER_PROFILE and _GGUF_READER_PROFILE_COUNT < _GGUF_READER_PROFILE_LIMIT
        t0 = time.perf_counter() if profile else 0.0
        nbytes = rows * blocks_per_row * 56
        data = self._read_at(offset, nbytes)
        if profile:
            t_read = time.perf_counter()

        n_blocks = rows * blocks_per_row
        blocks = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, 56)
        qs = blocks[:, :32]
        qh = blocks[:, 32:48]
        scales = blocks[:, 48:56].view(np.uint16)

        # Reconstruct the shared f16 super-block scale from the high nibbles of
        # four uint16 scale words.
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[:, 0] | d[:, 1] | d[:, 2] | d[:, 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

        # Low 12 bits of the scale words contain 4 packed 3-bit local scales.
        local_scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        local_scales = (local_scales & 0x07).reshape((n_blocks, -1))
        dl = d * (2 * local_scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))

        qh_parts = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        qidx = qs.astype(np.uint16) | ((qh_parts & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))

        delta = np.where(qh_parts & 0x08 == 0, np.float32(0.125), np.float32(-0.125))
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))

        from src.loader.gguf.iq1_grid import iq1_grid_i8

        grid = iq1_grid_i8().astype(np.float32, copy=False)[qidx.reshape(-1)].reshape((n_blocks, -1, 2, 2, 8))
        out = (dl * (grid + delta)).reshape((rows, blocks_per_row, 256))
        if profile:
            t_done = time.perf_counter()
            _GGUF_READER_PROFILE_COUNT += 1
            print(
                f"gguf_reader_profile type=iq1_m rows={rows} blocks_per_row={blocks_per_row} bytes={nbytes} "
                f"read={t_read - t0:.6f}s decode={t_done - t_read:.6f}s total={t_done - t0:.6f}s",
                flush=True,
            )
        return out
