"""Safetensors checkpoint index and shard reader helpers.

Two access patterns live here. `iter_safetensors_shards` opens a shard with
`safetensors.safe_open` and hands the caller a framework reader, which is what a
loader that walks every tensor once wants. `MmapSafetensors` maps each shard once
and hands out torch views over the raw bytes, which is what a checkpoint too large
for RAM wants: a `view` faults in only the pages it touches, so reading one routed
expert out of a 7 GiB layer costs one expert's worth of I/O rather than a layer's.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
import os
import struct
import warnings
from typing import Iterator

import numpy as np
import torch
from safetensors import safe_open


# safetensors' dtype strings, each with the numpy dtype that reads its bytes and the torch dtype
# they mean. Nothing here is exotic: bf16 and every fp8 have no numpy dtype of their own, so they
# map through an unsigned integer of the same width and are reinterpreted by `Tensor.view`, and
# the fp4 checkpoint dtypes are stored as `I8` -- two codes per byte -- which is why that entry is
# a plain int8 rather than a float type.
SAFETENSORS_DTYPES: dict[str, tuple[np.dtype, torch.dtype]] = {
    "F64": (np.dtype("<f8"), torch.float64),
    "F32": (np.dtype("<f4"), torch.float32),
    "F16": (np.dtype("<f2"), torch.float16),
    "BF16": (np.dtype("<u2"), torch.bfloat16),
    "F8_E4M3": (np.dtype("<u1"), torch.float8_e4m3fn),
    "F8_E5M2": (np.dtype("<u1"), torch.float8_e5m2),
    "F8_E8M0": (np.dtype("<u1"), torch.float8_e8m0fnu),
    "I64": (np.dtype("<i8"), torch.int64),
    "I32": (np.dtype("<i4"), torch.int32),
    "I16": (np.dtype("<i2"), torch.int16),
    "I8": (np.dtype("<i1"), torch.int8),
    "U8": (np.dtype("<u1"), torch.uint8),
    "BOOL": (np.dtype("?"), torch.bool),
}

# `Tensor.view` only reinterprets between equal-width types, so these are the dtypes whose mapped
# numpy array needs the extra step. Everything else is already its own numpy dtype.
_VIEW_DTYPES = frozenset(
    {torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu}
)



@dataclass(frozen=True)
class SafetensorsIndex:
    root: str
    weight_map: dict[str, str]
    file_to_keys: dict[str, list[str]]

    def shard_path(self, file_name: str) -> str:
        return os.path.join(self.root, file_name)


def group_weight_map_by_file(weight_map: dict[str, str]) -> dict[str, list[str]]:
    file_to_keys: dict[str, list[str]] = {}
    for key, file_name in weight_map.items():
        file_to_keys.setdefault(file_name, []).append(key)
    return file_to_keys


def read_safetensors_index(root: str) -> SafetensorsIndex:
    root = os.path.abspath(root)
    weight_map_path = os.path.join(root, "model.safetensors.index.json")
    with open(weight_map_path) as f:
        weight_map = json.load(f)["weight_map"]
    return SafetensorsIndex(root=root, weight_map=weight_map, file_to_keys=group_weight_map_by_file(weight_map))


class SafetensorsShardReader:
    def __init__(self, path: str, *, device: str = "cpu") -> None:
        self.path = path
        self.device = device
        self._reader = None

    def __enter__(self):
        self._reader = safe_open(self.path, framework="pt", device=self.device)
        return self._reader.__enter__()

    def __exit__(self, exc_type, exc, tb):
        assert self._reader is not None
        return self._reader.__exit__(exc_type, exc, tb)


def iter_safetensors_shards(
    index: SafetensorsIndex,
    file_to_keys: dict[str, list[str]] | None = None,
    *,
    device: str = "cpu",
) -> Iterator[tuple[int, int, str, list[str], object]]:
    selected = file_to_keys if file_to_keys is not None else index.file_to_keys
    total_files = len(selected)
    for file_idx, (file_name, keys) in enumerate(selected.items(), 1):
        file_path = index.shard_path(file_name)
        with safe_open(file_path, framework="pt", device=device) as reader:
            yield file_idx, total_files, file_name, keys, reader


def filter_file_to_keys(
    file_to_keys: dict[str, list[str]],
    predicate,
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    for file_name, keys in file_to_keys.items():
        selected = [key for key in keys if predicate(key)]
        if selected:
            filtered[file_name] = selected
    return filtered


@dataclass(frozen=True)
class TensorEntry:
    """Where one tensor lives: which shard, and its byte range inside that shard's data section."""

    file_name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


class MmapSafetensors:
    """Read-only view over a sharded safetensors checkpoint.

    Every shard is mapped once, lazily, and `view` returns a torch tensor aliasing the mapped bytes
    -- no copy, no device transfer. Slicing that view only faults in the pages it touches, so
    reading one expert out of a layer costs one expert's worth of I/O rather than a whole layer's.
    That is the point: the checkpoints this serves are hundreds of GiB and the compute device has
    tens.

    Reading a header is the only eager work at construction: 48 shards is a few MB of JSON for the
    largest checkpoint here, and it gives every tensor's dtype and shape without touching a byte of
    data. `nbytes_total` and `group_bytes` turn that into the placement accounting a caller needs
    before deciding what stays on the device.

    The mapping is read-only, so the tensors handed out must not be written through; callers that
    want to mutate copy first (`load`).
    """

    def __init__(self, root: str, *, index: SafetensorsIndex | None = None) -> None:
        self.root = os.path.abspath(root)
        # Only ever used to report which shard a tensor came from, so a caller that has already read
        # the index can pass it in and save the second parse of a multi-megabyte JSON.
        self.index = index
        self.entries: dict[str, TensorEntry] = {}
        self._maps: dict[str, np.memmap] = {}
        self._data_offsets: dict[str, int] = {}
        self._views: dict[str, torch.Tensor] = {}
        self._index_headers()

    def _shard_files(self) -> list[str]:
        if self.index is not None:
            return sorted(set(self.index.weight_map.values()))
        if os.path.exists(os.path.join(self.root, "model.safetensors.index.json")):
            return sorted(set(read_safetensors_index(self.root).weight_map.values()))
        # An unsharded checkpoint has no index at all; `save_file` writes one file and nothing else.
        if os.path.exists(os.path.join(self.root, "model.safetensors")):
            return ["model.safetensors"]
        raise FileNotFoundError(f"no safetensors checkpoint at {self.root}")

    def _index_headers(self) -> None:
        for file_name in self._shard_files():
            path = os.path.join(self.root, file_name)
            with open(path, "rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(header_len))
            self._data_offsets[file_name] = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                if key in self.entries:
                    # Two shards cannot both hold the same tensor name; if they do, the checkpoint
                    # is not one checkpoint, and silently keeping the last one would hide that.
                    raise ValueError(
                        f"{key} appears in both {self.entries[key].file_name} and {file_name}"
                    )
                begin, end = meta["data_offsets"]
                self.entries[key] = TensorEntry(
                    file_name=file_name,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    begin=begin,
                    end=end,
                )

    def _map(self, file_name: str) -> np.memmap:
        mm = self._maps.get(file_name)
        if mm is None:
            mm = np.memmap(os.path.join(self.root, file_name), dtype=np.uint8, mode="r")
            self._maps[file_name] = mm
        return mm

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self):
        return self.entries.keys()

    def entry(self, key: str) -> TensorEntry:
        try:
            return self.entries[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in the checkpoint at {self.root}") from None

    def entry_view(self, entry: TensorEntry) -> torch.Tensor:
        """Like `view`, for a caller that already resolved the entry.

        Deliberately uncached: the tensors this serves are the ones too large to hold views of, and
        a caller that wants one expert out of a 7 GiB layer should not leave a handle on the whole
        layer behind. `view` is the cached form, for the small tensors that are read repeatedly.
        """
        np_dtype, torch_dtype = SAFETENSORS_DTYPES[entry.dtype]
        base = self._data_offsets[entry.file_name]
        raw = self._map(entry.file_name)[base + entry.begin : base + entry.end]
        arr = raw.view(np_dtype)
        # The mapping is read-only; torch warns about that but we never write through these views.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable.*")
            tensor = torch.from_numpy(arr)
        if torch_dtype in _VIEW_DTYPES:
            tensor = tensor.view(torch_dtype)
        return tensor.reshape(entry.shape)

    def view(self, key: str) -> torch.Tensor:
        """Torch tensor aliasing the mapped bytes (no copy, no device transfer)."""
        cached = self._views.get(key)
        if cached is not None:
            return cached
        tensor = self.entry_view(self.entry(key))
        self._views[key] = tensor
        return tensor

    def nbytes_total(self) -> int:
        return sum(entry.nbytes for entry in self.entries.values())

    def group_bytes(self, group_of) -> dict[object, int]:
        """Sum every tensor's bytes into the group `group_of(key)` names.

        The caller supplies the grouping, because what counts as one placement unit -- a layer's
        dense weights, its routed experts, an embedding table -- is a property of the model, not of
        the file format.
        """
        totals: dict[object, int] = {}
        for key, entry in self.entries.items():
            group = group_of(key)
            if group is None:
                continue
            totals[group] = totals.get(group, 0) + entry.nbytes
        return totals

    def advise_dontneed(self, key: str) -> int:
        """Drop this process's page-table entries for one tensor's byte range.

        Used after a tensor has been copied into resident host memory: without it the process keeps
        the mapped pages referenced on top of the copy. This only drops *this* process's
        references -- the pages stay in the shared page cache, so a sibling rank reading the same
        region is not forced back to the disk. Returns the bytes advised.
        """
        return self.advise_dontneed_entry(self.entry(key))

    def advise_dontneed_entry(self, entry: TensorEntry) -> int:
        raw = self._map(entry.file_name)
        handle = getattr(raw, "_mmap", None)
        if handle is None or not hasattr(handle, "madvise"):
            return 0
        base = self._data_offsets[entry.file_name]
        page = mmap.PAGESIZE
        # Align inwards so only pages wholly inside this tensor are dropped.
        start = -(-(base + entry.begin) // page) * page
        end = ((base + entry.end) // page) * page
        if end <= start:
            return 0
        try:
            handle.madvise(mmap.MADV_DONTNEED, start, end - start)
        except (OSError, ValueError):
            return 0
        return end - start

    def load(self, key: str, *, device=None, dtype=None) -> torch.Tensor:
        """Copy a tensor out of the mapping, optionally to a device/dtype."""
        tensor = self.view(key)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        else:
            tensor = tensor.clone()
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def close(self) -> None:
        """Release the mappings. Views handed out earlier are no longer backed by anything."""
        self._views.clear()
        self._maps.clear()

