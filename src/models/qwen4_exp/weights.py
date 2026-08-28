"""Zero-copy safetensors access for Qwen4-Exp.

The checkpoint is 335 GiB, of which 225 GiB is routed experts and 95 GiB is the
PLE n-gram table.  Neither fits in 4x22 GiB of VRAM, and copying them into
process memory would double the footprint, so both are read through `mmap`: the
host page cache (1 TB of RAM here) becomes the expert/embedding store and the
GPU only ever sees the rows a step actually touches.

`MmapSafetensors` maps each shard once and hands out torch views over the raw
bytes.  BF16 has no numpy dtype, so the mapping is `uint16` and reinterpreted
with `Tensor.view(torch.bfloat16)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import struct

import numpy as np
import torch

_ST_DTYPES: dict[str, tuple[np.dtype, torch.dtype]] = {
    "F64": (np.dtype("<f8"), torch.float64),
    "F32": (np.dtype("<f4"), torch.float32),
    "F16": (np.dtype("<f2"), torch.float16),
    "BF16": (np.dtype("<u2"), torch.bfloat16),
    "I64": (np.dtype("<i8"), torch.int64),
    "I32": (np.dtype("<i4"), torch.int32),
    "I16": (np.dtype("<i2"), torch.int16),
    "I8": (np.dtype("<i1"), torch.int8),
    "U8": (np.dtype("<u1"), torch.uint8),
    "BOOL": (np.dtype("?"), torch.bool),
}


@dataclass(frozen=True)
class TensorEntry:
    file_name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class MmapSafetensors:
    """Read-only view over a sharded safetensors checkpoint.

    Tensors are materialized lazily as torch views over the mmap; slicing a view
    only faults in the pages it touches, so `expert_rows` costs one expert's
    worth of I/O rather than a whole layer's.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.entries: dict[str, TensorEntry] = {}
        self._maps: dict[str, np.memmap] = {}
        self._data_offsets: dict[str, int] = {}
        self._views: dict[str, torch.Tensor] = {}
        self._index_files()

    def _index_files(self) -> None:
        index_path = os.path.join(self.root, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        for file_name in files:
            path = os.path.join(self.root, file_name)
            with open(path, "rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(header_len))
            self._data_offsets[file_name] = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
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

    def keys(self):
        return self.entries.keys()

    def view(self, key: str) -> torch.Tensor:
        """Torch tensor aliasing the mapped bytes (no copy, no device transfer)."""
        cached = self._views.get(key)
        if cached is not None:
            return cached
        entry = self.entries[key]
        np_dtype, torch_dtype = _ST_DTYPES[entry.dtype]
        base = self._data_offsets[entry.file_name]
        raw = self._map(entry.file_name)[base + entry.begin : base + entry.end]
        arr = raw.view(np_dtype)
        # The mapping is read-only; torch warns about that but we never write
        # through these views (callers copy before mutating).
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable.*")
            tensor = torch.from_numpy(arr)
        if torch_dtype is torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        tensor = tensor.reshape(entry.shape)
        self._views[key] = tensor
        return tensor

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


class Qwen4ExpCheckpoint:
    """Names and lazily reads Qwen4-Exp tensors from a checkpoint directory."""

    LM_PREFIX = "model.language_model"

    def __init__(self, root: str, *, store: MmapSafetensors | None = None) -> None:
        self.root = root
        self.store = store if store is not None else MmapSafetensors(root)

    # -- naming -----------------------------------------------------------

    def layer_prefix(self, layer_idx: int) -> str:
        return f"{self.LM_PREFIX}.layers.{layer_idx}"

    def expert_key(self, layer_idx: int, which: str) -> str:
        return f"{self.layer_prefix(layer_idx)}.mlp.experts.{which}"

    # -- routed experts ---------------------------------------------------

    def expert_rows(self, layer_idx: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One expert's `(gate_up, down)` as host views (no copy)."""
        gate_up = self.store.view(self.expert_key(layer_idx, "gate_up_proj"))[expert_id]
        down = self.store.view(self.expert_key(layer_idx, "down_proj"))[expert_id]
        return gate_up, down

    # -- PLE n-gram table -------------------------------------------------

    def ngram_shard_keys(self, layer_idx: int) -> list[str]:
        prefix = f"{self.layer_prefix(layer_idx)}.ple.ple_embedding.ngram_embedding.shard_"
        keys = [k for k in self.store.keys() if k.startswith(prefix)]
        return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))


class HostNGramTable:
    """Row lookup into the sharded PLE embedding, kept in host RAM.

    Shards are equal-height slices of one logical `(total_rows, head_dim)` table,
    so a row id maps to `(shard, row_in_shard)` by integer division.  Gathering
    on the host keeps 95 GiB off the GPU; only the gathered rows cross PCIe.
    """

    def __init__(
        self,
        shards: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        assert shards, "PLE table needs at least one shard"
        self.shards = shards
        self.rows_per_shard = shards[0].shape[0]
        self.head_dim = shards[0].shape[1]
        self.device = device
        self.dtype = dtype
        self.total_rows = sum(s.shape[0] for s in shards)

    def __call__(self, row_ids: torch.Tensor) -> torch.Tensor:
        """row_ids: (..., ngram_heads) -> (..., ngram_heads, head_dim) on device."""
        flat = row_ids.reshape(-1).to("cpu", dtype=torch.long)
        out = torch.empty(flat.shape[0], self.head_dim, dtype=self.shards[0].dtype)
        shard_idx = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local_idx = flat - shard_idx * self.rows_per_shard
        for s in shard_idx.unique().tolist():
            picks = (shard_idx == s).nonzero(as_tuple=True)[0]
            if s >= len(self.shards):
                # Row lands in the divisor padding past the real table; upstream
                # leaves those rows at their (zero) init value.
                out[picks] = 0
                continue
            out[picks] = self.shards[s].index_select(0, local_idx[picks])
        out = out.to(self.device, dtype=self.dtype, non_blocking=True)
        return out.reshape(*row_ids.shape, self.head_dim)


class HostEmbedding:
    """Token embedding gathered on the host, moved to device per step."""

    def __init__(self, weight: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        self.weight = weight
        self.device = device
        self.dtype = dtype

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat = token_ids.reshape(-1).to("cpu", dtype=torch.long)
        rows = self.weight.index_select(0, flat)
        return rows.to(self.device, dtype=self.dtype).reshape(*token_ids.shape, self.weight.shape[1])


class DeviceEmbedding:
    """Token embedding resident on the compute device."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(token_ids.to(self.weight.device), self.weight)
