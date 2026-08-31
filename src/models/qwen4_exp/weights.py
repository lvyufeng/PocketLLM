"""Safetensors access for Qwen4-Exp: mmap reads plus host-resident experts.

The checkpoint is 335 GiB, of which 225 GiB is routed experts and 95 GiB is the
PLE n-gram table.  Neither fits in 4x22 GiB of VRAM, so both live on the host and
the GPU only ever sees the rows a step actually touches.

`MmapSafetensors` maps each shard once and hands out torch views over the raw
bytes.  BF16 has no numpy dtype, so the mapping is `uint16` and reinterpreted
with `Tensor.view(torch.bfloat16)`.

Reading experts through the mapping alone is not enough: the mapping is backed by
a mechanical disk here, so a page that is not resident costs a seek.
`HostExpertShard` therefore copies this rank's routed experts into host RAM once
at startup (pinned when the memlock limit allows), and `Qwen4ExpCheckpoint`
serves `expert_rows` from that copy afterwards.  Under TP the shards are disjoint
(`expert_id % world_size == rank`), so the four ranks together hold exactly one
copy of the expert set.
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

    def advise_dontneed(self, key: str) -> int:
        """Drop this process's page-table entries for one tensor's byte range.

        Used after a tensor has been copied into resident host memory: without it
        the process keeps ~56 GiB of mapped expert pages referenced on top of the
        56 GiB copy.  This only drops *this* process's references — the pages stay
        in the shared page cache, so a sibling rank reading interleaved rows in
        the same region is not forced back to the disk.  Returns the bytes advised.
        """
        import mmap as _mmap

        entry = self.entries[key]
        raw = self._map(entry.file_name)
        handle = getattr(raw, "_mmap", None)
        if handle is None or not hasattr(handle, "madvise"):
            return 0
        base = self._data_offsets[entry.file_name]
        page = _mmap.PAGESIZE
        # Align inwards so only pages wholly inside this tensor are dropped.
        start = -(-(base + entry.begin) // page) * page
        end = ((base + entry.end) // page) * page
        if end <= start:
            return 0
        try:
            handle.madvise(_mmap.MADV_DONTNEED, start, end - start)
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


class HostExpertShard:
    """This rank's routed experts, copied out of the mapping into host RAM.

    One pair of contiguous host tensors per layer, indexed by *local* expert id
    (`expert_id // world_size`), so a 48-layer TP4 shard is 96 allocations rather
    than 12,288 expert-row ones.  Pinned memory is requested because the H2D
    staging in `HostExpertMoE` is the consumer, but a 56 GiB pin can exceed
    `ulimit -l`; on failure the shard falls back to pageable memory once and says
    so, rather than dying at load time.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        rank: int = 0,
        world_size: int = 1,
        pin_memory: bool = True,
    ) -> None:
        if world_size < 1 or not (0 <= rank < world_size):
            raise ValueError(f"bad shard placement: rank={rank} world_size={world_size}")
        self.num_layers = int(num_layers)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pin_requested = bool(pin_memory)
        self.pinned = False
        self.resident_bytes = 0
        self.num_local_experts = 0
        self._gate_up: dict[int, torch.Tensor] = {}
        self._down: dict[int, torch.Tensor] = {}
        self._loaded: set[int] = set()
        self._pin_fallback_reported = False

    # -- placement --------------------------------------------------------

    def owns(self, expert_id: int) -> bool:
        return (int(expert_id) % self.world_size) == self.rank

    def local_index(self, expert_id: int) -> int:
        return int(expert_id) // self.world_size

    def local_expert_ids(self, num_experts: int) -> list[int]:
        return [e for e in range(int(num_experts)) if self.owns(e)]

    # -- allocation -------------------------------------------------------

    def _fall_back_to_pageable(self, exc: RuntimeError) -> None:
        """Convert already loaded layers before retrying the failed allocation."""
        self.pin_requested = False
        if not self._pin_fallback_reported:
            print(
                f"[qwen4exp] pinning the expert shard failed ({exc}); "
                "falling back to pageable host memory",
                flush=True,
            )
            self._pin_fallback_reported = True
        if self.pinned:
            # Do not leave a mixed shard behind: pageable clones replace all
            # earlier pinned layers before the current layer is allocated.
            self._gate_up = {layer: tensor.clone() for layer, tensor in self._gate_up.items()}
            self._down = {layer: tensor.clone() for layer, tensor in self._down.items()}
            self.pinned = False
            import gc

            gc.collect()

    def _alloc_pair(
        self,
        gate_up_shape: tuple[int, ...],
        gate_up_dtype: torch.dtype,
        down_shape: tuple[int, ...],
        down_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both of a layer's buffers, pinned together or pageable together.

        Allocating them separately could leave a layer half-pinned if the memlock
        limit is hit between the two, so a failure on either discards the pair.
        """
        if self.pin_requested:
            try:
                gate_up = torch.empty(gate_up_shape, dtype=gate_up_dtype, pin_memory=True)
                down = torch.empty(down_shape, dtype=down_dtype, pin_memory=True)
                self.pinned = True
                return gate_up, down
            except RuntimeError as exc:  # memlock limit, most likely
                # Release a half-allocated pair first: if gate_up pinned and down
                # did not, holding it would keep locked pages during the clones.
                gate_up = down = None
                self._fall_back_to_pageable(exc)
        return (
            torch.empty(gate_up_shape, dtype=gate_up_dtype),
            torch.empty(down_shape, dtype=down_dtype),
        )

    def load_layer(
        self,
        layer_idx: int,
        gate_up_all: torch.Tensor,
        down_all: torch.Tensor,
    ) -> int:
        """Copy this rank's rows of one layer in; returns the bytes copied."""
        if layer_idx in self._loaded:
            return 0
        num_experts = gate_up_all.shape[0]
        if down_all.shape[0] != num_experts:
            raise ValueError(
                f"expert count mismatch in layer {layer_idx}: "
                f"gate_up {gate_up_all.shape[0]} vs down {down_all.shape[0]}"
            )
        ids = self.local_expert_ids(num_experts)
        gate_up, down = self._alloc_pair(
            (len(ids), *gate_up_all.shape[1:]),
            gate_up_all.dtype,
            (len(ids), *down_all.shape[1:]),
            down_all.dtype,
        )
        # Sequential in expert id, which is also sequential in file offset: the
        # backing store is a mechanical disk, so ordering matters more than
        # parallelism here.
        for local, expert_id in enumerate(ids):
            gate_up[local].copy_(gate_up_all[expert_id])
            down[local].copy_(down_all[expert_id])
        self._gate_up[layer_idx] = gate_up
        self._down[layer_idx] = down
        self._loaded.add(layer_idx)
        self.num_local_experts = len(ids)
        copied = gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
        self.resident_bytes += copied
        return copied

    # -- access -----------------------------------------------------------

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._loaded

    def rows(self, layer_idx: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local_index(expert_id)
        return self._gate_up[layer_idx][local], self._down[layer_idx][local]

    def layer_tensors(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this rank's contiguous local expert tensors for grouped prefill."""
        return self._gate_up[layer_idx], self._down[layer_idx]

    def stats(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "layers": len(self._loaded),
            "local_experts": self.num_local_experts,
            "resident_bytes": self.resident_bytes,
            "pinned": self.pinned,
        }


class Qwen4ExpCheckpoint:
    """Names and lazily reads Qwen4-Exp tensors from a checkpoint directory."""

    LM_PREFIX = "model.language_model"

    def __init__(self, root: str, *, store: MmapSafetensors | None = None) -> None:
        self.root = root
        self.store = store if store is not None else MmapSafetensors(root)
        self.expert_shard: HostExpertShard | None = None

    # -- naming -----------------------------------------------------------

    def layer_prefix(self, layer_idx: int) -> str:
        return f"{self.LM_PREFIX}.layers.{layer_idx}"

    def expert_key(self, layer_idx: int, which: str) -> str:
        return f"{self.layer_prefix(layer_idx)}.mlp.experts.{which}"

    # -- routed experts ---------------------------------------------------

    def preload_experts(
        self,
        num_layers: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        pin: bool = True,
        progress: bool = False,
        barrier=None,
        release_mapping: bool = True,
    ) -> HostExpertShard:
        """Copy this rank's routed experts into host RAM and serve them from there.

        Called once at load time.  Afterwards `expert_rows` no longer touches the
        mapping for owned experts, so steady-state staging reads RAM instead of
        risking a disk seek.  Foreign experts are never loaded; under TP they are
        another rank's responsibility and `ShardedMoE` masks them out.

        `barrier` (when given) is called after each layer.  The ranks interleave
        rows within one layer, so keeping them on the same layer keeps the
        mechanical disk's head in one region instead of letting four processes
        seek across the whole 225 GiB expert set independently.

        `release_mapping` drops each layer's mapped pages from this process once
        it has been copied.  Without it a rank ends up holding its 56 GiB copy
        *and* 56 GiB of mapped expert pages (measured RSS 113 GiB), which at four
        ranks is most of the machine's RAM for no benefit.
        """
        shard = HostExpertShard(
            num_layers=num_layers,
            rank=rank,
            world_size=world_size,
            pin_memory=pin,
        )
        for layer_idx in range(num_layers):
            gate_up_key = self.expert_key(layer_idx, "gate_up_proj")
            down_key = self.expert_key(layer_idx, "down_proj")
            shard.load_layer(layer_idx, self.store.view(gate_up_key), self.store.view(down_key))
            if barrier is not None:
                # Siblings finish the same layer before anyone drops its pages, so
                # MADV_DONTNEED never pulls a region a sibling is still reading.
                barrier()
            if release_mapping:
                self.store.advise_dontneed(gate_up_key)
                self.store.advise_dontneed(down_key)
            if progress:
                gib = shard.resident_bytes / (1 << 30)
                print(
                    f"[qwen4exp] rank {rank}: experts resident through layer "
                    f"{layer_idx} ({gib:.2f} GiB)",
                    flush=True,
                )
        self.expert_shard = shard
        return shard

    def expert_rows(self, layer_idx: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One expert's `(gate_up, down)` as host tensors (no copy, no device transfer).

        Served from the resident shard when this rank owns the expert and the
        shard is loaded; otherwise from the mapping, which is both the
        `--mmap-experts` diagnostic path and what the tiny test fixtures use.
        """
        shard = self.expert_shard
        if shard is not None and shard.has_layer(layer_idx) and shard.owns(expert_id):
            return shard.rows(layer_idx, expert_id)
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
