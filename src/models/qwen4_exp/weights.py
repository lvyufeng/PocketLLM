"""Safetensors access for Qwen4-Exp: mmap reads plus host-resident experts.

The checkpoint is 335 GiB, of which 225 GiB is routed experts and 95 GiB is the
PLE n-gram table.  Neither fits in 4x22 GiB of VRAM, so both live on the host and
the GPU only ever sees the rows a step actually touches.

`MmapSafetensors` maps each shard once and hands out torch views over the raw
bytes.  BF16 has no numpy dtype, so the mapping is `uint16` and reinterpreted
with `Tensor.view(torch.bfloat16)`; `float8_e4m3fn` goes the same way through
`uint8`.

The FP8 build of the same model halves both of those: routed experts drop from
229.7 to 114.9 GiB and the PLE table from 95.4 to 47.7 GiB.  It also changes the
expert *layout* — `experts.{id}.{gate,up,down}_proj.weight` with a
`weight_scale_inv` each, instead of two packed `[512, ...]` tensors — so the
reader assembles the runtime's packed `[2*inter, hidden]` convention itself and
everything downstream sees one shape regardless of which checkpoint is loaded.

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

from src.models.qwen4_exp.quant import FP8Tensor, fp8_scalar_dequantize, pack_gate_up_fp8

_ST_DTYPES: dict[str, tuple[np.dtype, torch.dtype]] = {
    "F64": (np.dtype("<f8"), torch.float64),
    "F32": (np.dtype("<f4"), torch.float32),
    "F16": (np.dtype("<f2"), torch.float16),
    "BF16": (np.dtype("<u2"), torch.bfloat16),
    # Neither bf16 nor fp8 has a numpy dtype, so both map through an unsigned
    # integer of the same width and are reinterpreted by `Tensor.view`.
    "F8_E4M3": (np.dtype("<u1"), torch.float8_e4m3fn),
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
        if torch_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            tensor = tensor.view(torch_dtype)
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

    An FP8 checkpoint adds a second buffer per projection for the block scales,
    so a layer holds four tensors instead of two.  They are still allocated as
    one group: a shard where the codes are pinned and the scales are not would
    stall every staging copy on the unpinned half.
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
        self.block_size: tuple[int, int] | None = None
        self._gate_up: dict[int, torch.Tensor] = {}
        self._down: dict[int, torch.Tensor] = {}
        self._gate_up_scale: dict[int, torch.Tensor] = {}
        self._down_scale: dict[int, torch.Tensor] = {}
        self._loaded: set[int] = set()
        self._pin_fallback_reported = False

    @property
    def is_fp8(self) -> bool:
        return self.block_size is not None

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
            for store in (self._gate_up, self._down, self._gate_up_scale, self._down_scale):
                for layer, tensor in list(store.items()):
                    store[layer] = tensor.clone()
            self.pinned = False
            import gc

            gc.collect()

    def _alloc_group(
        self,
        specs: list[tuple[tuple[int, ...], torch.dtype]],
    ) -> list[torch.Tensor]:
        """All of a layer's buffers, pinned together or pageable together.

        Allocating them one at a time could leave a layer half-pinned if the
        memlock limit is hit partway through, so a failure on any of them
        discards the whole group and the layer is retried as pageable.
        """
        if self.pin_requested:
            allocated: list[torch.Tensor] = []
            try:
                for shape, dtype in specs:
                    allocated.append(torch.empty(shape, dtype=dtype, pin_memory=True))
                self.pinned = True
                return allocated
            except RuntimeError as exc:  # memlock limit, most likely
                # Release the partial group first: holding pinned buffers here
                # would keep locked pages during the clones below.
                allocated.clear()
                self._fall_back_to_pageable(exc)
        return [torch.empty(shape, dtype=dtype) for shape, dtype in specs]

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
        gate_up, down = self._alloc_group(
            [
                ((len(ids), *gate_up_all.shape[1:]), gate_up_all.dtype),
                ((len(ids), *down_all.shape[1:]), down_all.dtype),
            ]
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

    def load_layer_fp8(
        self,
        layer_idx: int,
        num_experts: int,
        expert_reader,
        *,
        block_size: tuple[int, int],
    ) -> int:
        """Copy this rank's FP8 rows of one layer in; returns the bytes copied.

        `expert_reader(expert_id)` yields `(gate_up, down)` as `FP8Tensor`s
        already packed into the runtime's `[2*inter, hidden]` convention.  The
        codes and the scale grids each get one contiguous buffer per projection,
        so a layer is four allocations no matter how many experts it holds.
        """
        if layer_idx in self._loaded:
            return 0
        ids = self.local_expert_ids(num_experts)
        if not ids:
            raise ValueError(
                f"rank {self.rank} of {self.world_size} owns no experts out of {num_experts}"
            )
        first_gate_up, first_down = expert_reader(ids[0])
        buffers = self._alloc_group(
            [
                ((len(ids), *first_gate_up.code.shape), first_gate_up.code.dtype),
                ((len(ids), *first_down.code.shape), first_down.code.dtype),
                ((len(ids), *first_gate_up.scale.shape), first_gate_up.scale.dtype),
                ((len(ids), *first_down.scale.shape), first_down.scale.dtype),
            ]
        )
        gate_up, down, gate_up_scale, down_scale = buffers
        for local, expert_id in enumerate(ids):
            pair = (first_gate_up, first_down) if local == 0 else expert_reader(expert_id)
            gate_up[local].copy_(pair[0].code)
            gate_up_scale[local].copy_(pair[0].scale)
            down[local].copy_(pair[1].code)
            down_scale[local].copy_(pair[1].scale)
        self._gate_up[layer_idx] = gate_up
        self._down[layer_idx] = down
        self._gate_up_scale[layer_idx] = gate_up_scale
        self._down_scale[layer_idx] = down_scale
        self._loaded.add(layer_idx)
        self.num_local_experts = len(ids)
        self.block_size = (int(block_size[0]), int(block_size[1]))
        copied = sum(t.numel() * t.element_size() for t in buffers)
        self.resident_bytes += copied
        return copied

    # -- access -----------------------------------------------------------

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._loaded

    def rows(self, layer_idx: int, expert_id: int):
        """One expert's `(gate_up, down)`; `FP8Tensor`s for an FP8 shard."""
        local = self.local_index(expert_id)
        gate_up = self._gate_up[layer_idx][local]
        down = self._down[layer_idx][local]
        if self.block_size is None:
            return gate_up, down
        return (
            FP8Tensor(gate_up, self._gate_up_scale[layer_idx][local], self.block_size),
            FP8Tensor(down, self._down_scale[layer_idx][local], self.block_size),
        )

    def layer_tensors(self, layer_idx: int):
        """This rank's contiguous local expert tensors, for grouped prefill.

        Returns `FP8Tensor`s for an FP8 shard, whose leading dimension is the
        local expert index rather than a matrix row.
        """
        gate_up = self._gate_up[layer_idx]
        down = self._down[layer_idx]
        if self.block_size is None:
            return gate_up, down
        return (
            FP8Tensor(gate_up, self._gate_up_scale[layer_idx], self.block_size),
            FP8Tensor(down, self._down_scale[layer_idx], self.block_size),
        )

    def stats(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "layers": len(self._loaded),
            "local_experts": self.num_local_experts,
            "resident_bytes": self.resident_bytes,
            "pinned": self.pinned,
            "fp8": self.is_fp8,
        }


class Qwen4ExpCheckpoint:
    """Names and lazily reads Qwen4-Exp tensors from a checkpoint directory."""

    LM_PREFIX = "model.language_model"

    def __init__(
        self,
        root: str,
        *,
        store: MmapSafetensors | None = None,
        fp8: "FP8QuantSpec | None" = None,
    ) -> None:
        self.root = root
        self.store = store if store is not None else MmapSafetensors(root)
        self.expert_shard: HostExpertShard | None = None
        self.fp8 = fp8 if fp8 is not None else self._detect_fp8()

    def _detect_fp8(self) -> "FP8QuantSpec | None":
        """Read `quantization_config` from the checkpoint's own `config.json`.

        Detection is by config, not by tensor naming: the config is what declares
        the block size and the skip list, and a checkpoint that names tensors one
        way but scales them another would be a packaging bug we want to surface
        rather than guess around.
        """
        from src.models.qwen4_exp.config import FP8QuantSpec

        path = os.path.join(self.root, "config.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            raw = json.load(f)
        return FP8QuantSpec.from_dict(raw.get("quantization_config"))

    @property
    def is_fp8(self) -> bool:
        return self.fp8 is not None

    # -- naming -----------------------------------------------------------

    def layer_prefix(self, layer_idx: int) -> str:
        return f"{self.LM_PREFIX}.layers.{layer_idx}"

    def expert_key(self, layer_idx: int, which: str) -> str:
        return f"{self.layer_prefix(layer_idx)}.mlp.experts.{which}"

    def fp8_expert_key(self, layer_idx: int, expert_id: int, proj: str) -> str:
        """Name of one FP8 expert projection (`gate`, `up` or `down`)."""
        return f"{self.expert_key(layer_idx, str(int(expert_id)))}.{proj}_proj.weight"

    # -- routed experts ---------------------------------------------------

    def fp8_expert_rows(
        self, layer_idx: int, expert_id: int
    ) -> tuple[FP8Tensor, FP8Tensor]:
        """One FP8 expert straight from the mapping, packed like the BF16 layout.

        The checkpoint stores `gate_proj` and `up_proj` separately; the runtime
        expects them row-concatenated as `[2*inter, hidden]` with gate first,
        which is the order the BF16 build ships and which `swiglu_expert`'s
        `chunk(2)` assumes.
        """
        if self.fp8 is None:
            raise ValueError("checkpoint is not FP8; use expert_rows instead")
        block = self.fp8.block_size

        def read(proj: str) -> FP8Tensor:
            key = self.fp8_expert_key(layer_idx, expert_id, proj)
            return FP8Tensor(
                self.store.view(key),
                self.store.view(f"{key}_scale_inv"),
                block,
            )

        return pack_gate_up_fp8(read("gate"), read("up")), read("down")

    def fp8_expert_keys(self, layer_idx: int, expert_id: int) -> list[str]:
        """Every mapped tensor name backing one FP8 expert (codes and scales)."""
        keys = []
        for proj in ("gate", "up", "down"):
            key = self.fp8_expert_key(layer_idx, expert_id, proj)
            keys.extend((key, f"{key}_scale_inv"))
        return keys

    def preload_experts(
        self,
        num_layers: int,
        *,
        num_experts: int | None = None,
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

        `num_experts` is only consulted for an FP8 checkpoint, where the expert
        count cannot be read off a packed tensor's leading dimension.
        """
        shard = HostExpertShard(
            num_layers=num_layers,
            rank=rank,
            world_size=world_size,
            pin_memory=pin,
        )
        for layer_idx in range(num_layers):
            if self.fp8 is not None:
                self._preload_layer_fp8(
                    shard,
                    layer_idx,
                    num_experts=num_experts,
                    barrier=barrier,
                    release_mapping=release_mapping,
                )
            else:
                self._preload_layer_packed(
                    shard,
                    layer_idx,
                    barrier=barrier,
                    release_mapping=release_mapping,
                )
            # Drop local aliases before the next layer.  The mmap views remain
            # cached by the store, but no temporary packed/FP8 row assembly is
            # retained by the preload loop.
            import gc

            gc.collect()
            if progress:
                gib = shard.resident_bytes / (1 << 30)
                print(
                    f"[qwen4exp] rank {rank}: experts resident through layer "
                    f"{layer_idx} ({gib:.2f} GiB)",
                    flush=True,
                )
        self.expert_shard = shard
        return shard

    def _preload_layer_packed(
        self,
        shard: HostExpertShard,
        layer_idx: int,
        *,
        barrier,
        release_mapping: bool,
    ) -> None:
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

    def _preload_layer_fp8(
        self,
        shard: HostExpertShard,
        layer_idx: int,
        *,
        num_experts: int | None,
        barrier,
        release_mapping: bool,
    ) -> None:
        assert self.fp8 is not None
        if num_experts is None:
            raise ValueError(
                "preload_experts needs num_experts for an FP8 checkpoint: the count "
                "is not recoverable from a packed tensor's shape"
            )
        shard.load_layer_fp8(
            layer_idx,
            int(num_experts),
            lambda expert_id: self.fp8_expert_rows(layer_idx, expert_id),
            block_size=self.fp8.block_size,
        )
        if barrier is not None:
            barrier()
        if release_mapping:
            # 6 keys per expert rather than 2 per layer, so this is the one place
            # where the FP8 layout costs materially more bookkeeping.
            for expert_id in shard.local_expert_ids(int(num_experts)):
                for key in self.fp8_expert_keys(layer_idx, expert_id):
                    self.store.advise_dontneed(key)

    def expert_rows(self, layer_idx: int, expert_id: int):
        """One expert's `(gate_up, down)` as host tensors (no copy, no device transfer).

        Served from the resident shard when this rank owns the expert and the
        shard is loaded; otherwise from the mapping, which is both the
        `--mmap-experts` diagnostic path and what the tiny test fixtures use.

        Returns `FP8Tensor`s for an FP8 checkpoint; `HostExpertMoE` dequantizes.
        """
        shard = self.expert_shard
        if shard is not None and shard.has_layer(layer_idx) and shard.owns(expert_id):
            return shard.rows(layer_idx, expert_id)
        if self.fp8 is not None:
            return self.fp8_expert_rows(layer_idx, expert_id)
        gate_up = self.store.view(self.expert_key(layer_idx, "gate_up_proj"))[expert_id]
        down = self.store.view(self.expert_key(layer_idx, "down_proj"))[expert_id]
        return gate_up, down

    # -- PLE n-gram table -------------------------------------------------

    def ngram_shard_keys(self, layer_idx: int) -> list[str]:
        prefix = f"{self.layer_prefix(layer_idx)}.ple.ple_embedding.ngram_embedding.shard_"
        keys = [k for k in self.store.keys() if k.startswith(prefix)]
        return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))

    def ngram_scale(self, layer_idx: int) -> torch.Tensor | None:
        """The FP8 n-gram table's single shared scale, or None when unquantized.

        Unlike the experts, the table is quantized per tensor: all 128 shards
        share one scalar, stored once next to them.
        """
        key = f"{self.layer_prefix(layer_idx)}.ple.ple_embedding.ngram_embedding.weight_scale"
        if key not in self.store:
            return None
        return self.store.view(key).reshape(())


class HostNGramTable:
    """Row lookup into the sharded PLE embedding, kept in host RAM.

    Shards are equal-height slices of one logical `(total_rows, head_dim)` table,
    so a row id maps to `(shard, row_in_shard)` by integer division.  Gathering
    on the host keeps 95 GiB off the GPU; only the gathered rows cross PCIe.

    An FP8 table halves that to 47.7 GiB.  Its 128 shards share one scalar scale,
    so the gather is done on the raw codes and the scale is applied once on the
    device, after the (much smaller) gathered rows have crossed PCIe.
    """

    def __init__(
        self,
        shards: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        scale: torch.Tensor | None = None,
    ) -> None:
        assert shards, "PLE table needs at least one shard"
        self.shards = shards
        self.rows_per_shard = shards[0].shape[0]
        self.head_dim = shards[0].shape[1]
        self.device = device
        self.dtype = dtype
        self.total_rows = sum(s.shape[0] for s in shards)
        self.is_fp8 = shards[0].dtype is torch.float8_e4m3fn
        if self.is_fp8 and scale is None:
            raise ValueError("an FP8 n-gram table needs its weight_scale")
        self.scale = scale
        # `index_select` has no float8 CPU kernel, so gather over a byte alias of
        # the same storage and reinterpret afterwards.  This is a view, not a copy.
        self._gather_shards = (
            [s.view(torch.uint8) for s in shards] if self.is_fp8 else shards
        )
        self._gather_dtype = torch.uint8 if self.is_fp8 else shards[0].dtype

    def __call__(self, row_ids: torch.Tensor) -> torch.Tensor:
        """row_ids: (..., ngram_heads) -> (..., ngram_heads, head_dim) on device."""
        flat = row_ids.reshape(-1).to("cpu", dtype=torch.long)
        out = torch.empty(flat.shape[0], self.head_dim, dtype=self._gather_dtype)
        shard_idx = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local_idx = flat - shard_idx * self.rows_per_shard
        for s in shard_idx.unique().tolist():
            picks = (shard_idx == s).nonzero(as_tuple=True)[0]
            if s >= len(self._gather_shards):
                # Row lands in the divisor padding past the real table; upstream
                # leaves those rows at their (zero) init value.
                out[picks] = 0
                continue
            out[picks] = self._gather_shards[s].index_select(0, local_idx[picks])
        if self.is_fp8:
            codes = out.to(self.device, non_blocking=True).view(torch.float8_e4m3fn)
            out = fp8_scalar_dequantize(codes, self.scale.to(self.device), self.dtype)
        else:
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
