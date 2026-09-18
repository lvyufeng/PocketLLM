"""The released checkpoint's routed experts and Engram tables, resident in host RAM, shared by ranks.

`/mnt/data3` is an SMR disk. `DeviceRoutedExperts` already consumes the checkpoint's packed fp4
without expanding it, so what a step costs on the disk is not arithmetic any more -- but it is still
a read of the shard mapping per expert miss, and the mapping is the slowest resource on this host:
measured with `O_DIRECT` over `model-00003-of-00048.safetensors`, **213.7 MiB/s**, against 14 GiB/s
for the same copy when the pages are already resident. A step stages 40 layers x 6 experts x 17.93
MiB = **4.20 GiB**, which is 0.30 s warm and 1.31 s cold *per row*, and the first token of a run
pays the cold number for every one of its 240 experts. The Engram tables pay the same tax once per
gathered row: 21 to 49 ms for a row whose page is not resident against 0.004 ms for one that is, and
a 512-token prefill gathers 12,288 rows per table. The directive this module exists for is that the
checkpoint should be in host memory and not reached for again.

What is resident, and how it is laid out
----------------------------------------

`/mnt/data3/DeepSeek-V4.1-Flash` stores each layer's 384 routed experts in **one shard**:
`layers.L.*` is in `model-000{03+L}-of-00048.safetensors`, and inside that shard the 2,304 expert
tensors are not interleaved per expert but grouped by kind -- all 1,152 scales (`w1`, `w2`, `w3` for
every expert, 405 MiB) then all 1,152 packed weights (6.33 GiB) -- as two contiguous runs separated
by a 154.53 MiB hole holding that layer's own dense tensors (layer 0: scales at 8015576, hole at
432688856, weights at 594728408, ending at 7389500888).

That is what makes this a *page* of copies rather than a scatter: the bank stores each layer's
expert bytes **in the order the shard stores them**, so filling one layer is two `pread`s of
contiguous file range into contiguous shared memory, 80 reads for the whole checkpoint and no
per-tensor addressing at all. The experts are 40 x 6,885.0 MiB = **268.95 GiB**, which is exactly
the expert bytes and not one byte more -- the dense hole is skipped, because the run boundary is
taken from the shard's own `data_offsets` rather than assumed from a stride.

The two Engram tables are stored the same way and are held here for the same reason: each is in a
shard of its own (`layers.1.engram.embed.*` in `model-00047`, `layers.14.engram.embed.*` in
`model-00048`), with the 91.55 GiB of fp8 codes at `begin = 0` and the 2.86 GiB of E8M0 scales
immediately after them, so a table is two more `pread`s and nothing else in those shards is wanted.
They are 94.42 GiB each, **188.83 GiB** together, and the segment is 491,535,866,896 bytes =
457.78 GiB, which is what `/dev/shm`'s 504 GB tmpfs holds with 11.6 GiB to spare. (That the weight
begins at the data section's first byte is a coincidence of the release -- shard 47 holds nothing
else -- and the run offsets are taken from the index rather than from that, so a release that moved
them would still be read correctly.)

Why shared memory rather than a per-process allocation
------------------------------------------------------

One process driving four cards needs all 384 experts of all 40 layers, and four processes -- the
shape the TP work needs -- need the same bytes four times over. Four copies is 1.8 TiB of a 1 TiB
host, so the bank is one POSIX shared-memory segment that every rank attaches to and reads its own
slice of. That also makes "the disk is read once" a property of the module rather than of the
launcher: the segment is created, filled and marked ready by one process, and every other rank opens
it by name.

Both regions are always in the segment together. A second variant that held only the experts would
need its own name, its own ready marker and its own header, so that a rank which asked for one and
attached to the other gets a byte range that is real and wrong -- and the thing being traded away is
per-process replication of the Engram tables, which is the 756 GiB this exists to prevent.

The fill is deliberately single-process. Four concurrent sequential readers on a shingled disk only
seek; one reader at 213.7 MiB/s is 457.78 GiB in **36.6 minutes**, paid once per boot of the segment
and not once per run: both the creating process and every rank that attaches drop the tracker's
handle, so the segment's lifetime is `/dev/shm`'s rather than the process's, and a second run
attaches in milliseconds. `rm -rf /dev/shm/pocketllm_v41_experts`, or `close(unlink=True)`, is how
the 458 GiB goes back.

Correctness of the layout is asserted rather than assumed
----------------------------------------------------------

The expert order inside a shard is a property of the release and not of this code, so rather than
relying on it, `_layout` checks it: every layer must have exactly `n_experts` experts of each of the
three projections in each of the two kinds, they must sort by file offset into **precisely**
`w1, w2, w3` per expert in expert order, each kind's group must be contiguous, and all of a layer's
expert tensors must come from one shard. A release laid out differently raises here, naming the
layer and the order it found, instead of staging a permutation of the right bytes into a kernel.

What this does and does not buy, measured
-----------------------------------------

It buys the disk read: cold staging becomes resident staging, and the 27.2 s first token that was
99% paging becomes the forward it is. It does **not** remove the `page cache -> pinned` copy that
`DeviceRoutedExperts._stage` makes -- that copy is 0.30 s/step whether its source is the page cache
or this bank, because 14 GiB/s is a memcpy's rate and not a disk's. That prediction is now measured
on the device path with the page cache emptied (`/tmp/fadvise_drop.py`): the same 8-token row goes
from **17.01 s a step and 9.91 s of `_stage`** to **782.9 ms and 242.1 ms** -- 21.7x and 41x --
while warm the pair is 804.5 against 754.0 ms and `_stage` 224.7 against 244.0, which is to say no
difference at all. Deleting *that* needs the bank to be pinned and the H2D to read it directly, which
is a separate measurement with its own risk: the packed fp4 rows are the op's ABI and a per-expert
copy has to replace a per-card one.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Callable, Iterator

import torch

__all__ = [
    "ResidentExpertBank",
    "enabled",
    "open_expert_bank",
    "parse_engram_key",
    "parse_key",
    "resident_bytes",
]

# A magic and a version, because a stale segment from an older layout is a plausible thing to find
# on a host that keeps them across restarts, and the failure it would otherwise cause is a kernel
# reading a plausible permutation of the right bytes.
#
# The fields are `magic, version, n_layers, n_experts, expert_bytes, n_tables, table_bytes, size,
# reserved` as native little-endian `Q`s. Everything a rank needs to refuse a segment that is not
# its own is in the first eight, so the mismatch is caught at attach time rather than by a kernel
# reading far past the end of the segment; `reserved` keeps the struct 64-bit aligned for a field
# added later. The table fields went in at version 2, and a version 1 segment -- which is the same
# bytes with the Engram region missing -- is refused rather than half-read.
_HEADER_MAGIC = b"PKTL41EX"
_HEADER_VERSION = 2
_HEADER_STRUCT = struct.Struct("<8sQQQQQQQQ")
_HEADER_BYTES = 4096

_ALIGN = 64

# The checkpoint's own names, in its own order. `w1` and `w3` are `[inter, dim]` and `w2` is
# `[dim, inter]`; the bank never looks at the shapes, it only has to reproduce the order.
PROJECTIONS = ("w1", "w2", "w3")

# `s` is the E8M0 scale beside a projection and `q` is the packed codes; two kinds, always.
KINDS = ("s", "q")

# The two tensors an Engram table is, in the order the release stores them.
TABLE_LEAVES = ("weight", "scale")

ENABLE_ENV = "DEEPSEEK_V41_RESIDENT_EXPERTS"
DIR_ENV = "DEEPSEEK_V41_RESIDENT_EXPERTS_DIR"
DEFAULT_DIR = "/dev/shm/pocketllm_v41_experts"

# What a `pread` is allowed to be. Large enough that one syscall is tens of milliseconds of disk and
# four sequential readers' worth of seek is avoided, small enough that the kernel's readahead stays
# ahead of it.
_READ_CHUNK = 32 << 20


def enabled() -> bool:
    """Whether a run wants the checkpoint's experts held in host memory."""
    return os.getenv(ENABLE_ENV, "0").lower() in {"1", "true", "yes"}


def _root_dir(root_dir: str | None) -> str:
    if root_dir is None:
        root_dir = os.getenv(DIR_ENV) or DEFAULT_DIR
    return os.path.abspath(root_dir)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _align_up(value: int, align: int = _ALIGN) -> int:
    return ((int(value) + align - 1) // align) * align


@dataclass(frozen=True)
class _Run:
    """One contiguous range of one shard that becomes one contiguous range of the bank."""

    file_name: str
    file_offset: int
    bank_offset: int
    nbytes: int


@dataclass(frozen=True)
class _Layer:
    layer_id: int
    base: int
    nbytes: int
    runs: tuple[_Run, ...]
    # `(expert, which, kind) -> (offset within the layer's own bytes, shape)`.
    slots: dict[tuple[int, str, str], tuple[int, tuple[int, ...]]]


@dataclass(frozen=True)
class _Table:
    """One Engram layer's `embed.weight` and `embed.scale`, as two contiguous runs."""

    layer_id: int
    base: int
    nbytes: int
    runs: tuple[_Run, ...]
    # `'weight'|'scale' -> (offset within the table's own bytes, shape)`.
    slots: dict[str, tuple[int, tuple[int, ...]]]


def _scan(reader) -> dict[int, dict[tuple[int, str, str], object]]:
    """One pass over the checkpoint's index, grouped by layer.

    The index is 96,085 names and reading it is 0.7 s; this is one walk of it rather than 40
    prefix-filtered ones, and it allocates nothing until it finds an expert tensor.
    """
    layers: dict[int, dict[tuple[int, str, str], object]] = {}
    for key, entry in reader.entries.items():
        slot = parse_key(key)
        if slot is None:
            continue
        layer_id, expert, which, kind = slot
        layers.setdefault(layer_id, {})[(expert, which, kind)] = entry
    return layers


def entries_of_kind(entries, kind: str) -> list[tuple[int, str, object]]:
    """One layer's tensors of one kind as `(expert, projection, entry)`."""
    return [
        (expert, which, entry)
        for (expert, which, slot_kind), entry in entries.items()
        if slot_kind == kind
    ]


def parse_key(key: str) -> tuple[int, int, str, str] | None:
    """`layers.L.ffn.experts.E.wN.weight|scale` as `(L, E, 'wN', 'q'|'s')`, else `None`.

    The mirror of `_scan`'s name test, for a caller that has a checkpoint key and wants to know
    whether the bank is where its bytes should come from.
    """
    parts = key.split(".")
    if len(parts) != 7 or parts[0] != "layers" or parts[2] != "ffn" or parts[3] != "experts":
        return None
    if parts[5] not in PROJECTIONS or parts[6] not in ("weight", "scale"):
        return None
    return int(parts[1]), int(parts[4]), parts[5], "q" if parts[6] == "weight" else "s"


def parse_engram_key(key: str) -> tuple[int, str] | None:
    """`layers.L.engram.embed.weight|scale` as `(L, 'weight'|'scale')`, else `None`.

    Five parts and `embed` in the fourth, so an Engram layer's `wkv`/`q_weight`/`k_weight` -- which
    the bank does not hold -- are not mistaken for its table.
    """
    parts = key.split(".")
    if len(parts) != 5 or parts[0] != "layers" or parts[2] != "engram" or parts[3] != "embed":
        return None
    if parts[4] not in TABLE_LEAVES:
        return None
    return int(parts[1]), parts[4]


def _expert_layout(reader) -> tuple[list[_Layer], int]:
    """Every layer's expert bytes and where they go, derived from the shard index and checked.

    Returns the layers in order and the offset just past the region's last byte.

    The order this walks in is the **shard's**, taken from `data_offsets`, and it is not the numeric
    expert order: the release sorts expert names as strings, so layer 0 opens
    `0, 1, 10, 100, 101, ... 109, 11, ...` and `layers.0.ffn.experts.100.*` sits between expert 10's
    and expert 11's. `slots` therefore records where each expert actually landed, and nothing here
    assumes it landed at `expert * per_expert_bytes`. What *is* asserted is the property the fill
    depends on -- each kind is one contiguous range with no gap -- and the property that makes the
    slot map a coherent grouping rather than a coincidence, which is that the three projections
    cycle `w1, w2, w3` across the range. A release that stored `w1` for every expert and then `w2`
    for every expert would still be contiguous and would put expert 5's w1 into expert 0's w3 slot.
    """
    scanned = _scan(reader)
    if not scanned:
        raise ValueError("the checkpoint has no `layers.*.ffn.experts.*` tensors to keep resident")

    layers: list[_Layer] = []
    cursor = _HEADER_BYTES
    for layer_id in sorted(scanned):
        entries = scanned[layer_id]
        placed = sorted({expert for expert, _, _ in entries})
        if placed != list(range(len(placed))):
            first = next(i for i, expert in enumerate(placed) if expert != i)
            raise ValueError(
                f"layer {layer_id}'s experts are not numbered 0..{len(placed) - 1} without gaps: "
                f"position {first} holds expert {placed[first]}"
            )
        n_experts = len(placed)

        runs: list[_Run] = []
        slots: dict[tuple[int, str, str], tuple[int, tuple[int, ...]]] = {}
        local = 0
        for kind in KINDS:
            missing = [
                (expert, which)
                for expert in range(n_experts)
                for which in PROJECTIONS
                if (expert, which, kind) not in entries
            ]
            if missing:
                raise ValueError(
                    f"layer {layer_id} is missing {len(missing)} {kind!r} tensors, first {missing[0]}"
                )
            ordered = sorted(entries_of_kind(entries, kind), key=lambda item: item[2].begin)
            cycle = [which for _, which, _ in ordered]
            expected_cycle = [PROJECTIONS[i % len(PROJECTIONS)] for i in range(len(ordered))]
            if cycle != expected_cycle:
                first = next(i for i, (a, b) in enumerate(zip(cycle, expected_cycle)) if a != b)
                raise ValueError(
                    f"layer {layer_id}'s {kind!r} tensors do not cycle "
                    f"{'/'.join(PROJECTIONS)} in file order: position {first} is {cycle[first]!r} "
                    f"where the bank needs {expected_cycle[first]!r}. The bank reproduces the "
                    f"shard's order, so a release that groups by projection has to be copied "
                    f"tensor by tensor instead."
                )
            files = {entry.file_name for _, _, entry in ordered}
            if len(files) != 1:
                raise ValueError(
                    f"layer {layer_id}'s {kind!r} tensors span {len(files)} shards; the bank reads "
                    f"one contiguous range per kind and needs them in one"
                )
            start, end = ordered[0][2].begin, ordered[-1][2].end
            # `TensorEntry.begin` counts from the shard's data section, not from the file, so a
            # `pread` of this run has to start past the header. Getting this wrong reads a
            # self-consistent tensor out of the wrong place and every shape still checks out.
            base = reader.data_offset(ordered[0][2].file_name)
            walk = start
            for expert, which, entry in ordered:
                if entry.begin != walk:
                    raise ValueError(
                        f"layer {layer_id}'s {kind!r} group has a {entry.begin - walk} byte gap "
                        f"before expert {expert}'s {which}; the bank copies the range in one read"
                    )
                slots[(expert, which, kind)] = (local + entry.begin - start, tuple(entry.shape))
                walk = entry.end
            runs.append(_Run(ordered[0][2].file_name, base + start, cursor + local, end - start))
            local += end - start

        layers.append(
            _Layer(layer_id=layer_id, base=cursor, nbytes=local, runs=tuple(runs), slots=slots)
        )
        cursor = _align_up(cursor + local)
    return layers, cursor


def _table_layout(reader, cursor: int) -> tuple[list[_Table], int]:
    """Every Engram table's bytes and where they go, continuing from the expert region's end.

    Each table is two runs and the only thing asserted is what a single tensor's own `data_offsets`
    already says -- both of its tensors are in one shard and each is contiguous -- so a release that
    stored the scale before the weight is read correctly rather than refused. The layer is picked up
    from the index and not from a list of layer ids, so an Engram layer added to the model is held
    without a change here.
    """
    scanned = _scan_tables(reader)
    tables: list[_Table] = []
    for layer_id in sorted(scanned):
        entries = scanned[layer_id]
        missing = [leaf for leaf in TABLE_LEAVES if leaf not in entries]
        if missing:
            raise ValueError(
                f"engram layer {layer_id} has no {missing[0]}; a table with codes and no scales, or "
                f"the reverse, cannot be dequantized, and holding half of one is not useful"
            )
        files = {entries[leaf].file_name for leaf in TABLE_LEAVES}
        if len(files) != 1:
            raise ValueError(
                f"engram layer {layer_id}'s table spans {len(files)} shards; the bank reads one "
                f"contiguous range per tensor and needs them in one"
            )
        runs: list[_Run] = []
        slots: dict[str, tuple[int, tuple[int, ...]]] = {}
        local = 0
        for leaf in TABLE_LEAVES:
            entry = entries[leaf]
            base = reader.data_offset(entry.file_name)
            runs.append(_Run(entry.file_name, base + entry.begin, cursor + local, entry.nbytes))
            slots[leaf] = (local, tuple(entry.shape))
            local += entry.nbytes
        tables.append(
            _Table(layer_id=layer_id, base=cursor, nbytes=local, runs=tuple(runs), slots=slots)
        )
        cursor = _align_up(cursor + local)
    return tables, cursor


def _scan_tables(reader) -> dict[int, dict[str, object]]:
    """The `layers.L.engram.embed.*` half of the index, grouped by layer."""
    tables: dict[int, dict[str, object]] = {}
    for key, entry in reader.entries.items():
        slot = parse_engram_key(key)
        if slot is None:
            continue
        layer_id, leaf = slot
        tables.setdefault(layer_id, {})[leaf] = entry
    return tables


def _layout(reader) -> tuple[list[_Layer], int, int]:
    """The expert region alone: the layers in order, its size, and the expert count.

    The segment holds the Engram tables after it (`layout`), but the expert region is what the
    layout assertions above are about and what a caller measuring only the experts reasons about.
    """
    layers, cursor = _expert_layout(reader)
    return layers, cursor, n_experts_of(layers)


def layout(reader) -> tuple[list[_Layer], list[_Table], int, int]:
    """Both regions in the order the segment holds them, the segment size, and the expert count."""
    layers, cursor = _expert_layout(reader)
    tables, cursor = _table_layout(reader, cursor)
    return layers, tables, cursor, n_experts_of(layers)


def n_experts_of(layers: list[_Layer]) -> int:
    """How many experts a layer holds, checked to be the same number on every layer."""
    counts = [len({expert for expert, _, _ in layer.slots}) for layer in layers]
    if len(set(counts)) != 1:
        raise ValueError(
            f"the layers do not all hold the same number of experts: "
            f"{ {layer.layer_id: n for layer, n in zip(layers, counts)} }"
        )
    return counts[0]


def resident_bytes(reader) -> int:
    """What the bank would occupy for `reader`, without building it: both regions and the header."""
    return layout(reader)[2]


class ResidentExpertBank:
    """One POSIX shared-memory segment holding every routed expert and both Engram tables.

    Not a `torch.Tensor` and not an `nn.Module`: the bytes belong to no layer's tree, they are read
    through `tensor()` and `table_tensor()` as `uint8` views in exactly the shapes the checkpoint
    stores, and the kernel that consumes them takes them that way. A view is cached per slot -- the
    expert fill is 92,160 slots and a step touches 240 of them, and the Engram tables are two slots
    read every step -- so the cache is small and its lookup is the only per-read work this class
    does.
    """

    def __init__(
        self,
        root_dir: str,
        layers: list[_Layer],
        size: int,
        shm_name: str,
        *,
        create: bool,
        n_routed_experts: int,
        engram: list[_Table] | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.layers = {layer.layer_id: layer for layer in layers}
        self.tables = {table.layer_id: table for table in (engram or ())}
        self.size = int(size)
        # `size` has the header and the per-region padding in it; these two are the payload bytes
        # alone, which is what a residency budget and a disk-read projection are both about.
        self.total_expert_bytes = sum(layer.nbytes for layer in self.layers.values())
        self.total_table_bytes = sum(table.nbytes for table in self.tables.values())
        self.shm_name = shm_name
        self.n_routed_experts = int(n_routed_experts)
        self.ready_path = os.path.join(root_dir, "bank.ready")
        self.meta_path = os.path.join(root_dir, "bank.json")
        self._shm: shared_memory.SharedMemory | None = None
        self._buffer = None
        self._tensors: dict[tuple, torch.Tensor] = {}
        self._open(create=create)

    # -- opening ----------------------------------------------------------------------------

    def _open(self, create: bool) -> None:
        if create:
            os.makedirs(self.root_dir, exist_ok=True)
            self._unlink_stale()
        self._shm = shared_memory.SharedMemory(
            name=self.shm_name, create=create, size=self.size if create else 0
        )
        # Neither side of this is the process that opened it. `multiprocessing` registers every
        # segment it creates with a tracker that unlinks it when the creating process exits, which
        # would make the 36.6 minute fill cost 36.6 minutes *per run* and make the ready file a lie --
        # so both the creator and a rank that only attaches give up the handle. The segment then
        # lives as long as the host's `/dev/shm`, which is what `mark_ready` is claiming.
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        if not create:
            if self._shm.size != self.size:
                raise RuntimeError(
                    f"shared bank {self.shm_name} is {self._shm.size} bytes where this checkpoint "
                    f"needs {self.size}; it belongs to a different checkpoint or layout"
                )
        self._buffer = self._shm.buf
        if create:
            self._write_header()
            self._write_metadata()
        else:
            self._validate_header()

    def _unlink_stale(self) -> None:
        try:
            stale = shared_memory.SharedMemory(name=self.shm_name, create=False)
        except FileNotFoundError:
            return
        stale.close()
        stale.unlink()

    def _header(self) -> tuple:
        return _HEADER_STRUCT.unpack_from(self._buffer, 0)

    def _write_header(self) -> None:
        _HEADER_STRUCT.pack_into(
            self._buffer,
            0,
            _HEADER_MAGIC,
            _HEADER_VERSION,
            len(self.layers),
            self.n_routed_experts,
            self.total_expert_bytes,
            len(self.tables),
            self.total_table_bytes,
            self.size,
            0,
        )

    def _validate_header(self) -> None:
        magic, version, n_layers, n_experts, expert_bytes, n_tables, table_bytes, size, _ = (
            self._header()
        )
        if magic != _HEADER_MAGIC:
            raise RuntimeError(f"shared bank {self.shm_name} has invalid magic {magic!r}")
        if version != _HEADER_VERSION:
            raise RuntimeError(
                f"shared bank {self.shm_name} is version {version}, this code writes "
                f"{_HEADER_VERSION}"
            )
        if (n_layers, n_experts, expert_bytes, n_tables, table_bytes, size) != (
            len(self.layers),
            self.n_routed_experts,
            self.total_expert_bytes,
            len(self.tables),
            self.total_table_bytes,
            self.size,
        ):
            raise RuntimeError(
                f"shared bank {self.shm_name} holds {n_layers} layers of {n_experts} experts and "
                f"{n_tables} tables, {expert_bytes} + {table_bytes} payload bytes in {size}, where "
                f"this checkpoint needs {len(self.layers)}, {self.n_routed_experts}, "
                f"{len(self.tables)}, {self.total_expert_bytes}, {self.total_table_bytes}, "
                f"{self.size}"
            )

    def _metadata(self) -> dict:
        return {
            "version": _HEADER_VERSION,
            "shm_name": self.shm_name,
            "size": self.size,
            "n_layers": len(self.layers),
            "n_routed_experts": self.n_routed_experts,
            "layers": sorted(self.layers),
            "layer_bytes": {str(k): v.nbytes for k, v in sorted(self.layers.items())},
            "runs": sum(len(layer.runs) for layer in self.layers.values())
            + sum(len(table.runs) for table in self.tables.values()),
            "engram_layers": sorted(self.tables),
            "engram_bytes": {str(k): v.nbytes for k, v in sorted(self.tables.items())},
        }

    def _write_metadata(self) -> None:
        tmp = f"{self.meta_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self._metadata(), f, indent=2, sort_keys=True)
        os.replace(tmp, self.meta_path)

    # -- filling ----------------------------------------------------------------------------

    @property
    def payload_bytes(self) -> int:
        """The bytes the bank exists to hold: every routed expert and both Engram tables."""
        return self.total_expert_bytes + self.total_table_bytes

    def fill(self, checkpoint, progress: Callable[[str], None] | None = None) -> int:
        """One sequential pass over every shard that holds experts or a table, into the segment.

        `pread` into the shared mapping and not `mmap` of the shard: a mapping would leave the same
        bytes in the page cache under this copy, so 457.78 GiB of bank would cost 457.78 GiB of page
        cache as well -- the trap `qwen4exp_host_resident_expert_plan` records. `POSIX_FADV_DONTNEED`
        over each range as soon as it is copied is the other half, and it is given the range rather
        than the whole file because the whole file would also drop the readahead that has already run
        into the next run of the same shard.

        The Engram tables come last rather than interleaved by layer, because layer 1's table is in
        shard 47 while its experts are in shard 4: walking the two regions separately leaves the disk
        reading forward, and one pass over it is the whole point.
        """
        moved = 0
        started = time.perf_counter()
        for layer in sorted(self.layers.values(), key=lambda item: item.layer_id):
            moved += self._copy_runs(layer.runs, checkpoint.root)
            if progress is not None:
                progress(self._progress(f"layer {layer.layer_id}", moved, started))
        for table in sorted(self.tables.values(), key=lambda item: item.layer_id):
            moved += self._copy_runs(table.runs, checkpoint.root)
            if progress is not None:
                progress(self._progress(f"engram {table.layer_id}", moved, started))
        return moved

    def _copy_runs(self, runs: tuple[_Run, ...], root: str) -> int:
        """One region's runs, in order, and how many bytes they were."""
        moved = 0
        for run in runs:
            fd = os.open(os.path.join(root, run.file_name), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, run.file_offset, run.nbytes, os.POSIX_FADV_SEQUENTIAL)
                done = 0
                while done < run.nbytes:
                    chunk = min(_READ_CHUNK, run.nbytes - done)
                    view = self._buffer[run.bank_offset + done : run.bank_offset + done + chunk]
                    got = os.preadv(fd, [view], run.file_offset + done)
                    if got != chunk:
                        raise RuntimeError(
                            f"{run.file_name}: asked for {chunk} bytes at "
                            f"{run.file_offset + done}, got {got}"
                        )
                    done += chunk
                os.posix_fadvise(fd, run.file_offset, run.nbytes, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            moved += run.nbytes
        return moved

    def _progress(self, label: str, moved: int, started: float) -> str:
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = moved / elapsed / 2**20
        remaining = (self.payload_bytes - moved) / 2**20 / max(rate, 1e-9)
        return (
            f"resident checkpoint: {label} in, "
            f"{moved / 2**30:.1f}/{self.payload_bytes / 2**30:.1f} GiB at "
            f"{rate:.0f} MiB/s, {remaining / 60:.0f} min left"
        )

    def mark_ready(self) -> None:
        self._write_metadata()
        tmp = f"{self.ready_path}.tmp"
        with open(tmp, "w") as f:
            f.write("ready\n")
        os.replace(tmp, self.ready_path)

    @staticmethod
    def wait_until_ready(root_dir: str, timeout_s: float = 3600.0) -> None:
        path = os.path.join(_root_dir(root_dir), "bank.ready")
        deadline = time.time() + timeout_s
        while not os.path.exists(path):
            if time.time() > deadline:
                raise TimeoutError(f"waited {timeout_s:.0f}s for the resident expert bank at {path}")
            time.sleep(0.2)

    # -- reading ----------------------------------------------------------------------------

    def has_expert(self, layer_id: int, expert: int, which: str, kind: str) -> bool:
        """Whether this bank holds that slot rather than raising for one it does not.

        A bank built over a subset of the checkpoint -- every test's, and one day a rank that holds
        only its own layers -- has to be able to answer "not mine" without the caller reading it out
        of a `KeyError`.
        """
        layer = self.layers.get(int(layer_id))
        return layer is not None and (int(expert), which, kind) in layer.slots

    def has_table(self, layer_id: int, leaf: str) -> bool:
        """Whether this bank holds that Engram tensor; see `has_expert`."""
        table = self.tables.get(int(layer_id))
        return table is not None and leaf in table.slots

    def tensor(self, layer_id: int, expert: int, which: str, kind: str) -> torch.Tensor:
        """One expert's packed codes or E8M0 scale as a `uint8` view of the segment.

        `uint8` and not the checkpoint's own dtype, for the same reason `DeviceRoutedExperts._stage`
        travels this way: a `float8_e8m0fnu` has no CPU `copy_`, and the kernel reads bytes.
        """
        key = (int(layer_id), int(expert), which, kind)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached
        layer = self.layers.get(key[0])
        if layer is None:
            raise KeyError(f"layer {key[0]} is not in the resident bank")
        slot = layer.slots.get((key[1], key[2], key[3]))
        if slot is None:
            raise KeyError(f"expert {key[1]}'s {key[2]}.{key[3]} is not in the resident bank")
        offset, shape = slot
        tensor = torch.frombuffer(
            self._buffer, dtype=torch.uint8, count=_numel(shape), offset=layer.base + offset
        ).view(shape)
        self._tensors[key] = tensor
        return tensor

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        """One expert's six tensors, keyed `(which, kind)`, which is how the staging loop wants them."""
        return {
            (which, kind): self.tensor(layer_id, expert, which, kind)
            for which in PROJECTIONS
            for kind in KINDS
        }

    def table_tensor(self, layer_id: int, leaf: str) -> torch.Tensor:
        """One Engram layer's `weight` or `scale` as a `uint8` view of the segment.

        `uint8` for the same reason `tensor()` is, and so the caller can index the codes one byte
        wide -- the CPU has no float8 `index_select`, so an Engram gather that is to read this
        instead of the mapping has to be able to address it as bytes.
        """
        key = (int(layer_id), leaf)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached
        table = self.tables.get(key[0])
        if table is None:
            raise KeyError(f"engram layer {key[0]} is not in the resident bank")
        slot = table.slots.get(key[1])
        if slot is None:
            raise KeyError(f"engram layer {key[0]}'s embed.{key[1]} is not in the resident bank")
        offset, shape = slot
        tensor = torch.frombuffer(
            self._buffer, dtype=torch.uint8, count=_numel(shape), offset=table.base + offset
        ).view(shape)
        self._tensors[key] = tensor
        return tensor

    def close(self, unlink: bool = False) -> None:
        """Drop this process's handle on the segment, and optionally the segment itself.

        `_open` gives up the tracker's handle on every path, because the segment's lifetime is the
        host's and not this process's, so `unlink=True` is the explicit way to say "this host is done
        with 458 GiB" -- from a test, or from an operator. It is not the default, and the default is
        not a leak: the next run attaches instead of paying the 36.6 minute fill again.

        Not a guarantee that the memory is returned to the kernel this instant: `torch.frombuffer`
        exports the mapping and CPython refuses to release a buffer with exports, so a tensor a caller
        kept alive holds the segment mapped until that tensor is dropped. Both releases are therefore
        attempted and a `BufferError` is not an error -- it means a view is still in flight, and
        dropping it is the caller's job.
        """
        self._tensors.clear()
        if self._buffer is not None:
            try:
                self._buffer.release()
            except BufferError:
                pass
            self._buffer = None
        if self._shm is not None:
            try:
                self._shm.close()
            except BufferError:
                pass
            if unlink:
                # `SharedMemory.unlink` unregisters the name unconditionally, on a tracker whose entry
                # `_open` has already removed -- which the tracker answers with a `KeyError` printed
                # from its own process at shutdown. Putting the name back first is what makes the two
                # cancellations add up; it is also what a name that was never registered needs.
                try:
                    resource_tracker.register(self._shm._name, "shared_memory")
                except Exception:
                    pass
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _shm_name(checkpoint) -> str:
    """Deterministic in the checkpoint, so every rank computes the same name without a handshake."""
    return f"pocketllm_v41_experts_{_safe(os.path.basename(os.path.abspath(checkpoint.root).rstrip(os.sep)))}"


def _segment_path(name: str) -> str:
    """Where Linux backs a POSIX shared-memory name, for a cheap existence check.

    The ready file is not the segment: a host that set `DIR_ENV` to a path outside `/dev/shm` keeps
    the marker across a reboot while the segment itself is gone, and a rank that trusted the marker
    alone would then fail to attach for the rest of the host's life rather than refill.
    """
    return os.path.join("/dev/shm", name)


def open_expert_bank(
    checkpoint,
    *,
    rank: int = 0,
    root_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
    timeout_s: float = 3600.0,
) -> ResidentExpertBank | None:
    """The resident bank for `checkpoint`, filled by rank 0 and attached by everyone else.

    Returns `None` when `DEEPSEEK_V41_RESIDENT_EXPERTS` is unset, which keeps every caller's
    existing behaviour one environment variable away. A segment that is already marked ready is
    attached rather than refilled, so restarting a server is instant and the 36.6 minute pass is
    paid once per host boot rather than once per run -- `ResidentExpertBank` gives up the tracker's
    handle on the segment for exactly this reason. `rm -rf <root_dir>` is how it goes away early.

    Every rank computes the layout independently from the checkpoint's own index -- 0.24 s -- rather
    than taking it from the ready file, because the layout is what `_validate_header` checks the
    segment against and a rank that trusted a file for it would be checking nothing.
    """
    if not enabled():
        return None
    root = _root_dir(root_dir)
    name = _shm_name(checkpoint)
    layers, tables, size, n_experts = layout(checkpoint.reader)
    gi_b = size / 2**30
    runs = sum(len(layer.runs) for layer in layers) + sum(len(table.runs) for table in tables)

    ready = os.path.join(root, "bank.ready")
    held = os.path.exists(ready) and os.path.exists(_segment_path(name))
    if int(rank) != 0 or held:
        if not held:
            if progress is not None:
                progress(f"resident checkpoint: waiting for rank 0 to fill {gi_b:.1f} GiB")
            ResidentExpertBank.wait_until_ready(root, timeout_s)
        bank = ResidentExpertBank(
            root, layers, size, name, create=False, n_routed_experts=n_experts, engram=tables
        )
        if progress is not None:
            progress(f"resident checkpoint: attached {gi_b:.1f} GiB at {name}")
        return bank

    bank = ResidentExpertBank(
        root, layers, size, name, create=True, n_routed_experts=n_experts, engram=tables
    )
    if progress is not None:
        progress(
            f"resident checkpoint: filling {gi_b:.1f} GiB from {checkpoint.root} "
            f"({runs} sequential reads)"
        )
    started = time.perf_counter()
    bank.fill(checkpoint, progress=progress)
    bank.mark_ready()
    if progress is not None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        progress(
            f"resident checkpoint: {gi_b:.1f} GiB resident in {elapsed:.0f} s "
            f"({gi_b / elapsed:.2f} GiB/s)"
        )
    return bank


def iter_slots(checkpoint) -> Iterator[tuple[int, int, str, str, tuple[int, ...]]]:
    """Every slot the bank holds, for a caller that wants to check the layout without building it."""
    layers, _, _ = _layout(checkpoint.reader)
    for layer in layers:
        for (expert, which, kind), (_, shape) in layer.slots.items():
            yield layer.layer_id, expert, which, kind, shape


def iter_table_slots(checkpoint) -> Iterator[tuple[int, str, tuple[int, ...]]]:
    """Every Engram slot the bank holds, in the same spirit as `iter_slots`."""
    tables, _ = _table_layout(checkpoint.reader, 0)
    for table in tables:
        for leaf, (_, shape) in table.slots.items():
            yield table.layer_id, leaf, shape
