"""The released checkpoint's routed experts, resident in host memory, shared by ranks.

The device path for this checkpoint is heterogeneous by necessity and not by taste.
The 47 routed layers hold 47 x 256 experts of 12.75 MiB, which is **149.81 GiB**,
and four RTX 2080 Ti hold 88 GiB between them -- so the experts do not fit on the
device at any tensor-parallel width, and a step that reads them from `/mnt/data3`
pays the disk for every draw. That disk is an SMR array: measured on this host,
`O_DIRECT` over a 13 TiB shard set reads at **213.7 MiB/s**, which is 12 minutes
for one pass over the experts and 12 minutes again on the next run, because the
read leaves the page cache evictable while the bank does not.

So the experts live in one POSIX shared-memory segment that every rank attaches
to, filled once per boot of the host and read by the DMA engines in place. Four
ranks reading the same bytes out of one segment is the whole point: a per-process
allocation would be four copies of 149.81 GiB, and a rank only ever reads its own
quarter anyway.

What is resident, and how it is laid out
----------------------------------------

`MiMoV2ExpertLayout` has already verified the property this module depends on, at
open time and against the shards' own headers: shard `ep{N}` holds experts
`4N..4N+3` of every routed layer, and those four experts are **one contiguous run**
of the file, each expert being six tensors in file order (`down_proj.weight`,
`down_proj.weight_scale`, `gate_proj.*`, `up_proj.*`). Filling a layer is therefore
64 `pread`s of 51 MiB into 64 places in the segment, and the whole checkpoint is
3008 reads with no per-tensor addressing anywhere.

Inside that run the order is the **shard's own**, and it is not numeric: safetensors
sort their keys as strings, so `ep2` stores experts `10, 11, 8, 9` in that order and
expert 8 is the third of the four. A bank that put expert `e` at `e * expert_bytes`
would hand the device path four experts that are each the wrong one -- silently,
since all of them are real experts of the right layer with the right shapes. The
positions are therefore read off the file offsets (`_layout`, which asserts that
each expert sits exactly where its position implies), and `_Layer.positions` is that
map. It is 47 x 256 integers, paid once at open, and it is what makes the fill one
read per (shard, layer) instead of four.

The destination jumps around a shard and the source does not, which is the
direction to prefer: the fill walks **shard-major**, so each file is read front to
back exactly once and the segment is written in 47 scattered places. A disk that
sees one sequential stream per file is a disk that does not seek, and on an SMR
array that is the difference between 12 minutes and an evening.

Why shared memory rather than a per-process allocation
------------------------------------------------------

The TP4 launcher runs four processes on this host, one per card, and all four need
the experts. `expert_bytes` is 12.75 MiB and the arithmetic is unforgiving: four
private copies is 600 GiB, on a host with 1007 GiB of it and 458 GiB already spent
on another model's resident bank.

The fill is deliberately single-process. Four concurrent sequential readers on a
shingled disk only seek, and one reader at 213.7 MiB/s is 149.81 GiB in **12.0
minutes**, paid once per boot of the segment and not once per run: both the
creating process and every attaching rank give up the tracker's handle, so the
segment's lifetime is `/dev/shm`'s and a second run attaches in milliseconds. The
ready file is what says so; `rm -rf /dev/shm/pocketllm_mimo_experts` is how the
149.81 GiB goes back and `open_expert_bank` will refill.

This segment and the other model's are separate files under the same tmpfs, and
they do not fit together: 149.81 + 457.78 GiB is 607 GiB of a tmpfs whose default
size is half of RAM. A host that runs both either raises `size=` on the mount or
frees one; the segment's name carries the checkpoint's directory so the two are
never confused for each other.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import time
from dataclasses import dataclass, field
from multiprocessing import resource_tracker, shared_memory
from typing import Callable

import torch

from src.models.mimo_v2.loader import MimoV2Checkpoint

__all__ = [
    "MimoV2ExpertBank",
    "PinResult",
    "enabled",
    "open_expert_bank",
    "pin_enabled",
    "resident_bytes",
]

# A magic and a version, because a stale segment from an older layout is a plausible thing to find on
# a host that keeps them across restarts, and the failure it would otherwise cause is a kernel reading
# a plausible permutation of the right bytes. The fields are `magic, version, n_layers, n_experts,
# expert_bytes, total_bytes, size, reserved` as native little-endian `Q`s. Everything a rank needs to
# refuse a segment that is not its own is in the first seven, so the mismatch is caught at attach time
# rather than by a kernel reading past the end of the segment.
_HEADER_MAGIC = b"PKTLM2EX"
_HEADER_VERSION = 1
_HEADER_STRUCT = struct.Struct("<8sQQQQQQQ")
_HEADER_BYTES = 4096

_ALIGN = 64

ENABLE_ENV = "POCKETLLM_MIMO_RESIDENT_EXPERTS"
DIR_ENV = "POCKETLLM_MIMO_RESIDENT_EXPERTS_DIR"
PIN_ENV = "POCKETLLM_MIMO_PIN_RESIDENT_EXPERTS"
DEFAULT_DIR = "/dev/shm/pocketllm_mimo_experts"

# What a `pread` is allowed to be. Large enough that one syscall is tens of milliseconds of disk, small
# enough that the kernel's readahead stays ahead of it. A shard's share of one layer is 51 MiB, so this
# only ever splits a run in two on the last read.
_READ_CHUNK = 64 << 20


def enabled() -> bool:
    """Whether a run wants the checkpoint's experts held in host memory."""
    return os.getenv(ENABLE_ENV, "0").lower() in {"1", "true", "yes"}


def pin_enabled() -> bool:
    """Whether a run wants the bank's mapping pinned, so the DMA engine can read it in place.

    On by default, and that is the opposite of `enabled`, deliberately: this is only asked by a process
    that has already decided to read the bank through the device path, and there it is what the reads
    are worth -- see `MimoV2ExpertBank.pin`. `0` leaves the mapping pageable, which is correct and
    slower, because the copy then stages through PyTorch's own pinned ring.
    """
    return os.getenv(PIN_ENV, "1").lower() not in {"0", "false", "no"}


def _root_dir(root_dir: str | None) -> str:
    if root_dir is None:
        root_dir = os.getenv(DIR_ENV) or DEFAULT_DIR
    return os.path.abspath(root_dir)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _align_up(value: int, align: int = _ALIGN) -> int:
    return ((int(value) + align - 1) // align) * align


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _shm_name(checkpoint: MimoV2Checkpoint) -> str:
    """Deterministic in the checkpoint, so every rank computes the same name without a handshake."""
    root = os.path.abspath(checkpoint.root).rstrip(os.sep)
    return f"pocketllm_mimo_experts_{_safe(os.path.basename(root))}"


def _segment_path(name: str) -> str:
    """Where Linux backs a POSIX shared-memory name, for a cheap existence check.

    The ready file is not the segment: a host that set `DIR_ENV` outside `/dev/shm` keeps the marker
    across a reboot while the segment itself is gone, and a rank that trusted the marker alone would
    then fail to attach for the rest of the host's life rather than refill.
    """
    return os.path.join("/dev/shm", name)


@dataclass(frozen=True)
class PinResult:
    """What `cudaHostRegister` answered, and how long it took."""

    rc: int
    seconds: float
    bytes: int

    @property
    def ok(self) -> bool:
        return self.rc == 0


@dataclass(frozen=True)
class _Layer:
    """One routed layer's region of the segment, and where each of its tensors landed."""

    layer_id: int
    base: int
    nbytes: int
    expert_bytes: int
    n_experts: int
    #: `(expert, projection, kind) -> (offset within the expert's own bytes, shape)`.
    slots: dict[tuple[int, str, str], tuple[int, tuple[int, ...]]]
    #: `expert -> its index in the layer's stored order`. Not the expert's id: a shard stores its four
    #: experts in *name* order, so `ep2` holds 10, 11, 8, 9 and expert 8 is not at `8 * expert_bytes`.
    positions: dict[int, int]
    #: `(shard index, source file, source offset, first stored position, expert count)` per fill.
    fills: tuple[tuple[int, str, int, int, int], ...]

    def position(self, expert: int) -> int:
        try:
            return self.positions[int(expert)]
        except KeyError:
            raise KeyError(f"expert {expert} is not in the resident bank's layer {self.layer_id}") from None

    def offset(self, expert: int, proj: str, kind: str) -> int:
        """The byte offset of one tensor inside the segment."""
        slot = self.slots.get((int(expert), proj, kind))
        if slot is None:
            raise KeyError(f"expert {expert}'s {proj}.{kind} is not in the resident bank")
        return self.base + self.position(expert) * self.expert_bytes + slot[0]

    def shape(self, proj: str, kind: str) -> tuple[int, ...]:
        for (_, name, name_kind), (_, shape) in self.slots.items():
            if name == proj and name_kind == kind:
                return shape
        raise KeyError(f"{proj}.{kind} is not a tensor any expert of layer {self.layer_id} has")


@dataclass
class MimoV2ExpertBank:
    """Every routed layer's experts, in one shared segment, read in place by the device path.

    The class owns both halves of the thing: the layout that says where each expert's bytes go, and
    the mapping they go into. It is opened `create=True` exactly once, by whichever process wins the
    fill, and `create=False` by every rank that attaches to the result.
    """

    checkpoint: MimoV2Checkpoint
    shm_name: str
    root_dir: str
    size: int
    n_routed_experts: int
    layers: dict[int, _Layer]
    create: bool
    ready_path: str = field(init=False)
    meta_path: str = field(init=False)
    total_expert_bytes: int = field(init=False)
    _shm: shared_memory.SharedMemory | None = field(default=None, init=False, repr=False)
    _buffer: memoryview | None = field(default=None, init=False, repr=False)
    _tensors: dict[tuple, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _pin: PinResult | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.ready_path = os.path.join(self.root_dir, "bank.ready")
        self.meta_path = os.path.join(self.root_dir, "bank.json")
        self.total_expert_bytes = sum(layer.nbytes for layer in self.layers.values())
        self._open(self.create)

    # -- opening ----------------------------------------------------------------------------

    def _open(self, create: bool) -> None:
        if create:
            os.makedirs(self.root_dir, exist_ok=True)
            self._unlink_stale()
        self._shm = shared_memory.SharedMemory(
            name=self.shm_name, create=create, size=self.size if create else 0
        )
        # Neither side of this is the process that opened it. `multiprocessing` registers every segment
        # it creates with a tracker that unlinks it when the creating process exits, which would make
        # the 12-minute fill cost 12 minutes *per run* and make the ready file a lie -- so both the
        # creator and a rank that only attaches give up the handle. The segment then lives as long as
        # the host's `/dev/shm`, which is what `mark_ready` is claiming.
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        if not create and self._shm.size != self.size:
            raise RuntimeError(
                f"shared bank {self.shm_name} is {self._shm.size} bytes where this checkpoint needs "
                f"{self.size}; it belongs to a different checkpoint or layout"
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
            self.layers[sorted(self.layers)[0]].expert_bytes,
            self.total_expert_bytes,
            self.size,
            sum(len(layer.fills) for layer in self.layers.values()),
        )

    def _validate_header(self) -> None:
        magic, version, n_layers, n_experts, expert_bytes, total, size, _ = self._header()
        if magic != _HEADER_MAGIC:
            raise RuntimeError(
                f"shared bank {self.shm_name} does not carry this module's header; it belongs to "
                f"another model or was written by a much older version"
            )
        if version != _HEADER_VERSION:
            raise RuntimeError(
                f"shared bank {self.shm_name} is layout version {version}, this build writes "
                f"{_HEADER_VERSION}; remove {self.root_dir} and let it refill"
            )
        mine = (
            len(self.layers),
            self.n_routed_experts,
            self.layers[sorted(self.layers)[0]].expert_bytes,
            self.total_expert_bytes,
            self.size,
        )
        theirs = (n_layers, n_experts, expert_bytes, total, size)
        if mine != theirs:
            raise RuntimeError(
                f"shared bank {self.shm_name} holds {theirs} where this checkpoint needs {mine} "
                f"(layers, experts, expert bytes, total bytes, segment bytes)"
            )

    def _write_metadata(self) -> None:
        tmp = f"{self.meta_path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(self._metadata(), handle, indent=2, sort_keys=True)
        os.replace(tmp, self.meta_path)

    def _metadata(self) -> dict:
        return {
            "checkpoint": os.path.abspath(self.checkpoint.root),
            "segment": self.shm_name,
            "size": self.size,
            "payload_bytes": self.payload_bytes,
            "n_routed_experts": self.n_routed_experts,
            "layers": {
                str(layer.layer_id): {
                    "base": layer.base,
                    "nbytes": layer.nbytes,
                    "expert_bytes": layer.expert_bytes,
                    "n_experts": layer.n_experts,
                    "fills": [
                        {
                            "shard": shard,
                            "file": name,
                            "offset": offset,
                            "first_expert": first,
                            "experts": count,
                        }
                        for shard, name, offset, first, count in layer.fills
                    ],
                }
                for layer in sorted(self.layers.values(), key=lambda item: item.layer_id)
            },
        }

    # -- filling ----------------------------------------------------------------------------

    @property
    def payload_bytes(self) -> int:
        """The bytes the bank exists to hold: every routed expert and not one byte more."""
        return self.total_expert_bytes

    def fill(self, progress: Callable[[str], None] | None = None) -> int:
        """One sequential pass over every shard, into the segment.

        `pread` into the shared mapping and not `mmap` of the shard: a mapping would leave the same
        bytes in the page cache under this copy, so 149.81 GiB of bank would cost 149.81 GiB of page
        cache as well. `POSIX_FADV_DONTNEED` over each range as soon as it is copied is the other
        half, and it is given the range rather than the whole file because the whole file would also
        drop the readahead that has already run into the next layer of the same shard.

        Shard-major, so each of the 64 files is read front to back exactly once; the destination
        scatters across the layer regions, which costs nothing. See the module docstring.
        """
        moved = 0
        started = time.perf_counter()
        for shard in range(len(self.checkpoint.layout.files)):
            file_name = self.checkpoint.layout.files[shard]
            fd = os.open(os.path.join(self.checkpoint.root, file_name), os.O_RDONLY)
            try:
                for layer in sorted(self.layers.values(), key=lambda item: item.layer_id):
                    for fill_shard, name, offset, first, count in layer.fills:
                        if fill_shard != shard:
                            continue
                        moved += self._copy_run(fd, name, offset, layer, first, count)
            finally:
                os.close(fd)
            if progress is not None:
                progress(self._progress(f"shard {shard}", moved, started))
        return moved

    def _copy_run(
        self, fd: int, file_name: str, file_offset: int, layer: _Layer, first: int, count: int
    ) -> int:
        """One shard's share of one layer -- `count` experts, one read, one destination.

        `file_offset` is a `data_offsets` value, which is relative to the shard's data section and
        not to the file, so the header's own length is added here rather than assumed to be absent.
        The distinction is worth a sentence because the read that skips it succeeds, returns the
        right byte count, and lands a block of somebody else's tensors in the segment -- at the right
        place, with the right shape, and with no error anywhere.
        """
        nbytes = count * layer.expert_bytes
        bank_offset = layer.base + first * layer.expert_bytes
        source = self.checkpoint.mmap.data_offset(file_name) + file_offset
        os.posix_fadvise(fd, source, nbytes, os.POSIX_FADV_SEQUENTIAL)
        done = 0
        while done < nbytes:
            chunk = min(_READ_CHUNK, nbytes - done)
            view = self._buffer[bank_offset + done : bank_offset + done + chunk]
            got = os.preadv(fd, [view], source + done)
            if got != chunk:
                raise RuntimeError(
                    f"{file_name}: asked for {chunk} bytes at {source + done}, got {got}"
                )
            done += chunk
        os.posix_fadvise(fd, source, nbytes, os.POSIX_FADV_DONTNEED)
        return nbytes

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
        with open(tmp, "w") as handle:
            handle.write("ready\n")
        os.replace(tmp, self.ready_path)

    @staticmethod
    def wait_until_ready(root_dir: str | None, timeout_s: float = 3600.0) -> None:
        path = os.path.join(_root_dir(root_dir), "bank.ready")
        deadline = time.time() + timeout_s
        while not os.path.exists(path):
            if time.time() > deadline:
                raise TimeoutError(f"waited {timeout_s:.0f}s for the resident expert bank at {path}")
            time.sleep(0.2)

    # -- reading ----------------------------------------------------------------------------

    def has_expert(self, layer_id: int, expert: int, proj: str, kind: str) -> bool:
        """Whether this bank holds that slot rather than raising for one it does not."""
        layer = self.layers.get(int(layer_id))
        return layer is not None and (int(expert), proj, kind) in layer.slots

    def tensor(self, layer_id: int, expert: int, proj: str, kind: str) -> torch.Tensor:
        """One expert's packed codes or E8M0 scale as a `uint8` view of the segment.

        `uint8` and not the checkpoint's own dtype: an E8M0 scale has no CPU `copy_`, and every
        consumer of these bytes -- the dequantizer and the device kernel alike -- reads them as bytes.
        """
        key = (int(layer_id), int(expert), proj, kind)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached
        layer = self.layers.get(key[0])
        if layer is None:
            raise KeyError(f"layer {key[0]} is not in the resident bank")
        shape = layer.shape(proj, kind)
        tensor = torch.frombuffer(
            self._buffer,
            dtype=torch.uint8,
            count=_numel(shape),
            offset=layer.offset(expert, proj, kind),
        ).view(shape)
        self._tensors[key] = tensor
        return tensor

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        """One expert's six tensors, keyed `(projection, kind)`, which is how the staging loop wants them."""
        return {
            (proj, kind): self.tensor(layer_id, expert, proj, kind)
            for proj in self.checkpoint.layout.projections
            for kind in self.checkpoint.layout.kinds
        }

    def pin(self) -> PinResult:
        """Register the whole mapping as pinned host memory, so the DMA can read it in place.

        This is what makes a bank tensor a legal source for a `non_blocking` H2D: without it the copy
        still works but PyTorch stages it through its own pinned ring, which is the copy this
        registration exists to delete. One call covers all 149.81 GiB -- the driver's pin walk is
        comparable to the other model's measured 130.1 ms a GiB, so about 20 s -- and it is idempotent,
        so 47 layers sharing the one segment pin it once.

        Failure is not raised. A driver that refuses the registration leaves a bank that is still
        correct and still resident, just read through the staging ring, and a run that aborts at
        startup over 20 s of a possibly-avoidable cost would be the worse trade. The caller reports the
        `rc`; `PinResult.ok` is the test. A *failed* registration is worth reading carefully before
        blaming a limit: a pin walk over a region whose pages are not yet resident faults them in, so on
        a host whose tmpfs is full the call fails after a long walk with a code that looks exactly like
        a `RLIMIT_MEMLOCK` refusal. Check `df -h /dev/shm` before `ulimit -l`. This bank's own pages are
        resident by construction -- `fill` wrote them, or a previous run did.
        """
        if self._pin is not None and self._pin.ok:
            return self._pin
        if self._buffer is None:
            raise RuntimeError("the bank is closed; there is no mapping to pin")
        # `memoryview` is what the segment hands back, and `from_buffer` gives the address the mapping
        # actually has -- which is the address every cached view in `_tensors` also has, so registration
        # moves nothing and invalidates nothing.
        address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
        size = len(self._buffer)
        started = time.perf_counter()
        try:
            libcudart = ctypes.CDLL("libcudart.so")
            code = libcudart.cudaHostRegister(
                ctypes.c_void_p(address), ctypes.c_size_t(size), ctypes.c_uint(0)
            )
        except OSError:
            # No CUDA runtime to register with at all -- a host that built this engine for another
            # backend, or a test. `-1` is not a `cudaHostRegister` code and is not meant to be read as
            # one; it is `ok` being false, which is the only thing a caller acts on.
            code = -1
        self._pin = PinResult(int(code), time.perf_counter() - started, size)
        return self._pin

    @property
    def pin_result(self) -> PinResult | None:
        """What `pin()` did, or `None` if it has not been called."""
        return self._pin

    def pin_if_enabled(self) -> PinResult | None:
        """`pin()` when this run wants the mapping pinned, and nothing when it does not.

        The gate belongs to the bank and not to its callers, so a device path that reads the
        bank can ask once and honour `POCKETLLM_MIMO_PIN_RESIDENT_EXPERTS=0` without knowing
        the variable's name. `None` is "not asked to", which is not the same answer as a
        failed registration -- that comes back as a `PinResult` with `ok` false.
        """
        if not pin_enabled():
            return None
        return self.pin()

    @property
    def resident_bytes(self) -> int:
        """The segment's bytes, header and padding included."""
        return self.size

    def close(self, unlink: bool = False) -> None:
        """Drop this process's handle on the segment, and optionally the segment itself.

        `_open` gives up the tracker's handle on every path, because the segment's lifetime is the
        host's and not this process's, so `unlink=True` is the explicit way to say "this host is done
        with 149.81 GiB". It is not the default, and the default is not a leak: the next run attaches
        instead of paying the 12-minute fill again.
        """
        if self._pin is not None and self._pin.ok and self._buffer is not None:
            try:
                address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
                ctypes.CDLL("libcudart.so").cudaHostUnregister(ctypes.c_void_p(address))
            except Exception:
                # A process on its way out has no work left that needs the region pageable.
                pass
        self._pin = None
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

    def __enter__(self) -> MimoV2ExpertBank:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _layout(checkpoint: MimoV2Checkpoint) -> tuple[dict[int, _Layer], int]:
    """Where every routed layer's experts go, derived from the loader's own runs and checked.

    The property this depends on is `MimoV2ExpertLayout`'s: a shard's share of a layer is one
    contiguous run of the file, with no other layer's or expert's bytes in between. What is checked
    here is that the run covers exactly the shard's four experts, and that each of them sits at the
    offset the run's start plus its stored position implies -- so the fill is one read per (shard,
    layer) and the destination is `position * expert_bytes`, not a per-expert scatter. What is also
    checked is that every expert of every routed layer is the same size, which is what lets one
    `expert_bytes` describe the whole bank.

    The **stored order is the shard's name order and not the numeric one**, which is a trap this
    module exists downstream of: the shard sorts its tensors as strings, so `ep2` holds experts
    `10, 11, 8, 9` and expert 8 is the third of them. Filling by `id * expert_bytes` would put 8's
    bytes where 10's belong and hand a kernel four experts that are each the wrong one -- silently,
    since every one of them is a real expert of the right layer and the right shape.
    """
    layout = checkpoint.layout
    moe_layers = list(layout.moe_layers)
    if not moe_layers:
        raise ValueError("the checkpoint has no routed layers to keep resident")

    layers: dict[int, _Layer] = {}
    cursor = _HEADER_BYTES
    expert_bytes = layout.expert_bytes
    for layer_id in moe_layers:
        runs = layout.layer_runs(layer_id)
        if len(runs) != checkpoint.layout.n_experts:
            raise ValueError(
                f"layer {layer_id} has {len(runs)} expert runs where the config says "
                f"{checkpoint.layout.n_experts}"
            )
        slots: dict[tuple[int, str, str], tuple[int, tuple[int, ...]]] = {}
        for run in runs:
            for begin, end, proj, kind in run.parts:
                shape = layout.shapes[proj] if kind == "weight" else layout.scale_shapes[proj]
                slots[(run.expert, proj, kind)] = (begin - run.begin, shape)

        fills: list[tuple[int, str, int, int, int]] = []
        positions: dict[int, int] = {}
        for shard in range(len(layout.files)):
            owned = [run for run in runs if run.file_name == layout.files[shard]]
            if not owned:
                continue
            expected = checkpoint.experts_of_shard(shard)
            if sorted(run.expert for run in owned) != sorted(expected):
                raise ValueError(
                    f"layer {layer_id}'s experts in {layout.files[shard]} are "
                    f"{sorted(run.expert for run in owned)} where shard {shard} owns {expected}"
                )
            # The shard stores its four experts in *name* order, which for a ten-plus shard is not
            # numeric -- `ep2` holds 10, 11, 8, 9 -- so the stored order is read off the file offsets
            # rather than assumed from the ids. What is asserted is the property the single-read fill
            # needs: the four are one contiguous range with no gap, occupying exactly four experts'
            # worth of bytes.
            stored = sorted(owned, key=lambda run: run.begin)
            span = stored[-1].end - stored[0].begin
            if span != len(stored) * expert_bytes:
                raise ValueError(
                    f"layer {layer_id}'s experts in {layout.files[shard]} span {span} bytes where "
                    f"{len(stored)} experts are {len(stored) * expert_bytes}; the bank fills them with "
                    f"one read"
                )
            for index, run in enumerate(stored):
                if run.begin != stored[0].begin + index * expert_bytes:
                    raise ValueError(
                        f"layer {layer_id} expert {run.expert} is at {run.begin} in "
                        f"{layout.files[shard]}, not at the {stored[0].begin + index * expert_bytes} "
                        f"its position implies; the bank fills the shard's experts as one run"
                    )
                positions[run.expert] = shard * layout.experts_per_shard + index
            fills.append(
                (
                    shard,
                    stored[0].file_name,
                    stored[0].begin,
                    shard * layout.experts_per_shard,
                    len(stored),
                )
            )

        if any(run.nbytes != expert_bytes for run in runs):
            raise ValueError(
                f"layer {layer_id}'s experts are not all {expert_bytes} bytes; the bank indexes an "
                f"expert by `position * expert_bytes`"
            )
        if len(positions) != layout.n_experts:
            raise ValueError(
                f"layer {layer_id} has {len(positions)} experts with a stored position, expected "
                f"{layout.n_experts}"
            )
        nbytes = _align_up(layout.n_experts * expert_bytes)
        layers[layer_id] = _Layer(
            layer_id=layer_id,
            base=cursor,
            nbytes=nbytes,
            expert_bytes=expert_bytes,
            n_experts=layout.n_experts,
            slots=slots,
            positions=positions,
            fills=tuple(fills),
        )
        cursor += nbytes

    return layers, cursor


def open_expert_bank(
    checkpoint: MimoV2Checkpoint,
    *,
    root_dir: str | None = None,
    progress: Callable[[str], None] | None = None,
    force_fill: bool = False,
) -> MimoV2ExpertBank:
    """The bank for this checkpoint, filled if the host does not already have one.

    A segment that is already marked ready is attached and not refilled, and the layout is rebuilt from
    the checkpoint either way rather than read back out of `bank.json`, because the layout is what
    `_validate_header` checks the segment against -- a metadata file that agreed with itself and not
    with the checkpoint would be worse than no file at all.

    The fill is serialized by the ready marker and not by a lock: `os.replace` is atomic, so a rank
    that finds no marker creates the segment, fills it and marks it, and a rank that arrives in the
    middle waits for the marker rather than filling a second copy. Two processes that both find no
    marker both fill, which is wasteful and correct -- the second one's `_unlink_stale` is what makes
    it safe, and the window is the microseconds between the `exists` and the create.
    """
    root = _root_dir(root_dir)
    name = _shm_name(checkpoint)
    layers, size = _layout(checkpoint)
    ready = os.path.join(root, "bank.ready")
    held = os.path.exists(ready) and os.path.exists(_segment_path(name))

    if held and not force_fill:
        return MimoV2ExpertBank(
            checkpoint=checkpoint,
            shm_name=name,
            root_dir=root,
            size=size,
            n_routed_experts=checkpoint.layout.n_experts,
            layers=layers,
            create=False,
        )

    if os.path.exists(ready) and not force_fill:
        # The marker is there and the segment is not: a host that keeps `root_dir` somewhere that
        # survives a reboot has a ready file describing memory that no longer exists.
        MimoV2ExpertBank.wait_until_ready(root, timeout_s=1.0)

    bank = MimoV2ExpertBank(
        checkpoint=checkpoint,
        shm_name=name,
        root_dir=root,
        size=size,
        n_routed_experts=checkpoint.layout.n_experts,
        layers=layers,
        create=True,
    )
    bank.fill(progress=progress)
    bank.mark_ready()
    return bank


def resident_bytes(checkpoint: MimoV2Checkpoint) -> int:
    """The segment this checkpoint's experts would need, without opening anything."""
    _, size = _layout(checkpoint)
    return size
