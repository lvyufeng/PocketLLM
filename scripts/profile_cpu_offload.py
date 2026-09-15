#!/usr/bin/env python3
"""Measure the ceiling for layer-granularity CPU offloading on this host.

Issue #156 ("CPU Offloading with Prefetch Pipeline") promises four things: a 70B
FP4 checkpoint runnable on a 2080 Ti, decode at >= 5 tok/s, prefetch hiding more
than 80% of the H2D traffic, and token parity.  None of those has been measured
here, and the roadmap arithmetic behind them
(`docs/architecture/pocketllm_roadmap_old_hardware.md`, section 2.1) does not
close on its own terms: it budgets ~100 ms of H2D and ~100 ms of compute per
layer, which for a 70B FP4 layer is wrong in both terms.

Roadmap entry 15 of #151 asks for exactly this before any throughput is
promised:

    Start with real-checkpoint memory and transfer profiles, distinguish cold
    storage from warm host-cache behavior, and establish the overlap ceiling
    before promising throughput.

This script produces those numbers.  It changes no runtime path and is not an
engine feature: it exists so that a go/no-go on layer-granularity offload can be
made from measurements rather than from the plan's arithmetic.

Note the distinction the existing offload paths in this repository rely on.
Every CPU/GPU mixed path here is *expert*-granular -- GLM's active-expert
staging, the Qwen4-Exp host-resident expert shard, the cpp_engine GGUF Q2
staging.  Expert granularity works because MoE routes each token to a handful of
experts, so most expert bytes never move at all.  *Layer* granularity (llama.cpp
`--gpu-layers N`) moves every weight of every offloaded layer on every token.
The two are not comparable budgets, and only the second is what #156 asks for.

Stages
------
`--stage storage`
    Sequential read bandwidth per mount, cold and warm.  Cold means the sampled
    file's own pages were evicted with `posix_fadvise(POSIX_FADV_DONTNEED)`;
    this never drops the whole page cache, which would perturb unrelated work on
    a shared machine.

`--stage h2d`
    Host-to-device bandwidth against copy size, pinned and pageable, one GPU and
    four GPUs concurrently, plus a NUMA-local/remote A/B for pinned buffers.
    Establishes where copies stop being latency-bound, which sets the smallest
    useful double-buffer chunk.

`--stage overlap`
    The core number: copy-only, compute-only, serial and double-buffered overlap
    for a real layer's byte count and a real layer's arithmetic, at decode (1
    row) and prefill (512 rows) shapes.  Reports the fraction of the copy the
    pipeline hides.

`--stage budget`
    Metadata only -- no tensor data is read.  Per-layer byte counts, the dense
    versus routed-expert split, per-rank footprint at TP1/2/4, how many layers
    fit in a 22 GiB card, and the arithmetic token-rate ceiling given the
    bandwidth measured above.

Usage
-----
    python scripts/profile_cpu_offload.py --stage all \
        --json /tmp/cpu_offload_profile.json

See `docs/performance/cpu_offload_profile.md` for the recorded run and
`docs/guides/benchmarking.md` for the reporting rules this follows.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import glob
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GiB = 1 << 30
MiB = 1 << 20

# Layer geometry of the checkpoints present on this host.  Only metadata is
# read at run time; these constants exist so that the overlap stage can be run
# without rescanning a 434 GiB shard set, and each carries its provenance.

# GLM-5.2 UD-Q4_K_M, `blk.N` tensor table of the 11 GGUF shards.  A MoE block
# is 6,070,800,384 B of which 5,838,471,168 B is the `ffn_{gate,up,down}_exps`
# tensors; the three leading dense blocks are 426,576,896 B each.  The 8 active
# experts of a 256-expert block are `expert_used_count = 8`.
GLM52_MOE_LAYER_BYTES = 6_070_800_384
GLM52_MOE_LAYER_ROUTED_EXPERT_BYTES = 5_838_471_168
GLM52_HIDDEN = 6144
GLM52_EXPERT_FFN = 2048
GLM52_EXPERTS = 256
GLM52_ACTIVE_EXPERTS = 8

# Llama-family 70B dense geometry (hidden 8192, FFN 28672, 80 layers, 64 query
# heads over 8 KV heads).  No such checkpoint exists on this host, so these are
# published shapes, not measurements -- #156's acceptance criterion names a 70B
# FP4 model that cannot be run here.  Weights at 4-bit weight-only with FP8
# per-32-element scales cost 0.53125 B/parameter.
LLAMA70B_LAYERS = 80
LLAMA70B_HIDDEN = 8192
LLAMA70B_FFN = 28672
LLAMA70B_KV_HIDDEN = 1024
FP4_BYTES_PER_PARAM = 0.5 + 0.03125


@dataclass(frozen=True)
class LayerSpec:
    """One offloadable layer: what it costs to move, and what it costs to run.

    `layer_bytes` is what has to cross PCIe when the whole layer is offloaded --
    that is the definition of layer granularity, and it is the number that makes
    a MoE block expensive.  `needed_bytes` is what the layer's arithmetic
    actually has to read for the tokens at hand: for a dense layer the two are
    the same, while a 256-expert MoE block computes only its 8 routed experts,
    so offloading it moves every expert to use one thirty-second of them.
    `gemms` are `(K, N)` weight shapes for that arithmetic; the row count varies
    with the phase (1 for decode, 512 for prefill).
    """

    name: str
    source: str
    layer_bytes: int
    needed_bytes: int
    gemms: tuple[tuple[int, int], ...]

    @property
    def move_amplification(self) -> float:
        return self.layer_bytes / self.needed_bytes if self.needed_bytes else 0.0


def _llama70b_gemms() -> tuple[tuple[int, int], ...]:
    """Per-layer dense 70B weights, in (K, N) form.

    Attention is QKV plus output; the FFN is gate, up and down.  Grouped-query
    attention with 8 KV heads over 64 query heads gives the wide Q/O and narrow
    K/V shapes below.
    """
    q = LLAMA70B_HIDDEN * LLAMA70B_HIDDEN
    kv = LLAMA70B_HIDDEN * LLAMA70B_KV_HIDDEN
    ffn = LLAMA70B_HIDDEN * LLAMA70B_FFN
    params = 2 * q + 2 * kv + 2 * ffn + ffn
    return (
        (LLAMA70B_HIDDEN, LLAMA70B_HIDDEN),          # q_proj
        (LLAMA70B_HIDDEN, LLAMA70B_KV_HIDDEN),       # k_proj
        (LLAMA70B_HIDDEN, LLAMA70B_KV_HIDDEN),       # v_proj
        (LLAMA70B_HIDDEN, LLAMA70B_HIDDEN),          # o_proj
        (LLAMA70B_HIDDEN, LLAMA70B_FFN),             # gate_proj
        (LLAMA70B_HIDDEN, LLAMA70B_FFN),             # up_proj
        (LLAMA70B_FFN, LLAMA70B_HIDDEN),             # down_proj
    ), params


def _llama70b_layer_bytes() -> int:
    _, params = _llama70b_gemms()
    return int(params * FP4_BYTES_PER_PARAM)


def _glm52_needed_bytes() -> int:
    """What a GLM-5.2 MoE block has to read for one token, whole-layer aside.

    Everything outside `ffn_*_exps` is dense and always needed; the routed
    experts are needed only for the tokens that route to them, so at a batch
    small enough to keep the routing disjoint the needed fraction is
    `active_experts / experts`.
    """
    routed = GLM52_MOE_LAYER_ROUTED_EXPERT_BYTES
    dense = GLM52_MOE_LAYER_BYTES - routed
    return dense + routed * GLM52_ACTIVE_EXPERTS // GLM52_EXPERTS


def default_layer_specs() -> list[LayerSpec]:
    return [
        LayerSpec(
            name="glm52-q4-moe",
            source="GLM-5.2-UD-Q4_K_M GGUF blk.3..78 tensor table",
            layer_bytes=GLM52_MOE_LAYER_BYTES,
            needed_bytes=_glm52_needed_bytes(),
            gemms=tuple(
                (GLM52_HIDDEN, 2 * GLM52_EXPERT_FFN) if kind != "down"
                else (GLM52_EXPERT_FFN, GLM52_HIDDEN)
                for _expert in range(GLM52_ACTIVE_EXPERTS)
                for kind in ("gate_up", "down")
            ),
        ),
        LayerSpec(
            name="llama70b-fp4-dense",
            source="#156's named model: published Llama-70B geometry at 4-bit "
                   "weight-only (no such checkpoint on this host)",
            layer_bytes=_llama70b_layer_bytes(),
            # A dense layer reads all of its weights, so offloading it moves
            # exactly what the arithmetic needs.
            needed_bytes=_llama70b_layer_bytes(),
            gemms=_llama70b_gemms()[0],
        ),
    ]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _numa_cpus() -> dict[int, list[int]]:
    nodes: dict[int, list[int]] = {}
    for path in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
        node = int(re.search(r"node(\d+)", path).group(1))
        cpus: list[int] = []
        for part in open(path).read().strip().split(","):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-"))
                cpus.extend(range(lo, hi + 1))
            elif part:
                cpus.append(int(part))
        if cpus:
            nodes[node] = cpus
    return nodes


class _affinity:
    """Bind the calling thread to a NUMA node's CPUs for first-touch placement.

    `cudaHostAlloc` places pages according to the calling thread's policy, so an
    affinity-restricted thread is how a pinned buffer ends up on a chosen node
    without `numactl` re-exec.  That is a request, not a result -- the run
    record reports the node the pages actually landed on, read back with
    `_numa_node_of`.
    """

    def __init__(self, cpus: list[int] | None):
        self._cpus = cpus
        self._saved: set[int] | None = None

    def __enter__(self):
        if self._cpus:
            self._saved = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(self._cpus))
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.sched_setaffinity(0, self._saved)
        return False


def _numa_pages_of(addr: int) -> dict[int, int]:
    """Resident page count per NUMA node for the mapping holding `addr`.

    `/proc/self/numa_maps` lists one line per VMA as `addr policy N0=pages
    N1=pages`.  The VMA covering an address is the nearest one starting at or
    below it -- keying off the largest page count instead would report whichever
    big mapping happens to be in the file.  The counts are returned rather than
    a single node so the caller can check that the dominant count matches the
    allocation it just made; a 64 MiB buffer is `N<n>=16384`.
    """
    try:
        lines = open("/proc/self/numa_maps").read().splitlines()
    except OSError:
        return {}
    best: tuple[int, str] | None = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            start = int(parts[0], 16)
        except ValueError:
            continue
        if start <= addr and (best is None or start > best[0]):
            best = (start, line)
    if best is None:
        return {}
    return {int(m.group(1)): int(m.group(2))
            for m in re.finditer(r"\bN(\d+)=(\d+)", best[1])}


def _numa_node_of(addr: int) -> int | None:
    pages = _numa_pages_of(addr)
    return max(pages, key=lambda n: pages[n]) if pages else None


class _NodePinned:
    """An anonymous pinned buffer whose pages are on a chosen NUMA node.

    `pin_memory=True` cannot express this.  `cudaHostAlloc` places the pages
    itself and, on this driver, puts them all on node 0 whichever CPU the
    calling thread is bound to -- measured, and reported by
    `_pin_memory_placement`.  So the pages are placed first with
    `numa_alloc_onnode` and then pinned where they landed with
    `cudaHostRegister`.

    This is the bounded anonymous staging ring the repository allows.  It is
    nothing like registering a checkpoint's file-backed mmap, which
    `cpp_engine/tests/probe_host_register.cpp` rejects: that pins file pages
    and poisons the kernel's dirty-page accounting for the whole machine.
    """

    def __init__(self, ptr: int, size: int, view, pages: dict[int, int]):
        self.ptr = ptr
        self.size = size
        self.view = view
        self.pages = pages
        self.placed_node = max(pages, key=lambda n: pages[n]) if pages else None

    def release(self, torch) -> None:
        try:
            torch.cuda.cudart().cudaHostUnregister(self.ptr)
        finally:
            _libnuma().numa_free(ctypes.c_void_p(self.ptr),
                                 ctypes.c_size_t(self.size))


_libnuma_cache: Any = None


def _libnuma():
    """libnuma via ctypes -- the two calls needed, no bindings to install."""
    global _libnuma_cache
    if _libnuma_cache is None:
        lib = ctypes.CDLL("libnuma.so.1")
        lib.numa_alloc_onnode.restype = ctypes.c_void_p
        lib.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
        lib.numa_free.restype = None
        lib.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libnuma_cache = lib
    return _libnuma_cache


def _pinned_on_node(torch, size: int, node: int) -> _NodePinned | None:
    lib = _libnuma()
    ptr = lib.numa_alloc_onnode(ctypes.c_size_t(size), ctypes.c_int(node))
    if not ptr:
        return None
    ctypes.memset(ctypes.c_void_p(ptr), 1, size)
    pages = _numa_pages_of(ptr)
    if torch.cuda.cudart().cudaHostRegister(ptr, size, 0) != 0:
        lib.numa_free(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
        return None
    arr = (ctypes.c_ubyte * size).from_address(ptr)
    view = torch.frombuffer(arr, dtype=torch.uint8)
    return _NodePinned(ptr, size, view, pages)


def _pin_memory_placement(torch, numa_cpus: dict[int, list[int]]) -> list[dict]:
    """Where `pin_memory=True` lands under each CPU affinity.

    Reported because the answer is not the obvious one: if thread affinity did
    control placement, a staging ring could be steered towards the GPUs that
    read it.  Measured here so the run record can say whether that holds.
    """
    out = []
    size = 64 * MiB
    for node, cpus in sorted(numa_cpus.items()):
        with _affinity(cpus):
            buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            buf.fill_(1)
            pages = _numa_pages_of(buf.data_ptr())
        out.append({
            "thread_node": node,
            "placed_node": max(pages, key=lambda n: pages[n]) if pages else None,
            "pages": {str(k): v for k, v in sorted(pages.items())},
            "bytes": size,
        })
        del buf
    return out



def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# stage: storage
# ---------------------------------------------------------------------------


def _largest_file(root: str, budget: int = 200_000) -> str | None:
    """The largest regular file under `root`, stopping after `budget` entries.

    Sampling the largest file keeps cold reads long enough to leave the
    latency-dominated regime; the traversal is bounded so that a mount holding
    hundreds of thousands of files does not turn into its own benchmark.
    """
    best: tuple[int, str] | None = None
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            seen += 1
            if seen > budget:
                return best[1] if best else None
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            if best is None or st.st_size > best[0]:
                best = (st.st_size, path)
    return best[1] if best else None


def _read_range(path: str, offset: int, nbytes: int, chunk: int = MiB) -> None:
    """Read a byte range, one reused buffer per thread.

    `preadv` into an existing buffer rather than `pread`, which would allocate
    and free a fresh bytes object per chunk and turn the measurement into an
    allocator benchmark.
    """
    fd = os.open(path, os.O_RDONLY)
    buf = bytearray(min(chunk, nbytes))
    view = memoryview(buf)
    try:
        remaining = nbytes
        pos = offset
        while remaining > 0:
            want = min(len(buf), remaining)
            got = os.preadv(fd, [view[:want]], pos)
            if not got:
                break
            pos += got
            remaining -= got
    finally:
        os.close(fd)


def _read_parallel(path: str, offset: int, nbytes: int, threads: int,
                   chunk: int = 8 * MiB) -> float:
    """Wall time for `threads` threads reading disjoint slices concurrently.

    This is the host-side rate at which weights can be pulled out of page cache
    into staging buffers.  It is a copy within RAM, so it bounds the CPU side of
    the pipeline; the disk rate bounds the cold case instead.
    """
    per = nbytes // threads
    if per <= 0:
        threads, per = 1, nbytes
    t0 = time.perf_counter()
    workers = [
        threading.Thread(target=_read_range,
                         args=(path, offset + i * per, per, chunk))
        for i in range(threads)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return time.perf_counter() - t0


def _block_device(target: str) -> dict:
    """The physical device behind a path, with its rotational flag.

    Whether a mount is an SSD is a hardware fact, not something to infer from
    a read rate: a warm page-cache read of a file on a spinning disk is
    indistinguishable from a cold read of the same file on an SSD.  Two things
    are needed to get it right.  The probe is a *file* rather than the mount
    point, because a mount reached through autofs reports the autofs source
    (`systemd-1`) for the directory.  And a partition such as `/dev/sdb1` is
    resolved to its parent disk with `PKNAME`, since `lsblk -d` reports no model
    for a partition.
    """
    try:
        src = subprocess.run(["findmnt", "-no", "SOURCE", "--target", target],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        return {"block_device_error": str(exc)}
    # An autofs target prints more than one line -- the autofs mount (`systemd-1`)
    # and then the real filesystem behind it -- so take the first line that is a
    # device rather than the first line.
    device = next((line.strip() for line in src.stdout.splitlines()
                   if line.strip().startswith("/dev/")), "")
    if not device:
        return {"source": src.stdout.strip().splitlines()[0] if
                src.stdout.strip() else None}

    disk = device
    try:
        parent = subprocess.run(["lsblk", "-no", "PKNAME", device],
                                capture_output=True, text=True, check=True)
        name = parent.stdout.strip().splitlines()[0] if parent.stdout.strip() else ""
        if name:
            disk = f"/dev/{name}"
    except (subprocess.CalledProcessError, IndexError, OSError):
        pass

    try:
        info = subprocess.run(["lsblk", "-dno", "ROTA,MODEL", disk],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return {"source": device, "disk": disk}
    rota, _, model = info.strip().partition(" ")
    return {
        "source": device,
        "disk": disk,
        "rotational": rota.strip() or None,
        "model": model.strip() or None,
        "kind": "rotational" if rota.strip() == "1" else "solid-state",
    }


def _evict_range(path: str, offset: int, nbytes: int) -> None:
    """Drop this file's page cache for one range only.

    `POSIX_FADV_DONTNEED` on a single file's range, never a global
    `/proc/sys/vm/drop_caches`: the machine is shared and a global drop would
    perturb every other process running against these mounts.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def stage_storage(mounts: Iterable[str], sample_bytes: int, threads: int) -> dict:
    out: list[dict] = []
    for mount in mounts:
        if not os.path.isdir(mount):
            out.append({"mount": mount, "status": "not a directory"})
            continue
        entry: dict[str, Any] = {"mount": mount}
        try:
            st = os.statvfs(mount)
            entry["free_gib"] = st.f_bavail * st.f_frsize / GiB
        except OSError as exc:
            out.append({"mount": mount, "status": f"statvfs failed: {exc}"})
            continue
        path = _largest_file(mount)
        if path is None:
            out.append({"mount": mount, "status": "no file found"})
            continue
        # Resolve the device from the sampled file, not the mount point: an
        # autofs mount reports its autofs source for the directory.
        entry.update(_block_device(path))
        size = os.path.getsize(path)
        if size < sample_bytes + MiB:
            out.append({
                "mount": mount,
                "status": f"largest file {path} is {size} B, too small to sample",
            })
            continue
        offset = min(64 * MiB, size // 10)
        nbytes = min(sample_bytes, size - offset)
        entry.update({"file": path, "file_gib": size / GiB, "offset": offset,
                      "sample_bytes": nbytes, "threads": threads})

        _evict_range(path, offset, nbytes)
        t0 = time.perf_counter()
        _read_range(path, offset, nbytes)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        _read_range(path, offset, nbytes)
        warm = time.perf_counter() - t0
        warm_parallel = _read_parallel(path, offset, nbytes, threads)

        entry["cold_seconds"] = cold
        entry["warm_seconds"] = warm
        entry["warm_parallel_seconds"] = warm_parallel
        entry["cold_gib_s"] = nbytes / GiB / cold if cold > 0 else None
        entry["warm_gib_s"] = nbytes / GiB / warm if warm > 0 else None
        entry["warm_parallel_gib_s"] = (
            nbytes / GiB / warm_parallel if warm_parallel > 0 else None
        )
        out.append(entry)
    return {"sample_bytes": sample_bytes, "threads": threads, "mounts": out}


# ---------------------------------------------------------------------------
# stage: h2d
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SystemExit(f"the h2d/overlap stages need torch: {exc}")
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        raise SystemExit("the h2d/overlap stages need CUDA")
    return torch


def _wall(fn: Callable[[], None], iters: int, warmup: int,
          device: str | Sequence[str] | None = None) -> float:
    """Wall seconds per iteration of `fn`, with the device(s) drained.

    `torch.cuda.synchronize()` without an argument drains only the *current*
    device.  Timing a copy on `cuda:3` that way returns in microseconds, because
    nothing waits for the copy that was just enqueued -- the loop finishes as
    soon as the kernels are enqueued, and the rate comes out four orders of
    magnitude too high.
    """
    torch = _require_torch()
    targets = device if isinstance(device, (list, tuple)) else (device,)

    def sync() -> None:
        for d in targets:
            torch.cuda.synchronize(d)

    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def _sweep_plan(lo: int, hi: int) -> list[int]:
    sizes = []
    n = lo
    while n <= hi:
        sizes.append(n)
        n *= 4
    return sizes


def _copy_traffic_iters(size: int, target_bytes: int, lo: int, hi: int) -> int:
    """Iterations that approximate `target_bytes` of traffic at this size.

    The floor keeps the smallest copies measurable at all -- 400 iterations of
    4 KiB is 1.6 MiB, which is mostly timer noise -- and the ceiling keeps the
    latency-dominated points from running for minutes.
    """
    return max(lo, min(hi, max(1, target_bytes // max(size, 1))))


def stage_h2d(args: argparse.Namespace) -> dict:
    torch = _require_torch()
    devices = [f"cuda:{i}" for i in range(min(args.gpus, torch.cuda.device_count()))]
    if not devices:
        raise SystemExit("no CUDA device visible")

    sweep: list[dict] = []
    for size in _sweep_plan(args.copy_min_bytes, args.copy_max_bytes):
        iters = _copy_traffic_iters(size, args.copy_target_bytes, 5, 20000)
        row: dict[str, Any] = {"bytes": size, "iters": iters}
        for label, pinned in (("pinned", True), ("pageable", False)):
            host = torch.empty(size, dtype=torch.uint8, pin_memory=pinned)
            # Touch both buffers.  A never-written anonymous page reads back as
            # the shared zero page, so an untouched pageable source would make
            # the read side of the copy free and flatter the pageable curve.
            host.fill_(1)
            dev = torch.empty(size, dtype=torch.uint8, device=devices[0])
            secs = _wall(lambda: dev.copy_(host, non_blocking=True), iters, 2,
                         devices[0])
            row[f"{label}_seconds"] = secs
            row[f"{label}_gib_s"] = size / GiB / secs if secs > 0 else None
            del host, dev
        sweep.append(row)
        torch.cuda.empty_cache()

    concurrent: dict[str, Any] = {}
    if len(devices) > 1:
        size = args.concurrent_bytes
        iters = _copy_traffic_iters(size, args.copy_target_bytes, 5, 100)
        for label, pinned in (("pinned", True), ("pageable", False)):
            hosts = [torch.empty(size, dtype=torch.uint8, pin_memory=pinned)
                     for _ in devices]
            devs = [torch.empty(size, dtype=torch.uint8, device=d) for d in devices]
            for h in hosts:
                h.fill_(1)

            def _run():
                threads = []
                for h, d in zip(hosts, devs):
                    t = threading.Thread(target=lambda h=h, d=d: d.copy_(h, non_blocking=True))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()

            secs = _wall(_run, iters, 1, devices)
            concurrent[f"{label}_seconds_per_round"] = secs
            concurrent[f"{label}_aggregate_gib_s"] = len(devices) * size / GiB / secs
            concurrent[f"{label}_per_gpu_gib_s"] = size / GiB / secs
            del hosts, devs
            torch.cuda.empty_cache()
        concurrent["devices"] = devices
        concurrent["bytes"] = size

    numa_cpus = _numa_cpus()
    numa: list[dict] = []
    placement: list[dict] = []
    if len(numa_cpus) > 1 and args.numa_bytes > 0:
        size = args.numa_bytes
        iters = _copy_traffic_iters(size, args.copy_target_bytes, 5, 200)
        targets = [devices[0]] + ([devices[-1]] if len(devices) > 1 else [])
        placement = _pin_memory_placement(torch, numa_cpus)
        for node in sorted(numa_cpus):
            buf = _pinned_on_node(torch, size, node)
            if buf is None:
                numa.append({"node": node, "bytes": size,
                             "error": "numa_alloc_onnode failed"})
                continue
            try:
                for target in targets:
                    dev = torch.empty(size, dtype=torch.uint8, device=target)
                    secs = _wall(
                        lambda: dev.copy_(buf.view, non_blocking=True), iters, 2,
                        target)
                    numa.append({
                        "node": node,
                        "placed_node": buf.placed_node,
                        "placed_pages": {str(k): v
                                         for k, v in sorted(buf.pages.items())},
                        "target": target,
                        "bytes": size,
                        "iters": iters,
                        "gib_s": size / GiB / secs if secs > 0 else None,
                    })
                    del dev
            finally:
                buf.release(torch)
        torch.cuda.empty_cache()

    return {
        "devices": devices,
        "topology": _topology(),
        "sweep": sweep,
        "concurrent": concurrent,
        "numa_placement": placement,
        "numa": numa,
    }


def _topology() -> list[dict]:
    """Record GPU indices, PCIe bus ids and clocks, as a text table.

    The bus id is what says whether a pair is PCIe (PHB) or NVLink (NV2) and
    which NUMA node a card sits on; a bandwidth number without it is not
    comparable across machines.
    """
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,clocks.sm,clocks.mem",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
    except Exception as exc:  # pragma: no cover - depends on the host
        return [{"error": str(exc)}]
    return [{"line": line.strip()} for line in res.stdout.strip().splitlines()]


# ---------------------------------------------------------------------------
# stage: overlap
# ---------------------------------------------------------------------------


def _make_weights(torch, spec: LayerSpec, device: str, dtype) -> list:
    """Device-resident weights for one layer's arithmetic.

    Allocated at the layer's real shapes in `dtype`.  They are *not* the bytes
    being copied -- the copy target is a separate buffer sized to the layer's
    on-disk footprint -- because the two streams are timed independently and
    what matters is how long each stream's work takes.
    """
    return [torch.empty((k, n), dtype=dtype, device=device) for k, n in spec.gemms]


def _make_inputs(torch, spec: LayerSpec, rows: int, dtype, device: str) -> dict:
    """One activation per distinct input width used by this layer.

    The GEMMs of a real layer chain -- attention output feeds the FFN, and a
    MoE expert's `down_proj` consumes that expert's `gate_up` output.  Chaining
    them here would need the intermediate layouts (a 4096-wide SwiGLU result is
    not a 6144-wide activation), so each GEMM instead reads an activation of
    its own declared width.  The multiply-accumulate count is unchanged, which
    is what the compute term is made of; what is lost is the data dependency
    between one kernel and the next, which is the same for either choice here.
    """
    widths = sorted({k for k, _ in spec.gemms})
    return {k: torch.randn((rows, k), dtype=dtype, device=device) * 0.02
            for k in widths}


def _run_layer(torch, spec: LayerSpec, weights: list, xs: dict, out) -> None:
    """Run one layer's GEMMs, writing into a fixed output buffer.

    `out` is cleared first and every GEMM lands inside it, so no part of the
    arithmetic can be elided as dead.
    """
    out.zero_()
    for (k, n), w in zip(spec.gemms, weights):
        out.narrow(1, 0, n).add_(xs[k] @ w)


def _layer_output_shape(spec: LayerSpec, rows: int) -> tuple[int, int]:
    return (rows, max(n for _, n in spec.gemms))


def measure_overlap(torch, spec: LayerSpec, rows: int, layers: int,
                    device: str, dtype, copy_buffer_bytes: int) -> dict:
    weights = _make_weights(torch, spec, device, dtype)
    xs = _make_inputs(torch, spec, rows, dtype, device)
    scratch = torch.zeros(_layer_output_shape(spec, rows), dtype=dtype,
                          device=device)

    src = torch.empty(copy_buffer_bytes, dtype=torch.uint8, pin_memory=True)
    src.fill_(1)
    dst = [torch.empty(copy_buffer_bytes, dtype=torch.uint8, device=device)
           for _ in range(2)]

    def copy_only():
        for _ in range(layers):
            dst[0].copy_(src, non_blocking=True)

    def compute_only():
        for _ in range(layers):
            _run_layer(torch, spec, weights, xs, scratch)

    def serial():
        for _ in range(layers):
            dst[0].copy_(src, non_blocking=True)
            _run_layer(torch, spec, weights, xs, scratch)

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    def overlap():
        copy_ev: list[Any] = [torch.cuda.Event() for _ in range(layers)]
        comp_ev: list[Any] = [torch.cuda.Event() for _ in range(layers)]
        with torch.cuda.stream(copy_stream):
            dst[0].copy_(src, non_blocking=True)
            copy_ev[0].record(copy_stream)
        for i in range(layers):
            if i + 1 < layers:
                with torch.cuda.stream(copy_stream):
                    # Do not overwrite the buffer the previous GEMM is still
                    # reading.  Without this the pipeline would be measuring a
                    # race, not an overlap.
                    if i >= 1:
                        copy_stream.wait_event(comp_ev[i - 1])
                    dst[(i + 1) % 2].copy_(src, non_blocking=True)
                    copy_ev[i + 1].record(copy_stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(copy_ev[i])
                _run_layer(torch, spec, weights, xs, scratch)
                comp_ev[i].record(compute_stream)
        torch.cuda.synchronize(device)

    timings = {}
    for label, fn in (("copy_only", copy_only), ("compute_only", compute_only),
                      ("serial", serial), ("overlap", overlap)):
        secs = _wall(fn, 1, 1, device)
        timings[label] = {
            "seconds": secs,
            "per_layer_ms": secs * 1000.0 / layers,
        }

    copy_s = timings["copy_only"]["seconds"]
    serial_s = timings["serial"]["seconds"]
    overlap_s = timings["overlap"]["seconds"]
    hidden = (serial_s - overlap_s) / copy_s if copy_s > 0 else 0.0
    timings["hidden_fraction"] = hidden
    timings["rows"] = rows
    timings["layers"] = layers
    timings["copy_buffer_bytes"] = copy_buffer_bytes
    timings["copy_gib_s"] = (
        layers * copy_buffer_bytes / GiB / copy_s if copy_s > 0 else None
    )

    del weights, xs, scratch, src, dst
    torch.cuda.empty_cache()
    return timings


def stage_overlap(args: argparse.Namespace) -> dict:
    torch = _require_torch()
    device = f"cuda:{args.gpu}"
    dtype = torch.float16
    specs = default_layer_specs()
    if args.overlap_spec:
        wanted = set(args.overlap_spec.split(","))
        specs = [s for s in specs if s.name in wanted]

    out = []
    for spec in specs:
        rows_out: dict[str, Any] = {}
        for rows in args.overlap_rows:
            rows_out[str(rows)] = measure_overlap(
                torch, spec, rows, args.overlap_layers, device, dtype,
                spec.layer_bytes,
            )
            print(f"  {spec.name} rows={rows}: "
                  f"copy {rows_out[str(rows)]['copy_only']['per_layer_ms']:.2f} ms/layer, "
                  f"compute {rows_out[str(rows)]['compute_only']['per_layer_ms']:.2f} ms/layer, "
                  f"hidden {rows_out[str(rows)]['hidden_fraction'] * 100:.1f}%")
        out.append({
            "name": spec.name,
            "source": spec.source,
            "layer_bytes": spec.layer_bytes,
            "layer_gib": spec.layer_bytes / GiB,
            "needed_bytes": spec.needed_bytes,
            "move_amplification": spec.move_amplification,
            "gemms": list(spec.gemms),
            "rows": rows_out,
        })
    return {"device": device, "dtype": str(dtype), "specs": out}


# ---------------------------------------------------------------------------
# stage: budget
# ---------------------------------------------------------------------------


@dataclass
class CheckpointFacts:
    name: str
    path: str
    kind: str
    n_layers: int = 0
    hidden: int | None = None
    experts: int | None = None
    top_k: int | None = None
    total_bytes: int = 0
    layer_bytes: list[int] = field(default_factory=list)
    routed_expert_bytes: int = 0
    host_only_bytes: int = 0
    notes: list[str] = field(default_factory=list)


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_BLK_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")
_ROUTED_RE = re.compile(r"(?:^|\.)experts\.|_exps\.")
_SHARED_RE = re.compile(r"shared_expert|ffn_.*_shexp|_shexp\.")
# Tables that live in a layer's namespace but are not that layer's weights.
_HOST_ONLY_RE = re.compile(r"\.ple\.")


def _is_routed(name: str) -> bool:
    if _SHARED_RE.search(name):
        return False
    return bool(_ROUTED_RE.search(name))


def _scan_safetensors(path: str) -> tuple[dict[str, int], int]:
    """Read only the safetensors headers of a sharded checkpoint.

    Each shard starts with an 8-byte little-endian header length and a JSON
    tensor table; the tensor data is never touched.  A 131-shard, 335 GiB
    checkpoint costs ~7 ms.
    """
    sizes: dict[str, int] = {}
    total = 0
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(shard, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                continue
            header_len = struct.unpack("<Q", raw)[0]
            header = json.loads(fh.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            sizes[name] = end - start
            total += end - start
    return sizes, total


def _read_config(path: str) -> dict:
    """Flatten the checkpoint's config, including a nested `text_config`.

    Qwen4-Exp keeps the language-model geometry under `text_config` and leaves
    the top level to multimodal keys, so reading only the top level silently
    reports no hidden size and no expert count.
    """
    for candidate in ("config.json", "configuration.json"):
        p = os.path.join(path, candidate)
        if not os.path.exists(p):
            continue
        try:
            cfg = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cfg, dict) or not cfg:
            continue
        text = cfg.get("text_config")
        if isinstance(text, dict):
            merged = dict(cfg)
            merged.update(text)
            return merged
        return cfg
    return {}


def scan_checkpoint(name: str, path: str) -> CheckpointFacts:
    if not os.path.isdir(path):
        return CheckpointFacts(name=name, path=path, kind="missing",
                               notes=[f"{path} does not exist on this host"])
    ggufs = sorted(glob.glob(os.path.join(path, "*.gguf")))
    if ggufs:
        return _scan_gguf(name, path, ggufs)
    return _scan_st(name, path)


def _scan_gguf(name: str, path: str, shards: list[str]) -> CheckpointFacts:
    from src.loader.gguf.reader import GGUFReader

    facts = CheckpointFacts(name=name, path=path, kind="gguf")
    sizes: dict[str, int] = {}
    metadata: dict[str, Any] = {}
    for shard in shards:
        gf = GGUFReader(shard).read()
        metadata.update(gf.metadata)
        for ti in gf.tensors:
            sizes[ti.name] = ti.nbytes or 0

    arch = metadata.get("general.architecture")
    if arch:
        facts.hidden = metadata.get(f"{arch}.embedding_length")
        facts.experts = metadata.get(f"{arch}.expert_count")
        facts.top_k = metadata.get(f"{arch}.expert_used_count")
        facts.n_layers = metadata.get(f"{arch}.block_count", 0)
    facts.total_bytes = sum(sizes.values())
    facts.notes.append(f"{len(shards)} GGUF shards, no tensor data read")

    per_layer: dict[int, int] = collections.defaultdict(int)
    routed = 0
    for tname, nbytes in sizes.items():
        m = _BLK_RE.search(tname)
        if not m:
            continue
        per_layer[int(m.group(1))] += nbytes
        if _is_routed(tname):
            routed += nbytes
    facts.layer_bytes = [per_layer[i] for i in sorted(per_layer)]
    facts.routed_expert_bytes = routed
    if not facts.n_layers:
        facts.n_layers = len(facts.layer_bytes)
    return facts


def _scan_st(name: str, path: str) -> CheckpointFacts:
    facts = CheckpointFacts(name=name, path=path, kind="safetensors")
    sizes, total = _scan_safetensors(path)
    cfg = _read_config(path)
    facts.total_bytes = total
    facts.hidden = cfg.get("hidden_size")
    facts.n_layers = cfg.get("num_hidden_layers", 0)
    facts.experts = cfg.get("num_experts") or cfg.get("num_local_experts")
    facts.top_k = cfg.get("num_experts_per_tok")
    facts.notes.append(f"{len(sizes)} tensors, headers only, no tensor data read")

    per_layer: dict[int, int] = collections.defaultdict(int)
    routed = 0
    host_only = 0
    for tname, nbytes in sizes.items():
        m = _LAYER_RE.search(tname)
        if not m:
            continue
        # `mtp.layers.N` is the multi-token-prediction head, not a trunk layer.
        if re.search(r"(?:^|\.)mtp\.", tname):
            continue
        # Qwen4-Exp's per-layer embedding table is registered under
        # `layers.1.ple.*` but is a 95 GiB host-side vocabulary table that
        # every layer reads, not 95 GiB of layer-1 weights.  Counting it as a
        # layer would make layer-granularity placement look absurd for the one
        # layer it happens to hang off.
        if _HOST_ONLY_RE.search(tname):
            host_only += nbytes
            continue
        per_layer[int(m.group(1))] += nbytes
        if _is_routed(tname):
            routed += nbytes
    facts.layer_bytes = [per_layer[i] for i in sorted(per_layer)]
    if not facts.n_layers:
        facts.n_layers = len(facts.layer_bytes)
    facts.routed_expert_bytes = routed
    facts.host_only_bytes = host_only
    return facts


def _synthetic_70b() -> CheckpointFacts:
    gemms, params = _llama70b_gemms()
    per_layer = int(params * FP4_BYTES_PER_PARAM)
    facts = CheckpointFacts(
        name="llama70b-fp4 (arithmetic model)",
        path="<none on this host>",
        kind="synthetic",
        n_layers=LLAMA70B_LAYERS,
        hidden=LLAMA70B_HIDDEN,
        experts=None,
        top_k=None,
    )
    facts.layer_bytes = [per_layer] * LLAMA70B_LAYERS
    facts.total_bytes = per_layer * LLAMA70B_LAYERS
    facts.notes.append(
        "#156 names a 70B FP4 checkpoint; no such checkpoint exists on this "
        "host, so this entry is published Llama-70B geometry at 4-bit "
        "weight-only, not a measurement"
    )
    return facts


def _ceiling(layer_bytes: int, offloaded_layers: int, bandwidth_gib_s: float,
             compute_ms_per_layer: float) -> dict:
    """Bound the token rate of a layer-granularity pipeline two ways.

    `serial` moves the layer then computes it.  `perfect_overlap` is the most
    optimistic case #156 can ask for: the transfer of layer i+1 is entirely
    hidden behind the compute of layer i, so the token rate is set by the
    slower of the two streams rather than by their sum.  No real pipeline
    beats this, so a target above it cannot be reached by prefetching harder.
    """
    transfer_bytes = layer_bytes * offloaded_layers
    transfer_ms = transfer_bytes / GiB / bandwidth_gib_s * 1000.0
    compute_total_ms = compute_ms_per_layer * offloaded_layers
    serial_ms = transfer_ms + compute_total_ms
    overlap_ms = max(transfer_ms, compute_total_ms)
    return {
        "offloaded_layers": offloaded_layers,
        "transfer_bytes": transfer_bytes,
        "transfer_gib_per_token": transfer_bytes / GiB,
        "transfer_ms_per_token": transfer_ms,
        "compute_ms_per_token": compute_total_ms,
        "serial_ms_per_token": serial_ms,
        "perfect_overlap_ms_per_token": overlap_ms,
        "serial_tok_per_s": 1000.0 / serial_ms if serial_ms > 0 else None,
        "perfect_overlap_tok_per_s": 1000.0 / overlap_ms if overlap_ms > 0 else None,
        # The compute term is a borrowed proxy (see `compute_source` in the
        # run's assumptions), so the ceiling is only trustworthy while the
        # transfer dominates.  This is that check, per checkpoint.
        "compute_fraction_of_serial": (
            compute_total_ms / serial_ms if serial_ms > 0 else None
        ),
    }


def stage_budget(args: argparse.Namespace, bandwidth_gib_s: float,
                 compute_ms_per_layer: float, bandwidth_source: str,
                 compute_source: str) -> dict:
    checkpoints = list(args.checkpoint or [
        "GLM-5.2-GGUF-UD-Q4_K_M=/mnt/data3/GLM-5.2-GGUF/UD-Q4_K_M",
        "Qwen3.8-Flash-Next=/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next",
        "Qwen3.8-27B-FP8=/mnt/data2/Qwen3.8-27B-FP8",
    ])

    rows: list[dict] = []
    scanned = []
    for item in checkpoints:
        name, _, path = item.partition("=")
        facts = scan_checkpoint(name, path or name)
        scanned.append(facts)
    if not args.no_synthetic_70b:
        scanned.append(_synthetic_70b())

    for facts in scanned:
        row: dict[str, Any] = {
            "name": facts.name,
            "path": facts.path,
            "kind": facts.kind,
            "n_layers": facts.n_layers,
            "hidden": facts.hidden,
            "experts": facts.experts,
            "top_k": facts.top_k,
            "total_gib": facts.total_bytes / GiB,
            "notes": facts.notes,
        }
        if facts.layer_bytes:
            ordered = sorted(facts.layer_bytes)
            row["layer_bytes"] = {
                "min": ordered[0],
                "median": ordered[len(ordered) // 2],
                "max": ordered[-1],
                "mean": sum(ordered) / len(ordered),
                "dense_share": (sum(ordered) - facts.routed_expert_bytes)
                / sum(ordered),
            }
            row["routed_expert_gib"] = facts.routed_expert_bytes / GiB
            if facts.experts and facts.top_k:
                row["active_expert_gib"] = (
                    facts.routed_expert_bytes * facts.top_k / facts.experts / GiB
                )
        row["per_rank"] = {}
        for tp in (1, 2, 4):
            rank_bytes = facts.total_bytes / tp
            entry = {"weights_gib": rank_bytes / GiB}
            if facts.routed_expert_bytes:
                entry["routed_expert_gib"] = facts.routed_expert_bytes / tp / GiB
            row["per_rank"][str(tp)] = entry

        if facts.host_only_bytes:
            row["host_only_gib"] = facts.host_only_bytes / GiB

        # Layer-granular placement: how much of the model fits on a card and
        # what the remaining layers cost to stream on every token.  The whole
        # layer has to fit, not the mean layer, so a checkpoint whose largest
        # layer does not fit cannot be placed at all at that TP width.
        if facts.layer_bytes:
            mean_layer = sum(facts.layer_bytes) / len(facts.layer_bytes)
            largest_layer = max(facts.layer_bytes)
            budget = args.gpu_budget_gib * GiB
            row["placement"] = {}
            for tp in (1, 2, 4):
                rank_budget = budget
                # Every layer exists once per rank at TP1 only; at TP>1 the
                # per-rank layer is a slice, so the shard is the layer over TP.
                per_rank_largest = largest_layer / tp
                resident = int(rank_budget // per_rank_largest)
                resident = min(resident, facts.n_layers)
                offloaded = max(0, facts.n_layers - resident)
                entry = {
                    "rank_layer_gib": mean_layer / tp / GiB,
                    "rank_largest_layer_gib": per_rank_largest / GiB,
                    "resident_layers": resident,
                    "offloaded_layers": offloaded,
                }
                entry["ceiling"] = _ceiling(
                    int(mean_layer / tp), offloaded, bandwidth_gib_s,
                    compute_ms_per_layer,
                )
                row["placement"][str(tp)] = entry
        rows.append(row)

    return {
        "assumptions": {
            "bandwidth_gib_s": bandwidth_gib_s,
            "compute_ms_per_layer_1row": compute_ms_per_layer,
            "gpu_budget_gib": args.gpu_budget_gib,
            "bandwidth_source": bandwidth_source,
            "compute_source": compute_source,
            "note": "resident_layers counts whole layers against the GPU "
                    "budget with no allowance for KV cache, activations or "
                    "the compute buffers, so it is an upper bound on how much "
                    "can stay resident and a lower bound on the offloaded "
                    "share",
        },
        "checkpoints": rows,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", default="all",
                   choices=["all", "storage", "h2d", "overlap", "budget"])
    p.add_argument("--json", dest="json_path", default=None,
                   help="write the full result tree here")
    p.add_argument("--gpus", type=int, default=4,
                   help="GPUs used by the h2d stage")
    p.add_argument("--gpu", type=int, default=0,
                   help="single GPU used by the overlap stage")

    g = p.add_argument_group("storage")
    g.add_argument("--storage-mounts", default="/mnt/data1,/mnt/data2,/mnt/data3")
    g.add_argument("--storage-sample-mb", type=int, default=2048,
                   help="bytes sampled per mount (default 2 GiB)")
    g.add_argument("--storage-threads", type=int, default=4,
                   help="threads used for the parallel warm read")

    g = p.add_argument_group("h2d")
    g.add_argument("--copy-min-bytes", type=int, default=4 * 1024)
    g.add_argument("--copy-max-bytes", type=int, default=512 * MiB)
    g.add_argument("--copy-target-bytes", type=int, default=256 * MiB,
                   help="traffic per measurement point; iteration count is "
                        "chosen to approximate this")
    g.add_argument("--concurrent-bytes", type=int, default=64 * MiB)
    g.add_argument("--numa-bytes", type=int, default=64 * MiB,
                   help="0 disables the NUMA A/B")

    g = p.add_argument_group("overlap")
    g.add_argument("--overlap-rows", type=int, nargs="+", default=[1, 512])
    g.add_argument("--overlap-layers", type=int, default=4)
    g.add_argument("--overlap-spec", default=None,
                   help="comma-separated subset of the built-in layer specs")

    g = p.add_argument_group("budget")
    g.add_argument("--checkpoint", action="append", default=None,
                   metavar="NAME=PATH", help="repeatable; replaces the defaults")
    g.add_argument("--no-synthetic-70b", action="store_true")
    g.add_argument("--gpu-budget-gib", type=float, default=22.0)
    g.add_argument("--bandwidth-gib-s", type=float, default=None,
                   help="override the measured H2D bandwidth used by the "
                        "ceiling arithmetic")
    g.add_argument("--compute-ms-per-layer", type=float, default=None,
                   help="override the measured 1-row per-layer compute time")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stages = (["storage", "h2d", "overlap", "budget"]
              if args.stage == "all" else [args.stage])

    results: dict[str, Any] = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "commit": _git_commit(),
            "command": " ".join(sys.argv),
            "host": os.uname().nodename,
            "cpu": _cpu_model(),
            "python": sys.version.split()[0],
        }
    }
    try:
        import torch
        results["metadata"]["torch"] = torch.__version__
        results["metadata"]["cuda"] = torch.version.cuda
    except ImportError:
        pass

    bandwidth = args.bandwidth_gib_s
    bandwidth_source = "argument" if bandwidth else "unset"
    compute_ms = args.compute_ms_per_layer
    compute_source = "argument" if compute_ms else "unset"

    for stage in stages:
        print(f"[stage] {stage}")
        t0 = time.perf_counter()
        if stage == "storage":
            results["storage"] = stage_storage(
                args.storage_mounts.split(","),
                args.storage_sample_mb * MiB,
                args.storage_threads,
            )
        elif stage == "h2d":
            results["h2d"] = stage_h2d(args)
            best = max((r["pinned_gib_s"] or 0) for r in results["h2d"]["sweep"])
            if bandwidth is None and best:
                bandwidth = best
                bandwidth_source = "measured: best pinned point of the sweep"
        elif stage == "overlap":
            results["overlap"] = stage_overlap(args)
            if compute_ms is None:
                for spec in results["overlap"]["specs"]:
                    if spec["name"] == "glm52-q4-moe":
                        compute_ms = spec["rows"]["1"]["compute_only"]["per_layer_ms"]
                        compute_source = ("measured: glm52-q4-moe, 1 row, "
                                          "per-layer GEMM wall")
                        break
        elif stage == "budget":
            if bandwidth is None:
                # Recorded in docs/performance/qwen4_exp_performance.md.
                bandwidth = 10.61
                bandwidth_source = ("fallback constant: pinned H2D per GPU, "
                                    "docs/performance/qwen4_exp_performance.md")
            if compute_ms is None:
                compute_ms = 1.0
                compute_source = "fallback constant: 1 ms/layer, unmeasured"
            results["budget"] = stage_budget(
                args, bandwidth, compute_ms, bandwidth_source, compute_source,
            )
        results.setdefault("timings", {})[stage] = time.perf_counter() - t0
        print(f"[stage] {stage} done in {results['timings'][stage]:.2f} s")

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
