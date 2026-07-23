"""Pull GGUF shard files into the OS page cache.

GLM-5.2 (and other large GGUF bundles) live on a spinning disk here, and the
MoE routed-expert hot path reads expert weights through an ``mmap`` view on
every forward. The first forward is dominated by cold HDD reads (page faults
that hit the platter); once the relevant pages are resident in the Linux page
cache, subsequent forwards run from RAM and are ~60x faster.

This module sequentially reads shard files so their pages become resident.
It does not change the mmap semantics, does not pin host memory, and does not
lock pages -- it only warms the cache. The page cache is shared system-wide,
so under tensor parallelism a single rank prewarming benefits every rank.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.loader.gguf.bundle import GGUFBundle

_DEFAULT_CHUNK_BYTES = 32 << 20  # 32 MiB sequential reads


def _fadvise(fd: int) -> None:
    """Hint the kernel to read ahead sequentially and prefetch the whole file.

    ``posix_fadvise`` is POSIX-only; on platforms without it the sequential
    reads below still make the pages resident, just without the readahead hint.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    except OSError:
        # Advisory only -- failure here does not affect correctness.
        pass


def prewarm_paths(
    paths: Sequence[str],
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    use_fadvise: bool = True,
    log: bool = True,
) -> dict[str, float]:
    """Sequentially read files to pull them into the OS page cache.

    Args:
        paths: Files to warm (e.g. ``bundle.paths``).
        chunk_bytes: Read size per ``os.read`` call.
        use_fadvise: Issue ``POSIX_FADV_SEQUENTIAL``/``WILLNEED`` hints.
        log: Print per-shard byte count, elapsed seconds, and bandwidth.

    Returns:
        Mapping of each path to the elapsed seconds spent reading it.

    Raises:
        FileNotFoundError: If a path does not exist.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")

    elapsed: dict[str, float] = {}
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            if use_fadvise:
                _fadvise(fd)
            t0 = time.perf_counter()
            total = 0
            while True:
                chunk = os.read(fd, chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
            dt = time.perf_counter() - t0
        finally:
            os.close(fd)
        elapsed[path] = dt
        if log:
            gib = total / 1024**3
            bw = (total / 1024**2) / dt if dt > 0 else 0.0
            name = os.path.basename(path)
            print(
                f"prewarm shard={name} bytes={gib:.2f}GB elapsed={dt:.1f}s bw={bw:.0f}MB/s",
                flush=True,
            )
    return elapsed


def prewarm_bundle(
    bundle: "GGUFBundle",
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    use_fadvise: bool = True,
    log: bool = True,
) -> dict[str, float]:
    """Prewarm every shard file backing a :class:`GGUFBundle`.

    A single sequential pass is intentional: on a spinning disk one stream
    reaches full sequential bandwidth, whereas several concurrent streams
    across ranks would thrash the head and lower aggregate throughput. Under
    tensor parallelism, warm on one rank only (the page cache is shared
    system-wide) and let the others wait at a barrier.
    """
    return prewarm_paths(
        bundle.paths,
        chunk_bytes=chunk_bytes,
        use_fadvise=use_fadvise,
        log=log,
    )
