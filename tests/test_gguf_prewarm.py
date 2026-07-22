"""Tests for the GGUF page-cache prewarm utility."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.loader.gguf.prewarm import prewarm_paths

REAL_GLM_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")


def test_prewarm_paths_reads_file(tmp_path):
    payload = os.urandom(3 << 20)  # 3 MiB
    f = tmp_path / "shard.bin"
    f.write_bytes(payload)

    elapsed = prewarm_paths([str(f)], chunk_bytes=1 << 20, log=False)

    assert str(f) in elapsed
    assert elapsed[str(f)] >= 0.0
    assert len(elapsed) == 1


def test_prewarm_paths_multiple_files(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"shard_{i}.bin"
        f.write_bytes(os.urandom(1 << 20))
        files.append(str(f))

    elapsed = prewarm_paths(files, chunk_bytes=1 << 20, log=False)

    assert set(elapsed.keys()) == set(files)
    assert all(v >= 0.0 for v in elapsed.values())


def test_prewarm_missing_file_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist.bin")
    with pytest.raises(FileNotFoundError):
        prewarm_paths([missing], log=False)


def test_prewarm_rejects_nonpositive_chunk(tmp_path):
    f = tmp_path / "shard.bin"
    f.write_bytes(b"data")
    with pytest.raises(ValueError):
        prewarm_paths([str(f)], chunk_bytes=0, log=False)


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="real GLM bundle not present")
def test_prewarm_bundle_smoke():
    from src.loader.gguf.bundle import read_gguf_bundle

    bundle = read_gguf_bundle(REAL_GLM_PATH)
    # Only warm the small header shard (00001, ~9 MB) to avoid reading 49 GB.
    small = min(bundle.paths, key=os.path.getsize)
    elapsed = prewarm_paths([small], log=False)
    assert small in elapsed
    assert elapsed[small] >= 0.0
