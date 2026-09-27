from __future__ import annotations

from pathlib import Path

import pytest

from src.loader.gguf.bundle import read_gguf_bundle
from src.components.moe.registry import detect_spec, get_spec, known_architectures, load_bundle
from tests.gguf_test_utils import write_gguf, write_minimax_bundle


def test_known_architectures_include_deepseek_v4_and_minimax() -> None:
    assert "deepseek4" in known_architectures()
    assert "minimax-m2" in known_architectures()


def test_get_spec_normalizes_architecture() -> None:
    assert get_spec("MiniMax-M2").architecture == "minimax-m2"
    assert get_spec("deepseek4").architecture == "deepseek4"


def test_detect_spec_from_minimax_bundle(tmp_path: Path) -> None:
    root = write_minimax_bundle(tmp_path / "bundle", n_layers=1)
    bundle = read_gguf_bundle(root)

    spec = detect_spec(bundle)

    assert spec.architecture == "minimax-m2"


def test_detect_spec_from_deepseek_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ds4.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "deepseek4", "deepseek4.block_count": 1},
        tensors=[],
    )
    bundle = read_gguf_bundle(path)

    spec = detect_spec(bundle)

    assert spec.architecture == "deepseek4"


def test_unknown_architecture_error(tmp_path: Path) -> None:
    path = tmp_path / "unknown.gguf"
    write_gguf(path, metadata={"general.architecture": "dense-thing"}, tensors=[])
    bundle = read_gguf_bundle(path)

    with pytest.raises(ValueError, match="unsupported MoE architecture"):
        detect_spec(bundle)


def test_load_bundle_is_the_registry_reachable_way_to_read_a_checkpoint(tmp_path: Path) -> None:
    """`src/runtime/` must not name a container format, so the generation driver cannot import
    `src.loader` -- and the registry is the one thing above the loader it is allowed to import. That
    only works if the reading path is reachable from here, which is what this pins: the same file
    through both spellings, and the architecture resolved off the result, so the seam is a full
    substitute for the import it replaced rather than a partial one."""
    path = tmp_path / "ds4.gguf"
    write_gguf(
        path,
        metadata={"general.architecture": "deepseek4", "deepseek4.block_count": 1},
        tensors=[],
    )

    bundle = load_bundle(path)

    assert bundle.paths == read_gguf_bundle(path).paths
    assert detect_spec(bundle).architecture == "deepseek4"


def test_load_bundle_reports_a_missing_checkpoint_rather_than_a_type_error(tmp_path: Path) -> None:
    """A `str` path is the spelling every caller uses, so it has to be accepted as one."""
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "not-there.gguf")
