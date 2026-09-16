"""Tests for the partial-checkpoint mode of `scripts/audit_dsv41_headers.py`.

The checkpoint is 475 GiB over 48 shards and the audit is meant to be useful
while it is still arriving, so the tool distinguishes three outcomes rather than
two: a check whose evidence is local and agrees *passes*, one whose evidence is
local and disagrees *fails*, and one whose evidence sits in a shard that has not
been downloaded is *undecided* -- reported separately and never counted as a pass.
The split runs through the whole script: `model.safetensors.index.json` answers
presence for all 96,085 tensors, and only the headers of the shards on disk can
answer shape.

What these tests have to pin down is therefore not any single number but the
boundary between those outcomes, and in particular that a defect is never
deferred: a bad shape inside a *downloaded* shard must fail while other shards are
still missing. The cases below are built by taking the real header-prefix tree
(`/tmp/dsv41`, the same 48 headers the documented `curl -r 0-3000000` recipe
fetches), rebuilding the index from it -- which is what an index is a statement
about -- and then keeping only one or two shards on disk. That is a real partial
checkpoint, one byte-for-byte unmodified header at a time, and it needs no
download.

The unit half of the file needs neither tree nor checkpoint: the three-state
`Report`, the shard labels, the subset-until-complete rule and the coverage
lookup are pure functions.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "scripts" / "audit_dsv41_headers.py"

# The real checkpoint, if this host has it, and the header-prefix tree it can be
# audited from without one. The tree is what the partial cases are built out of.
CHECKPOINT = Path("/mnt/data3/DeepSeek-V4.1-Flash")
HF_CONFIG = CHECKPOINT / "config.json"
HEADER_TREE = Path("/tmp/dsv41")

SHARD_1 = "model-00001-of-00048.safetensors"  # the vision tower and the aligner
SHARD_3 = "model-00003-of-00048.safetensors"  # backbone layer 0
ENGRAM_SHARD = "model-00047-of-00048.safetensors"  # the layer-1 Engram table

requires_header_tree = pytest.mark.skipif(not HEADER_TREE.is_dir(), reason="no header-prefix tree at /tmp/dsv41")
requires_config = pytest.mark.skipif(not HF_CONFIG.is_file(), reason="no released config to audit against")
requires_checkpoint = pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="no released checkpoint on this host")


def _load_audit():
    """The script as a module, so its helpers can be exercised directly."""
    spec = importlib.util.spec_from_file_location("audit_dsv41_headers", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT_MODULE = _load_audit()


# ---------------------------------------------------------------------------
# the three outcomes
# ---------------------------------------------------------------------------


def test_a_pass_with_its_evidence_missing_is_undecided_not_passed():
    report = AUDIT_MODULE.Report()
    outcome = report.check("a shape", True, "observed 3", waiting=[ENGRAM_SHARD])

    assert outcome is None
    assert report.passed == 0
    assert report.failures == []
    assert report.undecided == [("a shape", "undecided: 1 shard(s) not downloaded (47/48); observed 3")]


def test_a_failure_is_never_deferred_to_a_missing_shard():
    report = AUDIT_MODULE.Report()
    outcome = report.check("a shape", False, "(512, 5121) != (512, 5120)", waiting=[ENGRAM_SHARD])

    assert outcome is False
    assert report.undecided == []
    assert report.failures == [("a shape", "(512, 5121) != (512, 5120); 1 shard(s) also not downloaded")]


def test_the_three_outcomes_are_counted_apart():
    report = AUDIT_MODULE.Report()
    report.check("passes", True)
    report.check("fails", False, "detail")
    report.check("waits", True, "", [ENGRAM_SHARD])

    assert (report.passed, len(report.failures), len(report.undecided)) == (1, 1, 1)
    assert len(report.checks) == 3


def test_shard_labels_shorten_the_release_names_only():
    assert AUDIT_MODULE.shard_label(SHARD_1) == "1/48"
    assert AUDIT_MODULE.shard_label("h00015.bin") == "h00015.bin"


def test_a_set_claim_is_subset_until_the_evidence_is_all_in():
    settles = AUDIT_MODULE.settles
    expected = {(32, 32)}

    # Nothing read yet is an absence of evidence, not a contradiction.
    assert settles(set(), expected, complete=False)
    assert not settles(set(), expected, complete=True)
    # A different block is a contradiction whether or not shards are outstanding.
    assert not settles({(16, 16)}, expected, complete=False)
    assert settles(expected, expected, complete=False)


# ---------------------------------------------------------------------------
# coverage: which shard must land before a check can be decided
# ---------------------------------------------------------------------------

WEIGHT_MAP = {
    "layers.0.attn.wkv.scale": SHARD_3,
    "layers.0.attn.wkv.weight": SHARD_3,
    "layers.1.engram.embed.weight": ENGRAM_SHARD,
    "vision.patch_embed.proj.weight": SHARD_1,
}


def test_coverage_names_the_shards_a_check_is_waiting_on():
    coverage = AUDIT_MODULE.Coverage(WEIGHT_MAP)
    coverage.identify({"h00003.bin": {name: None for name in WEIGHT_MAP if name.startswith("layers.0")}})

    assert coverage.local == {SHARD_3}
    assert coverage.pending == [SHARD_1, ENGRAM_SHARD]
    assert coverage.waiting(["layers.0.attn.wkv.weight"]) == []
    assert coverage.waiting(["layers.1.engram.embed.weight"]) == [ENGRAM_SHARD]
    assert coverage.waiting(["layers.1.engram.embed.weight", "vision.patch_embed.proj.weight"]) == [
        SHARD_1,
        ENGRAM_SHARD,
    ]
    # Names the index does not list cannot be waited on.
    assert coverage.waiting(["layers.9.nonexistent"]) == []


def test_coverage_finds_whole_prefixes_at_once():
    coverage = AUDIT_MODULE.Coverage(WEIGHT_MAP)
    coverage.identify({"h00003.bin": {name: None for name in WEIGHT_MAP if name.startswith("layers.0")}})

    assert coverage.waiting_under("layers.") == [ENGRAM_SHARD]
    assert coverage.waiting_under("vision.") == [SHARD_1]
    assert coverage.waiting_under("nothing.") == []


def test_a_file_mixing_two_index_shards_is_left_unmatched():
    """`identify` matches on tensors, so a file that mixes shards matches nothing."""
    coverage = AUDIT_MODULE.Coverage(WEIGHT_MAP)
    coverage.identify({"mixed.bin": {"layers.0.attn.wkv.weight": None, "vision.patch_embed.proj.weight": None}})

    assert coverage.shard_of == {}
    assert coverage.local == set()
    assert coverage.pending == sorted(set(WEIGHT_MAP.values()))


def test_presence_comes_from_the_index_and_shape_from_the_headers():
    coverage = AUDIT_MODULE.Coverage(WEIGHT_MAP)
    coverage.identify({"h00003.bin": {"layers.0.attn.wkv.weight": None}})
    ckpt = AUDIT_MODULE.Checkpoint(
        {"layers.0.attn.wkv.weight": ("F8_E4M3", (512, 5120), 2621440, "h00003.bin")},
        {},
        True,
        coverage,
    )

    assert "layers.1.engram.embed.weight" in ckpt  # shipped, but not downloaded
    assert ckpt.shape("layers.1.engram.embed.weight") is None
    assert ckpt.shape("layers.0.attn.wkv.weight") == (512, 5120)
    assert ckpt.waiting("layers.1.engram.embed.weight") == [ENGRAM_SHARD]


def test_without_an_index_presence_falls_back_to_the_headers():
    ckpt = AUDIT_MODULE.Checkpoint({"a.weight": ("BF16", (2,), 4, "s.bin")}, {}, True, AUDIT_MODULE.Coverage())

    assert "a.weight" in ckpt
    assert "b.weight" not in ckpt
    assert ckpt.waiting("b.weight") == []


# ---------------------------------------------------------------------------
# a real partial checkpoint, built from the header-prefix tree
# ---------------------------------------------------------------------------


def index_from_headers(header_tree: Path) -> dict:
    """The tensor -> shard map the 48 headers imply, under the release's names."""
    weight_map = {}
    for path in sorted(header_tree.glob("h*.bin")):
        header, _length, _complete = AUDIT_MODULE.read_header(str(path))
        number = int(re.search(r"(\d+)", path.stem).group(1))
        shard = f"model-{number:05d}-of-00048.safetensors"
        for name in header:
            if name != "__metadata__":
                assert name not in weight_map, f"{name} is in two shards"
                weight_map[name] = shard
    return weight_map


def rewrite_header(path: Path, mutate) -> None:
    """Apply `mutate` to a safetensors header in place, keeping the payload bytes."""
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    mutate(header)
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + header_len :])


def run_audit(directory: Path, contract: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict | None]:
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--checkpoint-dir", str(directory), "--json", str(contract), *extra],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    report = json.loads(contract.read_text(encoding="utf-8")) if contract.exists() else None
    return result, report


def statuses(report: dict) -> dict[str, str]:
    return {check["name"]: check["status"] for check in report["checks"]}


@pytest.fixture
def partial(tmp_path):
    """An index over all 48 shards, with only two of them on disk.

    The index is rebuilt from the header tree rather than read from the checkpoint,
    so this needs no download: it is the same 96,085 tensors as the released
    `model.safetensors.index.json`, assembled from the headers the release's own
    documentation says how to fetch. Shards 1 and 3 are kept -- the vision tower
    and backbone layer 0 -- which is one shape-checkable layer, one shape-checkable
    aligner, and 46 shards worth of checks that must not be called failures.
    """
    shutil.copyfile(HF_CONFIG, tmp_path / "config.json")
    for name in ("h00001.bin", "h00003.bin"):
        shutil.copyfile(HEADER_TREE / name, tmp_path / name)
    weight_map = index_from_headers(HEADER_TREE)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": weight_map}), encoding="utf-8"
    )
    return tmp_path


@requires_header_tree
@requires_config
def test_a_partial_checkpoint_fails_nothing(partial, tmp_path):
    """The point of the mode: 46 shards missing is not 46 shards' worth of defects."""
    result, report = run_audit(partial, tmp_path / "contract.json")

    assert result.returncode == 0, result.stdout + result.stderr
    assert report is not None
    assert [check for check in report["checks"] if check["status"] == "fail"] == []
    assert [check for check in report["checks"] if check["status"] == "undecided"]
    assert report["index"]["local_shards"] == [SHARD_1, SHARD_3]
    assert len(report["index"]["pending_shards"]) == 46
    assert report["tensors"] < report["index"]["shipped_tensors"] == 96085


@requires_header_tree
@requires_config
def test_the_index_alone_decides_the_presence_checks(partial, tmp_path):
    """Which layers own a compressor, how many experts: index questions, all decided."""
    _result, report = run_audit(partial, tmp_path / "contract.json")
    decided = statuses(report)

    for name in (
        "inventory: every backbone layer has its full tensor set",
        "csa2: compressor.wkv is present exactly on kv_source_layers",
        "csa2: compressor.wgate is present exactly on the ratio>1 KV sources",
        "csa2: indexer.wk is present exactly on kv_source_layers",
        "csa2: indexer.wq_b is present exactly on index_source_layers",
        "csa2: Full/Reindex/Reuse partition the backbone",
        "engram: the tables sit on exactly engram_layer_ids",
        "experts: every backbone layer has 384 routed experts",
        "experts: every MTP layer has 128 routed experts",
        "index: every local shard holds exactly the tensors the index assigns it",
    ):
        assert decided[name] == "pass", f"{name}: {decided[name]}"

    # ...and the shape checks that need a shard nobody has are waiting, not failed.
    for name in (
        "engram: the tables are F8_E4M3 with one E8M0 scale per 32 channels",
        "dspark: the Markov and confidence heads match the config",
    ):
        assert decided[name] == "undecided", f"{name}: {decided[name]}"


@requires_header_tree
@requires_config
def test_a_check_waits_for_its_shard_rather_than_failing(partial, tmp_path):
    """Drop the one layer that was on disk: every shape it answered becomes undecided."""
    (partial / "h00003.bin").unlink()
    result, report = run_audit(partial, tmp_path / "contract.json")

    assert result.returncode == 0, result.stdout + result.stderr
    decided = statuses(report)
    assert "fail" not in decided.values()
    assert decided["inventory: the known shapes match the config"] == "undecided"
    assert decided["experts: w1/w2/w3 are FP4 packed into I8 with FP4-block-32 E8M0 scales"] == "undecided"
    assert decided["experts: every backbone layer has 384 routed experts"] == "pass"

    # `--require-complete` is the same run read strictly: undecided becomes failure.
    strict, _strict_report = run_audit(partial, tmp_path / "strict.json", "--require-complete")
    assert strict.returncode == 1
    assert "counted as failures" in strict.stderr


@requires_header_tree
@requires_config
def test_a_wrong_shape_inside_a_downloaded_shard_still_fails(partial, tmp_path):
    """A missing shard defers a check; a shard that is here and wrong does not."""
    rewrite_header(
        partial / "h00003.bin",
        lambda header: header["layers.0.attn.wkv.weight"].update({"shape": [512, 5121]}),
    )
    result, report = run_audit(partial, tmp_path / "contract.json")

    assert result.returncode == 1
    failed = {check["name"]: check["detail"] for check in report["checks"] if check["status"] == "fail"}
    assert "(512, 5121) != (512, 5120)" in failed["inventory: the known shapes match the config"]
    # The byte extent no longer matches the shape, and the scale no longer blocks.
    assert "packing: every tensor's byte extent matches its shape and dtype" in failed
    assert "scales: every weight/scale pair blocks evenly" in failed


@requires_header_tree
@requires_config
def test_a_header_that_disagrees_with_the_index_fails(partial, tmp_path):
    """Presence is the index's to state, and a header contradicting it is a defect.

    Renaming one tensor leaves the index saying the checkpoint ships
    `layers.0.attn.wkv.weight` on a shard that is on disk while the header does not
    have it -- a real inconsistency, and one that only the index makes visible.
    """
    rewrite_header(
        partial / "h00003.bin",
        lambda header: header.update({"layers.0.attn.wkv.weight_renamed": header.pop("layers.0.attn.wkv.weight")}),
    )
    result, report = run_audit(partial, tmp_path / "contract.json")

    assert result.returncode == 1
    decided = statuses(report)
    assert decided["index: every local shard holds exactly the tensors the index assigns it"] == "fail"
    assert decided["inventory: the known shapes match the config"] == "fail"
    # Presence is still the index's answer, so this one did not move.
    assert decided["inventory: every backbone layer has its full tensor set"] == "pass"


# ---------------------------------------------------------------------------
# the released checkpoint itself
# ---------------------------------------------------------------------------


@requires_checkpoint
def test_the_released_checkpoint_passes_on_the_shards_it_has(tmp_path):
    """Whatever has been downloaded must not produce a failure.

    This asserts the invariant rather than a count, because the count moves as the
    download advances: it is a failure of the tool if a shard arriving turns a pass
    into a defect. `--require-complete` is checked to agree with the report, so a
    fully downloaded checkpoint exits 0 both ways.
    """
    result, report = run_audit(CHECKPOINT, tmp_path / "contract.json")

    assert [check for check in report["checks"] if check["status"] == "fail"] == [], result.stdout[-4000:]
    assert result.returncode == 0, result.stdout[-4000:]
    assert report["index"]["indexed"] is True
    assert report["index"]["shipped_tensors"] == 96085
    assert len(report["index"]["local_shards"]) + len(report["index"]["pending_shards"]) == 48
    # Every shard on disk is exactly the shard the index says it is, by name.
    assert report["index"]["local_shards"] == sorted(report["index"]["local_shards"])

    undecided = [check for check in report["checks"] if check["status"] == "undecided"]
    strict, _strict_report = run_audit(CHECKPOINT, tmp_path / "strict.json", "--require-complete")
    assert strict.returncode == (1 if undecided else 0)
