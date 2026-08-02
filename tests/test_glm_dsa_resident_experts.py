"""Correctness lock for the GLM-DSA resident-expert staging refactor.

The decode hot path was changed from re-reading each active expert's raw blocks
from mmap + ``torch.stack`` every step to a lazily-built resident CPU cache that
is indexed + async-H2D'd.  This test asserts the resident cache is byte-identical
to the original per-expert mmap reads, so the refactor cannot silently corrupt
expert weights.

Runs only when the real GLM-5.2 GGUF bundle is present (CPU-only; no CUDA
required — we compare the CPU-side staged blocks before the H2D copy).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.loader.gguf.bundle import read_gguf_bundle
from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader
from src.models.glm_dsa import architecture as arch


REAL_GLM_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")


class _StagingHarness:
    """Minimal stand-in exposing exactly the attributes that
    ``_build_resident_experts`` / ``_stage_active_experts`` read, so we can test
    the staging logic without building the full CUDA MoE layer."""

    _build_resident_experts = arch.GLMDSARawBlockMoE._build_resident_experts
    _stage_active_experts = arch.GLMDSARawBlockMoE._stage_active_experts

    def __init__(self, w1_name, w3_name, w2_name, expert_start, expert_count):
        self.w1_name = w1_name
        self.w3_name = w3_name
        self.w2_name = w2_name
        self.expert_start = expert_start
        self.expert_count = expert_count
        self.device = torch.device("cpu")
        self._type_ids = GGUF_DENSE_TYPE_IDS
        self._resident_w1 = None
        self._resident_w3 = None
        self._resident_w2 = None
        self._resident_meta = None
        self._disable_resident = False
        self._pin_resident = False


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="real GLM GGUF not available")
def test_resident_cache_matches_mmap_reads() -> None:
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    # blk.3 is the first routed MoE layer (leading_dense=3).
    prefix = "blk.3"
    w1_name = f"{prefix}.ffn_gate_exps.weight"
    w3_name = f"{prefix}.ffn_up_exps.weight"
    w2_name = f"{prefix}.ffn_down_exps.weight"
    shard = bundle.tensors_by_name[w1_name].shard_path
    reader = get_cached_gguf_tensor_reader(shard)

    expert_start, expert_count = 0, 4
    h = _StagingHarness(w1_name, w3_name, w2_name, expert_start, expert_count)

    # Build the resident cache and stage a subset of active experts.
    active = [3, 1, 0, 2]  # deliberately unordered to exercise index_select
    w1_r, w3_r, w2_r, meta_r = h._stage_active_experts(reader, active)

    # Reference: fresh per-expert mmap reads in the same active order.
    def ref_stack(name):
        blocks = []
        tn = in_dim = None
        for lid in active:
            b, tn, in_dim = reader.read_routed_expert_blocks(name, lid + expert_start)
            blocks.append(b.clone())
        return torch.stack(blocks, dim=0).contiguous(), tn, in_dim

    w1_ref, w1_tn, w1_in = ref_stack(w1_name)
    w3_ref, w3_tn, w3_in = ref_stack(w3_name)
    w2_ref, w2_tn, w2_in = ref_stack(w2_name)

    assert torch.equal(w1_r, w1_ref), "resident w1 blocks differ from mmap reads"
    assert torch.equal(w3_r, w3_ref), "resident w3 blocks differ from mmap reads"
    assert torch.equal(w2_r, w2_ref), "resident w2 blocks differ from mmap reads"

    assert meta_r == (
        (GGUF_DENSE_TYPE_IDS[w1_tn], int(w1_in)),
        (GGUF_DENSE_TYPE_IDS[w3_tn], int(w3_in)),
        (GGUF_DENSE_TYPE_IDS[w2_tn], int(w2_in)),
    )


@pytest.mark.skipif(not REAL_GLM_PATH.exists(), reason="real GLM GGUF not available")
def test_resident_disabled_matches_enabled() -> None:
    """The GLM_DISABLE_RESIDENT_EXPERTS fallback (per-step mmap re-read) must
    produce identical staged blocks to the resident path."""
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    prefix = "blk.3"
    w1_name = f"{prefix}.ffn_gate_exps.weight"
    w3_name = f"{prefix}.ffn_up_exps.weight"
    w2_name = f"{prefix}.ffn_down_exps.weight"
    shard = bundle.tensors_by_name[w1_name].shard_path
    reader = get_cached_gguf_tensor_reader(shard)
    active = [2, 0, 3]

    h_res = _StagingHarness(w1_name, w3_name, w2_name, 0, 4)
    w1a, w3a, w2a, meta_a = h_res._stage_active_experts(reader, active)

    h_dis = _StagingHarness(w1_name, w3_name, w2_name, 0, 4)
    h_dis._disable_resident = True
    w1b, w3b, w2b, meta_b = h_dis._stage_active_experts(reader, active)

    assert torch.equal(w1a, w1b)
    assert torch.equal(w3a, w3b)
    assert torch.equal(w2a, w2b)
    assert meta_a == meta_b
