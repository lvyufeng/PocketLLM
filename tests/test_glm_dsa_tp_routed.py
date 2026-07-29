"""Numerical equivalence of GLM-DSA routed-expert TP (inter-dim slicing) vs the
full/EP computation, plus the TP slice-alignment guards.

The routed MoE can shard two ways across TP ranks:
  * EP (original): each rank owns a contiguous expert range, full inter dim.
  * TP (default, GLM_ROUTED_TP=1): every rank owns ALL experts but only its
    1/world slice of the inter (feed-forward) dimension; w2 becomes a partial
    sum along inter and the per-layer all_reduce combines the ranks.

Correctness invariant: summing the per-rank TP inter-slice outputs must equal
the single full-inter grouped forward (the value EP's all_reduce reconstructs),
up to float summation-order noise.  We drive the same
gguf_moe_prefill_grouped_forward entry point used by _routed_forward, once at
full inter and once per rank slice, and assert the slice-sum matches.

The w13 slice is a contiguous out_dim (== inter) row range; the w2 slice keeps
only the matching block-columns of every output row (in_dim == inter).  This
mirrors GLMDSARawBlockMoE._stage_active_experts under self._tp_routed.

Real GLM bundle + CUDA extension required for the numeric test; the alignment
guard test runs anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.loader.gguf.bundle import read_gguf_bundle
from src.loader.gguf.tensor_reader import (
    GGUFTensorDataReader,
    get_iq2xs_iq3xxs_signed_grid_tensor,
)


REAL_GLM_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")


def _cuda_gguf_ext_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    return cuda_mod is not None and hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward")


def _type_id(tn: str) -> int:
    from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS

    return GGUF_DENSE_TYPE_IDS[tn]


def _stack_expert_blocks(reader, name, experts):
    blocks = []
    tn = in_dim = None
    for eid in experts:
        b, tn, in_dim = reader.read_routed_expert_blocks(name, eid)
        blocks.append(b.clone())
    return torch.stack(blocks, dim=0).contiguous(), tn, int(in_dim)


def _run_grouped(cuda_mod, x, rt, rw, seg, w1, w3, w2, meta, grid):
    (w1_tid, w1_in), (w3_tid, w3_in), (w2_tid, w2_in) = meta
    backups = {}
    for flag in ("DEEPSEEK_GGUF_IQ2_XS_W13_DP4A", "DEEPSEEK_GGUF_IQ3_XXS_W2_DP4A"):
        backups[flag] = os.environ.get(flag)
        os.environ[flag] = "1"
    try:
        return cuda_mod.gguf_moe_prefill_grouped_forward(
            x, rt, rw, seg, w1, w3, w2,
            int(w1_in), int(w1_tid),
            int(w3_in), int(w3_tid),
            int(w2_in), int(w2_tid),
            grid, 0.0,
        )
    finally:
        for flag, val in backups.items():
            if val is not None:
                os.environ[flag] = val
            else:
                os.environ.pop(flag, None)


@pytest.mark.skipif(
    not (REAL_GLM_PATH.exists() and _cuda_gguf_ext_available()),
    reason="real GLM GGUF or CUDA extension not available",
)
def test_glm_tp_inter_slice_sum_matches_full() -> None:
    from src.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    prefix = "blk.3"  # first routed MoE layer (leading_dense=3)
    w1_name = f"{prefix}.ffn_gate_exps.weight"
    w3_name = f"{prefix}.ffn_up_exps.weight"
    w2_name = f"{prefix}.ffn_down_exps.weight"
    shard = bundle.tensors_by_name[w1_name].shard_path
    reader = GGUFTensorDataReader(shard)
    device = torch.device("cuda:0")

    try:
        experts = [0, 1]
        w1_cpu, w1_tn, dim = _stack_expert_blocks(reader, w1_name, experts)
        w3_cpu, w3_tn, _ = _stack_expert_blocks(reader, w3_name, experts)
        w2_cpu, w2_tn, inter_dim = _stack_expert_blocks(reader, w2_name, experts)
    finally:
        reader.close()

    assert w1_tn == "iq2_xs" and w3_tn == "iq2_xs" and w2_tn == "iq3_xxs"
    # w13: [E, inter, w13_blocks_per_row, 74]; w2: [E, dim, w2_blocks_per_row, 98]
    assert w1_cpu.size(1) == inter_dim
    assert w2_cpu.size(1) == dim
    w2_blocks_per_row = w2_cpu.size(2)
    assert w2_blocks_per_row * 256 == inter_dim, (w2_blocks_per_row, inter_dim)

    grid = get_iq2xs_iq3xxs_signed_grid_tensor().to(device=device, dtype=torch.int8).contiguous()

    tokens, topk, n_experts = 4, 2, 2
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], device=device, dtype=torch.long)
    weights = torch.rand(tokens, topk, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    grouped = cuda_mod.moe_group_routes(indices, weights, 0, n_experts)
    _local_ids, route_tokens, route_weights, seg_starts = grouped

    torch.manual_seed(1234)
    x = torch.randn(tokens, dim, device=device, dtype=torch.float16)

    # Reference: full inter dimension (what EP's all_reduce reconstructs).
    meta_full = ((_type_id(w1_tn), dim), (_type_id(w3_tn), dim), (_type_id(w2_tn), inter_dim))
    y_full = _run_grouped(
        cuda_mod, x, route_tokens, route_weights, seg_starts,
        w1_cpu.to(device), w3_cpu.to(device), w2_cpu.to(device), meta_full, grid,
    )

    # TP: slice inter into `world` equal pieces, run each, sum the outputs.
    world = 4
    assert inter_dim % world == 0
    inter_slice = inter_dim // world
    assert inter_slice % 256 == 0
    blk_slice = inter_slice // 256

    y_tp = torch.zeros_like(y_full)
    for r in range(world):
        r0, r1 = r * inter_slice, (r + 1) * inter_slice
        b0, b1 = r * blk_slice, (r + 1) * blk_slice
        # w13: contiguous out_dim (inter) row slice.
        w1_s = w1_cpu[:, r0:r1, :, :].contiguous().to(device)
        w3_s = w3_cpu[:, r0:r1, :, :].contiguous().to(device)
        # w2: keep the matching block-columns of every output row.
        w2_s = w2_cpu[:, :, b0:b1, :].contiguous().to(device)
        meta_s = ((_type_id(w1_tn), dim), (_type_id(w3_tn), dim), (_type_id(w2_tn), inter_slice))
        y_r = _run_grouped(
            cuda_mod, x, route_tokens, route_weights, seg_starts,
            w1_s, w3_s, w2_s, meta_s, grid,
        )
        y_tp = y_tp + y_r

    assert y_tp.shape == y_full.shape == (tokens, dim)
    assert torch.isfinite(y_tp).all() and torch.isfinite(y_full).all()
    assert y_full.abs().sum() > 0.0 and y_tp.abs().sum() > 0.0

    abs_diff = (y_tp - y_full).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mask = y_full.abs() > 1e-2
    rel = abs_diff[mask] / (y_full[mask].abs() + 1e-8)
    p99_rel = float(torch.quantile(rel, 0.99).item()) if rel.numel() else 0.0

    # Same value, only float summation order differs (partial-inter sums vs one
    # full-inter reduction), so the tolerance is tight.
    assert max_abs < 5.0e-2, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"
    assert mean_abs < 1.0e-3, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"
    assert p99_rel < 5.0e-2, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"

    print(f"✓ GLM routed TP inter-slice sum matches full: max_abs={max_abs:.4e} "
          f"mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}")


def test_glm_tp_inter_slice_alignment_guard() -> None:
    """A non-256-aligned inter slice must be rejected at MoE construction."""
    from src.models.glm_dsa.architecture import GLMDSAArgs, GLMDSARawBlockMoE

    # GLMDSAArgs is a frozen dataclass; build one with the fields the guard uses
    # (n_routed_experts, moe_inter_dim, dim, top_k, expert_weights_*) set and the
    # rest as harmless placeholders.
    args = GLMDSAArgs(
        n_layers=4, leading_dense_layers=3, dim=64, vocab_size=32, n_heads=1,
        n_kv_heads=1, head_dim=16, q_lora_rank=0, kv_lora_rank=0, key_mla_dim=0,
        value_dim=0, value_mla_dim=0, rope_dim=0, rope_base=10000.0,
        indexer_heads=0, indexer_key_dim=0, dense_inter_dim=128,
        n_routed_experts=8, top_k=2, moe_inter_dim=2048, n_shared_experts=0,
        norm_eps=1e-5, context_length=128, expert_weights_norm=True,
        expert_weights_scale=1.0,
    )

    gate = torch.zeros(args.n_routed_experts, args.dim)
    grid = torch.zeros(1, dtype=torch.int8)
    device = torch.device("cpu")

    common = dict(
        gguf_path="/nonexistent.gguf",
        w1_name="w1", w3_name="w3", w2_name="w2",
        signed_grid=grid, device=device, dtype=torch.float16,
        expert_start=0, expert_count=args.n_routed_experts,
    )

    # 300 is not a multiple of 256 -> must raise.
    with pytest.raises(ValueError, match="256-block aligned"):
        GLMDSARawBlockMoE(args, 3, gate, None, None, None, None,
                          inter_start=0, inter_count=300, **common)

    # 512 is aligned -> constructs fine and records the block-column range.
    moe = GLMDSARawBlockMoE(args, 3, gate, None, None, None, None,
                            inter_start=512, inter_count=512, **common)
    assert moe._tp_routed is True
    assert moe._w2_block_start == 2 and moe._w2_block_end == 4
