"""Numerical correctness of the GLM-DSA iq2_xs (w13) and iq3_xxs (w2) DP4A
grouped-MoE kernels vs the fp32 general branch.

GLM-5.2's routed experts are iq2_xs (w1/w3, 74 B/256) + iq3_xxs (w2, 98 B/256),
which the DP4A fast path did not previously cover (it gated on iq2_xxs 66 B
only), so decode fell to the slow fp32 gguf_moe_w13_kernel.  These tests drive
the same gguf_moe_prefill_grouped_forward entry point with the new env gates
DEEPSEEK_GGUF_IQ2_XS_W13_DP4A / DEEPSEEK_GGUF_IQ3_XXS_W2_DP4A on vs off and
assert the DP4A output matches the fp32 baseline within int8-quant tolerance.

Real GLM bundle + CUDA extension required.
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


def _stack_expert_blocks(reader, name, experts):
    """Stack [E, N, K_blocks, block_bytes] like _stage_active_experts does."""
    blocks = []
    tn = in_dim = None
    for eid in experts:
        b, tn, in_dim = reader.read_routed_expert_blocks(name, eid)
        blocks.append(b.clone())
    return torch.stack(blocks, dim=0).contiguous(), tn, int(in_dim)


def _type_id(tn: str) -> int:
    from src.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS

    return GGUF_DENSE_TYPE_IDS[tn]


def _run_grouped(cuda_mod, x, rt, rw, seg, w1, w3, w2, meta, grid, env_flags_on):
    (w1_tid, w1_in), (w3_tid, w3_in), (w2_tid, w2_in) = meta
    backups = {}
    for flag in ("DEEPSEEK_GGUF_IQ2_XS_W13_DP4A", "DEEPSEEK_GGUF_IQ3_XXS_W2_DP4A"):
        backups[flag] = os.environ.get(flag)
        if env_flags_on:
            os.environ[flag] = "1"
        else:
            os.environ.pop(flag, None)
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
def test_glm_iq2xs_iq3xxs_dp4a_matches_fp32() -> None:
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

    w1 = w1_cpu.to(device)
    w3 = w3_cpu.to(device)
    w2 = w2_cpu.to(device)
    meta = (
        (_type_id(w1_tn), dim),
        (_type_id(w3_tn), dim),
        (_type_id(w2_tn), inter_dim),
    )
    grid = get_iq2xs_iq3xxs_signed_grid_tensor().to(device=device, dtype=torch.int8).contiguous()

    tokens, topk, n_experts = 4, 2, 2
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], device=device, dtype=torch.long)
    weights = torch.rand(tokens, topk, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    grouped = cuda_mod.moe_group_routes(indices, weights, 0, n_experts)
    _local_ids, route_tokens, route_weights, seg_starts = grouped
    routes = int(route_tokens.numel())
    assert routes == tokens * topk

    torch.manual_seed(1234)
    x = torch.randn(tokens, dim, device=device, dtype=torch.float16)

    y_float = _run_grouped(cuda_mod, x, route_tokens, route_weights, seg_starts,
                           w1, w3, w2, meta, grid, env_flags_on=False)
    y_dp4a = _run_grouped(cuda_mod, x, route_tokens, route_weights, seg_starts,
                          w1, w3, w2, meta, grid, env_flags_on=True)

    assert y_float.shape == y_dp4a.shape == (tokens, dim)
    assert torch.isfinite(y_float).all() and torch.isfinite(y_dp4a).all()
    assert y_float.abs().sum() > 0.0 and y_dp4a.abs().sum() > 0.0

    abs_diff = (y_dp4a - y_float).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mask = y_float.abs() > 1e-2
    rel = abs_diff[mask] / (y_float[mask].abs() + 1e-8)
    p99_rel = float(torch.quantile(rel, 0.99).item()) if rel.numel() else 0.0

    # int8 activation quant + DP4A over two quant dtypes (w13 iq2_xs AND w2
    # iq3_xxs) accumulates more error than a single-stage kernel; keep tolerance
    # modest but strict enough to catch a wrong grid/scale/layout.
    assert max_abs < 0.5, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"
    assert mean_abs < 3.0e-2, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"
    assert p99_rel < 0.35, f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}"

    print(f"✓ GLM iq2_xs/iq3_xxs DP4A match: max_abs={max_abs:.4e} "
          f"mean_abs={mean_abs:.4e} p99_rel={p99_rel:.4e}")
