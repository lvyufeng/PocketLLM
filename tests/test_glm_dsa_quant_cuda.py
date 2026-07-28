from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.loader.gguf.bundle import read_gguf_bundle
from src.loader.gguf.tensor_reader import (
    GGUFTensorDataReader,
    get_iq2xs_iq3xxs_signed_grid_tensor,
)


REAL_GLM_PATH = Path("/mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL")

# Representative tensors per GLM GGUF dtype (read-only /mnt/data3).
IQ2_XS_ROUTED = "blk.3.ffn_gate_exps.weight"   # type_id 5
IQ3_XXS_ROUTED = "blk.3.ffn_down_exps.weight"  # type_id 6
Q6_K_DENSE = "blk.0.ffn_down.weight"           # type_id 8
Q8_0_ATTN = "blk.0.attn_q_b.weight"            # q8_0 attention projection
IQ4_XS_ROUTED = "blk.8.ffn_down_exps.weight"   # type_id 7, iq4_xs (4 layers only)


def _cuda_gemm_available() -> bool:
    if not torch.cuda.is_available():
        return False
    mod = load_cuda_kernel()
    return mod is not None and hasattr(mod, "gguf_quant_gemm_forward")


def _merged_grid_cuda() -> torch.Tensor:
    return get_iq2xs_iq3xxs_signed_grid_tensor().to(device="cuda", dtype=torch.int8).contiguous()


def _empty_grid_cuda() -> torch.Tensor:
    return torch.empty((0,), device="cuda", dtype=torch.int8)


@pytest.mark.skipif(
    not (REAL_GLM_PATH.exists() and _cuda_gemm_available()),
    reason="real GLM GGUF or CUDA extension not available",
)
@pytest.mark.parametrize(
    ("name", "type_id", "type_name", "routed"),
    [
        (IQ2_XS_ROUTED, 5, "iq2_xs", True),
        (IQ3_XXS_ROUTED, 6, "iq3_xxs", True),
        (IQ4_XS_ROUTED, 7, "iq4_xs", True),
        (Q6_K_DENSE, 8, "q6_k", False),
    ],
)
def test_glm_quant_gemm_matches_reference(name: str, type_id: int, type_name: str, routed: bool) -> None:
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    tensor = bundle.tensors_by_name[name]
    reader = GGUFTensorDataReader(tensor.shard_path)
    rows = 16
    try:
        if routed:
            blocks, tn, in_dim = reader.read_routed_expert_blocks(tensor.name, 0, 0, rows)
            ref = reader.read_routed_expert(tensor.name, 0, 0, rows).float()
        else:
            blocks, tn, in_dim = reader.read_quantized_matrix_block_rows(tensor.name, 0, rows)
            ref = reader.read_quantized_matrix_rows_reference(tensor.name, 0, rows).float()
    finally:
        reader.close()

    assert tn == type_name
    assert ref.shape == (rows, in_dim)

    mod = load_cuda_kernel()
    grid = _merged_grid_cuda() if type_id in (5, 6) else _empty_grid_cuda()
    blocks_cuda = blocks.cuda()
    ref_cuda = ref.cuda()

    x = torch.randn((5, in_dim), device="cuda", dtype=torch.float16)
    expected = x.float() @ ref_cuda.t()

    y = mod.gguf_quant_gemm_forward(x, blocks_cuda, in_dim, type_id, grid).float()
    yp = mod.gguf_quant_gemm_prefill_forward(x, blocks_cuda, in_dim, type_id, grid).float()

    assert bool(torch.isfinite(y).all().item())
    assert bool(torch.isfinite(yp).all().item())

    scale = expected.abs().max().clamp_min(1.0e-3)
    rel_decode = (y - expected).abs().max() / scale
    rel_prefill = (yp - expected).abs().max() / scale
    assert float(rel_decode.item()) < 2.0e-2, f"{type_name} decode rel err {rel_decode.item()}"
    assert float(rel_prefill.item()) < 2.0e-2, f"{type_name} prefill rel err {rel_prefill.item()}"


@pytest.mark.skipif(
    not (REAL_GLM_PATH.exists() and _cuda_gemm_available()),
    reason="real GLM GGUF or CUDA extension not available",
)
def test_glm_q8_0_gemm_matches_reference() -> None:
    """q8_0 uses its own kernel (q8_0_gemm_forward), not the grid/type-id path."""
    from src.kernels.ops import q8_0_weight_dequantize

    bundle = read_gguf_bundle(REAL_GLM_PATH)
    tensor = bundle.tensors_by_name[Q8_0_ATTN]
    reader = GGUFTensorDataReader(tensor.shard_path)
    rows = 16
    try:
        blocks = reader.read_q8_0_block_rows(tensor.name, 0, rows)
        row_elems = int(tensor.dimensions[0])
        ref = q8_0_weight_dequantize(blocks, row_elems=row_elems).float()
    finally:
        reader.close()

    assert blocks.shape == (rows, row_elems // 32, 34)
    assert ref.shape == (rows, row_elems)

    mod = load_cuda_kernel()
    blocks_cuda = blocks.cuda()
    ref_cuda = ref.cuda()

    x = torch.randn((5, row_elems), device="cuda", dtype=torch.float16)
    expected = x.float() @ ref_cuda.t()
    y = mod.q8_0_gemm_forward(x, blocks_cuda, row_elems).float()

    x1 = torch.randn((1, row_elems), device="cuda", dtype=torch.float16)
    expected1 = x1.float() @ ref_cuda.t()
    y1 = mod.q8_0_gemm_forward(x1, blocks_cuda, row_elems).float()

    assert bool(torch.isfinite(y).all().item())
    assert bool(torch.isfinite(y1).all().item())

    scale = expected.abs().max().clamp_min(1.0e-3)
    rel_prefill = (y - expected).abs().max() / scale
    scale1 = expected1.abs().max().clamp_min(1.0e-3)
    rel_decode = (y1 - expected1).abs().max() / scale1
    assert float(rel_prefill.item()) < 2.0e-2, f"q8_0 prefill rel err {rel_prefill.item()}"
    assert float(rel_decode.item()) < 2.0e-2, f"q8_0 decode rel err {rel_decode.item()}"


@pytest.mark.skipif(
    not REAL_GLM_PATH.exists(),
    reason="real GLM GGUF not available",
)
def test_glm_iq4_xs_reference_decode() -> None:
    """iq4_xs reference decode must produce finite values (correctness fix for blk.8/75/76/77)."""
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    tensor = bundle.tensors_by_name[IQ4_XS_ROUTED]
    reader = GGUFTensorDataReader(tensor.shard_path)
    rows = 8
    try:
        ref = reader.read_routed_expert(tensor.name, 0, 0, rows).float()
    finally:
        reader.close()

    assert ref.shape[0] == rows
    assert bool(torch.isfinite(ref).all().item()), "iq4_xs reference decode produced NaN/Inf"
    absmax = ref.abs().max().item()
    assert absmax > 0.0 and absmax < 1.0, f"iq4_xs decode absmax {absmax} out of expected range"


@pytest.mark.skipif(
    not REAL_GLM_PATH.exists(),
    reason="real GLM GGUF not available",
)
def test_glm_iq3_xxs_reference_decode_golden() -> None:
    """IQ3_XXS reference decode must match llama.cpp block layout.

    The ``block_iq3_xxs`` layout stores 64 bytes of grid indices followed by
    32 bytes of aux (scales_and_signs) as SEPARATE regions, not interleaved
    12-byte sub-blocks.  Decoding with the interleaved layout mixes grid
    indices with sign/scale bytes and corrupts the down-projection (verified
    against the llama.cpp golden l_out-3, which regressed from 1.5% -> 27%
    relative error when this layout was wrong).  These golden values are
    captured from the corrected decode of blk.3 expert 0 (down, iq3_xxs).
    """
    bundle = read_gguf_bundle(REAL_GLM_PATH)
    tensor = bundle.tensors_by_name[IQ3_XXS_ROUTED]
    assert tensor.type_name == "iq3_xxs"
    reader = GGUFTensorDataReader(tensor.shard_path)
    try:
        ref = reader.read_routed_expert(tensor.name, 0, 0, 8).float()
    finally:
        reader.close()

    expected_row0 = torch.tensor(
        [-0.001674, -0.001674, 0.001674, 0.008369, 0.001674, -0.001674, -0.011717, 0.005021],
        dtype=torch.float32,
    )
    assert torch.allclose(ref[0, :8], expected_row0, atol=1.0e-5), (
        f"IQ3_XXS decode row0 {ref[0, :8].tolist()} != golden {expected_row0.tolist()} "
        "(check block_iq3_xxs grid/aux layout)"
    )
    assert abs(float(ref.sum()) - (-0.535039)) < 1.0e-3, f"IQ3_XXS decode sum {float(ref.sum())} != golden -0.535039"


@pytest.mark.skipif(
    not (REAL_GLM_PATH.exists() and _cuda_gemm_available()),
    reason="real GLM GGUF or CUDA extension not available",
)
def test_glm_lm_head_vocab_sharding() -> None:
    """lm_head vocab sharding: each rank loads a disjoint vocab row slice that
    tiles the full vocab, and every rank's local logits match the full lm_head's
    corresponding slice."""
    from src.models.glm_dsa.gguf_model import GLMDSAGGUFModelLoader

    bundle = read_gguf_bundle(REAL_GLM_PATH)
    world = 4
    total_rows = int(bundle.tensors_by_name["output.weight"].dimensions[1])

    # Full (unsharded) reference lm_head.
    full_loader = GLMDSAGGUFModelLoader(bundle, world=1, rank=0)
    try:
        full_head = full_loader._quant_lm_head("output.weight")
    finally:
        full_loader.close()
    assert full_head.row_start == 0
    assert full_head.out_dim == total_rows

    dim = int(full_head.in_dim)
    x = torch.randn((3, dim), device="cuda", dtype=torch.float16)
    full_logits = full_head(x).float()
    assert full_logits.shape == (3, total_rows)

    covered = 0
    prev_end = 0
    for rank in range(world):
        loader = GLMDSAGGUFModelLoader(bundle, world=world, rank=rank)
        try:
            head = loader._quant_lm_head("output.weight")
        finally:
            loader.close()
        start = head.row_start
        count = head.out_dim
        # Slices tile the vocab contiguously with no gaps/overlap.
        assert start == prev_end, f"rank {rank} start {start} != prev_end {prev_end}"
        prev_end = start + count
        covered += count

        local = head(x).float()
        assert local.shape == (3, count)
        ref_slice = full_logits[:, start:start + count]
        scale = ref_slice.abs().max().clamp_min(1.0e-3)
        rel = (local - ref_slice).abs().max() / scale
        assert float(rel.item()) < 1.0e-3, f"rank {rank} logit slice rel err {rel.item()}"

    assert prev_end == total_rows
    assert covered == total_rows
