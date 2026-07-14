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
