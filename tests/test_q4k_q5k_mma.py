"""Numerical correctness of the Q4_K / Q5_K MMA tensor-core prefill path vs the float baseline.

The MMA path (env GGUF_Q4K_Q5K_MMA=1) and the existing float dequant+dot path
consume the same raw GGUF weight blocks. Because MMA quantizes the activation
to int8 and accumulates in int32 before applying scales, its output differs
from the float path by a small, bounded amount. This test asserts that bound.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REAL_MINIMAX_PATH = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")


def _cuda_gguf_ext_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()
    return cuda_mod is not None and hasattr(cuda_mod, "gguf_quant_gemm_prefill_forward")


def _load_qk_weight_rows(tensor_name: str, row_count: int):
    """Load the first ``row_count`` weight rows of a real MiniMax Q4_K/Q5_K tensor.

    Returns (blocks[N, blocks_per_row, block_bytes] uint8 on cuda, row_elems, type_id).
    """
    from pocketllm.loader.gguf.bundle import read_gguf_bundle
    from pocketllm.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

    bundle = read_gguf_bundle(REAL_MINIMAX_PATH)
    # Find the shard that actually holds this tensor name.
    shard_path = None
    for sh in bundle.shards:
        file = sh.file
        names = {t.name for t in file.tensors} if hasattr(file, "tensors") else set()
        if tensor_name in names:
            shard_path = sh.path
            break
    assert shard_path is not None, f"tensor {tensor_name} not found in any shard"

    reader = get_cached_gguf_tensor_reader(shard_path)
    blocks, type_name, row_elems = reader.read_quantized_matrix_block_rows(tensor_name, 0, row_count)
    type_name_l = type_name.lower()
    assert type_name_l in ("q4_k", "q5_k"), f"expected q4_k/q5_k, got {type_name}"
    type_id = 3 if type_name_l == "q4_k" else 4
    blocks = blocks.contiguous().to("cuda", copy=True)  # [N, blocks_per_row, block_bytes]
    return blocks, int(row_elems), type_id


@pytest.mark.skipif(
    not (REAL_MINIMAX_PATH.exists() and _cuda_gguf_ext_available()),
    reason="real MiniMax-M2.7 GGUF bundle or CUDA extension not available",
)
def test_q5k_mma_vs_float():
    """Q5_K MMA prefill output matches float baseline within int8 tolerance."""
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()

    # MiniMax attention q/k/v/o_proj are Q5_K. Pick a known tensor name.
    # The first transformer block's q_proj: "blk.0.attn_q.weight"
    blocks, K, type_id = _load_qk_weight_rows("blk.0.attn_q.weight", row_count=512)
    assert type_id == 4, "expected Q5_K attention projection"
    N = blocks.size(0)
    torch.manual_seed(0)
    # rows must be > 1 for the prefill path; use 64 tokens.
    rows = 64
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16)
    grid = torch.empty(0, dtype=torch.int8, device="cuda")

    # Float baseline (gate OFF)
    env_backup = os.environ.get("GGUF_Q4K_Q5K_MMA")
    os.environ.pop("GGUF_Q4K_Q5K_MMA", None)
    y_float = cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid)

    # MMA path (gate ON)
    os.environ["GGUF_Q4K_Q5K_MMA"] = "1"
    y_mma = cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid)

    if env_backup is not None:
        os.environ["GGUF_Q4K_Q5K_MMA"] = env_backup
    else:
        os.environ.pop("GGUF_Q4K_Q5K_MMA", None)

    assert y_float.shape == y_mma.shape == (rows, N)

    abs_diff = (y_mma.to(torch.float32) - y_float.to(torch.float32)).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    # Relative error is only meaningful for non-tiny outputs; mask on |y|>0.1.
    mask = y_float.abs() > 0.1
    rel_err = abs_diff[mask] / (y_float[mask].abs().to(torch.float32) + 1e-8)
    p99_rel = float(torch.quantile(rel_err, 0.99).item()) if rel_err.numel() else 0.0

    assert torch.isfinite(y_mma).all(), "MMA output has non-finite values"
    assert y_mma.abs().sum() > 0.0, "MMA output is all-zero (kernel did not run)"
    # int8 activation quantization + int32 accumulate; Q5_K is 5-bit weights.
    assert max_abs < 0.5, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    assert mean_abs < 3.0e-2, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    assert p99_rel < 0.25, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    print(f"✓ Q5_K MMA vs float: max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}")


@pytest.mark.skipif(
    not (REAL_MINIMAX_PATH.exists() and _cuda_gguf_ext_available()),
    reason="real MiniMax-M2.7 GGUF bundle or CUDA extension not available",
)
def test_q4k_mma_vs_float():
    """Q4_K MMA prefill output matches float baseline within int8 tolerance."""
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    cuda_mod = load_cuda_kernel()

    # output_proj / lm_head / embeddings are Q4_K in MiniMax. Try the lm_head.
    blocks, K, type_id = _load_qk_weight_rows("output.weight", row_count=512)
    assert type_id == 3, "expected Q4_K"
    N = blocks.size(0)
    torch.manual_seed(1)
    rows = 64
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16)
    grid = torch.empty(0, dtype=torch.int8, device="cuda")

    env_backup = os.environ.get("GGUF_Q4K_Q5K_MMA")
    os.environ.pop("GGUF_Q4K_Q5K_MMA", None)
    y_float = cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid)

    os.environ["GGUF_Q4K_Q5K_MMA"] = "1"
    y_mma = cuda_mod.gguf_quant_gemm_prefill_forward(x, blocks, K, type_id, grid)

    if env_backup is not None:
        os.environ["GGUF_Q4K_Q5K_MMA"] = env_backup
    else:
        os.environ.pop("GGUF_Q4K_Q5K_MMA", None)

    assert y_float.shape == y_mma.shape == (rows, N)

    abs_diff = (y_mma.to(torch.float32) - y_float.to(torch.float32)).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mask = y_float.abs() > 0.1
    rel_err = abs_diff[mask] / (y_float[mask].abs().to(torch.float32) + 1e-8)
    p99_rel = float(torch.quantile(rel_err, 0.99).item()) if rel_err.numel() else 0.0

    assert torch.isfinite(y_mma).all(), "MMA output has non-finite values"
    assert y_mma.abs().sum() > 0.0, "MMA output is all-zero (kernel did not run)"
    assert max_abs < 0.6, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    assert mean_abs < 3.0e-2, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    assert p99_rel < 0.30, f"max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}"
    print(f"✓ Q4_K MMA vs float: max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}, p99_rel={p99_rel:.4e}")


@pytest.mark.skipif(True, reason="Q4_K/Q5_K DP4A decode kernel not wired up (Q5_K 5th-bit unpack WIP); float decode used")
def test_q5k_dp4a_decode_vs_float():
    """Placeholder: Q5_K DP4A decode (rows=1) — disabled, see gguf_mma_wrapper.cu and task #70 notes."""
    pass
