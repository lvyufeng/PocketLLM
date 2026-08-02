"""GLM-DSA RMSNorm fused-CUDA path must match the PyTorch reference.

GLM decode calls RMSNorm ~4x per layer x 78 layers; the fused kernel replaces
the fp32 pow/mean/rsqrt/mul chain (many tiny launches) with one block/token.
This asserts the fused output matches the reference math within fp16 tolerance
for the norm dims GLM actually uses (hidden, q/kv lora ranks).
"""
from __future__ import annotations

import pytest
import torch

from src.models.glm_dsa.architecture import RMSNorm


def _cuda_rmsnorm_available() -> bool:
    if not torch.cuda.is_available():
        return False
    from src.kernels.cuda_loader import load_cuda_kernel

    mod = load_cuda_kernel()
    return mod is not None and hasattr(mod, "fused_rms_norm_forward")


@pytest.mark.skipif(not _cuda_rmsnorm_available(), reason="CUDA fused_rms_norm not available")
@pytest.mark.parametrize("dim", [5120, 1536, 512, 4096])
@pytest.mark.parametrize("rows", [1, 13])
def test_glm_fused_rmsnorm_matches_reference(dim: int, rows: int, monkeypatch) -> None:
    # The fused path is opt-in (neutral for GLM decode); enable it for the test.
    monkeypatch.setenv("GLM_FUSED_RMSNORM", "1")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    weight = torch.randn(dim, device=device)
    eps = 1.0e-5
    norm = RMSNorm(weight, eps, out_dtype=torch.float16)

    x = torch.randn(1, rows, dim, device=device, dtype=torch.float16)
    y_fused = norm(x)

    # Reference (bypass the fused path explicitly).
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (xf * inv * weight.float()).to(torch.float16)

    assert y_fused.shape == y_ref.shape
    assert torch.isfinite(y_fused).all()
    abs_diff = (y_fused.float() - y_ref.float()).abs()
    max_abs = float(abs_diff.max().item())
    # fp16 rounding of a normalized value; a couple ULPs is expected.
    assert max_abs < 5.0e-3, f"dim={dim} rows={rows} max_abs={max_abs:.4e}"
