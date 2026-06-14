"""Test MiniMax fused half-split RoPE kernel correctness."""

import torch
from src.kernels.cuda_loader import load_cuda_kernel


def test_fused_rope_vs_pytorch_gqa():
    """Compare fused RoPE kernel with PyTorch reference (GQA: different head counts for q/k)."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Test both q (48 heads) and k (8 heads)
    for name, H in [("q", 48), ("k", 8)]:
        B, S = 1, 1
        D = 128
        rope_dim = 128
        rope_base = 10000.0
        start_pos = 8

        # Input tensor
        x_ref = torch.randn(B, S, H, D, device=device, dtype=dtype)
        x_fused = x_ref.clone().contiguous()

        # Pre-compute freqs
        half = rope_dim // 2
        positions = torch.arange(start_pos, start_pos + S, device=device, dtype=torch.float32)
        j = torch.arange(half, device=device, dtype=torch.float32)
        inv = torch.pow(torch.full_like(j, float(rope_base)), -(2.0 * j) / float(rope_dim))
        freqs = positions[:, None] * inv[None, :]  # [S, half]
        freqs_cos = torch.cos(freqs).contiguous()
        freqs_sin = torch.sin(freqs).contiguous()

        # PyTorch reference
        sin_ref = freqs_sin.to(dtype)[None, :, None, :]
        cos_ref = freqs_cos.to(dtype)[None, :, None, :]
        x1 = x_ref[..., :half]
        x2 = x_ref[..., half:rope_dim]
        y1 = x1 * cos_ref - x2 * sin_ref
        y2 = x1 * sin_ref + x2 * cos_ref
        if rope_dim < D:
            y_ref = torch.cat((y1, y2, x_ref[..., rope_dim:]), dim=-1)
        else:
            y_ref = torch.cat((y1, y2), dim=-1)

        # Fused kernel
        cuda_ext = load_cuda_kernel()
        cuda_ext.fused_minimax_rope_halfsplit_inplace(x_fused, freqs_cos, freqs_sin)

        # Compare
        max_diff = (y_ref - x_fused).abs().max().item()
        mean_diff = (y_ref - x_fused).abs().mean().item()

        print(f"\n[{name} H={H}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # Detailed comparison for first few elements of first head
        print(f"First 10 elements of rope result (head 0):")
        print(f"PyTorch:   {y_ref[0, 0, 0, :10].float().cpu().numpy()}")
        print(f"Fused:     {x_fused[0, 0, 0, :10].float().cpu().numpy()}")
        print(f"Diff:      {(y_ref - x_fused)[0, 0, 0, :10].float().abs().cpu().numpy()}")

        assert max_diff < 1e-2, f"Fused RoPE mismatch for {name}: max_diff={max_diff}"
        print(f"✓ Fused RoPE matches PyTorch reference for {name}")


if __name__ == "__main__":
    test_fused_rope_vs_pytorch_gqa()
