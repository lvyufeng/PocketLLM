"""Test GQA-aware decode attention kernels vs PyTorch baseline."""

import torch
from src.kernels.cuda_loader import load_cuda_kernel


def test_gqa_qk_gemv():
    """Test Q·K^T kernel correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # MiniMax decode: B=1, n_heads=48, n_kv_heads=8, head_dim=128, context=9
    B, n_heads, n_kv_heads, head_dim = 1, 48, 8, 128
    T = 9  # context length

    # Random Q, K
    Q = torch.randn(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    K = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    scale = 1.0 / (head_dim ** 0.5)

    # Baseline: expand K via repeat_interleave
    repeat = n_heads // n_kv_heads
    K_expanded = K.repeat_interleave(repeat, dim=1)  # [B, n_heads, T, head_dim]
    scores_baseline = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale  # [B, n_heads, 1, T]

    # Custom GQA kernel
    cuda_ext = load_cuda_kernel()
    scores_gqa = torch.empty(B, n_heads, 1, T, device=device, dtype=dtype)
    cuda_ext.gqa_decode_qk_gemv(Q, K, scores_gqa, n_heads, n_kv_heads, T, head_dim, scale)

    # Compare
    max_diff = (scores_baseline - scores_gqa).abs().max().item()
    mean_diff = (scores_baseline - scores_gqa).abs().mean().item()

    print(f"\nQ·K^T: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"Baseline scores[0,0,0,:5]: {scores_baseline[0,0,0,:5].float().cpu().numpy()}")
    print(f"GQA scores[0,0,0,:5]:      {scores_gqa[0,0,0,:5].float().cpu().numpy()}")

    assert max_diff < 1e-2, f"GQA Q·K^T mismatch: max_diff={max_diff}"
    print("✓ GQA Q·K^T matches baseline")


def test_gqa_attn_v_gemv():
    """Test attn·V kernel correctness."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float16

    B, n_heads, n_kv_heads, head_dim = 1, 48, 8, 128
    T = 9

    # Random attn_weights, V
    attn_weights = torch.randn(B, n_heads, 1, T, device=device, dtype=dtype)
    attn_weights = torch.softmax(attn_weights, dim=-1)  # normalize
    V = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)

    # Baseline: expand V via repeat_interleave
    repeat = n_heads // n_kv_heads
    V_expanded = V.repeat_interleave(repeat, dim=1)  # [B, n_heads, T, head_dim]
    out_baseline = torch.matmul(attn_weights, V_expanded)  # [B, n_heads, 1, head_dim]

    # Custom GQA kernel
    cuda_ext = load_cuda_kernel()
    out_gqa = torch.empty(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    cuda_ext.gqa_decode_attn_v_gemv(attn_weights, V, out_gqa, n_heads, n_kv_heads, T, head_dim)

    # Compare
    max_diff = (out_baseline - out_gqa).abs().max().item()
    mean_diff = (out_baseline - out_gqa).abs().mean().item()

    print(f"\nattn·V: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"Baseline out[0,0,0,:5]: {out_baseline[0,0,0,:5].float().cpu().numpy()}")
    print(f"GQA out[0,0,0,:5]:      {out_gqa[0,0,0,:5].float().cpu().numpy()}")

    assert max_diff < 1e-2, f"GQA attn·V mismatch: max_diff={max_diff}"
    print("✓ GQA attn·V matches baseline")


def test_gqa_full_attention():
    """Test full attention path (Q·K^T + softmax + attn·V) vs PyTorch baseline."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float16

    B, n_heads, n_kv_heads, head_dim = 1, 48, 8, 128
    T = 9

    Q = torch.randn(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    K = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    V = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    scale = 1.0 / (head_dim ** 0.5)

    # Baseline: PyTorch SDPA with repeat_interleave
    repeat = n_heads // n_kv_heads
    K_expanded = K.repeat_interleave(repeat, dim=1)
    V_expanded = V.repeat_interleave(repeat, dim=1)
    out_baseline = torch.nn.functional.scaled_dot_product_attention(
        Q, K_expanded, V_expanded, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )

    # Custom GQA path
    cuda_ext = load_cuda_kernel()
    scores_gqa = torch.empty(B, n_heads, 1, T, device=device, dtype=dtype)
    cuda_ext.gqa_decode_qk_gemv(Q, K, scores_gqa, n_heads, n_kv_heads, T, head_dim, scale)
    attn_weights_gqa = torch.softmax(scores_gqa, dim=-1)
    out_gqa = torch.empty(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    cuda_ext.gqa_decode_attn_v_gemv(attn_weights_gqa, V, out_gqa, n_heads, n_kv_heads, T, head_dim)

    # Compare
    max_diff = (out_baseline - out_gqa).abs().max().item()
    mean_diff = (out_baseline - out_gqa).abs().mean().item()

    print(f"\nFull attention: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"Baseline out[0,0,0,:10]: {out_baseline[0,0,0,:10].float().cpu().numpy()}")
    print(f"GQA out[0,0,0,:10]:      {out_gqa[0,0,0,:10].float().cpu().numpy()}")

    assert max_diff < 1e-2, f"GQA full attention mismatch: max_diff={max_diff}"
    print("✓ GQA full attention matches baseline")


if __name__ == "__main__":
    test_gqa_qk_gemv()
    test_gqa_attn_v_gemv()
    test_gqa_full_attention()
    print("\n✅ All GQA kernel tests passed")
