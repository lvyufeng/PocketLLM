"""
Test fused decode GQA attention against real MiniMaxAttention forward pass.
"""
import torch
import torch.nn.functional as F
from src.kernels.cuda_loader import load_cuda_kernel

def test_against_minimax_attention():
    """Compare fused kernel against the exact ops MiniMaxAttention.__call__ does."""
    torch.manual_seed(789)

    # MiniMax-M2 config
    n_heads = 48
    n_kv_heads = 8
    head_dim = 128
    repeat = n_heads // n_kv_heads
    kv_len = 50
    max_seq = 256

    # Inputs (all in MiniMaxAttention layout)
    q = torch.randn(1, n_heads, 1, head_dim, device='cuda', dtype=torch.float16)
    k_new = torch.randn(1, 1, n_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    v_new = torch.randn(1, 1, n_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    cache_k = torch.randn(1, max_seq, n_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    cache_v = torch.randn(1, max_seq, n_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    start_pos = kv_len - 1

    # PyTorch reference (exact MiniMaxAttention.__call__ ops for decode)
    cache_k_ref = cache_k.clone()
    cache_v_ref = cache_v.clone()
    # k_new/v_new: [B, 1, Hkv, D], cache: [B, max_seq, Hkv, D]
    cache_k_ref[:, start_pos:start_pos+1, :, :] = k_new
    cache_v_ref[:, start_pos:start_pos+1, :, :] = v_new

    k_full = cache_k_ref[:, :kv_len]  # [B, T, Hkv, D]
    v_full = cache_v_ref[:, :kv_len]

    q_t = q  # already [B, Hq, S=1, D]
    k_t = k_full.transpose(1, 2).contiguous()  # [B, Hkv, T, D]
    v_t = v_full.transpose(1, 2).contiguous()
    k_t = k_t.repeat_interleave(repeat, dim=1)  # [B, Hq, T, D]
    v_t = v_t.repeat_interleave(repeat, dim=1)

    sm_scale = 1.0 / (head_dim ** 0.5)
    expected = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=None, dropout_p=0.0, is_causal=False,
        scale=sm_scale
    )  # [B, Hq, 1, D]

    # Fused kernel
    cuda_mod = load_cuda_kernel()
    out = cuda_mod.fused_decode_gqa_attention(q, k_new, v_new, cache_k, cache_v, start_pos, sm_scale)

    # Compare
    diff = (expected - out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, kv_len={kv_len}")
    print(f"Max diff:  {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    if max_diff > 0.05:
        print("\n❌ FAILED: diff too large")
        # Debug which heads fail
        for h in range(0, n_heads, 6):
            h_diff = diff[0, h, 0].max().item()
            if h_diff > 0.05:
                print(f"  Head {h} (kv_h={h//repeat}): max_diff={h_diff:.3f}")
                print(f"    Expected[:5]: {expected[0, h, 0, :5].tolist()}")
                print(f"    Kernel[:5]:   {out[0, h, 0, :5].tolist()}")
        return False
    else:
        print("✅ PASSED")
        return True

if __name__ == "__main__":
    test_against_minimax_attention()
