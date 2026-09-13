"""Test PyTorch SDPA native GQA support vs manual repeat_interleave."""

import torch

torch.manual_seed(42)
device = torch.device("cuda:0")
dtype = torch.float16

# MiniMax decode scenario: B=1, S=1, n_heads=48, n_kv_heads=8, head_dim=128, context=9
B, S_q, S_kv = 1, 1, 9
n_heads, n_kv_heads, head_dim = 48, 8, 128
repeat = n_heads // n_kv_heads  # 6

# Random Q, K, V
q = torch.randn(B, n_heads, S_q, head_dim, device=device, dtype=dtype)
k = torch.randn(B, n_kv_heads, S_kv, head_dim, device=device, dtype=dtype)
v = torch.randn(B, n_kv_heads, S_kv, head_dim, device=device, dtype=dtype)

scale = 1.0 / (head_dim ** 0.5)

# Method 1: Native GQA (enable_gqa=True)
out_gqa = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True
)

# Method 2: Manual repeat_interleave (current MiniMax implementation)
k_repeated = k.repeat_interleave(repeat, dim=1)
v_repeated = v.repeat_interleave(repeat, dim=1)
out_repeat = torch.nn.functional.scaled_dot_product_attention(
    q, k_repeated, v_repeated, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
)

# Compare
max_diff = (out_gqa - out_repeat).abs().max().item()
mean_diff = (out_gqa - out_repeat).abs().mean().item()

print(f"max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
print(f"out_gqa[0,0,0,:10]:   {out_gqa[0,0,0,:10].float().cpu().numpy()}")
print(f"out_repeat[0,0,0,:10]: {out_repeat[0,0,0,:10].float().cpu().numpy()}")

# Check if results are identical (or within fp16 epsilon)
if max_diff < 1e-5:
    print("✓ Native GQA matches repeat_interleave (bit-exact or fp16 epsilon)")
else:
    print(f"⚠ Difference {max_diff:.6f} may affect token parity")
