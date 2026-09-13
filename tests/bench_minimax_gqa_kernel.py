"""Benchmark GQA-aware decode attention kernels vs PyTorch baseline."""

import torch
import time
from src.kernels.cuda_loader import load_cuda_kernel


def benchmark_gqa_attention(B=1, n_heads=48, n_kv_heads=8, head_dim=128, T=1024, warmup=10, iters=100):
    """Benchmark full attention: Q·K^T + softmax + attn·V."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float16

    Q = torch.randn(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    K = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    V = torch.randn(B, n_kv_heads, T, head_dim, device=device, dtype=dtype)
    scale = 1.0 / (head_dim ** 0.5)

    cuda_ext = load_cuda_kernel()
    repeat = n_heads // n_kv_heads

    # Warmup
    for _ in range(warmup):
        K_expanded = K.repeat_interleave(repeat, dim=1)
        V_expanded = V.repeat_interleave(repeat, dim=1)
        _ = torch.nn.functional.scaled_dot_product_attention(
            Q, K_expanded, V_expanded, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )

    # Baseline: PyTorch SDPA with repeat_interleave
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        K_expanded = K.repeat_interleave(repeat, dim=1)
        V_expanded = V.repeat_interleave(repeat, dim=1)
        out_baseline = torch.nn.functional.scaled_dot_product_attention(
            Q, K_expanded, V_expanded, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    baseline_ms = (t1 - t0) / iters * 1000

    # Warmup GQA
    scores_gqa = torch.empty(B, n_heads, 1, T, device=device, dtype=dtype)
    out_gqa = torch.empty(B, n_heads, 1, head_dim, device=device, dtype=dtype)
    for _ in range(warmup):
        cuda_ext.gqa_decode_qk_gemv(Q, K, scores_gqa, n_heads, n_kv_heads, T, head_dim, scale)
        attn_weights_gqa = torch.softmax(scores_gqa, dim=-1)
        cuda_ext.gqa_decode_attn_v_gemv(attn_weights_gqa, V, out_gqa, n_heads, n_kv_heads, T, head_dim)

    # Custom GQA path
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        cuda_ext.gqa_decode_qk_gemv(Q, K, scores_gqa, n_heads, n_kv_heads, T, head_dim, scale)
        attn_weights_gqa = torch.softmax(scores_gqa, dim=-1)
        cuda_ext.gqa_decode_attn_v_gemv(attn_weights_gqa, V, out_gqa, n_heads, n_kv_heads, T, head_dim)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gqa_ms = (t1 - t0) / iters * 1000

    speedup = baseline_ms / gqa_ms
    print(f"T={T:5d}: Baseline={baseline_ms:.3f}ms, GQA={gqa_ms:.3f}ms, Speedup={speedup:.2f}×")


if __name__ == "__main__":
    print("Benchmarking GQA attention kernels (MiniMax decode scenario)")
    print("=" * 70)
    for T in [128, 512, 1024, 2048, 4096, 8192, 16384]:
        benchmark_gqa_attention(T=T)
    print("=" * 70)
