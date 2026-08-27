"""Python binding for INT8 per-token-head Triton attention to cpp_engine."""

import os
import sys
import torch
from typing import Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import Triton kernel
_TRITON_AVAILABLE = False
_TRITON_IMPORT_ERROR = None
try:
    from src.kernels.int8_per_token_head_triton import qwen_int8_per_token_head_decode_triton
    _TRITON_AVAILABLE = True
except Exception as e:
    _TRITON_IMPORT_ERROR = str(e)


def int8_per_token_head_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    context_len: int,
    use_triton: bool = True,
) -> torch.Tensor:
    """INT8 per-token-head decode attention with Triton optimization.

    Args:
        query: [q_heads, head_dim] FP16
        key_cache: [max_context, kv_heads, head_dim] INT8
        value_cache: [max_context, kv_heads, head_dim] INT8
        k_scale: [max_context, kv_heads] FP16 per-position per-head scales
        v_scale: [max_context, kv_heads] FP16 per-position per-head scales
        context_len: Actual context length
        use_triton: Use Triton kernel if available (default True)

    Returns:
        output: [q_heads, head_dim] FP16
    """
    q_heads, head_dim = query.shape
    max_context = key_cache.shape[0]
    output = torch.empty_like(query)

    if use_triton and _TRITON_AVAILABLE:
        # Allocate score scratch buffer
        score_scratch = torch.empty(
            (q_heads, max_context), dtype=torch.float32, device=query.device
        )
        qwen_int8_per_token_head_decode_triton(
            query, key_cache, value_cache, k_scale, v_scale, context_len, output, score_scratch
        )
        return output
    else:
        # Fallback to PyTorch implementation (slower)
        return _int8_decode_torch_fallback(
            query, key_cache, value_cache, k_scale, v_scale, context_len
        )


def _int8_decode_torch_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    context_len: int,
) -> torch.Tensor:
    """PyTorch fallback for INT8 decode (slow, for correctness reference)."""
    q_heads, head_dim = query.shape
    max_context, kv_heads, _ = key_cache.shape
    scale = 1.0 / (head_dim ** 0.5)

    output = torch.zeros_like(query)

    for q_head in range(q_heads):
        kv_head = q_head % kv_heads
        q = query[q_head]  # [head_dim]

        # Dequant K and compute scores
        k = key_cache[:context_len, kv_head, :].float() * k_scale[:context_len, kv_head, None]
        scores = torch.matmul(q.float(), k.t()) * scale  # [context_len]

        # Softmax
        scores = torch.softmax(scores, dim=0)

        # Dequant V and weighted sum
        v = value_cache[:context_len, kv_head, :].float() * v_scale[:context_len, kv_head, None]
        output[q_head] = torch.matmul(scores, v).half()

    return output


def test_int8_triton_correctness():
    """Test Triton kernel against PyTorch reference."""
    if not _TRITON_AVAILABLE:
        print(f"Triton not available: {_TRITON_IMPORT_ERROR}")
        return

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    q_heads, kv_heads, head_dim = 40, 8, 128
    max_context, context_len = 4096, 2048

    # Generate test data
    query = torch.randn(q_heads, head_dim, dtype=torch.float16, device=device)

    # INT8 cache with per-token-head scales
    key_fp16 = torch.randn(max_context, kv_heads, head_dim, dtype=torch.float16, device=device)
    value_fp16 = torch.randn(max_context, kv_heads, head_dim, dtype=torch.float16, device=device)

    k_scale = torch.rand(max_context, kv_heads, dtype=torch.float16, device=device) * 0.1
    v_scale = torch.rand(max_context, kv_heads, dtype=torch.float16, device=device) * 0.1

    # Quantize
    key_cache = (key_fp16 / k_scale[:, :, None]).clamp(-127, 127).round().to(torch.int8)
    value_cache = (value_fp16 / v_scale[:, :, None]).clamp(-127, 127).round().to(torch.int8)

    # Run both implementations
    output_torch = int8_per_token_head_decode_attention(
        query, key_cache, value_cache, k_scale, v_scale, context_len, use_triton=False
    )

    if _TRITON_AVAILABLE:
        output_triton = int8_per_token_head_decode_attention(
            query, key_cache, value_cache, k_scale, v_scale, context_len, use_triton=True
        )

        # Compare
        max_diff = (output_torch - output_triton).abs().max().item()
        mean_diff = (output_torch - output_triton).abs().mean().item()

        print(f"INT8 Triton correctness test:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  Output norm (torch): {output_torch.norm().item():.2f}")
        print(f"  Output norm (triton): {output_triton.norm().item():.2f}")

        if max_diff < 0.1:  # FP16 tolerance
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
    else:
        print("Triton not available, skipping correctness test")


if __name__ == "__main__":
    test_int8_triton_correctness()
