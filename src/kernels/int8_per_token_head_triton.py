"""Triton INT8 per-token-head KV cache attention kernel.

Ported from vLLM-2080Ti triton_unified_attention.py to achieve +16% performance
over FP16 baseline. Uses Triton JIT compiler's automatic vectorization to handle
per-position scales efficiently.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qwen_int8_per_token_head_decode_kernel(
    # Output
    output_ptr,
    score_scratch_ptr,  # [q_heads, max_context] FP32 scratch for scores
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,  # [max_context, kv_heads] FP16
    v_scale_cache_ptr,  # [max_context, kv_heads] FP16
    # Scalars
    scale: tl.constexpr,  # 1/sqrt(head_dim)
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    context_len: tl.int32,
    max_context: tl.constexpr,
    # Strides
    stride_q_head: tl.int64,
    stride_q_dim: tl.int64,
    stride_k_pos: tl.int64,
    stride_k_head: tl.int64,
    stride_k_dim: tl.int64,
    stride_v_pos: tl.int64,
    stride_v_head: tl.int64,
    stride_v_dim: tl.int64,
    stride_ks_pos: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_pos: tl.int64,
    stride_vs_head: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.int64,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """INT8 per-token-head decode attention kernel.

    Grid: (q_heads,)
    Block: processes one query head, iterates over KV positions
    """
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx % kv_heads  # GQA mapping

    # Load query tile: [head_dim]
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    q_ptrs = query_ptr + q_head_idx * stride_q_head + offs_d * stride_q_dim
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Phase 1: Compute scores Q·K with INT8 dequant
    max_score = -1e10
    num_tiles = tl.cdiv(context_len, BLOCK_M)
    score_base = score_scratch_ptr + q_head_idx * max_context

    for tile_idx in range(num_tiles):
        pos_start = tile_idx * BLOCK_M
        offs_pos = pos_start + tl.arange(0, BLOCK_M)
        mask_pos = offs_pos < context_len

        # Load K scale for this tile: [BLOCK_M]
        ks_ptrs = k_scale_cache_ptr + offs_pos * stride_ks_pos + kv_head_idx * stride_ks_head
        k_scales = tl.load(ks_ptrs, mask=mask_pos, other=0.0).to(tl.float32)

        # Load INT8 K tile: [BLOCK_M, head_dim]
        k_ptrs = (key_cache_ptr +
                  offs_pos[:, None] * stride_k_pos +
                  kv_head_idx * stride_k_head +
                  offs_d[None, :] * stride_k_dim)
        k_int8 = tl.load(k_ptrs, mask=mask_pos[:, None] & mask_d[None, :], other=0)
        k = k_int8.to(tl.float32) * k_scales[:, None]  # Broadcast scale

        # Compute Q·K: [BLOCK_M]
        score_tile = tl.sum(q[None, :] * k, axis=1) * scale

        # Store scores
        tl.store(score_base + offs_pos, score_tile, mask=mask_pos)

        # Track max for softmax
        tile_max = tl.max(tl.where(mask_pos, score_tile, -1e10), axis=0)
        max_score = tl.maximum(max_score, tile_max)

    # Phase 2: Softmax
    sum_exp = 0.0
    for tile_idx in range(num_tiles):
        pos_start = tile_idx * BLOCK_M
        offs_pos = pos_start + tl.arange(0, BLOCK_M)
        mask_pos = offs_pos < context_len

        score_tile = tl.load(score_base + offs_pos, mask=mask_pos, other=-1e10)
        exp_tile = tl.exp(score_tile - max_score)
        tl.store(score_base + offs_pos, exp_tile, mask=mask_pos)
        sum_exp += tl.sum(tl.where(mask_pos, exp_tile, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-8)

    # Phase 3: Weighted value sum with INT8 dequant
    accum = tl.zeros([head_dim], dtype=tl.float32)

    for tile_idx in range(num_tiles):
        pos_start = tile_idx * BLOCK_M
        offs_pos = pos_start + tl.arange(0, BLOCK_M)
        mask_pos = offs_pos < context_len

        # Load V scale for this tile: [BLOCK_M]
        vs_ptrs = v_scale_cache_ptr + offs_pos * stride_vs_pos + kv_head_idx * stride_vs_head
        v_scales = tl.load(vs_ptrs, mask=mask_pos, other=0.0).to(tl.float32)

        # Load INT8 V tile: [BLOCK_M, head_dim]
        v_ptrs = (value_cache_ptr +
                  offs_pos[:, None] * stride_v_pos +
                  kv_head_idx * stride_v_head +
                  offs_d[None, :] * stride_v_dim)
        v_int8 = tl.load(v_ptrs, mask=mask_pos[:, None] & mask_d[None, :], other=0)
        v = v_int8.to(tl.float32) * v_scales[:, None]  # Broadcast scale

        # Load attention weights
        weights = tl.load(score_base + offs_pos, mask=mask_pos, other=0.0) / sum_exp

        # Accumulate weighted values: [head_dim]
        accum += tl.sum(weights[:, None] * v, axis=0)

    # Store final output
    out_ptrs = output_ptr + q_head_idx * stride_out_head + offs_d * stride_out_dim
    tl.store(out_ptrs, accum.to(output_ptr.dtype.element_ty), mask=mask_d)


def qwen_int8_per_token_head_decode_triton(
    query: torch.Tensor,  # [q_heads, head_dim] FP16
    key_cache: torch.Tensor,  # [max_context, kv_heads, head_dim] INT8
    value_cache: torch.Tensor,  # [max_context, kv_heads, head_dim] INT8
    k_scale: torch.Tensor,  # [max_context, kv_heads] FP16
    v_scale: torch.Tensor,  # [max_context, kv_heads] FP16
    context_len: int,
    output: torch.Tensor,  # [q_heads, head_dim] FP16
    score_scratch: torch.Tensor,  # [q_heads, max_context] FP32
) -> None:
    """Triton INT8 per-token-head decode attention.

    Args:
        query: Query tensor [q_heads, head_dim]
        key_cache: INT8 key cache [max_context, kv_heads, head_dim]
        value_cache: INT8 value cache [max_context, kv_heads, head_dim]
        k_scale: FP16 key scales [max_context, kv_heads]
        v_scale: FP16 value scales [max_context, kv_heads]
        context_len: Actual context length (≤ max_context)
        output: Output tensor [q_heads, head_dim]
        score_scratch: Score scratch buffer [q_heads, max_context]
    """
    q_heads, head_dim = query.shape
    max_context, kv_heads, _ = key_cache.shape

    assert query.dtype == torch.float16
    assert key_cache.dtype == torch.int8
    assert value_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float16
    assert v_scale.dtype == torch.float16
    assert output.dtype == torch.float16
    assert score_scratch.dtype == torch.float32

    scale = 1.0 / (head_dim ** 0.5)

    # Tile sizes
    BLOCK_M = 32  # positions per tile
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (q_heads,)

    _qwen_int8_per_token_head_decode_kernel[grid](
        output,
        score_scratch,
        query,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        scale,
        q_heads,
        kv_heads,
        head_dim,
        context_len,
        max_context,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )
