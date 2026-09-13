"""Profile MiniMax-M2 prefill attention internals (256 tokens)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.loader.gguf.bundle import read_gguf_bundle
from src.models.minimax_m2.spec import MiniMaxM2Spec


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        world = 1
        rank = 0
        local_rank = 0
    return world, rank, local_rank, torch.device("cuda", local_rank)


def main():
    world, rank, local_rank, device = setup_dist()

    gguf_path = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")
    bundle = read_gguf_bundle(gguf_path)
    spec = MiniMaxM2Spec()

    runtime = spec.build_token_runtime(
        bundle, world=world, rank=rank, device=device,
        dtype=torch.float16, n_layers=None, gpu_memory_gib=22.0,
    )
    model = runtime.model

    seq_len = 256
    prompt = torch.tensor([list(range(1, seq_len + 1))], device=device, dtype=torch.long)

    # Warmup
    model.reset_cache(1, seq_len + 16)
    with torch.inference_mode():
        _ = model.forward(prompt, 0)
    sync()

    layer = model.layers[0]
    attn = layer.attention
    x = model.embedding(prompt).to(model.dtype)

    num_runs = 5
    timings = {k: [] for k in ["qkv_proj", "qkv_norm", "rope", "cache_copy", "transpose",
                                "repeat", "sdpa", "o_proj", "total"]}

    with torch.inference_mode():
        for _ in range(num_runs):
            sync()
            t_total = time.perf_counter()

            # QKV projections (separate calls)
            t0 = time.perf_counter()
            q_raw = attn.q_proj(x)
            k_raw = attn.k_proj(x)
            v_raw = attn.v_proj(x)
            sync()
            timings["qkv_proj"].append(time.perf_counter() - t0)

            # Norms + reshape
            t0 = time.perf_counter()
            q = attn.q_norm(q_raw).view(1, seq_len, attn.args.n_heads, attn.args.head_dim)
            k = attn.k_norm(k_raw).view(1, seq_len, attn.args.n_kv_heads, attn.args.head_dim)
            v = v_raw.view(1, seq_len, attn.args.n_kv_heads, attn.args.head_dim).to(attn.dtype)
            sync()
            timings["qkv_norm"].append(time.perf_counter() - t0)

            # RoPE
            t0 = time.perf_counter()
            q = attn._apply_rope(q, 0)
            k = attn._apply_rope(k, 0)
            sync()
            timings["rope"].append(time.perf_counter() - t0)

            # Cache copy
            t0 = time.perf_counter()
            attn.cache_k[:1, :seq_len].copy_(k)
            attn.cache_v[:1, :seq_len].copy_(v)
            k_full = attn.cache_k[:1, :seq_len]
            v_full = attn.cache_v[:1, :seq_len]
            sync()
            timings["cache_copy"].append(time.perf_counter() - t0)

            # Transpose
            t0 = time.perf_counter()
            q_t = q.transpose(1, 2).contiguous()
            k_t = k_full.transpose(1, 2).contiguous()
            v_t = v_full.transpose(1, 2).contiguous()
            sync()
            timings["transpose"].append(time.perf_counter() - t0)

            # Repeat
            t0 = time.perf_counter()
            repeat = attn.args.n_heads // attn.args.n_kv_heads
            if repeat != 1:
                k_t = k_t.repeat_interleave(repeat, dim=1)
                v_t = v_t.repeat_interleave(repeat, dim=1)
            sync()
            timings["repeat"].append(time.perf_counter() - t0)

            # SDPA
            t0 = time.perf_counter()
            out = F.scaled_dot_product_attention(
                q_t.to(attn.dtype), k_t.to(attn.dtype), v_t.to(attn.dtype),
                attn_mask=None, dropout_p=0.0, is_causal=True,
                scale=1.0 / (attn.args.head_dim ** 0.5),
            )
            sync()
            timings["sdpa"].append(time.perf_counter() - t0)

            # Output reshape + o_proj
            t0 = time.perf_counter()
            out = out.transpose(1, 2).contiguous().view(1, seq_len, attn.args.n_heads * attn.args.head_dim)
            out = attn.o_proj(out)
            sync()
            timings["o_proj"].append(time.perf_counter() - t0)

            sync()
            timings["total"].append(time.perf_counter() - t_total)

    if rank == 0:
        avg_total = sum(timings["total"]) / num_runs * 1000
        print(f"\n=== Prefill {seq_len} tok: Attention Internals (layer 0, avg {num_runs} runs) ===")
        print(f"{'Component':<16} {'ms':<10} {'% of total':<12}")
        print("-" * 40)
        for key in ["qkv_proj", "qkv_norm", "rope", "cache_copy", "transpose",
                    "repeat", "sdpa", "o_proj"]:
            avg_ms = sum(timings[key]) / num_runs * 1000
            pct = avg_ms / avg_total * 100 if avg_total > 0 else 0
            print(f"{key:<16} {avg_ms:<10.3f} {pct:<12.1f}")
        print("-" * 40)
        print(f"{'TOTAL':<16} {avg_total:<10.3f} {100.0:<12.1f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
