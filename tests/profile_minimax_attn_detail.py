"""Profile MiniMax-M2 attention internals during decode (single token)."""

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

    if rank == 0:
        print(f"n_heads={model.args.n_heads} n_kv_heads={model.args.n_kv_heads} "
              f"head_dim={model.args.head_dim} dim={model.args.dim} "
              f"repeat={model.args.n_heads // model.args.n_kv_heads}", flush=True)

    # Warmup
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device, dtype=torch.long)
    model.reset_cache(1, 512)
    with torch.inference_mode():
        _ = model.forward(prompt, 0, return_next_token=True)
        _ = model.forward(torch.tensor([[9]], device=device), len(prompt[0]), return_next_token=True)
    sync()

    # Profile attention internals for layer 0, decode step
    layer = model.layers[0]
    attn = layer.attention
    start_pos = len(prompt[0])  # 8
    end_pos = start_pos + 1

    # Prepare x
    inp = torch.tensor([[9]], device=device, dtype=torch.long)
    x = model.embedding(inp).to(model.dtype)

    num_runs = 10
    timings = {k: [] for k in ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
                                "rope_q", "rope_k", "cache_copy", "transpose",
                                "repeat", "sdpa", "o_transpose", "o_proj", "total"]}

    for _ in range(num_runs):
        sync()
        t_total = time.perf_counter()

        # q_proj
        t0 = time.perf_counter()
        q_raw = attn.q_proj(x)
        sync()
        timings["q_proj"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        q = attn.q_norm(q_raw).view(1, 1, attn.args.n_heads, attn.args.head_dim)
        sync()
        timings["q_norm"].append(time.perf_counter() - t0)

        # k_proj, v_proj
        t0 = time.perf_counter()
        k_raw = attn.k_proj(x)
        v_raw = attn.v_proj(x)
        sync()
        timings["k_proj"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        k = attn.k_norm(k_raw).view(1, 1, attn.args.n_kv_heads, attn.args.head_dim)
        v = v_raw.view(1, 1, attn.args.n_kv_heads, attn.args.head_dim).to(attn.dtype)
        sync()
        timings["k_norm"].append(time.perf_counter() - t0)

        # RoPE
        t0 = time.perf_counter()
        q = attn._apply_rope(q, start_pos)
        k = attn._apply_rope(k, start_pos)
        sync()
        timings["rope_q"].append(time.perf_counter() - t0)

        # Cache copy
        t0 = time.perf_counter()
        attn.cache_k[:1, start_pos:end_pos].copy_(k)
        attn.cache_v[:1, start_pos:end_pos].copy_(v)
        k_full = attn.cache_k[:1, :end_pos]
        v_full = attn.cache_v[:1, :end_pos]
        sync()
        timings["cache_copy"].append(time.perf_counter() - t0)

        # Transpose
        t0 = time.perf_counter()
        q_t = q.transpose(1, 2).contiguous()
        k_t = k_full.transpose(1, 2).contiguous()
        v_t = v_full.transpose(1, 2).contiguous()
        sync()
        timings["transpose"].append(time.perf_counter() - t0)

        # Repeat (GQA expansion)
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
            attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=1.0 / (attn.args.head_dim ** 0.5),
        )
        sync()
        timings["sdpa"].append(time.perf_counter() - t0)

        # Output transpose
        t0 = time.perf_counter()
        out = out.transpose(1, 2).contiguous().view(1, 1, attn.args.n_heads * attn.args.head_dim)
        sync()
        timings["o_transpose"].append(time.perf_counter() - t0)

        # o_proj
        t0 = time.perf_counter()
        out = attn.o_proj(out)
        sync()
        timings["o_proj"].append(time.perf_counter() - t0)

        sync()
        timings["total"].append(time.perf_counter() - t_total)

    if rank == 0:
        print(f"\n=== Attention Internal Timing (layer 0, decode, avg {num_runs} runs) ===")
        print(f"{'Component':<16} {'ms':<10} {'% of total':<12}")
        print("-" * 40)
        avg_total = sum(timings["total"]) / num_runs * 1000
        for key in ["q_proj", "q_norm", "k_proj", "k_norm", "rope_q",
                    "cache_copy", "transpose", "repeat", "sdpa", "o_transpose", "o_proj"]:
            avg_ms = sum(timings[key]) / num_runs * 1000
            pct = avg_ms / avg_total * 100 if avg_total > 0 else 0
            print(f"{key:<16} {avg_ms:<10.3f} {pct:<12.1f}")
        print("-" * 40)
        print(f"{'TOTAL':<16} {avg_total:<10.3f} {100.0:<12.1f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
