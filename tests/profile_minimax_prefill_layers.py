"""Profile MiniMax-M2 prefill per-layer timing breakdown (256 tokens)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

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

    # Prefill 256 tokens
    seq_len = 256
    prompt = torch.tensor([list(range(1, seq_len + 1))], device=device, dtype=torch.long)

    # Warmup
    model.reset_cache(1, seq_len + 16)
    with torch.inference_mode():
        _ = model.forward(prompt, 0)
    sync()

    num_profile_layers = 5
    num_runs = 3
    all_runs = []

    for run_id in range(num_runs):
        model.reset_cache(1, seq_len + 16)
        results = []
        with torch.inference_mode():
            h = model.embedding(prompt).to(model.dtype)
            for layer_id in range(min(num_profile_layers, len(model.layers))):
                layer = model.layers[layer_id]

                # Attention
                sync()
                t0 = time.perf_counter()
                attn_out = layer.attention(layer.attn_norm(h), 0)
                sync()
                attn_ms = (time.perf_counter() - t0) * 1000
                h = (h + attn_out).to(layer.dtype)

                # MoE
                sync()
                t0 = time.perf_counter()
                moe_out = layer.moe(layer.ffn_norm(h))
                sync()
                moe_ms = (time.perf_counter() - t0) * 1000
                h = (h + moe_out).to(layer.dtype)

                results.append({"attn_ms": attn_ms, "moe_ms": moe_ms,
                                "total_ms": attn_ms + moe_ms})
        all_runs.append(results)

    if rank == 0:
        n_layers = len(model.layers)
        print(f"\n=== Prefill {seq_len} tok: Per-Layer Timing (avg {num_runs} runs, first {num_profile_layers} layers) ===")
        print(f"{'Layer':<6} {'Attn(ms)':<12} {'MoE(ms)':<12} {'Total(ms)':<12}")
        print("-" * 50)
        for layer_id in range(num_profile_layers):
            a = sum(r[layer_id]["attn_ms"] for r in all_runs) / num_runs
            m = sum(r[layer_id]["moe_ms"] for r in all_runs) / num_runs
            print(f"{layer_id:<6} {a:<12.2f} {m:<12.2f} {a+m:<12.2f}")
        avg_attn = sum(sum(r[i]["attn_ms"] for i in range(num_profile_layers)) for r in all_runs) / (num_runs * num_profile_layers)
        avg_moe = sum(sum(r[i]["moe_ms"] for i in range(num_profile_layers)) for r in all_runs) / (num_runs * num_profile_layers)
        print("-" * 50)
        print(f"{'AVG':<6} {avg_attn:<12.2f} {avg_moe:<12.2f} {avg_attn+avg_moe:<12.2f}")
        print(f"\nBreakdown: Attn={avg_attn/(avg_attn+avg_moe)*100:.1f}% MoE={avg_moe/(avg_attn+avg_moe)*100:.1f}%")
        full_ms = (avg_attn + avg_moe) * n_layers
        print(f"Estimated full-model prefill: {full_ms:.1f} ms ({seq_len}/{full_ms*1000:.2f} = {seq_len*1000/full_ms:.1f} TPS)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
