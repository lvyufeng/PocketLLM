"""Profile MiniMax-M2 decode per-layer timing breakdown.

Measures:
- Attention compute (replicated, no communication)
- MoE EP compute (local expert computation, no all_reduce)
- All-reduce communication
- Python overhead

Run with TP4:
    torchrun --standalone --nproc_per_node=4 tests/profile_minimax_decode_layers.py
"""

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


def profile_decode_single_step(model, x_in: torch.Tensor, start_pos: int, num_layers: int = 5):
    """Profile first `num_layers` decode layers, breaking down timing."""
    results = []
    x = x_in.clone()

    for layer_id in range(min(num_layers, len(model.layers))):
        layer = model.layers[layer_id]

        # === Attention ===
        sync()
        t0 = time.perf_counter()
        attn_in = layer.attn_norm(x)
        attn_out = layer.attention(attn_in, start_pos)
        sync()
        attn_ms = (time.perf_counter() - t0) * 1000

        x = (x + attn_out).to(layer.dtype)

        # === MoE: split compute and all_reduce ===
        moe_in = layer.ffn_norm(x)
        moe_in_flat = moe_in.reshape(-1, moe_in.shape[-1]).contiguous()

        # Route (CPU, negligible)
        indices, weights = layer.moe.route(moe_in_flat)

        # MoE EP compute (local experts only)
        sync()
        t0 = time.perf_counter()

        moe_layer = layer.moe.cache.layer(layer.moe.layer_id)
        route_slots = indices[0, :].contiguous()
        route_weights_1d = weights[0, :].contiguous()

        expert_start = int(layer.moe.cache.expert_start)
        expert_end = expert_start + int(layer.moe.cache.expert_count)
        local_mask = (route_slots >= expert_start) & (route_slots < expert_end)
        local_slots = route_slots[local_mask] - expert_start
        local_weights = route_weights_1d[local_mask]

        if local_slots.numel() == 0:
            y = torch.zeros((1, layer.moe.args.dim), device=moe_in.device, dtype=torch.float32)
        else:
            grid = layer.moe.cache._quant_grid(moe_layer.w1.type_name)
            y = layer.moe._cuda.gguf_moe_single_token_iq2_q2k_forward(
                moe_in_flat,
                local_slots,
                local_weights,
                moe_layer.w1.blocks,
                moe_layer.w3.blocks,
                moe_layer.w2.blocks,
                grid,
                0.0,
            )
        sync()
        moe_compute_ms = (time.perf_counter() - t0) * 1000

        # === All-reduce ===
        sync()
        t0 = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(y)
        sync()
        allreduce_ms = (time.perf_counter() - t0) * 1000

        x = (x + y.reshape(moe_in.shape).to(layer.dtype)).to(layer.dtype)

        results.append({
            "layer": layer_id,
            "attn_ms": attn_ms,
            "moe_compute_ms": moe_compute_ms,
            "allreduce_ms": allreduce_ms,
            "total_ms": attn_ms + moe_compute_ms + allreduce_ms,
        })

    return results, x


def main():
    world, rank, local_rank, device = setup_dist()

    gguf_path = Path("/mnt/data1/dsv4_inference/gguf_hfd/MiniMax-M2.7-GGUF/UD-IQ1_M")
    if not gguf_path.exists():
        if rank == 0:
            print(f"GGUF path not found: {gguf_path}")
        return

    if rank == 0:
        print(f"profile_start world={world} path={gguf_path}", flush=True)

    bundle = read_gguf_bundle(gguf_path)
    spec = MiniMaxM2Spec()

    runtime = spec.build_token_runtime(
        bundle,
        world=world,
        rank=rank,
        device=device,
        dtype=torch.float16,
        n_layers=None,
        gpu_memory_gib=22.0,
    )

    model = runtime.model

    # Warm-up: run one full forward pass
    if rank == 0:
        print("warmup_start", flush=True)

    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device, dtype=torch.long)
    model.reset_cache(1, 512)
    with torch.inference_mode():
        _ = model.forward(prompt, 0, return_next_token=True)
        # First decode step
        _ = model.forward(torch.tensor([[9]], device=device), len(prompt[0]), return_next_token=True)

    sync()
    if rank == 0:
        print("warmup_done", flush=True)

    # Profile decode: 5 layers, 3 runs
    num_profile_layers = 5
    num_runs = 3

    all_runs = []
    for run_id in range(num_runs):
        # Reset and prefill
        model.reset_cache(1, 512)
        with torch.inference_mode():
            h = model.embedding(prompt).to(model.dtype)
            for layer in model.layers:
                h = layer(h, 0)
            pos = len(prompt[0])

            # Decode step with detailed profiling
            inp = torch.tensor([[9]], device=device, dtype=torch.long)
            h_decode = model.embedding(inp).to(model.dtype)

            results, _ = profile_decode_single_step(model, h_decode, pos, num_profile_layers)
            all_runs.append(results)

    # Aggregate and print
    if rank == 0:
        print("\n=== Per-Layer Decode Timing (averaged over 3 runs) ===")
        print(f"{'Layer':<6} {'Attn(ms)':<12} {'MoE Compute(ms)':<18} {'All-Reduce(ms)':<16} {'Total(ms)':<12}")
        print("-" * 70)

        for layer_id in range(num_profile_layers):
            attn_vals = [run[layer_id]["attn_ms"] for run in all_runs]
            moe_vals = [run[layer_id]["moe_compute_ms"] for run in all_runs]
            ar_vals = [run[layer_id]["allreduce_ms"] for run in all_runs]
            total_vals = [run[layer_id]["total_ms"] for run in all_runs]

            print(f"{layer_id:<6} {sum(attn_vals)/len(attn_vals):<12.2f} "
                  f"{sum(moe_vals)/len(moe_vals):<18.2f} "
                  f"{sum(ar_vals)/len(ar_vals):<16.2f} "
                  f"{sum(total_vals)/len(total_vals):<12.2f}")

        # Summary
        avg_attn = sum(sum(r[i]["attn_ms"] for i in range(num_profile_layers)) for r in all_runs) / (num_runs * num_profile_layers)
        avg_moe = sum(sum(r[i]["moe_compute_ms"] for i in range(num_profile_layers)) for r in all_runs) / (num_runs * num_profile_layers)
        avg_ar = sum(sum(r[i]["allreduce_ms"] for i in range(num_profile_layers)) for r in all_runs) / (num_runs * num_profile_layers)
        avg_total = avg_attn + avg_moe + avg_ar

        print("-" * 70)
        print(f"{'AVG':<6} {avg_attn:<12.2f} {avg_moe:<18.2f} {avg_ar:<16.2f} {avg_total:<12.2f}")
        print()
        print(f"Breakdown: Attn={avg_attn/avg_total*100:.1f}% MoE={avg_moe/avg_total*100:.1f}% AllReduce={avg_ar/avg_total*100:.1f}%")
        print(f"Estimated full-model decode time: {avg_total * 62:.1f} ms/token ({1000/(avg_total*62):.2f} TPS)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
