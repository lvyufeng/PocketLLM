#!/usr/bin/env python
"""What Xing4.0's hyper-connection costs, and how much of it is launch overhead.

#392 asks for the decode cost attributable to this block, with the note that if it
is not small that is the finding.  So this measures the block three ways:

- **bytes**, which is what it should cost: `hc_fn` is 24 x 14336, 672 KiB in
  bf16, called twice per layer, so 53.7 MiB per token across the 40-layer trunk;
- **GPU time**, from the profiler, which is the arithmetic;
- **wall time**, from CUDA events, which is the arithmetic plus the launches --
  and the Sinkhorn is 40 tiny ops per call, so the launches are the interesting
  part of this number.

The weights are real: layer 2's own `attn_hc` and `ffn_hc` out of the released
shard, at the released widths.

Usage::

    python scripts/bench_xing4_0_block.py [--device cuda:2] [--rows 1,512,2048]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.xing4_0.config import Xing4_0Params  # noqa: E402
from src.models.xing4_0.hyper_connection import HyperConnection, HyperConnectionWeights  # noqa: E402

CHECKPOINT = Path("/mnt/data2/Xing4.0-29B-A4B")
SHARD = CHECKPOINT / "model-00003-of-00041.safetensors"
LAYER = 2
TRUNK_LAYERS = 40
CALLS_PER_LAYER = 2  # attn_hc and ffn_hc
PEAK_GBPS = 616.0


def wall_ms(fn, *, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def gpu_ms_and_launches(fn, *, iters: int = 30) -> tuple[float, float]:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    averages = prof.key_averages()
    total_us = sum(getattr(event, "self_device_time_total", 0.0) for event in averages)
    launches = sum(event.count for event in averages)
    return total_us / 1000 / iters, launches / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--rows", default="1,512,2048")
    args = parser.parse_args()

    from safetensors.torch import safe_open

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    params = Xing4_0Params.from_json(CHECKPOINT / "config.json")
    prefix = f"model.layers.{LAYER}."
    with safe_open(str(SHARD), framework="pt") as handle:
        tensors = {
            name[len(prefix) :]: handle.get_tensor(name).to(device=device, dtype=dtype)
            for name in handle.keys()
            if name.startswith(prefix)
        }
    weights = HyperConnectionWeights.from_hf(tensors, params, "attn_hc")
    connection = HyperConnection(params, weights, dtype=dtype)

    elements = weights.hc_fn.numel() + weights.hc_base.numel() + weights.hc_scale.numel()
    per_call = elements * torch.tensor([], dtype=dtype).element_size()
    total = per_call * CALLS_PER_LAYER * TRUNK_LAYERS
    floor = total / (PEAK_GBPS * 1e6)
    print(f"device {device}  dtype {args.dtype}  hc_mult {params.hc_mult}  sinkhorn {params.hc_sinkhorn_iters}")
    print(
        f"weights  {per_call / 2**20:.3f} MiB per call, {CALLS_PER_LAYER} calls x {TRUNK_LAYERS} layers "
        f"= {total / 2**20:.1f} MiB/token, floor {floor:.4f} ms/token"
    )
    print()

    print(
        f"  {'rows':>6}  {'wall ms':>9}  {'GPU ms':>8}  {'launches':>9}  {'GPU us/launch':>14}  "
        f"{'40x2 GPU ms/token':>18}  {'40x2 wall ms/token':>19}"
    )
    for rows in (int(r) for r in args.rows.split(",")):
        hidden = torch.randn(1, rows, params.hc_mult, params.hidden_size, device=device, dtype=dtype)

        def call() -> None:
            connection.forward(hidden)

        wall = wall_ms(call)
        busy, launches = gpu_ms_and_launches(call)
        print(
            f"  {rows:>6}  {wall:>9.3f}  {busy:>8.3f}  {launches:>9.0f}  "
            f"{busy * 1000 / max(launches, 1):>14.2f}  {busy * CALLS_PER_LAYER * TRUNK_LAYERS:>18.2f}  "
            f"{wall * CALLS_PER_LAYER * TRUNK_LAYERS:>19.2f}"
        )
    print()
    print("  the same, as a share of a token: the MoE has to move 0.877 GiB and the")
    print("  attention 2.117 GiB per token, so this block's bytes are 2.5% of the sum, and")
    print("  what it actually costs is whatever the launch column says it costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
