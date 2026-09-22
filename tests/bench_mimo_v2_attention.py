#!/usr/bin/env python
"""What one layer's attention costs on a card, prefill and decode, per family.

This is a baseline, not a result: the attention is torch with an online softmax and
no kernel, so what it establishes is what the *shape* of the work costs. Two things
make the shape interesting.

A sliding-window layer reads a window, not a prefix, so its prefill should not grow
with the context the way a global layer's does. The block loop gets that from the
query bounds -- a key block is answered by a slice of the query rows -- and the
number it prints is the blocks it actually visited against the blocks a full walk
would have visited, which is the ratio the skipping is worth.

A global layer is quadratic and honestly is. At 256k it is also the reason the cache
exists: a decode step reads 256k keys once, and a prefill chunk reads them for every
chunk of queries.

Usage:

    python tests/bench_mimo_v2_attention.py                  # the default sweep
    python tests/bench_mimo_v2_attention.py --seqs 4096 --chunks 512 1024
    python tests/bench_mimo_v2_attention.py --decode 4096 32768 262144

The checkpoint is the default asset path; without it the script exits 0 and says so.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.device_attention import (  # noqa: E402
    MimoV2DeviceAttention,
    MimoV2KVCache,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
#: One layer of each family: 5 is a global layer with no sink, 2 is a windowed one.
LAYERS = (2, 5)
#: The whole model: 39 windowed layers and 9 global ones.
FAMILY_COUNTS = {"swa": 39, "ga": 9}


def fill(cache, layer: int, positions: int, dtype: torch.dtype) -> None:
    """Give a layer's cache `positions` entries, which is what a prefix leaves behind.

    The keys are random rather than computed, because this measures the attention step
    and not the layer that would have produced them. They are appended in blocks so a
    256k fill does not materialise 256k rows to write them.
    """
    done = 0
    while done < positions:
        size = min(8192, positions - done)
        shape = cache._shapes[layer]
        cache.append(
            layer,
            torch.randn(shape.num_kv_heads, size, shape.head_dim, dtype=dtype, device=cache.device) * 0.1,
            torch.randn(shape.num_kv_heads, size, shape.v_head_dim, dtype=dtype, device=cache.device) * 0.1,
        )
        done += size


def synchronise(function, rounds: int = 3) -> float:
    """Fastest of `rounds` runs, in milliseconds -- the one least interfered with."""
    best = float("inf")
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - started) * 1e3)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seqs", type=int, nargs="+", default=(1024, 4096, 8192))
    parser.add_argument("--chunks", type=int, nargs="+", default=(512, 1024, 2048))
    parser.add_argument("--decode", type=int, nargs="+", default=(4096, 32768, 65536))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--block", type=int, default=1024)
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.checkpoint, "config.json")):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0
    dtype = getattr(torch, args.dtype)
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    config = checkpoint.layer

    attentions = {
        layer: MimoV2DeviceAttention(checkpoint, layer, "cuda", dtype, block=args.block)
        for layer in LAYERS
    }

    print("## Prefill: a chunk appended to a cache, one layer at a time")
    print()
    print("| layer | family | context | chunk | ms | tokens/s | pairs vs a dense pass |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    # A timing runs the call several times and every call appends its chunk, so a
    # global layer's cache needs room for all of them or the append refuses.
    room = 8 * max(args.chunks) + 4096
    for layer, attention in attentions.items():
        shape = attention.shape
        for context in args.seqs:
            for chunk in args.chunks:
                cache = MimoV2KVCache(config, context + room, [layer], "cuda", dtype)
                hidden = torch.randn(chunk, config.hidden_size, dtype=dtype, device="cuda") * 0.5
                fill(cache, layer, context, dtype)
                attention.forward(hidden, start_pos=context, cache=cache)
                elapsed = synchronise(
                    lambda: attention.forward(hidden, start_pos=context, cache=cache)
                )
                print(
                    f"| {layer} | {shape.family} | {context} | {chunk} | {elapsed:.2f} | "
                    f"{chunk / elapsed * 1e3:.0f} | {attention.last_stats.pairs / 1e6:.1f}M / "
                    f"{attention.last_stats.full_pairs / 1e6:.1f}M |"
                )
    print()

    print("## Decode: one token against a cache of `context` positions")
    print()
    print("| layer | family | context | ms | tokens/s | path |")
    print("| ---: | --- | ---: | ---: | ---: | --- |")
    for layer, attention in attentions.items():
        shape = attention.shape
        for context in args.decode:
            cache = MimoV2KVCache(config, context + 4096, [layer], "cuda", dtype)
            fill(cache, layer, context, dtype)
            step = torch.randn(1, config.hidden_size, dtype=dtype, device="cuda") * 0.5
            attention.forward(step, start_pos=context, cache=cache)
            elapsed = synchronise(lambda: attention.forward(step, start_pos=context, cache=cache))
            print(
                f"| {layer} | {shape.family} | {context} | {elapsed:.2f} | {1e3 / elapsed:.1f} | "
                f"{attention.last_stats.path} |"
            )
    print()

    print("A whole-token estimate under the same torch, from one layer of each family:")
    print("attention alone, with the cache pre-filled and no projections to hide behind.")
    print()
    print("| context | prefill ms/token (chunk 1024) | decode ms/token |")
    print("| ---: | ---: | ---: |")
    for context in args.decode:
        totals = {}
        for mode in ("prefill", "decode"):
            total = 0.0
            for layer, attention in attentions.items():
                shape = attention.shape
                cache = MimoV2KVCache(config, context + 4096, [layer], "cuda", dtype)
                fill(cache, layer, context, dtype)
                if mode == "prefill":
                    chunk = 1024
                    rows = torch.randn(chunk, config.hidden_size, dtype=dtype, device="cuda") * 0.5
                    elapsed = synchronise(
                        lambda a=attention, h=rows, c=cache: a.forward(h, start_pos=context, cache=c),
                        rounds=2,
                    ) / chunk
                else:
                    step = torch.randn(1, config.hidden_size, dtype=dtype, device="cuda") * 0.5
                    elapsed = synchronise(
                        lambda a=attention, h=step, c=cache: a.forward(h, start_pos=context, cache=c),
                        rounds=2,
                    )
                total += elapsed * FAMILY_COUNTS[shape.family]
            totals[mode] = total
        print(f"| {context} | {totals['prefill']:.3f} | {totals['decode']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
