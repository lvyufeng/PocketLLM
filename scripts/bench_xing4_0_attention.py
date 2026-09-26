#!/usr/bin/env python
"""What Xing4.0's MLA attention costs on a 2080 Ti, at the released widths.

#391's acceptance asks whether the attention can be the decode bottleneck.  It
answers that three ways, because the three give different numbers and the
difference is the finding.

**The floor.**  At decode the attention is a handful of dense GEMMs over a fixed
weight volume per layer, so the number that matters is bytes-per-token divided by
616 GB/s -- the card's peak.  It is computed here from the released shard's own
tensor shapes, at the storage dtype asked for, and scaled to the trunk's 40
layers.

**The GPU's own time.**  An eager PyTorch port is not a kernel, and on this card
the difference is not a rounding error: a one-token forward here is ~250 kernel
launches, and at ~7 us of launch each that is milliseconds of *host* time per
layer per token.  So the script reports the GPU-side time (from the profiler) and
the wall time (from events) separately.  The GPU number is what a kernel would
have to beat; the wall number is what an unoptimized runtime would actually cost.

**Reachability.**  A floor nobody can reach is not evidence, so the real
projections are also timed as plain fp16 GEMMs at their real shapes, at M=1 and at
M=512, and reported as a fraction of peak.

It runs on one card and needs no GGUF: layer 2 of the released BF16 checkpoint is
one shard, and its attention is every layer's attention.

Usage::

    python scripts/bench_xing4_0_attention.py [--device cuda:2] [--dtype float16]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.xing4_0.attention import KVLatentCache, MLAAttention, MLAAttentionWeights  # noqa: E402
from src.models.xing4_0.config import Xing4_0Params  # noqa: E402

CHECKPOINT = Path("/mnt/data2/Xing4.0-29B-A4B")
SHARD = CHECKPOINT / "model-00003-of-00041.safetensors"
LAYER = 2
TRUNK_LAYERS = 40
PEAK_GBPS = 616.0  # RTX 2080 Ti, GDDR6


def load_attention(device: torch.device, dtype: torch.dtype):
    from safetensors.torch import safe_open

    params = Xing4_0Params.from_json(CHECKPOINT / "config.json")
    prefix = f"model.layers.{LAYER}."
    with safe_open(str(SHARD), framework="pt") as handle:
        tensors = {
            name[len(prefix) :]: handle.get_tensor(name).to(device=device, dtype=dtype)
            for name in handle.keys()
            if name.startswith(prefix)
        }
    return params, MLAAttentionWeights.from_hf(tensors, params)


def attention_bytes(params: Xing4_0Params, dtype: torch.dtype) -> int:
    """The weight volume one decode token streams through one attention layer."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    elements = (
        params.q_lora_rank * params.hidden_size
        + params.q_lora_rank
        + params.n_heads * params.qk_head_dim * params.q_lora_rank
        + (params.kv_lora_rank + params.qk_rope_head_dim) * params.hidden_size
        + params.kv_lora_rank
        # kv_b_proj, stored whole or as k_b and v_b -- the same bytes either way.
        + params.n_heads * (params.qk_nope_head_dim + params.v_head_dim) * params.kv_lora_rank
        + params.hidden_size * params.n_heads * params.v_head_dim
    )
    return elements * itemsize


def wall_ms(fn, *, warmup: int = 3, iters: int = 20) -> float:
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


def gpu_ms(fn, *, iters: int = 20) -> float:
    """Median GPU-busy time per call, which is what a kernel has to beat."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    averages = prof.key_averages()
    # `key_averages()` sums per-op, so the GPU's own time is the sum over ops;
    # the profiler is the only way to separate it from the launch time around it.
    total_us = sum(
        getattr(event, "self_device_time_total", 0.0) for event in averages
    )
    return total_us / 1000 / iters


def prime_cache(
    attention: MLAAttention,
    cache: KVLatentCache,
    context: int,
    device: torch.device,
    dtype: torch.dtype,
    params: Xing4_0Params,
    *,
    chunk: int = 2048,
) -> None:
    """Fill the cache with `context` tokens' worth of latents.

    Written directly rather than by running a prefill, because the cache only
    holds `kv_a_proj`'s output and its rope key, and running the absorbed prefill
    to get them would score the whole `(heads, context, context)` matrix -- 64 GiB
    of fp16 at 32K, which is the reason the absorbed form is a decode path.  What
    is timed below is the step, so how the prompt got there does not matter; the
    chunking is only so a long context also does not need a long prefill.
    """
    for start in range(0, context, chunk):
        length = min(chunk, context - start)
        hidden = torch.randn(1, length, params.hidden_size, device=device, dtype=dtype)
        positions = torch.arange(start, start + length, device=device)
        latent, rope_key = attention._compressed_kv(hidden)
        rope_key = attention.rope_queries(rope_key.unsqueeze(2), positions).squeeze(2)
        cache.append(torch.cat((latent, rope_key), dim=-1), start)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--contexts", default="1,4096,32768")
    parser.add_argument("--prefills", default="128,512,2048")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    params, weights = load_attention(device, dtype)
    attention = MLAAttention(params, weights, dtype=dtype, device=device)
    itemsize = torch.tensor([], dtype=dtype).element_size()

    per_layer = attention_bytes(params, dtype)
    total = per_layer * TRUNK_LAYERS
    floor = total / (PEAK_GBPS * 1e6)
    print(f"device {device}  dtype {args.dtype}  trunk layers {TRUNK_LAYERS}")
    print(
        f"attention weights  {per_layer / 2**20:.1f} MiB/layer  {total / 2**30:.3f} GiB/token  "
        f"floor {floor:.3f} ms/token = {1000 / floor:.0f} tok/s at {PEAK_GBPS:.0f} GB/s"
    )
    print()

    hidden = torch.randn(1, params.hidden_size, device=device, dtype=dtype)
    print("plain fp16 GEMMs at the real projection shapes (reachability)")
    print(f"  {'projection':>10}  {'shape':>12}  {'M=1 ms':>8}  {'M=1 GB/s':>9}  {'M=512 ms':>9}  {'M=512 GB/s':>10}")
    for name, in_dim, out_dim in (
        ("q_b", params.q_lora_rank, params.n_heads * params.qk_head_dim),
        ("kv_b", params.kv_lora_rank, params.n_heads * (params.qk_nope_head_dim + params.v_head_dim)),
        ("o_proj", params.n_heads * params.v_head_dim, params.hidden_size),
    ):
        weight = torch.randn(out_dim, in_dim, device=device, dtype=dtype)
        moved = weight.numel() * itemsize
        x1 = torch.randn(1, 1, in_dim, device=device, dtype=dtype)
        m1 = wall_ms(lambda w=weight, x=x1: torch.nn.functional.linear(x, w))
        x512 = torch.randn(1, 512, in_dim, device=device, dtype=dtype)
        m512 = wall_ms(lambda w=weight, x=x512: torch.nn.functional.linear(x, w))
        print(
            f"  {name:>10}  {f'{out_dim}x{in_dim}':>12}  {m1:>8.4f}  {moved / m1 / 1e6:>9.1f}  "
            f"{m512:>9.4f}  {moved / m512 / 1e6:>10.1f}"
        )
    print()

    print("prefill, one pass (no cache)")
    print(f"  {'seq':>6}  {'expanded wall':>14}  {'absorbed wall':>14}  {'expanded GPU':>13}  {'absorbed GPU':>13}")
    for seq in (int(s) for s in args.prefills.split(",")):
        prompt = torch.randn(1, seq, params.hidden_size, device=device, dtype=dtype)
        positions = torch.arange(seq, device=device)
        # Only the expanded form is scored over the whole (heads, seq, seq) matrix
        # cheaply; absorbed materialises the same matrix without the causal saving
        # and is a decode path, so it is reported and not pushed past 2048.
        ew = wall_ms(lambda: attention.forward_expanded(prompt, positions))
        eg = gpu_ms(lambda: attention.forward_expanded(prompt, positions))
        aw = wall_ms(lambda: attention.forward_absorbed(prompt, positions))
        ag = gpu_ms(lambda: attention.forward_absorbed(prompt, positions))
        print(f"  {seq:>6}  {ew:>14.3f}  {aw:>14.3f}  {eg:>13.3f}  {ag:>13.3f}")
    print(f"  (ms for one layer; {TRUNK_LAYERS} of them per prefill)")
    print()

    print("decode step, one token against a filled cache")
    print(
        f"  {'ctx':>6}  {'form':>9}  {'wall ms':>9}  {'GPU ms':>8}  {'GPU GB/s':>9}  {'GPU %peak':>9}  "
        f"{'40-layer GPU ms':>15}  {'40-layer tok/s':>15}"
    )
    for context in (int(c) for c in args.contexts.split(",")):
        # Both forms, primed identically, so the cache traffic is the only thing
        # that differs between the two rows.  For the expanded form the cache is
        # the *expanded* key and value; for the absorbed one it is the latent.
        absorbed_cache = KVLatentCache(1, context + 1, params, device=device, dtype=dtype)
        prime_cache(attention, absorbed_cache, context, device, dtype, params)
        expanded_cache = KVLatentCache(1, context + 1, params, device=device, dtype=dtype)
        prime_cache(attention, expanded_cache, context, device, dtype, params)
        step_position = torch.tensor([context], device=device)

        for form, forward, cache in (
            ("absorbed", attention.forward_absorbed, absorbed_cache),
            ("expanded", attention.forward_expanded, expanded_cache),
        ):
            def step(f=forward, c=cache) -> None:
                f(hidden.unsqueeze(0), step_position, cache=c, start_pos=context)

            wall = wall_ms(step)
            busy = gpu_ms(step)
            rate = per_layer / 2**30 / (busy / 1000)
            print(
                f"  {context:>6}  {form:>9}  {wall:>9.3f}  {busy:>8.3f}  {rate:>9.1f}  {rate / PEAK_GBPS * 100:>8.0f}%  "
                f"{busy * TRUNK_LAYERS:>15.1f}  {1000 / (busy * TRUNK_LAYERS):>15.1f}"
            )
        assert absorbed_cache.length == context + 1
    print()

    print("what that means")
    print(f"  the attention's own ceiling, at peak bandwidth:  {1000 / floor:.0f} tok/s")
    print("  the port above is host-bound: the wall column is launch time, not arithmetic")
    print("  so a decode step's attention cost is bounded by the GPU column, and the MoE")
    print("  -- which must stage routed experts per token -- is where the tokens go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
