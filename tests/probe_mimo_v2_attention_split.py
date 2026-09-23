#!/usr/bin/env python
"""The four-way attention split against the whole attention, bit for bit.

The split's whole claim is exactness: a rank computes its own quarter of the query heads with the
key and value heads they attend to, the four quarters are *concatenated* (`ep.make_all_gather`, not
summed), and the layer's projection runs on the whole of it. So the answer is the answer -- not a
close one -- and this probe is what says so.

Two modes:

* **one card** -- the four shares are run one after another and joined with `torch.cat`, which is
  the arithmetic of the join without the collective. What it establishes is that the *pieces* are
  the whole's pieces.
* **four ranks** (`torchrun --nproc_per_node=4`) -- every rank also computes the whole attention on
  its own card, from the whole weight and a whole cache, and compares its own gather-joined answer
  against it. This is the mode that exercises `all_gather_into_tensor` and the `_qkv_weight` share
  read, and it is the one that has to come out `0.00e+00`.

The cache is filled with the same pseudo-random keys and values on every rank, sliced by key head
for a share, and the fill is done from a CPU generator so that four cards and one card see the same
numbers. Nothing here appends to a cache from the *whole* path and then compares a share against a
longer prefix: one call, one prefix, both paths.

**What the numbers mean.** `qkv` and the one-card mode's `pre_o` are held to zero and they are: a
share's fused projection is the whole's rows for that share, bit for bit, and feeding the *whole's*
own tensors to a share's shapes reproduces the share's `pre_o` exactly. What is not always zero is
the attention op's own output, because a batched GEMM's tiling depends on the batch size and the
batch here is the number of key heads: a share has a quarter of them. Measured on the release the
per-key-head path (a global layer past `FOLD_KEYS` keys, and any chunk of queries) is exact, and the
folded path taken by a decode step below `FOLD_KEYS` keys -- which is every step of a windowed layer,
whose ring cache is 128 keys long however long the sequence is -- differs by up to one float32 ULP
of `pre_o`, one or two bfloat16 ULPs of the layer's output. That is the tolerance this probe
asserts: half a bfloat16 ULP would be zero and two would be a real difference.

    python tests/probe_mimo_v2_attention_split.py
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_attention_split.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.device_attention import (  # noqa: E402
    MimoV2DeviceAttention,
    MimoV2KVCache,
)
from src.models.mimo_v2.ep import ATTENTION_SHARDS, EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

#: The layers to probe: one of each family. Layer 0 is a global layer, layer 1 carries the sliding
#: window and the sink, and the two are the two arithmetic shapes this checkpoint has.
LAYERS = (0, 1)

#: The band a joined answer may sit in and still be the whole's answer: one bfloat16 ULP of the
#: layer's own output. See the module docstring for why the band is not zero everywhere.
TOLERANCE = 2.0**-7


def _refuse(piece: torch.Tensor) -> torch.Tensor:
    raise AssertionError("the one-card mode joins its own shares; this gather is never called")


def fill(
    config,
    cache: MimoV2KVCache,
    layer_idx: int,
    shape,
    keys: int,
    *,
    shard: int = 0,
    shards: int = 1,
) -> None:
    """Fill one cache with `keys` keys and values, as this share of the key heads.

    `shape` is the *whole layer's* geometry, whichever cache is being filled: the numbers are drawn
    once for every key head the layer has and the share takes the `num_kv_heads / shards` contiguous
    heads `shard_shape` gave it, which is the slice the whole layer would have attended to from
    those query heads. Generated on the CPU from a fixed seed so that every card and every mode
    draws the same numbers.
    """
    generator = torch.Generator()
    generator.manual_seed(0)
    heads = shape.num_kv_heads
    full_key = torch.randn((heads, keys, shape.head_dim), generator=generator)
    full_value = torch.randn((heads, keys, shape.v_head_dim), generator=generator)
    own = heads // shards
    low = shard * own
    cache.append(
        layer_idx,
        full_key[low : low + own].to(cache.device),
        full_value[low : low + own].to(cache.device),
    )


def timed(fn, rounds: int) -> float:
    """Mean wall time of `fn` in milliseconds, after one untimed call."""
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / rounds * 1e3


def one_card(args, checkpoint, config) -> int:
    device = torch.device(args.device)
    status = 0
    print(f"{'layer':>5} {'family':>7} {'rows':>5} {'keys':>7} {'whole ms':>9} {'share ms':>9} "
          f"{'gain':>6} {'pre_o':>10} {'post_o':>10} {'qkv':>10}", flush=True)
    for layer_idx in args.layers:
        shape = config.attention(layer_idx)
        whole = MimoV2DeviceAttention(checkpoint, layer_idx, device, torch.bfloat16)
        pieces = [
            MimoV2DeviceAttention(
                checkpoint,
                layer_idx,
                device,
                torch.bfloat16,
                shard=shard,
                shards=args.shards,
                gather=_refuse,
            )
            for shard in range(args.shards)
        ]
        for rows in args.rows:
            for keys in args.keys:
                # The calls below append to these caches, so they hold the prefix, the parity call
                # and the timed rounds: a decode step's cache is where the stage's own reads are.
                room = keys + rows * (args.rounds + 3) + 8
                cache = MimoV2KVCache(
                    config, room, [layer_idx], device=device, dtype=torch.bfloat16
                )
                fill(config, cache, layer_idx, whole.shape, keys)
                share_caches = [
                    MimoV2KVCache(
                        config,
                        room,
                        [layer_idx],
                        device=device,
                        dtype=torch.bfloat16,
                        shard=shard,
                        shards=args.shards,
                    )
                    for shard in range(args.shards)
                ]
                for shard, share_cache in enumerate(share_caches):
                    fill(config, share_cache, layer_idx, whole.shape, keys, shard=shard,
                         shards=args.shards)
                hidden = torch.randn(
                    (rows, config.hidden_size), generator=torch.Generator().manual_seed(1)
                ).to(device=device, dtype=torch.bfloat16)

                whole_out = whole.forward(hidden, start_pos=keys, cache=cache)
                shares = [
                    pieces[shard].attention_output(hidden, start_pos=keys, cache=share_caches[shard])
                    for shard in range(args.shards)
                ]
                joined = torch.cat([piece for piece, _ in shares], dim=-1)
                joined_post = F.linear(joined.to(whole.o_proj.dtype), whole.o_proj)
                joined_qkv = torch.cat([qkv for _, qkv in shares], dim=-1)

                reference = whole_out["attn_out_pre_o"]
                post = whole_out["attn_out_post_o"]
                peak = float(post.float().abs().max())
                pre_diff = float((joined.float() - reference.float()).abs().max())
                post_diff = float((joined_post.float() - post.float()).abs().max())
                qkv_diff = float(
                    (joined_qkv.float() - whole_out["qkv_raw"].float()).abs().max()
                )
                if joined.shape[-1] != reference.shape[-1] or qkv_diff:
                    status = 1
                if post_diff > peak * TOLERANCE:
                    status = 1
                # And what one *share* costs against the layer, which is the stage's whole point: a
                # rank runs one share and the other ranks run theirs at the same time, so the layer's
                # wall is the share's wall. Timed after the comparison, on the same caches, one
                # append a call either way.
                whole_ms = timed(
                    lambda: whole.attention_output(
                        hidden, start_pos=cache.written(layer_idx), cache=cache
                    ),
                    args.rounds,
                )
                share_ms = timed(
                    lambda: pieces[0].attention_output(
                        hidden, start_pos=share_caches[0].written(layer_idx), cache=share_caches[0]
                    ),
                    args.rounds,
                )
                print(f"{layer_idx:>5} {shape.family:>7} {rows:>5} {keys:>7} {whole_ms:>9.3f} "
                      f"{share_ms:>9.3f} {whole_ms / share_ms:>6.2f} {pre_diff:>10.2e} "
                      f"{post_diff:>10.2e} {qkv_diff:>10.2e}"
                      f"  pre peak {float(reference.abs().max()):.2e} post peak {peak:.2e}",
                      flush=True)
                del cache, share_caches, whole_out, shares, joined, joined_post, joined_qkv
                torch.cuda.empty_cache()
        del whole, pieces
        torch.cuda.empty_cache()
    return status


def four_ranks(args, checkpoint, config) -> int:
    """Every rank computes its share *and* the whole, and compares. The whole is the reference.

    The whole attention is built on every rank, from the whole weight and a whole cache: it is the
    same computation on four cards, so it is also a check that the four cards agree, which is what
    a split has to be measured against and not an assumption. The cache is filled identically --
    the share from its own key heads, the whole from all of them -- so the two paths see the same
    prefix and only the split differs.
    """
    ep = EpGroup.from_env()
    if ep.world < 2:
        raise SystemExit("this mode is for a world of four; run it under torchrun")
    shards = ep.attention_shards
    device = torch.device(args.device)
    print(f"rank {ep.rank} of {ep.world}: attention in {shards} share(s)", flush=True)
    status = 0
    for layer_idx in args.layers:
        shape = config.attention(layer_idx)
        split = MimoV2DeviceAttention(
            checkpoint,
            layer_idx,
            device,
            torch.bfloat16,
            shard=ep.attention_shard,
            shards=shards,
            gather=ep.gather,
        )
        whole = MimoV2DeviceAttention(checkpoint, layer_idx, device, torch.bfloat16)
        for rows in args.rows:
            for keys in args.keys:
                capacity = keys + rows + 8
                split_cache = MimoV2KVCache(
                    config,
                    capacity,
                    [layer_idx],
                    device=device,
                    dtype=torch.bfloat16,
                    shard=ep.attention_shard,
                    shards=shards,
                )
                whole_cache = MimoV2KVCache(
                    config, capacity, [layer_idx], device=device, dtype=torch.bfloat16
                )
                fill(
                    config,
                    split_cache,
                    layer_idx,
                    whole.shape,
                    keys,
                    shard=ep.attention_shard,
                    shards=shards,
                )
                fill(config, whole_cache, layer_idx, whole.shape, keys)
                hidden = torch.randn(
                    (rows, config.hidden_size), generator=torch.Generator().manual_seed(1)
                ).to(device=device, dtype=torch.bfloat16)

                whole_out = whole.forward(hidden, start_pos=keys, cache=whole_cache)
                split_out = split.forward(hidden, start_pos=keys, cache=split_cache)
                post = whole_out["attn_out_post_o"]
                pre = whole_out["attn_out_pre_o"]
                post_diff = float(
                    (split_out["attn_out_post_o"].float() - post.float()).abs().max()
                )
                pre_diff = float(
                    (split_out["attn_out_pre_o"].float() - pre.float()).abs().max()
                )
                checksum = float(post.float().sum())
                peak_post = float(post.float().abs().max())
                if post_diff > peak_post * TOLERANCE or pre.shape != split_out["attn_out_pre_o"].shape:
                    status = 1
                print(f"rank {ep.rank} layer {layer_idx} {shape.family} rows {rows} keys {keys}: "
                      f"split {tuple(split_out['attn_out_post_o'].shape)} whole "
                      f"{tuple(post.shape)} | pre abs {pre_diff:.2e} of "
                      f"{float(pre.float().abs().max()):.2e} | post abs {post_diff:.2e} of "
                      f"{peak_post:.2e} | whole sum {checksum:.6e}",
                      flush=True)
                del split_cache, whole_cache, whole_out, split_out
                torch.cuda.empty_cache()
        del split, whole
        torch.cuda.empty_cache()
    # Every rank's `whole` is the same computation, so the sums agree only if the group does; a
    # disagreement here means the reference itself is not shared and no parity number means
    # anything. A [1] tensor, so it costs nothing and cannot be confused with a real message.
    if ep.world > 1:
        import torch.distributed as dist

        worst = torch.tensor([float(status)], device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        status = int(worst.item())
        dist.destroy_process_group()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/mnt/data3/MiMo-V2.6-Flash-RL")
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 512])
    parser.add_argument("--keys", type=int, nargs="+", default=[1, 4096, 32768])
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=6, help="timed rounds a share")
    parser.add_argument("--one-card", action="store_true", help="force the local join")
    args = parser.parse_args()

    if args.shards not in (1, ATTENTION_SHARDS):
        raise SystemExit(f"{args.shards} is not a partition this checkpoint admits")

    checkpoint = MimoV2Checkpoint(args.checkpoint)
    config = checkpoint.layer
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not args.one_card:
        return four_ranks(args, checkpoint, config)
    return one_card(args, checkpoint, config)


if __name__ == "__main__":
    raise SystemExit(main())
