#!/usr/bin/env python
"""What a decode token's cost is, on the host, and what two cuts to it are worth.

The step is not arithmetic-bound and it is not collective-bound. At 32768 positions on four ranks
the numbers below come out at roughly 190 to 200 ms a token, of which the *returning call* is 183 to
192 ms and the wait behind the queue is 6: the host is the whole of the step, and the card is idle
behind it. What the host is doing is dominated by kernel launches -- about 5,300 to 5,600 a token,
110 to 120 a layer -- and by the dispatcher work around them.

Two of those dispatches are per-call work the attention does not have to do: a RoPE table rebuilt
every layer for a position that has not changed since the last token, and a nineteen-slice cut of
the fused qkv projection where one gather is the same three tensors. Both are toggled here by
writing the two attributes the production code reads, so the arms are the shipped paths and not a
monkeypatch of them.

The arms are interleaved, and that is the point of the script rather than a detail of it: the same
configuration measured 194 ms and 252 ms in two processes on the same afternoon, so a process's
first number is not its baseline and a single A-B inside one process is not evidence. Four
configurations, `--rounds` times round, and the comparison that counts is between arms of the same
round.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_host.py --depth 32768
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_host.py --depth 32768 --abab
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.bench_mimo_v2_model import fill_cache  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]
COMBINATIONS = ((False, False), (True, False), (False, True), (True, True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT",
                                                               DEFAULT_CHECKPOINT))
    parser.add_argument("--depth", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=6, help="decode steps an arm measures")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2, help="times round the four arms")
    parser.add_argument("--rows", type=int, default=20, help="op rows to report")
    parser.add_argument("--abab", action="store_true", help="run the interleaved arms")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(checkpoint, device=device, expert_source=bank, ep=ep)
    torch.cuda.synchronize()

    depth = args.depth
    capacity = depth + 1 + 8 * (args.warmup + args.steps * len(COMBINATIONS) * args.rounds)
    cache = model.cache(capacity)
    model.greedy(PROMPT_IDS, max_tokens=1, cache=cache)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()
    cache.reset()
    fill_cache(cache, model.config, [layer.layer_idx for layer in model.layers], depth)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()
    print(f"[r{rank}] cache at {depth} positions, {cache.memory_bytes / 2**30:.2f} GiB; "
          f"{sum(1 for layer in model.layers if layer.attention._rope_table is not None)} layers "
          f"share a RoPE table", flush=True)

    # The two features a layer reads, and nothing else: the arms below are the shipped paths.
    was = [
        (layer.attention._rope_table, layer.attention._qkv_order) for layer in model.layers
    ]

    def configure(rope_table: bool, qkv_gather: bool) -> None:
        for layer, (table, order) in zip(model.layers, was):
            layer.attention._rope_table = table if rope_table else None
            layer.attention._qkv_order = order if qkv_gather else None

    def arm(
        rope_table: bool, qkv_gather: bool, position: int, steps: int
    ) -> tuple[float, float, int]:
        """One configuration, `steps` steps long, split into the host's own time and the wait.

        Returns the two times and the position the cache is at afterwards, which is what the next
        arm starts from: the arms append to one cache and a step cannot read a position that has
        not been appended.
        """
        configure(rope_table, qkv_gather)
        logits = None
        step_position = position
        for _ in range(args.warmup):
            feed = PROMPT_IDS[0] if logits is None else int(logits.argmax())
            logits = model.step(feed, start_pos=step_position, cache=cache)[-1]
            step_position += 1
        torch.cuda.synchronize()
        if world > 1:
            torch.distributed.barrier()
        host = tail = 0.0
        for _ in range(steps):
            started = time.perf_counter()
            logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
            queued = time.perf_counter()
            torch.cuda.synchronize()
            host += (queued - started) * 1e3
            tail += (time.perf_counter() - queued) * 1e3
            step_position += 1
        if world > 1:
            torch.distributed.barrier()
        return host / steps, tail / steps, step_position

    host, tail, position = arm(True, True, depth, args.steps)
    print(f"[r{rank}] a step is {host:7.1f} ms on the host and {tail:6.1f} ms behind the queue "
          f"({host + tail:7.1f} ms, {1000 / (host + tail):5.2f} tok/s)", flush=True)

    if args.rows:
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_stack=False
        ) as prof:
            logits = None
            for _ in range(2):
                feed = PROMPT_IDS[0] if logits is None else int(logits.argmax())
                logits = model.step(feed, start_pos=position, cache=cache)[-1]
                position += 1
            torch.cuda.synchronize()
        if rank == 0:
            table = prof.key_averages()
            total = sum(entry.self_cpu_time_total for entry in table)
            print(f"[r0] {sum(entry.count for entry in table) / 2:.0f} ops a token, "
                  f"{total / 1e3 / 2:.1f} ms of host time a token by the profiler's own count",
                  flush=True)
            for entry in sorted(table, key=lambda e: -e.self_cpu_time_total)[: args.rows]:
                print(f"[r0]   {entry.key[:56]:56s} {entry.count / 2:8.0f} x  "
                      f"{entry.self_cpu_time_total / 1e3 / 2:7.2f} ms a token", flush=True)

    if not args.abab:
        return 0

    timings: dict[tuple[bool, bool], list[float]] = {key: [] for key in COMBINATIONS}
    for _ in range(args.rounds):
        for rope_table, qkv_gather in COMBINATIONS:
            host, tail, position = arm(rope_table, qkv_gather, position, args.steps)
            timings[(rope_table, qkv_gather)].append(host + tail)
            print(f"[r{rank}] rope_table={int(rope_table)} qkv_gather={int(qkv_gather)}  "
                  f"{host + tail:7.1f} ms/token  {1000 / (host + tail):5.2f} tok/s", flush=True)
    configure(True, True)
    if rank == 0:
        for key in COMBINATIONS:
            values = timings[key]
            print(f"[r0] rope_table={int(key[0])} qkv_gather={int(key[1])}: "
                  f"best {min(values):.1f}, mean {sum(values) / len(values):.1f} ms "
                  f"over {len(values)} rounds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
