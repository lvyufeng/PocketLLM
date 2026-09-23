#!/usr/bin/env python
"""What one 16 KiB all-reduce costs on this host's four cards, in isolation and in a chain.

The decode profile bills a token ~47 ms of device time inside the collective -- 1 ms a layer for
16 KiB, which is two orders of magnitude above a good all-reduce's latency. This measures the bare
primitive on the same four ranks, in three shapes: a back-to-back chain (throughput floor), a chain
with a kernel between the messages (what the model does), and the same with the peers' P2P paths
disabled, so that "is the hop through the host" is a measurement rather than a guess.

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_allreduce.py --messages 47
    NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 tests/probe_mimo_v2_allreduce.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

SIZE = 4096  # dim, one [1, 4096] fp32 row = 16 KiB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=47, help="messages a round")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--work", type=int, default=1, help="1 also runs a kernel between messages")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    device = torch.device(f"cuda:{local}")
    pair = torch.randint(0, 100, (1, SIZE), device=device, dtype=torch.float32)
    scratch = torch.zeros((256, 256), device=device)

    def kernel() -> None:
        """A stand-in for one layer's expert kernel: real work, on the same stream."""
        for _ in range(2):
            torch.mm(scratch, scratch.t())

    for _ in range(3):
        dist.all_reduce(pair)
    torch.cuda.synchronize()
    dist.barrier()

    def run(with_kernel: bool) -> float:
        best = 1e9
        for _ in range(args.rounds):
            dist.barrier()
            start = time.perf_counter()
            for _ in range(args.messages):
                dist.all_reduce(pair)
                if with_kernel:
                    kernel()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        return best

    chain = run(False)
    with_kernel = run(True)
    if rank == 0:
        print(
            f"[r0] world {world}, {args.messages} x 16 KiB all_reduce: chain {chain * 1e3:.2f} ms "
            f"({chain / args.messages * 1e6:.1f} us a message), with a kernel between "
            f"{with_kernel * 1e3:.2f} ms ({with_kernel / args.messages * 1e6:.1f} us a message)",
            flush=True,
        )

    # The same measurement at a few sizes, so that "it is the message" and "it is the latency" are
    # distinguishable. **Every rank runs the sweep and only rank 0 prints it.** A rank that left
    # after the first line would take its side of the next `barrier()` with it, and the run would
    # hang at whatever the survivors reached next rather than report the table.
    for size in (SIZE, SIZE * 16, SIZE * 1024):
        buffer = torch.zeros((1, size), device=device, dtype=torch.float32)
        for _ in range(3):
            dist.all_reduce(buffer)
        torch.cuda.synchronize()
        dist.barrier()
        best = 1e9
        for _ in range(args.rounds):
            dist.barrier()
            start = time.perf_counter()
            for _ in range(args.messages):
                dist.all_reduce(buffer)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
        if rank == 0:
            print(
                f"[r0] {size * 4 / 1024:8.1f} KiB a message: {best / args.messages * 1e6:8.1f} us a "
                f"message, {size * 4 * args.messages / best / 1e9:6.2f} GB/s",
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
