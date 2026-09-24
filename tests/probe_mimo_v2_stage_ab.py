#!/usr/bin/env python
"""What the expert staging costs the *token*, with the arms interleaved in one process.

The copy is 1198.5 MiB a rank a token at 10.4 GiB/s, which is 115 ms of the link's time whatever
the step's total is. That is not the same question as what it costs the step: if the host is busy
elsewhere while the copy runs, most of it is free, and a resident expert set that deletes 80% of it
would buy nothing. The two arms are one process because this box has measured the same
configuration 184 and 245 ms on one afternoon; `--rounds` times round, and the pairs of a round are
what is compared.

The off arm replaces `_stage` with a function that does nothing, so the kernel reads whatever the
slot held last -- the arithmetic is wrong and the point is not the token, it is the clock. The
kernel call, the deal, the `tolist`, the collective and the attention all stay.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_stage_ab.py --steps 6 --rounds 3
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

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
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
    experts = model.experts

    cache = model.cache(max(args.depth, args.prompt) + 4 * args.rounds * args.steps + 16)
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    model.greedy(prompt_ids, max_tokens=1, cache=cache)
    cache.reset()
    logits = None
    for position, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
    position = len(prompt_ids)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    shipped = experts._stage

    def nothing(slot, layer_id, experts_arg):
        return None

    def arm(staging: bool, position: int, steps: int, warmup: int, logits):
        """One arm, `steps` measured after `warmup`, split into the host's time and the queue."""
        experts._stage = shipped if staging else nothing
        step_position = position
        for _ in range(warmup):
            logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
            step_position += 1
        torch.cuda.synchronize()
        if world > 1:
            torch.distributed.barrier()
        host = tail = 0.0
        for _ in range(steps):
            began = time.perf_counter()
            logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
            queued = time.perf_counter()
            torch.cuda.synchronize()
            host += (queued - began) * 1e3
            tail += (time.perf_counter() - queued) * 1e3
            step_position += 1
        if world > 1:
            torch.distributed.barrier()
        return (host + tail) / steps, host / steps, tail / steps, step_position, logits

    arms: dict[bool, list[tuple[float, float, float]]] = {True: [], False: []}
    for _ in range(args.rounds):
        for staging in (True, False):
            token, host, tail, position, logits = arm(
                staging, position, args.steps, args.warmup, logits
            )
            arms[staging].append((token, host, tail))
    experts._stage = shipped

    for index in range(args.rounds):
        on, off = arms[True][index][0], arms[False][index][0]
        print(
            f"[r{rank}] round {index + 1}: staging {on:7.1f} ms against none {off:7.1f} ms "
            f"({on - off:+6.1f} ms, {on / off:.3f}x)",
            flush=True,
        )
    for staging in (True, False):
        got = arms[staging]
        print(
            f"[r{rank}] {'staging' if staging else 'none   '}: "
            f"{sum(entry[0] for entry in got) / len(got):7.1f} ms a token "
            f"({1000 / (sum(entry[0] for entry in got) / len(got)):5.2f} tok/s), "
            f"host {sum(entry[1] for entry in got) / len(got):7.1f}, "
            f"queue {sum(entry[2] for entry in got) / len(got):5.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
