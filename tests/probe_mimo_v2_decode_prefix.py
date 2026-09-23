#!/usr/bin/env python
"""What the attention's prefix copy costs a decode step, and what not making it is worth.

`MimoV2KVCache.append_and_span` hands the attention the span it is about to read -- the prefix with
this call's own keys on the end of it -- as a view of the cache buffer whenever nothing has wrapped,
so the attention does not have to build that span with a `torch.cat` first. The bytes are the same
bytes in the same order, which is what makes the cut cheap to take: at 262144 a global layer's prefix
is 160 MiB of keys and values with the attention split over the four ranks, read and written every
token, and nine of the forty-eight layers are that shape.

The arms are interleaved, and the arm that is not the new path is not a monkeypatch of the
arithmetic: it replaces the cache's method with one that answers `None`, which is what a caller sees
when the buffer has wrapped -- so the `cat` arm is the code that ran before the view existed, and a
step in either arm is the shipped path. The comparison that counts is between arms of the same round,
because this box has measured the same configuration 194 ms and 252 ms in two processes on one
afternoon.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_prefix.py --depth 262144
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_prefix.py --depth 8192 --rounds 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_attention import MimoV2KVCache  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.bench_mimo_v2_model import PhaseTimer, fill_cache  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT",
                                                               DEFAULT_CHECKPOINT))
    parser.add_argument("--depth", type=int, default=262144)
    parser.add_argument("--steps", type=int, default=4, help="decode steps an arm measures")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2, help="times round the two arms")
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
    capacity = depth + 1 + 8 * (args.warmup + args.steps * 2 * args.rounds)
    cache = model.cache(capacity)
    model.greedy(PROMPT_IDS, max_tokens=1, cache=cache)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()
    cache.reset()
    fill_cache(cache, [layer.layer_idx for layer in model.layers], depth)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()
    print(f"[r{rank}] cache at {depth} positions, {cache.memory_bytes / 2**30:.2f} GiB, "
          f"world {world}, attention in {ep.attention_shards} share(s)", flush=True)

    timer = PhaseTimer(model)
    # The shipped method, and the answer a wrapped buffer gives: both are paths a caller can meet.
    shipped = MimoV2KVCache.append_and_span

    def wrapped(self, layer, key, value, *, start_pos):
        return None

    def configure(span: bool) -> None:
        MimoV2KVCache.append_and_span = shipped if span else wrapped

    def arm(span: bool, position: int, steps: int) -> tuple[float, float, int]:
        """One arm, `steps` steps long, and the attention its share of the step was.

        The arms append to one cache -- a step cannot read a position that has not been appended --
        so the position they start from is the previous arm's, and the depth is `--depth` at the
        first step of the first arm and a few positions deeper at the last.
        """
        configure(span)
        logits = None
        step_position = position
        for _ in range(args.warmup):
            feed = PROMPT_IDS[0] if logits is None else int(logits.argmax())
            logits = model.step(feed, start_pos=step_position, cache=cache)[-1]
            step_position += 1
        torch.cuda.synchronize()
        if world > 1:
            torch.distributed.barrier()
        timer.read()
        started = time.perf_counter()
        for _ in range(steps):
            logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
            step_position += 1
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        phases = timer.read()
        return seconds / steps * 1e3, phases["attn"] / steps, step_position

    position = depth
    arms: dict[bool, list[tuple[float, float]]] = {True: [], False: []}
    for _ in range(args.rounds):
        for span in (True, False):
            token_ms, attn_ms, position = arm(span, position, args.steps)
            arms[span].append((token_ms, attn_ms))
    configure(True)

    # The pair of the same round, and then the means: the drift this box has between processes is
    # why the pairing is printed at all.
    for index in range(args.rounds):
        span_ms, cat_ms = arms[True][index][0], arms[False][index][0]
        print(
            f"[r{rank}] round {index + 1}: span {span_ms:7.1f} ms against cat {cat_ms:7.1f} ms "
            f"({cat_ms - span_ms:+6.1f} ms, {cat_ms / span_ms:.3f}x)",
            flush=True,
        )
    for span in (True, False):
        token = sum(entry[0] for entry in arms[span]) / len(arms[span])
        attn = sum(entry[1] for entry in arms[span]) / len(arms[span])
        print(
            f"[r{rank}] {'span' if span else 'cat '}: {token:7.1f} ms a token "
            f"({1000 / token:5.2f} tok/s), attention {attn:5.1f} ms of it",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
