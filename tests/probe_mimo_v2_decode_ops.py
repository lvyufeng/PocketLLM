#!/usr/bin/env python
"""Where one MiMo-V2.6 decode step's device time goes, op by op.

The phase probe says a token is 286 ms: the copy stall 102, the attention 63-100, the collective
47, the router 23, the expert kernel 15 and 35 ms of "everything else". This one names those from
the profiler's own tables, so the question "what would a kernel buy" has an answer in microseconds
rather than in a guess.

**Two tables, because at this size the host is the suspect.** Device time says how much of the card
a step uses, and it is small -- 522 ms of an at-1-rank step that is mostly `Memcpy HtoD`. CPU time
says how long the host spends *asking* for the step, and when that is the larger number the step is
launch-bound and not arithmetic-bound. A step that spends 160 ms a token of host time in 47 layers
is spending 3.4 ms a layer to issue a layer's worth of small kernels, and that is worth knowing
before any kernel is written.

One rank, one card: the op mix of a layer is the same on every rank, and a single process keeps the
table readable.

    python tests/probe_mimo_v2_decode_ops.py --steps 3 --prompt 8

The checkpoint is the default asset path; without it the script exits 0 and says so.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.bench_mimo_v2_model import fill_cache  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="positions to fill the cache with instead of prefilling `--prompt`",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--resident-rows", type=int, default=0)
    parser.add_argument(
        "--stub-experts",
        action="store_true",
        help="replace the routed experts with the zero an empty rank returns",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    rank = ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")

    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint, device=device, expert_source=bank, ep=ep, resident_rows=args.resident_rows
    )
    if args.stub_experts:
        experts_module = model.experts

        def stubbed(hidden, indices, weights, **kwargs):
            return torch.zeros(
                (hidden.shape[0], experts_module.dim),
                dtype=torch.float32,
                device=hidden.device,
            )

        experts_module.forward = stubbed
    torch.cuda.synchronize()

    depth = int(args.depth or 0)
    cache = model.cache(max(depth, args.prompt) + args.warmup + args.steps * 3 + 8)
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    model.greedy(prompt_ids[:1] if depth else prompt_ids, max_tokens=1, cache=cache)
    cache.reset()
    if depth:
        fill_cache(cache, [layer.layer_idx for layer in model.layers], depth)
        position = depth
        logits = None
        for _ in range(args.warmup):
            logits = model.step(PROMPT_IDS[0], start_pos=position, cache=cache)[-1]
            position += 1
        print(f"[r{rank}] cache at {depth} positions", flush=True)
    else:
        logits = None
        for position, token in enumerate(prompt_ids):
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
        position = len(prompt_ids)
    torch.cuda.synchronize()

    drawn = [int(logits.argmax())]
    for step in range(args.steps):
        logits = model.step(drawn[-1], start_pos=position + step, cache=cache)[-1]
        drawn.append(int(logits.argmax()))

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step in range(args.steps):
            logits = model.step(drawn[-1], start_pos=position + step, cache=cache)[-1]
            drawn.append(int(logits.argmax()))
        torch.cuda.synchronize()

    events = list(prof.key_averages())
    for title, field in (("host", "self_cpu_time_total"), ("device", "self_device_time_total")):
        rows = sorted(
            ((getattr(event, field), event.key, event.count) for event in events), reverse=True
        )
        total = sum(row[0] for row in rows)
        print(
            f"\n[r{rank}] {args.steps} steps, self {title} time {total / 1e3:.1f} ms "
            f"({total / 1e3 / args.steps:.1f} ms a step), {len(rows)} distinct ops"
        )
        print(f"{'ms/step':>9} {'us/call':>9} {'calls/step':>11}  op")
        for micros, key, count in rows[: args.top]:
            print(
                f"{micros / 1e3 / args.steps:9.3f} {micros / max(count, 1):9.1f} "
                f"{count / args.steps:11.1f}  {key[:78]}"
            )
        # And the same table by how many times a step asks for it, because the two questions are
        # different: a step can be bound by the op that costs the most and by the op it launches
        # the most of, and the second is the one a launch-bound step has to answer.
        print(f"\n{'calls/step':>11} {'us/call':>9} {'ms/step':>9}  op, most calls first")
        for micros, key, count in sorted(rows, key=lambda row: row[2], reverse=True)[: args.top]:
            print(
                f"{count / args.steps:11.1f} {micros / max(count, 1):9.1f} "
                f"{micros / 1e3 / args.steps:9.3f}  {key[:78]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
