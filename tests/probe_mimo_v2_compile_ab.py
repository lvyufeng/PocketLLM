#!/usr/bin/env python
"""Whether `torch.compile` pays on a MiMo-V2.6 decode step, timed against eager in one process.

`probe_mimo_v2_ablate.py` prices the regions and the answer is not "the arithmetic": deleting every
layer leaves 12.5 ms and deleting any single region leaves 155 to 200 of a 217 ms token, while a
4096-wide elementwise add on this box costs 10 us of host time and the reference's `rms_norm` costs
123. A step is forty thousand eager dispatches at five microseconds, and the card is idle for most
of them -- which is the exact shape `torch.compile` exists for: the pointwise chains fuse into one
kernel, the dispatches go away, and the graph breaks land on the parts that are already talking to
the host.

So this is the cheap version of the experiment: compile `step` and time it against eager, in the
same process, arms alternating by round. `--mode` passes straight through, and `reduce-overhead`
is the one that uses CUDA graphs -- worth trying and expected to fall back, because a routed layer
reads its own draw back to the host and a graph cannot contain a decision.

Correctness is not what this measures. A fused reduction is a different order of additions, and
`rms_norm` accumulates a mean over 4096 elements; if the number below is good the next step is a
parity run over `tests/test_mimo_v2_*`, not this file.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_compile_ab.py --steps 5 --rounds 2
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


def arm(step, cache, position: int, steps: int, warmup: int, logits):
    """`steps` measured steps after `warmup`, split into the host's time and the queue behind it."""
    step_position = position
    for _ in range(warmup):
        logits = step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        step_position += 1
    torch.cuda.synchronize()
    host = tail = 0.0
    for _ in range(steps):
        began = time.perf_counter()
        logits = step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - began) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        step_position += 1
    return (host + tail) / steps, host / steps, tail / steps, step_position, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--mode", default="default", help="inductor mode: default, reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument(
        "--resident-rows",
        type=int,
        default=0,
        help="held experts a routed layer; the step compiled here is the one the set leaves, and "
        "the set is built before the first compiled call so its counters are not part of the graph",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        resident_rows=args.resident_rows,
    )
    torch.cuda.synchronize()

    span = max(args.prompt, 8) + 2 * (args.warmup + args.steps) * (args.rounds + 2) + 16
    cache = model.cache(span)
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

    if rank == 0:
        print(f"[r0] compiling `step` with mode={args.mode} fullgraph={args.fullgraph}", flush=True)
    began = time.perf_counter()
    compiled = torch.compile(model.step, mode=args.mode, fullgraph=args.fullgraph, dynamic=True)
    # The first call is the compile; it is on the clock here so that a failure is a failure and not
    # a silent fall back to eager inside the measurement below.
    try:
        compiled(int(logits.argmax()), start_pos=position, cache=cache)
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        print(f"[r{rank}] torch.compile refused `step`: {type(error).__name__}: {error}", flush=True)
        return 0
    if rank == 0:
        print(f"[r0] compiled in {time.perf_counter() - began:.1f}s", flush=True)

    took: dict[str, list[tuple[float, float, float]]] = {"eager": [], "compiled": []}
    for _ in range(args.rounds):
        for name, step in (("eager", model.step), ("compiled", compiled)):
            try:
                token, host, tail, position, logits = arm(
                    step, cache, position, args.steps, args.warmup, logits
                )
            except Exception as error:  # noqa: BLE001
                print(f"[r{rank}] the {name} arm failed: {type(error).__name__}: {error}", flush=True)
                raise
            took[name].append((token, host, tail))

    mean = {name: sum(entry[0] for entry in got) / len(got) for name, got in took.items()}
    for index in range(args.rounds):
        print(
            f"[r{rank}] round {index + 1}: eager {took['eager'][index][0]:7.1f} ms  "
            f"compiled {took['compiled'][index][0]:7.1f} ms",
            flush=True,
        )
    for name in ("eager", "compiled"):
        got = took[name]
        print(
            f"[r{rank}] {name:9s} {mean[name]:7.1f} ms a token  host "
            f"{sum(entry[1] for entry in got) / len(got):7.1f}  queue "
            f"{sum(entry[2] for entry in got) / len(got):5.1f}  {1000 / mean[name]:5.2f} tok/s",
            flush=True,
        )
    if "eager" in mean:
        print(
            f"[r{rank}] compiled is {mean['eager'] / mean['compiled']:.3f}x eager "
            f"({mean['compiled'] - mean['eager']:+.1f} ms)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
