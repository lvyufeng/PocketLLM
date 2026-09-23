#!/usr/bin/env python
"""What a decode token costs with pieces of it removed, measured without a profiler in the way.

The profiler answers "which op" and inflates the answer by forty percent, and the timeline probe
can only price the gaps it can see. Both agree the step is nothing like its kernels: 6,449 of them
a token and a step of 286 ms at four ranks, where the arithmetic is a handful of milliseconds.

So this measures the arms instead. One model, one cache, the same drawn token three times over, with
the routed experts replaced by the zero the deal's empty rank already returns and then the attention
replaced by a pass-through. The difference between two arms is what the piece costs *in situ*,
launches and stalls included, which is the number a fix has to move.

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_arms.py --steps 8 --prompt 8

The checkpoint is the default asset path; without it the script exits 0 and says so.
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


def time_steps(model, prompt_ids, steps) -> float:
    """Seconds for `steps` single-token steps, on a cache filled with the same prompt each time.

    A fresh cache an arm, because the arms leave the cache in different states -- a pass-through
    attention never appends its key -- and a shared one would compare two different contexts. The
    prompt is eight tokens, so refilling it is nothing.

    The token fed is the prompt's own draw and then itself: every step is the same shape, which is
    what the arms are about, and a live draw would let the arms diverge into different work.
    """
    cache = model.cache(len(prompt_ids) + steps + 8)
    for position, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
    token = int(logits.argmax())
    model.step(token, start_pos=len(prompt_ids), cache=cache)
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    started = time.perf_counter()
    for step in range(steps):
        model.step(token, start_pos=len(prompt_ids) + 1 + step, cache=cache)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="run `forward` through `torch.compile`, and time a fourth arm for it",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    rank, world = ep.rank, ep.world
    device = ep.device if ep.device is not None else torch.device("cuda:0")

    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(checkpoint, device=device, expert_source=bank, ep=ep)
    torch.cuda.synchronize()

    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]

    full = time_steps(model, prompt_ids, args.steps)

    # The experts replaced by the answer an empty rank already gives: this rank holding no share.
    # This is the whole table, and it is deliberately the only arm. An arm that also zeroes the
    # attention was tried and removed: the residual stream then stops moving, every layer routes
    # on the same row, and the arm measures a *different* model -- it came back 82 ms a token
    # *slower* than the arm that kept the attention, which is not a decomposition of anything.
    # A subtraction whose second term is not the same model is worse than no subtraction.
    experts = model.experts
    real_forward = experts.forward

    def stubbed(hidden, indices, weights, **kwargs):
        return torch.zeros((hidden.shape[0], experts.dim), dtype=torch.float32, device=hidden.device)

    experts.forward = stubbed
    without_experts = time_steps(model, prompt_ids, args.steps)
    experts.forward = real_forward

    step_ms = lambda seconds: seconds / args.steps * 1e3
    print(
        f"\n[r{rank}] {args.steps} steps an arm at {len(prompt_ids)} tokens of context, "
        f"world {world}",
        flush=True,
    )
    print(f"{'ms a token':>11} {'tok/s':>8}  arm")
    for name, seconds in (
        ("everything", full),
        ("the routed experts stubbed out", without_experts),
    ):
        print(
            f"{step_ms(seconds):11.1f} {args.steps / seconds:8.2f}  {name}",
            flush=True,
        )
    print(
        f"\n[r{rank}] the routed path -- the draw's copies, the kernel, and the host round trip "
        f"that decides them -- costs {step_ms(full) - step_ms(without_experts):.1f} ms a token; "
        f"the rest of the layer is {step_ms(without_experts):.1f}",
        flush=True,
    )

    if args.compile:
        # A step this size launches five thousand kernels, so the question a compiled arm answers
        # is not "is the arithmetic faster" -- it is whether the launches can be made to stop
        # being the arithmetic. `start_pos` is a Python int that changes every token, which is
        # exactly the argument dynamo specializes on, so the compile count is part of the answer.
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = 512
        model.forward = torch.compile(model.forward, dynamic=False)
        compiled = time_steps(model, prompt_ids, args.steps)
        print(
            f"{step_ms(compiled):11.1f} {args.steps / compiled:8.2f}  compiled forward",
            flush=True,
        )
        for counter, value in sorted(dynamo.utils.counters.items()):
            if value:
                print(f"[r{rank}] dynamo {counter}: {value}", flush=True)
    if world > 1:
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
