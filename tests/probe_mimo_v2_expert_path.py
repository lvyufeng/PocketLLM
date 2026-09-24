#!/usr/bin/env python
"""Where the expert wrapper's 73 ms a token goes, inside the call that spends it.

`probe_mimo_v2_host_phases.py` wraps `MimoV2DeviceExperts.forward` and reports 73.1 ms a token for
it, of which `_stage` is 16.4 and the kernel's own entry 3.7. That leaves 53 ms inside one Python
function that is thirty lines long, and the list of things in it is short: a `tolist`, a deal, the
slot bookkeeping, a `wait_event`, an `index_select` and the kernel call. A number that large has to
be a *sync*: `drawn = indices.tolist()` is a device-to-host round trip, and a round trip on a stream
that has work queued on it is the queued work's time arriving on the host.

This probe re-implements `forward`'s body with a `perf_counter` between each step, so the 53 ms is
attributed instead of inferred. The re-implementation is a copy and it is the risk of the probe: a
region that is timed around the wrong call is a region that is priced wrong. What makes the copy
safe is that its total is printed next to the shipped method's own total, measured in the same
window; if the two are not within a few percent, the copy has drifted and the split below is not
read.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_expert_path.py --steps 8
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
from src.models.mimo_v2.ep import EpGroup, owned_positions  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]

#: `(name, what it covers)`, in the order `forward` pays them.
STEPS = (
    ("tolist", "`indices.tolist()` -- the draw, read back to the host, and a sync"),
    ("deal", "`owned_positions` and the two checks around it"),
    ("slots", "`_take_slots`"),
    ("stage", "`_stage`: the six copies an expert, on the copy stream"),
    ("wait+select", "`compute.wait_event` and `weights.index_select(0, picked)`"),
    ("kernel", "the kernel entry, which builds its own static buffers and launches"),
)


def instrument(experts, tally: dict[str, list[float]]) -> None:
    """Replace `forward` with the same body, timed a step at a time."""
    shipped = experts.forward
    arena_rows, top_k = experts.arena_rows, experts.top_k
    rank, world, deal = experts.rank, experts.world, experts.deal
    picked_host, picked_device = experts._picked_host, experts._picked_device
    rows_all = experts._rows
    kernel = experts._kernel

    def forward(hidden, indices, weights, *, layer_id=None):
        layer = experts.layer_id if layer_id is None else int(layer_id)
        indices = indices.reshape(-1)
        weights = weights.reshape(-1)

        began = time.perf_counter()
        drawn = indices.tolist()
        now = time.perf_counter()
        tally["tolist"].append(now - began)

        mine = owned_positions(drawn, rank=rank, world=world, deal=deal)
        took = time.perf_counter()
        tally["deal"].append(took - now)
        if not mine:
            return torch.zeros((1, experts.dim), dtype=torch.float32, device=experts.device)

        slot = experts._take_slots(len(mine))
        now = time.perf_counter()
        tally["slots"].append(now - took)
        experts._stage(slot, layer, [drawn[position] for position in mine])
        took = time.perf_counter()
        tally["stage"].append(took - now)

        arena = experts._arenas[slot]
        compute = torch.cuda.current_stream(experts.device)
        compute.wait_event(experts._copy_events[slot])
        picked_host[: len(mine)] = torch.tensor(mine, dtype=torch.int64)
        picked = picked_device[: len(mine)].copy_(picked_host[: len(mine)], non_blocking=True)
        rows = rows_all[: len(mine)]
        now = time.perf_counter()
        tally["wait+select"].append(now - took)

        out = kernel.moe_single_token_fp4_forward(
            hidden.to(experts.device),
            rows,
            weights.index_select(0, picked).to(experts.device, dtype=torch.float32),
            arena[("gate_proj", "weight")],
            arena[("gate_proj", "weight_scale")],
            arena[("down_proj", "weight")],
            arena[("down_proj", "weight_scale")],
            arena[("up_proj", "weight")],
            arena[("up_proj", "weight_scale")],
            0,
            15.0,
        )
        tally["kernel"].append(time.perf_counter() - now)
        experts._events[slot].record(compute)
        experts._pending[slot] = True
        return out

    experts.forward = forward
    del arena_rows, top_k, shipped


def arm(model, cache, position: int, steps: int, warmup: int, logits):
    step_position = position
    for _ in range(warmup):
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        step_position += 1
    torch.cuda.synchronize()
    host = tail = 0.0
    for _ in range(steps):
        began = time.perf_counter()
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
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

    cache = model.cache(max(args.prompt, 8) + 2 * (args.warmup + args.steps) + 16)
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

    # The shipped method's own total first, so the copy below has something to be checked against.
    tally: dict[str, list[float]] = {name: [] for name, _ in STEPS}
    _, host, tail, position, logits = arm(model, cache, position, args.steps, args.warmup, logits)
    shipped_ms = host + tail
    print(f"[r{rank}] shipped: {shipped_ms:7.1f} ms a token", flush=True)

    instrument(model.experts, tally)
    _, host, tail, position, logits = arm(model, cache, position, args.steps, args.warmup, logits)
    copied_ms = host + tail
    print(
        f"[r{rank}] the copy of `forward`: {copied_ms:7.1f} ms a token "
        f"({copied_ms - shipped_ms:+.1f} against the shipped one)",
        flush=True,
    )

    # `arm` counts the warmup calls into the tally too, so the divisor is the arm's whole run and
    # not the measured window. The two totals above are the check that the copy is faithful; this
    # is the divisor that keeps the split honest.
    seen = args.steps + args.warmup
    routed = len(model.layers) - 1
    total = 0.0
    for name, what in STEPS:
        got = tally[name]
        if not got:
            continue
        per_step = sum(got) / seen * 1e3
        total += per_step
        print(
            f"[r{rank}]   {name:12s} {per_step:7.1f} ms a token  "
            f"{len(got) / args.steps:5.1f} calls  max {max(got) * 1e3:6.3f} ms  {what}",
            flush=True,
        )
    print(
        f"[r{rank}]   {'sum':12s} {total:7.1f} ms a token over {routed} layers",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
