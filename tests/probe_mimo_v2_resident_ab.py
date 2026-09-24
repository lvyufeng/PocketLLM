#!/usr/bin/env python
"""Is the resident expert set free of the bytes it removes, and what does it buy?

The claim the resident set rests on is an equality and not a tolerance: a resident row holds the
bytes the staging row would have held, the kernel is handed rows, so an expert that was kept on
the card is the same arithmetic as an expert that was copied. If that is true then the two arms
below -- the same model, the same weights, the same draws, only `experts._residents` set or
cleared -- must agree on every logit to the last bit, and any policy can be scored without a
correctness argument.

The A/B is in one process on purpose. This box has reported the same configuration 184 and 245 ms
an afternoon apart, so a number is only readable next to a number taken minutes before it in the
same process; and because a *resident* row and a *staging* row are disjoint ranges of the same
arena, clearing `_residents` mid-run is the shipped path exactly -- `forward` then stages every
draw into the slot's own rows, which is what it did before residents existed. The only difference
between the arms is the copies the set removes.

The token chain is fixed rather than sampled, so both arms see the same draws at the same
position and the comparison is about residency and not about the sampler's luck. It is derived
from one greedy pass, then replayed into a reset cache a round at a time, which is also what makes
the timing interleaved: `off, on, off, on, ...` so the round's own drift lands on both arms.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_resident_ab.py --resident-rows 16 --steps 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_experts import _Residents  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def walk(model, cache, prompt_ids, chain, position, *, steps=None):
    """Prefill the prompt from a reset cache, then replay `chain`, timing the replayed steps.

    `chain` is fed rather than sampled, so the position sequence is identical between calls and
    two calls differ only in the state the module was left in. The returned logits are the last
    one each step produced, which is what the equality is read off.
    """
    cache.reset()
    logits = None
    for index, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=index, cache=cache)[-1]
    torch.cuda.synchronize()
    host = tail = 0.0
    seen = chain if steps is None else chain[:steps]
    outs = []
    for offset, token in enumerate(seen):
        began = time.perf_counter()
        logits = model.forward(torch.tensor([token]), start_pos=position + offset, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - began) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        outs.append(logits.clone())
    return outs, host / max(1, len(seen)), tail / max(1, len(seen))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--resident-rows", type=int, default=16)
    parser.add_argument("--chain", type=int, default=0, help="tokens in the replayed chain")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    rows = args.resident_rows
    model = MimoV2DeviceModel(checkpoint, device=device, expert_source=bank, ep=ep, resident_rows=rows)
    torch.cuda.synchronize()
    module = model.experts
    if module.resident_rows != rows:
        print(f"[r{rank}] the module was built with {module.resident_rows} rows, not {rows}")
        return 1

    free, total = torch.cuda.mem_get_info(device)
    print(
        f"[r{rank}] {rows} rows a routed layer x {module.resident_layers} layers = "
        f"{module.resident_bytes / 2**30:.2f} GiB resident, arena "
        f"{module.arena_bytes / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free of {total / 2**30:.2f}",
        flush=True,
    )

    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    chain_tokens = args.chain or max(args.steps * (args.rounds + 1), 4 * args.steps)
    span = args.prompt + chain_tokens + 8
    cache = model.cache(span)

    # The chain, derived once with the set off: every arm replays it, so the draws are the same.
    module._residents = None
    logits = None
    for index, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=index, cache=cache)[-1]
    chain = []
    for index in range(chain_tokens):
        token = int(logits.argmax())
        chain.append(token)
        logits = model.forward(torch.tensor([token]), start_pos=args.prompt + index, cache=cache)[-1]
    torch.cuda.synchronize()
    print(f"[r{rank}] the chain is {chain[:8]}...", flush=True)
    if world > 1:
        torch.distributed.barrier()

    base, _, _ = walk(model, cache, prompt_ids, chain, args.prompt)
    fresh = _Residents(rows, module.resident_layers)
    module._residents = fresh
    got, _, _ = walk(model, cache, prompt_ids, chain, args.prompt)
    worst = 0.0
    exact = True
    for index, (a, b) in enumerate(zip(base, got)):
        exact = exact and bool(torch.equal(a, b))
        worst = max(worst, float((a - b).abs().max()))
    print(
        f"[r{rank}] {len(base)} steps: bit-exact {exact}, worst |delta| {worst:.3e}, "
        f"resident hits {fresh.hits} of {fresh.drawn} draws, {fresh.held} rows held, "
        f"{fresh.swaps} swaps",
        flush=True,
    )
    if world > 1:
        torch.distributed.barrier()

    # Timing, interleaved by round so the drift lands on both arms.
    took: dict[str, list[tuple[float, float, float]]] = {"off": [], "on": []}
    for _ in range(args.rounds):
        module._residents = None
        _, host, tail = walk(model, cache, prompt_ids, chain, args.prompt, steps=args.steps)
        took["off"].append((host + tail, host, tail))
        module._residents = fresh
        _, host, tail = walk(model, cache, prompt_ids, chain, args.prompt, steps=args.steps)
        took["on"].append((host + tail, host, tail))
    if world > 1:
        torch.distributed.barrier()

    for name in ("off", "on"):
        got = took[name]
        token = sum(entry[0] for entry in got) / len(got)
        host = sum(entry[1] for entry in got) / len(got)
        tail = sum(entry[2] for entry in got) / len(got)
        print(
            f"[r{rank}] residents {name:3s} {token:7.1f} ms a step  host {host:7.1f}  "
            f"queue {tail:5.1f}  {1000 / token:5.2f} tok/s",
            flush=True,
        )
    off = sum(entry[0] for entry in took["off"]) / len(took["off"])
    on = sum(entry[0] for entry in took["on"]) / len(took["on"])
    print(
        f"[r{rank}] {rows} rows a layer is {on - off:+7.1f} ms a step ({on / off:.3f}x), "
        f"copies a step {module.staged_experts} experts",
        flush=True,
    )
    report = module.resident_report()
    print(f"[r{rank}] policy {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
