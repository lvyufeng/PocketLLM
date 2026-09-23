#!/usr/bin/env python
"""Where a MiMo-V2.6 decode step's 275 ms goes, with the copy's share separated from its hiding.

`bench_mimo_v2_ep.py` reports a token as `attention` and `FFN and staging` and the second of those
holds the expert copy *and* everything the copy is not: the router, the kernel, the slot wait and the
collective. This probe separates them on one process and one card per rank.

The instrument is event pairs on the compute stream and not kernel timings:

* `stall` is an event recorded after `_stage` has enqueued the copy and an event recorded immediately
  before the kernel launch. The compute stream reaches the second one only once the wait on the copy
  event has cleared, so `stall` is the copy the kernel *waited* for -- the part of the copy that is
  not hidden behind the previous layer's kernel. If the slots are doing their job this is small and
  the layer is the kernel; if it is not, this is the layer and the copy is the floor.
* `kernel` is the launch's own pair, and `reduce` is the collective's.
* `host` is wall time spent inside `_stage` on the host, which is enqueue work and does not stall the
  card unless it outruns it.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_phases.py --steps 8
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


class Counter:
    """Event pairs, summed by key, for one region of one step.

    The pairs are held and resolved after a synchronise rather than at the record, because
    `elapsed_time` refuses an event that has not completed -- and reading one early would be the
    stream drain this probe exists to avoid paying in the middle of a measured step.
    """

    def __init__(self) -> None:
        self.pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        # Host wall time, kept apart from `sums` because `read` rebuilds that from the pairs.
        self.host: dict[str, float] = {}

    def host_time(self, key: str, seconds: float) -> None:
        self.host[key] = self.host.get(key, 0.0) + seconds

    def record(self, key: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self.pairs.append((key, start, end))

    def read(self) -> None:
        self.sums.clear()
        self.counts.clear()
        for key, start, end in self.pairs:
            self.sums[key] = self.sums.get(key, 0.0) + start.elapsed_time(end)
            self.counts[key] = self.counts.get(key, 0) + 1
        self.pairs.clear()

    def clear(self) -> None:
        self.pairs.clear()
        self.sums.clear()
        self.counts.clear()


def instrument(experts, counter: Counter, device) -> None:
    """Separate `_stage`'s enqueue from the wait the kernel does on what it enqueued."""
    inner_stage = experts._stage
    inner_kernel = experts._kernel

    class Kernel:
        def __getattr__(self, name):
            inner = getattr(inner_kernel, name)
            # Both entry points: a decode step calls the single-token kernel and a prefill calls
            # the grouped one, and the question this probe asks -- how much of the copy the
            # compute waited for -- is the same question at either shape.
            if name not in ("moe_single_token_fp4_forward", "moe_multi_token_fp4_forward"):
                return inner

            def called(*args, **kwargs):  # noqa: ANN002, ANN003
                stream = torch.cuda.current_stream(device)
                before = torch.cuda.Event(True)
                before.record(stream)
                counter.record("stall", pending["before"], before)
                out = inner(*args, **kwargs)
                after = torch.cuda.Event(True)
                after.record(stream)
                counter.record("kernel", before, after)
                return out

            return called

    pending: dict[str, torch.cuda.Event] = {}
    kernel = Kernel()

    def staged(slot, layer_id, experts_arg):
        host = time.perf_counter()
        inner_stage(slot, layer_id, experts_arg)
        counter.host_time("stage", time.perf_counter() - host)
        # The compute stream is now, in stream time, just before the wait that follows.
        before = torch.cuda.Event(True)
        before.record(torch.cuda.current_stream(device))
        pending["before"] = before

    experts._stage = staged
    experts._kernel = kernel
    return inner_stage, inner_kernel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--slots", type=int, default=2)
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
        checkpoint, device=device, expert_source=bank, ep=ep, slots=args.slots
    )
    torch.cuda.synchronize()
    experts = model.experts
    print(
        f"[r{rank}] world {world} deal `{experts.deal}`, {experts.arena_bytes / 2**20:.0f} MiB arena, "
        f"{model.memory_bytes / 2**20:.0f} MiB on the card, {experts.slots} slots",
        flush=True,
    )

    cache = model.cache(args.prompt + args.steps + 8)
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    model.greedy(prompt_ids, max_tokens=1, cache=cache)
    cache.reset()
    logits = None
    for position, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    counter = Counter()
    instrument(experts, counter, device)

    # The layer's other two regions, so that "everything else" has a name: the attention is the
    # module's own forward and the router is the gate the MLP dispatches on.
    def timed(module, name, key):
        inner = getattr(module, name)

        def called(*args, **kwargs):  # noqa: ANN002, ANN003
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(True)
            start.record(stream)
            out = inner(*args, **kwargs)
            end = torch.cuda.Event(True)
            end.record(stream)
            counter.record(key, start, end)
            return out

        setattr(module, name, called)

    for layer in model.layers:
        timed(layer.attention, "forward", "attn")
        if layer.kind != "dense":
            timed(layer, "route", "route")

    # The collective, wrapped where the model calls it.
    def reduce_wrap():
        original = ep.reduce

        def called(tensor):
            if world == 1:
                return tensor
            stream = torch.cuda.current_stream(device)
            start = torch.cuda.Event(True)
            start.record(stream)
            began = time.perf_counter()
            out = original(tensor)
            counter.host_time("reduce", time.perf_counter() - began)
            end = torch.cuda.Event(True)
            end.record(stream)
            counter.record("reduce", start, end)
            return out

        return called

    if world > 1:
        model.ep.reduce = reduce_wrap()

    drawn = [int(logits.argmax())]
    counter.clear()
    if world > 1:
        torch.distributed.barrier()
    started = time.perf_counter()
    for step in range(args.steps):
        logits = model.step(drawn[-1], start_pos=len(prompt_ids) + step, cache=cache)[-1]
        drawn.append(int(logits.argmax()))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    counter.read()

    steps = args.steps
    token_ms = seconds / steps * 1e3
    print(f"\n[r{rank}] decode {steps} steps in {seconds:.3f}s = {token_ms:.1f} ms a token", flush=True)
    kernel_ms = counter.sums.get("kernel", 0.0) / steps
    stall_ms = counter.sums.get("stall", 0.0) / steps
    reduce_ms = counter.sums.get("reduce", 0.0) / steps
    host_ms = counter.host.get("stage", 0.0) / steps
    reduce_host_ms = counter.host.get("reduce", 0.0) / steps
    attn_ms = counter.sums.get("attn", 0.0) / steps
    route_ms = counter.sums.get("route", 0.0) / steps
    calls = counter.counts.get("kernel", 0) / steps
    print(
        f"[r{rank}] per token: kernel {kernel_ms:.1f} ms in {calls:.0f} calls, "
        f"copy stall {stall_ms:.1f} ms, collective {reduce_ms:.1f} ms, "
        f"attention {attn_ms:.1f} ms, router {route_ms:.1f} ms\n"
        f"[r{rank}] host per token: {reduce_host_ms:.1f} ms blocked inside the collective "
        f"(device {reduce_ms:.1f} ms), {host_ms:.1f} ms in `_stage`",
        flush=True,
    )
    accounted = kernel_ms + stall_ms + reduce_ms + attn_ms + route_ms
    print(
        f"[r{rank}] accounted {accounted:.1f} of {token_ms:.1f} ms "
        f"({accounted / token_ms * 100:.0f}%), rest {token_ms - accounted:.1f} ms "
        f"(dense linears, the head, the sampler, the host)",
        flush=True,
    )
    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(
            gathered,
            (
                round(token_ms, 1),
                round(attn_ms, 1),
                round(route_ms, 1),
                round(stall_ms, 1),
                round(kernel_ms, 1),
                round(reduce_ms, 1),
            ),
        )
        if rank == 0:
            print(
                f"[r0] (token, attn, router, stall, kernel, collective) by rank: {gathered}",
                flush=True,
            )
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
