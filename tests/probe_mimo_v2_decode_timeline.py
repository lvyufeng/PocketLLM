#!/usr/bin/env python
"""The decode step's GPU timeline: how much of it the card is busy, and what the gaps wait on.

The op table says the collective's kernel occupies 182 ms of a step whose wall clock is 286 ms, and
the bare probe says the same all-reduce costs 135 us when the ranks are in step. Both cannot be
about NCCL: a kernel that spends its life spinning for a peer is *busy* and *idle* at once, and the
op table cannot tell the two apart because it bills the spin as device time.

This one reads the trace instead. Every kernel, memcpy and collective is placed on its stream with a
timestamp, so the step's wall clock is the union of those intervals and everything outside it is
either a gap the card was waiting through or a launch the host had not made yet. The gaps are
printed largest first with the kernels that bracket them, which is the question a fix has to answer.

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_timeline.py --steps 2

`--depth` is the same measurement over a prefix the prompt cannot reach, which is where the
question changes shape: at eight positions the card is behind the host and the host is the step,
and at 4096 the host spends 112 ms a token of which only six are the queue. Whether the *card* is
busy through those 112 ms is what says whether the remaining work is the host's Python or the
device's, and it is the one thing a wall clock around `model.step` cannot answer.

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


def intervals(prof) -> list[tuple[float, float, str]]:
    """Every interval that occupied the card, as `(start_us, end_us, name)`.

    Read from the profiler's own event list and not from the exported trace: the trace of a step
    this size is thirteen megabytes of JSON, and `export_chrome_trace` truncated it mid-object the
    one time it was asked. The individual events carry the same timestamps and no serialisation,
    which is the whole of what a timeline question needs.

    Host events are excluded -- a CPU op's interval is when the *launch* happened, and counting it
    as the card being busy is exactly the confusion this probe exists to undo.
    """
    from torch.profiler import DeviceType

    spans = []
    for event in prof.events():
        if event.device_type != DeviceType.CUDA:
            continue
        start, end = float(event.time_range.start), float(event.time_range.end)
        if end > start:
            spans.append((start, end, event.name))
    spans.sort()
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="fill the cache to this many positions with `bench_mimo_v2_model.fill_cache` "
        "instead of feeding a prompt, which is the answer at a depth a prompt cannot reach in a "
        "probe's lifetime: the question the section below asks -- whether the card is idle while "
        "the host is busy -- is not the same question at 8 positions and at 4096",
    )
    parser.add_argument(
        "--resident-rows",
        type=int,
        default=0,
        help="hold this many of each routed layer's hottest experts on the card; the gaps this "
        "probe prints are not the same shape with the copies thinned out",
    )
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
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        resident_rows=args.resident_rows,
    )
    if args.stub_experts:
        experts = model.experts

        def stubbed(hidden, indices, weights, **kwargs):
            return torch.zeros(
                (hidden.shape[0], experts.dim), dtype=torch.float32, device=hidden.device
            )

        experts.forward = stubbed
    torch.cuda.synchronize()

    cache = model.cache(max(args.depth, args.prompt) + 2 * args.steps + 16)
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    if args.depth:
        # `--depth` is the same step over a prefix that means nothing: the cache is filled
        # directly, which is what makes a 4096-position step seconds instead of minutes. Its own
        # `position` is the depth, and the prompt is not fed at all.
        fill_cache(cache, [layer.layer_idx for layer in model.layers], args.depth)
        position = args.depth
    else:
        for index, token in enumerate(prompt_ids):
            logits = model.forward(torch.tensor([token]), start_pos=index, cache=cache)[-1]
        position = len(prompt_ids)
    logits = model.step(prompt_ids[0], start_pos=position, cache=cache)[-1]
    torch.cuda.synchronize()

    drawn = [int(logits.argmax())]
    for offset in range(2):
        logits = model.step(drawn[-1], start_pos=position + 1 + offset, cache=cache)[-1]
        drawn.append(int(logits.argmax()))
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for step in range(args.steps):
            logits = model.step(drawn[-1], start_pos=position + 3 + step, cache=cache)[-1]
            drawn.append(int(logits.argmax()))
        torch.cuda.synchronize()

    spans = intervals(prof)
    if not spans:
        print(f"[r{rank}] the profile holds nothing that ran on the card")
        return 0

    first = min(span[0] for span in spans)
    last = max(span[1] for span in spans)
    wall = last - first
    # The union, not the sum: the copy stream and the compute stream overlap, and counting both
    # would price the same microsecond twice.
    busy = 0.0
    cursor = first
    for start, end, _name in spans:
        if end <= cursor:
            continue
        busy += end - max(start, cursor)
        cursor = max(cursor, end)

    if rank == 0:
        print(
            f"[r{rank}] {args.steps} steps: wall {wall / 1e3:.1f} ms "
            f"({wall / 1e3 / args.steps:.1f} ms a step), the card busy {busy / 1e3:.1f} ms "
            f"({busy / wall * 100:.1f}%), idle {wall - busy:.0f} us in "
            f"{len(spans)} intervals",
            flush=True,
        )
        # What the busy time is made of. The union above says the card is 72% busy; this says
        # busy with *what*, which is the other half of the question and the one a kernel answers.
        # The intervals overlap across streams, so this column sums to more than `busy` -- a
        # memcpy that ran under a kernel is counted in both, and that is the point: it says which
        # work exists, not which work is on the critical path.
        by_name: dict[str, list[float]] = {}
        for start, end, name in spans:
            by_name.setdefault(name, []).append(end - start)
        ranked = sorted(
            ((sum(sizes), len(sizes), name) for name, sizes in by_name.items()), reverse=True
        )
        print(
            f"\ncard time by interval name, the largest {args.top} of {len(ranked)}, "
            f"summed over {args.steps} step(s):"
        )
        print(f"{'ms total':>9} {'intervals':>10} {'us each':>9}  name")
        for total, count, name in ranked[: args.top]:
            print(f"{total / 1e3:9.2f} {count:10d} {total / count:9.1f}  {name[:60]}")

    # Every gap between the end of one interval and the start of the next that begins after it. A
    # gap's bracket is the kernel that finished and the kernel that waited, which is what names the
    # dependency the host or the fabric put in the middle.
    #
    # Billed by bracket and not one gap at a time, because the number of gaps is a launch count and
    # the question is which *pair* of ops the card keeps idling between: one 2.5 ms stall and four
    # thousand 50 us ones are different problems with different fixes, and a list sorted by size
    # shows only the first of them.
    pairs: dict[tuple[str, str], list[float]] = {}
    covered = first
    previous = "the start of the profile"
    for start, end, name in spans:
        if start > covered:
            pairs.setdefault((previous, name), []).append(start - covered)
        if end > covered:
            covered = end
            previous = name

    billed = sorted(
        ((sum(sizes), len(sizes), key) for key, sizes in pairs.items()), reverse=True
    )
    idle = sum(row[0] for row in billed)
    if rank == 0:
        print(
            f"\nidle {idle / 1e3:.1f} ms of the {wall / 1e3:.1f} ms wall in "
            f"{len(billed)} distinct brackets, priced by bracket:"
        )
        print(f"{'ms total':>9} {'gaps':>7} {'us each':>9}  ends -> begins")
        for total, count, (before, after) in billed[: args.top]:
            print(
                f"{total / 1e3:9.2f} {count:7d} {total / count:9.1f}  "
                f"{before[:44]} -> {after[:44]}",
                flush=True,
            )
    if ep.world > 1:
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
