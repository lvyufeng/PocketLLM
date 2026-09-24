#!/usr/bin/env python
"""Where a decode step's attention gets slower with depth, layer by layer and by family.

`probe_mimo_v2_host_phases.py` says the attention is 33.9 ms of a short-context token's host and
41.5 to 42.9 at 4096. Which layers that is matters, because the two families are not the same case
and the model does not treat them the same way: a windowed layer reads a 128-slot ring whatever the
context, and a global layer reads the whole prefix -- and above `FOLD_KEYS` a global layer's decode
step stops taking `decode_output` at all and falls back to `attention_output`'s chunk path, which is
`apply_partial_rope` with its broadcast axes, the bounds, and the transposes. Thirty-nine of the
forty-eight layers are windowed and nine are global, so a table of all forty-eight is what says
whether the 8 ms is the nine or the thirty-nine.

**Two caches, one process.** The comparison this probe exists for is depth against no depth, and this
box has measured the same configuration 10 to 15 ms apart between processes -- so the two caches
live in one run, one filled to `--depth` and one fed `--prompt` tokens, and every step is timed on
both. `--depth` is reached with `bench_mimo_v2_model.fill_cache` and not with a prefill, which is
what makes a 4096 step a few seconds rather than a few minutes; the step it measures is the same
step -- the same bytes the attention reads, the same experts staged -- over a prefix that means
nothing.

The number a layer reports is the whole of `attention.forward` around the clock, taken as the best
of the rounds rather than the mean, because the host's own jitter is asymmetric and a minimum is the
statistic that survives it. `--rounds` is what says whether a difference is one.

**A null and a confound.** Four arms run, and two of them -- `short` and `short-nofold` -- are one
configuration: at eight positions the fold bound is not in question, so both take `decode_foldable`
and every layer on both walks the same path. The gap between them is therefore this run's noise with
the code under test held fixed, and it is printed as `the null`. A delta which is not larger than
the null is not a measurement of anything, and a run whose null is milliseconds wide -- this box
does that, and the round's own first-arm penalty is most of it, which is why the arms rotate with
the round -- has nothing to say about a lever smaller than that.

The confound is the same one every arm here has: the two caches hold different hidden states, so
their routers draw different experts, so they stage different bytes. Copies are the largest single
term in this token, so the staged count and the resident hit count are printed beside every arm's
clock; an arm that answers more of its draws on the card is a faster arm without the code under
test having done anything. The `fold` column of the table is the difference between the same layer
on two arms whose copies agree -- read it only where they do.

Every arm also prints the branch it actually took, per family. `None` of the deltas above mean
anything without it: `set_fold` patches a predicate three frames up and `attention_output` has a
second clause of its own -- the RoPE table has to reach the position -- so a patch that misses, a
table that is too short, and an effect that is genuinely absent all look like the same column of
zeros.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_attention_depth.py --depth 4096 --rounds 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.bank import open_expert_bank  # noqa: E402
from src.models.mimo_v2.device_attention import (  # noqa: E402
    FOLD_KEYS,
    MimoV2DeviceAttention,
)
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.ep import EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.bench_mimo_v2_model import fill_cache  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def meter(model) -> dict[int, float]:
    """Wrap every layer's attention around the clock, keyed by `layer_idx`."""
    spent: dict[int, float] = {}

    for layer in model.layers:
        attention = layer.attention
        was = attention.forward
        index = layer.layer_idx

        def timed(*args, _was=was, _index=index, **kwargs):  # noqa: ANN002, ANN003
            began = time.perf_counter()
            out = _was(*args, **kwargs)
            spent[_index] = spent.get(_index, 0.0) + (time.perf_counter() - began) * 1e3
            return out

        attention.forward = timed
    return spent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--depth", type=int, default=4096, help="context the long cache is filled to")
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--resident-rows", type=int, default=16)
    parser.add_argument(
        "--force-decode",
        action="store_true",
        help="answer `decode_foldable` true for every step and clear `_decode_ops`, which puts a "
        "global layer's long span on `decode_output`'s torch softmax block instead of the chunk "
        "path it takes above `FOLD_KEYS`. The kernel is cleared rather than left in because the "
        "span this is asking about is past its shared-memory bound, so the arm prices the decode "
        "path's *arithmetic* and not the kernel's dispatch -- the two are the same block",
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
    layers = [layer.layer_idx for layer in model.layers]
    families = {layer.layer_idx: layer.kind for layer in model.layers}
    windows = {layer.layer_idx: layer.attention.shape.sliding_window for layer in model.layers}

    # Two caches and two positions. The short one is fed `--prompt` real tokens so its step is a
    # step the model would take; the long one is filled, so its step is a step of the right shape.
    # One cache an arm, because two arms sharing one would interleave their appends and a slot
    # would stop meaning the position it was written for. `--rounds` rounds of `--steps` steps each,
    # plus the warmup, plus the prompt the short arms are fed: the capacity is the whole run and not
    # one round of it.
    walked = args.warmup + args.rounds * args.steps
    arms = ("long", "long-nofold", "short", "short-nofold")
    caches = {
        name: model.cache((args.depth if name.startswith("long") else args.prompt) + walked + 8)
        for name in arms
    }
    # The tables are shared per *family* and `cache()` re-shares them every time it is called, so
    # four caches leave the last one's capacity live for the whole stack -- forty-four positions
    # here. `attention_output` stands aside for a step whose `start_pos` is past the table, so
    # without this line every 4096-position step takes the chunk path and neither the fold bound
    # nor `decode_output` is reachable, whatever `decode_foldable` answers. One capacity for the
    # run, the longest one, and the tally below is what says it took.
    model.share_rope_tables(args.depth + walked + 8)
    model.greedy(PROMPT_IDS[: args.prompt], max_tokens=1, cache=caches["short"])
    logits = None
    for name in ("short", "short-nofold"):
        caches[name].reset()
        for position, token in enumerate(PROMPT_IDS[: args.prompt]):
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=caches[name])[-1]
    for name in ("long", "long-nofold"):
        fill_cache(caches[name], layers, args.depth)
    torch.cuda.synchronize()
    print(
        f"[r{rank}] {len(layers)} layers, {args.resident_rows} resident rows a layer, "
        f"short caches at {args.prompt}, long caches at {args.depth}, "
        f"{caches['long'].memory_bytes / 2**30:.2f} GiB of long cache",
        flush=True,
    )
    if world > 1:
        torch.distributed.barrier()

    if args.force_decode:
        MimoV2DeviceAttention.decode_foldable = lambda self, start_pos, cache: True
        for layer in model.layers:
            layer.attention._decode_ops = None
        print(
            f"[r{rank}] every step takes the one-row path, on the torch block: a global layer's "
            f"span past `FOLD_KEYS` is not something `decode_foldable` admits on its own",
            flush=True,
        )

    spent = meter(model)
    best: dict[str, dict[int, float]] = {name: {} for name in arms}
    attentions: dict[str, list[float]] = {name: [] for name in arms}
    tokens: dict[str, list[float]] = {}
    was_fold = MimoV2DeviceAttention.decode_foldable

    # The other thing that moves a token: how many of the eight draws a layer a shared expert came
    # over PCIe and how many the resident set answered. The two caches route over different hidden
    # states, so a pair of arms is only a pair of arms if these agree -- the counters are printed
    # beside the clock rather than written into a caveat, because on this stack the copies are
    # larger than every kernel here.
    module = model.experts
    residents = module._residents
    copies: dict[str, list[tuple[int, int, int]]] = {name: [] for name in arms}

    def counters() -> tuple[int, int, int]:
        if residents is None:
            return module.staged_experts, 0, 0
        return module.staged_experts, residents.hits, residents.drawn

    # And the branch each arm actually took, per family, because a delta of nothing is either a
    # lever worth nothing or a lever that was never pulled -- `set_fold` patches a predicate three
    # frames up, and a patch that misses looks exactly like an effect that is absent. `last_stats`
    # is what `decode_output` and `attention_output` each stamp, so the tally is the proof.
    paths: dict[str, dict[str, int]] = {name: {} for name in arms}

    def tally(name: str) -> None:
        for layer in model.layers:
            family = "windowed" if windows[layer.layer_idx] is not None else "global"
            key = f"{family}:{layer.attention.last_stats.path}"
            paths[name][key] = paths[name].get(key, 0) + 1

    def set_fold(past: bool) -> None:
        """`past=False` is the bound before this change: `FOLD_KEYS` for every one-row step."""
        if past:
            MimoV2DeviceAttention.decode_foldable = was_fold
        else:
            MimoV2DeviceAttention.decode_foldable = (
                lambda self, start_pos, cache: min(
                    int(start_pos), cache.slots(self.layer_idx)
                ) + 1 <= FOLD_KEYS
            )

    # Four arms, interleaved round by round: the two caches, and the fold bound on and off. The
    # depth question and the bound question are one process apart from each other and from the
    # round's own drift, which on this box is larger than either.
    positions = {
        name: (args.depth if name.startswith("long") else args.prompt) for name in arms
    }
    for name in arms:
        tokens[name] = []
    for name in arms:  # a warm pass each, so no arm starts cold
        set_fold(name != "long-nofold" and name != "short-nofold")
        position = positions[name]
        for _ in range(args.warmup):
            logits = model.step(int(logits.argmax()), start_pos=position, cache=caches[name])[-1]
            position += 1
        positions[name] = position
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    for round_index in range(args.rounds):
        # The order rotates with the round. Whoever goes first in a round pays for whatever the
        # round's first barrier and first kernel launch cost, and over four rounds that is the
        # largest single term in the delta -- larger than either question being asked.
        for name in arms[round_index % len(arms):] + arms[:round_index % len(arms)]:
            set_fold(name != "long-nofold" and name != "short-nofold")
            position = positions[name]
            spent.clear()
            before = counters()
            began = time.perf_counter()
            for _ in range(args.steps):
                logits = model.step(int(logits.argmax()), start_pos=position, cache=caches[name])[-1]
                position += 1
            # The wall the whole step took and not the attention's share of it: synchronised after
            # the clock, so what is measured is a step and the queue behind it.
            queued = time.perf_counter()
            torch.cuda.synchronize()
            after = counters()
            tally(name)
            positions[name] = position
            tokens[name].append((queued - began) * 1e3 / args.steps)
            copies[name].append(tuple(b - a for a, b in zip(before, after)))
            for index, ms in spent.items():
                best[name][index] = min(best[name].get(index, ms / args.steps), ms / args.steps)
            attentions[name].append(sum(spent.values()) / args.steps)
        if world > 1:
            torch.distributed.barrier()
    set_fold(True)

    for name in arms:
        got = tokens[name]
        staged, hits, drawn = (sum(c[i] for c in copies[name]) for i in range(3))
        print(
            f"[r{rank}] {name:12s} token: best {min(got):7.1f} ms ({1000 / min(got):5.2f} tok/s), "
            f"mean {sum(got) / len(got):7.1f} ({1000 * len(got) / sum(got):5.2f}), over "
            f"{args.rounds} rounds ({', '.join(f'{entry:.1f}' for entry in got)}); "
            f"{staged:6d} staged, {hits:6d}/{drawn} resident",
            flush=True,
        )
    short = sum(tokens["short"]) / args.rounds
    long_ = sum(tokens["long"]) / args.rounds
    print(
        f"[r{rank}] the depth costs {long_ - short:+7.1f} ms a token at the mean, and the fold "
        f"bound is worth {sum(tokens['long-nofold']) / args.rounds - long_:+7.1f} at {args.depth} "
        f"against {sum(tokens['short-nofold']) / args.rounds - short:+7.1f} at {args.prompt}",
        flush=True,
    )
    # `short` and `short-nofold` are the same configuration -- both take `was_fold` at eight
    # positions, where the bound is not in question -- so the gap between them is this run's own
    # noise, measured rather than assumed. A delta in the row above that is not larger than this
    # number is a delta this run cannot read, on either the depth or the bound.
    null = sum(tokens["short-nofold"]) / args.rounds - short
    print(
        f"[r{rank}] the null, two arms that are one configuration: {null:+7.1f} ms a token, so "
        f"nothing under {abs(null):.1f} ms in this run is a measurement",
        flush=True,
    )
    for name in arms:
        took = ", ".join(f"{key} {count}" for key, count in sorted(paths[name].items()))
        print(f"[r{rank}] {name:12s} took {took}", flush=True)

    families = {layer.layer_idx: layer.kind for layer in model.layers}
    windows = {layer.layer_idx: layer.attention.shape.sliding_window for layer in model.layers}
    print(f"\n[r{rank}] the attention a layer, best of {args.rounds} rounds of {args.steps} steps")
    print(
        f"[r{rank}] {'layer':>5} {'family':>8} {'window':>7} {'short':>8} {'long':>8} "
        f"{'delta':>8} {'fold':>8}"
    )
    for index in layers:
        window = windows[index]
        short_ms = best["short"].get(index, 0.0)
        long_ms = best["long"].get(index, 0.0)
        loose = best["long-nofold"].get(index, 0.0)
        print(
            f"[r{rank}] {index:5d} {families[index]:>8} "
            f"{('-' if window is None else str(window)):>7} "
            f"{short_ms:8.3f} {long_ms:8.3f} {long_ms - short_ms:+8.3f} "
            f"{loose - long_ms:+8.3f}",
            flush=True,
        )
    for name in arms:
        got = attentions[name]
        print(
            f"[r{rank}] {name:12s} attention a step: best {min(got):7.2f} ms, mean "
            f"{sum(got) / len(got):7.2f}, over {args.rounds} rounds "
            f"({', '.join(f'{entry:.1f}' for entry in got)})",
            flush=True,
        )
    windowed = [i for i in layers if windows[i] is not None]
    global_ = [i for i in layers if windows[i] is None]
    for domain, group in (("windowed", windowed), ("global", global_)):
        for stem, other in (("long", "long-nofold"), ("short", "short-nofold")):
            folded = sum(best[stem].get(i, 0.0) for i in group)
            loose = sum(best[other].get(i, 0.0) for i in group)
            depth = args.depth if stem.startswith("long") else args.prompt
            print(
                f"[r{rank}] {domain:8s} {len(group):2d} layers at {depth:5d}: fold-past {folded:7.2f} "
                f"ms ({folded / len(group):5.3f} a layer), the old bound {loose:7.2f} "
                f"({loose / len(group):5.3f} a layer), {loose - folded:+7.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
