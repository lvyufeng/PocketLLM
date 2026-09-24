#!/usr/bin/env python
"""How much of a MiMo-V2.6 token's expert copy a resident set on the cards would delete.

A decode token stages 1198.5 MiB a rank -- 47 layers of the two experts the deal gave it -- and
`bench_mimo_v2_staging_copy.py` says the link is at 10.4 GiB/s whatever the copy granularity is. So
the bytes are the floor and the only lever left on that floor is *not copying*: hold the experts the
router keeps asking for on the card and stage only the rest.

That is only worth anything if the router asks for the same experts again. `probe_mimo_v2_expert_reuse.py`
asked the *previous step* and found 9 to 13.5% of a rank's rows, which is why a prefetch cannot be
issued from it -- but a resident set is a different question with a different shape: it is not
"predict the next draw from the last one", it is "how much of the marginal distribution does the
hottest K experts of a layer cover", and a skewed enough distribution makes that far larger than a
uniform one would.

This probe answers that, on a real prompt and a real greedy continuation, four ranks. For each layer
it counts every expert the layer routed to over `--tokens` steps and reports, for a sweep of `K`:

* `hit` -- the share of that layer's draws that land on its top `K`, and the same over the whole
  model, which is the share of the 1198.5 MiB that would never be staged.
* `bytes` -- what a rank would move a token at that `K`, and the token's copy time at the 10.4 GiB/s
  the link sustains.
* `GiB` -- what the resident set costs the four cards in all, `47 * K * one expert`.

The top-`K` sets are chosen on the *same* steps they are scored on, which makes every number an
upper bound and not an estimate: a set chosen from one prompt need not hold on the next. The number
to act on is therefore the shape -- how fast the curve rises with `K` -- and not the last point.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_expert_residency.py --depth 8192 --tokens 128
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
MIB = 1 << 20

#: Where the residency curve is read: the sizes a card can actually afford.
SWEEP = (8, 16, 24, 32, 48, 64, 96, 128, 192)


def tokenize(path: str, checkpoint: str, want: int) -> list[int]:
    """`want` tokens of a real document, or `PROMPT_IDS` repeated if there is no tokenizer."""
    text = None
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    fallback = (PROMPT_IDS * (want // len(PROMPT_IDS) + 1))[:want]
    if text is None:
        return fallback
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return fallback
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ids = tokenizer(text)["input_ids"]
    if len(ids) < want:
        ids = (ids * (want // len(ids) + 1))[:want]
    return list(ids[:want])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--depth", type=int, default=8192, help="prompt tokens to prefill")
    parser.add_argument("--tokens", type=int, default=128, help="steps to count draws over")
    parser.add_argument("--chunk", type=int, default=2048, help="prefill chunk")
    parser.add_argument(
        "--deal",
        default="sorted",
        help="`sorted` (the served decode deal) or `id`; `id` cannot hold a set and a chunk band "
        "at once, so its arm prefills a row at a time",
    )
    parser.add_argument("--prompt-file", default="docs/models/mimo-v2.6-flash.md")
    parser.add_argument(
        "--resident-rows",
        type=int,
        default=0,
        help="also run the real thing: a set of this width, reporting what it actually hit",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    started = time.perf_counter()
    ep = EpGroup.from_env()
    device = ep.device if ep.device is not None else torch.device("cuda", ep.rank)
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    # A set and a chunk band want the same rows, so an `id` arm gives up the band and prefills a
    # row at a time. That is the cost of measuring the deal `id` gives a set, and it is paid once.
    band = None if args.deal == "id" else args.chunk
    step = 1 if band is None else args.chunk
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        deal=args.deal,
        chunk_rows=band,
        resident_rows=args.resident_rows,
    )
    torch.cuda.synchronize()
    experts = model.experts
    per_expert = experts.expert_bytes
    top_k = model.config.num_experts_per_tok
    print(
        f"[r{ep.rank}] world {ep.world}, deal `{experts.deal}`, {top_k} of "
        f"{model.config.n_routed_experts} experts a layer, {per_expert / MIB:.2f} MiB an expert, "
        f"built in {time.perf_counter() - started:.0f}s",
        flush=True,
    )

    prompt = tokenize(args.prompt_file, args.checkpoint, args.depth)
    position = len(prompt)
    cache = model.cache(args.depth + args.tokens + 8)
    logits = model.prefill(prompt, cache=cache, chunk=step)
    torch.cuda.synchronize()
    if ep.world > 1:
        torch.distributed.barrier()

    # Every draw of every layer, as the *host* sees it: the ids the layer chose, whatever the deal
    # did with them. A resident set is a property of the layer and not of a rank's share of it.
    counts: dict[int, list[int]] = {}

    def capturing(inner):
        def forward(hidden, indices, weights, **kwargs):
            layer = kwargs.get("layer_id")
            drawn = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            row = counts.setdefault(int(layer), [0] * model.config.n_routed_experts)
            for expert in drawn:
                row[int(expert)] += 1
            return inner(hidden, indices, weights, **kwargs)

        return forward

    experts.forward = capturing(experts.forward)
    # The set's own counters, taken before the decode loop so that the rate below is the rate a
    # *decode* draw saw. A prefill through the same module is thousands of draws over a short
    # context and it would otherwise be most of the denominator.
    before = experts.resident_report() if args.resident_rows else None
    for _ in range(args.tokens):
        logits = model.step(int(logits.argmax()), start_pos=position, cache=cache)[-1]
        position += 1
    torch.cuda.synchronize()
    if ep.world > 1:
        torch.distributed.barrier()

    layers = sorted(counts)
    total_draws = sum(sum(counts[layer]) for layer in layers)
    print(
        f"[r{ep.rank}] {args.tokens} steps over {len(layers)} routed layers, "
        f"{total_draws} drawings, {total_draws / args.tokens:.1f} a layer a step",
        flush=True,
    )
    if ep.rank == 0:
        print(f"{'K':>5} {'hit':>8} {'MiB a rank a token':>19} {'copy ms':>8} {'GiB resident':>13}")
        for k in SWEEP:
            hits = sum(sum(sorted(row, reverse=True)[:k]) for row in counts.values())
            share = hits / total_draws
            staged = sum(sum(row) - sum(sorted(row, reverse=True)[:k]) for row in counts.values())
            # A rank's own share of the drawings, and a *token's* worth of them: `staged` is every
            # copy the run made, which is `--tokens` tokens of them and four ranks' worth.
            rank_bytes = staged / ep.world / args.tokens * per_expert
            print(
                f"{k:5d} {share * 100:7.1f}% {rank_bytes / MIB:19.0f} "
                f"{rank_bytes / (10.4 * 2**30) * 1e3:8.1f} "
                f"{len(layers) * k * per_expert / 2**30:13.1f}",
                flush=True,
            )
        # The shape of the marginal distribution itself, per layer: how concentrated the router is.
        effective = []
        for layer in layers:
            row = counts[layer]
            total = sum(row)
            effective.append(1.0 / sum((value / total) ** 2 for value in row))
        if before is not None:
            report = experts.resident_report()
            hits = report["hits"] - before["hits"]
            drawn = report["drawn"] - before["drawn"]
            print(
                f"[r0] the set at {args.resident_rows} rows a layer, over the {args.tokens} "
                f"decode steps: hit {100 * hits / max(drawn, 1):.1f}% of this rank's {drawn:.0f} "
                f"draws, {report['held']:.0f} rows held, {report['swaps']:.0f} swaps",
                flush=True,
            )
        effective.sort()
        print(
            f"[r0] effective experts a layer (1/Sum p^2): min {effective[0]:.1f}, "
            f"median {effective[len(effective) // 2]:.1f}, max {effective[-1]:.1f}, "
            f"uniform would be {model.config.n_routed_experts}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
