#!/usr/bin/env python
"""How much of a decode step's expert copy a rank has already staged once before.

A decode step at 262144 costs 204.7 ms, and `probe_mimo_v2_decode_phases.py` says what is in it:
**the copy stall is 99.9 of it** -- the part of the expert H2D the kernel actually waited for -- and
nothing else is close (attention 45.3, kernel 12.5, collective ~10, router ~8, rest ~28). The bytes
are the draw's minimum: two experts a rank a layer, 25.5 MiB, 1198.5 MiB a token, and the link is at
12 GiB/s already.

They are also *unhideable as the step is scheduled*, and that is a dependency and not a mistake in the
slot code. Layer L's draw needs layer L's attention, so the copy cannot be issued before the attention
it would have to hide behind; and the host's `indices.tolist()` drains the stream at that point, so
the device has no other work in flight either. The one thing a rank does know early is what it staged
for the *previous token* -- same layer, same rank, the same two rows of the same arena -- so the
question this probe answers is whether that is a prediction worth acting on.

What it measures, on a real prompt and a real greedy continuation, four ranks: for every layer and
every step, how many of the rank's `ceil(top_k / world)` rows hold the same expert as the step before.
The accounting is per *position* rather than per set, because a position that holds the same expert
needs no copy at all -- its row is already right -- while a set that matches in a different order
would need the rows rewritten. The row a slot's expert lands in is `sorted`'s and is stable across
steps (row `i` is the rank's sorted slot `rank + i * world`); under `id` a rank's rows are its own
experts in draw order, so a draw that hands it three shifts every row after the first and the
position-wise question is not the one to ask there.

**The answer is no.** On 8192 tokens of a real document and 32 greedy steps, a row holds the expert
it held a step earlier 9 to 13.5 times in a hundred (0.18 to 0.27 of a rank's two, 76 to 84% of the
pairs keeping neither row) and a rank's whole set repeats in 1.9 to 2.4% of the draws -- three to four
times the ~3% a re-draw would score on its own, and far too little to issue a copy from. The design a
better number would have justified: a prefetch of layer L+1's rows on the copy stream when layer L's
draw is read, landing behind layer L's kernel and layer L+1's attention, and at layer L+1's own draw a
copy only of the positions that moved. A hit would then cost no copy at all and the floor would be
the copy's *unpredicted* fraction rather than all of it. This router's draws are too scattered for
that fraction to be small -- which is also why a wider batch does not help: a chunk's tokens draw
nearly disjoint experts, so staging them together is one copy either way.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_expert_reuse.py --depth 8192 --tokens 32
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_expert_reuse.py --depth 8192 --tokens 32 \
        --prompt-file docs/models/mimo-v2.6-flash.md
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


def tokenize(path: str, checkpoint: str, want: int) -> list[int]:
    """`want` tokens of a real document, or `PROMPT_IDS` repeated if there is no tokenizer.

    A drawn prompt would be the wrong instrument here: the draws of a random id stream are the
    model's answer to noise, and two consecutive tokens of noise have no reason to route alike.
    """
    text = None
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    if text is None:
        return (PROMPT_IDS * (want // len(PROMPT_IDS) + 1))[:want]
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return (PROMPT_IDS * (want // len(PROMPT_IDS) + 1))[:want]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ids = tokenizer(text)["input_ids"]
    if len(ids) < want:
        ids = (ids * (want // len(ids) + 1))[:want]
    return list(ids[:want])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT",
                                                               DEFAULT_CHECKPOINT))
    parser.add_argument("--depth", type=int, default=8192, help="prompt tokens to prefill")
    parser.add_argument("--tokens", type=int, default=32, help="tokens to decode after it")
    parser.add_argument("--chunk", type=int, default=2048, help="prefill chunk")
    parser.add_argument("--prompt-file", default="docs/models/mimo-v2.6-flash.md")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    started = time.perf_counter()
    ep = EpGroup.from_env()
    device = ep.device if ep.device is not None else torch.device("cuda", ep.rank)
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint, device=device, expert_source=bank, ep=ep, chunk_rows=args.chunk
    )
    torch.cuda.synchronize()
    cache = model.cache(args.depth + args.tokens + 8)
    experts = model.experts
    print(
        f"[r{ep.rank}] world {ep.world}, deal `{experts.deal}`, {experts.arena_rows} rows a slot, "
        f"built in {time.perf_counter() - started:.0f}s",
        flush=True,
    )

    prompt = tokenize(args.prompt_file, args.checkpoint, args.depth)
    logits = model.prefill(prompt, cache=cache, chunk=args.chunk)
    torch.cuda.synchronize()
    if ep.world > 1:
        torch.distributed.barrier()

    # The draws, per layer, as the host reads them -- which is the only place they exist in a form
    # a prefetch could be issued from.
    seen: list[dict[int, list[int]]] = []
    current: dict[int, list[int]] = {}

    def capturing(inner):
        def forward(hidden, indices, weights, **kwargs):
            layer = kwargs.get("layer_id")
            drawn = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            # The ids, in arena-row order: row `i` of the slot holds `drawn[mine[i]]`, and that is
            # what a prefetch would have to have put there.
            mine = owned_positions(drawn, rank=experts.rank, world=experts.world, deal=experts.deal)
            current[layer] = [drawn[position] for position in mine]
            return inner(hidden, indices, weights, **kwargs)

        return forward

    original = experts.forward
    experts.forward = capturing(original)

    tokens = [int(logits.argmax())]
    position = args.depth
    for step in range(args.tokens):
        current = {}
        logits = model.step(tokens[-1], start_pos=position, cache=cache)[-1]
        seen.append(current)
        tokens.append(int(logits.argmax()))
        position += 1
    experts.forward = original
    torch.cuda.synchronize()
    if ep.world > 1:
        torch.distributed.barrier()

    # Per layer and step: how many of this rank's rows hold the expert the same row held a step
    # earlier. The first step has nothing to compare against and is not counted.
    rows = experts.arena_rows
    matched: list[int] = []
    per_layer: dict[int, list[int]] = {}
    sets_same = 0
    for step in range(1, len(seen)):
        for layer, mine in seen[step].items():
            before = seen[step - 1].get(layer)
            if before is None:
                continue
            hits = sum(1 for a, b in zip(mine, before) if a == b)
            matched.append(hits)
            per_layer.setdefault(layer, []).append(hits)
            sets_same += int(sorted(mine) == sorted(before))

    total = len(matched)
    if total:
        counts = [matched.count(k) for k in range(rows + 1)]
        share = [count / total for count in counts]
        print(
            f"[r{ep.rank}] rows unchanged per (layer, step): "
            + ", ".join(f"{k}: {counts[k]} ({share[k] * 100:.1f}%)" for k in range(rows + 1))
            + f"; mean {sum(matched) / total:.2f} of {rows}",
            flush=True,
        )
        worst = sorted(per_layer.items(), key=lambda item: sum(item[1]) / len(item[1]))[:3]
        best = sorted(per_layer.items(), key=lambda item: -sum(item[1]) / len(item[1]))[:3]
        print(
            f"[r{ep.rank}] coldest layers "
            + ", ".join(f"{layer}: {sum(v) / len(v):.2f}" for layer, v in worst)
            + "; hottest "
            + ", ".join(f"{layer}: {sum(v) / len(v):.2f}" for layer, v in best),
            flush=True,
        )
        # The same question asked of the whole draw rather than of this rank's rows: a set that
        # repeats still has to be *placed* here, so this is the looser number.
        print(
            f"[r{ep.rank}] draws whose rank-own *set* repeats: "
            f"{sets_same / total * 100:.1f}% ({sets_same} of {total})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
