#!/usr/bin/env python
"""What a MiMo-V2.6 token costs when the experts are dealt out over the cards.

The same stack and the same prompt as `bench_mimo_v2_model.py`, run under `torchrun` with a rank
a card, so the two runs are the two columns of one table: one rank holds the whole draw and moves
4.68 GiB a token, four ranks hold a quarter of it each and move a quarter of the bytes. The router,
the attention and the dense layer are replicated on every rank, so what this measures is the expert
traffic divided and nothing else -- the attention is still paid four times over, and the stage that
fixes that is tensor parallelism.

Three things are checked rather than assumed, and the third is why this is a verification and not
only a bench:

* Where the bytes went. `staged MiB/token` is the rank's own counter, and under a `sorted` deal a
  top-8 draw over four ranks is two experts a rank and not eight -- 47 x 2 x 12.75 MiB, which is
  the quarter the deal is supposed to buy.
* Which deal. `--deal` sets `POCKETLLM_MIMO_EXPERT_DEAL`, and the two rows of the summary are the
  two deals on the same weights: `id` partitions the experts and `sorted` partitions the draw.
* That the answer did not change. Every rank computes the *same* logits -- the collective returns
  the same sum everywhere -- so the ranks are gathered after the first decode step and held against
  each other bit for bit, and their greedy continuations have to be the same ids. A rank that read
  another rank's experts, staged a row it did not own or summed a partial twice disagrees here, and
  a decoder that agrees across four ranks while disagreeing with the one-rank run is what the
  token list at the end is for.

Usage:

    torchrun --nproc_per_node=4 tests/bench_mimo_v2_ep.py --steps 8
    torchrun --nproc_per_node=4 tests/bench_mimo_v2_ep.py --deal sorted
    python tests/bench_mimo_v2_ep.py                      # world 1: the control column

`--depth` is the same decode step at a context too long to prefill: the cache is written directly
at `depth` positions and the prompt is skipped, so a 256k step costs minutes rather than hours. The
step it measures is the same step -- the bytes read, the experts staged, the collective -- and the
logits it produces are not an answer. See `bench_mimo_v2_model.fill_cache`.

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
from src.models.mimo_v2.ep import DEAL_ENV, EpGroup  # noqa: E402
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from tests.bench_mimo_v2_model import PhaseTimer, fill_cache, h2d_rate  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"

#: The same eight ids `bench_mimo_v2_model.py` feeds, so the two scripts' token lists compare.
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS), help="prompt tokens to feed")
    parser.add_argument("--steps", type=int, default=8, help="decode steps to measure")
    parser.add_argument("--slots", type=int, default=2, help="expert arena slots")
    parser.add_argument("--deal", default=None, help="`id` or `sorted`; the environment's by default")
    parser.add_argument("--no-pin", action="store_true", help="leave the bank pageable")
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="positions to fill the cache with instead of prefilling `--prompt`",
    )
    parser.add_argument("--warmup", type=int, default=2, help="decode steps before the measured ones")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0
    if args.deal is not None:
        os.environ[DEAL_ENV] = args.deal

    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    tag = f"[r{rank}]"

    started = time.perf_counter()
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        slots=args.slots,
        pin=not args.no_pin,
    )
    torch.cuda.synchronize()
    built = time.perf_counter() - started
    arena = model.experts.arena_bytes / 2**20
    pin = (
        f"rc={model.pin_result.rc} ok={model.pin_result.ok} in {model.pin_result.seconds:.1f}s"
        if model.pin_result is not None
        else "not asked for"
    )
    print(
        f"{tag} world {world} deal `{model.experts.deal}`: {len(model.layers)} layers in "
        f"{built:.1f}s, {model.memory_bytes / 2**20:.0f} MiB on the card, {arena:.0f} MiB arena",
        flush=True,
    )
    print(f"{tag} bank pin: {pin}", flush=True)

    # The link, measured while every rank is measuring it. One rank's 10.4 GiB/s is a
    # single-rank number; four ranks pulling at once is a question about the host's memory
    # bandwidth and the root complex, and it is what the copy column below has to be read
    # against.
    rate, expert_mib = h2d_rate(bank, model.config.moe_layer_indices[0])
    rates = [None] * world
    if world > 1:
        torch.distributed.all_gather_object(rates, round(rate, 2))
    print(
        f"{tag} expert pages to the card: {rate:.2f} GiB/s"
        + (f", all ranks {rates}" if world > 1 else ""),
        flush=True,
    )

    counters = model.experts
    depth = int(args.depth or 0)
    cache = model.cache(max(depth, args.prompt) + args.warmup + args.steps + 8)

    # A discarded pass: the first token pays the CUDA context, the first staging and whatever the
    # driver does once. Its routing is also what warms the collective, so the measured region is
    # not paying a communicator's first message either.
    model.greedy(PROMPT_IDS[: args.prompt], max_tokens=1, cache=cache)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    if depth:
        # A depth reached without a prompt: the cache is filled directly, because prefilling to
        # 256k is 1 h 47 m at the release's own rate. The step measured below is then the same
        # step -- the same bytes read, the same two experts a rank staged, the same collective --
        # over logits that mean nothing and a draw that is noise. `fill_cache` says what that
        # does and does not preserve; what it does not is any claim about an answer.
        # `reset` and not a fresh cache: the discard pass above already appended to this one, and
        # the fill appends rather than writes in place, so the depth it asks for is a depth *past*
        # whatever the discarded prompt left behind.
        cache.reset()
        fill_cache(cache, [layer.layer_idx for layer in model.layers], depth)
        position = depth
        logits = None
        for _ in range(args.warmup):
            logits = model.step(PROMPT_IDS[0], start_pos=position, cache=cache)[-1]
            position += 1
        torch.cuda.synchronize()
        if world > 1:
            torch.distributed.barrier()
        print(
            f"{tag} cache at {depth} positions after {args.warmup} warm steps, "
            f"{cache.memory_bytes / 2**30:.2f} GiB of cache",
            flush=True,
        )
        prompt_ids = []
        prefill_s = prefill_bytes = 0.0
    else:
        prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
        cache.reset()
        mark = (counters.staged_experts, counters.staged_bytes)

        if world > 1:
            torch.distributed.barrier()
        started = time.perf_counter()
        for position, token in enumerate(prompt_ids):
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - started
        prefill_bytes = counters.staged_bytes - mark[1]
        position = len(prompt_ids)

    drawn = [int(logits.argmax())]
    first = logits.clone()
    timer = PhaseTimer(model)
    mark = (counters.staged_experts, counters.staged_bytes)
    if world > 1:
        torch.distributed.barrier()
    started = time.perf_counter()
    for step in range(args.steps):
        logits = model.step(drawn[-1], start_pos=position + step, cache=cache)[-1]
        drawn.append(int(logits.argmax()))
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - started
    phases = timer.read()
    decode_experts = counters.staged_experts - mark[0]
    decode_bytes = counters.staged_bytes - mark[1]

    # `args.steps` forwards and not `len(drawn)`: the first drawn token is the prefill's own last
    # row, so dividing the loop's seconds by the tokens it produced would report a token that the
    # timed region never computed. What the loop did is `steps` tokens, and that is the rate.
    steps = args.steps
    if not depth:
        print(
            f"{tag} prefill {len(prompt_ids)} tokens in {prefill_s:.3f}s = "
            f"{len(prompt_ids) / prefill_s:.2f} tok/s",
            flush=True,
        )
    print(
        f"{tag} decode  {steps} steps in {decode_s:.3f}s = {steps / decode_s:.2f} tok/s "
        f"({decode_s / steps * 1e3:.1f} ms/token), {decode_experts / steps:.1f} experts a step"
        + (f" at {depth} positions" if depth else ""),
        flush=True,
    )
    if not depth:
        print(
            f"{tag} staged  {decode_bytes / steps / 2**20:.1f} MiB a step, "
            f"{prefill_bytes / max(1, len(prompt_ids)) / 2**20:.1f} MiB a prompt token",
            flush=True,
        )
    print(
        f"{tag} phases  attention {phases['attn'] / steps:.1f} ms a step, "
        f"FFN and staging {phases['mlp'] / steps:.1f} ms a step",
        flush=True,
    )

    # The collective's own check, and the reason a four-rank run is a verification: every rank
    # holds the same sum, so every rank's first step is the same row bit for bit. A rank that
    # staged a row it did not own, dropped one it did, or summed a partial twice lands here.
    if world > 1:
        import torch.distributed as dist

        gathered = [torch.empty_like(first) for _ in range(world)]
        dist.all_gather(gathered, first.contiguous())
        disagree = [
            (other, float((gathered[other] - gathered[0]).abs().max()))
            for other in range(1, world)
            if not torch.equal(gathered[other], gathered[0])
        ]
        print(
            f"{tag} ranks agree on the first step's logits: {not disagree}"
            + (f" -- {disagree}" if disagree else ""),
            flush=True,
        )
        drawn_at_rank = [None] * world
        staged_at_rank = [None] * world
        dist.all_gather_object(drawn_at_rank, drawn)
        dist.all_gather_object(staged_at_rank, decode_bytes / steps / 2**20)
        same = all(row == drawn_at_rank[0] for row in drawn_at_rank)
        print(f"{tag} ranks agree on {len(drawn)} generated tokens: {same}", flush=True)
        # Gathered on every rank and printed on one: a rank that left while rank 0 was still
        # gathering would hang the run rather than fail it.
        if rank == 0:
            print(
                f"[r0] MiB a step by rank: {[round(value, 1) for value in staged_at_rank]}",
                flush=True,
            )
            print(f"[r0] tokens by rank: {drawn_at_rank}", flush=True)

    if rank == 0:
        print(f"[r0] drew {len(drawn)} tokens: {drawn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
