#!/usr/bin/env python
"""What a decode step actually costs, with nothing in the loop that measures it.

Every other probe in this directory wraps something: a region in a `perf_counter` pair, a phase in a
CUDA event pair, an op in the profiler's tables. On a step whose host is the larger half, the
instrument is inside the number it reports -- `probe_mimo_v2_host_phases.py` puts a clock around
every one of a token's regions and reads 118.3 ms at 4096 where the same configuration, the same
sixteen resident rows and the same depth read **107.3, 97.3 and 96.5** with nothing wrapped. So this
probe is the number the others are checked against, and it is deliberately the simplest thing here:
`--steps` calls to `model.step`, one `perf_counter` pair around the whole loop, one `synchronize`
after it.

Two of the numbers it prints are not on the clock and are the reason it is worth having on its own:

* **How many experts a step staged and how many of its draws the resident set answered.** The copy
  is the floor under a step that no host work hides -- 47 layers of the `sorted` deal's two experts
  is 1198.5 MiB a rank -- so a run whose hit rate is not the one a table was taken at is a different
  step, and residency is worth more than every kernel on this path put together.
* **The four ranks' own steps.** A step is a chain of barriers, so the slowest rank is the step.
  The spread is printed because a rank that is behind is a configuration problem -- the deal, the
  resident set, the host -- and a rate quoted from rank 0 alone would hide it.

A step at a depth needs a prefix, and **the prefix has to be a real one**. The obvious way to reach a
depth is `bench_mimo_v2_model.fill_cache`, which writes the positions without running the model and
turns a 262144-position step into seconds instead of the 1 h 47 m this page's prefill rate would
charge. What it does not give is a step a serving loop would take: the states are random, so the
router's draw is nearly the same experts over and over, the resident set answers almost all of them,
and the step comes out *cheaper* than the same step over a prompt -- 96.5 ms at eight positions
against 117.8 over a real eight-token prompt, at the same sixteen rows. A rate read off a filled
cache is therefore an upper bound and not the token.

So `--depth` prefills `--tokens` of a real document through `model.prefill` in `--chunk`-wide
chunks, which is a real context and costs what a real context costs: 4096 tokens is about half a
minute. `--fill` is the escape hatch for the depths where that is not affordable -- 262144 is 42
minutes of prefill against 5.65 GiB of cache -- and a run that uses it says so in its own line.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_token.py --steps 200
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_token.py --depth 4096 --resident-rows 16
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_token.py --depth 262144 --fill \
        --resident-rows 8 --steps 120

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
from tests.bench_mimo_v2_model import fill_cache, tokenize  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="prefill this many tokens of a real document before the measured steps; 4096 is about "
        "half a minute and 262144 is 42, which is what `--fill` is for",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="write the positions into the cache instead of prefilling them: seconds instead of "
        "minutes, at the cost of a step whose draws are the router's answer to noise. The rate it "
        "prints is an upper bound -- the same step over a real prompt is dearer, because the set "
        "answers fewer of its draws",
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--chunk", type=int, default=2048, help="width a prefill chunk goes at")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=0,
        help="the band a chunk's arena holds, which is what builds the second expert module a "
        "prefill needs. The step module keeps its own arena -- a decode module is the only kind "
        "that can hold a resident set -- so this costs card memory and does not change the step",
    )
    parser.add_argument(
        "--prompt-file", default="docs/models/mimo-v2.6-flash.md", help="the document to tokenize"
    )
    parser.add_argument("--steps", type=int, default=120, help="steps the clock covers")
    parser.add_argument("--warmup", type=int, default=6, help="untimed steps before it")
    parser.add_argument(
        "--resident-rows",
        type=int,
        default=0,
        help="how many of each routed layer's hottest experts to keep on the card. Default 0 is "
        "what `pocketllm serve --backend mimo` ships, because 16 rows is 9.36 GiB and a "
        "262144-token cache is 5.65 of a 22 GiB card; a short-context deployment is what should "
        "pay for it, and at 8 rows a 262144-token step still fits",
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
        chunk_rows=args.chunk_rows,
    )
    layers = [layer.layer_idx for layer in model.layers]

    # One cache, sized for the whole run. `share_rope_tables` after it re-shares the tables at the
    # run's capacity -- the tables are one a *family* and every `cache()` call re-shares them at
    # *its* capacity, so a smaller cache built later would leave the stack unable to reach a deep
    # position on the one-row path, and every step would silently take the chunk path.
    span = max(args.depth, args.prompt) + args.warmup + args.steps + 16
    cache = model.cache(span)
    model.share_rope_tables(span)
    filled = False
    if args.depth:
        if args.fill:
            fill_cache(cache, layers, args.depth)
            filled = True
        else:
            ids = tokenize(args.checkpoint, args.depth, args.prompt_file)
            logits = model.prefill(ids, cache=cache, chunk=args.chunk)
        position = args.depth
    else:
        prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
        position = 0
        logits = None
        for token in prompt_ids:
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
            position += 1
    torch.cuda.synchronize()

    module = model.experts
    residents = module._residents

    def counters() -> tuple[int, int, int]:
        """Copies over PCIe, and the draws the resident set answered."""
        if residents is None:
            return module.staged_experts, 0, 0
        return module.staged_experts, residents.hits, residents.drawn

    # A greedy chain, not a fixed token: the draws the router makes have to be the ones a real
    # decode makes them, or the residency numbers below are about a prompt nobody would send.
    token = PROMPT_IDS[0]
    for offset in range(args.warmup):
        logits = model.step(token, start_pos=position + offset, cache=cache)[-1]
        token = int(logits.argmax())
    torch.cuda.synchronize()

    before = counters()
    began = time.perf_counter()
    for step in range(args.steps):
        logits = model.step(token, start_pos=position + args.warmup + step, cache=cache)[-1]
        token = int(logits.argmax())
    torch.cuda.synchronize()
    took = time.perf_counter() - began
    after = counters()

    staged, hits, drawn = (b - a for a, b in zip(before, after))
    token_id = int(logits.argmax())
    # The slowest rank is the step: every routed layer closes with a barrier, so a rank that is
    # behind is a rank everyone waits for, and a rate read off rank 0 alone would hide it.
    worst = torch.tensor([took], device=device)
    if ep.world > 1:
        torch.distributed.all_reduce(worst, op=torch.distributed.ReduceOp.MAX)
    per_rank = [None] * ep.world
    per_rank[rank] = took
    gathered: list = []
    if ep.world > 1:
        holder = [None] * ep.world
        torch.distributed.all_gather_object(holder, took)
        gathered = holder
    spread = ""
    if gathered and all(entry is not None for entry in gathered):
        spread = (
            f", the ranks {', '.join(f'{entry * 1e3 / args.steps:.1f}' for entry in gathered)} ms"
        )
    if rank == 0:
        print(
            f"[r{rank}] {args.steps} steps at {args.depth or args.prompt} positions with "
            f"{args.resident_rows} resident rows: {worst.item() * 1e3 / args.steps:7.1f} ms a token "
            f"({args.steps / worst.item():5.2f} tok/s), staged {staged / args.steps:5.2f} experts a "
            f"step, resident {hits}/{drawn} = {100 * hits / max(drawn, 1):.1f}%{spread}; "
            f"the last token is {token_id}",
            flush=True,
        )
        if filled:
            print(
                f"[r{rank}] the prefix was *written*, not prefilled: its draws are the router's "
                f"answer to noise and the set answers more of them than it would over a prompt, so "
                f"this rate is an upper bound on the same step over real text",
                flush=True,
            )
    if ep.world > 1:
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
