"""Run DeepSeek-V4.1-Flash on the four cards: the dense tree cut across them, one process per card.

This is the launcher the rest of the round exists for. `python -m src.models.deepseek_v4_1.generate`
drives the same model on the host with no process group; this one reads the rank torchrun left in the
environment, puts the tree's quarter and the experts' share on the card that rank owns, and starts the
same decode loop. Nothing in the loop changes -- `load_backbone(world=..., rank=...)` is the whole
difference, and `world=1` reproduces the host run exactly.

    torchrun --nproc_per_node=4 -m src.cli.generate_v41 \\
        --checkpoint /path/to/DeepSeek-V4.1-Flash --prompt "The capital of France is" \\
        --max-new-tokens 8

**Why one process per card and not one process driving four.** A single process issuing four ranks'
forwards pays the Python dispatch serially -- 1027 kernel launches a layer at ~16 us of Python each,
which `/tmp/probe_tp4_block.py` measures as the whole cost of the step, since device-busy is 11%. Four
processes pay it in parallel. It is also the shape the repository already has for TP:
`src/cli/generate_glm.py`, `src/models/qwen4_exp/runtime.py`'s `make_all_reduce`.

**The card is `LOCAL_RANK`, and it is the same card for both halves.** The dense tree is built on
`cuda:{local_rank}` (Phase 2.1's `device`), and the experts land on `cuda:{base + rank}` because the
loader's one-process-per-*node* shape spells `--expert-device cuda:1 --expert-world 4` as cards 1
through 4 (`loader.py`'s `on_device`). The base this launcher hands over is therefore `local_rank -
rank` -- zero on a single node, which is the only shape this launcher supports, and it says so rather
than driving the wrong card.

**The experts default to the cards here.** Under torchrun with a world above one, `--expert-device`
unset means the rank's own card, because that is the configuration the plan's numbers are about and a
four-process run whose experts quietly stayed on the host would look exactly like one that worked.
`--expert-device host` is the way back to the host path, and it is the control column.

**The ranks stay in lockstep without a broadcast.** Every rank computes the same logits: the tree is
cut so that each collective's result is identical on all four (`tp.py`), the head and the embedding
are replicated rather than cut, and a ring all-reduce sums in the same order everywhere. Greedy
decoding is an argmax over that, so the four agree on the token by construction rather than by being
told, and one collective a step is not paid to enforce what is already true.

**What the four cards come to.** At 8 tokens of context, greedy decode is **722-747 ms per step** --
409-444 of it in `DeviceRoutedExperts` and 303-313 in the tree -- against the **1060-1140 ms per step**
`docs/performance/deepseek_v4_1_flash_device_experts.md` records with the tree on the host, so cutting
it across the cards halved the half that was the dense tree's. Prefill is the standing problem and
always was: the same probe measures **3.0 tok/s at 128 tokens**, 41.85 of its 42.86 s inside
`DeviceRoutedExperts`, because the class stages 4.2 GiB per row and a prefill is 128 rows of it.

That 722-747 ms is a warm-page-cache figure and only as durable as the cache: `DeviceRoutedExperts`
stages out of the checkpoint mapping, and this host holds 68-100 GiB of the checkpoint's 475.25 GiB
because 457.78 GiB of its RAM is already the resident bank's tmpfs segment. With the pages dropped
(`/tmp/fadvise_drop.py`) the same 8-token row measures **17.01 s a step**, 9.91 s of it in `_stage`
against 0.30 s resident, while `_upload` and `_launch` do not move at all. Staging from a resident
bank rather than from the mapping is the follow-on that makes the headline hold; until then, quote it
with the cache it was measured on.

**Pass `--threads`, because torchrun does not.** `torch.distributed.run` sets `OMP_NUM_THREADS` to 1
for every worker unless the environment already had it, and a share of a step is host work -- the
expert staging, the gate, the head, the layer glue. The same three tokens on this host, with the
resident bank in place so the two columns differ only in the flag, measure **5.7-5.8 s at
`--threads 22` against 6.5-6.6 s at one**; every other number in this round used 22, and the flag is
how a run says so out loud instead of inheriting it.

Be careful what this flag is credited with. With no resident bank -- the class staging out of the
checkpoint mapping, so the read is `/mnt/data3` -- those same three tokens are 6.5 s at 22 threads and
**44.1 s at one**, 14.70 s/token: one thread serializes a per-row read out of an SMR disk. That 6.8x
is the flag's when there is no resident source, and it is the resident bank's under the same
conditions, so the two cannot be credited to each other. An earlier pair of runs here recorded 5.1 s
against 6.6 s as if it were a thread effect; they differed in the bank as well.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from typing import Sequence

import torch

from src.models.deepseek_v4_1.generate import generate

__all__ = ["main", "setup_distributed"]


def setup_distributed() -> tuple[int, int, int, torch.device | None]:
    """`(world, rank, local_rank, device)` from the environment torchrun left behind.

    `world=1` outside torchrun is the host configuration: no process group, no card, and `device` is
    `None`, which is the same value `load_backbone` takes when nobody asked for a card.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world <= 1:
        return 1, 0, 0, None
    if not torch.cuda.is_available():
        raise RuntimeError(f"a TP world of {world} needs the cards, and this host has none")
    count = torch.cuda.device_count()
    if local_rank >= count:
        raise RuntimeError(f"rank {rank} was told to drive card {local_rank} of {count} on this host")

    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Two hours, not the ten-minute default: rank 0 may be paying the resident bank's one-time
        # fill -- 36 minutes of an SMR disk -- while the rest wait at the first collective, and a
        # store timeout would tear the communicator down underneath a run that is behaving.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return world, rank, local_rank, torch.device("cuda", local_rank)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchrun --nproc_per_node=4 -m src.cli.generate_v41",
        description="Generate text from DeepSeek-V4.1-Flash with the dense tree cut across the cards.",
    )
    parser.add_argument("--checkpoint", required=True, help="the released checkpoint directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 is greedy, which is the default and the reproducible one")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="context the run allocates for. The caches are preallocated at this "
                             "length, so it is a memory budget and not a limit that grows: the "
                             "default is the prompt plus the tokens asked for, rounded up, and "
                             "`--max-seq-len` is how a caller asks for the model's full 1M")
    parser.add_argument("--device", default=None, metavar="DEVICE",
                        help="build the dense tree on DEVICE in this process, with no process group "
                             "and an undivided tree. One card cannot hold the whole tree -- it runs "
                             "out of memory in `Head` -- so this is a debugging flag for the "
                             "single-process shape, not a run. Under torchrun the card comes from "
                             "LOCAL_RANK and this is ignored")
    parser.add_argument("--resident-engram", action="store_true",
                        help="copy the Engram tables into this process's RAM instead of reading them "
                             "out of the resident bank; 189.1 GiB per rank, so it is the thing four "
                             "ranks must not each pay -- see `resident_bank`")
    parser.add_argument("--expert-cache", type=int, default=None,
                        help="dequantized experts one layer keeps on the host, on the host path")
    parser.add_argument("--expert-device", default=None, metavar="DEVICE",
                        help="put the routed experts on DEVICE instead of the host. Default: the "
                             "rank's own card when running under torchrun, the host otherwise. "
                             "`host` forces the host path, which is the control column")
    parser.add_argument("--expert-world", type=int, default=None,
                        help="cards to deal the routed experts over. Default: the tree's world, "
                             "which is what a sharded tree requires")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="host threads this rank may use. `torchrun` sets OMP_NUM_THREADS to 1 "
                             "unless the environment already had one, and the expert staging, the "
                             "gate and the head are host work -- 22 is what this round's numbers "
                             "used. Unset leaves whatever the environment says")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the progress and timing lines on every rank")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    world, rank, local_rank, device = setup_distributed()
    sharded = world > 1
    if device is None and args.device is not None:
        device = torch.device(args.device)
    card = None if device is None else (device.index if device.index is not None else 0)

    def say(message: str) -> None:
        if not args.quiet:
            print(f"[rank {rank}] {message}" if sharded else message, file=sys.stderr, flush=True)

    import torch.distributed as dist

    from transformers import AutoTokenizer

    from src.models.deepseek_v4_1.config import load_config
    from src.models.deepseek_v4_1.loader import DEFAULT_EXPERT_CACHE, V41Checkpoint, build_hasher
    from src.models.deepseek_v4_1.loader import load_backbone
    from src.models.deepseek_v4_1 import resident_bank

    expert_device = args.expert_device
    if expert_device is not None and expert_device.strip().lower() in ("host", "cpu", "none"):
        expert_device = None
    if expert_device is None and card is not None:
        # `loader.on_device` counts a rank's card as `base + rank`, so the base that lands rank `r`
        # on the card the tree is on is `card - rank`. Zero under torchrun on one node, which is the
        # shape this launcher supports: a multi-node run would need a base below zero, which is not
        # a card, and it is refused here rather than silently misplaced.
        base = card - rank
        if base < 0:
            raise SystemExit(
                f"rank {rank} drives card {card}, so the loader's rank-offset card arithmetic "
                "cannot name its own card: this launcher runs one node"
            )
        expert_device = f"cuda:{base}"
    expert_world = args.expert_world
    if expert_world is None:
        expert_world = world if expert_device is not None else 1

    config = load_config(f"{args.checkpoint}/config.json").text
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    _, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    # Every rank tokenizes the same prompt: it is a deterministic CPU step and cheaper than a
    # broadcast, which is why the GLM entry point does the same thing. It is done before the load
    # because the sequence length the caches are sized at comes out of it.
    prompt_ids = tokenizer(args.prompt)["input_ids"]
    if not prompt_ids:
        raise SystemExit("--prompt tokenized to nothing")
    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        # The caches are `register_buffer`-ed at this length rather than grown, and the default the
        # loader would otherwise take is `max_position_embeddings` -- 1M, four cards' worth of
        # 268 MB `freqs_cis` tables and 536 MB compress caches per source layer. A run that only
        # asked for 40 tokens should not pay for a context it will never reach. Rounded up so that
        # a token's worth of slack stays slack rather than becoming an off-by-one at the last step.
        max_seq_len = -(-(len(prompt_ids) + args.max_new_tokens) // 64) * 64

    say(
        f"network {world} rank{'s' if world > 1 else ''}, tree on "
        f"{'the host' if device is None else device}, experts on "
        f"{'the host' if expert_device is None else f'{expert_device} x{expert_world}'}, "
        f"context {max_seq_len}, resident checkpoint "
        f"{'on' if resident_bank.enabled() else 'off'}"
    )
    started = time.perf_counter()
    front = load_backbone(
        config,
        checkpoint,
        device=device,
        max_seq_len=max_seq_len,
        hasher=hasher,
        resident_engram=args.resident_engram,
        expert_cache=DEFAULT_EXPERT_CACHE if args.expert_cache is None else args.expert_cache,
        expert_device=expert_device,
        expert_world=expert_world,
        # The bank is filled by rank 0 and attached by everyone else, so a rank here is both the
        # tree's rank and the stagger the bank wants: four ranks must not read `/mnt/data3` at once.
        expert_rank=rank,
        world=world,
        rank=rank,
        progress=say,
    )
    say(f"loaded in {time.perf_counter() - started:.1f} s")

    # `--expert-device host` is the other failure this flag has, so the line reports the objects that
    # were built rather than the flag that asked: `DeviceRoutedExperts` is deliberately not an
    # `nn.Module`, and `load_backbone` unpacks the dict into the layers.
    held = [layer.ffn.routed for layer in front.model.layers]
    kinds = sorted({type(r).__name__ for r in held})
    where = ""
    if expert_device is not None:
        world_held = getattr(held[0], "world", None)
        ranks_held = getattr(held[0], "ranks", None)
        where = f", world {world_held}"
        if ranks_held is not None and world_held is not None and len(ranks_held) < world_held:
            where += f", holding rank {ranks_held[0]} of the deal"
    say(f"routed experts: {', '.join(kinds)} on {len(held)} layers{where}")
    say(f"prompt {len(prompt_ids)} tokens")

    if sharded and dist.is_initialized():
        # Before the first collective rather than inside it: one rank can still be reading a quarter
        # of the tree off `/mnt/data3` while another is ready to prefill, and the wait belongs here
        # where it is a load-time cost rather than a stall in the middle of a forward.
        dist.barrier()

    started = time.perf_counter()
    result = generate(
        front,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started
    if result.tokens:
        per = elapsed / len(result.tokens)
        say(f"{len(result.tokens)} tokens in {elapsed:.1f} s ({per:.2f} s/token), "
            f"stopped on {result.stopped}")

    if rank == 0:
        print(tokenizer.decode(prompt_ids + result.tokens), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
