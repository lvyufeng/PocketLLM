#!/usr/bin/env python
"""The split moves the clock and not the tokens: the greedy stream, with the split and without.

`probe_mimo_v2_attention_split.py` pins the arithmetic layer by layer -- a share's piece is the
whole's piece bit for bit, and a windowed layer's decode step can differ in the last bit of a
float32 because a batched GEMM's tiling follows the batch size it is handed. This is the other half
of the question and the one a serving path is judged on: does a *token* move. It builds the
released model at the world it was launched with, prefills a real token stream through the chunked
path to `--depth`, decodes `--tokens` greedily, and prints what it drew.

The comparison is between two processes -- one with `POCKETLLM_MIMO_ATTENTION_SHARDS=1`, one without
-- because a second engine in one process is the same 18 GiB of a 22 GiB card twice over:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_split_tokens.py --depth 8192 --tokens 16 \
        --out /tmp/tokens_split.txt
    POCKETLLM_MIMO_ATTENTION_SHARDS=1 torchrun --nproc_per_node=4 \
        tests/probe_mimo_v2_split_tokens.py --depth 8192 --tokens 16 --out /tmp/tokens_whole.txt
    diff /tmp/tokens_split.txt /tmp/tokens_whole.txt

The cache is filled by a *real* prefill and never by `fill_cache`: a random cache is drawn from a
seeded generator whose stream advances with the shape it draws, so two arms with different key head
counts would be handed two different contexts and the comparison would mean nothing.

`--random-prompt` is the harsher arm and the one to run when the answer matters. The released
tokenizer's text gives the model a confident distribution where the argmax has a margin of whole
logits; a prompt drawn from a fixed seed and fed as ids gives a nearly flat one where a token sits
within a rounding step of its neighbour, which is exactly where a last-bit difference could show. A
stream that survives the second survives the first.

Every rank prints its own line, and that is a check of its own: the routed experts' split is joined
by an fp32 all-reduce, so four ranks that agree on a token agree on everything below it.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT",
                                                               DEFAULT_CHECKPOINT))
    parser.add_argument("--depth", type=int, default=8192, help="prompt tokens to prefill")
    parser.add_argument("--tokens", type=int, default=16, help="tokens to decode after it")
    parser.add_argument("--chunk", type=int, default=512, help="prefill chunk, and the expert band")
    parser.add_argument("--random-prompt", action="store_true",
                        help="draw the prompt from a seed instead of repeating `PROMPT_IDS`")
    parser.add_argument("--seed", type=int, default=0, help="the seed `--random-prompt` draws from")
    parser.add_argument("--out", default=None, help="write the token line here as well")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to compare")
        return 0

    started = time.perf_counter()
    ep = EpGroup.from_env()
    device = ep.device if ep.device is not None else torch.device("cuda", ep.rank)
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        device=device,
        expert_source=bank,
        ep=ep,
        chunk_rows=args.chunk,
    )
    cache = model.cache(args.depth + args.tokens + 8)
    print(f"[r{ep.rank}] world {ep.world}, attention in {ep.attention_shards} share(s), "
          f"{len(model.layers)} layers, cache {cache.memory_bytes / 2**30:.2f} GiB, "
          f"built in {time.perf_counter() - started:.0f}s", flush=True)

    prompt = (PROMPT_IDS * (args.depth // len(PROMPT_IDS) + 1))[: args.depth]
    if args.random_prompt:
        # A CPU generator, so the two arms get the same ids on every rank whatever the card.
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        prompt = torch.randint(
            0, model.vocab_size, (args.depth,), generator=generator
        ).tolist()
    started = time.perf_counter()
    logits = model.prefill(prompt, cache=cache, chunk=args.chunk)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - started
    # What the prefill left behind, before any token is fed back: the two arms' answers are only
    # comparable up to this point, and a divergence after it is the decode's and not the prompt's.
    prompt_best = torch.topk(logits.float().reshape(-1), 2).values
    print(f"[r{ep.rank}] prefill logits |sum| {float(logits.float().abs().sum()):.6e} "
          f"peak {float(prompt_best[0]):.6e} margin {float(prompt_best[0] - prompt_best[1]):.6e} "
          f"argmax {int(logits.argmax())}", flush=True)

    drawn = []
    started = time.perf_counter()
    for step in range(args.tokens):
        drawn.append(int(logits.argmax()))
        if step + 1 < args.tokens:
            logits = model.step(
                drawn[-1], start_pos=args.depth + step, cache=cache
            )[-1]
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - started

    if ep.world > 1:
        torch.distributed.barrier()
    line = " ".join(str(token) for token in drawn)
    # The margin the last draw had, and the reason it is printed: a token stream is the coarsest
    # observable there is -- it is the argmax of 152,576 logits -- so "the tokens agree" says
    # nothing about how close the second one was. A margin of whole logits is a stream that
    # survives any last-bit difference; a margin of nothing is a coin toss that says nothing.
    best = torch.topk(logits.float().reshape(-1), 2).values
    print(f"[r{ep.rank}] prefill {args.depth} tokens in {prefill_s:.1f}s "
          f"({args.depth / prefill_s:.1f} tok/s), decode {args.tokens} in {decode_s:.2f}s "
          f"({args.tokens / decode_s:.2f} tok/s)", flush=True)
    print(f"[r{ep.rank}] logits |sum| {float(logits.float().abs().sum()):.6e} "
          f"peak {float(best[0]):.6e} margin {float(best[0] - best[1]):.6e} "
          f"argmax {drawn[-1]}", flush=True)
    print(f"TOKENS {line}", flush=True)
    if args.out and ep.rank == 0:
        with open(args.out, "w") as handle:
            handle.write(f"depth {args.depth} tokens {args.tokens} "
                         f"shards {ep.attention_shards} "
                         f"prompt {'random' if args.random_prompt else 'real'} seed {args.seed}\n"
                         f"{line}\n")
        print(f"wrote {args.out}", flush=True)
    if ep.world > 1:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
