#!/usr/bin/env python
"""Run the released MiMo-V2.6 checkpoint through the host reference and decode.

Stage 1 pinned the reference against the oracle fixture, which is the only place
the *arithmetic* can be checked. This is the other half: the fixture says nothing
about whether the loader reads the released tensors, and a checkpoint that is read
at the wrong offset or with the wrong nibble order still produces finite logits.
The observable that catches that is text -- the real checkpoint on a real prompt
either continues it coherently or does not, and no amount of shape checking
substitutes for that.

So this runs the full 48-layer backbone on the release, greedily, on the host, and
prints the continuation. It is slow by construction: nothing is cached, every
routed expert is dequantized from the packed checkpoint on the step that selects
it, and the whole thing is float32 on CPU. That is the point of a reference -- the
device path is what gets optimized, and it is diffed against this.

    python scripts/verify_mimo_v2_real_checkpoint.py
    python scripts/verify_mimo_v2_real_checkpoint.py --layers 2 --tokens 4

Exit status is 0 when the run completes; non-finite logits raise rather than being
reported as a number. A truncated stack (`--layers`) prints what it produced but
says so, because a partial stack's text means nothing.

The checkpoint is the default asset path and not a small download, so a repository
without it skips with status 0 rather than failing, matching how the tests treat a
missing asset.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.layers import build_attention_masks  # noqa: E402
from src.models.mimo_v2.weights import host_model_from_checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
DEFAULT_PROMPT = (
    "The capital of France is Paris, and the capital of Japan is"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=8, help="tokens to generate")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="truncate the stack (debugging only; the text is not meaningful)",
    )
    parser.add_argument("--expert-cache", action="store_true", help="keep dequantized experts")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def describe_tensors(checkpoint: MimoV2Checkpoint) -> None:
    print(checkpoint.describe())
    layer0 = "model.layers.0.self_attn.qkv_proj.weight"
    print(f"layer 0 qkv  {tuple(checkpoint.entry(layer0).shape)} {checkpoint.entry(layer0).dtype}")


@torch.no_grad()
def greedy(
    model,
    input_ids: torch.Tensor,
    steps: int,
    mask: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[float]]:
    """Greedy decode, one full forward per token.

    No KV cache: re-running the prefix is the honest thing for a reference whose
    job is to agree with the fixture, and at these lengths it costs nothing.

    `mask` is built once at the final length and sliced down, because the masks are
    position-based and a prefix of a longer one is the shorter mask exactly.
    """
    generated = input_ids
    logits_seen: list[float] = []
    for step in range(steps):
        length = generated.shape[1]
        step_mask = {key: value[:, :, :length, :length] for key, value in mask.items()}
        positions = torch.arange(length).unsqueeze(0)
        started = time.time()
        logits = model(generated, attention_mask=step_mask, position_ids=positions)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite logits at length {length}")
        last = logits[:, -1]
        logits_seen.append(float(last.max()))
        next_id = int(last.argmax(dim=-1))
        # Progress is printed per token rather than at the end: this runs for tens
        # of minutes per token when the checkpoint's pages are cold, and a script
        # that is silent for an hour is indistinguishable from one that has hung.
        print(
            f"  step {step + 1}/{steps}  len {length}  "
            f"{time.time() - started:.1f}s  next {next_id}",
            flush=True,
        )
        generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)
    return generated, logits_seen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dtype = getattr(torch, args.dtype)
    if not os.path.isdir(args.checkpoint):
        print(f"SKIP: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 0

    t0 = time.time()
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    describe_tensors(checkpoint)
    print(f"headers      {time.time() - t0:.2f}s")

    config = checkpoint.layer
    tokenizer_path = os.path.join(checkpoint.root, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"SKIP: no tokenizer.json beside the shards at {checkpoint.root}", file=sys.stderr)
        return 0

    layers = None if args.layers is None else list(range(args.layers))
    t0 = time.time()
    model = host_model_from_checkpoint(
        checkpoint, dtype=dtype, device="cpu", layers=layers, expert_cache=args.expert_cache
    )
    print(f"assembled    {len(model.layers)} layers in {time.time() - t0:.1f}s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint.root)
    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    print(f"prompt       {ids.shape[1]} tokens: {args.prompt!r}")

    t0 = time.time()
    mask = build_attention_masks(ids.shape[1] + args.tokens, config.resolved_window, dtype)
    print(f"masks        {sorted(mask)} in {time.time() - t0:.2f}s")

    t0 = time.time()
    out, tops = greedy(model, ids, args.tokens, mask)
    elapsed = time.time() - t0
    new_ids = out[0, ids.shape[1] :].tolist()
    rate = args.tokens / elapsed
    print(f"decoded      {args.tokens} tokens in {elapsed:.1f}s ({rate:.2f} tok/s)")
    print(f"max logit    min {min(tops):.3f} max {max(tops):.3f}")
    print(f"ids          {new_ids}")
    print(f"text         {tokenizer.decode(new_ids)!r}")

    if args.layers is not None:
        print(f"NOTE: stack truncated to {args.layers} layers; the text is not meaningful")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
