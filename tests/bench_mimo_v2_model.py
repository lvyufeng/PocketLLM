#!/usr/bin/env python
"""What a whole MiMo-V2.6 token costs on one card, and which part of it is the copy.

The stack runs end to end here -- embedding, forty-eight decoder layers over a KV cache,
the final norm and the head -- so the number this prints is a real token and not a layer.
It is also, deliberately, a baseline: the routed experts are the single-token kernel, so a
prompt is fed one token at a time and the prefill column below is a floor and not a
prefill. What the run is here to establish is the *shape* of the cost, and the shape has
one dominant term.

A decode step draws eight experts a layer. That is 12.75 MiB each, 102 MiB a layer and
**4.8 GiB a token** of host-to-device traffic, against roughly 250 MiB of weights that are
read on the card. So a token is a copy and everything else is noise in it, which is what
the split columns are for: `attn` and `mlp` are measured on the compute stream and the
rest is the difference.

Three things are measured rather than assumed:

* The link rate, from the bank's own pages to the card, over the same tensors the staging
  copies. A pinned bank hands the DMA engine the pages in place; without the pin the copy
  goes through PyTorch's staging ring, which is the same bytes at a lower rate, and the
  difference is one of the columns.
* `staged MiB/token`, the counters' total divided by the tokens decoded. It should be
  47 x 8 x 12.75 MiB for a full stack, and if it is not, some layer is not routing the way
  the model says it does.
* The kernel time, from CUDA events around the attention and the FFN of every layer, so
  "the copy is the token" is a claim with a number on both sides.

Usage:

    python tests/bench_mimo_v2_model.py                       # the full stack, 8 + 8 tokens
    python tests/bench_mimo_v2_model.py --layers 0,1,2,5 --steps 4
    python tests/bench_mimo_v2_model.py --slots 4 --no-pin    # what depth and the pin are worth

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
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
#: Prompt ids are drawn from the vocabulary and not tokenized: this measures the stack, and
#: a tokenizer that disagreed with the checkpoint would show up as a different prompt and
#: not as a wrong number.
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]


def h2d_rate(bank, layer: int, *, rounds: int = 16) -> tuple[float, float]:
    """Host-to-device rate over one expert's own pages, in GiB/s, with its size in MiB.

    One expert is 12.75 MiB and that is what a draw copies, so the probe is the smallest
    thing that has the same shape as the work: the six tensors, copied into device buffers
    of their own, exactly as the staging does it. Nothing is concatenated first, because a
    `cat` would allocate pageable memory and the copy would then be of that and not of the
    bank.
    """
    views = list(bank.expert_views(layer, 0).values())
    targets = [torch.empty(view.shape, dtype=view.dtype, device="cuda") for view in views]
    size = sum(view.numel() * view.element_size() for view in views)

    def copy() -> None:
        for view, target in zip(views, targets):
            target.copy_(view, non_blocking=True)

    copy()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        copy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return rounds * size / elapsed / 2**30, size / 2**20


class PhaseTimer:
    """Time every layer's attention and FFN on the compute stream, at event granularity.

    Event pairs measure stream time and not kernel time, so a layer waiting for its experts
    to land reports that wait inside `mlp` -- which is the honest place for it, since the
    wait is the copy and the copy is what the stage is made of.
    """

    def __init__(self, model) -> None:
        self._marks: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for layer in model.layers:
            self._wrap(layer.attention, "forward", "attn")
            self._wrap(layer, "mlp", "mlp")

    def _wrap(self, module, name: str, key: str) -> None:
        inner = getattr(module, name)

        def timed(*args, **kwargs):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            out = inner(*args, **kwargs)
            end.record()
            self._marks.append((key, start, end))
            return out

        setattr(module, name, timed)

    def read(self) -> dict[str, float]:
        totals = {"attn": 0.0, "mlp": 0.0}
        for key, start, end in self._marks:
            totals[key] += start.elapsed_time(end)
        self._marks.clear()
        return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="all", help="`all` or a comma-separated list")
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS), help="prompt tokens to feed")
    parser.add_argument("--steps", type=int, default=8, help="decode steps to measure")
    parser.add_argument("--slots", type=int, default=2, help="expert arena slots")
    parser.add_argument("--no-pin", action="store_true", help="leave the bank pageable")
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    layers = None if args.layers == "all" else [int(part) for part in args.layers.split(",")]

    started = time.perf_counter()
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint,
        layers=layers,
        device=args.device,
        expert_source=bank,
        slots=args.slots,
        pin=not args.no_pin,
    )
    torch.cuda.synchronize()
    arena = model.experts.arena_bytes / 2**20 if model.experts is not None else 0.0
    print(
        f"built {len(model.layers)} layers in {time.perf_counter() - started:.1f}s: "
        f"{model.memory_bytes / 2**20:.0f} MiB on the card, {arena:.0f} MiB of it arena"
    )
    if model.pin_result is not None:
        print(
            f"bank pin: rc={model.pin_result.rc} ok={model.pin_result.ok} "
            f"in {model.pin_result.seconds:.1f}s over {model.pin_result.bytes / 2**30:.1f} GiB"
        )
    else:
        print("bank pin: not asked for")

    routed = model.config.moe_layer_indices[0]
    rate, expert_mib = h2d_rate(bank, routed)
    print(f"host to device over an expert's own pages: {rate:.2f} GiB/s ({expert_mib:.2f} MiB)")

    cache = model.cache(args.prompt + args.steps + 8)
    print(f"cache: {cache.memory_bytes / 2**20:.1f} MiB over {len(cache)} layers")

    # A discarded first pass: the first token pays the CUDA context, the first staging and
    # whatever the driver does once, and every number below is a mean of what follows.
    model.greedy(PROMPT_IDS[: args.prompt], max_tokens=1, cache=cache)
    torch.cuda.synchronize()

    timer = PhaseTimer(model)
    counters = model.experts
    bank_marks = (counters.staged_experts, counters.staged_bytes) if counters else (0, 0)

    # The prompt, one token at a time: the only shape the single-token expert kernel has.
    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    cache.reset()
    started = time.perf_counter()
    for position, token in enumerate(prompt_ids):
        model.forward(torch.tensor([token]), start_pos=position, cache=cache)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - started
    prefill_phases = timer.read()

    staged_experts = counters.staged_experts if counters else 0
    staged_bytes = counters.staged_bytes if counters else 0
    prompt_experts = staged_experts - bank_marks[0]
    prompt_bytes = staged_bytes - bank_marks[1]

    # The decode loop: greedy, one token a step, the token it just drew fed back in.
    started = time.perf_counter()
    logits = model.forward(
        torch.tensor([prompt_ids[-1]]), start_pos=len(prompt_ids) - 1, cache=cache
    )[-1]
    drawn = [int(logits.argmax())]
    for step in range(args.steps):
        logits = model.step(drawn[-1], start_pos=len(prompt_ids) + step, cache=cache)[-1]
        drawn.append(int(logits.argmax()))
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - started
    decode_phases = timer.read()

    decode_experts = (counters.staged_experts if counters else 0) - staged_experts
    decode_bytes = (counters.staged_bytes if counters else 0) - staged_bytes

    print()
    print(f"{'phase':>8} {'tokens':>7} {'seconds':>9} {'ms/token':>9} {'tok/s':>8} {'attn ms':>8} {'mlp ms':>8}")
    for name, seconds, tokens, phases in (
        ("prefill", prefill_s, len(prompt_ids), prefill_phases),
        ("decode", decode_s, args.steps + 1, decode_phases),
    ):
        print(
            f"{name:>8} {tokens:>7} {seconds:>9.3f} {seconds / tokens * 1e3:>9.1f}"
            f" {tokens / seconds:>8.2f} {phases['attn'] / tokens:>8.2f} {phases['mlp'] / tokens:>8.2f}"
        )

    print()
    print(f"{'phase':>8} {'experts/token':>14} {'MiB/token':>10} {'copy ms/token':>14} {'share of token':>15}")
    for name, tokens, experts, byte_count, seconds in (
        ("prefill", len(prompt_ids), prompt_experts, prompt_bytes, prefill_s),
        ("decode", args.steps + 1, decode_experts, decode_bytes, decode_s),
    ):
        per_token = byte_count / tokens / 2**20
        copy_ms = byte_count / tokens / 2**30 / rate * 1e3
        print(
            f"{name:>8} {experts / tokens:>14.1f} {per_token:>10.1f} {copy_ms:>14.1f}"
            f" {copy_ms / (seconds / tokens * 1e3) * 100:>14.1f}%"
        )

    print()
    print(f"drew {len(drawn)} tokens, last 8: {drawn[-8:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
