#!/usr/bin/env python
"""What the fused decode attention is worth on the token, as a difference of two shipped paths.

`mimo_decode_attention` is the only kernel in the MiMo path that is not bit-exact, and what it
buys is host time: `probe_mimo_v2_attention_path.py` puts `scores` + `softmax` + `out` + `view` at
19.1 ms of a token's host at sixteen resident rows, which is eighteen eager dispatches a layer. The
arm below is the path this model took before the kernel existed -- `_decode_ops = None`, so
`decode_output` takes the torch block -- and the kernel arm minus it is the kernel's price.

**The two arms are not bit-identical**, which is the whole difficulty of reading this one, and the
probe is shaped around it. The kernel's float32 agrees with the torch block's to about 3e-7
relative, but the answer that leaves the layer is bfloat16, so a fraction of the elements land on
the other side of a rounding boundary and the arms' hidden states are not the same tensor. The
draws the router then takes are *usually* the same and can differ, and a differing draw changes how
much the step copies over PCIe -- which is 34 ms of the token on its own. So a price read off the
clock alone would be a price of "the kernel plus however many copies its last bits moved".

Three things in this probe are what make the number readable:

* **The chain is fixed**, derived once with the kernel shipped, and both arms replay the same
  tokens at the same positions. Neither arm samples, so the comparison is not about the sampler's
  luck.
* **The draws are compared, not assumed.** Each arm reports the resident hit rate and the number of
  experts staged, and a pair whose hit rates differ is a pair whose copies differ -- the line the
  probe prints between the arms is the confounder's size rather than a caveat.
* **The rounds are interleaved** `off, on, off, on, ...`, so the drift this box has between two
  runs of the same configuration lands on both arms, and the arms are timed in one process minutes
  apart rather than in two processes an afternoon apart.

The clock is split into the host's time -- the wall between the step's first dispatch and its last,
which is where a host-bound decode step spends the kernel's saving -- and the queue behind it.

Measured at sixteen resident rows on four ranks, one process, `--steps 8`: the **held** pair reads
**108.0 -> 91.9 ms a step** (9.26 to 10.88 tok/s) and **107.4 -> 92.9** (9.31 to 10.77) in two runs,
all of it on the host, with the queue unchanged at 4.5 ms. The **free** pair reads **-19.9** and
**-19.1 ms** in the same two runs -- bigger, and to be read as "the kernel plus the copies its last
bits moved", because the run whose counters are printed staged 996 experts with the torch block
against 853 with the kernel over its 32-step replay. The held pair agreeing to 1.6 ms across two
runs while the free pair ranged over ten is the reason the hold is there.

The same box, the same afternoon, `probe_mimo_v2_ablate.py --steps 10 --warmup 3 --rounds 4
--arms shipped` reads **125.8 ms a token over four rounds with a 110.4 to 141.3 spread**: an
absolute number on this host moves further across rounds than the kernel is worth, which is why
nothing above is quoted as a rate on its own.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_attention_ab.py --resident-rows 16 --steps 8
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


def reach(model, *, fused: bool) -> None:
    """Set every attention module's switch, which is the whole of the two arms.

    A module whose loader found no extension has `_decode_ops is None` already, so the `fused` arm
    of a build without the op is the reference twice -- the probe says so above the table rather
    than returning a difference of zero and letting it read as a result.
    """
    from src.kernels.cuda_loader import load_cuda_kernel

    ops = load_cuda_kernel()
    for layer in model.layers:
        layer.attention._decode_ops = (
            ops if (fused and ops is not None and hasattr(ops, "mimo_decode_attention")) else None
        )


def record(model) -> tuple[dict, list]:
    """Install a recorder on every routed layer; return the store and the undo.

    This is what makes the clean arm possible. The two arms replay the same tokens at the same
    positions, so `route` is called the same number of times in the same order in both, and a queue
    of the recorded draws hands both arms the *same* experts -- which removes the copies from the
    difference. The queued tensors are the router's own device tensors rather than a recomputation
    of them, so the held arm still pays for the router.
    """
    held: dict[int, list] = {}
    undo: list[tuple[object, object]] = []
    for layer in model.layers:
        if layer.kind != "moe":
            continue
        undo.append((layer, layer.route))
        real = layer.route

        def rec(hidden, _real=real, _held=held, _name=layer.layer_idx):  # noqa: ANN001
            out = _real(hidden)
            _held.setdefault(_name, []).append((out[0].clone(), out[1].clone()))
            return out

        layer.route = rec
    return held, undo


def hold(model, held: dict) -> list:
    """Hand every `route` the recorded draw instead of the router's. Undo is the returned list."""
    undo = []
    for layer in model.layers:
        if layer.kind != "moe":
            continue
        undo.append((layer, layer.route))
        queue = list(held.get(layer.layer_idx, []))

        def stuck(hidden, _queue=queue):  # noqa: ANN001
            return _queue.pop(0)

        layer.route = stuck
    return undo


def unhold(undo: list) -> None:
    for layer, was_route in undo:
        layer.route = was_route


def walk(model, cache, prompt_ids, chain, position, *, steps=None):
    """Prefill the prompt from a reset cache, then replay `chain`, timing the replayed steps.

    `chain` is fed rather than sampled, so the position sequence is identical between calls and two
    calls differ only in the switch. The returned logits are the last one each step produced, which
    is what the equality is read off.
    """
    cache.reset()
    logits = None
    for index, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=index, cache=cache)[-1]
    torch.cuda.synchronize()
    host = tail = 0.0
    seen = chain if steps is None else chain[:steps]
    outs = []
    for offset, token in enumerate(seen):
        began = time.perf_counter()
        logits = model.forward(torch.tensor([token]), start_pos=position + offset, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - began) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        outs.append(logits.clone())
    return outs, host / max(1, len(seen)), tail / max(1, len(seen))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--resident-rows", type=int, default=16)
    parser.add_argument("--chain", type=int, default=0, help="tokens in the replayed chain")
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
        checkpoint, device=device, expert_source=bank, ep=ep, resident_rows=args.resident_rows
    )
    torch.cuda.synchronize()
    module = model.experts

    fused = sum(layer.attention._decode_ops is not None for layer in model.layers)
    print(f"[r{rank}] {fused} of {len(model.layers)} layers reach mimo_decode_attention", flush=True)
    if world > 1:
        torch.distributed.barrier()

    prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
    chain_tokens = args.chain or max(args.steps * (args.rounds + 1), 4 * args.steps)
    span = args.prompt + chain_tokens + 8
    cache = model.cache(span)

    # The chain, derived once with the kernel shipped: every round replays it, so the positions
    # are the same in both arms and the draws are compared rather than aligned by luck.
    reach(model, fused=True)
    logits = None
    for index, token in enumerate(prompt_ids):
        logits = model.forward(torch.tensor([token]), start_pos=index, cache=cache)[-1]
    chain = []
    for index in range(chain_tokens):
        token = int(logits.argmax())
        chain.append(token)
        logits = model.forward(torch.tensor([token]), start_pos=args.prompt + index, cache=cache)[-1]
    torch.cuda.synchronize()
    print(f"[r{rank}] the chain is {chain[:8]}...", flush=True)
    if world > 1:
        torch.distributed.barrier()

    # What the two arms do to the answer, and to the copies. The hit rate and the number of
    # experts staged are how many of the replay's draws the resident set answered and how many had
    # to come over PCIe, which is the quantity a differing draw would move -- so a pair whose two
    # numbers differ is a pair whose copies differ, and the price below is the kernel plus that.
    residents = module._residents

    def snapshot() -> tuple[int, int, int]:
        """The counters the two arms have to agree on: copies over PCIe, and resident draws."""
        if residents is None:
            return module.staged_experts, 0, 0
        return module.staged_experts, residents.hits, residents.drawn

    def measure(fused: bool):
        before = snapshot()
        reach(model, fused=fused)
        outs, _, _ = walk(model, cache, prompt_ids, chain, args.prompt)
        after = snapshot()
        return outs, tuple(b - a for a, b in zip(before, after))

    plain, plain_copies = measure(False)
    shipped, shipped_copies = measure(True)
    worst = max(float((a - b).abs().max()) for a, b in zip(plain, shipped))
    exact = all(bool(torch.equal(a, b)) for a, b in zip(plain, shipped))
    peak = max(float(a.abs().max()) for a in plain)
    print(
        f"[r{rank}] {len(plain)} steps: bit-exact {exact}, worst |delta| {worst:.3e} of a {peak:.1f} "
        f"peak; the replay staged {plain_copies[0]} experts with the torch block against "
        f"{shipped_copies[0]} with the kernel, resident hits {plain_copies[1]}/{plain_copies[2]} "
        f"against {shipped_copies[1]}/{shipped_copies[2]}",
        flush=True,
    )
    if world > 1:
        torch.distributed.barrier()

    # The recorded draws, taken off a kernel-arm replay so the hold is a draw the shipped model
    # really took. Both timed arms below replay them, which is what leaves the attention's own
    # dispatch as the only difference between the two clocks.
    reach(model, fused=True)
    saved_route, undo = record(model)
    try:
        walk(model, cache, prompt_ids, chain, args.prompt)
    finally:
        unhold(undo)
    print(f"[r{rank}] held {sum(len(v) for v in saved_route.values())} draws", flush=True)

    # Timing, interleaved by round so the drift lands on both arms, and twice: free-running, which
    # is what the model does and where the draws may move, and held, which is what the kernel is
    # worth with the copies removed from the difference.
    took: dict[str, list[tuple[float, float, float]]] = {}
    for label, held_draws in (("free", False), ("held", True)):
        took[label + ":torch"] = []
        took[label + ":kernel"] = []
        for _ in range(args.rounds):
            for arm, fused in (("torch", False), ("kernel", True)):
                reach(model, fused=fused)
                held = hold(model, saved_route) if held_draws else None
                try:
                    _, host, tail = walk(
                        model, cache, prompt_ids, chain, args.prompt, steps=args.steps
                    )
                finally:
                    if held is not None:
                        unhold(held)
                took[label + ":" + arm].append((host + tail, host, tail))
        if world > 1:
            torch.distributed.barrier()

    for label in ("free", "held"):
        seen = []
        for arm in ("torch", "kernel"):
            got = took[label + ":" + arm]
            token = sum(entry[0] for entry in got) / len(got)
            host = sum(entry[1] for entry in got) / len(got)
            tail = sum(entry[2] for entry in got) / len(got)
            seen.append((token, host, tail))
            print(
                f"[r{rank}] {label:4s} {arm:6s} {token:7.1f} ms a step  host {host:7.1f}  "
                f"queue {tail:5.1f}  {1000 / token:5.2f} tok/s",
                flush=True,
            )
        (off, host_off, _), (on, host_on, _) = seen
        print(
            f"[r{rank}] {label}: the fused attention is {on - off:+7.1f} ms a step "
            f"({on / off:.3f}x), {host_on - host_off:+7.1f} of it on the host",
            flush=True,
        )
    return 0
    print(
        f"[r{rank}] the fused attention is {on - off:+7.1f} ms a step ({on / off:.3f}x), "
        f"{host_on - host_off:+7.1f} of it on the host",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
