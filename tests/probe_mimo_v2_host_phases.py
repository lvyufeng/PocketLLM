#!/usr/bin/env python
"""What a MiMo-V2.6 decode token's host is doing, split by the module that asked for it.

`probe_mimo_v2_decode_host.py` says the step is the host: at a short context a token is 178 ms of a
returning `model.step()` and 6 ms of queue behind it. That probe answers *how much*; this one answers
*where*, because the answer decides what a kernel is worth.

The instrument is `time.perf_counter` around each module's own entry point and nothing else -- no CUDA
events, no profiler. That is deliberate: the profiler's own accounting is inflated by a factor that
varies between processes (the same `cudaMemcpyAsync` has been recorded at 32.9 and at 96.3 ms a
token), and a CUDA event pair on a host-bound step measures the host's own starvation rather than the
region it was put around. Wall time around a call attributes exactly what the caller paid.

The regions nest, and the print says which is inside which:

    layer            the whole layer, attention and MLP together
      attention        `MimoV2DeviceAttention.forward`, gather included
      router           `layer.route`, the gate and the draw
      expert wrapper   `MimoV2DeviceExperts.forward` -- the `tolist`, the deal and the kernel call
        expert stage     `_stage`, which is host time only: the copies themselves are on the stream
        expert kernel    the kernel's own Python entry
      norms            the two `normalise` calls a layer

so `layer` minus its children is the residuals and the loop's own Python, and the token minus every
`layer` is the embedding, the final norm, the head and the sampler.

The overhead of the probe is itself measured and printed as `-- `, because a probe that adds 280
`perf_counter` pairs a token is a probe whose own number is not the step's: on a host-bound step the
honest total is `probe_mimo_v2_decode_host.py`'s, and this one is read for its shares.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_host_phases.py --steps 8
    torchrun --nproc_per_node=4 tests/probe_mimo_v2_host_phases.py --steps 8 --depth 32768
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
from tests.bench_mimo_v2_model import fill_cache  # noqa: E402

DEFAULT_CHECKPOINT = "/mnt/data3/MiMo-V2.6-Flash-RL"
PROMPT_IDS = [8374, 4021, 95012, 1288, 77431, 5502, 19904, 61783]

#: `(name, indent)` in the order a token pays them; the indent is the nesting.
REGIONS = (
    ("layer", 0),
    ("attention", 1),
    ("router", 1),
    ("expert wrapper", 1),
    ("expert stage", 2),
    ("expert kernel", 2),
    ("collective reduce", 1),
    ("collective gather", 1),
    ("norms", 1),
)


class Meter:
    """Wall time per named region, summed over a measured window, and the calls that made it."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.host: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.max: dict[str, float] = {}

    def region(self, name: str, run):
        """`run()` timed on the host, and the answer it returned."""

        def called(*args, **kwargs):  # noqa: ANN002, ANN003
            began = time.perf_counter()
            out = run(*args, **kwargs)
            took = time.perf_counter() - began
            self.host[name] = self.host.get(name, 0.0) + took
            self.calls[name] = self.calls.get(name, 0) + 1
            self.max[name] = max(self.max.get(name, 0.0), took)
            return out

        return called

    def clear(self) -> None:
        self.host.clear()
        self.calls.clear()
        self.max.clear()


def instrument(model, meter: Meter) -> None:
    """Wrap each named region where the model calls it, so the arms are the shipped paths."""
    from src.models.mimo_v2 import device_model as device_model_module

    for layer in model.layers:
        layer.forward = meter.region("layer", layer.forward)
        layer.attention.forward = meter.region("attention", layer.attention.forward)
        if layer.kind != "dense":
            layer.route = meter.region("router", layer.route)
    for module in (model.experts, model.chunk_experts):
        if module is None:
            continue
        module.forward = meter.region("expert wrapper", module.forward)
        module._stage = meter.region("expert stage", module._stage)
        # The kernel is an attribute on `_kernel`; wrap the two entry points the two paths use.
        for name in ("moe_single_token_fp4_forward", "moe_multi_token_fp4_forward"):
            inner = getattr(module._kernel, name, None)
            if inner is not None:
                setattr(module._kernel, name, meter.region("expert kernel", inner))
    if model.ep is not None:
        if model.ep.reduce is not None:
            model.ep.reduce = meter.region("collective reduce", model.ep.reduce)
        if model.ep.gather is not None:
            model.ep.gather = meter.region("collective gather", model.ep.gather)
    # The norms are called through `normalise`, the model's own fused entry point, so that is
    # the call site the meter has to wrap -- not `layers.rms_norm`, which is what the host
    # reference uses and which a decode step does not reach.
    device_model_module.normalise = meter.region("norms", device_model_module.normalise)


def arm(
    model, cache, position: int, steps: int, warmup: int, logits=None
) -> tuple[float, float, int, object]:
    """`steps` measured steps after `warmup`, split into the host's time and the wait behind it."""
    step_position = position
    for _ in range(warmup):
        feed = PROMPT_IDS[0] if logits is None else int(logits.argmax())
        logits = model.step(feed, start_pos=step_position, cache=cache)[-1]
        step_position += 1
    torch.cuda.synchronize()
    host = tail = 0.0
    for _ in range(steps):
        started = time.perf_counter()
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - started) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        step_position += 1
    return host / steps, tail / steps, step_position, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--depth", type=int, default=0, help="context the cache is filled to")
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--resident-rows", type=int, default=0)
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
    experts = model.experts
    print(
        f"[r{rank}] world {world} deal `{experts.deal}`, attention in "
        f"{ep.attention_shards} share(s), {experts.resident_rows} resident rows a layer",
        flush=True,
    )

    cache = model.cache(max(args.depth, args.prompt) + args.warmup + args.steps + 8)
    if args.depth:
        fill_cache(cache, [layer.layer_idx for layer in model.layers], args.depth)
        position = args.depth
    else:
        prompt_ids = (PROMPT_IDS * (args.prompt // len(PROMPT_IDS) + 1))[: args.prompt]
        model.greedy(prompt_ids, max_tokens=1, cache=cache)
        cache.reset()
        logits = None
        for position, token in enumerate(prompt_ids):
            logits = model.forward(torch.tensor([token]), start_pos=position, cache=cache)[-1]
        position = len(prompt_ids)
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    # The bare cost of the probe: the same steps with the wrappers in place and their sums thrown
    # away, so the number below can be read against it instead of against the honest one.
    meter = Meter(device)
    instrument(model, meter)
    _, _, position, logits = arm(model, cache, position, args.warmup, 0, logits)
    meter.clear()
    started = time.perf_counter()
    for _ in range(args.steps):
        logits = model.step(int(logits.argmax()), start_pos=position, cache=cache)[-1]
        position += 1
    torch.cuda.synchronize()
    behind = (time.perf_counter() - started) * 1e3 / args.steps

    meter.clear()
    host, tail, position, logits = arm(model, cache, position, args.steps, 0, logits)
    print(
        f"\n[r{rank}] with the wrappers: {host:7.1f} ms on the host, {tail:6.1f} ms behind "
        f"({host + tail:6.1f} ms a token, {1000 / (host + tail):5.2f} tok/s); a step with them in "
        f"place and not summed is {behind:6.1f} ms",
        flush=True,
    )
    per_step = {
        name: meter.host[name] / args.steps * 1e3 for name, indent in REGIONS if name in meter.host
    }
    for name, indent in REGIONS:
        if name not in per_step:
            continue
        print(
            f"[r{rank}]   {'  ' * indent}{name:16s} host {per_step[name]:7.1f} ms  "
            f"{meter.calls[name] / args.steps:6.1f} calls  "
            f"max {meter.max[name] * 1e3:7.3f} ms",
            flush=True,
        )
    print(
        f"[r{rank}]   {'outside a layer':16s} host "
        f"{host - per_step.get('layer', 0.0):7.1f} ms  "
        f"(embedding, head, the loop and the sampler)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
