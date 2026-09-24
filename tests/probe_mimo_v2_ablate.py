#!/usr/bin/env python
"""What each region of a MiMo-V2.6 decode token costs the *step*, by deleting it and timing.

Two things this box has measured are worth more than a profile here. The first is that the profiler's
own per-op accounting is inflated by a factor that moves between processes -- the same
`cudaMemcpyAsync` has been recorded at 32.9 and at 96.3 ms a token -- so a table of `self_cpu_time`
is a table of shares and not of milliseconds. The second is that this box has reported the same
configuration 184 and 245 ms in one afternoon, so a number is only readable next to another number
taken in the same process minutes apart.

So the arms run interleaved, one process, and each arm is the shipped model with one region replaced
by a function that returns the right shape and does nothing else. What it returns is wrong and the
point is not the token, it is the clock: the difference between an arm and the shipped arm, taken a
round at a time, is what that region costs the step. Every stub is installed on every rank at once,
so the collectives stay symmetric.

The stubs, and what each one deletes:

    copies      every region but the copy: the draws are the router's own and the copies, the
                ordering and the kernel are the shipped ones, so this is the floor a token cannot
                go under while the experts come over PCIe. With a resident set the copy it prices
                is the miss, which is the point of the pair below it.
    copyfloor   the same, with the draw *cached*: a repeated draw is a resident hit, so the only
                copies left are the admits. What it measures is the step if every draw were held.
    stage       `_stage`: the six copies a row. The kernel still runs, on stale arena rows.
    attention   `MimoV2DeviceAttention.forward`: the qkv projection, the RoPE, the softmax, the
                gather and the output projection. The cache still advances, on zeros, so the next
                step's bounds are the ones it would have had -- which is why this arm does *not*
                price the append, and `kv` does.
    kv          `MimoV2KVCache.append`'s two `index_copy_`, with the write cursor still advanced.
    router      `route`, deleted: the gate linear, the correction bias, the three `topk` and the draw.
                The stub caches the first draw and returns it, so it is *not* the router's price
                alone -- the expert path then stages the same two experts every layer and the resident
                set hits every layer, which is `copyfloor`'s saving on top of the router's. Use
                `pyrouter` for the router's own price and `tests/probe_mimo_v2_router_ab.py` for the
                one call's.
    pyrouter    `route` still called and still returned, but from `layers.gate_and_route` rather than
                from the C++ transcription: `_route_ops = None` on every routed layer. This is the
                one arm that is a difference of two *shipped* paths, the draw is a real draw, and
                nothing downstream of the router knows which of the two answered it -- so `shipped`
                minus this arm is exactly what the transcription buys. It prices 0 when the extension
                is not built, which the line the probe prints above the table will say.
    rope        `attention.rotate`, both calls: the rotation of the query and of the key. The
                unrotated row comes back instead, so the attention is wrong; what is priced is the
                rotation's own work, whichever way this build dispatches it -- and on a build with
                no `rotate` to patch it is `rope_rows` that is stubbed.
    pyrope      the rotation through `rope_rows` rather than through the kernel, which is a
                *restore*: `shipped` minus this is what `mimo_rope_rows` is worth. It prices 0 on
                a build whose layers carry no rotation switch, which is a build without the
                kernel -- the same shape `pyrouter` has at line 41.
    norms       the two `rms_norm` a layer -- `device_model.normalise`, which is where they are
                called from the layer.
    reduce      `ep.reduce`, the layer's `all_reduce` of the expert partial.
    experts     the whole `MimoV2DeviceExperts.forward`: the `tolist`, the deal, the stage, the kernel.
    all         every one of the above at once, which is the floor of the loop, the embedding, the
                head and the sampler with the layers reduced to two adds.

And the arms that are not deletions. Three of them restore a path the shipped build does not take
and are each priced as a difference of two *shipped* builds, which is the only difference this box
can read:

    chunk-decode  forces every decode step through `attention_output`'s chunk path by answering
                  `decode_foldable` false, which is what the model did before `decode_output`
                  existed. `shipped` minus this arm is what the lean decode attention is worth,
                  and it is exact: the two arms are bit-identical, checked over 25 steps of a real
                  model before the arm was written.
    pyrouter      as above: the router off the card and back in Python.
    pyrope        as above: the rotation off the card and back in `rope_rows`.
    inference-mode
                  the step run under `torch.inference_mode` rather than `torch.no_grad`, which is
                  the same promise about autograd with less bookkeeping on every dispatch. It is
                  exact and it is not a deletion, which makes it the only arm here that a shipped
                  build can simply adopt -- and it is the one arm that goes quiet once the shipped
                  build has adopted it, since a second `inference_mode` costs nothing. Read it
                  against the `shipped` column's own decorator, not as a claim about the mode.
    nosync        hands `forward` a *host* indices tensor holding the last draw it took, so the
                  `tolist` inside it returns without touching the stream. The experts staged are
                  then the previous draw's, so the arithmetic is wrong; what the arm prices is the
                  device-to-host round trip a layer, which is 47 syncs a token. A CPU indices
                  tensor is what makes this work at all -- the shipped path passes `rows` and not
                  the ids to the kernel, so the only thing that reads `indices` is the `tolist`.

The arms are subtractive and they do not add up to the shipped token: two of them overlap (the
`experts` arm contains the `stage` arm) and the residual is the loop's own Python, the residuals and
whatever the collection of stubs leaves behind, which is the number to look at when the sum of the
columns is short of the token.

Usage:

    torchrun --nproc_per_node=4 tests/probe_mimo_v2_ablate.py --steps 5 --rounds 2
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

#: The arms, in the order they are printed. `shipped` is the arm every other one is subtracted from.
ARMS = (
    "shipped",
    "copies",
    "copyfloor",
    "nosync",
    "chunk-decode",
    "stage",
    "attention",
    "kv",
    "router",
    "pyrouter",
    "rope",
    "pyrope",
    "inference-mode",
    "norms",
    "reduce",
    "experts",
    "all",
)


def install(model, arm: str) -> callable:
    """Replace one region with a shape-correct no-op; return the undo.

    Every patch records `(owner, attribute, what was there)` at the moment it is made and the undo
    replays that list, rather than closing over a local that a later patch overwrites -- the first
    cut of this closed over one `kept` and the `all` arm's own undo put the expert kernel on the
    `rms_norm` call site.
    """
    if arm == "shipped":
        return lambda: None

    import src.models.mimo_v2.device_model as device_model_module

    if arm in ("copies", "copyfloor"):
        # Everything that is not the expert copy: the attention is a zero of the right shape down
        # the cache, the norms and the reduce are identity, and the draw is taken from the real
        # router. What is left is the copies, their ordering and the kernel, which is the floor a
        # token cannot go under while the experts come over PCIe.
        #
        # `copyfloor` goes one step further and *caches* the draw, which is what makes it readable
        # as a floor: with a resident set a repeated draw is a resident hit, so the arm's own
        # copies are the admits and nothing else, and the number is what the step would be if every
        # draw were held. The two arms differ in exactly that, and the difference is the copies.
        patch = []
        module = model.experts
        was_forward = module.forward

        if arm == "copyfloor":
            held = {}

            def cached_draw(hidden, indices, weights, *, layer_id=None, _was=was_forward, _h=held):
                key = int(layer_id)
                if key not in _h:
                    _h[key] = indices.tolist()
                return _was(
                    hidden, torch.tensor(_h[key], dtype=torch.int64), weights, layer_id=layer_id
                )

            patch.append((module, "forward", was_forward))
            module.forward = cached_draw

        for layer in model.layers:
            kept = layer.attention.forward

            def nothing(hidden, _layer=layer, *, start_pos=0, cache=None, **kwargs):  # noqa: ANN001, ANN003
                shape = cache.shape(_layer.layer_idx)
                rows = hidden.shape[0]
                cache.append(
                    _layer.layer_idx,
                    torch.zeros(
                        (shape.num_kv_heads, rows, shape.head_dim),
                        dtype=cache.dtype,
                        device=hidden.device,
                    ),
                    torch.zeros(
                        (shape.num_kv_heads, rows, shape.v_head_dim),
                        dtype=cache.dtype,
                        device=hidden.device,
                    ),
                )
                return {"attn_out_post_o": torch.zeros_like(hidden)}

            patch.append((layer.attention, "forward", kept))
            layer.attention.forward = nothing

        patch.append((device_model_module, "normalise", device_model_module.normalise))
        device_model_module.normalise = lambda hidden, weight, eps: hidden
        if model.ep is not None and model.ep.reduce is not None:
            patch.append((model.ep, "reduce", model.ep.reduce))
            model.ep.reduce = lambda out: out

        def restore_copyfloor() -> None:
            for owner, attribute, was in reversed(patch):
                setattr(owner, attribute, was)

        return restore_copyfloor

    if arm == "nosync":
        module = model.experts
        was = module.forward
        held: dict[tuple[str, int], list[int]] = {}

        def nosync_forward(hidden, indices, weights, *, layer_id=None, _was=was, _held=held):
            key = ("ids", int(layer_id))
            if key not in _held:
                _held[key] = indices.tolist()
            # On the host, so the `tolist` the shipped path does inside is a no-op rather than a
            # second round trip. The kernel is handed rows and not ids, so nothing else reads it.
            fake = torch.tensor(_held[key], dtype=torch.int64)
            return _was(hidden, fake, weights, layer_id=layer_id)

        module.forward = nosync_forward
        return lambda: setattr(module, "forward", was)

    if arm == "chunk-decode":
        from src.models.mimo_v2.device_attention import MimoV2DeviceAttention

        was = MimoV2DeviceAttention.decode_foldable
        MimoV2DeviceAttention.decode_foldable = lambda self, start_pos, cache: False

        def unpatch() -> None:
            MimoV2DeviceAttention.decode_foldable = was

        return unpatch

    patch: list[tuple[object, str, object]] = []

    def set_(owner, attribute: str, value) -> None:
        patch.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, value)

    stub_stage = arm in ("stage", "all")
    stub_attention = arm in ("attention", "all")
    stub_kv = arm in ("kv", "all")
    stub_router = arm in ("router", "all")
    stub_rope = arm in ("rope", "all")
    stub_pyrouter = arm == "pyrouter"
    stub_pyrope = arm == "pyrope"
    stub_inference_mode = arm == "inference-mode"
    stub_norms = arm in ("norms", "all")
    stub_reduce = arm in ("reduce", "all")
    stub_experts = arm in ("experts", "all")

    if stub_kv:
        from src.models.mimo_v2.device_attention import MimoV2KVCache

        def no_append(cache, layer, key, value):  # noqa: ANN001
            """The write cursor and the ring's trim, without the two `index_copy_`."""
            slots = cache._slots[layer]
            start = cache._written[layer]
            count = key.shape[1]
            if count > slots:
                start += count - slots
                count = slots
            cache._written[layer] = start + count
            return cache._written[layer]

        set_(MimoV2KVCache, "append", no_append)

    for layer in model.layers:
        if stub_attention:

            def no_attention(hidden, _layer=layer, *, start_pos=0, positions=None, cache=None, **kwargs):  # noqa: ANN001, ANN003
                # Zeros down the cache rather than a skipped append: the next step's bounds, the
                # ring's trim and `_check_readable` all read the cursor this sets, and an arm that
                # left it behind would be measuring a shorter sequence.
                shape = cache.shape(_layer.layer_idx)
                rows = hidden.shape[0]
                key = torch.zeros(
                    (shape.num_kv_heads, rows, shape.head_dim),
                    dtype=cache.dtype,
                    device=hidden.device,
                )
                value = torch.zeros(
                    (shape.num_kv_heads, rows, shape.v_head_dim),
                    dtype=cache.dtype,
                    device=hidden.device,
                )
                cache.append(_layer.layer_idx, key, value)
                return {"attn_out_post_o": torch.zeros_like(hidden)}

            set_(layer.attention, "forward", no_attention)
        if stub_router and layer.kind != "dense":
            real = layer.route
            held: dict[str, tuple] = {}

            def no_route(hidden, _real=real, _held=held):  # noqa: ANN001
                # Shape and dtype from the real router, taken once, so the expert path is fed
                # something a draw could have produced without paying for a draw.
                if "out" not in _held:
                    _held["out"] = _real(hidden)
                indices, weights = _held["out"]
                return indices.clone(), weights.clone()

            set_(layer, "route", no_route)
        if stub_pyrouter and layer.kind != "dense":
            # `_route_ops = None` is the whole arm: `route` then takes its own fallback, which is
            # the reference the transcription was checked against and the path this model took
            # before the extension existed. Nothing else about the layer changes.
            set_(layer, "_route_ops", None)

    if stub_norms:
        set_(device_model_module, "normalise", lambda hidden, weight, eps: hidden)

    if stub_rope:
        import src.models.mimo_v2.device_attention as attention_module

        # The rotation is skipped and the unrotated row is handed back, which is the wrong answer
        # in exactly the way the arm is supposed to be: the shape, the dtype and the cache
        # bookkeeping are the shipped ones, so what the arm prices is the rotation itself and not a
        # shorter attention.
        #
        # Which switch that is depends on the build, and both are covered rather than one being
        # assumed: a layer that can rotate through the kernel reaches it through `rotate`, so
        # patching the reference would price zero, and a build without the dispatch has the
        # reference as its only path. An arm that quietly prices zero is worse than one that
        # raises, so neither case is left to `getattr` failing.
        for layer in model.layers:
            if hasattr(layer.attention, "_rope_ops"):
                set_(layer.attention, "_rope_ops", None)
        if hasattr(attention_module.MimoV2DeviceAttention, "rotate"):
            set_(
                attention_module.MimoV2DeviceAttention,
                "rotate",
                lambda self, states, cos, sin, dim: states,
            )
        else:
            set_(attention_module, "rope_rows", lambda states, cos, sin, dim: states)

    if stub_pyrope:
        # `_rope_ops = None` is the whole arm and it is the *positive* one: the layer then rotates
        # through `rope_rows`, which is the reference and is what the model did before the kernel
        # existed. `shipped` minus this is what the kernel buys, on a step where nothing else moved.
        # A build whose layers carry no switch has nothing to hand back, and the arm prices 0 there.
        for layer in model.layers:
            if hasattr(layer.attention, "_rope_ops"):
                set_(layer.attention, "_rope_ops", None)

    if stub_inference_mode:
        # The one arm that changes nothing about the arithmetic: the step is already under
        # `torch.no_grad`, and `torch.inference_mode` is the same promise with less bookkeeping on
        # every one of its dispatches. A trivial `torch.add` on this box is 15.9 us under `no_grad`
        # and 9.8 under `inference_mode`, and a decode step is several thousand of those.
        set_(model, "step", torch.inference_mode()(model.step))

    for module in (model.experts, model.chunk_experts):
        if module is None:
            continue
        if stub_stage:
            set_(module, "_stage", lambda *args, **kwargs: None)
        if stub_experts:
            dim = module.dim

            def no_experts(hidden, *args, _dim=dim, **kwargs):  # noqa: ANN002, ANN003
                return torch.zeros(
                    (hidden.shape[0], _dim), dtype=torch.float32, device=hidden.device
                )

            set_(module, "forward", no_experts)

    if stub_reduce and model.ep is not None and model.ep.reduce is not None:
        set_(model.ep, "reduce", lambda out: out)

    def restore() -> None:
        for owner, attribute, was in reversed(patch):
            setattr(owner, attribute, was)

    return restore


def arm(model, cache, position: int, steps: int, warmup: int, logits):
    """`steps` measured steps after `warmup`, split into the host's time and the queue behind it."""
    step_position = position
    for _ in range(warmup):
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        step_position += 1
    torch.cuda.synchronize()
    host = tail = 0.0
    for _ in range(steps):
        began = time.perf_counter()
        logits = model.step(int(logits.argmax()), start_pos=step_position, cache=cache)[-1]
        queued = time.perf_counter()
        torch.cuda.synchronize()
        host += (queued - began) * 1e3
        tail += (time.perf_counter() - queued) * 1e3
        step_position += 1
    return (host + tail) / steps, host / steps, tail / steps, step_position, logits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", default=os.environ.get("POCKETLLM_MIMO_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--prompt", type=int, default=len(PROMPT_IDS))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--resident-rows", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint):
        print(f"no checkpoint at {args.checkpoint}; nothing to measure")
        return 0

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    ep = EpGroup.from_env()
    world, rank = ep.world, ep.rank
    device = ep.device if ep.device is not None else torch.device("cuda:0")
    checkpoint = MimoV2Checkpoint(args.checkpoint)
    bank = open_expert_bank(checkpoint)
    model = MimoV2DeviceModel(
        checkpoint, device=device, expert_source=bank, ep=ep, resident_rows=args.resident_rows
    )
    torch.cuda.synchronize()

    routed = [layer for layer in model.layers if layer.kind == "moe"]
    on_card = sum(1 for layer in routed if layer._route_ops is not None)
    print(
        f"[r{rank}] router: {on_card} of {len(routed)} routed layers route through the C++ "
        f"transcription, {len(routed) - on_card} through `layers.gate_and_route`",
        flush=True,
    )
    if all(hasattr(layer.attention, "_rope_ops") for layer in model.layers):
        rotated = sum(1 for layer in model.layers if layer.attention._rope_ops is not None)
        print(
            f"[r{rank}] rope: {rotated} of {len(model.layers)} layers rotate through "
            f"`mimo_rope_rows`, {len(model.layers) - rotated} through `rope_rows`",
            flush=True,
        )
    else:
        print(
            f"[r{rank}] rope: no layer carries a rotation switch, so all {len(model.layers)} "
            f"rotate through `rope_rows` and the `pyrope` arm prices 0",
            flush=True,
        )

    span = max(args.prompt, 8) + 2 * (args.warmup + args.steps) * (args.rounds + 1) * len(arms) + 16
    cache = model.cache(span)
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

    took: dict[str, list[tuple[float, float, float]]] = {name: [] for name in arms}
    # `shipped` runs first in every round so the round's own drift is visible as the spread of the
    # shipped column rather than charged to whichever arm happened to run late.
    order = ["shipped"] + [name for name in arms if name != "shipped"]
    for _ in range(args.rounds):
        for name in order:
            restore = install(model, name)
            try:
                token, host, tail, position, logits = arm(
                    model, cache, position, args.steps, args.warmup, logits
                )
            except Exception:
                print(f"[r{rank}] the {name} arm failed", flush=True)
                raise
            finally:
                restore()
            took[name].append((token, host, tail))
    torch.cuda.synchronize()
    if world > 1:
        torch.distributed.barrier()

    mean = {name: sum(entry[0] for entry in got) / len(got) for name, got in took.items()}
    if "shipped" in mean:
        base = mean["shipped"]
        for index in range(args.rounds):
            print(
                f"[r{rank}] round {index + 1}: "
                + "  ".join(f"{name} {took[name][index][0]:6.1f}" for name in order)
                + f"   (shipped this round {took['shipped'][index][0]:.1f})",
                flush=True,
            )
        print(
            f"[r{rank}] shipped {base:7.1f} ms a token ({1000 / base:5.2f} tok/s) over "
            f"{args.rounds} rounds, spread "
            f"{min(entry[0] for entry in took['shipped']):.1f}-"
            f"{max(entry[0] for entry in took['shipped']):.1f}",
            flush=True,
        )
    for name in order:
        got = took[name]
        token = mean[name]
        host = sum(entry[1] for entry in got) / len(got)
        tail = sum(entry[2] for entry in got) / len(got)
        delta = "" if name == "shipped" or "shipped" not in mean else f"  {token - base:+7.1f} ms"
        print(
            f"[r{rank}] {name:10s} {token:7.1f} ms a token  host {host:7.1f}  queue {tail:5.1f}  "
            f"{1000 / token:5.2f} tok/s{delta}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
