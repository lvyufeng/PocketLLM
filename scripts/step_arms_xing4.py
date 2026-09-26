"""One decode step, three ways to submit it, interleaved inside one process.

The end-to-end bench puts each arm in its own process, which is right for a rate and wrong for a small
lever: this host's own spread across processes is 141 to 193 ms on the same step, so a comparison of
two arms that differ by 1% has to be taken with the arms alternating in one process, on one model.
That is what this script does.

    python scripts/step_arms_xing4.py --device cuda:2 --context 4096 --steps 24

`eager` is `Xing4_0GGUFModel.forward` with a Python position, `bucket` the same forward reading the
cache at the rung a graph would read, and `graph` the captured step replayed.  The gap between the
first two is what the *bucket* costs; between the last two, what the *recording* buys.

**Each arm gets its own cache, and that is not tidiness.** All three write the row their position names
and none of them is bit-identical to the others in every low bit -- a wider read is a different cuBLAS
tiling -- so a shared cache would let whichever arm ran last decide the row the next step of the others
reads.  Three caches of 0.19 GiB at a 4096-token context fit beside a 17.84 GiB model; the prompt is
forwarded once and the rows are copied in, so the three arms start from the same 4096 rows.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.xing4_0.decode_pos import Pos                        # noqa: E402
from src.models.xing4_0.gguf_model import Xing4_0GGUFModel           # noqa: E402
from src.models.xing4_0.graphs import DecodeGraphs                   # noqa: E402

DEFAULT_GGUF = "/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
DEFAULT_RELEASE = "/mnt/data2/Xing4.0-29B-A4B"
CORPUS = Path(__file__).resolve().parent.parent / "docs" / "models" / "qwen3.8-27b-fp8.md"
CHUNK = 128
ARMS = ("eager", "bucket", "graph")


def _clone(template, model, capacity: int, context: int):
    """A cache holding exactly the template's first `context` rows, and nothing of anyone else's."""
    cache = model.make_cache(capacity, batch=1)
    for layer, source in zip(cache, template):
        layer.latent[:, :context].copy_(source.latent[:, :context])
        layer.length = context
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="positions the cache is sized at; 0 gives the run exactly the room it needs, which puts "
        "the tail rung beside the context and hides the bucket's cost -- 8192 against a 4096-token "
        "context is the widest rung the ladder offers and reads 2.00x the rows a step needs",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.release)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.release, trust_remote_code=True)
    pool: list[int] = []
    text = CORPUS.read_text(encoding="utf-8")
    while len(pool) < args.context + args.steps:
        pool.extend(int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"])
    ids = pool[: args.context]

    torch.cuda.set_device(torch.device(args.device).index)
    started = time.perf_counter()
    model = Xing4_0GGUFModel(
        args.gguf, device=args.device, config_path=str(Path(args.release) / "config.json"), use_kernel=True
    )
    torch.cuda.synchronize()
    print(f"loaded {model.nbytes / 2**30:.2f} GiB in {time.perf_counter() - started:.1f} s on {args.device}")

    # The prompt is forwarded once, into a template the three caches are filled from -- the rows a step
    # reads are activations the model produced and not a fixture, and all three arms read the same ones.
    steps = args.rounds * args.steps
    capacity = args.capacity or args.context + steps + 8
    template = model.make_cache(capacity, batch=1)
    for offset in range(0, args.context, CHUNK):
        model.forward(ids[offset : offset + CHUNK], cache=template, start_pos=offset)
    torch.cuda.synchronize()

    caches, holders = {}, {}
    for arm in ARMS:
        caches[arm] = _clone(template, model, capacity, args.context)
        if arm != "eager":
            holders[arm] = DecodeGraphs(model, caches[arm], device=args.device)
    torch.cuda.empty_cache()
    token = int(ids[0])
    # Every rung the run can reach is recorded on the first step that needs one, so the capture cost
    # lands in the warm-up below rather than in a timed step. The bucket arm needs the ladder and not
    # a recording, because it submits the same ops eagerly.
    holders["graph"].reserve(args.context + steps)

    def take(arm: str, position: int) -> float:
        began = time.perf_counter()
        if arm == "eager":
            model.forward([token], cache=caches[arm], start_pos=position)
        elif arm == "bucket":
            model.forward(
                [token],
                cache=caches[arm],
                start_pos=Pos.bucket(position, holders[arm].bucket_for(position + 1)),
            )
        else:
            holders[arm].step(token, caches[arm], position)
        torch.cuda.synchronize()
        return (time.perf_counter() - began) * 1000

    # First touches: the rungs are recorded here, on the first step that needs them, and every arm is
    # warmed before anything is timed.
    for arm in ARMS:
        take(arm, args.context)

    holder = holders["graph"]
    rungs = holder.recorded
    print(
        f"context {args.context}, {len(rungs)} rungs {rungs[0]}..{rungs[-1]}, "
        f"capture {holder.capture_seconds:.1f} s over {len(rungs)} rungs "
        f"({holder.capture_seconds / len(rungs):.2f} s a rung), "
        f"pool {holder.pool_bytes / 2**20:.1f} MiB in {len(model.blocks)} blocks"
    )
    print(f"the largest rung reads {rungs[-1] / (args.context + 1):.2f}x the rows a step needs\n")

    for rung in sorted(holder.rung_bytes):
        print(f"  rung {rung:>6}: {holder.rung_bytes[rung] / 2**20:6.2f} MiB of pool")
    print()

    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for round_index in range(args.rounds):
        shift = round_index % len(ARMS)
        for index in range(args.steps):
            position = args.context + round_index * args.steps + index
            # Rotated rather than fixed, so a monotone drift in the host lands on all three arms.
            for arm in ARMS[shift:] + ARMS[:shift]:
                samples[arm].append(take(arm, position))

    print(f"{'arm':>7}  {'median':>8}  {'min':>7}  {'max':>7}  {'tok/s':>7}  {'vs eager':>8}  {'n':>3}")
    base = statistics.median(samples["eager"])
    for arm in ARMS:
        ms = samples[arm]
        print(
            f"{arm:>7}  {statistics.median(ms):8.1f}  {min(ms):7.1f}  {max(ms):7.1f}  "
            f"{1000 / statistics.median(ms):7.2f}  {base / statistics.median(ms):7.2f}x  {len(ms):3d}"
        )

    # The two parities. The replay against the same width submitted one launch at a time is the
    # capture's own claim; the bucket against the unbucketed forward is the *serving* question, and the
    # two are different questions -- one asks whether the recording is faithful, the other whether a
    # wider read changes the answer.
    graph_cache = caches["graph"]
    at = args.context
    first = holder.step_eager(token, graph_cache, at).clone()
    second = holder.step(token, graph_cache, at).clone()
    torch.cuda.synchronize()
    print(f"\nreplay vs same-width eager, max |dlogits| = {float((first - second).abs().max()):.3e}")

    # The bucket scan gets a **fresh cache per arm per position**, filled from the template. It has to:
    # a bucketed step writes a different row than an unbucketed one whenever their answers differ at
    # all in the low bits, so two long-lived caches would diverge from the first position and the scan
    # would then be comparing two different histories rather than two ways of reading one. The first
    # version of this scan did exactly that and reported a max delta of 26.
    rung = holders["bucket"].bucket_for(at + 1)
    worst, differ, changed = 0.0, 0, 0
    same, same_changed = 0.0, 0
    count = 16
    for position in range(at, at + count):
        clone = lambda: _clone(template, model, capacity, at)  # noqa: E731
        plain = model.forward([token], cache=clone(), start_pos=position).clone()
        wider = model.forward(
            [token], cache=clone(), start_pos=Pos.bucket(position, rung)
        )
        # The control: the bucket spelling at a width *equal* to the position, which is the width the
        # unbucketed read uses. If this is not exactly zero then the harness is comparing something
        # other than the width, and the number below means nothing.
        equal = model.forward([token], cache=clone(), start_pos=Pos.bucket(position, position + 1))
        torch.cuda.synchronize()
        delta = float((plain - wider).abs().max())
        worst = max(worst, delta)
        differ += delta > 0.0
        changed += int(plain.argmax()) != int(wider.argmax())
        same = max(same, float((plain - equal).abs().max()))
        same_changed += int(plain.argmax()) != int(equal.argmax())
    print(
        f"bucket vs unbucketed, {count} positions from {at}, a fresh cache a position:\n"
        f"    at rung {rung} ({rung / (at + 1):.2f}x the rows): max |dlogits| = {worst:.3e}, "
        f"{differ} of {count} differ at all, {changed} of {count} change the argmax\n"
        f"    at width {at + 1} (1.00x, the control): max |dlogits| = {same:.3e}, "
        f"{same_changed} of {count} change the argmax"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
