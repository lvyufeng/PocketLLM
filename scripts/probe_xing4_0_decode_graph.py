"""What a CUDA graph would buy Xing4.0's decode step, measured before it is built.

The stage-5 record leaves the decode step at 6.5 tok/s against a 152 tok/s byte
floor and attributes it to host submission: 178 ms of host against 46.5 ms of
device work in a 176 ms step.  That attribution is a *difference* between two
figures taken different ways, and the number that decides the next stage is not
the difference -- it is how much of the 178 ms a graph actually recovers.  This
probe is that number.

Three arms, one process, the same weights, interleaved A-B-A-B-A-A-C so drift
lands on every column:

    eager     the step as the server runs it, in a loop
    graph     the same forward captured at one frozen position and replayed
    compile   `torch.compile(mode="reduce-overhead")` around the same step

`graph` is deliberately not a decode loop.  Every replayed step is the same
position at the same token, which is what makes the capture legal before any of
the position-as-a-tensor work is done -- a Python `start_pos` is baked into the
capture, so the graph is only the same graph at every position once that work
exists.  It is the *upper bound*: it answers "how much of this step is launch
cost at all", and if the answer is small the rest of the plan is wrong.

What the capture needs is one thing, and it is not a model change: the token has
to arrive as a CUDA tensor.  `Xing4_0GGUFModel.embed` calls
`torch.as_tensor(input_ids, device=...)`, which is a pageable host-to-device copy
for a Python list and the identity for a tensor already on the card.  Nothing
else in the decode forward transfers or synchronises -- `generate.sample_token`'s
`.cpu()` is at the end of the step and stays outside this probe by construction.

    python scripts/probe_xing4_0_decode_graph.py --device cuda:2 --context 4096
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.xing4_0.gguf_model import Xing4_0GGUFModel

DEFAULT_GGUF = "/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
DEFAULT_RELEASE = "/mnt/data2/Xing4.0-29B-A4B"
# The same corpus the e2e bench measures on, so this probe's eager column is the
# number the stage-5 record already states rather than a second baseline.
CORPUS = Path(__file__).resolve().parents[1] / "docs" / "models" / "qwen3.8-27b-fp8.md"


def build_prompt(tokenizer, tokens: int) -> list[int]:
    """Real prose, repeated and truncated to exactly `tokens` ids."""
    text = CORPUS.read_text(encoding="utf-8")
    ids: list[int] = []
    while len(ids) < tokens:
        ids.extend(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids[:tokens]


def count_launches(body) -> tuple[int, int, int, float]:
    """What one call of `body` costs the host to submit, and the card to run.

    Four counts, because "launches" has meant all of them at different times and
    only one of them is the host's bill:

        submissions  `cudaLaunchKernel` calls -- what the host actually pays for
        kernels      GPU kernel instances, which is what a graph node replaces
        copies       device-to-device copies
        device_ms    the card's own time

    A CUDA-only profile of one step reports 11,231 rows, and the split below is
    why: 4,380 of them are kernels and the rest are the runtime API calls and
    copies around them.  `aten::` rows cannot appear here -- they are CPU-side
    dispatches -- and counting them needs
    `profile_xing4_0_decode_launches.py`, which profiles both activities.
    """
    import torch.profiler as prof

    with prof.profile(
        activities=[prof.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as p:
        body()
        torch.cuda.synchronize()
    submissions = kernels = copies = 0
    device_us = 0.0
    for row in p.key_averages():
        count = int(row.count)
        if row.key.startswith("cudaLaunchKernel"):
            submissions += count
        elif "Memcpy" in row.key or "Memset" in row.key:
            copies += count
        elif row.key.startswith("aten::"):
            continue
        else:
            kernels += count
        device_us += float(row.self_device_time_total)
    return submissions, kernels, copies, device_us / 1000.0


def timed(step, steps: int) -> float:
    """Wall seconds for `steps` calls of `step`, submitted back to back."""
    step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arms", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    # Before anything else, and it is not cosmetic: `torch.cuda.graph` captures
    # on the *current* device's stream, and naming a card in `--device` does not
    # set it.  With the current device left at 0 and the weights on cuda:2, every
    # op in the capture body fails with `cudaErrorStreamCaptureUnsupported`
    # reported at whatever call happens to come next.
    torch.cuda.set_device(torch.device(args.device).index)

    sys.path.insert(0, args.release)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.release, trust_remote_code=True)

    model = Xing4_0GGUFModel(
        args.gguf,
        device=args.device,
        config_path=str(Path(args.release) / "config.json"),
        use_kernel=True,
    )
    torch.cuda.synchronize()
    print(f"loaded {model.nbytes / 2**30:.2f} GiB on {args.device}")

    ids = build_prompt(tokenizer, args.context)
    cache = model.make_cache(args.max_model_len, batch=1)

    # The prompt, chunked the way the server does it, so the cache is in a real
    # state at the frozen position rather than a synthetic one.
    for offset in range(0, args.context, 128):
        model.forward(ids[offset : offset + 128], cache=cache, start_pos=offset)
    torch.cuda.synchronize()
    pos = args.context

    # One token, on the card, at a fixed address -- the only thing the capture
    # needs that the server does not already do.
    token = torch.tensor([ids[0]], dtype=torch.int64, device=args.device)

    def step_eager():
        return model.forward(token, cache=cache, start_pos=pos)

    # -- graph arm ---------------------------------------------------------- #
    # The pool rules are `graphs.py`'s: one handle for the tree, and a fresh one
    # per capture round because a pool's id has no lifetime of its own.
    pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step_eager()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    # `torch.cuda.graph` empties the cache on entry, so a baseline taken before it
    # is a baseline the capture has already thrown away -- the delta comes out
    # negative by exactly the warm-up's reservation.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_stats()["reserved_bytes.all.current"]
    with torch.cuda.graph(graph, pool=pool):
        graph_out = step_eager()
    torch.cuda.synchronize()
    # The pool's own cost, read while the graph that owns it is alive: the
    # allocator reserves for it and frees it when the last capture into that pool
    # is gone.  Read from `memory_stats`, not from `memory_reserved`, which drops
    # when a warmed-up segment is released and gives a negative delta.
    pool_bytes = torch.cuda.memory_stats()["reserved_bytes.all.current"] - reserved_before
    print(f"captured: one graph over the whole decode step, {pool_bytes / 2**20:.1f} MiB of pool")

    def step_graph():
        graph.replay()
        return graph_out

    # -- compile arm -------------------------------------------------------- #
    compiled = None
    if not args.no_compile:
        import torch._dynamo as dynamo

        # `start_pos` is a Python int, so this compiles once *at this position*.
        # A decode loop would recompile per position, which is the number the
        # record has to carry rather than hide.
        for mode in ("reduce-overhead", "default"):
            dynamo.reset()
            try:
                candidate = torch.compile(step_eager, mode=mode, fullgraph=False)
                started = time.perf_counter()
                candidate()
                torch.cuda.synchronize()
                graphs = dynamo.utils.counters["stats"].get("unique_graphs", None)
                print(f"compiled ({mode}) in {time.perf_counter() - started:.1f} s, "
                      f"{graphs if graphs is not None else '?'} inductor graphs")
                compiled = candidate
                break
            except Exception as error:  # noqa: BLE001 - a refused compile is a result
                print(f"compile refused ({mode}): {type(error).__name__}: {error}")
        if compiled is None:
            print("compile arm dropped")

    # -- parity -------------------------------------------------------------- #
    eager_out = step_eager().float()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    delta = float((eager_out - graph_out.float()).abs().max())
    print(f"graph against eager at the frozen position: max |dlogits| = {delta:.3e}")
    del eager_out

    arms = {"eager": step_eager, "graph": step_graph}
    if compiled is not None:
        arms["compile"] = lambda: compiled()

    for name, body in arms.items():
        for _ in range(args.warmup):
            body()
        torch.cuda.synchronize()
        # Time first and profile second: the profiler's own instrumentation is
        # host work, and the host is exactly what the eager column is measuring.
        step_ms = timed(body, args.steps) * 1000
        submissions, kernels, copies, device_ms = count_launches(body)
        print(
            f"{name:>8}: {step_ms:8.1f} ms/step  {1000 / step_ms:6.2f} tok/s   "
            f"{submissions:6d} submissions ({kernels} kernels, {copies} copies)  "
            f"{device_ms:6.1f} ms device  ({100.0 * device_ms / step_ms:4.0f}% busy)"
        )

    # A-B-A-B ordering, so a warm-up or a clock change is not read as an arm.
    print()
    print("interleaved, three rounds:")
    order = ["eager", "graph", "eager", "graph", "eager"] + (["compile", "compile"] if compiled else [])
    for name in order:
        step_ms = timed(arms[name], args.steps) * 1000
        print(f"  {name:>8}: {step_ms:8.1f} ms/step  {1000 / step_ms:6.2f} tok/s")

    # Last, because it perturbs the host: 200 replays submitted back to back with
    # no synchronise between them.  Whatever the host manages to queue ahead of
    # the device is hidden, so the number is the step's *device* time, and it is
    # what a long run of graphed steps converges to.
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(200):
        graph.replay()
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - started) / 200 * 1000
    print()
    print(f"one graph.replay() with the host free to run ahead: {replay_ms:6.1f} ms  "
          f"({1000 / replay_ms:5.2f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
