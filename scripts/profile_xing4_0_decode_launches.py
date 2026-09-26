"""Where a Xing4.0 decode step's kernel launches come from.

The stage-5 record priced the gap (178 ms of host submission against 46.5 ms of
device work) but not its distribution.  This walks one decode step at a time and
prints the PyTorch ops behind the count, which is what a fusion or a graph acts
on.

    python scripts/profile_xing4_0_decode_launches.py --device cuda:2 --context 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.xing4_0.gguf_model import Xing4_0GGUFModel

DEFAULT_GGUF = "/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
DEFAULT_RELEASE = "/mnt/data2/Xing4.0-29B-A4B"
CORPUS = Path(__file__).resolve().parents[1] / "docs" / "models" / "qwen3.8-27b-fp8.md"


def build_prompt(tokenizer, tokens: int) -> list[int]:
    """Real prose, repeated and truncated to exactly `tokens` ids."""
    text = CORPUS.read_text(encoding="utf-8")
    ids: list[int] = []
    while len(ids) < tokens:
        ids.extend(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids[:tokens]


def table(rows, header, key, top):
    print(f"{header[0]:<62} {header[1]:>9} {header[2]:>11} {header[3]:>12}")
    print("-" * 98)
    for name, count, cuda_us, cpu_us in rows[:top]:
        # Rounded, not averaged to a fraction: the count is one step's, and a
        # kernel that ran 26 times a step did not run 26.67 times.
        print(f"{name[:62]:<62} {int(round(count)):>9} {cuda_us / 1000:>11.2f} {cpu_us:>12.1f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--top", type=int, default=35)
    args = ap.parse_args()

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
    cache = model.make_cache(args.context + args.steps + 8, batch=1)
    # The prompt, outside the profile: this is about the decode step.
    model.forward(ids[:128], cache=cache, start_pos=0)
    for offset in range(128, args.context, 128):
        model.forward(ids[offset : offset + 128], cache=cache, start_pos=offset)
    torch.cuda.synchronize()
    n_blocks = len(model.blocks)

    import torch.profiler as prof

    steps = []
    for step in range(args.steps):
        with prof.profile(
            activities=[prof.ProfilerActivity.CUDA, prof.ProfilerActivity.CPU],
            record_shapes=False,
            with_stack=False,
        ) as p:
            out = model.forward([ids[0]], cache=cache, start_pos=args.context + step)
            out.sum().item()
        torch.cuda.synchronize()
        steps.append(p.key_averages())

    def merged(attr_count, attr_cuda, attr_cpu):
        totals: dict[str, list] = {}
        for avg in steps:
            for row in avg:
                entry = totals.setdefault(row.key, [0, 0.0, 0.0])
                entry[0] += int(getattr(row, attr_count))
                entry[1] += float(getattr(row, attr_cuda))
                entry[2] += float(getattr(row, attr_cpu))
        return totals

    n_steps = len(steps)

    # Everything below is *one* step, not the profiled run: the columns are
    # divided, because a per-op count that silently means three steps is the kind
    # of number a reader multiplies by forty and gets the wrong answer.
    cpu = merged("count", "self_device_time_total", "self_cpu_time_total")
    cpu = {k: [v[0] / n_steps, v[1] / n_steps, v[2] / n_steps] for k, v in cpu.items()}
    # `aten::` rows are the dispatcher calls; each is at least one kernel launch.
    aten = [(k, v[0], v[1], v[2]) for k, v in cpu.items() if k.startswith("aten::")]
    aten.sort(key=lambda r: -r[1])
    total_aten = int(sum(r[1] for r in aten))
    print()
    print(f"one decode step: {total_aten} aten dispatches, "
          f"{total_aten // n_blocks} a block over {n_blocks} blocks")
    print()
    table(aten, ("aten op", "calls", "cuda ms", "cpu us"), None, args.top)

    # The kernels themselves, by device time.
    ker = [(k, v[0], v[1], v[2]) for k, v in cpu.items() if not k.startswith("aten::")]
    ker.sort(key=lambda r: -r[2])
    table(ker, ("kernel / other", "count", "cuda ms", "cpu us"), None, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
