"""In-process A/B of the fused hyper-connection kernel, interleaved A-B-A-B.

One model is loaded, because two do not fit in 22 GiB, and the arm is switched by
rebinding `HyperConnection.cuda` on the same objects -- the same weights, the same
KV cache, the same prompt.  The arms alternate so a clock or thermal drift lands on
both, and only the arms' *difference* is read, never an absolute token count.
"""

import argparse
import statistics
import sys
import time

sys.path.insert(0, "/mnt/data2/Xing4.0-29B-A4B")
sys.path.insert(0, "/mnt/data1/dsv4_inference")

import torch

from src.models.xing4_0.gguf_model import Xing4_0GGUFModel

GGUF = "/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf"
PROMPT = (
    "The history of the telescope begins with the spectacle makers of the "
    "Low Countries, and it is a history that turns on a single question: how "
    "much light can one gather, and how sharply can one focus it? Explain, "
    "step by step, how a refractor and a reflector differ."
)


def build_model(device, block_count, use_kernel):
    t0 = time.perf_counter()
    model = Xing4_0GGUFModel(GGUF, device=device, block_count=block_count, use_kernel=use_kernel)
    print(f"loaded {model.nbytes / 2**30:.2f} GiB in {time.perf_counter() - t0:.1f} s")
    return model


def set_arm(model, on):
    from src.models.xing4_0.hyper_connection import _load_hyper_connection_kernel

    mod = _load_hyper_connection_kernel() if on else None
    n = 0
    for blk in model.blocks:
        for hc in (blk.attn_hc, blk.ffn_hc):
            hc.cuda = mod
            n += 1
    return n


@torch.inference_mode()
def run_arm(model, ids, steps, cache, start_pos):
    """Greedy decode from a primed cache; returns (seconds, tokens)."""
    token = ids[:, -1:]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    for i in range(steps):
        # `forward` returns `[tokens, vocab]`, so a one-token step's logits are a
        # row -- `logits[-1]`, not `logits[:, -1]`, which is column `vocab - 1`.
        step = model.forward(token, cache=cache, start_pos=start_pos + i)
        token = step.argmax(dim=-1).reshape(1, 1)
        out.append(int(token))
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--arms", type=int, default=3, help="A-B pairs; total arms = 2x this")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    from tokenization_xing4_0 import Xing4_0Tokenizer

    tok = Xing4_0Tokenizer("/mnt/data2/Xing4.0-29B-A4B/tokenizer.model")
    ids = torch.tensor([tok.encode(PROMPT)], device=args.device)
    print(f"prompt {ids.shape[1]} tokens")

    model = build_model(args.device, args.blocks, True)

    cache = model.make_cache(capacity=ids.shape[1] + args.steps + 8, batch=1)
    set_arm(model, True)
    t0 = time.perf_counter()
    model.forward(ids[:, :-1], cache=cache, start_pos=0)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    print(f"prefill {(ids.shape[1] - 1) / prefill_s:.1f} tok/s ({prefill_s * 1000:.0f} ms)")

    arms = {}
    texts = {}
    order = []
    for round_index in range(args.arms):
        for on in ((round_index % 2 == 0), (round_index % 2 == 1)):
            tag = "kernel" if on else "eager"
            n = set_arm(model, on)
            if round_index == 0 and not on:
                print(f"rebound {n} hyper-connections to the eager path")
            secs, tokens = run_arm(model, ids, args.warmup + args.steps, cache, ids.shape[1] - 1)
            per = secs / (args.warmup + args.steps) * 1000
            arms.setdefault(tag, []).append(per)
            order.append(tag)
            if round_index == 0:
                texts.setdefault(tag, tokens)
            print(f"  arm {len(order):2d} {tag:6s} {per:7.1f} ms/token  {1000 / per:5.2f} tok/s")

    sample = {tag: tok.decode(t[:16]) for tag, t in texts.items()}
    print()
    for tag in ("eager", "kernel"):
        print(f"{tag:6s} says: {sample[tag]!r}")
    if len(texts) == 2:
        same = next((i for i, (a, b) in enumerate(zip(texts['kernel'], texts['eager'])) if a != b), None)
        print(f"first differing token: {same}")
    for tag in ("eager", "kernel"):
        vals = arms[tag]
        print(f"{tag:6s} median {statistics.median(vals):7.1f} ms/token  min {min(vals):7.1f}  {len(vals)} arms")
    e, k = statistics.median(arms["eager"]), statistics.median(arms["kernel"])
    print(f"kernel/eager = {k / e:.3f}  ({e / k:.2f}x)")


if __name__ == "__main__":
    main()
