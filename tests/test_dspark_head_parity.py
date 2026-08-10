"""Numerical parity check: cpp_engine's DSpark output heads vs an fp32 reference.

The head is where a wrong draft turns into a *cheap* wrong draft instead of an
obviously broken one -- every way of getting it wrong still emits plausible
token ids. The pieces that can each be independently wrong:

  - hc_head collapses [block_size, hc, dim] with its own gate, a different
    parameterization from hc_pre's (4 mixes, one shared scale, no post/comb).
    Reusing hc_pre here would still produce tokens.
  - the loop is sequential: position i's markov bias is a bigram lookup on the
    token argmaxed at position i-1. Dropping the bias entirely, or computing
    every bias from the input token, both still produce tokens.
  - markov_w2 is stored [vocab, rank] and used untransposed. A transpose would
    only show in the numbers.
  - the confidence head reads concat(hc_head output, that position's markov
    embedding) -- the *pre-norm* hidden, not the normed one.

So this compares the drafted token ids exactly, not just norms. A norm check
alone would pass on several of the above.

Measured on the 0731 checkpoint: cpp/ref logits rel_l2 is 1.3e-7 (bf16 weights
against fp32 activations, no int8 anywhere). The three failure modes above read
8.3e-1, 5.7e-1, and 1.0e0 respectively, and each changes at least one drafted
token -- so the 1e-5 tolerance sits ~75x above the noise and ~5 orders below
any of them.

Usage:

    python tests/test_dspark_head_parity.py /mnt/data3/DeepSeek-V4-Flash-0731

Run under the `deepseek` conda env.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

MAGIC = 0x44534B48  # "DSKH"
CPP_BIN = os.path.join(os.path.dirname(__file__), "..", "cpp_engine", "build",
                       "tests", "test_dspark_head_parity")


def load_head_weights(ckpt_dir: str, stage_id: int, device):
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"mtp.{stage_id}."
    wanted = {
        "hc_head_fn": prefix + "hc_head_fn",
        "hc_head_scale": prefix + "hc_head_scale",
        "hc_head_base": prefix + "hc_head_base",
        "norm": prefix + "norm.weight",
        "markov_w1": prefix + "markov_head.markov_w1.weight",
        "markov_w2": prefix + "markov_head.markov_w2.weight",
        "conf": prefix + "confidence_head.proj.weight",
        "head": "head.weight",
    }

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for key, name in wanted.items():
        by_shard.setdefault(weight_map[name], []).append((key, name))

    out: dict[str, torch.Tensor] = {}
    for shard, items in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for key, name in items:
                out[key] = f.get_tensor(name).to(device).float()
    return out


def hc_head_reference(h4: torch.Tensor, w: dict, eps: float) -> torch.Tensor:
    """[rows, hc, dim] -> [rows, dim], mirroring ParallelHead.hc_head."""
    rows, hc, dim = h4.shape
    flat = h4.reshape(rows, hc * dim)
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = (flat @ w["hc_head_fn"].T) * rsqrt
    pre = torch.sigmoid(mixes * w["hc_head_scale"] + w["hc_head_base"]) + eps
    return (pre.unsqueeze(-1) * h4).sum(dim=1)


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * gamma


def head_reference(h4: torch.Tensor, input_token: int, w: dict, eps: float):
    """Returns (tokens [rows+1], logits [rows, vocab], confidence [rows])."""
    rows = h4.shape[0]
    x = hc_head_reference(h4, w, eps)
    logits = rmsnorm(x, w["norm"], eps) @ w["head"].T

    tokens = torch.empty(rows + 1, dtype=torch.long, device=h4.device)
    tokens[0] = input_token
    embeds = []
    for i in range(rows):
        embed = w["markov_w1"][tokens[i]]
        # markov_w2 is [vocab, rank]; used untransposed it maps rank -> vocab.
        logits[i] += w["markov_w2"] @ embed
        embeds.append(embed)
        tokens[i + 1] = logits[i].argmax()

    # Confidence reads the pre-norm hc_head output, concatenated with that
    # position's markov embedding.
    conf_in = torch.cat([x, torch.stack(embeds, dim=0)], dim=-1)
    confidence = (conf_in @ w["conf"].T).squeeze(-1)
    return tokens, logits, confidence


def run_cpp(cpp_bin: str, ckpt_dir: str, h4: torch.Tensor, input_token: int,
            vocab: int, tp_rank: int, tp_world: int):
    rows, hc, dim = h4.shape
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.bin")
        out_path = os.path.join(tmp, "out.bin")
        with open(in_path, "wb") as f:
            f.write(struct.pack("<5i", MAGIC, input_token, rows, hc, dim))
            f.write(h4.cpu().numpy().astype("<f4").tobytes())
        proc = subprocess.run([cpp_bin, ckpt_dir, in_path, out_path,
                               str(tp_rank), str(tp_world)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"cpp head parity binary failed ({proc.returncode})")
        raw = np.fromfile(out_path, dtype=np.uint8)

    off = 0
    tokens = raw[off:off + (rows + 1) * 4].view("<i4").copy()
    off += (rows + 1) * 4
    conf = raw[off:off + rows * 4].view("<f4").copy()
    off += rows * 4
    logits = raw[off:off + rows * vocab * 4].view("<f4").reshape(rows, vocab).copy()
    return tokens, conf, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--input-token", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--tp-world", type=int, default=1)
    # The head path is bf16 weights against fp32 activations throughout, with no
    # int8 activation quantization anywhere, so the measured error is ~1e-7 --
    # four orders below the 1e-3 that would merely "look small". The tolerance
    # is set where a real regression would land, not where the noise is.
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--cpp-bin", default=CPP_BIN)
    a = ap.parse_args()

    if not os.path.exists(a.cpp_bin):
        print(f"missing {a.cpp_bin}; build target test_dspark_head_parity first")
        return 2

    with open(os.path.join(a.ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    dim = cfg["hidden_size"]
    hc = cfg["hc_mult"]
    rows = cfg["dspark_block_size"]
    vocab = cfg["vocab_size"]
    eps = cfg["rms_norm_eps"]
    n_stages = 3

    device = "cuda"
    torch.manual_seed(a.seed)
    # Scale matches a block output after hc_post, which is O(1) per element.
    h4 = torch.randn(rows, hc, dim, dtype=torch.float32, device=device)

    cpp_tokens, cpp_conf, cpp_logits = run_cpp(a.cpp_bin, a.ckpt_dir, h4,
                                               a.input_token, vocab,
                                               a.tp_rank, a.tp_world)

    w = load_head_weights(a.ckpt_dir, n_stages - 1, device)
    with torch.inference_mode():
        ref_tokens, ref_logits, ref_conf = head_reference(h4, a.input_token, w, eps)

    ref_tokens_np = ref_tokens.cpu().numpy().astype(np.int32)
    failures = 0

    print(f"  ref tokens: {ref_tokens_np.tolist()}")
    print(f"  cpp tokens: {cpp_tokens.tolist()}")
    if not np.array_equal(ref_tokens_np, cpp_tokens):
        print("    [FAIL] drafted token ids differ")
        failures += 1
    else:
        print("    [ok] token ids match exactly")

    ref_logits_t = ref_logits.float()
    cpp_logits_t = torch.from_numpy(cpp_logits).to(device)
    lerr = ((cpp_logits_t - ref_logits_t).norm() / ref_logits_t.norm()).item()
    # The winning margin says how much slack the token ids had: a tiny margin
    # with matching ids means the match was luck, not agreement.
    top2 = ref_logits_t.topk(2, dim=-1)[0]
    margin = (top2[:, 0] - top2[:, 1]).min().item()
    print(f"  logits rel_l2={lerr:.3e} min top1-top2 margin={margin:.4f}")
    if not (lerr < a.tol):
        print(f"    [FAIL] logits rel_l2 {lerr:.3e} >= tol {a.tol}")
        failures += 1
    else:
        print("    [ok]")

    ref_conf_t = ref_conf.float()
    cpp_conf_t = torch.from_numpy(cpp_conf).to(device)
    cerr = ((cpp_conf_t - ref_conf_t).norm() / ref_conf_t.norm().clamp_min(1e-6)).item()
    print(f"  confidence rel_l2={cerr:.3e}")
    print(f"    ref: {[round(v, 4) for v in ref_conf_t.tolist()]}")
    print(f"    cpp: {[round(v, 4) for v in cpp_conf_t.tolist()]}")
    if not (cerr < a.tol):
        print(f"    [FAIL] confidence rel_l2 {cerr:.3e} >= tol {a.tol}")
        failures += 1
    else:
        print("    [ok]")

    print("PASS" if failures == 0 else f"FAIL ({failures} check(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
