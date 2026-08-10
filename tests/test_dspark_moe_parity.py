"""Numerical parity check: cpp_engine's DSpark MoE FFN vs an fp32 reference.

Weights loading and attention parity say nothing about the FFN. The MoE is
where a draft goes wrong quietly, because routing is discrete: send a token to
the wrong expert and the output is still a well-scaled vector of plausible
numbers -- just the wrong one. No norm-based check on the final output would
localize that, so this test compares the routing decisions directly *and* the
numerics.

The pieces that can each be independently wrong:
  - the gate's score function is sqrt(softplus(x)), not softmax or sigmoid
  - topk selects on score+bias but renormalizes the *unbiased* scores
  - weights are scaled by routed_scaling_factor after renormalization
  - routes are grouped expert-major before the batched expert GEMM
  - the shared expert is added on top, and (unlike routed experts) applies no
    swiglu clamp -- it is built with the default swiglu_limit=0

Usage:

    python tests/test_dspark_moe_parity.py /mnt/data3/DeepSeek-V4-Flash-0731

Run under the `deepseek` conda env (the FP8/FP4 dequant paths need its torch).
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAGIC = 0x44534B4D  # "DSKM"
CPP_BIN = os.path.join(os.path.dirname(__file__), "..", "cpp_engine", "build",
                       "tests", "test_dspark_moe_parity")


def bf16_to_f32(t: torch.Tensor) -> torch.Tensor:
    return t.float()


def dequant_fp4(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unpack an FP4 (e2m1, two nibbles per int8) matrix to fp32.

    Layout matches cpp_engine's fp4 kernels: qweight is [rows, cols/2] int8,
    each byte holding columns 2c (low nibble) and 2c+1 (high nibble); scale is
    [rows, cols/32] e8m0, one exponent per 32-column group.
    """
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float32, device=qweight.device)
    q = qweight.to(torch.uint8)
    lo = lut[(q & 0x0F).long()]
    hi = lut[(q >> 4).long()]
    rows = q.shape[0]
    # Interleave back to [rows, cols]: byte c carries columns 2c and 2c+1.
    out = torch.stack([lo, hi], dim=-1).reshape(rows, -1)
    # e8m0: stored byte b means 2^(b-127).
    exp = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    cols = out.shape[1]
    out = out.view(rows, cols // 32, 32) * exp.unsqueeze(-1)
    return out.reshape(rows, cols)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 weight with a [rows/128, cols/128] blockwise e8m0 scale."""
    from src.kernels.ops import soft_fp8_blockfp8_weight_dequant
    return soft_fp8_blockfp8_weight_dequant(weight, scale).float()


def load_stage_ffn(ckpt_dir: str, stage_id: int, device, experts_start: int, experts_end: int):
    """Load one stage's gate + shared expert, and keep routed experts quantized.

    Routed experts are dequantized on demand by `expert_fp32` below: fp32 for
    all 256 of them is ~12 GB, but a block of 5 tokens touches at most 30, and
    in practice far fewer.
    """
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"mtp.{stage_id}.ffn."
    wanted = [k for k in weight_map if k.startswith(prefix)]

    def keep(name: str) -> bool:
        rest = name[len(prefix):]
        if not rest.startswith("experts."):
            return True
        eid = int(rest.split(".")[1])
        return experts_start <= eid < experts_end

    wanted = [k for k in wanted if keep(k)]
    by_shard: dict[str, list[str]] = {}
    for name in wanted:
        by_shard.setdefault(weight_map[name], []).append(name)

    raw: dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for name in names:
                t = f.get_tensor(name)
                # Routed experts stay packed (and on host) until needed.
                stem = name[len(prefix):]
                raw[stem] = t if stem.startswith("experts.") else t.to(device)

    return {
        "gate_w": bf16_to_f32(raw["gate.weight"]),
        "gate_b": raw["gate.bias"].float(),
        "shared_w1": dequant_fp8(raw["shared_experts.w1.weight"], raw["shared_experts.w1.scale"]),
        "shared_w2": dequant_fp8(raw["shared_experts.w2.weight"], raw["shared_experts.w2.scale"]),
        "shared_w3": dequant_fp8(raw["shared_experts.w3.weight"], raw["shared_experts.w3.scale"]),
        "raw": raw,
        "device": device,
    }


def expert_fp32(ffn: dict, eid: int, which: str) -> torch.Tensor:
    """Dequantize one routed expert matrix on demand."""
    raw, device = ffn["raw"], ffn["device"]
    return dequant_fp4(raw[f"experts.{eid}.{which}.weight"].to(device),
                       raw[f"experts.{eid}.{which}.scale"].to(device))


def gate_reference(x: torch.Tensor, ffn: dict, topk: int, route_scale: float):
    """Returns (indices [rows, topk], weights [rows, topk]).

    sqrt(softplus(x @ W.T)); select topk on score+bias; renormalize the
    *unbiased* scores over the selection; scale by route_scale.
    """
    logits = x @ ffn["gate_w"].T
    original = torch.sqrt(torch.nn.functional.softplus(logits))
    scored = original + ffn["gate_b"]
    idx = scored.topk(topk, dim=-1)[1]
    w = original.gather(1, idx)
    w = w / w.sum(dim=-1, keepdim=True) * route_scale
    return idx, w


def moe_reference(x: torch.Tensor, ffn: dict, topk: int, route_scale: float,
                  swiglu_limit: float, experts_start: int, experts_end: int):
    """Full fp32 MoE: routed experts this rank owns, plus the shared expert."""
    idx, w = gate_reference(x, ffn, topk, route_scale)
    out = torch.zeros_like(x)

    for t in range(x.shape[0]):
        for k in range(topk):
            eid = int(idx[t, k])
            if not (experts_start <= eid < experts_end):
                continue  # another rank contributes this route
            w1 = expert_fp32(ffn, eid, "w1")
            w2 = expert_fp32(ffn, eid, "w2")
            w3 = expert_fp32(ffn, eid, "w3")
            gate = x[t] @ w1.T
            up = x[t] @ w3.T
            if swiglu_limit > 0:
                up = up.clamp(-swiglu_limit, swiglu_limit)
                gate = gate.clamp(max=swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up * w[t, k]
            out[t] += hidden @ w2.T
            del w1, w2, w3

    # Shared expert: no swiglu clamp (constructed with the default limit=0).
    sgate = x @ ffn["shared_w1"].T
    sup = x @ ffn["shared_w3"].T
    out += (torch.nn.functional.silu(sgate) * sup) @ ffn["shared_w2"].T
    return out, idx, w


def run_cpp(cpp_bin: str, ckpt_dir: str, stage: int, x: torch.Tensor,
            tp_rank: int, tp_world: int) -> torch.Tensor:
    rows, dim = x.shape
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.bin")
        out_path = os.path.join(tmp, "out.bin")
        with open(in_path, "wb") as f:
            f.write(struct.pack("<4i", MAGIC, stage, rows, dim))
            f.write(x.cpu().numpy().astype("<f4").tobytes())
        proc = subprocess.run([cpp_bin, ckpt_dir, in_path, out_path,
                               str(tp_rank), str(tp_world)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"cpp moe parity binary failed ({proc.returncode})")
        return torch.from_numpy(np.fromfile(out_path, dtype="<f4")).view(rows, dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--tp-world", type=int, default=1)
    # The routed path quantizes activations to int8 before the expert GEMM, so
    # a few 1e-3 of relative error is expected; wrong routing is >1e-1.
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--cpp-bin", default=CPP_BIN)
    a = ap.parse_args()

    if not os.path.exists(a.cpp_bin):
        print(f"missing {a.cpp_bin}; build target test_dspark_moe_parity first")
        return 2

    with open(os.path.join(a.ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    dim = cfg["hidden_size"]
    topk = cfg["num_experts_per_tok"]
    n_experts = cfg["n_routed_experts"]
    route_scale = cfg["routed_scaling_factor"]
    swiglu_limit = cfg["swiglu_limit"]

    per_rank = n_experts // a.tp_world
    experts_start = a.tp_rank * per_rank
    experts_end = experts_start + per_rank

    device = "cuda"
    torch.manual_seed(a.seed)

    failures = 0
    for stage in a.stages:
        # Scale matches post-RMSNorm activations.
        x = torch.randn(a.rows, dim, dtype=torch.float32, device=device)

        cpp_out = run_cpp(a.cpp_bin, a.ckpt_dir, stage, x, a.tp_rank, a.tp_world).to(device)

        ffn = load_stage_ffn(a.ckpt_dir, stage, device, experts_start, experts_end)
        with torch.inference_mode():
            ref, idx, w = moe_reference(x, ffn, topk, route_scale, swiglu_limit,
                                        experts_start, experts_end)

        err = ((cpp_out - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            cpp_out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
        local = [[int(e) for e in row if experts_start <= int(e) < experts_end]
                 for row in idx]
        print(f"  stage={stage} rel_l2={err:.5f} cos={cos:.6f} "
              f"mean|ref|={ref.abs().mean():.5f} mean|cpp|={cpp_out.abs().mean():.5f}")
        print(f"    routed experts per token (this rank): {local}")

        if not (err < a.tol):
            print(f"    [FAIL] rel_l2 {err:.5f} >= tol {a.tol}")
            failures += 1
        else:
            print("    [ok]")

        del ffn
        torch.cuda.empty_cache()

    print("PASS" if failures == 0 else f"FAIL ({failures} stage(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
