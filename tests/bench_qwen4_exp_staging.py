"""Where does Qwen4-Exp expert staging time actually go?

Four probes against the real Qwen3.8-Flash-Next checkpoint, each answering one
question that came up while profiling the heterogeneous TP4 path:

  sizes    - how many distinct experts does a chunk of N tokens touch, and how
             many bytes does that cost per layer / per 48-layer pass?
  source   - is staging bound by the checkpoint disk or by PCIe?  Compares a
             cold first touch, a warm page-cache touch, and a pure RAM->GPU copy.
  batched  - does gathering a layer's active experts into one contiguous host
             buffer and issuing a single H2D beat the per-expert loop?
  shard    - does `ShardedMoE` make each rank stage experts it never computes
             with (i.e. is there duplicated H2D across ranks)?

Single process, no TP: the staging path is per-rank anyway, so these numbers are
per-rank.  The aggregate PCIe figure only shows up in a real 4-rank run.

Run: python tests/bench_qwen4_exp_staging.py [--probe sizes,source,batched,shard]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.qwen4_exp.config import Qwen4ExpConfig
from src.models.qwen4_exp.layers import swiglu_expert
from src.models.qwen4_exp.moe import ShardedMoE
from src.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint

MODEL = os.environ.get("QWEN4EXP_MODEL", "/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next")
MIB = float(2**20)
GIB = float(2**30)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stage_loop(ckpt, layer, experts, device, dtype) -> tuple[float, int]:
    """Per-expert staging, the shape `HostExpertMoE._fetch` uses."""
    _sync(device)
    t0 = time.perf_counter()
    nbytes = 0
    for expert_id in experts:
        gate_up, down = ckpt.expert_rows(layer, expert_id)
        gate_up.to(device, dtype=dtype, non_blocking=True)
        down.to(device, dtype=dtype, non_blocking=True)
        nbytes += (gate_up.numel() + down.numel()) * 2
    _sync(device)
    return time.perf_counter() - t0, nbytes


def probe_sizes(ckpt, config, device, dtype) -> None:
    """Active expert count and byte cost per chunk size."""
    gate_up, down = ckpt.expert_rows(0, 0)
    per_expert = (gate_up.numel() + down.numel()) * 2
    layers = config.num_hidden_layers

    print(f"num_experts={config.num_experts} top_k={config.num_experts_per_tok} "
          f"hidden={config.hidden_size} moe_intermediate={config.moe_intermediate_size}")
    print(f"per-expert bf16 = {per_expert / MIB:.2f} MiB "
          f"(gate_up{tuple(gate_up.shape)} down{tuple(down.shape)})")
    print(f"full expert set = {config.num_experts * layers * per_expert / GIB:.1f} GiB\n")

    print(f"{'tokens':>7}  {'unique':>6}  {'MiB/layer':>10}  {'GiB/pass':>9}")
    for n_tok in (1, 8, 60, 512, 4096):
        generator = torch.Generator().manual_seed(0)
        # Uniform routing is the worst case; real routing is more concentrated.
        indices = torch.stack([
            torch.randperm(config.num_experts, generator=generator)[: config.num_experts_per_tok]
            for _ in range(n_tok)
        ])
        unique = int(torch.unique(indices).numel())
        print(f"{n_tok:>7}  {unique:>6}  {unique * per_expert / MIB:>10.1f}  "
              f"{layers * unique * per_expert / GIB:>9.2f}")
    print("\nnote: capped at num_experts, so past a few hundred tokens a larger chunk "
          "costs the same H2D but amortises it over more tokens.")


def probe_source(ckpt, config, device, dtype, *, layer: int = 17, count: int = 96) -> None:
    """Disk-bound or PCIe-bound?  cold vs warm vs pure RAM->GPU."""
    experts = list(range(count))
    cold_s, nbytes = _stage_loop(ckpt, layer, experts, device, dtype)
    warm_s, _ = _stage_loop(ckpt, layer, experts, device, dtype)

    gib = nbytes / GIB
    print(f"layer={layer} experts={count} payload={gib:.3f} GiB")
    print(f"cold (disk + H2D): {cold_s * 1e3:7.1f} ms  {gib / cold_s:5.2f} GiB/s  "
          f"{cold_s / count * 1e3:5.2f} ms/expert")
    print(f"warm (cache+ H2D): {warm_s * 1e3:7.1f} ms  {gib / warm_s:5.2f} GiB/s  "
          f"{warm_s / count * 1e3:5.2f} ms/expert")

    # Pure PCIe: host tensors already materialised outside the mapping.
    resident = [
        (ckpt.expert_rows(layer, e)[0].clone(), ckpt.expert_rows(layer, e)[1].clone())
        for e in experts
    ]
    _sync(device)
    t0 = time.perf_counter()
    for gate_up, down in resident:
        gate_up.to(device, dtype=dtype, non_blocking=True)
        down.to(device, dtype=dtype, non_blocking=True)
    _sync(device)
    h2d_s = time.perf_counter() - t0
    print(f"h2d  (RAM  -> GPU): {h2d_s * 1e3:7.1f} ms  {gib / h2d_s:5.2f} GiB/s  "
          f"{h2d_s / count * 1e3:5.2f} ms/expert")

    print(f"\ncold/warm = {cold_s / warm_s:.2f}x   warm/h2d = {warm_s / h2d_s:.2f}x")
    print("=> disk-bound" if cold_s > 2 * warm_s else "=> not disk-bound; PCIe is the floor")


def probe_batched(ckpt, config, device, dtype, *, layer: int = 10, count: int = 108) -> None:
    """One contiguous H2D vs the per-expert loop (count ~ a 512-token chunk)."""
    experts = list(range(count))
    ids = torch.tensor(experts, dtype=torch.long)
    gate_up_all = ckpt.store.view(ckpt.expert_key(layer, "gate_up_proj"))
    down_all = ckpt.store.view(ckpt.expert_key(layer, "down_proj"))

    def batched() -> tuple[float, float]:
        _sync(device)
        t0 = time.perf_counter()
        gate_up = torch.index_select(gate_up_all, 0, ids)
        down = torch.index_select(down_all, 0, ids)
        gathered = time.perf_counter()
        gate_up.to(device, dtype=dtype, non_blocking=True)
        down.to(device, dtype=dtype, non_blocking=True)
        _sync(device)
        return time.perf_counter() - t0, gathered - t0

    _stage_loop(ckpt, layer, experts, device, dtype)  # warm pages + allocator
    batched()
    loop = [_stage_loop(ckpt, layer, experts, device, dtype)[0] for _ in range(3)]
    batch = [batched() for _ in range(3)]

    gate_up, down = ckpt.expert_rows(layer, 0)
    gib = count * (gate_up.numel() + down.numel()) * 2 / GIB
    print(f"layer={layer} experts={count} payload={gib:.3f} GiB")
    print("loop:    " + "  ".join(f"{x * 1e3:6.1f} ms" for x in loop))
    print("batched: " + "  ".join(f"{x * 1e3:6.1f} ms" for x, _ in batch))
    print("  gather: " + "  ".join(f"{g * 1e3:6.1f} ms" for _, g in batch))
    speedup = min(loop) / min(x for x, _ in batch)
    print(f"\nbatched speedup = {speedup:.2f}x "
          f"({'win' if speedup > 1.05 else 'no win: host gather eats the savings'})")


class _FetchSpy:
    """Stands in for HostExpertMoE, recording which experts get fetched."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self.fetched: set[int] = set()

    def __call__(self, layer_idx, hidden_states, indices, weights):
        for expert_id in torch.unique(indices.reshape(-1)).tolist():
            if expert_id < self.num_experts:
                self.fetched.add(int(expert_id))
        return torch.zeros_like(hidden_states)


def probe_shard(ckpt, config, device, dtype, *, world_size: int = 4) -> None:
    """Does any rank stage an expert it never computes with?"""
    gate_up, down = ckpt.expert_rows(0, 0)
    per_expert = (gate_up.numel() + down.numel()) * 2
    num_experts, top_k = config.num_experts, config.num_experts_per_tok

    for n_tok in (1, 60, 512):
        generator = torch.Generator().manual_seed(1234)
        # Bias the scores so routing is concentrated, as it is in practice.
        logits = torch.randn(n_tok, num_experts, generator=generator)
        logits += torch.linspace(3.0, 0.0, num_experts).unsqueeze(0)
        indices = logits.topk(top_k, dim=-1).indices
        weights = torch.ones(n_tok, top_k)
        hidden = torch.zeros(n_tok, 8)

        unique = int(torch.unique(indices).numel())
        per_rank = []
        for rank in range(world_size):
            spy = _FetchSpy(num_experts)
            ShardedMoE(spy, rank, world_size)(0, hidden, indices, weights)
            per_rank.append(len(spy.fetched))

        print(f"tokens={n_tok:>4}  unique/layer={unique:>3}  per-rank={per_rank}  "
              f"sum={sum(per_rank)}  duplication={sum(per_rank) / max(1, unique):.2f}x")
        print(f"           rank0 {per_rank[0] * per_expert / MIB:7.1f} MiB/layer, "
              f"{config.num_hidden_layers * per_rank[0] * per_expert / GIB:5.2f} GiB/pass")


PROBES = {
    "sizes": probe_sizes,
    "source": probe_source,
    "batched": probe_batched,
    "shard": probe_shard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--probe", default="sizes,source,batched,shard",
                        help="comma-separated: " + ",".join(PROBES))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.model, "model.safetensors.index.json")):
        print(f"checkpoint not found at {args.model}; set --model or QWEN4EXP_MODEL")
        return

    config = Qwen4ExpConfig.from_pretrained(args.model).text_config
    ckpt = Qwen4ExpCheckpoint(args.model, store=MmapSafetensors(args.model))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    for name in [p.strip() for p in args.probe.split(",") if p.strip()]:
        if name not in PROBES:
            print(f"unknown probe {name!r}; choose from {','.join(PROBES)}")
            continue
        print(f"\n=== {name} ===")
        PROBES[name](ckpt, config, device, dtype)


if __name__ == "__main__":
    main()
