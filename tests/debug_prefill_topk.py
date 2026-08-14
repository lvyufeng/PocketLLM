#!/usr/bin/env python3
"""Print the top-k prefill logits per fixture so a first-token mismatch can be
classified as a near-tie (numerical drift) or a real divergence.

Same model construction as tests/bench_dspark_cpp_pytorch.py so the logits are
the ones that benchmark compares against.
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from src.models.deepseek_v4.dspark_loop import attach_dspark, dspark_state_dict_filter
from src.models.deepseek_v4.loader import load_model
from src.models.deepseek_v4.runtime import Transformer

from tests.bench_dspark_cpp_pytorch import (
    load_model_args,
    read_fixtures,
    reset_model_state,
    set_runtime_phase,
)


@torch.inference_mode()
def prefill_topk(model, prompt, topk):
    reset_model_state(model)
    input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    set_runtime_phase("prefill")
    logits = model.forward(input_ids, 0, return_next_token=False)
    if logits.dim() == 3:
        logits = logits[:, -1]
    logits = logits.float().reshape(-1)
    values, indices = torch.topk(logits, topk)
    return [int(i) for i in indices.tolist()], [float(v) for v in values.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--routed-experts-device", choices=("cpu", "gpu"), default="cpu"
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(days=1))
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(12345)

    fixtures = read_fixtures(args.fixtures)
    model_args = load_model_args(
        args.config, args.ckpt, fixtures, 1, args.routed_experts_device
    )
    with torch.device("cuda"):
        model = Transformer(model_args)
        attach_dspark(model, model_args)
    restore_state_dict = dspark_state_dict_filter(model, model_args)
    try:
        load_model(model, str(args.ckpt), world_size, rank, "safetensors")
    finally:
        restore_state_dict()
    if args.routed_experts_device == "cpu":
        model.prepare_cpu_expert_int8()
    if dist.is_initialized():
        dist.barrier()

    for name, prompt in fixtures:
        tokens, values = prefill_topk(model, prompt, args.topk)
        if rank == 0:
            record = {
                "case": name,
                "prompt_tokens": len(prompt),
                "top_tokens": tokens,
                "top_logits": values,
                "margin": values[0] - values[1] if len(values) > 1 else None,
            }
            print("TOPK_JSON " + json.dumps(record, separators=(",", ":")), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
