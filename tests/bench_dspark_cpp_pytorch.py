#!/usr/bin/env python3
"""TP plain/DSpark parity and decode-TPS runner for C++ fixtures.

The measured DSpark path is always-k: it drafts the checkpoint's full block and
submits it to one multi-token target forward. Adaptive and pre-draft gating are
deliberately excluded so acceptance measures draft fidelity rather than policy.
"""

import argparse
from dataclasses import fields
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from src.models.deepseek_v4.dspark_loop import (
    DSparkLoop,
    attach_dspark,
    dspark_state_dict_filter,
)
from src.models.deepseek_v4.loader import load_model
from src.models.deepseek_v4.runtime import ModelArgs, Transformer


def read_fixtures(path: Path) -> list[tuple[str, list[int]]]:
    fixtures = []
    for raw in path.read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        name, ids = raw.split("\t", 1)
        fixtures.append((name, [int(token) for token in ids.split(",") if token]))
    if not fixtures:
        raise ValueError(f"fixture file is empty: {path}")
    return fixtures


def sync() -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def set_runtime_phase(phase: str) -> None:
    """Match generation.py's prefill/decode phase contract.

    Active-expert GPU staging is deliberately decode-only. Calling
    Transformer.forward() directly without publishing this phase silently falls
    back to the CPU INT8 MoE path even when DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1.
    """
    if phase not in {"prefill", "decode"}:
        raise ValueError(f"invalid runtime phase: {phase}")
    os.environ["DEEPSEEK_PD_ACTIVE_PHASE"] = phase


def active_gpu_decode_layers(model: Transformer) -> int:
    """Fail loudly if the requested fast decode backend is not selectable."""
    requested = os.getenv("DEEPSEEK_GPU_MOE_DECODE_ACTIVE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if not requested:
        return 0
    set_runtime_phase("decode")
    layers = [
        module
        for module in model.modules()
        if getattr(module, "gpu_decode_active_moe_enabled", False)
    ]
    if not layers or any(module._pd_active_phase() != "decode" for module in layers):
        raise RuntimeError(
            "DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1 was requested, but the decode "
            "phase is not selecting active-expert GPU MoE"
        )
    return len(layers)


def reset_model_state(model: Transformer) -> None:
    """Restore all request-local main, compressor, indexer, and draft caches."""
    for module in model.modules():
        for name in ("kv_cache", "kv_state"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()
        score_state = getattr(module, "score_state", None)
        if isinstance(score_state, torch.Tensor):
            score_state.fill_(float("-inf"))


def hidden_summary(hidden: torch.Tensor) -> dict:
    value = hidden.detach().float()
    finite = torch.isfinite(value)
    return {
        "values": value.numel(),
        "finite": int(finite.sum().item()),
        "l2": float(torch.linalg.vector_norm(value).item()),
        "mean": float(value.mean().item()),
        "max_abs": float(value.abs().amax().item()),
    }


@torch.inference_mode()
def run_plain(
    model: Transformer, prompt: list[int], decode_tokens: int
) -> tuple[list[int], float]:
    reset_model_state(model)
    input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    set_runtime_phase("prefill")
    committed = model.forward(input_ids, 0, return_next_token=True)
    position = len(prompt)

    set_runtime_phase("decode")
    sync()
    start = time.perf_counter()
    output = []
    for _ in range(decode_tokens):
        committed = model.forward(
            committed.reshape(1, 1), position, return_next_token=True
        )
        output.append(int(committed.item()))
        position += 1
    sync()
    return output, time.perf_counter() - start


@torch.inference_mode()
def run_spec_always_k(
    model: Transformer,
    model_args: ModelArgs,
    prompt: list[int],
    decode_tokens: int,
    *,
    case: str,
    repeat: int,
    emit_trace: bool,
    rank: int,
) -> tuple[list[int], float, dict]:
    """Run DSpark with full k and C++ scheduler-compatible output semantics.

    DSparkLoop.step() returns the previously committed token plus accepted drafts.
    The C++ scheduler instead returns accepted drafts plus the target bonus token,
    so this runner drives the same private primitives explicitly.
    """
    reset_model_state(model)
    loop = DSparkLoop(model, model_args)
    input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    set_runtime_phase("prefill")
    loop.prefill(input_ids)

    set_runtime_phase("decode")
    sync()
    start = time.perf_counter()
    output: list[int] = []
    while len(output) < decode_tokens:
        round_index = loop.rounds
        position = loop._pos
        committed = int(loop._committed.item())
        seed_summary = (
            hidden_summary(loop._hidden) if emit_trace and rank == 0 else None
        )

        if emit_trace:
            sync()
            draft_start = time.perf_counter()
        draft_ids, confidence = loop._draft(
            loop._hidden, loop._committed[:, 0], loop._pos - 1
        )
        if emit_trace:
            sync()
            draft_seconds = time.perf_counter() - draft_start
        else:
            draft_seconds = None

        verify_in = torch.cat([loop._committed, draft_ids], dim=1)
        if emit_trace:
            sync()
            verify_start = time.perf_counter()
        v_greedy, v_hidden, v_logits = loop._main_forward(
            verify_in, loop._pos, want_hidden=True
        )
        if emit_trace:
            sync()
            verify_seconds = time.perf_counter() - verify_start
        else:
            verify_seconds = None

        n_drafted = draft_ids.size(1)
        n_ok = 0
        for i in range(n_drafted):
            if int(draft_ids[0, i]) != int(v_greedy[0, i]):
                break
            n_ok += 1

        generated = [int(token) for token in draft_ids[0, :n_ok].tolist()]
        generated.append(int(v_greedy[0, n_ok]))
        remaining = decode_tokens - len(output)
        output.extend(generated[:remaining])

        loop._write_draft_kv(v_hidden[:, : n_ok + 1], loop._pos)
        loop._pos += n_ok + 1
        loop._committed = v_greedy[:, n_ok : n_ok + 1]
        loop._hidden = v_hidden[:, n_ok : n_ok + 1]
        loop._logits = v_logits[:, n_ok : n_ok + 1]
        loop.rounds += 1
        loop.accepted_tokens += n_ok
        loop.drafted_tokens += n_drafted

        if emit_trace and rank == 0:
            trace = {
                "runtime": "pytorch",
                "path": "spec_always_k5",
                "case": case,
                "repeat": repeat,
                "round": round_index,
                "position": position,
                "committed_token": committed,
                "draft_tokens": [int(token) for token in draft_ids[0].tolist()],
                "confidence": [float(value) for value in confidence[0].tolist()],
                "target_successors": [int(token) for token in v_greedy[0].tolist()],
                "accepted_tokens": n_ok,
                "generated_tokens": generated,
                "seed_hidden": seed_summary,
                "draft_seconds": draft_seconds,
                "verify_seconds": verify_seconds,
            }
            print("TRACE_JSON " + json.dumps(trace, separators=(",", ":")), flush=True)

    sync()
    seconds = time.perf_counter() - start
    stats = {
        "rounds": loop.rounds,
        "accepted_tokens": loop.accepted_tokens,
        "drafted_tokens": loop.drafted_tokens,
        "accepted_per_round": loop.accepted_tokens / loop.rounds,
        "accept_rate": loop.accepted_tokens / loop.drafted_tokens,
    }
    return output, seconds, stats


def checkpoint_dspark_stages(ckpt: Path) -> int:
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    stages = {
        int(key.split(".")[1])
        for key in index["weight_map"]
        if key.startswith("mtp.")
    }
    if not stages or stages != set(range(max(stages) + 1)):
        raise ValueError(f"invalid DSpark stages in checkpoint: {sorted(stages)}")
    return len(stages)


def load_model_args(
    config_path: Path,
    ckpt: Path,
    fixtures: list[tuple[str, list[int]]],
    decode_tokens: int,
    routed_experts_device: str,
) -> ModelArgs:
    config = json.loads(config_path.read_text())
    checkpoint_config = json.loads((ckpt / "config.json").read_text())
    for key in (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ):
        config[key] = checkpoint_config[key]
    config["n_dspark_stages"] = checkpoint_dspark_stages(ckpt)
    config["max_batch_size"] = 1
    # Match the validated TP=4 PyTorch DSpark runtime. The config file omits
    # this field, whose ModelArgs default is `legacy`; the historical 303 ms
    # plain-decode baseline used baseline_4gpu explicitly.
    config["partition_policy"] = "baseline_4gpu"
    config["max_seq_len"] = max(
        2048, max(len(prompt) for _, prompt in fixtures) + decode_tokens + 16
    )
    config["n_mtp_layers"] = 0
    config["routed_experts_device"] = routed_experts_device
    accepted = {field.name for field in fields(ModelArgs)}
    unknown = sorted(set(config) - accepted)
    if unknown:
        raise ValueError(f"unsupported ModelArgs fields in {config_path}: {unknown}")
    return ModelArgs(**config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--routed-experts-device", choices=("cpu", "gpu"), default="cpu"
    )
    parser.add_argument("--trace", action="store_true")
    args_cli = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Transformer records this switch while its MoE layers are constructed.
    # Without it, DEEPSEEK_PD_ACTIVE_PHASE is ignored and decode silently uses
    # CPU INT8 experts instead of the requested active-expert GPU path.
    os.environ.setdefault("DEEPSEEK_PD_PHASE_AUTO_SELECT", "1")
    set_runtime_phase("prefill")
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(days=1))
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(12345)

    fixtures = read_fixtures(args_cli.fixtures)
    model_args = load_model_args(
        args_cli.config,
        args_cli.ckpt,
        fixtures,
        args_cli.decode_tokens,
        args_cli.routed_experts_device,
    )
    with torch.device("cuda"):
        model = Transformer(model_args)
        attach_dspark(model, model_args)
    restore_state_dict = dspark_state_dict_filter(model, model_args)
    try:
        load_model(model, str(args_cli.ckpt), world_size, rank, "safetensors")
    finally:
        restore_state_dict()
    if args_cli.routed_experts_device == "cpu":
        model.prepare_cpu_expert_int8()
    gpu_decode_layers = active_gpu_decode_layers(model)
    set_runtime_phase("prefill")
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        config_record = {
            "runtime": "pytorch",
            "world_size": world_size,
            "routed_experts_device": model_args.routed_experts_device,
            "partition_policy": model_args.partition_policy,
            "compress_ratios": list(model_args.compress_ratios[: model_args.n_layers]),
            "n_dspark_stages": model_args.n_dspark_stages,
            "dspark_block_size": model_args.dspark_block_size,
            "gate": "disabled",
            "pd_phase_auto_select": os.getenv("DEEPSEEK_PD_PHASE_AUTO_SELECT"),
            "gpu_decode_active_requested": os.getenv(
                "DEEPSEEK_GPU_MOE_DECODE_ACTIVE", "0"
            ),
            "gpu_decode_active_layers": gpu_decode_layers,
        }
        print("CONFIG_JSON " + json.dumps(config_record, separators=(",", ":")), flush=True)

    run_plain(model, fixtures[0][1], 1)
    run_spec_always_k(
        model,
        model_args,
        fixtures[0][1],
        1,
        case=fixtures[0][0],
        repeat=-1,
        emit_trace=False,
        rank=rank,
    )

    for name, prompt in fixtures:
        for repeat in range(args_cli.repeats):
            tokens, seconds = run_plain(model, prompt, args_cli.decode_tokens)
            record = {
                "runtime": "pytorch",
                "path": "plain",
                "case": name,
                "repeat": repeat,
                "prompt_tokens": len(prompt),
                "decode_tokens": args_cli.decode_tokens,
                "seconds": seconds,
                "tps": args_cli.decode_tokens / seconds,
                "tokens": tokens,
            }
            if rank == 0:
                print("RESULT_JSON " + json.dumps(record, separators=(",", ":")), flush=True)

            tokens, seconds, stats = run_spec_always_k(
                model,
                model_args,
                prompt,
                args_cli.decode_tokens,
                case=name,
                repeat=repeat,
                emit_trace=args_cli.trace,
                rank=rank,
            )
            record = {
                "runtime": "pytorch",
                "path": "spec_always_k5",
                "case": name,
                "repeat": repeat,
                "prompt_tokens": len(prompt),
                "decode_tokens": args_cli.decode_tokens,
                "seconds": seconds,
                "tps": args_cli.decode_tokens / seconds,
                "tokens": tokens,
                **stats,
            }
            if rank == 0:
                print("RESULT_JSON " + json.dumps(record, separators=(",", ":")), flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
