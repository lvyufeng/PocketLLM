"""Model-agnostic GGUF token-runtime helpers.

This module intentionally contains no model-specific checkpoint loading or
modeling imports. Per-architecture assembly is reached polymorphically through
``MoEModelSpec.build_token_runtime`` (resolved from ``general.architecture`` by
``src.components.moe.registry.detect_spec``), so the greedy decode / seed-file
driver below works for any registered raw-block CUDA architecture.

DeepSeek-V4's full text-generation orchestration (tokenizer, sampling,
interactive, PD scheduler) still lives in ``src.models.deepseek_v4.generation``;
this driver is the shared GGUF raw-block token-id smoke/perf entrypoint.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist

from src.loader.gguf.bundle import read_gguf_bundle


def setup_dist() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not torch.cuda.is_available():
        raise RuntimeError("GGUF raw-block runtime requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        # A long timeout so a slow one-time step on a single rank (e.g. reading
        # a multi-hundred-GB GGUF into the page cache while the other ranks wait
        # at the first barrier) does not trip the default 10-minute store
        # timeout and tear down the NCCL communicator.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return world, rank, local_rank, torch.device("cuda", local_rank)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def read_seed_tokens(path: str | Path) -> list[int]:
    data = Path(path).read_bytes()
    if len(data) == 0:
        raise ValueError(f"empty seed-file: {path}")
    if len(data) % 8 == 0:
        arr = np.frombuffer(data, dtype="<i8")
        if arr.size and int(arr.min()) >= 0 and int(arr.max()) < 2**31:
            return [int(x) for x in arr]
    if len(data) % 4 == 0:
        arr = np.frombuffer(data, dtype="<i4")
        return [int(x) for x in arr]
    raise ValueError(f"seed-file byte length must be divisible by 4 or 8, got {len(data)}")


def parse_tokens_csv(text: str) -> list[int]:
    vals = [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]
    if not vals:
        raise ValueError("--tokens did not contain any token ids")
    return [int(v) for v in vals]


@torch.inference_mode()
def greedy_generate_token_ids(
    model,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], dict[str, float]]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = model.device
    total_len = len(prompt_tokens) + max_new_tokens
    model.reset_cache(batch_size=1, max_seq_len=max(1, total_len))

    generated: list[int] = []
    prefill_time = 0.0
    decode_time = 0.0
    if max_new_tokens == 0:
        return generated, {"prefill_seconds": 0.0, "decode_seconds": 0.0, "prefill_tokens": 0.0, "decode_tokens": 0.0}

    prompt = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    sync()
    t0 = time.perf_counter()
    next_token = model.forward(prompt, 0, return_next_token=True)
    sync()
    prefill_time = time.perf_counter() - t0
    token = int(next_token.reshape(-1)[0].item())
    generated.append(token)

    for step in range(1, max_new_tokens):
        if eos_token_id is not None and generated[-1] == int(eos_token_id):
            break
        pos = len(prompt_tokens) + step - 1
        inp = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
        sync()
        t0 = time.perf_counter()
        next_token = model.forward(inp, pos, return_next_token=True)
        sync()
        decode_time += time.perf_counter() - t0
        generated.append(int(next_token.reshape(-1)[0].item()))

    stats = {
        "prefill_seconds": float(prefill_time),
        "decode_seconds": float(decode_time),
        "prefill_tokens": float(len(prompt_tokens)),
        "decode_tokens": float(max(0, len(generated) - 1)),
    }
    return generated, stats


@dataclass(frozen=True)
class GGUFTokenRuntime:
    """A loaded, ready-to-decode GGUF model plus the metadata the driver prints.

    ``model`` only needs the model-agnostic decode contract used by
    :func:`greedy_generate_token_ids`: ``device``, ``reset_cache(...)`` and
    ``forward(tokens, start_pos, return_next_token=...)``.
    """

    model: Any
    expert_start: int
    expert_count: int
    eos_token_id: int | None
    load_seconds: float


class GGUFTokenRuntimeSpec(Protocol):
    """Runtime-construction hook implemented by architecture specs.

    Specs that support raw-block CUDA token generation implement this so the
    generic driver can build their model without importing model-specific
    modules. Keeps TP expert-range planning and loader wiring inside each
    model package.
    """

    architecture: str
    display_name: str

    def build_token_runtime(
        self,
        bundle,
        *,
        world: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        n_layers: int | None,
        gpu_memory_gib: float,
    ) -> GGUFTokenRuntime:
        ...


def run_gguf_generation(
    gguf_path: str | Path,
    *,
    prompt_tokens: list[int],
    max_new_tokens: int,
    architecture: str = "auto",
    dtype: torch.dtype = torch.float16,
    n_layers: int | None = None,
    gpu_memory_gib: float = 22.0,
    prewarm: bool = False,
) -> tuple[list[int], dict[str, float]] | None:
    """Dispatch a GGUF checkpoint to its architecture spec and greedily decode.

    Returns ``(generated_token_ids, stats)`` on rank 0, ``None`` on other
    ranks. The architecture is resolved from ``general.architecture`` unless
    overridden, so the same entrypoint serves every registered model.
    """
    from src.components.moe.registry import detect_spec

    world, rank, local_rank, device = setup_dist()
    bundle = read_gguf_bundle(gguf_path)
    spec = detect_spec(bundle, architecture)
    if not hasattr(spec, "build_token_runtime"):
        raise ValueError(
            f"architecture {spec.architecture!r} does not support GGUF raw-block token generation"
        )

    model_context_tokens = 0
    if hasattr(spec, "parse_params"):
        params = spec.parse_params(bundle)
        model_context_tokens = int(getattr(params, "context_length", 0) or 0)
    request_tokens = len(prompt_tokens) + int(max_new_tokens)
    if model_context_tokens > 0 and request_tokens > model_context_tokens:
        raise ValueError(
            f"request_tokens={request_tokens} exceeds model_context_tokens={model_context_tokens} "
            f"for architecture={spec.architecture!r}"
        )

    if rank == 0:
        print(
            f"gguf_load_start architecture={spec.architecture} world={world} "
            f"layers={n_layers or 'full'} prompt_tokens={len(prompt_tokens)} "
            f"max_new_tokens={max_new_tokens} request_tokens={request_tokens} "
            f"model_context_tokens={model_context_tokens or 'unknown'} "
            f"dtype={str(dtype).removeprefix('torch.')}",
            flush=True,
        )

    if prewarm:
        # The page cache is shared system-wide, so warming on rank 0 benefits
        # every rank. Other ranks wait at the barrier until warming completes.
        # A single sequential reader is fastest on a spinning disk. Warming
        # 236 GB at HDD bandwidth takes many minutes, so setup_dist raises the
        # process-group timeout to keep the idle ranks from tripping it here.
        if rank == 0:
            from src.loader.gguf.prewarm import prewarm_bundle

            t0 = time.perf_counter()
            elapsed = prewarm_bundle(bundle)
            total_bytes = sum(os.path.getsize(p) for p in bundle.paths)
            print(
                f"gguf_prewarm_done shards={len(elapsed)} "
                f"total_bytes={total_bytes / 1024**3:.2f}GB "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        sync()

    runtime = spec.build_token_runtime(
        bundle,
        world=world,
        rank=rank,
        device=device,
        dtype=dtype,
        n_layers=n_layers,
        gpu_memory_gib=gpu_memory_gib,
    )
    sync()
    alloc_gib = torch.cuda.memory_allocated(device) / 1024**3
    reserved_gib = torch.cuda.memory_reserved(device) / 1024**3
    print(
        f"rank={rank} local_rank={local_rank} "
        f"expert_range=[{runtime.expert_start},{runtime.expert_start + runtime.expert_count}) "
        f"load_seconds={runtime.load_seconds:.3f} cuda_alloc_gib={alloc_gib:.3f} "
        f"cuda_reserved_gib={reserved_gib:.3f}",
        flush=True,
    )

    generated, stats = greedy_generate_token_ids(
        runtime.model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_id=runtime.eos_token_id,
    )
    result: tuple[list[int], dict[str, float]] | None = None
    if rank == 0:
        prefill_tps = stats["prefill_tokens"] / max(stats["prefill_seconds"], 1.0e-9)
        decode_tps = stats["decode_tokens"] / max(stats["decode_seconds"], 1.0e-9) if stats["decode_tokens"] > 0 else 0.0
        print(
            f"RESULT prefill_seconds={stats['prefill_seconds']:.6f} prefill_tokens={int(stats['prefill_tokens'])} "
            f"prefill_tps={prefill_tps:.2f} decode_seconds={stats['decode_seconds']:.6f} "
            f"decode_tokens={int(stats['decode_tokens'])} decode_tps={decode_tps:.2f}",
            flush=True,
        )
        print("generated_token_ids=" + ",".join(str(t) for t in generated), flush=True)
        result = (generated, stats)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return result
