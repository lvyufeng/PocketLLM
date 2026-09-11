#!/usr/bin/env python3
"""vLLM TP4 long-context prefill/decode benchmark for the PocketLLM comparison.

Runs one prompt length per process so initialization, JIT, and CUDA-graph
capture never land inside a measured request. Prompt token IDs come from a
shared fixture file so vLLM and PocketLLM see the exact same prompt.

Timing model, matched to the PocketLLM side:
  * prefill/TTFT -- wall time from request submission to the first output token.
  * decode       -- wall time from the first to the last output token.
  * decode TPS   -- (generated_tokens - 1) / decode_seconds.

This must run in an environment that has vLLM (the project's `qwen35` env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gpu_memory_sampler import GpuMemorySampler, read_used_bytes  # noqa: E402

DEFAULT_CKPT = "/mnt/data2/Qwen3.8-27B-FP8"
DEFAULT_LENGTHS = "8192,16384,32768,65536,131072,262016"
MODEL_MAX_CONTEXT = 262144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--fixture-dir", type=Path,
                        default=Path(".scratch/qwen_full_comparison/fixtures"))
    parser.add_argument("--work-dir", type=Path,
                        default=Path(".scratch/qwen_full_comparison/vllm"))
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    # The benchmark is single-request. vLLM's sampler warmup allocates for
    # max_num_seqs dummy requests, and the default 256 OOMs on a 2080 Ti.
    parser.add_argument("--max-num-seqs", type=int, default=1)
    # vLLM's "auto" picks flashqla_legacy on SM70/SM75, which imports the
    # separate `flash_qla` package. That package is not installed in this
    # environment, so auto raises ModuleNotFoundError inside the GDN prefill
    # call and the whole run fails. Triton/FLA is the working SM75 path.
    parser.add_argument("--gdn-prefill-backend",
                        choices=["auto", "triton", "flashqla_legacy", "flashinfer"],
                        default="triton",
                        help="GDN prefill kernel (default: triton, the working SM75 path)")
    parser.add_argument("--mtp-tokens", type=int, default=0,
                        help="0 runs plain; >0 enables qwen3_5_mtp with that K")
    parser.add_argument("--result", type=Path, default=None)
    # Internal: the worker mode for a single length inside a fresh process.
    parser.add_argument("--worker-spec", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_fixture(path: Path, length: int) -> list[int]:
    tokens = [int(item) for item in path.read_text(encoding="ascii").split(",") if item]
    if len(tokens) != length:
        raise RuntimeError(f"fixture {path} has {len(tokens)} tokens, expected {length}")
    return tokens


def fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_memory_bytes(devices: list[int]) -> dict[int, int]:
    return read_used_bytes(devices)


def run_worker(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    from vllm import LLM, SamplingParams

    prompt_tokens = spec["prompt_token_ids"]
    max_new = spec["max_new_tokens"]
    prompt_len = len(prompt_tokens)
    max_len = min(prompt_len + max_new, MODEL_MAX_CONTEXT)

    kwargs: dict[str, Any] = dict(
        model=spec["checkpoint"],
        tensor_parallel_size=spec["tp_world"],
        trust_remote_code=True,
        max_model_len=max_len,
        max_num_batched_tokens=spec["max_num_batched_tokens"],
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        max_num_seqs=spec["max_num_seqs"],
        disable_custom_all_reduce=spec["disable_custom_all_reduce"],
        # Async scheduling drives the batch-queue path, whose sample_tokens RPC
        # times out with a single in-flight sequence on this build.
        async_scheduling=False,
        additional_config={"gdn_prefill_backend": spec["gdn_prefill_backend"]},
    )
    if spec["mtp_tokens"] > 0:
        kwargs["speculative_config"] = {
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": spec["mtp_tokens"],
        }

    load_start = time.perf_counter()
    llm = LLM(**kwargs)
    load_seconds = time.perf_counter() - load_start

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new)
    warm_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4)

    request = [{"prompt_token_ids": prompt_tokens}]
    # Warmup exercises the same shapes so graph capture and Triton JIT happen
    # outside the measured request.
    llm.generate(request, warm_sampling)

    # RequestOutput.metrics is empty on this vLLM V1 build, and timing prefill
    # as a separate max_tokens=1 request is unusable at 128K (prefill noise
    # exceeds the whole decode window). Drive the engine step loop instead and
    # timestamp the first emitted token inside the one measured request.
    from vllm.sampling_params import RequestOutputKind

    step_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new)
    step_params.output_kind = RequestOutputKind.DELTA
    engine = llm.llm_engine
    engine.add_request("bench", {"prompt_token_ids": prompt_tokens}, step_params)

    start = time.perf_counter()
    first_token_ts = None
    generated: list[int] = []
    final_out = None
    while engine.has_unfinished_requests():
        for out in engine.step():
            new_ids = list(out.outputs[0].token_ids)
            if new_ids and first_token_ts is None:
                first_token_ts = time.perf_counter()
            generated.extend(new_ids)
            if out.finished:
                final_out = out
    end_ts = time.perf_counter()

    wall = end_ts - start
    prefill_wall = (first_token_ts - start) if first_token_ts else None
    decode_seconds = (end_ts - first_token_ts) if first_token_ts else None
    decode_tokens = max(len(generated) - 1, 0)

    out = final_out
    metric_fields: dict[str, Any] = {}
    metrics = getattr(out, "metrics", None) if out is not None else None
    if metrics is not None:
        for name in dir(metrics):
            if name.startswith("_"):
                continue
            value = getattr(metrics, name)
            if isinstance(value, (int, float, type(None))):
                metric_fields[name] = value

    result = {
        "engine": "vllm",
        "mode": "mtp" if spec["mtp_tokens"] > 0 else "plain",
        "mtp_k": spec["mtp_tokens"],
        "prompt_tokens": prompt_len,
        "max_model_len": max_len,
        "generated_tokens": len(generated),
        "tokens": generated,
        "e2e_wall_seconds": wall,
        "load_seconds": load_seconds,
        "ttft_seconds": prefill_wall,
        "prefill_seconds": prefill_wall,
        "prefill_tps": (prompt_len / prefill_wall) if prefill_wall else None,
        "decode_seconds": decode_seconds,
        "decode_tokens": decode_tokens,
        "decode_tps": (decode_tokens / decode_seconds)
        if decode_seconds and decode_tokens > 0 else None,
        "timing_method": "engine_step_loop_first_token",
        "raw_metrics": metric_fields,
        "num_cached_tokens": getattr(out, "num_cached_tokens", None) if out else None,
    }
    Path(spec["result_path"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("worker_ok " + json.dumps({k: v for k, v in result.items() if k != "tokens"}))


def main() -> int:
    args = parse_args()
    if args.worker_spec:
        run_worker(args.worker_spec)
        return 0

    lengths = [int(item) for item in args.lengths.split(",") if item]
    devices = [int(item) for item in args.devices.split(",") if item]
    args.work_dir.mkdir(parents=True, exist_ok=True)

    tag = "plain" if args.mtp_tokens == 0 else f"mtp{args.mtp_tokens}"
    results: list[dict[str, Any]] = []
    for length in lengths:
        fixture = args.fixture_dir / f"tokens_{length}.txt"
        prompt_tokens = read_fixture(fixture, length)
        spec_path = args.work_dir / f"spec_{tag}_{length}.json"
        result_path = args.work_dir / f"result_{tag}_{length}.json"
        log_path = args.work_dir / f"log_{tag}_{length}.txt"
        spec_path.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "prompt_token_ids": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "tp_world": args.tp_world,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "gdn_prefill_backend": args.gdn_prefill_backend,
            "mtp_tokens": args.mtp_tokens,
            "result_path": str(result_path),
        }), encoding="utf-8")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        # The first real forward JITs the Triton gated-delta kernels, which
        # exceeds the 300 s default execute_model RPC timeout on this GPU.
        env["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "3600"

        print(f"[run] vllm {tag} length={length}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            # The sampler spans the whole child lifetime, so the peak covers
            # weight load, KV reservation, and the measured prefill/decode.
            with GpuMemorySampler(devices) as sampler:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--worker-spec", str(spec_path)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
                )
        elapsed = time.perf_counter() - started
        memory = sampler.report()

        if completed.returncode != 0 or not result_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            results.append({
                "engine": "vllm", "mode": tag, "mtp_k": args.mtp_tokens,
                "prompt_tokens": length, "status": "failed",
                "returncode": completed.returncode,
                "process_seconds": elapsed,
                "failure_reason": "\n".join(tail),
                "gpu_memory": memory,
                "log": str(log_path),
            })
            print(f"[fail] vllm {tag} length={length} rc={completed.returncode}", flush=True)
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update({
            "status": "ok",
            "process_seconds": elapsed,
            "token_fixture": str(fixture),
            "token_fixture_sha256": fixture_sha256(fixture),
            "log": str(log_path),
            "gpu_memory": memory,
            "gpu_memory_bytes_after": memory["after_bytes"],
        })
        results.append(result)
        print(f"[ok] vllm {tag} length={length} "
              f"prefill_tps={result.get('prefill_tps')} "
              f"decode_tps={result.get('decode_tps')}", flush=True)

    payload = {
        "engine": "vllm",
        "checkpoint": str(args.checkpoint),
        "tp_world": args.tp_world,
        "devices": devices,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.max_num_batched_tokens,
        "mtp_tokens": args.mtp_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "gdn_prefill_backend": args.gdn_prefill_backend,
        "python": sys.executable,
        "results": results,
    }
    result_path = args.result or (args.work_dir / f"summary_{tag}.json")
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
