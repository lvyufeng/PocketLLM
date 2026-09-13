#!/usr/bin/env python3
"""Ascend-specific long-context Qwen benchmark with deterministic fixtures.

Differences from bench_qwen_long_context.py:
- Each rank receives its actual physical --device ID (no CUDA_VISIBLE_DEVICES)
- Uses ACL self-reported HBM allocation, not nvidia-smi
- Records Ascend environment switches (GQA, fusion, comm stream, overlap)
- Validates that scripts/ascend_env.sh was sourced before the run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_TEXT = (
    "Decode context parallelism partitions the key-value history across devices "
    "while tensor parallelism partitions attention heads and matrix weights. "
    "A correct implementation must merge the distributed softmax maximum, "
    "denominator, and weighted value sum without changing greedy generation. "
    "Long-context inference on PCIe-connected GPUs is useful only when the "
    "reduced attention scan costs more than the added collectives. This benchmark "
    "uses deterministic tokenizer output from a natural-language systems paragraph. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Qwen checkpoint directory")
    parser.add_argument(
        "--binary",
        default="cpp_engine/build-ascend/pocketllm_engine",
        help="built Ascend C++ engine executable",
    )
    parser.add_argument(
        "--lengths",
        default="4096,8192,32768",
        help="comma-separated prompt lengths",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-dtype", default="fp16")
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-top-p", type=float, default=1.0)
    parser.add_argument("--qwen-top-k", type=int, default=0)
    parser.add_argument("--qwen-seed", type=int, default=0)
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="serial fresh-process repetitions per prompt length",
    )
    parser.add_argument(
        "--topology-label",
        default="",
        help="human-readable topology label stored in the result artifact",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="environment override passed to every rank and recorded verbatim",
    )
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated physical NPU IDs; defaults to 0..tp_world-1",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="Qwen layer limit; keep 0 for the complete 64-layer model",
    )
    parser.add_argument(
        "--work-dir",
        default=".tmp/qwen_ascend_long",
        help="directory for token fixtures, HCCL IDs, logs, and results",
    )
    return parser.parse_args()


def check_ascend_env() -> dict[str, str]:
    """Validate CANN environment and collect runtime metadata."""
    required = ["ASCEND_TOOLKIT_HOME", "LD_LIBRARY_PATH"]
    env_info = {}

    for var in required:
        val = os.environ.get(var, "")
        if not val:
            print(f"error: {var} not set; source scripts/ascend_env.sh first", file=sys.stderr)
            sys.exit(1)
        env_info[var] = val

    # Collect Ascend tuning switches
    for switch in [
        "QWEN_GQA_OPTIMIZED",
        "QWEN_FUSE_AB_PROJECTION",
        "QWEN_NCCL_COMM_STREAM",
        "QWEN_COMM_OVERLAP_SLICES",
        "QWEN_GATED_DELTA_PRENORMALIZE",
    ]:
        env_info[switch] = os.environ.get(switch, "")

    return env_info


def generate_token_fixture(
    ckpt: str, length: int, text: str, work_dir: Path
) -> Path:
    """Create or reuse a deterministic tokenizer fixture."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    fixture_path = work_dir / f"fixture_{length}_{text_hash}.json"

    if fixture_path.exists():
        return fixture_path

    work_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    repeated_text = (text * ((length // len(text)) + 2))[:length * 10]
    token_ids = tokenizer.encode(repeated_text, add_special_tokens=False)[:length]

    if len(token_ids) < length:
        token_ids += [tokenizer.pad_token_id or 0] * (length - len(token_ids))

    fixture_data = {
        "text_hash": text_hash,
        "length": length,
        "token_count": len(token_ids),
        "token_ids": token_ids,
    }

    with open(fixture_path, "w") as f:
        json.dump(fixture_data, f)

    return fixture_path


def run_tp_benchmark(
    binary: str,
    ckpt: str,
    fixture_path: Path,
    devices: list[int],
    max_new_tokens: int,
    prefill_chunk: int,
    kv_dtype: str,
    layers: int,
    temp: float,
    top_p: float,
    top_k: int,
    seed: int,
    work_dir: Path,
    env_overrides: list[str],
) -> dict[str, Any]:
    """Launch TP ranks and collect timing/parity results."""
    tp_world = len(devices)
    run_id = int(time.time() * 1000) % 1000000
    id_path = work_dir / f"hccl_{run_id}.id"
    token_ids_file = work_dir / f"tokens_{run_id}.txt"
    log_dir = work_dir / f"logs_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HCCL_WHITELIST_DISABLE"] = "1"
    env["POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS"] = "12000"

    for override in env_overrides:
        if "=" in override:
            k, v = override.split("=", 1)
            env[k] = v

    # Load fixture tokens and write to file
    with open(fixture_path) as f:
        fixture = json.load(f)
    token_ids = fixture["token_ids"]

    with open(token_ids_file, "w") as f:
        f.write(",".join(map(str, token_ids)))

    common_args = [
        binary,
        "--ckpt", ckpt,
        "--smoke-forward",
        "--resident-bench",
        "--tp-world", str(tp_world),
        "--nccl-id-path", str(id_path),
        "--max-new-tokens", str(max_new_tokens),
        "--token-ids-file", str(token_ids_file),
    ]

    # Always pass --smoke-layers explicitly (0 means all 64 layers, >0 means N layers)
    common_args += ["--smoke-layers", str(layers)]

    # Each rank uses physical device ID
    procs = []
    for rank, device in enumerate(devices):
        rank_args = common_args + [
            "--tp-rank", str(rank),
            "--device", str(device),
        ]
        log_path = log_dir / f"rank{rank}.log"
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                rank_args,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        procs.append((proc, log_path))

    # Wait for all ranks
    exit_codes = []
    for proc, log_path in procs:
        code = proc.wait()
        exit_codes.append(code)

    if any(c != 0 for c in exit_codes):
        print(f"error: some ranks failed with codes {exit_codes}", file=sys.stderr)
        return {"error": "nonzero_exit", "exit_codes": exit_codes}

    # Parse rank 0 timing
    rank0_log = log_dir / "rank0.log"
    timing = parse_rank0_timing(rank0_log)

    # Verify token parity
    generated_tokens = []
    for i in range(tp_world):
        log_path = log_dir / f"rank{i}.log"
        tokens = extract_generated_tokens(log_path)
        generated_tokens.append(tokens)

    parity_ok = all(t == generated_tokens[0] for t in generated_tokens)

    return {
        "timing": timing,
        "parity_ok": parity_ok,
        "generated_tokens": generated_tokens[0] if parity_ok else generated_tokens,
        "log_dir": str(log_dir),
    }


def parse_rank0_timing(log_path: Path) -> dict[str, Any]:
    """Extract prefill/decode timing from rank 0 log."""
    with open(log_path) as f:
        content = f.read()

    timing = {}

    # ACL memory allocation
    match = re.search(r"ACL allocated (\d+) bytes", content)
    if match:
        timing["acl_allocated_bytes"] = int(match.group(1))

    # Prefill timing
    match = re.search(r"Prefill:\s+([\d.]+)\s+s,\s+([\d.]+)\s+tok/s", content)
    if match:
        timing["prefill_seconds"] = float(match.group(1))
        timing["prefill_tok_per_sec"] = float(match.group(2))

    # Decode timing (exclude first token)
    match = re.search(r"Decode\s+\(excluding first token\):\s+([\d.]+)\s+s,\s+([\d.]+)\s+ms/tok,\s+([\d.]+)\s+tok/s", content)
    if match:
        timing["decode_seconds"] = float(match.group(1))
        timing["decode_ms_per_tok"] = float(match.group(2))
        timing["decode_tok_per_sec"] = float(match.group(3))

    return timing


def extract_generated_tokens(log_path: Path) -> list[int]:
    """Parse generated token IDs from rank log."""
    with open(log_path) as f:
        content = f.read()

    tokens = []
    for line in content.splitlines():
        match = re.match(r"token\s+\d+:\s+(\d+)", line)
        if match:
            tokens.append(int(match.group(1)))

    return tokens


def main() -> None:
    args = parse_args()

    env_info = check_ascend_env()

    if not Path(args.binary).is_file():
        print(f"error: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.ckpt).is_dir():
        print(f"error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    devices = (
        [int(d) for d in args.devices.split(",")] if args.devices
        else list(range(args.tp_world))
    )

    if len(devices) != args.tp_world:
        print(f"error: device count {len(devices)} != tp_world {args.tp_world}", file=sys.stderr)
        sys.exit(1)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    lengths = [int(l) for l in args.lengths.split(",")]

    results = {
        "checkpoint": args.ckpt,
        "binary": args.binary,
        "devices": devices,
        "tp_world": args.tp_world,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "layers": args.layers,
        "topology_label": args.topology_label,
        "env_info": env_info,
        "env_overrides": args.env,
        "runs": [],
    }

    for length in lengths:
        print(f"\n=== Generating fixture for length {length} ===")
        fixture_path = generate_token_fixture(
            args.ckpt, length, DEFAULT_TEXT, work_dir
        )

        for rep in range(args.repetitions):
            print(f"\n=== Running length {length}, repetition {rep+1}/{args.repetitions} ===")
            run_result = run_tp_benchmark(
                args.binary,
                args.ckpt,
                fixture_path,
                devices,
                args.max_new_tokens,
                args.prefill_chunk_tokens,
                args.kv_cache_dtype,
                args.layers,
                args.qwen_temperature,
                args.qwen_top_p,
                args.qwen_top_k,
                args.qwen_seed,
                work_dir,
                args.env,
            )

            run_result["length"] = length
            run_result["repetition"] = rep
            results["runs"].append(run_result)

            if "timing" in run_result:
                print(f"  Prefill: {run_result['timing'].get('prefill_seconds', 'N/A')} s")
                print(f"  Decode: {run_result['timing'].get('decode_seconds', 'N/A')} s")
                print(f"  Parity: {'OK' if run_result['parity_ok'] else 'FAIL'}")

    result_path = work_dir / f"results_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results written to {result_path} ===")


if __name__ == "__main__":
    main()
