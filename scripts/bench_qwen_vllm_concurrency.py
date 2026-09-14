#!/usr/bin/env python3
"""vLLM-side counterpart of the native-server HTTP concurrency acceptance test.

`docs/cpp_openai_concurrency_validation.md` records PocketLLM's concurrent
throughput (1.43x/2.64x/2.93x wall speedup at 2/4/8 requests), but every number
in it is PocketLLM-only: the repository has never measured vLLM under
concurrency, so "PocketLLM multiplexes requests" and "PocketLLM multiplexes
requests as well as vLLM does" are not the same claim and only the first is
supported. This harness produces the missing half.

The client side is imported from `scripts/bench_cpp_openai_concurrency.py`
rather than reimplemented. That is deliberate: the prompt text, the request
payload, the completion validation, and the definition of "wall seconds" are
then provably identical on both sides, so a difference between the two records
comes from the engine and not from the harness. The only thing that differs is
which server answers on the port.

Both modes launch a real `vllm.entrypoints.openai.api_server` over HTTP, because
PocketLLM's published numbers were taken over HTTP and comparing an in-process
batch API against an HTTP server would credit vLLM with work it did not skip.
One server lifetime per mode, in a separate process, for the same reason the
PocketLLM harness does it: model construction, NCCL state, and GPU clocks must
not carry across a configuration change.

Example:
    python scripts/bench_qwen_vllm_concurrency.py \
        --python /home/lvyufeng/miniconda3/envs/vllm-2080ti-v015/bin/python \
        --mode batch --max-batch-size 8 \
        --json-out /tmp/vllm-http-concurrency-batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Reused verbatim from the PocketLLM harness so both sides measure the same
# thing: see the module docstring.
from bench_cpp_openai_concurrency import (  # noqa: E402
    http_request,
    parse_devices,
    require,
    run_concurrent,
    run_interleave,
    run_stream_cases,
)

VLLM_ENV = "/home/lvyufeng/miniconda3/envs/vllm-2080ti-v015/bin/python"
DEFAULT_CKPT = "/mnt/data2/Qwen3.8-27B-FP8"


@dataclass
class VllmServer:
    """A single fastapi/uvicorn server process.

    Duck-typed against `ServerGroup` from the PocketLLM harness so its
    `run_concurrent`, `run_interleave`, and `group_timeout` can be reused
    unchanged. `processes` holds one entry, which is what makes
    `wait_for_health` and the timeout arithmetic work.
    """

    processes: list[subprocess.Popen[bytes]]
    log_handles: list[Any]
    log_dir: pathlib.Path
    base_url: str
    log_path: pathlib.Path
    extra: dict[str, Any] = field(default_factory=dict)

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 30.0
        for process in self.processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        for handle in self.log_handles:
            handle.close()

    def tail_log(self, lines: int = 80) -> str:
        if not self.log_path.is_file():
            return "(no server log)"
        text = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(text[-lines:])


def wait_for_health(group: VllmServer, timeout: float) -> None:
    """Wait for `/health`, tolerating the long TP4 weight load.

    vLLM answers `/health` with 200 only after the engine is up; before that the
    port is not listening at all, so a connection error is the expected state
    rather than a failure. A server that dies instead of loading must be
    reported with its log, because the usual cause is an OOM or a missing
    kernel for the architecture and the traceback is the only place that says
    which.
    """
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        process = group.processes[0]
        if process.poll() is not None:
            raise RuntimeError(
                f"vllm server exited with {process.returncode} before readiness\n{group.tail_log()}"
            )
        try:
            result = http_request(group.base_url, "/health", timeout=5.0)
            if result.status == 200:
                return
            last_error = f"HTTP {result.status}: {result.text[:200]}"
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3.0)
    raise TimeoutError(f"vllm server did not become healthy: {last_error}\n{group.tail_log()}")


def start_server(args: argparse.Namespace, log_dir: pathlib.Path) -> VllmServer:
    devices = parse_devices(args.devices)
    log_path = log_dir / "vllm_server.log"
    handle = log_path.open("wb")

    command = [
        str(pathlib.Path(args.python).resolve()),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", str(pathlib.Path(args.checkpoint).resolve()),
        "--served-model-name", args.served_model_name,
        "--tensor-parallel-size", str(len(devices)),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        # The default ("auto") resolves to flashqla_legacy on SM75, which
        # imports the separate `flash_qla` package that this environment does
        # not have; Triton/FLA is the working SM75 GDN prefill path. Same
        # setting the single-request A/B uses.
        "--additional-config", json.dumps({"gdn_prefill_backend": args.gdn_prefill_backend}),
    ]
    if args.enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    else:
        command.append("--no-enable-chunked-prefill")
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    group = VllmServer([process], [handle], log_dir, f"http://127.0.0.1:{args.port}", log_path)
    try:
        started = time.perf_counter()
        wait_for_health(group, args.startup_timeout)
        group.extra["load_seconds"] = time.perf_counter() - started
    except Exception:
        group.stop()
        raise
    return group


def resolve_model_id(group: VllmServer, fallback: str) -> str:
    """Read the served id rather than assuming it.

    `validate_completion` compares the response's `model` against the expected
    id, so a mismatch here would look like a protocol failure later. vLLM
    reports the `--served-model-name`, but reading it back is free and removes
    the assumption.
    """
    result = http_request(group.base_url, "/v1/models", timeout=30.0)
    require(result.status == 200, f"/v1/models returned HTTP {result.status}")
    data = result.json().get("data") or []
    require(data, "/v1/models returned no models")
    return str(data[0].get("id") or fallback)


def run(args: argparse.Namespace) -> dict[str, Any]:
    log_dir = pathlib.Path(args.log_dir or tempfile.mkdtemp(prefix="vllm-http-concurrency-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    group: VllmServer | None = None
    started = time.perf_counter()
    record: dict[str, Any] = {
        "engine": "vllm",
        "mode": args.mode,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "python": str(pathlib.Path(args.python).resolve()),
        "tp_world": len(parse_devices(args.devices)),
        "devices": args.devices,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "gdn_prefill_backend": args.gdn_prefill_backend,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "max_tokens": args.max_tokens,
        "short_prompt_words": args.short_prompt_words,
        "concurrency_levels": list(args.concurrency),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(pathlib.Path(__file__).resolve().parent.parent)
        ).strip(),
        "log_dir": str(log_dir),
    }
    vllm_version = subprocess.run(
        [str(pathlib.Path(args.python).resolve()), "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True, check=False,
    )
    record["vllm_version"] = vllm_version.stdout.strip() or None

    try:
        group = start_server(args, log_dir)
        record["load_seconds"] = group.extra.get("load_seconds")
        model = resolve_model_id(group, args.served_model_name)
        record["model"] = model

        # Discarded passes over the measured ladder. vLLM pays a large
        # first-request cost (Triton JIT, graph replay setup, sampler warmup)
        # that is charged to whichever request happens to be first, and the
        # first request in the ladder is the count=1 case that the comparison
        # reads as "single-request latency". Without this, one warmup artifact
        # becomes a fabricated single-request regression and an equally
        # fabricated concurrency gain.
        if args.warmup_rounds > 0:
            warmups = []
            for _ in range(args.warmup_rounds):
                for n in [1] + list(args.concurrency):
                    warm = run_concurrent(group, model, n, args.short_prompt_words, args.max_tokens)
                    warmups.append({"count": n, "wall_seconds": warm["wall_seconds"]})
            record["warmup"] = warmups

        single = run_concurrent(group, model, 1, args.short_prompt_words, args.max_tokens)
        concurrency = [
            run_concurrent(group, model, n, args.short_prompt_words, args.max_tokens)
            for n in args.concurrency
        ]
        interleave = run_interleave(group, model, args.max_tokens)
        record.update({"single": single, "concurrency": concurrency, "interleave": interleave})
        if args.with_streaming:
            record["streaming"] = run_stream_cases(group, model, args.max_tokens)
        record["status"] = "pass"
    except Exception as exc:
        record.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        if group is not None:
            record["server_log_tail"] = group.tail_log()
        raise
    finally:
        if group is not None:
            group.stop()
        record["elapsed_seconds"] = time.perf_counter() - started
        if args.json_out:
            pathlib.Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--python", default=VLLM_ENV, help="interpreter that has vllm installed")
    parser.add_argument("--served-model-name", default="qwen3.8-27b-fp8")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=18381)
    parser.add_argument("--mode", choices=("serial", "batch"), default="batch")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gdn-prefill-backend", default="triton",
                        choices=["triton", "flashqla_legacy", "flashinfer", "auto"])
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--short-prompt-words", type=int, default=128)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup-rounds", type=int, default=1,
                        help="discarded passes over the measured ladder before measuring (0 disables)")
    parser.add_argument("--with-streaming", action="store_true",
                        help="also run the concurrent-SSE and client-disconnect cases")
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--log-dir")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    # Mirrors the PocketLLM harness: the serial control is the same client
    # behaviour against a server that can only hold one sequence, so the
    # difference between the two modes is the scheduler and nothing else.
    if args.mode == "serial":
        args.max_num_seqs = 1
    else:
        args.max_num_seqs = args.max_batch_size
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
