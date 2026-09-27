"""One request through one runtime, on the serial path or through the shared scheduler.

Two arms, and they are the same runtime, the same checkpoint and the same request: the only
difference is whether the adapter runs its own loop or submits to `BatchScheduler`. That is what
makes the token ids comparable and the timings worth reading -- a runtime that joins the scheduler
and returns a different answer has not joined it, it has replaced it.

One arm per process, deliberately. A second engine built in the same process runs about 10% slower
than the first, so two arms in one process measure the order they were constructed in as much as
the thing being compared.

    python scripts/bench_runtime_scheduler_path.py \
        --backend xing4 --arm serial \
        --model /mnt/data2/Xing4.0-29B-A4B-GGUF \
        --backend-option device=cuda:0 \
        --prompt "..." --max-tokens 32 --json-out /tmp/xing4-serial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--arm", choices=("serial", "batched"), required=True)
    parser.add_argument("--model", required=True, help="checkpoint path the backend opens")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discarded requests before the measured ones; the first one through a freshly "
        "loaded checkpoint pays kernel-load and allocator costs that land on whichever arm ran "
        "first and make it look slower",
    )
    parser.add_argument("--repeat", type=int, default=1, help="measured requests")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="repeatable; a different prompt each run keeps the prefix cache out of the answer",
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        help="name=value, repeatable; passed to the backend as it would be on the CLI",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def options(pairs):
    values = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not name:
            raise SystemExit(f"--backend-option {pair!r} is not name=value")
        lowered = value.lower()
        if lowered in ("true", "false"):
            values[name] = lowered == "true"
        else:
            try:
                values[name] = int(value)
            except ValueError:
                values[name] = value
    return values


def main(argv=None) -> int:
    args = parse_args(argv)
    from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
    from pocketllm.backends.factory import create_backend

    backend_options = options(args.backend_option)
    if args.arm == "batched":
        backend_options["enable_batching"] = True
    engine_args = EngineArgs(
        model=args.model,
        backend=args.backend,
        tokenizer_path=args.tokenizer_path,
        max_model_len=args.max_model_len,
        backend_options=backend_options,
    )
    backend = create_backend(engine_args)
    backend.prepare()

    capabilities = backend.capabilities
    print(f"backend {capabilities.name}: supports_batch={capabilities.supports_batch} "
          f"scheduler={capabilities.details.get('scheduler')}")

    request_template = args.prompt or ["Explain what a KV cache is in one paragraph."]

    def ask(index: int) -> GenerationRequest:
        # A different prompt per run, so a runtime with a prefix store does not answer the second
        # run out of the first one's cache -- that is a real speedup and not the one being measured.
        text = request_template[index % len(request_template)]
        return GenerationRequest(
            request_id=f"bench{index}",
            prompt=f"{text} (round {index})",
            sampling_params=SamplingParams(
                max_tokens=args.max_tokens, temperature=args.temperature
            ),
        )

    for index in range(args.warmup):
        backend.generate([ask(index)])

    runs = []
    first = None
    for index in range(args.repeat):
        request = ask(args.warmup + index)
        started = time.perf_counter()
        result = backend.generate([request])[0]
        wall = time.perf_counter() - started
        first = result if first is None else first
        runs.append(
            {
                "wall_seconds": wall,
                "prefill_seconds": float(result.timings.prefill_seconds),
                "decode_seconds": float(result.timings.decode_seconds),
                "ttft_seconds": float(result.timings.ttft_seconds),
                "completion_tokens": int(result.usage.completion_tokens),
                "prompt_tokens": int(result.usage.prompt_tokens),
                "token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
            }
        )

    def median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    record = {
        "backend": args.backend,
        "arm": args.arm,
        "supports_batch": bool(capabilities.supports_batch),
        "scheduler": capabilities.details.get("scheduler"),
        "repeat": args.repeat,
        "runs": runs,
        "token_ids": first.token_ids,
        # Identical answers are the claim that matters: a runtime that joins the scheduler and
        # returns something else has been replaced, not routed.
        "identical_answers": len({tuple(run["token_ids"]) for run in runs}) == 1,
        "wall_median_seconds": median([run["wall_seconds"] for run in runs]),
        "prefill_median_seconds": median([run["prefill_seconds"] for run in runs]),
        "decode_median_seconds": median([run["decode_seconds"] for run in runs]),
    }
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}, indent=2))
    print(f"wall {record['wall_median_seconds']:.3f}s "
          f"(min {min(r['wall_seconds'] for r in runs):.3f}, "
          f"max {max(r['wall_seconds'] for r in runs):.3f})")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
