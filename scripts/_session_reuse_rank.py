#!/usr/bin/env python3
"""One TP rank of the session-reuse check, used by verify_cpp_session_reuse.py.

Runs several greedy requests back to back through a single ``CppBackend`` and
records the tokens each one produced.  The point is the request that stops at
EOS: the native session keeps mutating for the whole token budget, so if the
adapter let it run past EOS the following request would resume from state the
caller never saw.  The parent process compares these tokens with the same
requests run in fresh backends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.cpp_backend import CppBackend


def load_cases(path: str) -> list[list[int]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append([int(item) for item in line.replace(",", " ").split()])
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tp-world", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--nccl-id-path", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp16")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="build a fresh backend per case instead of reusing one session",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases_file)

    def build() -> CppBackend:
        return CppBackend(
            EngineArgs(
                model=args.ckpt,
                backend="cpp",
                tensor_parallel_size=args.tp_world,
                tensor_parallel_rank=args.tp_rank,
                max_model_len=args.max_context,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                kv_cache_dtype=args.kv_cache_dtype,
                backend_options={"nccl_id_path": args.nccl_id_path},
            )
        )

    records: list[dict] = []
    eos_ids: list[int] = []
    eos_source = None

    def run_case(backend: CppBackend, index: int, prompt_ids: list[int]) -> dict:
        request = GenerationRequest(
            prompt_tokens=prompt_ids,
            request_id=f"session-{args.tp_rank}-{index}",
            sampling_params=SamplingParams(max_tokens=args.max_new_tokens),
        )
        result = backend.generate([request])[0]
        return {
            "index": index,
            "prompt_tokens": len(prompt_ids),
            "token_ids": [int(token) for token in result.token_ids],
            "finish_reason": result.finish_reason,
            "usage": [result.usage.prompt_tokens, result.usage.completion_tokens],
        }

    if args.isolated:
        for index, prompt_ids in enumerate(cases):
            backend = build()
            try:
                if not eos_ids:
                    eos_ids = sorted(backend.eos_token_ids)
                    eos_source = backend.capabilities.details.get("eos_source")
                records.append(run_case(backend, index, prompt_ids))
            finally:
                backend.close()
    else:
        backend = build()
        try:
            eos_ids = sorted(backend.eos_token_ids)
            eos_source = backend.capabilities.details.get("eos_source")
            for index, prompt_ids in enumerate(cases):
                records.append(run_case(backend, index, prompt_ids))
        finally:
            backend.close()

    Path(args.out).write_text(
        json.dumps(
            {
                "tp_rank": args.tp_rank,
                "mode": "isolated" if args.isolated else "shared",
                "eos_token_ids": eos_ids,
                "eos_source": eos_source,
                "cases": records,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
