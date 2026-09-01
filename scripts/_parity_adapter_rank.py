#!/usr/bin/env python3
"""One TP rank of the C++ adapter, used by verify_cpp_adapter_parity.py.

Runs a single greedy request through ``CppBackend`` and writes the resulting
tokens, finish reason, usage, and resolved EOS source to a JSON file so the
parent process can compare them with the reference binary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.cpp_backend import CppBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tp-world", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--nccl-id-path", required=True)
    parser.add_argument("--token-ids-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp16")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    prompt_ids = [
        int(item)
        for item in Path(args.token_ids_file).read_text().replace(",", " ").split()
        if item.strip()
    ]

    engine_args = EngineArgs(
        model=args.ckpt,
        backend="cpp",
        tensor_parallel_size=args.tp_world,
        tensor_parallel_rank=args.tp_rank,
        max_model_len=args.max_context,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
        backend_options={"nccl_id_path": args.nccl_id_path},
    )
    backend = CppBackend(engine_args)
    request = GenerationRequest(
        prompt_tokens=prompt_ids,
        request_id=f"parity-{args.tp_rank}",
        sampling_params=SamplingParams(max_tokens=args.max_new_tokens),
    )

    payload: dict = {
        "tp_rank": args.tp_rank,
        "eos_token_ids": sorted(backend.eos_token_ids),
        "eos_source": backend.capabilities.details.get("eos_source"),
        "prompt_tokens": len(prompt_ids),
    }
    try:
        if args.stream:
            token_ids: list[int] = []
            finish_reason = None
            usage = None
            for event in backend.stream(request):
                if event.token_id is not None:
                    token_ids.append(int(event.token_id))
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.usage is not None:
                    usage = [event.usage.prompt_tokens, event.usage.completion_tokens]
            payload.update(
                mode="stream",
                token_ids=token_ids,
                finish_reason=finish_reason,
                usage=usage,
            )
        else:
            result = backend.generate([request])[0]
            payload.update(
                mode="offline",
                token_ids=[int(token) for token in result.token_ids],
                text=result.text,
                finish_reason=result.finish_reason,
                usage=[result.usage.prompt_tokens, result.usage.completion_tokens],
            )
    finally:
        backend.close()

    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
