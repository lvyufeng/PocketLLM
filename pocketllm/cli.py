"""Command-line entry points for the unified PocketLLM API."""

from __future__ import annotations

import argparse
import json

from .api import EngineArgs
from .backends.factory import create_backend
from .server.openai import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pocketllm", description="PocketLLM unified inference interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the OpenAI-compatible server")
    serve_parser.add_argument("--model", required=True, help="checkpoint directory or model path")
    serve_parser.add_argument("--backend", choices=["auto", "torch", "cpp"], default="auto")
    serve_parser.add_argument("--tokenizer-path")
    serve_parser.add_argument("--config-path")
    serve_parser.add_argument("--model-format", choices=["auto", "safetensors", "gguf"], default="auto")
    serve_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    serve_parser.add_argument("--tensor-parallel-rank", type=int, default=0)
    serve_parser.add_argument("--device")
    serve_parser.add_argument("--max-model-len", type=int)
    serve_parser.add_argument("--dtype")
    serve_parser.add_argument("--kv-cache-dtype", default="auto")
    serve_parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    serve_parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    serve_parser.add_argument("--max-batch-size", type=int, default=1)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--engine-kind", default="qwen")
    serve_parser.add_argument("--routed-experts-device", choices=["gpu", "cpu"], default="gpu")
    serve_parser.add_argument("--pd-mode", choices=["off", "scheduler"], default="scheduler")
    serve_parser.add_argument("--attention-window", type=int, default=0)
    serve_parser.add_argument("--attention-sink-tokens", type=int, default=0)
    serve_parser.add_argument("--speculative-method", choices=["mtp", "dspark", "dflash2"])
    serve_parser.add_argument("--speculative-tokens", type=int, default=1)
    serve_parser.add_argument("--served-model-name", help="model id reported by /v1/models")
    serve_parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend-specific option; repeatable and parsed as JSON when possible",
    )
    return parser


def _backend_options(namespace: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {
        "engine_kind": namespace.engine_kind,
        "routed_experts_device": namespace.routed_experts_device,
        "pd_mode": namespace.pd_mode,
    }
    for item in namespace.backend_option or []:
        key, separator, raw = str(item).partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"--backend-option expects KEY=VALUE, got {item!r}")
        try:
            # JSON keeps numbers, booleans, and nested values typed; a bare
            # string stays a string so paths do not need quoting.
            value = json.loads(raw)
        except ValueError:
            value = raw
        options[key.strip()] = value
    return options


def _args(namespace: argparse.Namespace) -> EngineArgs:
    return EngineArgs(
        model=namespace.model,
        backend=namespace.backend,
        tokenizer_path=namespace.tokenizer_path,
        config_path=namespace.config_path,
        model_format=namespace.model_format,
        tensor_parallel_size=namespace.tensor_parallel_size,
        tensor_parallel_rank=namespace.tensor_parallel_rank,
        device=namespace.device,
        max_model_len=namespace.max_model_len,
        dtype=namespace.dtype,
        kv_cache_dtype=namespace.kv_cache_dtype,
        prefill_chunk_tokens=namespace.prefill_chunk_tokens,
        enable_prefix_caching=namespace.enable_prefix_caching,
        max_batch_size=namespace.max_batch_size,
        attention_window=namespace.attention_window,
        attention_sink_tokens=namespace.attention_sink_tokens,
        speculative_method=namespace.speculative_method,
        speculative_tokens=namespace.speculative_tokens,
        backend_options=_backend_options(namespace),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        engine_args = _args(args)
        backend = create_backend(engine_args)
        model_name = args.served_model_name or args.model
        try:
            serve(backend, host=args.host, port=args.port, model=model_name)
        except KeyboardInterrupt:
            backend.close()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
