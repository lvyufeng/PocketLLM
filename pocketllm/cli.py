"""Command-line entry points for the unified PocketLLM API."""

from __future__ import annotations

import argparse
import json

from .api import EngineArgs, device_hint
from .backends.cli_surface import add_declared_options, resolved_options
from .backends.factory import create_backend
from .server.openai import serve


#: The platforms ``--device`` accepts, which is ``EngineArgs``'s set: ``auto`` asks the build, and
#: an explicit value this build cannot serve is refused rather than retuned. Spelled here as well
#: because the parser has to render the set in ``--help`` and refuse a bad one before anything is
#: constructed -- and it is the *choices* that make the refusal reachable at all.
DEVICE_PLATFORMS = ("auto", "cuda", "ascend", "cpu")


def _device_platform(value: str) -> str:
    """``--device``'s type: the platform, or a refusal naming the flag that answers a card.

    ``choices`` alone would print ``invalid choice: 'cuda:2'``, which is true and unhelpful -- the
    whole of U3's migration is that the value is well formed and belongs to a different flag, and a
    message that does not say so is a message an operator answers by reading the source.
    """
    if value not in DEVICE_PLATFORMS:
        raise argparse.ArgumentTypeError(
            f"must be one of {', '.join(DEVICE_PLATFORMS)}, got {value!r}{device_hint(value)}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pocketllm", description="PocketLLM unified inference interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the OpenAI-compatible server")
    serve_parser.add_argument("--model", required=True, help="checkpoint directory or model path")
    # `auto` asks each runtime whether the checkpoint identifies it. The explicit values stay for a
    # checkpoint that declares nothing, and for forcing a runtime onto one to compare them.
    serve_parser.add_argument("--backend", default="auto")
    serve_parser.add_argument("--tokenizer-path")
    serve_parser.add_argument("--config-path")
    serve_parser.add_argument("--model-format", choices=["auto", "safetensors", "gguf"], default="auto")
    serve_parser.add_argument(
        "--device",
        default="auto",
        type=_device_platform,
        choices=DEVICE_PLATFORMS,
        help=(
            "platform this process runs on: auto, cuda, ascend or cpu; `auto` asks the build "
            "(default: auto). A card is not a platform -- `--device cuda:2` is refused by name, "
            "because `--device-ids` answers it"
        ),
    )
    serve_parser.add_argument(
        "--device-ids",
        default=None,
        metavar="L",
        help=(
            "cards this process may run on, comma-separated; this runtime owns one process, so the "
            "first entry is the one it takes"
        ),
    )
    serve_parser.add_argument("--max-model-len", type=int)
    serve_parser.add_argument("--dtype")
    serve_parser.add_argument("--kv-cache-dtype", default="auto")
    serve_parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    serve_parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    serve_parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "rows the batch scheduler may run at once. Above 1 this asks for the batch path on a "
            "backend that has one; a backend without a scheduler refuses it rather than accepting "
            "a width it cannot honour"
        ),
    )
    serve_parser.add_argument(
        "--enable-batching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "serve through the batch scheduler (default: on for backends that own one). "
            "--no-enable-batching forces the serialized session, and cannot be combined with a "
            "--max-batch-size above 1"
        ),
    )
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--attention-window", type=int, default=0)
    serve_parser.add_argument("--attention-sink-tokens", type=int, default=0)
    serve_parser.add_argument("--served-model-name", help="model id reported by /v1/models")
    serve_parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "backend-specific option, by the key a runtime declares it under; repeatable, parsed as "
            "JSON when possible, and beats the flag that spells the same option"
        ),
    )
    # The runtimes' own levers, one flag each, read out of the declarations they carry. Added last
    # so the host's flags above keep their order in `--help` and the generated sections follow them.
    add_declared_options(serve_parser)
    return parser


def _backend_options(namespace: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
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
        device=namespace.device,
        device_ids=namespace.device_ids,
        max_model_len=namespace.max_model_len,
        dtype=namespace.dtype,
        kv_cache_dtype=namespace.kv_cache_dtype,
        prefill_chunk_tokens=namespace.prefill_chunk_tokens,
        enable_prefix_caching=namespace.enable_prefix_caching,
        max_batch_size=namespace.max_batch_size,
        enable_batching=namespace.enable_batching,
        attention_window=namespace.attention_window,
        attention_sink_tokens=namespace.attention_sink_tokens,
        backend_options=_backend_options(namespace),
        # The flags generated from the runtimes' declarations, plus the host flags that spell one
        # (`--prefill-chunk-tokens`). They are a separate field from `backend_options` because they
        # are a different tier: `--backend-option` names an option outright and wins, which is the
        # order `decode_options` states once for every runtime.
        resolved_options=resolved_options(namespace),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "serve":
        return 2

    engine_args = _args(args)
    # One process owns one card. Nothing in this build launches a second rank, so `create_backend`
    # resolves the runtime and refuses an unroutable checkpoint here, before any weights are read.
    backend = create_backend(engine_args)
    try:
        model_name = args.served_model_name or args.model
        serve(backend, host=args.host, port=args.port, model=model_name)
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())