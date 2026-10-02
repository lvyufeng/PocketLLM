"""The command line: what this install has, and how to serve one checkpoint.

The old CLI's flags were *generated* from a runtime's option declarations
(``backends/cli_surface.py``), because there was machinery to declare per-runtime
levers and it had to reach ``--help``.  That machinery is gone with the runtimes
it described; what is left is a small parser whose ``--device`` and ``--backend``
choices are read from the backend **registry**, which is the same declaration the
engine dispatches against.  There is still one source of truth -- it just moved
from a per-runtime table to the registry that was always there.

The commands mirror the three questions a user has:

* ``devices`` -- which backends this host can actually open, and what the blocked
  ones are missing.  This is the command to run *when something is wrong*, so it
  must work with nothing installed, and it does: every check is a filesystem
  probe, and no runtime is imported.
* ``backends`` -- what each backend declares, whether or not the host can open it.
* ``run`` -- load a checkpoint and run it once, the interactive path.

``serve`` lives in :mod:`pocketllm.server`, which is the module that imports the
HTTP layer; importing this module does not import it, so ``pocketllm devices`` on
a phone does not pay for ``http.server`` either.
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import EngineArgs, device_kinds
from .backends import registry

__all__ = ["build_parser", "main"]


def _device_kinds() -> tuple[str, ...]:
    return device_kinds()


def _backend_names() -> tuple[str, ...]:
    return ("auto",) + registry.builtin()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocketllm",
        description="Run a large model on one accelerator: edge and mobile.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser(
        "devices",
        help="list the backends this host can open, and what the rest are missing",
    )
    devices.add_argument("--json", action="store_true", help="emit machine-readable rows")

    subparsers.add_parser("backends", help="list what each backend declares")
    subparsers.add_parser("architectures", help="list the buildable model architectures")

    run = subparsers.add_parser("run", help="load a checkpoint and run one prompt")
    _add_engine_flags(run)
    run.add_argument("--prompt", required=True, help="the prompt to run")
    run.add_argument("--max-tokens", type=int, default=16)

    serve = subparsers.add_parser("serve", help="start the OpenAI-compatible server")
    _add_engine_flags(serve)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", help="model id reported by /v1/models")

    ops = subparsers.add_parser(
        "ops",
        help="resolve an op against the backends, and say why each candidate did or did not",
    )
    ops.add_argument("--op", required=True, help="an op name from the ABI")
    ops.add_argument("--device", default="auto", help="the device to resolve against")
    ops.add_argument("--backend", action="append", default=[], help="restrict to a backend; repeatable")
    ops.add_argument(
        "--allow-reference-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the reference backend a candidate (default: on)",
    )

    return parser


def _add_engine_flags(parser: argparse.ArgumentParser) -> None:
    """The flags both ``run`` and ``serve`` share.

    Every value here reaches :class:`~pocketllm.api.EngineArgs`, which validates
    it; the parser's own ``choices`` are for ``--help`` and for failing before
    anything loads, and are read from the registry so the two cannot disagree.
    """
    parser.add_argument("--model", required=True, help="checkpoint directory or .gguf path")
    parser.add_argument("--backend", choices=_backend_names(), default="auto")
    parser.add_argument(
        "--device",
        choices=_device_kinds(),
        default="auto",
        help="the device this process runs on; `auto` asks the host (default: auto)",
    )
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--config-path")
    parser.add_argument("--model-format", choices=["auto", "gguf"], default="auto")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--kv-cache-dtype", default="auto")


def _args(namespace: argparse.Namespace) -> EngineArgs:
    return EngineArgs(
        model=namespace.model,
        backend=namespace.backend,
        tokenizer_path=namespace.tokenizer_path,
        config_path=namespace.config_path,
        model_format=namespace.model_format,
        device=namespace.device,
        max_model_len=namespace.max_model_len,
        dtype=namespace.dtype,
        kv_cache_dtype=namespace.kv_cache_dtype,
    )


# -- commands ----------------------------------------------------------------


def _cmd_devices(namespace: argparse.Namespace) -> int:
    rows = registry.describe()
    if namespace.json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(row["name"]) for row in rows) if rows else 4
    for row in rows:
        state = "available" if row["available"] else f"missing {row['missing']}"
        graph = row["graph"] if row["graph"] != "none" else "eager only"
        print(f"{row['name']:<{width}}  {row['kind'] or '-':<8}  {state:<28}  {graph:<16}  {row['ops']} ops")
    print(f"\nthis process opens one device; the kinds it knows are {', '.join(_device_kinds())}")
    return 0


def _cmd_backends(namespace: argparse.Namespace) -> int:
    for row in registry.describe():
        print(f"{row['name']}  [{row['source']}]")
        print(f"    {row['summary'] or 'no summary'}")
        print(f"    device: {row['kind'] or '-'}   ops: {row['ops']}   graph: {row['graph']}")
    return 0


def _cmd_architectures(namespace: argparse.Namespace) -> int:
    from .architectures import ARCHITECTURES, names

    for name in names():
        entry = ARCHITECTURES.get(name)
        summary = entry.summary if entry is not None else "entry point"
        print(f"{name:<12}  {summary}")
    return 0


def _cmd_ops(namespace: argparse.Namespace) -> int:
    """The debugging surface the old tree lacked: why this op did (not) resolve.

    Resolution is a pure function over declarations -- it reads no bytes and opens
    no device -- so this answers for a phone's backends from a development host.
    """
    from .kernels.device import Device, parse_device
    from .kernels.dispatch import Dispatcher
    from .kernels.registry import OPS

    if namespace.op not in OPS.names():
        print(f"no op named {namespace.op!r}; the ABI declares {', '.join(sorted(OPS.names()))}")
        return 2

    try:
        device = parse_device(namespace.device if namespace.device != "auto" else "cpu")
    except ValueError as exc:
        print(f"{exc}; pass a device like `cpu` or `cuda:1`")
        return 2

    if namespace.backend:
        backends = [registry.get(name, available_only=False) for name in namespace.backend]
    else:
        backends = [entry.factory() for entry in registry._entries()]

    dispatcher = Dispatcher(
        backends,
        allow_reference_fallback=namespace.allow_reference_fallback,
    )
    resolution = dispatcher.explain(namespace.op, [], device)
    chosen = resolution.chosen
    if chosen is None:
        print(resolution.reason())
        return 1
    print(f"{resolution.op} on {device} -> {chosen.backend.name} ({chosen.capability.rank})")
    for name, why in resolution.rejected:
        print(f"  rejected {name}: {why}")
    return 0


def _cmd_run(namespace: argparse.Namespace) -> int:
    """Load a checkpoint and generate.  Needs a model, which does not ship yet."""
    engine_args = _args(namespace)
    # The flags parse and validate -- that is real work, and the errors are the
    # same ones a working `run` would give.  What is missing is the other half:
    # a builder for the checkpoint's architecture, and the GGUF tokenizer.  Saying
    # which is missing is the difference between a stub and a lie.
    raise SystemExit(
        "`pocketllm run` has no model architecture or tokenizer yet; "
        f"parsed {engine_args.checkpoint_dir!r} on device {engine_args.device!r}. "
        "Run `pocketllm devices` to see what this host can open."
    )


def _cmd_serve(namespace: argparse.Namespace) -> int:
    """Start the OpenAI-compatible server.  Needs an engine backend, none ships yet."""
    engine_args = _args(namespace)
    raise SystemExit(
        "`pocketllm serve` has no engine backend yet; "
        f"parsed {engine_args.checkpoint_dir!r} on device {engine_args.device!r}. "
        "The HTTP surface itself is importable as pocketllm.server.openai.serve."
    )


_COMMANDS = {
    "devices": _cmd_devices,
    "backends": _cmd_backends,
    "architectures": _cmd_architectures,
    "ops": _cmd_ops,
    "run": _cmd_run,
    "serve": _cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    return _COMMANDS[namespace.command](namespace)


if __name__ == "__main__":
    sys.exit(main())