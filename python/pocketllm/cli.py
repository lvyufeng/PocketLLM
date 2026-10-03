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
* ``serve`` -- the same engine behind OpenAI-compatible HTTP, one request at a time.

``serve`` builds its adapter from :mod:`pocketllm.server`, which is the module that
imports the HTTP layer; importing this module does not import either, so
``pocketllm devices`` on a phone does not pay for ``http.server``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from .api import BackendUnavailableError, ConfigurationError, EngineArgs, SamplingParams, device_kinds
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
    # The sampling flags live on `run` alone and not on `_add_engine_flags`,
    # because `serve` has no decode loop to apply them in yet -- its session is
    # still a stub.  A flag that is parsed and ignored is worse than one that
    # does not exist.
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 means greedy (the default); anything above it samples",
    )
    run.add_argument("--top-k", type=int, help="keep the k most likely tokens (default: no limit)")
    run.add_argument("--top-p", type=float, help="keep the smallest set whose mass reaches p")
    run.add_argument("--min-p", type=float, help="drop tokens below this fraction of the top one")
    run.add_argument("--seed", type=int, help="seed for the sampling draw; unset means it is not reproducible")

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


def _run_device(engine_args: EngineArgs) -> str:
    """The backend name the C core is asked for, from the engine args.

    Two vocabularies meet here and only one of them is ours.  ``--device`` is
    ``EngineArgs.device`` -- a *kind*, from the set ``pocketllm devices`` lists,
    which is what a user picks from and what the Python registry validates.  The
    C core is asked for a *backend*, and it serves ``cpu`` and ``cuda`` today.

    ``auto`` is resolved here rather than passed down, because the C core has no
    auto: it refuses a name it does not know.  It resolves to ``cpu``, which
    every build of the library provides -- the conservative choice, and the one
    that cannot fail on a host where CUDA was never compiled in.  Preferring a
    card when one exists would mean probing, and a probe that opens a checkpoint
    to find out is 1.5 GB of work to answer a question ``--device cuda`` asks
    directly.
    """
    device = engine_args.device
    if device == "auto":
        device = "cpu"
    return device.split(":")[0]


def _cmd_run(namespace: argparse.Namespace) -> int:
    """Load a checkpoint and generate through the C core.

    The engine does the whole chain -- GGUF read, tokenize, graph walk, sample --
    and this function is the host half the ABI's header describes: it opens a
    session, drives the loop, and prints.  Nothing here knows what Qwen3 is.

    Greedy is the default and its output is bit-identical to what it was before
    sampling existed: a request with no flags takes the `argmax` path and never
    touches the sampler.  Anything above a temperature of zero branches to the
    sampler, and the draw comes from a `random.Random(seed)` held here rather
    than from the engine -- the library has no RNG, which is what keeps it a
    pure function of `(logits, uniform)` and makes this loop reproducible.
    """
    from . import native
    from .native import Engine, EngineUnavailable

    engine_args = _args(namespace)
    device = _run_device(engine_args)
    checkpoint = engine_args.checkpoint_dir

    # `SamplingParams` validates as it is constructed -- a `top_p` outside (0, 1]
    # is a `ConfigurationError` before the checkpoint is mapped -- so the flags
    # are read through it rather than checked here, and the greedy branch below
    # is the same predicate the server would use for the same request.
    sampling = SamplingParams(
        temperature=namespace.temperature,
        top_k=namespace.top_k,
        top_p=namespace.top_p,
        min_p=namespace.min_p,
        seed=namespace.seed,
    )
    draws = random.Random(sampling.seed)

    # **Two different failures wear the same exception.**  `Engine.open` raises
    # `EngineUnavailable` both when there is no library to load and when the
    # engine refuses the checkpoint -- and the second message is the *engine's
    # own*, written for a caller with a working library.  Asking whether the
    # library exists, before opening, is what tells them apart, so the advice
    # below is only printed for the case it is advice about.
    if not native.is_available():
        raise SystemExit(
            "`pocketllm run` needs the C engine, which is not built on this host.\n"
            "Build it with `cmake -B build -S src && cmake --build build`, or point "
            "POCKETLLM_CORE_LIB at an existing libpocketllm.so.\n"
            "(`pocketllm devices` lists the *Python* backends, which are a separate "
            "thing and still stubs.)"
        )

    try:
        with Engine.open(checkpoint, device) as engine:
            tokens = engine.encode(namespace.prompt, add_special=False, parse_special=True)
            if not tokens:
                print("the prompt tokenized to nothing", file=sys.stderr)
                return 1

            # The prompt is one batch: every token in it attends to the ones
            # before it, which is what makes a prefill one pass instead of n.
            logits = engine.forward(tokens)
            print(namespace.prompt, end="", flush=True)

            for _ in range(namespace.max_tokens):
                if sampling.greedy:
                    token = Engine.argmax(logits)
                else:
                    # The transform is a separate call from the sample so the
                    # logits stay the model's own -- a caller that later wants
                    # logprobs wants the unscaled ones -- and the sampler is
                    # handed a uniform draw from this loop's generator and no
                    # other source of randomness.
                    token = Engine.sample(
                        Engine.temperature(logits, sampling.temperature),
                        draws.random(),
                        top_k=sampling.top_k or 0,
                        top_p=sampling.top_p if sampling.top_p is not None else 1.0,
                        min_p=sampling.min_p if sampling.min_p is not None else 0.0,
                    )
                piece = engine.decode([token])
                print(piece, end="", flush=True)
                # One token per call, from the position the session is holding.
                # A local counter here would be a second copy of a fact the
                # engine already owns, and the two could disagree.
                logits = engine.forward([token])
            print()
            return 0
    except EngineUnavailable as exc:
        # Reaching here means the library loaded and the engine refused
        # something -- a checkpoint it cannot read, a backend this build does
        # not have.  The message is the engine's own and is already specific, so
        # it is passed through rather than rephrased by this side.
        raise SystemExit(f"`pocketllm run` failed: {exc}") from exc


def _cmd_serve(namespace: argparse.Namespace) -> int:
    """Load a checkpoint and serve it over the OpenAI-compatible HTTP surface.

    The server is the one in :mod:`pocketllm.server.openai`, which is device-neutral and does
    not change here; what is built is the adapter it drives.  Both imports are local because
    this module's docstring promises that ``pocketllm devices`` on a phone does not pay for
    ``http.server``, and that promise is kept by importing here rather than at the top.

    Sampling is not a CLI flag on this path: the client names it per request, and the adapter
    reads it from the body.  A server-wide default would be a second place the same number
    lives.
    """
    from .server.native_backend import NativeBackend
    from .server.openai import serve

    engine_args = _args(namespace)
    device = _run_device(engine_args)
    model_id = namespace.served_model_name or os.path.basename(engine_args.checkpoint_dir)

    # Construction is where the checkpoint is opened and the template read, so a bad path or an
    # unbuilt library fails here -- with the engine's own message -- rather than on the first
    # request, after the port has been advertised as ready.
    try:
        backend = NativeBackend(engine_args, device)
    except (BackendUnavailableError, ConfigurationError) as exc:
        raise SystemExit(f"`pocketllm serve` cannot start: {exc}") from exc

    health = backend.health()
    print(
        f"{health.message}\n"
        f"serving {model_id!r} on http://{namespace.host}:{namespace.port} "
        f"(one request at a time)",
        file=sys.stderr,
        flush=True,
    )
    try:
        serve(backend, host=namespace.host, port=namespace.port, model=model_id)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    return 0


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