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


#: The device kind served by the S600 delegate, ``libxlm.so``.
#:
#: The delegate is the one serving path this tree drives that is *not* its own C
#: engine, so it is named here rather than folded into the C engine's dispatch.
#: ``pocketllm.backends.horizon`` already claims this kind (its
#: ``device_kind`` is ``"horizon"`` and its probe finds ``libhbrt4.so`` on a
#: board), so ``--device horizon`` is a name a user already has from
#: ``pocketllm devices`` -- the selection below reuses it rather than inventing a
#: second vocabulary for the same silicon.
_DELEGATE_DEVICE = "horizon"


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


def _serve_adapter(engine_args: EngineArgs) -> str:
    """Which serving adapter ``serve`` builds: the C engine, or the S600 delegate.

    ``--device`` is the same *kind* a user already reads from ``pocketllm
    devices``, so it is what picks the adapter too -- there is no second flag to
    learn.  Two kinds map to an adapter and the rest are refused:

    * ``horizon`` -- the S600 delegate (:class:`~pocketllm.server.xlm_backend.XlmBackend`),
      which drives ``libxlm.so`` over a prebuilt ``.hbm``.
    * ``cpu``/``cuda`` (and ``auto``, which resolves to ``cpu`` for the same
      reason :func:`_run_device` says) -- this tree's own C engine
      (:class:`~pocketllm.server.native_backend.NativeBackend`).

    *Honest about the rest.*  ``qnn``, ``mps``, ``ascend`` and every other kind
    the registry knows are Python-ABI backends with no ``EngineBackend`` adapter
    at all -- ``serve`` cannot run them, and saying so by name is the point.  A
    silent fall-through to the C engine would be a server that accepts ``--device
    qnn`` and then runs the checkpoint on the CPU, which is worse than a refusal
    because the client cannot tell.  The extension seam is
    :data:`_SERVE_ADAPTERS`: a new adapter names the kinds it serves and the
    dispatch follows, rather than another ``if`` here.
    """
    device = engine_args.device.split(":")[0]
    if device == "auto":
        device = "cpu"
    if device not in _SERVE_ADAPTERS:
        raise ConfigurationError(
            f"`serve` has no adapter for --device {device!r}; it serves "
            f"{', '.join(sorted(_SERVE_ADAPTERS))} (and `auto`, which is cpu)"
        )
    return _SERVE_ADAPTERS[device]


#: The serving adapters, keyed by the device kind each one runs on.  The value is
#: the module that owns the adapter, imported only when it is selected -- the
#: same laziness :func:`_cmd_serve` already relies on so ``pocketllm devices``
#: does not pay for ``http.server``.
_SERVE_ADAPTERS: dict[str, str] = {
    "cpu": "pocketllm.server.native_backend",
    "cuda": "pocketllm.server.native_backend",
    _DELEGATE_DEVICE: "pocketllm.server.xlm_backend",
}


def _require_delegate_env() -> None:
    """Refuse to start the delegate without the two variables it cannot run without.

    Neither can be set usefully from here.  ``LD_LIBRARY_PATH`` is read by
    ``dlopen`` **once, before this process ran**, so writing it into
    ``os.environ`` now is a no-op -- measured: ``libxlm.so`` still fails to load
    its ``libopencv_world.so.409`` dependency when the variable is set in-process.
    ``HB_DNN_USER_DEFINED_L2M_SIZES`` may fare no better once ``libhbrt4`` has
    initialised.  So the honest thing is to require them **before** the adapter
    opens the ``.hbm`` and say where they come from, not to set them and pretend.

    The values are the SDK demo's own (``oellm_runtime/examples/llm_demo/
    run_llm.sh``): ``LD_LIBRARY_PATH`` must contain the SDK ``lib/`` directory,
    and the four-core Qwen3 ``.hbm``s need ``6:6:6:6``.
    """
    sdk_lib = "the SDK's lib/ directory"
    problems: list[str] = []
    if not os.environ.get("LD_LIBRARY_PATH"):
        problems.append(
            f"LD_LIBRARY_PATH must contain {sdk_lib} "
            "(the `libxlm.so` the delegate loads needs it)"
        )
    if not os.environ.get("HB_DNN_USER_DEFINED_L2M_SIZES"):
        problems.append(
            "HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6 "
            "(the L2m split the four-core Qwen3 `.hbm`s were built for)"
        )
    if not problems:
        return
    raise ConfigurationError(
        "the S600 delegate cannot start without, set before `pocketllm serve`: "
        + "; ".join(problems)
        + ". `source` the SDK's run_llm.sh, or export both and retry."
    )


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

    Which adapter is built follows ``--device`` (:func:`_serve_adapter`): the S600 delegate for
    ``horizon``, this tree's C engine for everything else it serves.  The delegate needs two
    environment variables that must be set *before this process started* -- see
    :func:`_require_delegate_env` -- so they are checked here, ahead of the ``.hbm`` load.
    """
    import importlib

    from .server.openai import serve

    engine_args = _args(namespace)
    device = _run_device(engine_args)
    model_id = namespace.served_model_name or os.path.basename(engine_args.checkpoint_dir)

    # Construction is where the checkpoint is opened and the template read, so a bad path or an
    # unbuilt library fails here -- with the engine's own message -- rather than on the first
    # request, after the port has been advertised as ready.  Adapter selection is inside the block
    # for the same reason: a device with no adapter must be a one-line refusal, not a traceback.
    try:
        adapter = _serve_adapter(engine_args)
        module = importlib.import_module(adapter)
        if adapter == "pocketllm.server.xlm_backend":
            _require_delegate_env()
            backend = module.XlmBackend(engine_args)
        else:
            backend = module.NativeBackend(engine_args, device)
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