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
import pathlib
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


#: The two runtimes a host entry point can drive, by the device kind each runs on.
#:
#: ``--device`` is the same *kind* a user already reads from ``pocketllm devices``,
#: so it is what picks the runtime too -- there is no second flag to learn.  It
#: maps to a *runtime*, not to a class, because ``run`` and ``serve`` drive the
#: same two runtimes through different seams: the C engine through ``native.py``
#: for ``run`` and through ``NativeBackend`` for ``serve``, the delegate through
#: ``XlmEngine`` for ``run`` and ``XlmBackend`` for ``serve``.  Keeping the mapping
#: to the runtime in one place is what stops the two commands from drifting into
#: two spellings of the same decision -- the gap this table closes.
#:
#: The *sampling* differences between the two are real and stay in each command;
#: only "which silicon does this name mean" is shared here.
_RUNTIMES: dict[str, str] = {
    "cpu": "native",
    "cuda": "native",
    "ascend": "native",
    _DELEGATE_DEVICE: "delegate",
}


def _dispatch_device(engine_args: EngineArgs, command: str) -> str:
    """The runtime ``command`` drives for these engine args: ``native`` or ``delegate``.

    *Honest about the rest.*  ``qnn``, ``mps`` and every other kind
    the registry knows are Python-ABI backends with no runtime a host entry point
    can drive at all -- ``run`` and ``serve`` cannot use them, and saying so by
    name is the point.  A silent fall-through to the C engine would accept
    ``--device qnn`` and run the checkpoint on the CPU (or refuse it for reasons
    about the wrong runtime), which is worse than a refusal because the caller
    cannot tell.  The extension seam is :data:`_RUNTIMES`: a new runtime names the
    kinds it serves and both commands follow.

    ``ascend`` is ``native``: the C engine carries an ascend backend, so
    ``--device ascend`` drives the same ctypes runtime as ``cpu``/``cuda`` -- the
    kernel backend underneath is the only difference, and the CLI names a
    *runtime*, not a kernel.  (It was left out of this table while the backend's
    context was thread-bound and ``serve`` could not reach it from a worker
    thread; see :func:`pocketllm.cli._cmd_serve`.)

    ``command`` is the word the refusal uses -- ``run`` or ``serve`` -- so the
    message names the command the user actually typed rather than a fixed one.
    """
    device = engine_args.device.split(":")[0]
    if device == "auto":
        # ``auto`` resolves to ``cpu`` for the reason :func:`_run_device` gives:
        # it is the choice that cannot fail on a host where CUDA was never built in.
        device = "cpu"
    if device not in _RUNTIMES:
        raise ConfigurationError(
            f"`{command}` has no runtime for --device {device!r}; it runs "
            f"{', '.join(sorted(_RUNTIMES))} (and `auto`, which is cpu)"
        )
    return _RUNTIMES[device]


def _serve_runtime(engine_args: EngineArgs) -> str:
    """Which runtime ``serve`` drives: the C engine, or the S600 delegate.

    A named wrapper rather than a bare call so the ``serve`` half of the dispatch
    reads as one thing at its call site, exactly as :func:`_run_runtime` does.
    """
    return _dispatch_device(engine_args, "serve")


def _run_runtime(engine_args: EngineArgs) -> str:
    """Which runtime ``run`` drives: the C engine, or the S600 delegate.

    ``run`` had the same gap ``serve`` did, one command over: it imported
    ``.native`` and drove the C engine unconditionally, so on the S600 -- where
    the delegate over a ``.hbm`` is what exists and the C engine is not built --
    ``run --device horizon`` refused a checkpoint it could have run.  The runtime
    is decided here and each branch below is the honest shape for that runtime:
    the C engine is a *token* loop this file owns, the delegate is a text-in/
    text-out call that owns its own decode.
    """
    return _dispatch_device(engine_args, "run")


def _require_delegate_env(command: str = "serve") -> None:
    """Refuse to start the delegate without the two variables it cannot run without.

    Neither can be set usefully from here.  ``LD_LIBRARY_PATH`` is read by
    ``dlopen`` **once, before this process ran**, so writing it into
    ``os.environ`` now is a no-op -- measured: ``libxlm.so`` still fails to load
    its ``libopencv_world.so.409`` dependency when the variable is set in-process.
    ``HB_DNN_USER_DEFINED_L2M_SIZES`` may fare no better once ``libhbrt4`` has
    initialised.  So the honest thing is to require them **before** the adapter
    opens the ``.hbm`` and say where they come from, not to set them and pretend.

    The values are the SDK demo's own (``oellm_runtime/examples/llm_demo/
    run_llm.sh``): ``LD_LIBRARY_PATH`` must contain the SDK's ``oellm_runtime/lib``
    directory (the libraries live there, *not* under a top-level ``lib/``), and the
    demo sets ``HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`` for the four-core Qwen3
    ``.hbm``s.

    **This checks that each variable is set, not that it has a particular value**,
    and the message says exactly that.  The split is the one value the delegate
    might be sensitive to, and it was measured not to be: on this board
    ``6:6:6:6``, ``0:0:0:0``, ``2:2:2:2``, ``1:1:1:1`` and ``12:12:12:12`` all
    initialise the model identically (the phase-2 ``ion_alloc`` refusal for 4B is
    at ``hbDNNInitializeFromFiles``, before the split is consulted).  Refusing
    anything but ``6:6:6:6`` would therefore assert a constraint nobody has shown
    -- the SDK's own documentation sets ``6:6:6:6`` for every model regardless of
    size, so it is a convention, not a per-checkpoint requirement.  The variable
    is still *required* rather than defaulted here because the delegate reads it
    at init and this process cannot set it in time.
    """
    sdk_lib = "the SDK's oellm_runtime/lib directory"
    problems: list[str] = []
    if not os.environ.get("LD_LIBRARY_PATH"):
        problems.append(
            f"LD_LIBRARY_PATH must contain {sdk_lib} "
            "(the `libxlm.so` the delegate loads needs it)"
        )
    if not os.environ.get("HB_DNN_USER_DEFINED_L2M_SIZES"):
        problems.append(
            "HB_DNN_USER_DEFINED_L2M_SIZES must be set to the L2m split the `.hbm` was "
            "built for (the SDK demo's run_llm.sh uses 6:6:6:6 for the four-core Qwen3 graphs)"
        )
    if not problems:
        return
    raise ConfigurationError(
        f"the S600 delegate cannot start without, set before `pocketllm {command}`: "
        + "; ".join(problems)
        + ". `source` the SDK's run_llm.sh, or export both and retry."
    )


def _cmd_run(namespace: argparse.Namespace) -> int:
    """Load a checkpoint and run one prompt through the runtime ``--device`` names.

    Two runtimes, and they are different shapes, not two implements of one
    interface: the C engine is a *token* loop this file owns (host tokenizes,
    forwards, draws), and the S600 delegate is a *text-in/text-out* call that
    owns its own decode on the BPU.  :func:`_run_runtime` picks between them by
    the device kind, the same table ``serve`` uses, so the two commands cannot
    drift into two spellings of "what does ``--device horizon`` mean".

    The *sampling* flags are the one place the two genuinely differ, and each
    branch states its own rule: the C engine applies them (see
    :func:`_run_native`), the delegate does not and refuses a non-greedy one
    (see :func:`_run_delegate`).  Keeping the difference inside the branch is
    what lets the shared dispatch stay a plain "which silicon" question.
    """
    engine_args = _args(namespace)
    # **The whole body, not just the dispatch, is inside one try.**  A
    # `ConfigurationError` is this command's own refusal -- a device with no
    # runtime, a flag the delegate cannot apply, an out-of-range flag, an env var
    # the delegate needs -- and every one of them is a caller's mistake that
    # belongs in a one-line `SystemExit`, not a traceback.  Wrapping it here
    # rather than at each raise is what keeps a *new* refusal from being added
    # later without the conversion: the branch functions raise, and this is the
    # one frame that has to catch.  `serve` does the same over its own whole
    # construction block (see :func:`_cmd_serve`), which is why the two commands
    # now fail the same way.
    #
    # Only `ConfigurationError` is caught: an `EngineUnavailable`, an `OSError`
    # and a bug are all left to propagate, so the conversion cannot hide a real
    # failure as a tidy refusal.
    try:
        runtime = _run_runtime(engine_args)
        if runtime == "delegate":
            return _run_delegate(namespace, engine_args)
        return _run_native(namespace, engine_args)
    except ConfigurationError as exc:
        raise SystemExit(f"`pocketllm run` cannot start: {exc}") from exc


def _run_native(namespace: argparse.Namespace, engine_args: EngineArgs) -> int:
    """The C engine's token loop: GGUF read, tokenize, graph walk, sample, print.

    This function is the host half the ABI's header describes: it opens a
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


def _run_delegate(namespace: argparse.Namespace, engine_args: EngineArgs) -> int:
    """Run one prompt through the S600 delegate over a prebuilt ``.hbm``.

    The delegate is text-in/text-out: it tokenizes, decodes on the BPU, applies
    its own chat template, and hands text back.  There is no logits surface, so
    there is nothing here to sample from, and ``max_tokens`` is the *model's*
    budget rather than this function's -- the delegate's sampler reads its own
    stop conditions from ``generation_config.json`` beside the tokenizer, and
    nothing in the ``xlm.h`` request struct carries a token cap.  The cap is
    therefore reported, not applied -- see below.

    **The sampling flags do not exist on this path, so a non-greedy one is
    refused by name.**  This is the same rule ``server/xlm_backend.py`` applies
    per request, and for the same reason: the delegate's sampler is fixed at open
    by the tokenizer directory's file, so honouring ``--temperature 0.7`` is not
    something this path can do, and silently generating with the file's sampler
    would answer a question the caller did not ask.  A value that names what the
    delegate does anyway (temperature 0, ``top_k 0``, ``top_p 1``) is accepted --
    spelling out the default is not a contradiction.  The check is the *shared*
    one (``_refuse_unsupported_sampling``), so ``run`` and ``serve`` cannot
    disagree about which values are refused.
    """
    from .server.xlm_backend import (
        _refuse_unsupported_sampling,
        _resolve_model,
    )
    from .xlm import XlmEngine, XlmInferenceError, XlmModelType, XlmUnavailable

    # The flag policy is settled **before** anything touches the disk: a bad flag
    # is a caller's mistake about the request, not about the checkpoint, and
    # reporting it first is what keeps the two failures from being confused.  The
    # note is built without the tokenizer directory here (it is not resolved yet),
    # which only drops the "is this session deterministic" clause from the message.
    body: dict[str, object] = {
        "temperature": namespace.temperature,
        "top_p": namespace.top_p,
        "top_k": namespace.top_k,
    }
    # `min_p` is not a delegate field and `_refuse_unsupported_sampling` does not
    # read it; a non-default one is refused separately, naming the same reason.
    if namespace.min_p not in (None, 0.0):
        raise SystemExit(
            f"`pocketllm run` cannot apply --min-p {namespace.min_p} on --device horizon: the "
            "delegate's sampler is fixed when the .hbm is loaded, so it cannot be set per call. "
            "Omit --min-p, or use a device kind whose engine samples on the host (cpu, cuda)."
        )
    if namespace.seed is not None:
        raise SystemExit(
            f"`pocketllm run` cannot apply --seed {namespace.seed} on --device horizon: the "
            "delegate has no RNG of its own to seed, and its reproducibility comes from the "
            "tokenizer directory's generation_config.json, not from a flag. Omit --seed, or make "
            "that file greedy (temperature 0, do_sample false) for a deterministic run."
        )
    refusal = _refuse_unsupported_sampling(body, _delegate_sampling_note(None))
    if refusal is not None:
        # The refusal names the *field* (``top_k``); a CLI caller typed a *flag*
        # (``--top-k``), and it is the flag the remedy has to name for the
        # message to be actionable without translating one spelling into the other.
        flag = "--" + refusal.field.replace("_", "-")
        raise SystemExit(
            f"`pocketllm run` cannot apply {flag} {refusal.requested}: {refusal.message} "
            f"Omit {flag}, or use a device kind whose engine samples on the host (cpu, cuda)."
        )

    # The same resolution `serve` uses: `--model` may be a `.hbm` or the SDK's
    # demo-style JSON config, and `--tokenizer-path`/`--config-path` override
    # what it says.  Reusing `_resolve_model` rather than re-deriving the three
    # paths is what makes `run` and `serve` accept the same `--model` spelling.
    #
    # Its `ConfigurationError` is raised uncaught on purpose: `_cmd_run` wraps
    # every branch call in one handler, so the conversion to `SystemExit` happens
    # for this refusal and for the ones below it in a single place rather than
    # three.  Catching here too would be the scattered shape this consolidation
    # removed.
    model = _resolve_model(engine_args)

    # Same gate ``serve`` applies, for the same reason: the two variables are read
    # by dlopen / libhbrt4 before this process started, so they must be *required*
    # here, ahead of the .hbm load, not set.
    _require_delegate_env("run")

    try:
        engine = XlmEngine.open(
            model_path=str(model.hbm),
            tokenizer_dir=str(model.tokenizer_dir),
            config_path=str(model.config),
            model_type=model.model_type,
        )
    except (XlmUnavailable, OSError) as exc:
        raise SystemExit(f"`pocketllm run` failed: {exc}") from exc

    try:
        text = engine.infer(namespace.prompt)
    except XlmInferenceError as exc:
        # A delegate whose graph did not run is a one-line refusal, not a traceback:
        # the same class as the load failure above, caught here because it can only
        # be observed after the session is open.
        raise SystemExit(f"`pocketllm run` failed: {exc}") from exc
    finally:
        engine.close()

    # `--max-tokens` is the model's budget on this path and the delegate does not
    # take one: the request struct in `xlm.h` carries no token cap, and the
    # delegate stops on the `generation_config.json`'s own conditions.  Saying so
    # is the honest option -- truncating the text *after* the delegate produced it
    # would be this CLI inventing a cap the model never saw, and a caller who set
    # `--max-tokens 16` and got 200 tokens should hear why.  The line is a warning
    # on stderr, so it does not contaminate the answer on stdout.
    if namespace.max_tokens != 16:
        print(
            f"`pocketllm run` on --device horizon does not cap generation: --max-tokens "
            f"{namespace.max_tokens} was not applied, and the delegate decodes until its own "
            "stop condition. The text below is the whole answer.",
            file=sys.stderr,
        )

    # Mirror the native path's contract: echo the prompt, then the completion.
    print(namespace.prompt, end="", flush=True)
    print(text)
    return 0


def _delegate_sampling_note(tokenizer_dir: os.PathLike[str] | str | None) -> str:
    """How to say, in a ``run`` refusal, what the delegate is doing instead of the flag.

    The ``serve`` adapter builds a note like this from its own model paths
    (:meth:`~pocketllm.server.xlm_backend.XlmBackend._engine_sampling_note`); this
    is the ``run`` side of the same sentence, kept here because ``run`` refuses a
    bad flag *before* it resolves the tokenizer directory.  ``None`` is that
    honest case -- the note then says what the delegate does without claiming to
    know whether *this* session is deterministic, rather than guessing.
    """
    if tokenizer_dir is None:
        behaviour = "whether this session is deterministic is decided by that file"
        return (
            "the S600 delegate builds its sampler from generation_config.json in the tokenizer "
            "directory, and that sampler is fixed when the model is loaded rather than per call: "
            f"{behaviour}. A value that would change the decode cannot be applied."
        )
    from .server.xlm_backend import _tokenizer_dir_is_deterministic

    deterministic = _tokenizer_dir_is_deterministic(pathlib.Path(tokenizer_dir))
    behaviour = (
        "the tokenizer directory's generation_config.json is deterministic, so this session "
        "decodes greedily"
        if deterministic
        else "this session samples from the tokenizer directory's generation_config.json, "
        "which is not deterministic"
    )
    return (
        "the S600 delegate builds its sampler from generation_config.json in the tokenizer "
        f"directory, and that sampler is fixed when the model is loaded rather than per call: "
        f"{behaviour}. A value that would change the decode cannot be applied."
    )


def _cmd_serve(namespace: argparse.Namespace) -> int:
    """Load a checkpoint and serve it over the OpenAI-compatible HTTP surface.

    The server is the one in :mod:`pocketllm.server.openai`, which is device-neutral and does
    not change here; what is built is the adapter it drives.  Both imports are local because
    this module's docstring promises that ``pocketllm devices`` on a phone does not pay for
    ``http.server``, and that promise is kept by importing here rather than at the top.

    Sampling is not a CLI flag on this path: the client names it per request, and the adapter
    reads it from the body.  A server-wide default would be a second place the same number
    lives.

    Which runtime is built follows ``--device`` (:func:`_serve_runtime`): the S600 delegate for
    ``horizon``, this tree's C engine for everything else it serves.  The delegate needs two
    environment variables that must be set *before this process started* -- see
    :func:`_require_delegate_env` -- so they are checked here, ahead of the ``.hbm`` load.
    """
    from .server.openai import serve

    engine_args = _args(namespace)
    device = _run_device(engine_args)
    model_id = namespace.served_model_name or os.path.basename(engine_args.checkpoint_dir)

    # Construction is where the checkpoint is opened and the template read, so a bad path or an
    # unbuilt library fails here -- with the engine's own message -- rather than on the first
    # request, after the port has been advertised as ready.  Runtime selection is inside the block
    # for the same reason: a device with no runtime must be a one-line refusal, not a traceback.
    try:
        if _serve_runtime(engine_args) == "delegate":
            _require_delegate_env()
            from .server.xlm_backend import XlmBackend

            backend = XlmBackend(engine_args)
        else:
            from .server.native_backend import NativeBackend

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