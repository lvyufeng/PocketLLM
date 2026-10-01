"""Backend selection and construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pocketllm.api import (
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    UnsupportedFeatureError,
)

from . import capabilities, cli_surface
from .capabilities import runtime_capabilities
from .xing4_backend import Xing4Backend


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------


def _refuse_a_capability_the_runtime_lacks(name: str, args: EngineArgs) -> None:
    """Refuse a request for something the runtime does not declare, before anything loads.

    The one instance so far, and it is the one that motivated the declaration: the CLI accepts
    ``--max-batch-size`` and ``--enable-batching`` on every backend, and on a runtime with no
    scheduler they used to be read by nothing. A width is a request for concurrency; a runtime that
    declares ``supports_batch=False`` cannot deliver it, and the refusal costs a process start
    rather than a model load.

    The width and the flag are checked separately because they are two different asks: a width
    above 1 is a request for rows, and ``--enable-batching`` on its own is a request for the batch
    *path* even at width 1. Only the second is why the flag exists rather than the width alone.
    """
    if runtime_capabilities(name).supports_batch:
        return
    asked: list[str] = []
    if args.max_batch_size > 1:
        asked.append(f"--max-batch-size {args.max_batch_size}")
    if args.enable_batching:
        asked.append("--enable-batching")
    if not asked:
        return
    raise UnsupportedFeatureError(
        f"backend={name!r} declares supports_batch=False: it runs one request at a time, so "
        f"{' and '.join(asked)} asks for concurrency it cannot deliver. Drop "
        f"{'that' if len(asked) == 1 else 'those'}"
    )


def _refuse_options_the_runtime_does_not_read(name: str, args: EngineArgs) -> None:
    """Refuse a flag the selected runtime does not read, in the parent, before anything loads.

    The counterpart of the flag generation in :mod:`pocketllm.backends.cli_surface`, and the reason
    one flat namespace is safe: every runtime's flags are on the same command line, so
    ``--expert-pool-rows`` is offered to a launch that would not read it. Silently ignoring it is
    the failure this whole surface exists to prevent -- the run is then measured on a lever nobody
    pulled -- and the refusal can be specific about why, because the declarations say who does read
    it.

    A ``--backend-option`` key of the same name is *not* this check's business: the adapter's own
    decoder refuses an undeclared key with the runtime's own message, and it has to keep doing so for
    a key that will never have a flag.
    """
    unread = cli_surface.unread_options(name, args)
    if not unread:
        return
    flags = ", ".join(cli_surface.cli_name(option) for option in unread)
    readers = cli_surface.readers_of(unread[0])
    if readers:
        where = (
            f"it is {readers[0]}'s" if len(readers) == 1 else f"it belongs to {', '.join(readers)}"
        )
    else:
        where = "no runtime declares it"
    raise ConfigurationError(
        f"backend={name!r} does not read {flags}: {where}. A tuning option that silently does "
        "nothing is how a run ends up measured on the wrong lever"
    )


def select_backend(args: EngineArgs) -> str:
    """Select a backend without silently changing an explicit user choice.

    Both questions -- which checkpoint is this, and can this runtime serve it -- are answered from
    ``capabilities.RUNTIMES``. The two halves used to be four predicate functions plus four
    ``_reject_unsupported_*`` bodies, which is two chances per adapter to disagree about which
    checkpoints are its.
    """
    if args.backend != "auto":
        refusal = capabilities.refusal(args.backend, args)
        if refusal:
            raise UnsupportedFeatureError(refusal)
        _refuse_a_capability_the_runtime_lacks(args.backend, args)
        _refuse_options_the_runtime_does_not_read(args.backend, args)
        return args.backend
    # `auto` asks a different question than the explicit path does: not "is this provably not
    # yours" but "does this checkpoint identify you", so a checkpoint presenting no evidence falls
    # through to the generic runtime rather than being routed on the strength of a backend being
    # importable.
    for name in capabilities.AUTO_ORDER:
        if capabilities.identify(name, args).routes_here:
            _refuse_a_capability_the_runtime_lacks(name, args)
            _refuse_options_the_runtime_does_not_read(name, args)
            return name
    # There is no generic fallback any more: the runtimes that read a checkpoint in any format are
    # the ones that moved to RelicLLM, so `auto` cannot answer with one. Saying so beats routing a
    # checkpoint to a runtime that will fail at the loader with a message about a missing file.
    raise UnsupportedFeatureError(
        f"no runtime in this build serves {args.checkpoint_dir!r}; pass --backend to name one"
    )


def create_backend(args: EngineArgs, **injected: Any):
    """Construct the selected adapter.

    ``injected`` is intentionally useful for tests and embedding applications;
    production callers normally only pass ``EngineArgs``.
    """
    selected = select_backend(args)
    if selected == "xing4":
        return Xing4Backend(
            args,
            loader=injected.get("loader"),
            tokenizer=injected.get("tokenizer"),
        )
    raise BackendUnavailableError(f"unsupported backend {selected!r}")