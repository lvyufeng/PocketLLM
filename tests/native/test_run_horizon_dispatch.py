"""``pocketllm run --device horizon`` drives the delegate; the checks before it, host-only.

``run`` had the same gap ``serve`` did, one command over: it imported ``.native``
and drove the C engine unconditionally, so on the S600 -- where the delegate over
a ``.hbm`` is what exists -- ``run --device horizon`` refused a checkpoint it could
have run.  The dispatch that fixes it is checked without a board in
``tests/serving/test_serve_backend_selection.py`` (both commands share the seam);
what is left for *this* file is the part unique to ``run``: the *flag* policy.

The delegate's sampler is fixed when the ``.hbm`` is loaded, so ``--temperature``,
``--top-k``, ``--top-p``, ``--min-p`` and ``--seed`` cannot be applied on this
path.  The honest behaviour is to refuse a value that would have changed the
answer and accept one that spells out what the delegate does anyway -- the same
rule ``server/xlm_backend.py`` applies per request, which is why the refusal is
the *shared* function and not a second one written here.  These tests hold that
the CLI states the refusal and names the flag, without needing the delegate loaded.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm.api import EngineArgs
from pocketllm.cli import _cmd_run, _run_runtime, build_parser

_DELEGATE = "delegate"


def _args(device: str) -> EngineArgs:
    return EngineArgs(model="checkpoint.hbm", device=device)


# -- the dispatch, host-only ------------------------------------------------


def test_horizon_selects_the_delegate_for_run() -> None:
    """The property the fix is for: ``--device horizon`` is the delegate on ``run``."""
    assert _run_runtime(_args("horizon")) == _DELEGATE


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_the_c_engines_kinds_still_select_native_for_run(device: str) -> None:
    """The existing kinds are unchanged: only ``horizon`` moved to the delegate."""
    assert _run_runtime(_args(device)) == "native"


def test_run_device_parses_the_horizon_flag() -> None:
    """``run --device horizon`` parses and dispatches, end to end through argparse."""
    namespace = build_parser().parse_args(
        ["run", "--model", "checkpoint.hbm", "--prompt", "hi", "--device", "horizon"]
    )
    assert _run_runtime(EngineArgs(model=namespace.model, device=namespace.device)) == _DELEGATE


# -- the flag policy, host-only ---------------------------------------------


def _run_delegate_args(**overrides: object):
    """A ``run`` namespace for the delegate path, defaulted to the greedy case."""
    argv = ["run", "--model", "checkpoint.hbm", "--prompt", "hi", "--device", "horizon"]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "overrides",
    [
        {},                                   # nothing asked -- greedy, what the delegate does
        {"temperature": 0},                   # the value that names the default
        {"top_p": 1.0},                        # the value that disables it
        {"top_k": 0},                          # likewise
        {"temperature": 0, "top_p": 1.0, "top_k": 0},
    ],
)
def test_a_greedy_flag_is_not_refused(overrides: dict) -> None:
    """Naming the delegate's own behaviour is not a contradiction.

    These reach the refusal check and pass it; the run then fails only because
    there is no delegate on this host, which is a *different* failure and is
    asserted separately.  The point here is that the flag check itself is silent.
    """
    namespace = _run_delegate_args(**overrides)
    # The whole command, not `_run_delegate` alone: a refusal is converted to
    # `SystemExit` at the `_cmd_run` boundary, so the boundary is where the
    # message can be read.  Calling the branch directly would see the raw
    # `ConfigurationError` that `_cmd_run` is responsible for converting.
    with pytest.raises(SystemExit) as raised:
        _cmd_run(namespace)
    message = str(raised.value)
    for field in ("temperature", "top_p", "top_k"):
        assert f"--{field.replace('_', '-')}" not in message, (
            f"a greedy {field} was refused by the run path: {message}"
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 40},
        {"min_p": 0.1},
        {"seed": 1234},
    ],
)
def test_a_non_greedy_flag_is_refused_naming_the_flag(overrides: dict) -> None:
    """A value the delegate cannot apply is refused, and the message names the flag.

    Refusing is the honest answer: the delegate would generate anyway, with the
    sampler its ``generation_config.json`` fixed, and a caller who asked for
    ``--temperature 0.7`` and got the file's behaviour cannot tell from the text.
    """
    namespace = _run_delegate_args(**overrides)
    with pytest.raises(SystemExit) as raised:
        _cmd_run(namespace)
    message = str(raised.value)
    flag = f"--{next(iter(overrides)).replace('_', '-')}"
    assert flag in message
    # The remedy has to be actionable without reading the source.
    assert "Omit" in message or "Make that file" in message


def test_a_pathless_model_is_refused_before_the_delegate_opens() -> None:
    """A bad ``--model`` fails with the resolver's message, not a delegate crash.

    ``_resolve_model`` is the same resolution ``serve`` uses, so ``run`` and
    ``serve`` accept the same ``--model`` spelling -- and refuse the same missing
    one -- at the same point: before a 1 GiB ``.hbm`` is mapped.
    """
    namespace = _run_delegate_args()
    namespace.model = str(pathlib.Path("/tmp/definitely-absent.hbm"))
    with pytest.raises(SystemExit) as raised:
        _cmd_run(namespace)
    assert "cannot start" in str(raised.value)


def test_the_env_refusal_names_run_not_serve() -> None:
    """The env message names the command the user typed.

    ``run`` and ``serve`` share the guard, so without the parameter it would
    always say ``serve`` -- advice that names the wrong command is one an operator
    has to translate, which is the cost the parameter removes.
    """
    from pocketllm.cli import _require_delegate_env

    import os

    names = ("LD_LIBRARY_PATH", "HB_DNN_USER_DEFINED_L2M_SIZES")
    saved = tuple(os.environ.pop(name, None) for name in names)
    try:
        with pytest.raises(Exception) as raised:
            _require_delegate_env("run")
        assert "pocketllm run" in str(raised.value)
        assert "pocketllm serve" not in str(raised.value)
    finally:
        for name, value in zip(names, saved):
            if value is not None:
                os.environ[name] = value