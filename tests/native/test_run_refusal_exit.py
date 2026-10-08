"""A ``run`` refusal is a one-line ``SystemExit``, never a traceback.

Every refusal ``pocketllm run`` makes is a caller's mistake -- a device with no
runtime, a flag the delegate cannot apply, an out-of-range flag, an env var the
delegate needs, a ``--model`` that is not a checkpoint.  Each of those is a
``ConfigurationError`` raised somewhere down the call chain, and before this fix
only the *dispatch* one was converted: the branch bodies raised the rest straight
past ``_cmd_run``'s handler, so ``pocketllm run --device horizon`` without the
delegate env printed a Python traceback ending in ``ConfigurationError`` instead
of the ``pocketllm run cannot start`` message.

The check is deliberately at the ``_cmd_run`` boundary, because that is the
contract: whatever the branch raises, the command exits with a message.  A test
that called ``_run_delegate`` directly would still see the raw
``ConfigurationError`` and prove nothing -- the conversion is ``_cmd_run``'s job,
and this file is the one that holds it to that.

Hardware-free: each case is made to fail *before* a checkpoint is opened (a bad
flag, a missing env var, a missing file), so none of them needs the delegate, the
C engine, or a board.
"""

from __future__ import annotations

import pytest

from pocketllm.api import ConfigurationError
from pocketllm.cli import _cmd_run, build_parser

#: The delegate's two variables, saved and cleared around a test that needs them
#: gone.  A host that has them set (the board) would otherwise pass the guard and
#: reach the checkpoint.
_DELEGATE_ENV = ("LD_LIBRARY_PATH", "HB_DNN_USER_DEFINED_L2M_SIZES")


@pytest.fixture
def without_delegate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DELEGATE_ENV:
        monkeypatch.delenv(name, raising=False)


def _run(argv: list[str]) -> None:
    """Parse ``argv`` and run it, so the test exercises the real entry point."""
    _cmd_run(build_parser().parse_args(argv))


def _resolvable_delegate_model(tmp_path) -> str:
    """A demo-style config that resolves *to disk* without a real ``.hbm``.

    ``_resolve_model`` checks the three paths it derives before returning, so a
    bare ``checkpoint.json`` fails there and never reaches the env guard this
    case is about.  A config pointing at files that exist gets past resolution,
    which is exactly the state ``--model`` is in on the board.
    """
    hbm = tmp_path / "model.hbm"
    hbm.write_bytes(b"")
    tokenizer = tmp_path / "tok"
    tokenizer.mkdir()
    config = tmp_path / "qwen3.json"
    config.write_text(
        '{"hbm_path": "model.hbm", "tokenizer_dir": "tok", "model_type": 9}',
        encoding="utf-8",
    )
    return str(config)


# -- Finding 1: the delegate's env refusal ---------------------------------


def test_a_missing_delegate_env_is_a_system_exit_not_a_traceback(
    tmp_path, without_delegate_env
) -> None:
    """The high finding, pinned: the env guard becomes a one-line refusal.

    Before the fix this raised ``ConfigurationError`` out of ``_cmd_run`` -- a
    traceback, because the dispatch-only ``try`` never saw the branch's raise.
    The message is asserted too, so a future change that converts to
    ``SystemExit`` with an empty or wrong-prefixed string still fails.
    """
    with pytest.raises(SystemExit) as raised:
        _run(
            [
                "run",
                "--model",
                _resolvable_delegate_model(tmp_path),
                "--device",
                "horizon",
                "--prompt",
                "hi",
            ]
        )
    message = str(raised.value)
    assert "`pocketllm run` cannot start" in message
    assert "LD_LIBRARY_PATH" in message
    assert "HB_DNN_USER_DEFINED_L2M_SIZES" in message


def test_a_missing_model_file_is_a_system_exit_not_a_traceback() -> None:
    """The delegate's ``_resolve_model`` refusal is converted by the same handler.

    ``_resolve_model`` raises a ``ConfigurationError`` of its own -- a ``--model``
    that is not a readable checkpoint -- and it is reached before the env guard,
    so this covers the second refusal in the body rather than only the first.
    """
    with pytest.raises(SystemExit) as raised:
        _run(
            [
                "run",
                "--model",
                "/tmp/definitely-absent-checkpoint.json",
                "--device",
                "horizon",
                "--prompt",
                "hi",
            ]
        )
    assert "`pocketllm run` cannot start" in str(raised.value)


def test_the_no_runtime_refusal_still_reads_the_same() -> None:
    """The refusal the old handler already caught is unchanged by the move.

    ``qnn`` is a registered kind with no host runtime; it was the one converted
    before, and it must stay converted and keep its message now that the handler
    covers the whole body.
    """
    with pytest.raises(SystemExit) as raised:
        _run(["run", "--model", "checkpoint", "--device", "qnn", "--prompt", "hi"])
    message = str(raised.value)
    assert "`pocketllm run` cannot start" in message
    assert "qnn" in message


# -- Finding 2: an out-of-range sampling flag (the native path) -------------


@pytest.mark.parametrize(
    "flag,value",
    [("--top-p", "2.0"), ("--temperature", "-1.0"), ("--top-k", "-5")],
)
def test_an_out_of_range_sampling_flag_is_a_system_exit(flag: str, value: str) -> None:
    """``SamplingParams`` raises on an out-of-range flag; ``_cmd_run`` converts it.

    This is the native path (``--device`` defaults to ``auto`` -> ``cpu``) and the
    flag is validated in ``_run_native``, *before* a checkpoint is opened, so the
    case needs no ``.gguf``.  It was a traceback before the fix for the same
    reason as Finding 1: the raise happened inside the branch, past the handler.
    """
    with pytest.raises(SystemExit) as raised:
        _run(["run", "--model", "x.gguf", "--prompt", "hi", flag, value])
    message = str(raised.value)
    assert "`pocketllm run` cannot start" in message
    # The message is ``SamplingParams``'s own, so it names the offending field.
    assert flag.lstrip("-").replace("-", "_") in message


# -- the boundary: only ConfigurationError is converted ---------------------


def test_a_non_configuration_error_is_not_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real failure must not be dressed up as a tidy refusal.

    The handler catches ``ConfigurationError`` alone.  A bug or a genuine runtime
    failure -- here an ``OSError`` forced out of the dispatch -- has to propagate,
    or a crash would exit 1 with a message that reads like the caller's mistake.
    """
    import pocketllm.cli as cli

    def boom(_engine_args):
        raise OSError("something broke that is not a configuration mistake")

    monkeypatch.setattr(cli, "_run_runtime", boom)
    with pytest.raises(OSError):
        _run(["run", "--model", "checkpoint", "--prompt", "hi"])