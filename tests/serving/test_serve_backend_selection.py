"""Which adapter ``pocketllm serve`` builds, checkable on any host.

``serve`` used to build :class:`~pocketllm.server.native_backend.NativeBackend`
unconditionally, so the S600 delegate adapter existed but no command could reach
it.  The selection that fixes that is pure host logic -- a device kind maps to an
adapter module -- and this checks it without a board, a ``.hbm`` or ``libxlm.so``:
the property under test is *which* adapter is named, never that it loads.

The two failure modes worth guarding are both silent if wrong.  A kind with no
adapter that fell through to the C engine would serve a checkpoint on the wrong
device and return a 200 the client cannot read as a mistake; and the delegate's
two environment variables are read by ``dlopen`` before this process starts, so
"set them and carry on" fails later with a message about a missing ``.so`` rather
than about the variable that is actually missing.
"""

from __future__ import annotations

import importlib

import pytest

from pocketllm.api import ConfigurationError, EngineArgs
from pocketllm.cli import _cmd_serve, _require_delegate_env, _serve_adapter, build_parser

_NATIVE = "pocketllm.server.native_backend"
_DELEGATE = "pocketllm.server.xlm_backend"


def _args(device: str) -> EngineArgs:
    return EngineArgs(model="checkpoint", device=device)


# -- the mapping, checkable on any host -------------------------------------


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_the_c_engines_kinds_select_the_native_adapter(device: str) -> None:
    """``auto``/``cpu``/``cuda`` are the C engine, exactly as ``run`` resolves them.

    ``auto`` lands on the native adapter because :func:`~pocketllm.cli._run_device`
    resolves it to ``cpu`` for the same reason: it is the choice that cannot fail
    on a host where CUDA was never built in.
    """
    assert _serve_adapter(_args(device)) == _NATIVE


def test_the_delegate_is_selected_by_the_horizon_kind() -> None:
    """``--device horizon`` is the S600 delegate.

    ``horizon`` is not a new name invented for this flag: it is the device kind
    :mod:`pocketllm.backends.horizon` already declares, so it is one a user reads
    from ``pocketllm devices`` and reuses, rather than a second vocabulary for the
    same silicon.
    """
    assert _serve_adapter(_args("horizon")) == _DELEGATE


@pytest.mark.parametrize("device", ["qnn", "mps", "ascend"])
def test_a_kind_with_no_adapter_is_refused_by_name(device: str) -> None:
    """Every other registered kind is refused, and the refusal names what is served.

    These are Python-ABI backends with no ``EngineBackend`` adapter; the failure
    this guards is the silent one -- falling through to the C engine would accept
    ``--device qnn`` and run the model on the CPU, which the client cannot tell.
    """
    with pytest.raises(ConfigurationError) as raised:
        _serve_adapter(_args(device))
    message = str(raised.value)
    assert device in message
    assert "horizon" in message  # the remedy names the adapters that do exist
    assert "cpu" in message


def test_the_selected_modules_export_the_adapter_they_are_named_for() -> None:
    """The table's value is a real module with the class ``_cmd_serve`` builds.

    A free check that keeps the two halves of the mapping honest: renaming the
    adapter module without updating the table would otherwise only surface on a
    board, at the point a session is about to load a 1 GiB ``.hbm``.
    """
    for module_name, attribute in ((_NATIVE, "NativeBackend"), (_DELEGATE, "XlmBackend")):
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute)


# -- the delegate's environment, checkable on any host ----------------------


def test_the_delegate_environment_is_required_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both variables must be present, and the message names the missing ones.

    They cannot be set from here usefully: ``LD_LIBRARY_PATH`` is read by
    ``dlopen`` before this process runs, so writing it now is a no-op, and
    ``HB_DNN_USER_DEFINED_L2M_SIZES`` is the L2m split fixed when the ``.hbm``
    was compiled.  Requiring them with a message is the honest failure; setting
    them would move the error to a confusing missing-``.so`` later.
    """
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("HB_DNN_USER_DEFINED_L2M_SIZES", raising=False)
    with pytest.raises(ConfigurationError) as raised:
        _require_delegate_env()
    message = str(raised.value)
    assert "LD_LIBRARY_PATH" in message
    assert "HB_DNN_USER_DEFINED_L2M_SIZES" in message
    assert "6:6:6:6" in message


def test_one_missing_variable_is_named_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the variable that is actually absent is reported.

    A message that lists both every time is one an operator learns to skim, and
    then misses the second failure when it is the real one.
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/sdk/lib")
    monkeypatch.delenv("HB_DNN_USER_DEFINED_L2M_SIZES", raising=False)
    with pytest.raises(ConfigurationError) as raised:
        _require_delegate_env()
    message = str(raised.value)
    assert "HB_DNN_USER_DEFINED_L2M_SIZES" in message
    assert "LD_LIBRARY_PATH" not in message


def test_a_fully_set_environment_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both variables present the guard is silent -- it is a gate, not a warning."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/sdk/lib")
    monkeypatch.setenv("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
    assert _require_delegate_env() is None


# -- the parse reaches the selection ----------------------------------------


def test_the_device_flag_flows_from_the_parser_to_the_selection() -> None:
    """``--device horizon`` parsed by the CLI selects the delegate.

    The seam this closes is exactly the one that was broken: the *parser* knew
    the kinds and the *engine* dispatch did not consult them for ``serve``.  This
    walks a real ``argv`` through ``build_parser`` so a rename on either side
    fails here rather than on a board.
    """
    namespace = build_parser().parse_args(
        ["serve", "--model", "checkpoint", "--device", "horizon"]
    )
    assert _serve_adapter(EngineArgs(model=namespace.model, device=namespace.device)) == _DELEGATE


def test_the_serve_command_is_still_wired() -> None:
    """``serve`` stays a dispatched command; the selection did not replace it."""
    assert _cmd_serve is not None