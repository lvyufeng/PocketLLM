"""The context-length refusal, from the ABI constant to the Python type.

An over-cap prompt is a *bad request*, not a broken engine, so the engine
reports it with its own return code rather than the generic ``-1`` and the host
raises a narrower exception for it. These tests pin the two halves that make
that work and neither of which needs a checkpoint or a built library: the
constant the Python side uses is the one the C header declares, and the bridge
turns that return code into :class:`ContextLengthExceeded` and only that code
into it.

The end-to-end behaviour -- the code leaving ``pocketllm_forward`` and the HTTP
server answering 400 -- is left to the board tests in ``docs/architecture``
rather than a unit test here, because both halves of it need a real checkpoint
on a real device.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm import native


class _LibReturning:
    """The one method ``Engine.forward`` calls, returning a fixed code.

    A stand-in for the loaded library so the mapping is tested without a
    session, an arena or a device: ``forward`` reads ``self._lib`` and
    ``self._handle`` and nothing else on the failure path.
    """

    def __init__(self, code: int) -> None:
        self._code = code

    def pocketllm_forward(self, handle, tokens, n, out, cap) -> int:  # noqa: ANN001
        return self._code


def test_the_python_code_matches_the_c_header() -> None:
    """``native._ERR_CONTEXT_LENGTH`` is a literal, so it is checked against the header.

    The value is duplicated rather than parsed out of the C source on purpose --
    importing ``pocketllm.native`` must not mean reading a header -- which makes
    a drift between the two a silent wrong answer: the bridge would raise the
    generic failure for a case the engine meant as recoverable. This is the
    check that makes the duplication safe.
    """
    header = pathlib.Path(__file__).resolve().parents[2] / "src" / "include" / "pocketllm.h"
    if not header.exists():
        pytest.skip("the C header is not in this checkout")
    text = header.read_text(encoding="utf-8")
    assert f"POCKETLLM_ERR_CONTEXT_LENGTH ({native._ERR_CONTEXT_LENGTH})" in text


def test_the_context_length_code_raises_the_narrow_type() -> None:
    """The engine's own verdict, not a guess from the prompt length."""
    engine = native.Engine(_LibReturning(native._ERR_CONTEXT_LENGTH), handle=1)
    with pytest.raises(native.ContextLengthExceeded):
        engine.forward([1, 2, 3])


def test_the_narrow_type_is_still_an_engine_unavailable() -> None:
    """Subclassing ``EngineUnavailable`` keeps every existing handler working.

    A caller written before this type existed catches ``EngineUnavailable`` and
    must go on catching it; only a caller that *can* act on the narrower case
    catches ``ContextLengthExceeded`` first.
    """
    assert issubclass(native.ContextLengthExceeded, native.EngineUnavailable)


def test_a_generic_failure_is_not_the_context_length_type() -> None:
    """``-1`` is the generic failure and must not masquerade as a bad request.

    If every negative return became ``ContextLengthExceeded`` the server would
    answer 400 for a genuinely broken engine, telling the client to shrink a
    prompt that was never the problem.
    """
    engine = native.Engine(_LibReturning(-1), handle=1)
    with pytest.raises(native.EngineUnavailable) as excinfo:
        engine.forward([1, 2, 3])
    assert not isinstance(excinfo.value, native.ContextLengthExceeded)