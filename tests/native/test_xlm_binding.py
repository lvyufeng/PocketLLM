"""The `libxlm.so` binding: what holds on every host, and what needs the board.

The binding has two failure surfaces and they need different tests.  The
*transcription* — the struct layouts ctypes reconstructs from the SDK header —
can be checked anywhere, and a slip in it is the specific failure that produces
a crash inside `xlm_init` rather than an exception, so it is worth a test that
runs on a laptop.  The *behaviour* — that a session opens, answers, and starts a
second answer clean — needs the delegate, the SDK, and the `.hbm`, and skips
without them.

The skip is the same shape `tests/native/` already uses for the C engine:
``pytest.mark.skipif`` on a filesystem probe, never an import that could fail.
"""

from __future__ import annotations

import ctypes
import pathlib

import pytest

from pocketllm import xlm

#: Where the D-Robotics installer puts the SDK, and what a session needs from it.
_SDK = pathlib.Path.home() / "llm_sdk" / "D-Robotics_LLM_S600_1.0.2_SDK" / "oellm_runtime"
_HBM = _SDK / "model/Qwen3_0.6B/Qwen3-0.6B_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm"
_CONFIG = _SDK / "examples/llm_demo/qwen3_0.6b_config.json"
_TOKENIZER = _SDK / "configs/Qwen3_config"

needs_delegate = pytest.mark.skipif(
    not xlm.is_available(), reason="libxlm.so is not installed on this host"
)
needs_checkpoint = pytest.mark.skipif(
    not _HBM.is_file(), reason=f"no .hbm at {_HBM}"
)


# -- the transcription, checkable on any host -------------------------------


def test_the_struct_sizes_match_the_sdk_header() -> None:
    """Each struct is the size the board's compiler reported for the header.

    These numbers come from ``offsetof``/``sizeof`` run on the S600 against
    `xlm.h`, not from reading the header by eye.  Importing the module already
    asserts them; this names the check so a failure points at the struct rather
    than at collection time.
    """
    assert ctypes.sizeof(xlm.CommonParams) == 176
    assert ctypes.sizeof(xlm.Sampling) == 44
    assert ctypes.sizeof(xlm.LmRequest) == 144
    assert ctypes.sizeof(xlm.Input) == 16
    assert ctypes.sizeof(xlm.Result) == 104
    assert ctypes.sizeof(xlm.Performance) == 88


def test_the_request_prompt_lands_at_the_headers_offset() -> None:
    """`prompt` is a `c_char_p` at byte 32 of `xlm_lm_request_t`.

    The union member is written through its computed address, and this is the
    check that the address is the one the delegate reads.  A union that is four
    bytes off does not fail to compile; it hands the SDK a pointer into the
    middle of another field.
    """
    request = xlm.LmRequest()
    ctypes.memset(ctypes.byref(request), 0, ctypes.sizeof(request))
    held = xlm._set_pointer(request, "_payload", "prompt", "hello")
    assert held == [b"hello"]

    offset = xlm.LmRequest._payload.offset + xlm._RequestUnion.prompt.offset
    assert offset == 32, "the prompt offset moved; re-check the header's layout"
    read_back = ctypes.cast(
        ctypes.addressof(request) + offset, ctypes.POINTER(ctypes.c_char_p)
    )[0]
    assert read_back == b"hello"


def test_a_missing_library_is_reported_not_raised_at_import() -> None:
    """Importing the module never needs the SDK; loading it says what to install.

    The property under test is the one the module docstring promises: a host
    without the board imports `pocketllm.xlm` fine, and only :func:`load`
    refuses — with a message naming the environment variable and the SDK path.
    """
    if xlm.is_available():
        pytest.skip("this host has libxlm.so, so the absent case cannot be exercised")

    with pytest.raises(xlm.XlmUnavailable) as raised:
        xlm.load()
    assert "POCKETLLM_XLM_LIB" in str(raised.value)


def test_model_type_qwen3_is_nine() -> None:
    """The one enum value a session cannot guess wrong.

    `xlm_model_type` is not dense -- PI0 is 8 and QWEN3 is 9 -- so a guestimate
    that assumed the natural ordering would name the wrong model family and the
    delegate would load the `.hbm` with the wrong head.
    """
    assert xlm.XlmModelType.QWEN3 == 9
    assert xlm.STATE_END == 1


# -- the behaviour, needs the board -----------------------------------------


@needs_delegate
@needs_checkpoint
def test_a_session_answers_and_starts_a_clean_second_turn() -> None:
    """Open, answer, and answer again without inheriting the first text.

    **The second answer is the assertion that matters.**  The delegate keeps
    conversation state between calls, so an `infer` that forgot to set
    ``new_chat`` would append to the previous turn, and one that failed to reset
    the accumulator would return the concatenation of every answer so far.
    Either is silent: the text is fluent and wrong.  Two unrelated prompts whose
    answers share no substring is what catches both.
    """
    engine = xlm.XlmEngine.open(
        model_path=str(_HBM),
        tokenizer_dir=str(_TOKENIZER),
        config_path=str(_CONFIG),
    )
    try:
        first = engine.infer("What is 2+2? Answer in one word.")
        second = engine.infer("Name the largest planet.")
    finally:
        engine.close()

    assert first.strip(), "the delegate returned nothing on the first prompt"
    assert second.strip(), "the delegate returned nothing on the second prompt"
    # Independent questions: neither answer may contain the other's subject.  A
    # naive substring overlap would be too strict (both mention "the"), so the
    # check is on the distinguishing content words.
    assert "four" in first.lower()
    assert "jupiter" in second.lower()
    assert "four" not in second.lower(), "the second answer inherited the first"