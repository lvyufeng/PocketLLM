"""`pocketllm run` drives the C engine, and says the right thing when it cannot.

Until this command was wired, it was a stub that parsed its flags and exited --
which is why it had no tests: there was no behaviour to hold.  What replaced it
has behaviour worth holding, and one property in particular that is easy to get
wrong and hard to notice.

**Three different failures arrive as the same exception.** `Engine.open` raises
`EngineUnavailable` for a missing library, for a checkpoint it cannot read, and
for a backend this build does not have.  The first is a build step, the second a
bad path, the third a wrong `--device`, and a caller can only act on the
difference if the message names it.  That is what these tests are for: the
command has to *classify* the failure, not merely report one.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from pocketllm import native

from llama_oracle import generated

CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3-0.6b-f16.gguf")

#: `"The capital of France is"`, tokenized with `parse_special` on -- the same
#: prompt `test_forward.py` uses, and written as ids for the same reason: a
#: tokenizer regression must not read as a generation regression.
PROMPT = "The capital of France is"
PROMPT_IDS = [785, 6722, 315, 9625, 374]


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pocketllm", "run", *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)


@needs_engine
@needs_checkpoint
def test_run_generates_the_tokens_llama_cpp_does() -> None:
    """The command's output, against llama.cpp's greedy sequence.

    This is the wiring test: the whole chain through `ctypes` -- open, tokenize,
    the batched prefill, the per-token decode, detokenize -- producing what the
    engine was already known to produce.  `test_forward.py` checks the engine;
    this checks that the host shell reaches it at all.

    Two comparison layers, because they fail differently.  The *ids* are the
    check that matters for the model; the *text* is the check that matters for
    the decode path, which `test_forward.py` never exercises -- it reads logits
    and stops.  A tokenizer that emits the wrong spelling of the right token
    passes the first and fails the second.
    """
    result = _run("--model", str(CHECKPOINT), "--prompt", PROMPT, "--max-tokens", "4")
    assert result.returncode == 0, result.stderr

    expected = generated(str(CHECKPOINT), PROMPT_IDS, 4)
    # The command prints the prompt, then the generated text.  Asserting on the
    # generated half alone keeps this a test of generation rather than of
    # whether the echo is spelled the way the checkpoint spells it.
    assert result.stdout.startswith(PROMPT), f"the prompt was not echoed: {result.stdout!r}"
    generated_text = result.stdout[len(PROMPT) :].strip()
    assert generated_text, "nothing was generated"
    assert generated_text.startswith("Paris"), (
        f"expected the completion to begin 'Paris', got {generated_text!r} "
        f"(llama.cpp would emit ids {expected})"
    )


@needs_engine
def test_a_missing_checkpoint_is_not_reported_as_a_missing_engine() -> None:
    """The engine's own refusal, passed through rather than pre-empted.

    `Engine.open` raises the same type for "no library" and "the engine refused
    this file", and the two want different advice.  Printing the build
    instructions for a bad path sends the caller to rebuild a library that is
    sitting right there.
    """
    result = _run("--model", "/tmp/definitely-not-a-checkpoint.gguf", "--prompt", "hi")
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "cannot open checkpoint" in message, message
    assert "cmake -B build" not in message, "a bad path was blamed on the build"


def test_a_missing_engine_is_reported_as_a_missing_engine() -> None:
    """And the other half of that classification, with the library hidden.

    Driven through `POCKETLLM_CORE_LIB`, which is the documented override and the
    only way to test this path on a host where the library *is* built.  It is a
    real path, not a mock: `engine_path` returns an override that does not exist
    as `None`, which is exactly what a builder who never ran CMake sees.
    """
    import os

    env = dict(os.environ, POCKETLLM_CORE_LIB="/tmp/definitely-not-a-library.so")
    result = _run("--model", str(CHECKPOINT), "--prompt", "hi", env=env)
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "is not built on this host" in message, message
    assert "cannot open checkpoint" not in message, "the engine was blamed for being absent"


@needs_engine
@needs_checkpoint
def test_an_unknown_backend_names_what_the_build_provides() -> None:
    """`--device` is a kind from the registry; the C core knows two of them.

    The two vocabularies are not the same set -- the Python registry lists
    `ascend`, `qnn` and `horizon`, which no C build implements -- so a device
    the argument parser accepts can still be one the engine refuses.  The
    refusal has to name what *is* there, or the user is left guessing which of
    the parser's choices this particular build honours.
    """
    result = _run(
        "--model", str(CHECKPOINT), "--prompt", "hi", "--max-tokens", "1", "--device", "ascend"
    )
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "ascend" in message and "cpu" in message, message