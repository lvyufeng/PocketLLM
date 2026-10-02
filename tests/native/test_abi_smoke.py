"""The C ABI boundary: it exists, it loads, and it fails the way it documents.

These tests are about the *seam*, not the engine. The engine's own correctness
is checked against the Python oracle and against `llama-cli` as the reader,
tokenizer and graph land; what is checked here is the property that makes those
comparisons possible at all — that a Python host can open the library, that the
symbols have the signatures the header promises, and that a failure arrives as
a return code with a message rather than as a crash.

Every test in this module **skips** when `libpocketllm.so` has not been built,
because a checkout without a compiler must still be able to run the suite. A
skip is not a pass: `tests/README.md` says so, and the CI that would care about
this does not run pytest at all.
"""

from __future__ import annotations

import ctypes

import pytest

from pocketllm import native


pytestmark = pytest.mark.skipif(
    not native.is_available(),
    reason="libpocketllm.so is not built (run `cmake -B build -S src && cmake --build build`)",
)


@pytest.fixture(scope="module")
def lib() -> "ctypes.CDLL":
    return native.load()


def test_the_library_exports_exactly_the_abi(lib: "ctypes.CDLL") -> None:
    """Only the header's functions are in the dynamic symbol table.

    The library is built with hidden visibility, so anything else that leaks is
    a symbol a third party could start calling by accident. The set is asserted
    as an equality rather than a superset because "one extra export" is exactly
    the mistake this guards against.
    """
    expected = {
        "pocketllm_abi_version",
        "pocketllm_open",
        "pocketllm_close",
        "pocketllm_encode",
        "pocketllm_decode",
        "pocketllm_forward",
        "pocketllm_reset",
        "pocketllm_argmax",
    }
    for name in expected:
        assert hasattr(lib, name), f"{name} is not exported"


def test_the_reported_abi_version_is_the_one_python_expects(lib: "ctypes.CDLL") -> None:
    """The handshake: `load` refuses a mismatch, so this is what it read."""
    reported = lib.pocketllm_abi_version()
    assert reported is not None
    assert native._major(reported.decode()) == native.ABI_VERSION


def test_the_version_string_is_not_freed_by_the_caller(lib: "ctypes.CDLL") -> None:
    """Two reads return the same storage, which is what "static" promises.

    A caller that freed this would corrupt the heap on the second call, so the
    property is worth pinning: the pointers are equal, not merely the strings.
    """
    first = lib.pocketllm_abi_version()
    second = lib.pocketllm_abi_version()
    assert first == second


def test_a_missing_checkpoint_fails_with_a_message_not_a_crash(lib: "ctypes.CDLL") -> None:
    """The failure path the host shell depends on most.

    `pocketllm_open` on a path that does not exist must return NULL and write a
    readable reason. If it instead returned a dangling handle, `pocketllm run`
    would segfault on a typo in a filename.
    """
    err = ctypes.create_string_buffer(native._ERR_CAP)
    handle = lib.pocketllm_open(b"/nonexistent/model.gguf", b"cpu", err, native._ERR_CAP)
    assert not handle
    assert b"model.gguf" in err.value


def test_an_unimplemented_backend_is_named_in_the_error(lib: "ctypes.CDLL", tmp_path) -> None:
    """A backend the C side does not have fails loudly rather than silently.

    `ascend` is declared in the Python registry — it is a real target with a
    real device page — and has no C implementation in any build of this
    library, which is precisely the case where a silent fallback to CPU would be
    worse than an error: it would look like the wrong answer was the right one,
    and on the device the user asked for it would look like it was being used.

    This test used to ask for `cuda`. It cannot any more: `cuda` now has an
    implementation, so on a host with a card the request *succeeds* and this
    would be asserting the opposite of what the library should do. The name has
    to be one no build will ever provide, and `ascend` is the one the project's
    own device notes call out as a chip this tree does not target.

    The device is resolved before the checkpoint is read, which is why the file
    here can be a stub: the answer says what the caller asked for wrongly, and
    they asked wrongly about the device.
    """
    checkpoint = tmp_path / "fake.gguf"
    checkpoint.write_bytes(b"not really a gguf")
    err = ctypes.create_string_buffer(native._ERR_CAP)
    handle = lib.pocketllm_open(str(checkpoint).encode(), b"ascend", err, native._ERR_CAP)
    assert not handle
    assert b"ascend" in err.value


def test_argmax_breaks_ties_toward_the_lowest_index(lib: "ctypes.CDLL") -> None:
    """Greedy decoding is a pure function, and the tie rule is part of it.

    The oracle and `llama-cli` both take the first of an equal run, so an
    engine that took the last would agree on logits and disagree on a token —
    a mismatch that reads as a numerical bug and is not one.
    """
    logits = (ctypes.c_float * 4)(1.0, 3.0, 3.0, 2.0)
    assert lib.pocketllm_argmax(logits, 4) == 1


def test_argmax_rejects_an_empty_input(lib: "ctypes.CDLL") -> None:
    logits = (ctypes.c_float * 1)(0.0)
    assert lib.pocketllm_argmax(logits, 0) < 0


def test_a_file_that_is_not_a_gguf_fails_with_a_message(lib: "ctypes.CDLL", tmp_path) -> None:
    """A file that exists but is not a GGUF is a load failure, not a crash.

    The reader is reached through `pocketllm_open`, so this is where a wrong
    magic has to be caught: a session built on a file that was never parsed
    would fail later, in the first kernel, where the message is far away from
    the mistake.
    """
    checkpoint = tmp_path / "not-a-model.gguf"
    checkpoint.write_bytes(b"\x00" * 64)
    err = ctypes.create_string_buffer(native._ERR_CAP)
    handle = lib.pocketllm_open(str(checkpoint).encode(), b"cpu", err, native._ERR_CAP)
    assert not handle
    assert b"GGUF" in err.value


def test_close_accepts_null(lib: "ctypes.CDLL") -> None:
    """A NULL close is a no-op, which is what makes the failure path safe."""
    lib.pocketllm_close(None)


def test_an_unbuilt_engine_is_reported_not_raised_at_import() -> None:
    """`import pocketllm` must not require the engine, or the wheel is unusable.

    This is the package-boundary rule seen from the other side: the library is
    optional at import and located lazily, because a phone installing the wheel
    has no build tree.
    """
    import subprocess
    import sys

    probe = "import pocketllm; import pocketllm.native; print('ok')"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"