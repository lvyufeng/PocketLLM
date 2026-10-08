"""`quiet_delegate_stdout` moves the delegate's fd-1 banner, and puts fd 1 back.

``libxlm.so`` writes its runtime banner (``[UCP]:``, ``[DNN]:``, and a
``[BPU][[BPU_MONITOR]][<address>]`` line whose address changes every run) to file
descriptor **1**, so ``pocketllm run --device horizon`` used to leak eight lines
of SDK noise into the stdout a caller pipes.  The fix redirects *fd 1*, not
``sys.stdout`` -- the library is C and never touches the Python object -- so what
this file checks is a file-descriptor fact, and it can be checked without the
library: an ``os.write(1, ...)`` inside the context is what the delegate does,
and where it lands is where the delegate's banner would land.

Two properties, and the second is the one that would be silent if wrong.  The
body's fd-1 writes must reach the chosen sink (stderr by default), and fd 1 must
be the *same file* afterwards as before -- a restore that left it pointed at the
sink would send every later print to stderr, which no test of the answer's text
would notice.
"""

from __future__ import annotations

import os

import pytest

from pocketllm.xlm import quiet_delegate_stdout


def _fd_target(fd: int) -> tuple[int, int]:
    """The (device, inode) fd 1 currently refers to -- a stable identity for 'same file'."""
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino)


@pytest.fixture
def restore_fd1():
    """Put the *real* fd 1 back even if a test asserts its way out mid-swap."""
    saved = os.dup(1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def test_a_write_to_fd_1_inside_the_context_reaches_the_target(
    tmp_path, restore_fd1
) -> None:
    """The property the fix exists for: a C `write(1, ...)` lands in the sink.

    This is exactly the delegate's behavior -- it writes to fd 1 with no knowledge
    of Python -- reproduced with `os.write`, so the test needs no SDK.
    """
    sink_path = tmp_path / "sink.txt"
    sink = os.open(sink_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with quiet_delegate_stdout(sink):
        os.write(1, b"[UCP]: log level = 3\n")
    os.close(sink)  # a caller-supplied target is the caller's to close, not the context's
    assert sink_path.read_bytes() == b"[UCP]: log level = 3\n"


def test_fd_1_is_the_same_file_after_the_context(tmp_path, restore_fd1) -> None:
    """fd 1 is restored to what it was, so later prints still reach stdout."""
    before = _fd_target(1)
    sink = os.open(tmp_path / "sink.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with quiet_delegate_stdout(sink):
        os.close(sink)  # the sink is consumed for the body; closed here, not after
    assert _fd_target(1) == before, "fd 1 was not restored; later output would go to the sink"


def test_fd_1_is_restored_even_when_the_body_raises(tmp_path, restore_fd1) -> None:
    """A delegate that fails mid-load must not leave the caller's stdout redirected.

    The failure path is the one that matters: `XlmEngine.open` raising on a bad
    `.hbm` is ordinary, and a restore only on success would send every diagnostic
    after it -- including our own error message -- to the wrong place.
    """
    before = _fd_target(1)
    sink = os.open(tmp_path / "sink.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    with pytest.raises(RuntimeError):
        with quiet_delegate_stdout(sink):
            os.close(sink)
            raise RuntimeError("the delegate refused the checkpoint")
    assert _fd_target(1) == before


def test_the_default_sink_is_stderr(tmp_path, restore_fd1) -> None:
    """With no target, the banner goes to fd 2 -- preserved, not dropped.

    Stderr is where the library's own `[E]` diagnostics already go and where a
    caller expects third-party chatter, so the default is to route rather than
    discard: `pocketllm run ... > out.txt` is clean, and `2>err.txt` still has the
    banner for anyone debugging a load.
    """
    err_path = tmp_path / "stderr.txt"
    err_file = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved_err = os.dup(2)
    try:
        os.dup2(err_file, 2)
        os.close(err_file)
        with quiet_delegate_stdout():
            os.write(1, b"[BPU][[BPU_MONITOR]][12345][INFO]BPULib\n")
    finally:
        os.dup2(saved_err, 2)
        os.close(saved_err)
    assert b"BPU_MONITOR" in err_path.read_bytes()


def test_nested_contexts_restore_in_order(tmp_path, restore_fd1) -> None:
    """Two swaps unwind to the original fd 1, not to each other.

    `open` and `infer` each take the context, and a caller may hold one across the
    other; the saved-descriptor stack has to behave like a stack.
    """
    before = _fd_target(1)
    a = os.open(tmp_path / "a", os.O_WRONLY | os.O_CREAT)
    b = os.open(tmp_path / "b", os.O_WRONLY | os.O_CREAT)
    try:
        with quiet_delegate_stdout(a):
            with quiet_delegate_stdout(b):
                pass
            # Still inside the outer swap: fd 1 is not the original yet.
            assert _fd_target(1) != before
    finally:
        os.close(a)
        os.close(b)
    assert _fd_target(1) == before