"""The generation watchdog: a stalling child is detected, killed, and leaves no leak.

The S600 delegate's lost-wake-up hang (#650) is a single blocking ``xlm_infer`` that never
returns, and it cannot be caught from inside the process that is in it.  These tests use a
**stand-in child** -- a plain Python process that either emits and exits, or emits once and
then sleeps forever -- because the real hang is a ~1-in-10 race on the BPU that CI has no
board to reproduce.  What they pin is the *mechanism* the mitigation rests on: the parent
sees a stream go quiet, kills the child's whole process group, and returns promptly with
``stalled`` set rather than waiting on a child that will never finish.  The board-side
confirmation that a killed delegate frees the BPU is the hardware gate, not this file.

Hardware-free: every child is ``python -c``, so none of this needs the delegate, the C
engine, or an S600.
"""

from __future__ import annotations

import pathlib
import sys
import time

from pocketllm.watchdog import WATCHDOG_STREAM_ENV, build_inner_argv, run_watchdogged


def _child(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_a_fast_child_is_not_stalled_and_streams_through(tmp_path: pathlib.Path) -> None:
    """A child that emits and exits is watched, forwarded, and left alone."""
    # Emits a prompt, then a piece, then exits -- the shape of a clean delegate run.
    child = _child(
        "import sys, time\n"
        "sys.stdout.write('PROMPT'); sys.stdout.flush()\n"
        "time.sleep(0.1)\n"
        "sys.stdout.write('ANSWER'); sys.stdout.flush()\n"
    )
    import io

    sink = io.StringIO()
    outcome = run_watchdogged(
        child,
        stream_path=tmp_path / "s.txt",
        stall_seconds=5.0,
        poll_seconds=0.05,
        stdout=sink,
    )
    assert outcome.stalled is False
    assert outcome.exit_code == 0
    assert outcome.text == "PROMPTANSWER"
    # The parent forwarded the child's output, so a caller sees a live stream.
    assert sink.getvalue() == "PROMPTANSWER"


def test_a_stalling_child_is_killed_promptly(tmp_path: pathlib.Path) -> None:
    """A child that goes quiet past the window is SIGKILLed, not waited on forever."""
    # Emits once, then sleeps far longer than the window and would never exit on its own.
    child = _child(
        "import sys, time\n"
        "sys.stdout.write('PARTIAL'); sys.stdout.flush()\n"
        "time.sleep(600)\n"
    )
    import io

    sink = io.StringIO()
    start = time.monotonic()
    outcome = run_watchdogged(
        child,
        stream_path=tmp_path / "s.txt",
        stall_seconds=1.0,
        poll_seconds=0.05,
        stdout=sink,
    )
    elapsed = time.monotonic() - start
    assert outcome.stalled is True
    # Killed with SIGKILL: the synthetic exit code is -signal.
    assert outcome.exit_code == -9
    assert outcome.killed_pids, "the killed pid must be reported"
    # The partial text before the stall is kept; that is all a lost generation produces.
    assert outcome.text == "PARTIAL"
    # Returned on the stall window, not after the child's own 600 s sleep.
    assert elapsed < 30, f"a stall must be detected promptly, took {elapsed:.1f}s"
    assert outcome.silent_for >= 1.0


def test_killing_a_stalled_child_does_not_block_the_next_run(tmp_path: pathlib.Path) -> None:
    """Recovery is the point: the watchdog is reusable after it has killed a child."""
    import io

    stalled = run_watchdogged(
        _child("import sys, time\nsys.stdout.write('x'); sys.stdout.flush()\ntime.sleep(600)\n"),
        stream_path=tmp_path / "a.txt",
        stall_seconds=1.0,
        poll_seconds=0.05,
        stdout=io.StringIO(),
    )
    assert stalled.stalled is True

    # A second, healthy child runs to completion right after -- the parent process is
    # intact and nothing from the killed run is left blocking it.
    healthy = run_watchdogged(
        _child("import sys\nsys.stdout.write('ok')\nsys.stdout.flush()\n"),
        stream_path=tmp_path / "b.txt",
        stall_seconds=5.0,
        poll_seconds=0.05,
        stdout=io.StringIO(),
    )
    assert healthy.stalled is False
    assert healthy.exit_code == 0
    assert healthy.text == "ok"


def test_the_child_is_told_to_stream(tmp_path: pathlib.Path) -> None:
    """The wrapper sets the env var the inner CLI reads to stream pieces live."""
    import io

    # A child that reports the variable it was handed, proving the wrapper set it.
    child = _child(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get(%r, 'UNSET')); sys.stdout.flush()\n" % WATCHDOG_STREAM_ENV
    )
    outcome = run_watchdogged(
        child,
        stream_path=tmp_path / "s.txt",
        stall_seconds=5.0,
        poll_seconds=0.05,
        stdout=io.StringIO(),
    )
    assert outcome.text == "1"


def test_build_inner_argv_forwards_the_flags_the_parent_parsed() -> None:
    """The child re-parses the same request, so the flags have to travel."""
    argv = build_inner_argv(
        model="/m/model.hbm",
        prompt="hello",
        device="horizon",
        tokenizer_path="/m/tok",
        config_path="/m/cfg.json",
        max_tokens=64,
        temperature=0.0,
    )
    assert argv[:4] == [sys.executable, "-m", "pocketllm", "run"]
    joined = " ".join(argv)
    assert "--model /m/model.hbm" in joined
    assert "--device horizon" in joined
    assert "--prompt hello" in joined
    assert "--tokenizer-path /m/tok" in joined
    assert "--config-path /m/cfg.json" in joined
    assert "--max-tokens 64" in joined
    # A greedy default is not forwarded: the child's own default already means greedy.
    assert "--temperature" not in joined