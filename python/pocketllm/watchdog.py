"""A stall watchdog for the S600 delegate: kill a lost-wake-up hang and recover.

The delegate has no timeout and no cancellation — :meth:`pocketllm.xlm.XlmEngine.infer`
is **one blocking ``ctypes`` call** (``xlm_infer``) that does not return until the answer
is whole, and ``libxlm.so`` exposes no way to stop it once it has started.  A
[#650 lost-wake-up hang] runs the process into that call and never comes out: measured, the
main thread joins a decode worker parked in ``std::atomic::wait`` with a null timeout, the
BPU is idle, and nothing is in flight.  No timeout on the call site can interrupt it,
because the call site is *inside* the blocked call.

So the only place a timeout can live is **outside the process**.  This module runs one
generation in a child process and watches that child's output stream: as long as the child
keeps emitting decoded pieces it is making progress; when it goes **silent for longer than
the stall window** it is hung, and the child is killed.  Killing is not a metaphor for
recovery here — a hung process holds the BPU only through its live BPU context, and
``SIGKILL`` tears that down with the process, so the *next* generation on the same cores
runs normally.  (Measured: after a hung 8B run was killed, the following run on the same
``bpu_core`` list produced a full answer.)

**Why a subprocess and not a thread.**  The other two shapes do not work:

- A **watchdog thread** in the same process cannot interrupt a blocked ``ctypes`` call.
  Python delivers no exception into a thread stuck in a C call, and ``xlm_infer`` reads no
  cancellation flag, so the thread burns to the end of the process either way.  It detects
  the hang and recovers nothing.
- A **thread running ``infer`` with a join-timeout** recovers the *caller* — the host can
  abandon the join and move on — but the abandoned thread is still parked in ``xlm_infer``
  holding the delegate's single session and its BPU context.  The session cannot be
  ``xlm_destroy``-ed while a call is inside it, so the thread is **unrecoverable**, and its
  BPU context is held for the life of the process.  That is not a mitigated failure; it is
  a leaked accelerator.

A child process is the smallest unit that can be *forcibly* torn down, which is exactly the
property the hang requires.  The cost is honest and bounded: the watchdog cannot stream the
inner CLI's banner readably and the caller sees the prompt only once the child prints it,
so there is a load-length delay before the first output — see :func:`run_watchdogged`.

**This is a mitigation, not a repair.**  The deadlock is inside the vendor ``libxlm.so`` and
is still there; a hung generation is *lost*, its partial text is all the caller gets, and
the vendor fix is the only thing that removes it.  What the watchdog buys is that the loss
is one request instead of the process.
"""

from __future__ import annotations

import codecs
import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

__all__ = [
    "WATCHDOG_STREAM_ENV",
    "WatchdogOutcome",
    "build_inner_argv",
    "run_watchdogged",
]

#: Set in the child's environment so ``pocketllm run`` streams decoded pieces to stdout as
#: the delegate produces them, instead of printing the whole answer once at the end.  The
#: watchdog detector needs the stream — a stall is *no new output*, and a child that prints
#: only at completion gives the watcher nothing to distinguish "slow" from "hung".  Unset
#: in every ordinary run, so the CLI's output is byte-for-byte what it was.
WATCHDOG_STREAM_ENV = "POCKETLLM_WATCHDOG_STREAM"

#: Seconds with no new output before the child is declared hung.  The value #650 used: a
#: long generation emits a piece far more often than this, so silence of this length is the
#: deadlock and not a slow step.
_DEFAULT_STALL_SECONDS = 25.0

#: How often the watcher looks at the child's stream.  Well under the stall window so the
#: granularity does not eat the margin.
_DEFAULT_POLL_SECONDS = 0.25


@dataclass
class WatchdogOutcome:
    """What a watched generation did.

    ``stalled`` is the whole point: ``True`` means the child went silent past the window and
    was killed, so ``exit_code`` is synthetic (``-signal``) and ``text`` is partial.  When
    ``False`` the child exited on its own and ``exit_code`` is its real status.
    """

    exit_code: int
    stalled: bool
    text: str
    #: The child's stderr, kept so a real refusal (a bad flag, a missing env var) is not
    #: swallowed by the wrapper.  Empty on the happy path.
    stderr: str = ""
    #: Seconds since the last output when the stall was declared (``stalled`` only).
    silent_for: float = 0.0
    #: Seconds the whole watched run took, load included.
    elapsed: float = 0.0
    #: Child pids that were signalled, for a test or a log to point at.
    killed_pids: list[int] = field(default_factory=list)


def build_inner_argv(
    *,
    model: str,
    prompt: str,
    device: str,
    tokenizer_path: str | None = None,
    config_path: str | None = None,
    backend: str = "auto",
    model_format: str = "auto",
    max_model_len: int | None = None,
    dtype: str | None = None,
    kv_cache_dtype: str = "auto",
    max_tokens: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    seed: int | None = None,
) -> list[str]:
    """The ``pocketllm run`` command line the child executes.

    Rebuilt from the parsed flags rather than copied from ``sys.argv`` so the same call
    works whether the CLI was reached as ``python -m pocketllm run …`` or invoked directly
    in a test.  Only non-default flags are forwarded, so the child parses the same request
    the parent did — including the sampling refusals, which then happen *in the child* and
    surface through its exit code.
    """
    argv = [
        sys.executable, "-m", "pocketllm", "run",
        "--model", model,
        "--device", device,
        "--prompt", prompt,
    ]
    if tokenizer_path:
        argv += ["--tokenizer-path", tokenizer_path]
    if config_path:
        argv += ["--config-path", config_path]
    if backend and backend != "auto":
        argv += ["--backend", backend]
    if model_format and model_format != "auto":
        argv += ["--model-format", model_format]
    if max_model_len is not None:
        argv += ["--max-model-len", str(max_model_len)]
    if dtype:
        argv += ["--dtype", dtype]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        argv += ["--kv-cache-dtype", kv_cache_dtype]
    if max_tokens is not None:
        argv += ["--max-tokens", str(max_tokens)]
    if temperature:
        argv += ["--temperature", repr(temperature)]
    if top_k is not None:
        argv += ["--top-k", str(top_k)]
    if top_p is not None:
        argv += ["--top-p", repr(top_p)]
    if min_p is not None:
        argv += ["--min-p", repr(min_p)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    return argv


def run_watchdogged(
    argv: list[str],
    *,
    stream_path: pathlib.Path,
    stall_seconds: float = _DEFAULT_STALL_SECONDS,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
    hard_cap_seconds: float | None = None,
    cwd: str | None = None,
    env: "dict[str, str] | None" = None,
    stdout=None,
) -> WatchdogOutcome:
    """Run ``argv`` as a watched child, stream its stdout, kill it if it stalls.

    The child's stdout is redirected to ``stream_path`` and this function **tails that
    file**, forwarding new bytes to ``stdout`` as they appear.  That is what makes the
    wrapper transparent: the child writes ``prompt`` then the decoded answer, the parent
    passes both through in order, and the caller sees exactly what the unwatched command
    would have printed.  The file is also the detector's sensor — a size that stops
    growing is a child that has stopped decoding.

    When the stream is silent for ``stall_seconds`` the child's **process group** is killed
    with ``SIGKILL`` (``start_new_session`` gives it its own group, so the BPU-holding
    threads die with it).  ``hard_cap_seconds`` is a second belt — a child that never writes
    at all still cannot outlive it — and defaults to ``None`` (no cap; the stall window is
    the contract).

    Returns a :class:`WatchdogOutcome`; it does not raise on a stall, so the caller decides
    what a lost generation means for its exit code.
    """
    stream_path = pathlib.Path(stream_path)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    # Start from an empty file so a stale one cannot be read as this run's opening output.
    with open(stream_path, "wb"):
        pass

    child_env = dict(os.environ if env is None else env)
    child_env[WATCHDOG_STREAM_ENV] = "1"

    out = sys.stdout if stdout is None else stdout
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    consumed = 0
    last_progress = time.monotonic()
    start = last_progress
    stalled = False
    silent_for = 0.0

    with open(stream_path, "rb") as reader:
        # The child's **stderr goes to a file, never a pipe we drain only at the end.**
        # The delegate moves its whole fd-1 banner to fd 2 (``quiet_delegate_stdout``), so
        # a long generation can write far more than a pipe buffer holds; a pipe left
        # unread would block the child on a stderr write, stop its stdout, and be killed by
        # this very watchdog as a stall it did not cause.  A file has no such ceiling.
        stderr_path = stream_path.with_suffix(stream_path.suffix + ".stderr")
        err_handle = open(stderr_path, "wb")
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=open(stream_path, "ab"),
                stderr=err_handle,
                cwd=cwd,
                env=child_env,
                start_new_session=True,
            )
            killed: list[int] = []
            try:
                while True:
                    code = proc.poll()
                    reader.seek(consumed)
                    new = reader.read()
                    if new:
                        consumed += len(new)
                        text = decoder.decode(new)
                        if text:
                            out.write(text)
                            out.flush()
                        last_progress = time.monotonic()
                    if code is not None:
                        break
                    now = time.monotonic()
                    if now - last_progress >= stall_seconds:
                        stalled = True
                        silent_for = now - last_progress
                        break
                    if hard_cap_seconds is not None and now - start >= hard_cap_seconds:
                        stalled = True
                        silent_for = now - last_progress
                        break
                    time.sleep(poll_seconds)

                if stalled:
                    # Kill the whole group: the hung decode worker is a child of the child,
                    # and a bare kill of the top pid would leave it holding the BPU.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    killed.append(proc.pid)
            finally:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover - kill already sent
                    proc.kill()
                    proc.wait()
                # Drain whatever ran after the last poll, so a clean run loses nothing that
                # landed between the final read and the exit.
                reader.seek(consumed)
                tail = decoder.decode(reader.read(), final=True)
                if tail:
                    out.write(tail)
                    out.flush()
        finally:
            err_handle.close()

        stderr_bytes = stderr_path.read_bytes() if stderr_path.exists() else b""

    exit_code = proc.returncode if proc.returncode is not None else -1
    if stalled:
        exit_code = -int(signal.SIGKILL)
    text = stream_path.read_text(encoding="utf-8", errors="replace")
    return WatchdogOutcome(
        exit_code=exit_code,
        stalled=stalled,
        text=text,
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        silent_for=silent_for,
        elapsed=time.monotonic() - start,
        killed_pids=killed,
    )