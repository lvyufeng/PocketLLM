"""Ranks must not survive a supervisor that dies without running cleanup.

SIGKILL and native segfaults bypass the supervisor's signal handlers and
__exit__, and start_new_session=True means the ranks share no process group with
it, so nothing else brings them down.  Without the PR_SET_PDEATHSIG guard these
processes leak, and in the real backend each one holds a share of device memory.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from pocketllm.supervisor import TensorParallelConfig, TensorParallelSupervisor


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG is Linux-specific",
)


# Stands in for a rank: announces readiness, then blocks the way a worker
# spinning on a collective does.
_RANK_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    sys.stdout.write("POCKETLLM_RANK_READY rank=%d\\n" % int(os.environ["TP_RANK"]))
    sys.stdout.flush()
    while True:
        time.sleep(0.05)
    """
).strip()

# Parent that starts a supervisor and reports the rank PIDs, so the test can
# outlive it and check them.  It must not be killed before the ranks are up.
_PARENT_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from pocketllm.supervisor import TensorParallelConfig, TensorParallelSupervisor

    config = TensorParallelConfig(
        command=(sys.executable, "-c", {rank_script!r}),
        world_size=2,
        startup_timeout=60.0,
    )
    supervisor = TensorParallelSupervisor(config)
    supervisor.start()
    print("PIDS " + ",".join(str(r.process.pid) for r in supervisor.processes), flush=True)
    while True:
        time.sleep(0.05)
    """
)


def _alive(pid: int) -> bool:
    """True only for a process still running.

    Reads /proc rather than using ``os.kill(pid, 0)``, which also succeeds for a
    zombie and would report a dead rank as alive.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            # The comm field can contain spaces and parens, so split after it.
            state = handle.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def _wait_gone(pids: list[int], timeout: float = 15.0) -> list[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return []
        time.sleep(0.1)
    return [pid for pid in pids if _alive(pid)]


@pytest.mark.parametrize("kill_signal", [signal.SIGKILL, signal.SIGSEGV])
def test_ranks_die_when_supervisor_is_killed_uncleanly(kill_signal: signal.Signals) -> None:
    """An unhandleable signal to the supervisor must still take the ranks down."""
    source = _PARENT_SCRIPT.format(rank_script=_RANK_SCRIPT)
    parent = subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    rank_pids: list[int] = []
    try:
        assert parent.stdout is not None
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            line = parent.stdout.readline()
            if not line:
                break
            if line.startswith("PIDS "):
                rank_pids = [int(p) for p in line[5:].strip().split(",")]
                break
        assert rank_pids, f"supervisor never reported ranks; stderr={parent.stderr.read()!r}"
        assert all(_alive(pid) for pid in rank_pids), "ranks not running before kill"

        # The supervisor gets no chance to clean up; only the kernel-level guard
        # can bring the ranks down from here.
        parent.send_signal(kill_signal)
        parent.wait(timeout=30)

        survivors = _wait_gone(rank_pids)
        assert not survivors, f"ranks leaked after supervisor {kill_signal.name}: {survivors}"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        for pid in rank_pids:
            if _alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_graceful_stop_still_reaps_ranks() -> None:
    """The guard must not disturb the normal shutdown path."""
    config = TensorParallelConfig(
        command=(sys.executable, "-c", _RANK_SCRIPT),
        world_size=2,
        startup_timeout=60.0,
    )
    supervisor = TensorParallelSupervisor(config)
    supervisor.start()
    rank_pids = [record.process.pid for record in supervisor.processes]
    assert all(_alive(pid) for pid in rank_pids)

    supervisor.cleanup()

    survivors = _wait_gone(rank_pids)
    assert not survivors, f"ranks survived graceful stop: {survivors}"
