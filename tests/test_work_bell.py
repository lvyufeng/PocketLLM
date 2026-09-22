"""The idle doorbell's protocol, exercised on real sockets and without a card or a process group.

What is under test here is the part of the doorbell that a hang would hide: rank 0's ring has to
reach a worker that is waiting, it has to reach one that is *not* waiting yet without being lost (the
channel counts rings rather than latching a flag), and a rank 0 that goes away has to end the wait
rather than park it forever. The waits that are supposed to block run on a daemon thread and are
joined with a timeout, so a protocol that stops delivering fails an assertion instead of hanging the
suite.

``tests/test_v41_backend.py`` covers the other half -- that ``_broadcast`` rings before it enters the
collective, and that ``_init_distributed`` opens the end of the doorbell its own rank needs.
"""

from __future__ import annotations

import os
import stat
import tempfile
import threading

import pytest

from pocketllm.api import ConfigurationError, TensorParallelSupervisorError
from pocketllm.work_bell import BellRinger, WorkerBell, bell_path


def _on_a_thread(call, *args):
    """Run ``call(*args)`` on a daemon thread; hand back the thread and what it finished with."""
    outcome: list = []

    def run() -> None:
        try:
            call(*args)
            outcome.append("returned")
        except BaseException as exc:  # noqa: BLE001 - the assertion is what reports it
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _two_waits(worker):
    def both() -> None:
        worker.wait()
        worker.wait()

    return both


def _finished(thread, outcome, seconds: float = 5.0):
    """Join, then insist the call came back rather than parking on a ring that never arrived."""
    thread.join(seconds)
    assert not thread.is_alive(), "the wait never returned"
    return outcome[0] if outcome else None


# ---------------------------------------------------------------------------- the ring


def test_a_ring_reaches_a_worker_that_is_already_waiting(tmp_path):
    path = str(tmp_path / "bell.sock")
    worker, ringer = WorkerBell(path), BellRinger([path])
    worker.bind()
    try:
        thread, outcome = _on_a_thread(worker.wait)
        ringer.ring()
        assert _finished(thread, outcome) == "returned"
    finally:
        ringer.close()
        worker.close()


def test_a_ring_that_arrives_before_the_wait_is_not_lost(tmp_path):
    """The channel counts rather than latches, which is what rank 0 ringing ahead depends on."""
    path = str(tmp_path / "bell.sock")
    worker, ringer = WorkerBell(path), BellRinger([path])
    worker.bind()
    try:
        ringer.ring()
        thread, outcome = _on_a_thread(worker.wait)
        assert _finished(thread, outcome) == "returned"
    finally:
        ringer.close()
        worker.close()


def test_two_rings_are_two_waits(tmp_path):
    """A worker still inside one broadcast when rank 0 rings for the next stays in step."""
    path = str(tmp_path / "bell.sock")
    worker, ringer = WorkerBell(path), BellRinger([path])
    worker.bind()
    try:
        ringer.ring()
        ringer.ring()
        thread, outcome = _on_a_thread(_two_waits(worker))
        assert _finished(thread, outcome) == "returned"
        # Both were consumed, so the next wait is a wait again rather than a stale byte.
        thread, _ = _on_a_thread(worker.wait)
        thread.join(0.5)
        assert thread.is_alive()
    finally:
        ringer.close()
        worker.close()


def test_one_ring_reaches_every_rank_rank_zero_sends_to(tmp_path):
    paths = [str(tmp_path / f"bell-r{rank}.sock") for rank in (1, 2, 3)]
    workers = [WorkerBell(path) for path in paths]
    for worker in workers:
        worker.bind()
    ringer = BellRinger(paths)
    try:
        waiting = [_on_a_thread(worker.wait) for worker in workers]
        ringer.ring()
        for thread, outcome in waiting:
            assert _finished(thread, outcome) == "returned"
    finally:
        ringer.close()
        for worker in workers:
            worker.close()


# ---------------------------------------------------------------------------- the ways out


def test_rank_zero_going_away_ends_the_wait_rather_than_parking_it(tmp_path):
    """EOF is not a ring: a worker that read it as one would enter a collective nobody is in."""
    path = str(tmp_path / "bell.sock")
    worker, ringer = WorkerBell(path), BellRinger([path])
    worker.bind()
    try:
        ringer.ring()
        thread, outcome = _on_a_thread(worker.wait)
        _finished(thread, outcome)
        ringer.close()

        thread, outcome = _on_a_thread(worker.wait)
        assert isinstance(_finished(thread, outcome), TensorParallelSupervisorError)
    finally:
        worker.close()


def test_a_rank_that_never_opened_its_end_says_so(tmp_path):
    with pytest.raises(ConfigurationError, match="never opened"):
        WorkerBell(str(tmp_path / "bell.sock")).wait()


def test_a_socket_left_by_a_killed_rank_does_not_stop_the_bind(tmp_path):
    """The owner of that file is gone, so taking it is not the next rank's mistake to report."""
    path = str(tmp_path / "bell.sock")
    with open(path, "wb"):
        pass
    worker = WorkerBell(path)
    try:
        worker.bind()  # not an error: the bind replaces it
        assert stat.S_ISSOCK(os.stat(path).st_mode)
    finally:
        worker.close()


def test_close_is_idempotent_and_takes_the_socket_with_it(tmp_path):
    path = str(tmp_path / "bell.sock")
    worker = WorkerBell(path)
    worker.bind()
    assert stat.S_ISSOCK(os.stat(path).st_mode)
    worker.close()
    worker.close()
    assert not os.path.exists(path)

    ringer = BellRinger([path])
    ringer.close()
    ringer.close()


def test_rank_zero_raises_rather_than_skipping_a_rank_it_cannot_reach(tmp_path):
    """A ring that did not arrive is a group about to desynchronize, not a rank to leave out."""
    ringer = BellRinger([str(tmp_path / "bell.sock")])
    try:
        for _ in range(2):  # once for the connect, once for the cached connection
            with pytest.raises(TensorParallelSupervisorError, match="could not reach"):
                ringer.ring()
    finally:
        ringer.close()


# ---------------------------------------------------------------------------- the name


def test_the_name_separates_two_groups_and_the_ranks_inside_one():
    assert bell_path("29500", 1) != bell_path("29500", 2)
    assert bell_path("29500", 1) != bell_path("29501", 1)
    assert bell_path("29500", 1) == bell_path(29500, 1)
    assert bell_path("29500", 1).endswith("r1.sock")


def test_a_name_too_long_for_a_unix_socket_is_refused(monkeypatch):
    """``sun_path`` is 108 bytes, and a bind that dies on the length says nothing about why."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/" + "d" * 200)
    with pytest.raises(ConfigurationError, match="Point TMPDIR"):
        bell_path("29500", 1)
