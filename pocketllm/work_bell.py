"""The idle wait for a sharded V4.1 rank, moved off the device.

A worker rank's loop is a blocking read of rank 0's next message, and that read used to *be* the
collective: ``dist.broadcast_object_list`` parks a non-root rank inside a device-side NCCL poll
kernel. That kernel is not idle. It holds a full host core and keeps the card's SMs busy at a boosted
clock for every second the service has nothing to do, so an idle four-rank service reads as three
cards at 100% utilization and 0% memory bandwidth, at 350 W of the 500 the four can draw. vLLM's
workers do not do this: they park in a host-side read on a notification socket -- ``shm_broadcast``'s
``SpinCondition``, one second of ``sched_yield`` and then an indefinite ``poller.poll`` -- and enter
a collective only once the engine has put work in the queue. This module is that wait, reduced to
what a rank needs: rank 0 rings, the worker wakes, and the collective that follows carries the
message. Nothing about the message changes; only where its reader sleeps.

The channel is one unix stream socket per worker rank, in the temporary directory, named after the
group's ``MASTER_PORT`` so that two engines on one host cannot ring each other. A socket rather than
a file of flags because the wait has to be a blocking read that costs nothing: a worker parked in
``recv`` is asleep in the kernel, at 0% of a core, and its card falls to the idle clock. The ring
itself is one byte per message, and it counts rather than latches -- two rings leave two bytes -- so a
worker that is still inside one broadcast when rank 0 rings for the next one consumes the two in the
same order rank 0 sent them and stays in step.

Ordering is the whole contract, and it is why nothing here retries: every rank opens its end in
``_init_distributed``, which runs before the barrier that ends the load, and rank 0's first ring
happens after that barrier -- so a worker's listener is always bound by the time rank 0 connects. Two
things this module therefore assumes about its caller: one ring per broadcast, and the load's barrier
between binding and the first ring. A worker that woke to a ring it could not answer, or that parked
waiting for one that was never sent, would be a hang rather than an error, so both ends raise loudly
if the socket goes away instead.
"""

from __future__ import annotations

import os
import socket
import tempfile
from collections.abc import Sequence

from pocketllm.api.errors import ConfigurationError, TensorParallelSupervisorError


#: One ring, one byte. The value is never read; the count is what matters.
_RING = b"\x01"

#: Enough for a rank that is still loading while rank 0 is already talking to its peers.
_BACKLOG = 8

#: ``sun_path`` is 108 bytes including the terminator; leave room for the name below.
_MAX_PATH = 100


def bell_path(master_port: str | int, rank: int) -> str:
    """Where ``rank``'s doorbell lives, derived from the group rather than from a handshake.

    The port is the group's rendezvous port: every rank of one process group has it, no two groups
    share it, and each rank already needs it to join the group at all, so the name costs no
    message and cannot arrive late.
    """
    path = os.path.join(
        tempfile.gettempdir(), f"pocketllm-v41-bell-{master_port}-r{int(rank)}.sock"
    )
    if len(path) > _MAX_PATH:
        raise ConfigurationError(
            f"the v41 idle doorbell's socket path is {len(path)} bytes, over the {_MAX_PATH} a "
            f"unix socket allows: {path!r}. Point TMPDIR at a shorter directory, or at one whose "
            "path every rank resolves the same way."
        )
    return path


class WorkerBell:
    """The receiving end. Every rank but 0 owns one, and rank 0 rings it."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None

    def bind(self) -> None:
        """Own the socket rank 0 will ring. Called while loading, before the load's barrier."""
        if self._listener is not None:
            return
        # A file left by a rank that was killed rather than closed would fail the bind, and its
        # owner is gone, so it is this rank's to take.
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self._path)
            listener.listen(_BACKLOG)
        except OSError as exc:
            listener.close()
            raise ConfigurationError(
                f"could not open the v41 idle doorbell at {self._path!r}: {exc}. A sharded rank "
                "waits on this socket instead of inside a collective, so there is no version of "
                "this launch without it; TMPDIR is where it is looked for."
            ) from exc
        self._listener = listener

    def wait(self) -> None:
        """Block until rank 0 rings. One call per message rank 0 broadcasts."""
        connection = self._connection
        if connection is None:
            listener = self._listener
            if listener is None:
                raise ConfigurationError(
                    f"the v41 idle doorbell at {self._path!r} was never opened on this rank"
                )
            try:
                connection, _ = listener.accept()
            except OSError as exc:
                raise TensorParallelSupervisorError(
                    f"the v41 idle doorbell at {self._path!r} could not accept rank 0: {exc}"
                ) from exc
            self._connection = connection
        try:
            ring = connection.recv(len(_RING))
        except OSError as exc:
            raise TensorParallelSupervisorError(
                f"the v41 idle doorbell at {self._path!r} broke while this rank was waiting: {exc}"
            ) from exc
        if ring != _RING:
            # EOF. Rank 0 closes the socket instead of ringing it when it goes away without a
            # shutdown broadcast, and a worker that read that as a ring would enter a collective
            # with no peer in it.
            raise TensorParallelSupervisorError(
                "rank 0 closed the v41 idle doorbell without sending a shutdown; there is no rank "
                "left to serve a request from"
            )

    def close(self) -> None:
        for sock in (self._connection, self._listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._connection = None
        self._listener = None
        try:
            os.unlink(self._path)
        except OSError:
            pass


class BellRinger:
    """Rank 0's end: one connection per worker rank, opened on the first ring."""

    def __init__(self, paths: Sequence[str]) -> None:
        self._paths = list(paths)
        self._connections: list[socket.socket | None] = [None] * len(self._paths)

    def ring(self) -> None:
        """Wake every worker rank. One call per message this rank broadcasts.

        A rank that cannot be rung is raised rather than skipped: the broadcast that follows is a
        collective every rank has to enter, so a ring that did not arrive is a group that is about
        to desynchronize, and the traceback is worth more than the hang.
        """
        for index, path in enumerate(self._paths):
            connection = self._connections[index]
            if connection is None:
                connection = self._connect(path)
                self._connections[index] = connection
            try:
                connection.sendall(_RING)
            except OSError as exc:
                raise TensorParallelSupervisorError(
                    f"rank 0 could not ring the v41 idle doorbell at {path!r}: {exc}"
                ) from exc

    def _connect(self, path: str) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(path)
        except OSError as exc:
            connection.close()
            raise TensorParallelSupervisorError(
                f"rank 0 could not reach the v41 idle doorbell at {path!r}: {exc}. Every rank "
                "binds its end while it loads, before the barrier that ends the load, and this "
                "rank rings only after that barrier; a missing socket means a rank did not get "
                "there, so the ranks this process group is waiting for are not all alive."
            ) from exc
        return connection

    def close(self) -> None:
        for index, connection in enumerate(self._connections):
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
                self._connections[index] = None


#: Either end of the doorbell, which is all the backend needs to name the attribute.
Bell = WorkerBell | BellRinger


__all__ = ["Bell", "BellRinger", "WorkerBell", "bell_path"]
