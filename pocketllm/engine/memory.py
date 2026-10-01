"""Device memory: an arena that reuses by size, and the liveness it feeds on.

Two things live here, and the second is the reason the first is small.

**The arena.**  A session allocates device buffers one at a time
(``BackendSession.alloc``); the arena owns the ``alloc``/``free`` calls on the
engine's behalf so the plan below has somewhere to hand a buffer back.  Its
policy is one line: a request of ``n`` bytes is served from the pool of freed
``n``-byte buffers if one is there, and allocated otherwise.  Exact size, not
"first fit" -- a subview of a larger buffer would alias a value the plan thinks
is dead, and the plan is the thing that has to be right.

**The liveness.**  A decode step's graph is *long and thin*: every value is
produced, read once or twice, and never read again.  Allocating one buffer per
value would make the high-water mark the sum of every activation in the model.
Measuring when each value is born and when it dies -- :func:`plan_memory` -- lets
a buffer be handed back the moment its last reader has run, which turns that sum
into the width of the widest single layer.

The plan is not a guess.  It is produced by *running the arena* against a session
that counts instead of allocating, so the numbers the executor observes and the
numbers the plan reports cannot drift: there is one policy, and the plan is a
measurement of it.

Nothing here names a device runtime.  The arena calls ``alloc``/``free`` on
whatever session it was handed, which is why the same plan serves a CUDA card and
a telephone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from pocketllm.kernels.buffer import Buffer
from pocketllm.kernels.graph import Node, Value
from pocketllm.kernels.tensor import TensorDesc

__all__ = [
    "Liveness",
    "MemoryPlan",
    "plan_memory",
    "value_liveness",
    "BufferArena",
    "CountingSession",
]


@dataclass(frozen=True, slots=True)
class Liveness:
    """When one value is born, when it dies, and how many bytes it holds."""

    name: str
    desc: TensorDesc
    #: Index, in the execution order, of the node that produces the value.  ``-1``
    #: for a graph input, which is born before the first node runs.
    born: int
    #: Index of the last node that reads the value, or ``-1`` when nothing reads
    #: it.  A value nothing reads dies at its own birth.
    dies: int

    @property
    def nbytes(self) -> int:
        return self.desc.nbytes

    def frees_after(self, position: int) -> bool:
        """Whether the value's buffer can go back to the arena after ``position``.

        A graph input is never freed here: the caller owns its buffer, and
        handing it back would free memory this plan did not allocate.
        """
        if self.born < 0:
            return False
        return self.dies <= position


def value_liveness(
    env: Mapping[str, TensorDesc],
    order: Sequence[Node],
) -> dict[str, Liveness]:
    """Birth and last-read index for every value in ``env``, in execution order.

    ``order`` must list the nodes in the order they will run.  A value read by a
    node is dead after that node returns; a value read by no node at all dies
    where it is made, which is what makes a dangling scratch output cost nothing.
    """
    # A value in the environment that no node produces is a **graph input**: this
    # function is not given the input list, and it needs none, because "produced
    # by no node" and "supplied by the caller" are the same set in a graph that
    # has been through ``Graph.verify``.  That is also the whole contract: a value
    # born at -1 is never freed, since its buffer is the caller's and returning it
    # to the arena would free memory this plan did not allocate.
    born: dict[str, int] = {name: -1 for name in env}
    dies: dict[str, int] = {name: -1 for name in env}
    for position, node in enumerate(order):
        for out in node.outputs:
            born[out] = position
            dies.setdefault(out, -1)
        for arg in node.args:
            if isinstance(arg, Value) and arg.name in env:
                dies[arg.name] = max(dies[arg.name], position)
    # A value nothing reads dies where it is made.  Leaving `dies` at -1 would
    # make it look immortal to the release pass, so a dead scratch output would
    # hold its buffer for the rest of the run.
    for name, position in born.items():
        if position >= 0 and dies[name] < 0:
            dies[name] = position
    return {name: Liveness(name, env[name], born[name], dies[name]) for name in env}


class CountingSession:
    """A session that satisfies the allocator half of the ABI without a device.

    The buffers it hands out are real :class:`~pocketllm.kernels.buffer.DeviceBuffer`
    objects over a single bytearray, so an arena run against it exercises the same
    code path as one run against a real device -- only the storage is a lie.  That
    is what lets :func:`plan_memory` report the numbers the executor will actually
    see rather than an estimate of them.
    """

    def __init__(self) -> None:
        self._storage = bytearray(1)
        self.allocated = 0
        self.freed = 0

    def alloc(self, nbytes: int, *, align: int = 64) -> Buffer:
        from pocketllm.kernels.buffer import DeviceBuffer
        from pocketllm.kernels.device import Device

        self.allocated += 1
        view = memoryview(self._storage)[:0]
        return DeviceBuffer(device=Device("cpu"), nbytes=int(nbytes), alignment=align, owner=self, _view=view)

    def free(self, buffer: Buffer) -> None:
        self.freed += 1


class BufferArena:
    """Allocates device buffers and reuses them by exact size.

    One arena belongs to one session.  The executor acquires a buffer per value
    and releases it when the plan says the value is dead; ``peak_live_bytes`` is
    what the model's activations actually cost, which is the number worth printing
    and the one a memory budget is set against.
    """

    def __init__(self, session, *, align: int = 64) -> None:
        self.session = session
        self.align = int(align)
        self._free: dict[int, list[Buffer]] = {}
        #: ``id(buffer) -> buffer``.  Held by identity, not by value: two buffers
        #: of the same size are equal as dataclasses, and keying them by value
        #: would make one look like the other.  Holding the reference keeps the id
        #: stable and the storage alive while the value is live.
        self._live: dict[int, Buffer] = {}
        self._live_bytes = 0
        self.allocations = 0
        self.reuses = 0
        self.total_bytes = 0
        self.peak_live_bytes = 0

    def acquire(self, nbytes: int, *, align: int | None = None) -> Buffer:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"arena request size must be non-negative, got {nbytes}")
        pool = self._free.get(nbytes)
        if pool:
            buffer = pool.pop()
            self.reuses += 1
        else:
            buffer = self.session.alloc(nbytes, align=self.align if align is None else int(align))
            self.allocations += 1
            self.total_bytes += nbytes
        self._live[id(buffer)] = buffer
        self._live_bytes += nbytes
        if self._live_bytes > self.peak_live_bytes:
            self.peak_live_bytes = self._live_bytes
        return buffer

    def release(self, buffer: Buffer) -> None:
        """Return a buffer to the pool, or ignore one the arena never handed out.

        Releasing a foreign buffer is not an error: a caller may hand the
        executor a tensor it uploaded itself (a weight, a preallocated output),
        and the executor's cleanup path must not care which buffers it owns.
        """
        owned = self._live.pop(id(buffer), None)
        if owned is None:
            return
        size = int(owned.nbytes)
        self._live_bytes -= size
        self._free.setdefault(size, []).append(owned)

    @property
    def live_bytes(self) -> int:
        return self._live_bytes

    def stats(self) -> dict[str, int]:
        return {
            "allocations": self.allocations,
            "reuses": self.reuses,
            "total_bytes": self.total_bytes,
            "peak_live_bytes": self.peak_live_bytes,
        }

    def close(self) -> None:
        for pool in self._free.values():
            for buffer in pool:
                self.session.free(buffer)
        for buffer in self._live.values():
            self.session.free(buffer)
        self._free.clear()
        self._live.clear()
        self._live_bytes = 0


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """What one graph's execution costs, measured rather than predicted."""

    values: tuple[Liveness, ...]
    #: Distinct ``alloc`` calls the arena had to make.  With reuse working this is
    #: the number of simultaneously-live values at the busiest point, not the
    #: number of values in the graph.
    allocations: int
    reuses: int
    #: Bytes held at the busiest point; the model's activation budget.
    peak_live_bytes: int
    #: Bytes the arena asked the session for, once each -- the *distinct* sizes,
    #: not one per value, since a recycled buffer is not allocated again.
    total_bytes: int

    @property
    def no_reuse_bytes(self) -> int:
        """What the peak would have been with one buffer per value.

        Values a caller keeps (the graph outputs, and every input) are resident
        either way, so they count in both figures; the difference is the scratch
        that reuse reclaims.  This is the comparison that says whether the
        liveness pass is earning its keep.
        """
        return sum(entry.nbytes for entry in self.values)

    @property
    def saved_bytes(self) -> int:
        """How much the reuse policy saved against one buffer per value."""
        return max(0, self.no_reuse_bytes - self.peak_live_bytes)

    def for_value(self, name: str) -> Liveness:
        for entry in self.values:
            if entry.name == name:
                return entry
        raise KeyError(f"no value named {name!r} in this plan")

    def summary(self) -> str:
        return (
            f"{len(self.values)} values, {self.allocations} buffers, "
            f"peak {self.peak_live_bytes} B of {self.total_bytes} B allocated "
            f"({self.reuses} reuses, {self.saved_bytes} B saved)"
        )


def plan_memory(
    env: Mapping[str, TensorDesc],
    order: Sequence[Node],
    *,
    outputs: Sequence[str] = (),
) -> MemoryPlan:
    """Measure one execution's memory, by running the arena against a counting session.

    ``outputs`` names values the caller keeps, which the plan therefore never
    frees -- at the end of the run they must still be readable.  Every other value
    goes back to the pool the moment its last reader has run.

    Simulating rather than deriving the peak is the point: the executor will drive
    the same arena in the same order, so a number reported here and a number
    measured there are the same number.  A derivation could be wrong in a way
    nothing would notice.
    """
    live = value_liveness(env, order)
    arena = BufferArena(CountingSession())
    keep = set(outputs)

    held: dict[str, Buffer] = {}
    for position, node in enumerate(order):
        for out in node.outputs:
            held[out] = arena.acquire(live[out].nbytes)
        for name, entry in live.items():
            if name in held and name not in keep and entry.frees_after(position):
                arena.release(held.pop(name))

    for name, buffer in held.items():
        arena.release(buffer)

    return MemoryPlan(
        values=tuple(live[name] for name in sorted(live)),
        allocations=arena.allocations,
        reuses=arena.reuses,
        peak_live_bytes=arena.peak_live_bytes,
        total_bytes=arena.total_bytes,
    )


def free_after(order: Sequence[Node], env: Mapping[str, TensorDesc]) -> Callable[[int], tuple[str, ...]]:
    """A step -> the values that die there, for the executor's release pass.

    The executor cannot call :func:`plan_memory` per step -- that would re-derive
    the whole plan every node -- so the plan is computed once and inverted into
    this lookup.
    """
    live = value_liveness(env, order)
    by_position: dict[int, list[str]] = {}
    for entry in live.values():
        if entry.born < 0:
            continue
        by_position.setdefault(entry.dies, []).append(entry.name)
    return lambda position: tuple(by_position.get(position, ()))