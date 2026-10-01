"""Walking a graph, op by op, on one session.

This is the always-works path.  It resolves each node through the dispatcher,
allocates the value's buffer from the arena, calls the chosen backend's
``run``, and releases the buffers of values that have just died.  It has no
device-specific code in it and no assumption that the backend can capture
anything: a backend with no graph path runs the whole model here, and that is
correct rather than degraded.

Two properties are load-bearing.

**Resolution is per node, not per graph.**  A model on a phone does not run
entirely on the NPU: the sampling tail is on the CPU, and an embedding gather may
be too.  Resolving per node is what lets one graph span several devices, and
:meth:`ExecutionTrace.crossed_a_device` reports where it did, because a host
round-trip was worth knowing about in the old tree and is worth knowing about
here.

**Nothing is copied that did not have to be.**  A graph input's tensor is bound
by reference; only a node's outputs are allocated.  A value that nothing reads is
freed at its own birth.  The one unavoidable copy is a cross-device one, and it
is named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pocketllm.kernels.buffer import Buffer
from pocketllm.kernels.device import Device
from pocketllm.kernels.dispatch import Dispatcher, ResolvedOp
from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import Tensor, TensorDesc

from .memory import BufferArena, free_after

__all__ = ["NodeResult", "ExecutionTrace", "Executor"]


@dataclass(slots=True)
class NodeResult:
    """What one node did, kept for the trace rather than for the caller."""

    node: Node
    backend: str
    device: Device
    outputs: tuple[str, ...]
    nbytes: int


@dataclass(slots=True)
class ExecutionTrace:
    """A record of one graph run: where it ran, and what it cost.

    ``crossed_a_device`` is true when two adjacent nodes resolved to different
    backends.  The executor does not *prevent* that -- composition across devices
    is the design -- but it is the case worth printing, so it is recorded rather
    than inferred later from a profile.
    """

    nodes: list[NodeResult] = field(default_factory=list)
    peak_bytes: int = 0
    allocations: int = 0
    reuses: int = 0

    @property
    def backends_used(self) -> tuple[str, ...]:
        seen: list[str] = []
        for result in self.nodes:
            if result.backend not in seen:
                seen.append(result.backend)
        return tuple(seen)

    @property
    def crossed_a_device(self) -> bool:
        return len(self.backends_used) > 1

    def summary(self) -> str:
        where = ", ".join(self.backends_used) or "nothing"
        extra = " (crosses devices)" if self.crossed_a_device else ""
        return (
            f"{len(self.nodes)} nodes on {where}{extra}, "
            f"peak {self.peak_bytes} B, {self.allocations} allocations, {self.reuses} reuses"
        )


class Executor:
    """Runs a verified :class:`Graph` on one session, one op at a time."""

    def __init__(self, session, *, registry=None, allow_reference_fallback: bool = True) -> None:
        self.session = session
        self.arena = BufferArena(session)
        self.dispatcher = Dispatcher(
            (session.backend,),
            registry=registry,
            allow_reference_fallback=allow_reference_fallback,
        )

    # -- the run ------------------------------------------------------------

    def run(
        self,
        graph: Graph,
        inputs: Mapping[str, Tensor],
        *,
        trace: bool = False,
    ) -> "Mapping[str, Tensor] | Execution":
        """Execute ``graph`` and return the values named by ``graph.outputs``.

        ```python
        y = executor.run(graph, {"x": tensor})["y"]        # the values
        run = executor.run(graph, {"x": tensor}, trace=True)  # the values and the trace
        ```

        The graph is verified first, so a shape or op mistake is reported before
        a byte is allocated.  ``inputs`` must cover ``graph.input_names``; an
        input whose descriptor disagrees with the graph's is refused, because the
        graph's own descriptor is what every downstream shape was inferred from.
        """
        env = graph.verify(self.dispatcher.registry)
        self._check_inputs(graph, inputs)
        for name in graph.input_names:
            desc = env[name]
            given = inputs[name].desc
            if given != desc:
                raise ValueError(
                    f"input {name!r} has descriptor {given}, but the graph declares {desc}"
                )

        evict = free_after(graph.nodes, env)
        keep = {value.name for value in graph.outputs}
        held: dict[str, Tensor] = {name: inputs[name] for name in graph.input_names}
        record = ExecutionTrace() if trace else None
        order = list(graph.nodes)

        for position, node in enumerate(order):
            args = [
                held[arg.name] if isinstance(arg, Value) and arg.name in held else arg
                for arg in node.args
            ]
            resolved = self._resolve(node, args)
            out_descs = self.dispatcher.registry.get(node.op).infer(args, node.attrs)

            out_tensors = self._allocate(node, out_descs)
            try:
                produced = self.session.run(node.op, args, out=out_tensors, attrs=node.attrs)
            except Exception:
                # A refused op must not leak the buffers it was about to fill.
                for tensor in out_tensors:
                    self.arena.release(tensor.buffer)
                raise
            for name, tensor in zip(node.outputs, produced):
                held[name] = tensor

            if record is not None:
                record.nodes.append(
                    NodeResult(
                        node=node,
                        backend=resolved.backend.name,
                        device=self.session.device,
                        outputs=node.outputs,
                        nbytes=sum(t.desc.nbytes for t in produced),
                    )
                )

            for name in evict(position):
                if name in keep or name not in held:
                    continue
                self.arena.release(held.pop(name).buffer)

        outputs = {value.name: held[value.name] for value in graph.outputs}
        record = record or ExecutionTrace()
        record.peak_bytes = self.arena.peak_live_bytes
        record.allocations = self.arena.allocations
        record.reuses = self.arena.reuses
        if trace:
            return Execution(outputs=outputs, trace=record, arena=self.arena)
        return outputs

    # -- pieces -------------------------------------------------------------

    def _resolve(self, node: Node, args: Sequence[Any]) -> ResolvedOp:
        return self.dispatcher.resolve(node.op, args, self.session.device, attrs=node.attrs)

    def _allocate(self, node: Node, descs: Sequence[TensorDesc]) -> list[Tensor]:
        """One buffer per output, taken from the arena at the descriptor's size."""
        tensors: list[Tensor] = []
        for desc in descs:
            buffer = self.arena.acquire(desc.nbytes)
            tensors.append(Tensor(desc, buffer))
        if len(tensors) != len(node.outputs):
            for tensor in tensors:
                self.arena.release(tensor.buffer)
            raise ValueError(
                f"{node.label()}: op {node.op!r} produces {len(tensors)} values, "
                f"node declares {len(node.outputs)}"
            )
        return tensors

    @staticmethod
    def _check_inputs(graph: Graph, inputs: Mapping[str, Tensor]) -> None:
        missing = [name for name in graph.input_names if name not in inputs]
        extra = [name for name in inputs if name not in graph.input_names]
        if missing or extra:
            raise ValueError(
                f"graph inputs are {list(graph.input_names)}; "
                f"missing {missing}, unexpected {extra}"
            )

    def close(self) -> None:
        self.arena.close()


@dataclass(slots=True)
class Execution:
    """The result of one :meth:`Executor.run`."""

    outputs: Mapping[str, Tensor]
    trace: ExecutionTrace
    arena: BufferArena

    def output(self, name: str) -> Tensor:
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise KeyError(f"the graph produces {sorted(self.outputs)}, not {name!r}") from exc

    def __getitem__(self, name: str) -> Tensor:
        return self.output(name)

    def release(self) -> None:
        """Return every buffer this run holds to the arena.

        Called by a session that wants the memory back before the next step;
        otherwise the arena's own ``close`` reclaims it.  The trace survives,
        because it holds sizes and names rather than bytes.
        """
        for tensor in self.outputs.values():
            self.arena.release(tensor.buffer)


#: ``Buffer`` is imported for the type of ``Execution.release``'s internals; kept
#: in the module namespace so the annotation resolves without a runtime import.
_ = Buffer