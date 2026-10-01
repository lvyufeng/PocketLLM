"""Deciding what runs eagerly and what a backend gets to take whole.

A backend with a graph path does not want the whole model handed to it: a CUDA
graph captures a decode step, a QNN context binary is the whole graph, and both
have ops they cannot absorb (a sampling step reads a random variate; a host-side
embedding gather has no device kernel at all).  So before execution the engine
splits the node list into **regions** and asks the backend's declared
:class:`~pocketllm.kernels.backend.GraphCapability` about each one.

The split is deliberately conservative and deliberately boring.  A region is a
maximal contiguous run of nodes whose ops are all in ``captures`` and which fits
``max_nodes``; anything else is a one-node eager region.  Two consequences worth
stating, because they are the reason this is not cleverer:

* **Adjacency is respected.**  A capturable op on either side of a sampling step
  does not get merged across it, even though the two ops could in principle be
  captured together, because replaying a capture does not re-run the sampler and
  the region's output would silently be stale.
* **Falling back is not a failure.**  A backend whose graph path is unsupported
  produces exactly one eager region and the plan says so.  :meth:`ExecutionPlan
  .captured_nodes` returning zero is the normal case for the reference backend and
  for a CPU, not an error to report.

The planner reads declarations only -- no device, no session -- so a plan for a
phone can be computed and printed on a CUDA host, which is what makes the
``ops resolve`` style of debugging work for hardware that is not present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pocketllm.kernels.backend import Backend, GraphCapability
from pocketllm.kernels.graph import Graph, GraphRegion, Node

__all__ = ["PlanRegion", "ExecutionPlan", "plan_execution"]


@dataclass(frozen=True, slots=True)
class PlanRegion:
    """One region of the plan, and the decision that put it there."""

    region: GraphRegion
    #: True when the backend's graph path will take this region whole.
    captured: bool
    #: Why it is eager, when it is.  Recorded rather than recomputed so
    #: ``--explain`` can print the reason for each region in order.
    reason: str = ""

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self.region.nodes


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """How a graph will be run on one backend."""

    regions: tuple[PlanRegion, ...]
    capability: GraphCapability

    @property
    def captured_nodes(self) -> int:
        return sum(len(r.nodes) for r in self.regions if r.captured)

    @property
    def eager_nodes(self) -> int:
        return sum(len(r.nodes) for r in self.regions if not r.captured)

    @property
    def uses_graph_path(self) -> bool:
        return self.captured_nodes > 0

    def summary(self) -> str:
        if not self.uses_graph_path:
            return f"{self.eager_nodes} nodes, all eager ({self.capability.mode.value})"
        return (
            f"{len(self.regions)} regions: {self.captured_nodes} captured, "
            f"{self.eager_nodes} eager ({self.capability.mode.value})"
        )

    def explain(self) -> str:
        lines = [self.summary()]
        for index, region in enumerate(self.regions):
            ops = ", ".join(node.op for node in region.nodes)
            mark = "capture" if region.captured else "eager"
            suffix = f" -- {region.reason}" if region.reason else ""
            lines.append(f"  region {index}: {mark} [{ops}]{suffix}")
        return "\n".join(lines)


def plan_execution(graph: Graph, backend: Backend) -> ExecutionPlan:
    """Split ``graph`` into regions for ``backend``'s declared graph path."""
    capability = backend.graph()
    nodes = list(graph.nodes)

    if not capability.supported or not capability.captures:
        if not nodes:
            return ExecutionPlan(regions=(), capability=capability)
        return ExecutionPlan(
            regions=(_eager_tuple(nodes, graph, "the backend declares no graph path"),),
            capability=capability,
        )

    regions: list[PlanRegion] = []
    run: list[Node] = []

    def flush() -> None:
        nonlocal run
        if run:
            regions.append(_captured_tuple(run, graph))
            run = []

    for node in nodes:
        if node.op not in capability.captures:
            # Not capturable at all: it breaks the run and runs on its own.
            flush()
            regions.append(_eager_tuple([node], graph, f"{node.op!r} is not in the capture set"))
            continue
        if capability.max_nodes and len(run) >= capability.max_nodes:
            # Capturable, but this region is full: close it and start another.
            # The node is *not* eager -- demoting it would be wrong, since the
            # backend declared it can take it.
            flush()
        run.append(node)
    flush()

    if not regions:
        return ExecutionPlan(regions=(), capability=capability)
    return ExecutionPlan(regions=tuple(regions), capability=capability)


def _captured_tuple(nodes: Sequence[Node], graph: Graph) -> PlanRegion:
    return PlanRegion(region=_region(nodes, graph), captured=True)


def _eager_tuple(nodes: Sequence[Node], graph: Graph, reason: str) -> PlanRegion:
    return PlanRegion(region=_region(nodes, graph), captured=False, reason=reason)


def _region(nodes: Sequence[Node], graph: Graph) -> GraphRegion:
    """The region a run of nodes forms: its nodes, and the values crossing its edge.

    The inputs are the run's arguments that are produced *outside* it -- graph
    inputs, or an earlier node's output -- and the outputs are the values
    produced inside and read outside it, which is a later node's argument or one
    of the graph's own outputs.  That is exactly what a backend needs to bind a
    captured region: what must be supplied, and what must be read back.

    The region is taken as a *contiguous slice* of the graph by position, so its
    boundary is the slice boundary and the two sets are computed from it.  A
    region's nodes are by construction adjacent and distinct, which is what makes
    the slice recoverable.
    """
    ordered = list(graph.nodes)
    start = ordered.index(nodes[0])
    stop = ordered.index(nodes[-1])
    inside_nodes = ordered[start : stop + 1]
    # Nodes given are assumed contiguous; a caller passing a scatter would get a
    # region wider than it asked for, so say so rather than silently widening.
    if [node for node in inside_nodes] != list(nodes):
        raise ValueError("a region's nodes must be a contiguous run of the graph")

    produced_here: set[str] = set()
    for node in inside_nodes:
        produced_here.update(node.outputs)

    inputs: list[str] = []
    for node in inside_nodes:
        for arg in node.args:
            name = getattr(arg, "name", None)
            if isinstance(name, str) and name not in produced_here and name not in inputs:
                inputs.append(name)

    read_later: set[str] = set()
    for later in ordered[stop + 1 :]:
        for arg in later.args:
            name = getattr(arg, "name", None)
            if isinstance(name, str):
                read_later.add(name)
    for value in graph.outputs:
        read_later.add(value.name)

    outputs = [name for name in _order_of_production(inside_nodes) if name in read_later]
    return GraphRegion(nodes=tuple(inside_nodes), inputs=tuple(inputs), outputs=tuple(outputs))


def _order_of_production(nodes: Sequence[Node]) -> list[str]:
    """Every value the run produces, in the order it is produced."""
    names: list[str] = []
    for node in nodes:
        for out in node.outputs:
            if out not in names:
                names.append(out)
    return names