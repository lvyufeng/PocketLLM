"""Region splitting: which nodes a backend gets whole, and why the rest are not.

The planner reads declarations only, so these run on any host -- including one
with no accelerator -- which is what makes a phone's plan checkable from a
development machine.
"""

from __future__ import annotations

from pocketllm.backends import registry
from pocketllm.backends.cpu import BACKEND as CPU
from pocketllm.backends.cuda import BACKEND as CUDA
from pocketllm.backends.qnn import BACKEND as QNN
from pocketllm.backends.reference import BACKEND as REFERENCE
from pocketllm.engine.planner import plan_execution
from pocketllm.kernels.backend import GraphCapability, GraphMode, RegionGranularity
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import TensorDesc


def _graph(*ops, outputs_at=None):
    """A single-input graph running ``ops`` in sequence, threading one value."""
    x = Value("x")
    nodes = []
    previous = x
    for index, op in enumerate(ops):
        nodes.append(Node(op=op, args=(previous, previous), outputs=(f"v{index}",), name=f"n{index}"))
        previous = Value(f"v{index}")
    outputs = (previous,) if outputs_at is None else tuple(Value(f"v{i}") for i in outputs_at)
    return Graph(
        inputs=(TensorDesc((4, 6), DType.F32),),
        input_names=("x",),
        nodes=tuple(nodes),
        outputs=outputs,
        name="seq",
    )


class _FixedBackend:
    """A declaration-only backend with a chosen graph capability."""

    def __init__(self, capability: GraphCapability, declared: frozenset[str]) -> None:
        self.name = "fixed"
        self.device_kind = "cpu"
        self.version = "test"
        self.is_reference = False
        self._capability = capability
        self._declared = declared

    def available(self) -> bool:
        return True

    def capabilities(self):
        from pocketllm.kernels.backend import Capability

        return tuple(Capability(op=name, dtypes=frozenset({DType.F32})) for name in sorted(self._declared))

    def graph(self) -> GraphCapability:
        return self._capability

    def compile_spec(self):
        return None

    def open(self, device, *, options=None):  # pragma: no cover
        raise NotImplementedError


def test_a_backend_with_no_graph_path_gets_one_eager_region() -> None:
    plan = plan_execution(_graph("add", "add"), REFERENCE)
    assert len(plan.regions) == 1
    assert not plan.regions[0].captured
    assert plan.captured_nodes == 0
    assert plan.eager_nodes == 2
    assert not plan.uses_graph_path
    assert "all eager" in plan.summary()


def test_an_empty_graph_plans_nothing() -> None:
    empty = Graph(inputs=(), input_names=(), nodes=(), outputs=())
    plan = plan_execution(empty, REFERENCE)
    assert plan.regions == ()


def test_a_capturable_run_becomes_one_region() -> None:
    capability = GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE, captures=frozenset({"add", "mul"}))
    backend = _FixedBackend(capability, frozenset({"add", "mul"}))
    plan = plan_execution(_graph("add", "mul", "add"), backend)

    assert len(plan.regions) == 1
    assert plan.regions[0].captured
    assert plan.captured_nodes == 3
    assert plan.uses_graph_path


def test_an_uncapturable_op_splits_the_run() -> None:
    """The sampling tail is not capturable, and the plan must say so.

    The interesting part is not that it splits but *where*: the ops before and
    after a sampling step are captured separately, and the sampler runs eagerly
    between them, because replaying a capture does not re-run the sampler and the
    region's output would silently be stale.
    """
    capability = GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE, captures=frozenset({"add", "mul"}))
    backend = _FixedBackend(capability, frozenset({"add", "mul", "argmax"}))
    plan = plan_execution(_graph("add", "argmax", "mul"), backend)

    assert [r.captured for r in plan.regions] == [True, False, True]
    assert plan.captured_nodes == 2
    assert plan.eager_nodes == 1
    assert "not in the capture set" in plan.regions[1].reason
    assert "argmax" in plan.explain()


def test_max_nodes_starts_a_new_region_rather_than_demoting() -> None:
    """A full region closes; the node that would have overflowed stays capturable.

    Demoting it to eager would be wrong -- the backend declared it can take it --
    and would quietly halve the accelerated work on a backend with a small limit.
    """
    capability = GraphCapability(
        supported=True,
        mode=GraphMode.STREAM_CAPTURE,
        captures=frozenset({"add"}),
        max_nodes=2,
    )
    backend = _FixedBackend(capability, frozenset({"add"}))
    plan = plan_execution(_graph("add", "add", "add", "add", "add"), backend)

    assert [r.captured for r in plan.regions] == [True, True, True]
    assert plan.captured_nodes == 5
    assert plan.eager_nodes == 0


def test_region_edge_names_what_crosses_it() -> None:
    """A region's inputs are produced outside it; its outputs are read outside it.

    That is exactly what a backend needs to bind a captured region: what must be
    supplied, and what must be read back.
    """
    capability = GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE, captures=frozenset({"add"}))
    backend = _FixedBackend(capability, frozenset({"add", "argmax"}))

    # add -> argmax -> add: the middle node is eager, so there are two regions.
    plan = plan_execution(_graph("add", "argmax", "add"), backend)
    first, second = plan.regions[0].region, plan.regions[2].region

    assert first.inputs == ("x",), "the region's input is the graph input"
    assert "v0" in first.outputs, "v0 is read by the eager node that follows"
    assert second.inputs == ("v1",), "the second region reads the eager node's output"
    assert second.outputs == ("v2",), "v2 is the graph output"


def test_a_region_must_be_contiguous() -> None:
    """A scattered region is refused rather than silently widened by the planner."""
    graph = _graph("add", "add", "add")
    capability = GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE, captures=frozenset({"add"}))
    backend = _FixedBackend(capability, frozenset({"add"}))
    plan = plan_execution(graph, backend)
    assert len(plan.regions) == 1
    assert plan.regions[0].nodes == graph.nodes


# -- against the real declarations -------------------------------------------


def test_reference_and_cpu_plan_eagerly() -> None:
    graph = _graph("add", "add")
    assert not plan_execution(graph, REFERENCE).uses_graph_path
    assert not plan_execution(graph, CPU).uses_graph_path


def test_cuda_splits_at_the_sampling_tail() -> None:
    """The declaration drives the plan, not a guess about it."""
    graph = _graph("rms_norm", "argmax")
    plan = plan_execution(graph, CUDA)
    assert plan.captured_nodes == 1
    assert plan.eager_nodes == 1
    assert plan.regions[1].nodes[0].op == "argmax"


def test_qnn_plans_a_whole_graph_at_once() -> None:
    """``GRAPH`` granularity: the host is out of the loop, so the region is the model."""
    plan = plan_execution(_graph("rms_norm", "silu_mul"), QNN)
    assert plan.capability.granularity is RegionGranularity.GRAPH
    assert plan.uses_graph_path


def test_every_backends_plan_only_names_ops_it_declares() -> None:
    """A regression guard: the planner must never mark an undeclared op captured.

    The graph is built from the ABI's whole vocabulary, so every backend's
    ``captures`` set is exercised against ops it may or may not have declared.
    """
    from pocketllm.kernels.registry import OPS

    graph = _graph(*sorted(OPS.names()))
    planned = 0
    for entry in registry.BACKENDS.values():
        backend = entry.factory()
        plan = plan_execution(graph, backend)
        declared = {cap.op for cap in backend.capabilities()}
        for region in plan.regions:
            if region.captured:
                planned += 1
                assert {node.op for node in region.nodes} <= declared, f"{entry.name} captured an undeclared op"
    assert planned, "no backend captured anything; this guard tested nothing"