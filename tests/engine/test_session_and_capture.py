"""Session selection, the fallback policy, and the right to decline a region.

Two things are checked here that nothing else can.  The **policy asymmetry** --
a slow answer is acceptable for ``run`` and not for ``serve`` -- is a decision
with a stated reason, and a decision nothing exercises is a comment.  And the
**decline path**: a backend whose graph path returns ``None`` must fall back to
eager execution with no branch at the call site, because that is what makes an
optional accelerator genuinely optional.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.backends import registry
from pocketllm.engine import EngineSession, SessionPolicy
from pocketllm.engine.captured import run_region
from pocketllm.engine.planner import plan_execution
from pocketllm.kernels.backend import GraphCapability, GraphMode
from pocketllm.kernels.buffer import DeviceBuffer
from pocketllm.kernels.device import Device
from pocketllm.kernels.graph import Graph, GraphRegion, Node, Value
from pocketllm.kernels.tensor import Tensor, TensorDesc
from pocketllm.kernels.dtypes import DType

from pocketllm.backends.reference import BACKEND as REFERENCE


# -- selection ---------------------------------------------------------------


def test_auto_selects_a_loadable_backend() -> None:
    session = EngineSession.open("auto", policy=SessionPolicy.for_run())
    try:
        assert session.backend.available()
        assert session.device.kind == session.backend.device_kind
    finally:
        session.close()


def test_a_named_backend_is_honoured() -> None:
    session = EngineSession.open(None, policy=SessionPolicy.for_run(backend="reference"))
    try:
        assert session.backend.name == "reference"
    finally:
        session.close()


def test_a_named_backend_that_is_not_loadable_refuses() -> None:
    """--backend qnn on a host without the SDK names the device that would fix it."""
    from pocketllm.engine.session import NoUsableBackend

    with pytest.raises(NoUsableBackend) as excinfo:
        EngineSession.open(None, policy=SessionPolicy.for_run(backend="qnn"))
    assert "qnn" in str(excinfo.value)


def test_serve_refuses_the_reference_fallback_that_run_allows() -> None:
    """The one place the engine has a performance opinion, checked in both directions.

    ``run`` accepts numpy on the host -- a slow answer beats none interactively.
    ``serve`` does not: a request that cannot return in time is worse than a
    refusal naming the device that would have worked.
    """
    serve = SessionPolicy.for_serve()
    assert not serve.allow_reference_fallback
    assert SessionPolicy.for_run().allow_reference_fallback

    # The policy reaches the dispatcher, not just the selector.
    session = EngineSession.open("cpu", policy=serve)
    try:
        assert session.executor().dispatcher.allow_reference_fallback is False
    finally:
        session.close()


def test_an_unimplemented_device_refuses_with_a_useful_message() -> None:
    """The refusal lists every candidate and why it was rejected."""
    from pocketllm.engine.session import NoUsableBackend

    with pytest.raises(NoUsableBackend) as excinfo:
        EngineSession.open("qnn", policy=SessionPolicy.for_serve())
    message = str(excinfo.value)
    assert "no usable backend for 'qnn'" in message
    assert "reference fallback disabled" in message
    for entry in registry.BACKENDS.values():
        assert entry.name in message, "the refusal should list every backend it considered"


def test_describe_states_whether_a_capture_path_exists() -> None:
    session = EngineSession.open("cpu")
    try:
        assert "eager only" in session.describe()
    finally:
        session.close()


def test_a_session_is_a_context_manager() -> None:
    with EngineSession.open("cpu") as session:
        assert session.backend.name in {"cpu", "reference"}
    session.close()  # idempotent


# -- declining a region ------------------------------------------------------


class _DecliningSession:
    """A session whose graph path exists and refuses, as a real one will.

    A CUDA capture that meets an unsupported shape, or an AOT backend whose
    toolchain is absent, both return ``None`` from the graph call.  The engine
    has to read that as "run it eagerly", not as an error -- so the test drives
    exactly that and checks the fallback decision, not the fallback's output.
    """

    def __init__(self, *, returns: int | None = None) -> None:
        self.backend = _DecliningBackend()
        self.device = Device("cpu")
        self._returns = returns

    def compile_graph(self, graph):
        return None

    def capture(self, region, *, warmup: int = 3):
        if self._returns is None:
            return None
        dummies = tuple(Tensor(TensorDesc((2, 2), DType.F32), DeviceBuffer(device=self.device, nbytes=16))
                        for _ in range(self._returns))

        class _Captured:
            def replay(self, inputs):
                return dummies

        return _Captured()


class _DecliningBackend:
    name = "declining"
    device_kind = "cpu"
    version = "test"
    is_reference = False

    def available(self) -> bool:
        return True

    def capabilities(self):
        from pocketllm.kernels.backend import Capability

        return (Capability(op="add", dtypes=frozenset({DType.F32})),)

    def graph(self) -> GraphCapability:
        return GraphCapability(supported=True, mode=GraphMode.STREAM_CAPTURE, captures=frozenset({"add"}))

    def compile_spec(self):
        return None

    def open(self, device, *, options=None):  # pragma: no cover
        raise NotImplementedError


def _region_and_inputs(count: int):
    nodes = tuple(
        Node(op="add", args=(Value("x") if i == 0 else Value(f"v{i - 1}"),) * 2, outputs=(f"v{i}",), name=f"n{i}")
        for i in range(count)
    )
    region = GraphRegion(nodes=nodes, inputs=("x",), outputs=(f"v{count - 1}",))
    desc = TensorDesc((2, 2), DType.F32)
    tensor = Tensor(desc, DeviceBuffer(device=Device("cpu"), nbytes=desc.nbytes))
    return region, {"x": tensor}


def test_a_declined_capture_falls_back_to_eager() -> None:
    session = _DecliningSession(returns=None)
    region, inputs = _region_and_inputs(2)
    assert run_region(session, region, inputs) is None


def test_a_capture_path_that_returns_the_wrong_arity_is_a_bug_not_a_fallback() -> None:
    """Absorbing an arity mismatch would turn a backend bug into a wrong answer.

    A decline is ``None``; a *result* with the wrong number of values is the
    backend contradicting its own declaration, and it raises.  The region has two
    nodes and one output, and the fake replay returns two values.
    """
    session = _DecliningSession(returns=2)
    region, inputs = _region_and_inputs(1)
    with pytest.raises(ValueError, match="returned 2 values"):
        run_region(session, region, inputs)


def test_an_unsupported_capability_is_declined_without_calling_anything() -> None:
    """``supported=False`` must short-circuit before either graph method is tried.

    Calling ``capture`` on a backend that declares no capture path would be
    asking a device to do something it has said it cannot -- cheap on the
    reference backend, a wasted driver round-trip on a real one.
    """

    class _NoGraph:
        def __init__(self):
            self.backend = _NoGraphBackend()

        def compile_graph(self, graph):
            pytest.fail("compile_graph was called on a backend with no graph path")

        def capture(self, region, *, warmup: int = 3):
            pytest.fail("capture was called on a backend with no graph path")

    class _NoGraphBackend(_DecliningBackend):
        def graph(self) -> GraphCapability:
            return GraphCapability()

    session = _NoGraph()
    region, inputs = _region_and_inputs(1)
    assert run_region(session, region, inputs) is None


# -- the reference session declares no graph path, and that is not an error ---


def test_the_reference_session_declines_by_returning_none() -> None:
    session = REFERENCE.open(Device("cpu"))
    try:
        region, inputs = _region_and_inputs(1)
        assert session.compile_graph(None) is None
        assert session.capture(region) is None
        assert run_region(session, region, inputs) is None
    finally:
        session.close()


def test_cudagraph_selection_reaches_the_planner() -> None:
    """A declarative check that cuda's capture path is wired through, not just declared."""
    from pocketllm.backends.cuda import BACKEND as CUDA

    graph = Graph(
        inputs=(TensorDesc((4, 6), DType.F32),),
        input_names=("x",),
        nodes=(
            Node(op="rms_norm", args=(Value("x"), Value("x")), outputs=("a",), name="n0"),
            Node(op="argmax", args=(Value("a"),), outputs=("t",), name="n1"),
        ),
        outputs=(Value("t"),),
    )
    plan = plan_execution(graph, CUDA)
    assert plan.uses_graph_path
    assert [r.captured for r in plan.regions] == [True, False]