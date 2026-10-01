"""Walking a graph: correctness, refusal, and the buffers on the way out.

The executor is the path every model takes when a backend cannot capture
anything -- which is the reference backend, the CPU, and MPS.  So these tests
are not about an optimisation; they are about the fallback being *complete*:
every op in the vocabulary must run through it, a graph with a mistake must be
refused before any allocation, and a refused op must not leak its buffers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.architectures.toy import ToyConfig, build
from pocketllm.backends.reference import BACKEND
from pocketllm.engine.executor import Executor, NodeResult
from pocketllm.kernels.backend import Capability, GraphCapability
from pocketllm.kernels.buffer import DeviceBuffer
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.errors import BackendNotImplementedError
from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import Tensor, TensorDesc


def _toy_session():
    return BACKEND.open(Device("cpu"))


class _RefusingBackend:
    """A declaration-only backend, enough for the executor to resolve against."""

    name = "refusing"
    device_kind = "cpu"
    version = "test"
    is_reference = True

    def available(self) -> bool:
        return True

    def capabilities(self):
        return (Capability(op="add", dtypes=frozenset({DType.F32})),)

    def graph(self):
        return GraphCapability()

    def compile_spec(self):
        return None

    def open(self, device, *, options=None):  # pragma: no cover - never opened
        raise NotImplementedError


class _RefusingSession:
    """Allocates honestly, then refuses every op -- a runtime that is present but bare.

    ``host_tensor`` is the one convenience the reference session has that this
    stub needs too: the executor takes :class:`Tensor` inputs, and building one
    from an array is not the thing under test.
    """

    def __init__(self) -> None:
        self.backend = _RefusingBackend()
        self.device = Device("cpu")
        self._storage = bytearray(1)

    def alloc(self, nbytes: int, *, align: int = 64) -> DeviceBuffer:
        return DeviceBuffer(
            device=self.device,
            nbytes=int(nbytes),
            alignment=align,
            owner=self,
            _view=memoryview(self._storage)[:0],
        )

    def free(self, buffer) -> None:
        return None

    def to_device(self, host, desc) -> Tensor:
        raise NotImplementedError

    def to_host(self, tensor) -> memoryview:
        raise NotImplementedError

    def host_tensor(self, array: np.ndarray) -> Tensor:
        array = np.ascontiguousarray(array)
        desc = TensorDesc(tuple(array.shape), DType.F32)
        return Tensor(desc, DeviceBuffer(device=self.device, nbytes=desc.nbytes, owner=array))

    def run(self, op, args, *, out=None, attrs=None):
        raise BackendNotImplementedError("refused on purpose")

    def compile_graph(self, graph):
        return None

    def capture(self, region, *, warmup: int = 3):
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _bind(spec, session, *, seed: int = 0):
    """Zero-or-random inputs for every graph input, including the weights."""
    env = spec.graph.verify()
    rng = np.random.default_rng(seed)
    bound = {}
    for name in spec.graph.input_names:
        desc = env[name]
        if desc.dtype is DType.I32:
            array = np.zeros(desc.shape, np.int32)
        else:
            array = rng.normal(size=desc.shape).astype(np.float32)
        bound[name] = session.tensor(array)
    return bound


def test_the_toy_graph_runs_end_to_end() -> None:
    spec = build(ToyConfig(hidden=8, ff=16, vocab=32))
    session = _toy_session()
    try:
        executor = Executor(session)
        out = executor.run(spec.graph, _bind(spec, session))
        y = np.asarray(session.array(out["y"]))
        assert y.shape == (1, 8)
        assert np.isfinite(y).all()
        assert np.any(y != 0), "an all-zero result usually means the weights were not read"
    finally:
        session.close()


def test_the_same_inputs_give_the_same_answer() -> None:
    """Determinism, which a capture path will later be checked against."""
    spec = build(ToyConfig())
    session = _toy_session()
    try:
        executor = Executor(session)
        inputs = _bind(spec, session)
        first = np.asarray(session.array(executor.run(spec.graph, inputs)["y"]))
        second = np.asarray(session.array(executor.run(spec.graph, inputs)["y"]))
        assert np.array_equal(first, second)
    finally:
        session.close()


def test_a_graph_is_verified_before_anything_is_allocated() -> None:
    """A malformed graph fails with a shape error, not a half-run allocation."""
    session = _toy_session()
    try:
        executor = Executor(session)
        # `add` requires two dimensions; a 1-d output is a graph the verifier
        # must reject.  The graph claims the op produced it, which it cannot.
        bad = Graph(
            inputs=(TensorDesc((4,), DType.F32),),
            input_names=("x",),
            nodes=(Node(op="silu_mul", args=(Value("x"), Value("x")), outputs=("y",), name="bad"),),
            outputs=(Value("y"),),
        )
        with pytest.raises(Exception) as excinfo:
            executor.run(bad, {"x": session.tensor(np.ones(4, np.float32))})
        assert "silu_mul" in str(excinfo.value) or "dims" in str(excinfo.value)
        assert executor.arena.allocations == 0, "verification must precede allocation"
    finally:
        session.close()


def test_missing_and_extra_inputs_are_named() -> None:
    spec = build(ToyConfig())
    session = _toy_session()
    try:
        executor = Executor(session)
        with pytest.raises(ValueError, match="missing"):
            executor.run(spec.graph, {"tokens": session.tensor(np.zeros(1, np.int32))})
        with pytest.raises(ValueError, match="unexpected"):
            executor.run(spec.graph, {**_bind(spec, session), "surprise": session.tensor(np.zeros(1, np.float32))})
    finally:
        session.close()


def test_an_input_with_the_wrong_descriptor_is_refused() -> None:
    """The graph's descriptor is what every downstream shape was inferred from."""
    spec = build(ToyConfig(hidden=8, ff=16, vocab=32))
    session = _toy_session()
    try:
        executor = Executor(session)
        inputs = _bind(spec, session)
        inputs["tokens"] = session.tensor(np.zeros((1, 1), np.int32))  # 2-d, graph says 1-d
        with pytest.raises(ValueError, match="descriptor"):
            executor.run(spec.graph, inputs)
    finally:
        session.close()


def test_a_session_that_cannot_run_an_op_does_not_leak_its_buffers() -> None:
    """A backend refusing mid-graph must not strand the buffers it was given.

    The executor allocates an output before asking the session to fill it, so a
    raise between those two steps would leave the arena holding a buffer nothing
    will ever release.  The cleanup path exists for exactly this, and it is
    checked by the arena's own live byte count.
    """
    session = _RefusingSession()
    executor = Executor(session)
    graph = Graph(
        inputs=(TensorDesc((4, 4), DType.F32), TensorDesc((4, 4), DType.F32)),
        input_names=("a", "b"),
        nodes=(Node(op="add", args=(Value("a"), Value("b")), outputs=("c",), name="n"),),
        outputs=(Value("c"),),
    )
    shared = session.host_tensor(np.zeros((4, 4), np.float32))
    with pytest.raises(BackendNotImplementedError):
        executor.run(graph, {"a": shared, "b": shared})
    assert executor.arena.live_bytes == 0, "the refused output buffer was not released"


def test_a_run_can_be_traced() -> None:
    spec = build(ToyConfig())
    session = _toy_session()
    try:
        executor = Executor(session)
        run = executor.run(spec.graph, _bind(spec, session), trace=True)
        assert len(run.trace.nodes) == len(spec.graph.nodes)
        assert run.trace.backends_used == ("reference",)
        assert not run.trace.crossed_a_device
        assert run.trace.peak_bytes > 0
        assert "reference" in run.trace.summary()
    finally:
        session.close()


def test_trace_can_report_a_cross_device_graph() -> None:
    """The trace names the case worth printing, rather than leaving it to a profile."""
    from pocketllm.engine.executor import ExecutionTrace

    trace = ExecutionTrace()
    from pocketllm.engine.executor import NodeResult

    trace.nodes.append(NodeResult(Node(op="add", args=(), outputs=("a",)), "cuda", Device("cuda", 0), ("a",), 4))
    trace.nodes.append(NodeResult(Node(op="argmax", args=(), outputs=("t",)), "reference", Device("cpu"), ("t",), 4))
    assert trace.crossed_a_device
    assert trace.backends_used == ("cuda", "reference")
    assert "crosses devices" in trace.summary()