"""The op graph: type-checking a model before any backend sees it.

``Graph.verify`` walks the nodes and resolves every argument through the
registry's schemas, so an inconsistent graph is caught on the host with no
device involved.  That is the property being tested: the check is available
before a backend is opened.
"""

from __future__ import annotations

import pytest

from pocketllm.kernels import DType, Graph, GraphRegion, Node, ShapeError, Tensor, TensorDesc, Value
from pocketllm.kernels.buffer import DeviceBuffer
from pocketllm.kernels.device import Device


def _desc(shape, dtype=DType.F32) -> TensorDesc:
    return TensorDesc(shape, dtype=dtype)


def test_two_op_graph_resolves_every_value() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)), _desc((4, 8))),
        input_names=("x", "y"),
        nodes=(
            Node(op="add", args=(Value("x"), Value("y")), outputs=("s",)),
            Node(op="mul", args=(Value("s"), Value("x")), outputs=("p",)),
        ),
        outputs=(Value("p"),),
    )
    env = graph.verify()
    assert env["s"].shape == (4, 8)
    assert env["p"].shape == (4, 8)


def test_gemm_node_infers_a_narrower_output() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)), _desc((16, 8))),
        input_names=("x", "w"),
        nodes=(Node(op="gemm", args=(Value("x"), Value("w")), outputs=("y",)),),
        outputs=(Value("y"),),
    )
    assert graph.verify()["y"].shape == (4, 16)


def test_unknown_input_is_refused_by_name() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)),),
        input_names=("x",),
        nodes=(Node(op="add", args=(Value("x"), Value("nope")), outputs=("s",)),),
        outputs=(Value("s"),),
    )
    with pytest.raises(ShapeError, match="nope"):
        graph.verify()


def test_shape_mismatch_across_nodes_is_refused() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)), _desc((4, 9))),
        input_names=("x", "y"),
        nodes=(Node(op="add", args=(Value("x"), Value("y")), outputs=("s",)),),
        outputs=(Value("s"),),
    )
    with pytest.raises(ShapeError, match="mismatch|dimension"):
        graph.verify()


def test_redefined_value_is_refused() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)),),
        input_names=("x",),
        nodes=(
            Node(op="add", args=(Value("x"), Value("x")), outputs=("s",)),
            Node(op="mul", args=(Value("x"), Value("x")), outputs=("s",)),
        ),
        outputs=(Value("s"),),
    )
    with pytest.raises(ShapeError, match="already defined"):
        graph.verify()


def test_dangling_output_is_refused() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)),),
        input_names=("x",),
        nodes=(Node(op="add", args=(Value("x"), Value("x")), outputs=("s",)),),
        outputs=(Value("gone"),),
    )
    with pytest.raises(ShapeError, match="gone"):
        graph.verify()


def test_too_many_arguments_is_refused() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)),),
        input_names=("x",),
        nodes=(Node(op="argmax", args=(Value("x"), Value("x"), Value("x"), Value("x")), outputs=("i",)),),
        outputs=(Value("i"),),
    )
    with pytest.raises(ShapeError):
        graph.verify()


def test_whole_graph_region_names_every_input_and_output() -> None:
    graph = Graph(
        inputs=(_desc((4, 8)),),
        input_names=("x",),
        nodes=(Node(op="add", args=(Value("x"), Value("x")), outputs=("s",)),),
        outputs=(Value("s"),),
    )
    region = GraphRegion.whole(graph)
    assert region.inputs == ("x",)
    assert region.outputs == ("s",)
    assert len(region.nodes) == 1


def test_graph_carries_no_device_or_buffer() -> None:
    """Construction and verification are pure: no device is touched."""
    descriptor = _desc((2, 2))
    graph = Graph(
        inputs=(descriptor,),
        input_names=("x",),
        nodes=(Node(op="add", args=(Value("x"), Value("x")), outputs=("s",)),),
        outputs=(Value("s"),),
    )
    assert graph.verify()["x"] is descriptor
    # A real tensor is only needed to *run* a graph, never to verify one.
    buffer = DeviceBuffer(device=Device("cpu", 0), nbytes=descriptor.nbytes)
    tensor = Tensor(descriptor, buffer)
    assert tensor.desc is descriptor