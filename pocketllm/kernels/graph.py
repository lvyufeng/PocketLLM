"""The op graph: what a model *is*, before any backend runs it.

A :class:`Graph` is data -- nodes, each an op with arguments that name other
nodes' outputs or the graph's inputs.  It carries no device, no buffer and no
execution policy.  The engine walks it (``engine/executor.py``) or hands a
:class:`GraphRegion` to a backend's graph path (``engine/captured.py``); the
backend never sees a graph it did not agree, through
:class:`~pocketllm.kernels.backend.GraphCapability`, to accept.

:meth:`Graph.verify` is the graph's own type checker: it walks the nodes,
resolves every argument to a :class:`TensorDesc` through the registry's schemas,
and reports the first place a shape or an op name does not hold.  That check runs
without a device, so a graph is validated before any backend touches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import ShapeError
from .registry import OPS, OpRegistry
from .schema import Kind
from .tensor import TensorDesc

__all__ = ["Value", "Node", "Graph", "GraphRegion"]


@dataclass(frozen=True, slots=True)
class Value:
    """A handle to one value in a graph: a graph input, or a node's output."""

    name: str


@dataclass(frozen=True, slots=True)
class Node:
    """One op application.  Arguments are :class:`Value` handles or Python scalars."""

    op: str
    args: tuple[Any, ...]
    outputs: tuple[str, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""

    def label(self) -> str:
        return self.name or f"{self.op}#{id(self):x}"


@dataclass(slots=True)
class Graph:
    """A directed acyclic graph of ops over named values."""

    inputs: tuple[TensorDesc, ...]
    input_names: tuple[str, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[Value, ...]
    name: str = ""

    def verify(self, registry: OpRegistry | None = None) -> dict[str, TensorDesc]:
        """Resolve every value's descriptor, or raise the first inconsistency.

        Returns a name -> descriptor map for the whole graph, which is what the
        executor uses to lay out buffers.  Raises :class:`ShapeError` naming the
        node and the value at fault.
        """
        registry = registry or OPS
        if len(self.inputs) != len(self.input_names):
            raise ShapeError("graph has a different number of inputs than input names")
        env: dict[str, TensorDesc] = dict(zip(self.input_names, self.inputs))
        for node in self.nodes:
            schema = registry.get(node.op)
            resolved_args = [_resolve(value, env, node, spec.name) for value, spec in _aligned(node, schema)]
            out_descs = schema.infer(resolved_args, node.attrs)
            if len(out_descs) != len(node.outputs):
                raise ShapeError(
                    f"{node.label()}: op {node.op!r} returns {len(out_descs)} values, "
                    f"node declares {len(node.outputs)}"
                )
            for out_name, desc in zip(node.outputs, out_descs):
                if out_name in env:
                    raise ShapeError(f"{node.label()}: output {out_name!r} is already defined")
                env[out_name] = desc
        for value in self.outputs:
            if value.name not in env:
                raise ShapeError(f"graph output {value.name!r} is not produced by any node")
        return env


@dataclass(frozen=True, slots=True)
class GraphRegion:
    """A contiguous run of nodes a backend's graph path is asked to take whole."""

    nodes: tuple[Node, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @classmethod
    def whole(cls, graph: Graph) -> "GraphRegion":
        return cls(
            nodes=graph.nodes,
            inputs=graph.input_names,
            outputs=tuple(value.name for value in graph.outputs),
        )


def _aligned(node: Node, schema) -> list[tuple[Any, Any]]:
    """Pair each of a node's arguments with its :class:`ArgSpec`, padding optionals."""
    if len(node.args) > len(schema.args):
        raise ShapeError(
            f"{node.label()}: op {node.op!r} takes {len(schema.args)} arguments, "
            f"node passes {len(node.args)}"
        )
    padded = list(node.args) + [None] * (len(schema.args) - len(node.args))
    return list(zip(padded, schema.args))


def _resolve(value: Any, env: Mapping[str, TensorDesc], node: Node, arg_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Value):
        if value.name not in env:
            raise ShapeError(f"{node.label()}: argument {arg_name!r} names unknown value {value.name!r}")
        return env[value.name]
    return value