"""The model IR: a graph, plus the names a checkpoint binds weights by.

There is deliberately no ``Model`` class in this tree.  A model *is* a
:class:`~pocketllm.kernels.graph.Graph` -- nodes, values and descriptors -- and
:class:`ModelSpec` is that graph with the two pieces of bookkeeping a serving
path needs on top of it:

* **which tensors are weights**, so the loader knows what to fill and the
  executor knows what to keep resident rather than free;
* **where the cache lives**, so a decode loop can truncate it between requests.

Everything else -- a layer count, a head count, a rope base -- is *input* to the
builder that produces the graph, not a property of the graph.  Keeping it that
way is what lets two checkpoints with the same structure share one builder and
two very different ones produce graphs the same executor runs.

The weights live in a :class:`WeightTable`: a name to descriptor map, with no
bytes.  Bytes are the loader's, and where they sit is the device's; this layer
names them and stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import TensorDesc

from .cache import CachePlan

__all__ = ["WeightSpec", "WeightTable", "ModelSpec", "GraphBuilder"]


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """One tensor the checkpoint must supply, and what it means."""

    name: str
    desc: TensorDesc
    #: A hint for the loader: which architecture parameter this is, so a
    #: checkpoint with a different naming convention can be mapped onto it.
    #: Free-form by design -- the mapping is the loader's problem.
    role: str = ""
    #: Whether the block layout may be quantized.  An embedding table and a
    #: projection usually may; a norm weight usually may not.
    quantizable: bool = True


class WeightTable:
    """The weights a model spec names, by name.  No bytes, and no device."""

    def __init__(self, specs: Sequence[WeightSpec] = ()) -> None:
        self._specs: dict[str, WeightSpec] = {}
        for spec in specs:
            self.add(spec)

    def add(self, spec: WeightSpec) -> None:
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"weight {spec.name!r} is declared twice, with different descriptors")
        self._specs[spec.name] = spec

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)

    def get(self, name: str) -> WeightSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"no weight named {name!r}; the table has {sorted(self._specs)}") from exc

    def names(self) -> frozenset[str]:
        return frozenset(self._specs)

    def specs(self) -> tuple[WeightSpec, ...]:
        return tuple(self._specs[name] for name in sorted(self._specs))

    def total_bytes(self) -> int:
        """How much storage the weights need as described (unquantized)."""
        return sum(spec.desc.nbytes for spec in self._specs.values())


@dataclass(slots=True)
class ModelSpec:
    """A model as the engine sees it: a graph, its weights, and its cache."""

    graph: Graph
    weights: WeightTable = field(default_factory=WeightTable)
    #: The value(s) a decode loop appends to each step.  Named rather than
    #: inferred: a paged cache is several values, and which one is which is the
    #: architecture's knowledge, not the engine's.
    cache_values: tuple[str, ...] = ()
    #: How large the cache must be, per tensor.  Empty for a model with no
    #: attention; the builder fills it from the config's layer geometry.
    cache: CachePlan = field(default_factory=CachePlan)
    #: Trusted only if ``verified``; the builder sets it after a successful
    #: ``graph.verify()`` so the executor is not the first thing to find a shape
    #: mistake at run time.
    verified: bool = False

    @property
    def name(self) -> str:
        return self.graph.name or "unnamed"

    def verify(self) -> "ModelSpec":
        """Run the graph's own type check and mark the spec verified."""
        self.graph.verify()
        self.verified = True
        return self

    def node(self, label: str) -> Node:
        for node in self.graph.nodes:
            if node.label() == label:
                return node
        raise KeyError(f"no node labelled {label!r} in {self.name}")

    def value(self, name: str) -> Value:
        if name not in self._value_names():
            raise KeyError(f"no value named {name!r} in {self.name}")
        return Value(name)

    def _value_names(self) -> set[str]:
        names = set(self.graph.input_names)
        for node in self.graph.nodes:
            names.update(node.outputs)
        return names

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self.graph.nodes)} nodes, "
            f"{len(self.weights)} weights ({self.weights.total_bytes()} B), "
            f"{len(self.cache_values)} cache values"
        )

    @property
    def weight_values(self) -> tuple[str, ...]:
        """The graph inputs the caller must bind from the checkpoint, not from a request."""
        return tuple(name for name in self.graph.input_names if name in self.weights)


class GraphBuilder:
    """Accumulates nodes and values into a :class:`ModelSpec`.

    A builder, rather than hand-assembling ``Node`` tuples, for two reasons that
    both bite on a real model: names have to be unique across a hundred layers,
    and a returned :class:`Value` has to carry the descriptor of what the node
    actually produces.  The builder does the first by construction and the second
    by running the schema's own shape rule, so a value's descriptor is never a
    guess.

    The tag each node is given is what makes ``layer3.attn.q`` a name rather than
    a counter, and it is what ``--explain`` prints, so a graph in a trace reads
    like the model.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: list[Node] = []
        self._env: dict[str, TensorDesc] = {}
        self._inputs: list[str] = []
        self._input_descs: list[TensorDesc] = []
        self._outputs: list[Value] = []
        self._weights = WeightTable()
        self._cache_values: list[str] = []
        self._used_labels: set[str] = set()

    # -- inputs and weights -------------------------------------------------

    def input(self, name: str, desc: TensorDesc) -> Value:
        """Declare a graph input.  Bound by the caller at run time."""
        if name in self._env:
            raise ValueError(f"value {name!r} is already declared")
        self._inputs.append(name)
        self._input_descs.append(desc)
        self._env[name] = desc
        return Value(name)

    def weight(self, name: str, desc: TensorDesc, *, role: str = "", quantizable: bool = True) -> Value:
        """Declare a weight.  It is a graph input, but the loader fills it."""
        if name in self._env:
            raise ValueError(f"value {name!r} is already declared")
        self._inputs.append(name)
        self._input_descs.append(desc)
        self._env[name] = desc
        self._weights.add(WeightSpec(name, desc, role=role, quantizable=quantizable))
        return Value(name)

    def constant(self, name: str, desc: TensorDesc) -> Value:
        """A value the caller supplies that is *not* a weight -- a rope table, a mask."""
        return self.input(name, desc)

    # -- nodes --------------------------------------------------------------

    def op(self, op: str, /, args: Sequence, *, outputs: Sequence[str], attrs: Mapping | None = None, tag: str = "") -> tuple[Value, ...]:
        """Append one node, infer its outputs' descriptors, and return handles.

        ``outputs`` names the values the node produces.  The number must match
        the op's declared returns, and each name must be new; both are checked
        here so the failure names the offending node rather than surfacing later
        as a mysterious shape error three layers down.
        """
        from pocketllm.kernels.registry import OPS

        schema = OPS.get(op)
        attrs = dict(attrs or {})
        resolved = [self._resolve(arg) for arg in args]
        descs = schema.infer(resolved, attrs)
        if len(outputs) != len(descs):
            raise ValueError(
                f"{tag or op}: op {op!r} returns {len(descs)} values, {len(outputs)} names were given"
            )
        for out in outputs:
            if out in self._env:
                raise ValueError(f"{tag or op}: output {out!r} is already declared")
        label = self._label(tag or op, outputs)
        node = Node(op=op, args=tuple(args), outputs=tuple(outputs), attrs=attrs, name=label)
        self._nodes.append(node)
        for out, desc in zip(outputs, descs):
            self._env[out] = desc
        return tuple(Value(out) for out in outputs)

    def one(self, op: str, /, *args, outputs: str, attrs: Mapping | None = None, tag: str = "") -> Value:
        """The single-output convenience over :meth:`op`."""
        (value,) = self.op(op, list(args), outputs=[outputs], attrs=attrs, tag=tag)
        return value

    # -- cache --------------------------------------------------------------

    def mark_cache(self, *names: str) -> None:
        """Record values the decode loop appends to between steps."""
        self._cache_values.extend(name for name in names if name not in self._cache_values)

    # -- the result ---------------------------------------------------------

    def output(self, *values: Value) -> None:
        for value in values:
            if value.name not in self._env:
                raise KeyError(f"graph output {value.name!r} is not produced by any node")
            if value not in self._outputs:
                self._outputs.append(value)

    def build(self, *, verify: bool = True) -> ModelSpec:
        graph = Graph(
            inputs=tuple(self._input_descs),
            input_names=tuple(self._inputs),
            nodes=tuple(self._nodes),
            outputs=tuple(self._outputs),
            name=self.name,
        )
        spec = ModelSpec(
            graph=graph,
            weights=self._weights,
            cache_values=tuple(self._cache_values),
        )
        if verify:
            spec.verify()
        return spec

    # -- internals ----------------------------------------------------------

    def _resolve(self, arg):
        if isinstance(arg, Value):
            if arg.name not in self._env:
                raise KeyError(f"argument {arg.name!r} has not been declared")
            return self._env[arg.name]
        return arg

    def _label(self, tag: str, outputs: Sequence[str]) -> str:
        label = tag
        if label in self._used_labels:
            suffix = 2
            while f"{label}.{suffix}" in self._used_labels:
                suffix += 1
            label = f"{label}.{suffix}"
        self._used_labels.add(label)
        return label