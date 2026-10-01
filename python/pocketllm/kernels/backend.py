"""What a backend *is*, and what it must declare.

A backend is a device implementation: it can allocate memory on a device and run
ops there.  It is discovered and loaded lazily, and it must be able to answer
:meth:`Backend.available` on a host that does not have its runtime installed --
so ``available`` may probe with ``find_spec``/``find_library``/a ``/dev`` check,
but it may not import the runtime.

The op-level ABI is the contract every backend fulfils; the graph path
(:class:`GraphCapability`) is the *optional* accelerated one.  A backend with no
graph support runs op-by-op, and that is correct, not degraded -- which is why
``GraphCapability.supported`` defaults to ``False`` and the engine treats a
``None`` from ``capture``/``compile_graph`` as "fall back", never as "error".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .buffer import Buffer
from .device import Device
from .dtypes import DType, QuantFormat
from .graph import Graph, GraphRegion
from .tensor import Tensor

__all__ = [
    "GraphMode",
    "RegionGranularity",
    "Capability",
    "GraphCapability",
    "CompileSpec",
    "Backend",
    "BackendSession",
    "CompiledGraph",
    "CapturedGraph",
]


class GraphMode(Enum):
    """How a backend accelerates a region, if at all."""

    NONE = "none"
    STREAM_CAPTURE = "stream_capture"  # replay a recorded stream (CUDA Graph, aclgraph)
    AOT_COMPILE = "aot_compile"        # compile ahead of time (QNN, Horizon, ...)


class RegionGranularity(Enum):
    """How much of a graph a backend wants handed to its graph path."""

    STEP = "step"    # one decode step at a time; the host stays in the loop
    GRAPH = "graph"  # the whole model at once; the host is out of the loop


@dataclass(frozen=True, slots=True)
class Capability:
    """One op a backend implements, and the domain it implements it over."""

    op: str
    dtypes: frozenset[DType] = frozenset()
    quants: frozenset[QuantFormat] = frozenset()
    #: A predicate over ``(args, attrs)`` for a domain the schema cannot express:
    #: "decode only (rows == 1)", "K a multiple of 256", "no mask".  ``None``
    #: means everything the schema admits.
    accepts: Callable[[Sequence[Any], Mapping[str, Any]], bool] | None = None
    #: Preference among equal matches; lower wins.  A documented relative cost,
    #: not a measurement.
    rank: int = 100

    def admits(self, dtypes: frozenset[DType], quants: frozenset[QuantFormat]) -> bool:
        """Whether this capability covers the given element and block types."""
        if self.dtypes and not dtypes <= self.dtypes:
            return False
        if quants and not quants <= self.quants:
            return False
        return True


@dataclass(frozen=True, slots=True)
class GraphCapability:
    """The optional accelerated path.  Default: none, and that is correct."""

    supported: bool = False
    mode: GraphMode = GraphMode.NONE
    #: Op names legal inside a capture region.  A region containing anything else
    #: is not capturable and the engine runs it eagerly.
    captures: frozenset[str] = frozenset()
    granularity: RegionGranularity = RegionGranularity.STEP
    max_nodes: int = 0
    rebuild_on_shape_change: bool = True

    def admits(self, region: "GraphRegion") -> bool:
        if not self.supported:
            return False
        if self.max_nodes and len(region.nodes) > self.max_nodes:
            return False
        return all(node.op in self.captures for node in region.nodes)


@dataclass(frozen=True, slots=True)
class CompileSpec:
    """What offline (host-side) compilation of a graph needs.

    AOT NPUs are typically compiled by an x86 host toolchain and *run* on an
    arm64 device, so the ABI never assumes the compile host equals the run host:
    this spec says what the compile step wants, and the artifact it produces is
    loaded by :meth:`BackendSession.open`-time device code.
    """

    host_toolchain: str
    target_arch: str
    artifact_format: str
    options: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """A device implementation, as the engine sees it."""

    name: str
    device_kind: str
    version: str

    def available(self) -> bool:
        """Whether this backend can be *loaded* here.  Must not import the runtime."""

    def capabilities(self) -> tuple[Capability, ...]:
        """The ops this backend implements, and over what domain."""

    def graph(self) -> GraphCapability:
        """The optional accelerated path; ``GraphCapability()`` when there is none."""

    def compile_spec(self) -> CompileSpec | None:
        """What an offline compile needs, or ``None`` when there is no AOT path."""

    def open(self, device: Device, *, options: Mapping[str, Any] | None = None) -> "BackendSession":
        """Open one session on one device."""


@runtime_checkable
class BackendSession(Protocol):
    """An opened device.  One session owns one device -- one process, one card."""

    backend: Backend
    device: Device

    # -- memory -------------------------------------------------------------
    def alloc(self, nbytes: int, *, align: int = 64) -> Buffer: ...
    def free(self, buffer: Buffer) -> None: ...
    def to_device(self, host: memoryview, desc) -> Tensor: ...
    def to_host(self, tensor: Tensor) -> memoryview: ...

    # -- the op-level ABI (always present) ----------------------------------
    def run(
        self,
        op: str,
        args: Sequence[Any],
        *,
        out: Sequence[Tensor] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, ...]: ...

    # -- the optional graph path --------------------------------------------
    def compile_graph(self, graph: Graph) -> "CompiledGraph | None": ...
    def capture(self, region: GraphRegion, *, warmup: int = 3) -> "CapturedGraph | None": ...

    def flush(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class CompiledGraph(Protocol):
    """An ahead-of-time compiled graph, runnable on the device."""

    def run(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]: ...
    def close(self) -> None: ...


@runtime_checkable
class CapturedGraph(Protocol):
    """A stream-captured region, replayed with new inputs of the same shape."""

    def replay(self, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]: ...