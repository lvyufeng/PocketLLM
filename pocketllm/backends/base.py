"""The machine every backend shares, so a backend writes only what is its own.

A backend has two halves: a *declaration* (which ops, over what domain, with
what graph path) and an *implementation*.  The declaration is where the stub
backends stop -- and it is not a throwaway, because dispatch, the conformance
harness and ``pocketllm devices`` all read it, so a stub is already testable
against the ABI before a line of native code exists.

:class:`DeclaredBackend` is that declaration half.  A backend subclass supplies
``name``/``device_kind``/``version`` and the tables; it gets ``available``,
``graph``, ``compile_spec`` and ``open`` for free.  :class:`UnimplementedSession`
is the other half of being a stub: every method raises
:class:`~pocketllm.kernels.errors.BackendNotImplementedError` naming the runtime
it is waiting for, which is what makes the pending work visible instead of
implicit in an ``ImportError``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pocketllm.kernels.backend import (
    Backend,
    Capability,
    CompileSpec,
    GraphCapability,
)
from pocketllm.kernels.buffer import Buffer
from pocketllm.kernels.device import Device
from pocketllm.kernels.errors import BackendNotImplementedError
from pocketllm.kernels.graph import Graph, GraphRegion
from pocketllm.kernels.tensor import Tensor

__all__ = [
    "BackendUnavailable",
    "RuntimeProbe",
    "DeclaredBackend",
    "StubBackend",
    "UnimplementedSession",
    "CAPABILITIES_BY_NAME",
]


class BackendUnavailable(RuntimeError):
    """A backend was selected but its runtime is not present on this host.

    Raised by :func:`pocketllm.backends.registry.get` in strict mode.  Distinct
    from :class:`~pocketllm.kernels.errors.BackendNotImplementedError`, which is
    "the runtime is here but this backend has not been written" -- the two want
    different fixes, so they are different errors.
    """


class RuntimeProbe:
    """A cheap "is this runtime here?" check that may not import the runtime.

    Three kinds of evidence, in the order a backend should prefer them: a Python
    module that can be *located* (``importlib.util.find_spec`` does not execute
    it), a shared library on the loader path (``ctypes.util.find_library`` reads
    the dynamic loader's cache without loading anything), and a device node.

    The restriction is real and testable: importing a runtime to ask whether it
    is installed is exactly the cost this design exists to avoid, and a probe
    that does it would make ``pocketllm devices`` take as long as a model load on
    the machines that have the runtime.
    """

    def __init__(
        self,
        *,
        modules: Sequence[str] = (),
        libraries: Sequence[str] = (),
        device_nodes: Sequence[str] = (),
        env: Sequence[str] = (),
        platforms: Sequence[str] = (),
    ) -> None:
        self.modules = tuple(modules)
        self.libraries = tuple(libraries)
        self.device_nodes = tuple(device_nodes)
        self.env = tuple(env)
        #: A hard gate: ``sys.platform.startswith()`` on one of these.  A
        #: backend can be structurally impossible on a host that nevertheless has
        #: its runtime installed -- torch is present on Linux and ``mps`` is not
        #: -- so a module probe alone would report a false positive there.
        self.platforms = tuple(platforms)

    def __call__(self) -> bool:
        if self.platforms and not any(sys.platform.startswith(p) for p in self.platforms):
            return False
        return bool(self.evidence()) or any(os.environ.get(name) for name in self.env)

    def evidence(self) -> tuple[str, ...]:
        """What was actually found, so ``pocketllm devices`` can name it."""
        found: list[str] = []
        for name in self.modules:
            try:
                if importlib.util.find_spec(name) is not None:
                    found.append(f"python module {name}")
            except (ImportError, ValueError):
                # A parent package that is absent makes find_spec raise rather
                # than return None.  That is still "not installed".
                continue
        for name in self.libraries:
            try:
                # find_library consults the dynamic loader's cache and the
                # standard paths; it does not dlopen, so a wrong name costs a
                # directory scan rather than a crash.  It is also only a name
                # match -- ``acl`` finds BSD's POSIX ACL library on Linux -- so
                # a backend with a short library name should pair it with a
                # device node or an env var, which is what the params after
                # ``libraries`` are for.
                import ctypes.util

                if ctypes.util.find_library(name):
                    found.append(f"library {name}")
            except OSError:
                continue
        for name in self.device_nodes:
            if Path(name).exists():
                found.append(f"device {name}")
        for name in self.env:
            if os.environ.get(name):
                found.append(f"env {name}")
        return tuple(found)


class DeclaredBackend:
    """A backend described by tables, with no implementation of its own.

    Subclasses override the class attributes and nothing else.  The methods are
    deliberately generic: a backend that needs different behaviour overrides
    them, but a stub never should.
    """

    #: The registry name, and what ``--backend`` takes.
    name: str = ""
    #: The device kind this backend opens.  May differ from ``name`` -- the
    #: reference backend is named ``reference`` and runs on ``cpu``.
    device_kind: str = ""
    #: Free-form, surfaced by ``pocketllm devices``.  A stub says so here.
    version: str = "0.0.0"
    #: One line on what the backend is for, for the device listing.
    summary: str = ""

    #: ``(op, dtypes, quants)`` triples, before the op domain table refines them.
    op_table: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    #: ``{op: rank}``, a documented relative cost among equal matches.
    ranks: Mapping[str, int] = {}
    #: ``(op, extra_dtypes)`` -- per-op dtypes that differ from the op table's.
    op_dtypes: Mapping[str, tuple[str, ...]] = {}
    #: The accelerated path, if any.  Default: none.
    graph_capability: GraphCapability = GraphCapability()
    #: What an offline compile needs, or ``None``.
    compile_spec_: CompileSpec | None = None
    #: How to tell whether this backend can be loaded here, without importing it.
    probe: RuntimeProbe = RuntimeProbe()
    #: Why it cannot be, in the operator's terms, when the probe says no.
    missing_dependency: str = ""
    #: An optional predicate over ``(args, attrs)`` refining one op's domain
    #: beyond its declared types.  Keyed by op name; see
    #: :class:`pocketllm.kernels.backend.Capability.accepts`.
    accepts: Mapping[str, Any] = {}

    is_reference: bool = False

    def available(self) -> bool:
        """Whether this backend's runtime is present.  Never imports it."""
        if os.environ.get("POCKETLLM_FAKE_BACKEND") == self.name:
            # An explicit override for tests and for CI that has no device: the
            # conformance harness runs the reference backend's own session here,
            # so a stub's *declaration* can be checked where its runtime is
            # absent.  It is opt-in by name and never on by default.
            return True
        return bool(self.probe())

    def capabilities(self) -> tuple[Capability, ...]:
        from pocketllm.kernels.dtypes import quant_format

        out: list[Capability] = []
        for op, dtypes, quants in self.op_table:
            names = self.op_dtypes.get(op, dtypes)
            out.append(
                Capability(
                    op=op,
                    dtypes=_dtypes(names),
                    quants=frozenset(quant_format(q) for q in quants),
                    accepts=self.accepts.get(op),
                    rank=int(self.ranks.get(op, 100)),
                )
            )
        return tuple(out)

    def graph(self) -> GraphCapability:
        return self.graph_capability

    def compile_spec(self) -> CompileSpec | None:
        return self.compile_spec_

    def open(self, device: Device, *, options: Mapping[str, Any] | None = None):
        raise BackendUnavailable(
            f"{self.name}: this backend declares capabilities but has no session; "
            f"see pocketllm/backends/{self.name}/README.md"
        )


class StubBackend(DeclaredBackend):
    """A declared backend that opens a session which refuses every call.

    This is the whole of a stub: the declaration is real (dispatch and the
    conformance harness read it), and the session is an
    :class:`UnimplementedSession` that names the missing runtime.  A backend
    graduates by overriding ``open`` with a real session, at which point this
    class stops being in its way.
    """

    def open(self, device: Device, *, options: Mapping[str, Any] | None = None) -> UnimplementedSession:
        return UnimplementedSession(self, device)


class UnimplementedSession:
    """The session a stub backend opens: every method names what it is waiting for.

    It holds the session's real state -- the backend and the device -- so a
    caller can still ask which device it failed to use, and it refuses each ABI
    call by name rather than with an ``AttributeError``.  ``describe`` is the
    single string every refusal repeats, and it is written to be read by an
    operator who has just seen the failure: what is missing, and where the
    instructions are.
    """

    def __init__(self, backend: DeclaredBackend, device: Device) -> None:
        self.backend = backend
        self.device = device

    @property
    def describe(self) -> str:
        backend = self.backend
        if backend.missing_dependency:
            return (
                f"{backend.name} session on {self.device}: needs {backend.missing_dependency}; "
                f"see pocketllm/backends/{backend.name}/README.md"
            )
        return f"{backend.name} session on {self.device}: not implemented yet"

    def alloc(self, nbytes: int, *, align: int = 64) -> Buffer:
        raise BackendNotImplementedError(self.describe)

    def free(self, buffer: Buffer) -> None:
        raise BackendNotImplementedError(self.describe)

    def to_device(self, host: memoryview, desc) -> Tensor:
        raise BackendNotImplementedError(self.describe)

    def to_host(self, tensor: Tensor) -> memoryview:
        raise BackendNotImplementedError(self.describe)

    def run(
        self,
        op: str,
        args: Sequence[Any],
        *,
        out: Sequence[Tensor] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, ...]:
        raise BackendNotImplementedError(self.describe)

    def compile_graph(self, graph: Graph):
        return None

    def capture(self, region: GraphRegion, *, warmup: int = 3):
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _dtypes(names: Sequence[str]) -> frozenset:
    from pocketllm.kernels.dtypes import DType

    by_name = {member.value: member for member in DType}
    try:
        return frozenset(by_name[name] for name in names)
    except KeyError as exc:
        raise KeyError(f"unknown dtype {exc.args[0]!r} in a backend declaration") from exc


#: ``DType`` values a quantized tensor's *activations* may take.  Quantized
#: backends and the reference backend share it: a block is decoded to the
#: activation's type, so this is the set they all admit.
FLOAT_DTYPES: tuple[str, ...] = ("f32", "f16", "bf16")

#: Every quant format the ABI names, used by the backends that consume all of
#: them.  Imported lazily so this module does not drag the format table in when
#: a caller only wanted the probe machinery.
CAPABILITIES_BY_NAME: dict[str, DeclaredBackend] = {}