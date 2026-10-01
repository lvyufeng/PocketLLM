"""Every backend's declaration, checked without needing its runtime.

A stub's declaration is the only part of it that exists, so it is the part that
has to be right.  These tests read the tables and check them against the ABI and
against the harness: an op name that is not in the registry, a quant format the
ABI does not name, a declared op the harness has no sample for, or a device kind
that does not match the registry key would all be invisible until somebody
plugged the hardware in -- which is exactly the moment you cannot debug it.

Nothing here opens a device, so it runs on any host, including CI, and covers
every backend in the table rather than the two that happen to be loadable.
"""

from __future__ import annotations

import pytest

from pocketllm.backends import registry
from pocketllm.backends.base import DeclaredBackend
from pocketllm.kernels.dtypes import DType, QUANT_FORMATS
from pocketllm.kernels.registry import OPS

from .conftest import sample_args


def _all_backends() -> list[tuple[str, DeclaredBackend]]:
    """Every backend in the tree, loaded regardless of availability."""
    return [(entry.name, entry.factory()) for entry in registry.BACKENDS.values()]


_IDS = [name for name, _ in _all_backends()]


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_declared_ops_are_in_the_abi(name, backend):
    unknown = {cap.op for cap in backend.capabilities()} - OPS.names()
    assert not unknown, f"{name} declares ops the ABI does not define: {sorted(unknown)}"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_declared_quants_are_named_by_the_abi(name, backend):
    for cap in backend.capabilities():
        unknown = {q.name for q in cap.quants} - set(QUANT_FORMATS)
        assert not unknown, f"{name} declares unknown quant formats for {cap.op}: {sorted(unknown)}"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_declared_dtypes_are_real(name, backend):
    known = {member.name for member in DType}
    for cap in backend.capabilities():
        unknown = {d.name for d in cap.dtypes} - known
        assert not unknown, f"{name} declares unknown dtypes for {cap.op}: {sorted(unknown)}"
        assert cap.dtypes, f"{name} declares {cap.op} with an empty dtype set, which admits nothing"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_op_names_are_unique_and_domains_are_sane(name, backend):
    seen: set[str] = set()
    for cap in backend.capabilities():
        assert cap.op not in seen, f"{name} declares {cap.op!r} twice"
        seen.add(cap.op)
        assert cap.rank >= 0, f"{name}: a negative rank for {cap.op} inverts the preference order"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_graph_capability_is_consistent(name, backend):
    graph = backend.graph()
    if not graph.supported:
        assert not graph.captures, f"{name} captures nothing but lists {sorted(graph.captures)}"
        return
    assert graph.captures, f"{name} claims a graph path but captures no ops"
    unknown = set(graph.captures) - OPS.names()
    assert not unknown, f"{name} captures ops the ABI does not define: {sorted(unknown)}"
    # A backend must not claim to capture an op it does not declare: the engine
    # would hand it a region it cannot run.
    declared = {cap.op for cap in backend.capabilities()}
    assert set(graph.captures) <= declared, (
        f"{name} captures ops it does not declare: {sorted(set(graph.captures) - declared)}"
    )
    # A compile spec describes an *offline* path and does not require the graph
    # mode to be AOT: a backend whose primary path is stream capture (Ascend's
    # aclgraph) can still offer `atc` as a second, offline route.  What must hold
    # is that an offline artifact is not described as a captured stream.
    spec = backend.compile_spec()
    if spec is not None:
        assert spec.artifact_format, f"{name} declares a compile spec that produces no artifact"
        assert spec.target_arch, f"{name} declares a compile spec that names no target"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_every_declared_op_has_a_conformance_sample(name, backend, reference_session):
    """The harness can build every call a backend claims it will run.

    Uses the *reference* session to build the samples, so this checks the
    declaration and the harness agree without needing the declared backend's
    runtime -- which is the only way to check a phone or an NPU declaration from
    a development host.
    """
    missing: list[str] = []
    for cap in backend.capabilities():
        quants = [q.name for q in sorted(cap.quants, key=lambda q: q.name)] or [None]
        for quant in quants:
            try:
                args, attrs = sample_args(cap.op, reference_session, quant=quant)
            except KeyError:
                missing.append(f"{cap.op}{'/' + quant if quant else ''}")
                continue
            # And the call is *well-formed* against its own schema, which is the
            # check that catches an arity or shape mistake in the harness rather
            # than in the backend.
            schema = OPS.get(cap.op)
            schema.infer(args, attrs)
    assert not missing, f"{name} declares ops the harness cannot build a call for: {missing}"


@pytest.mark.parametrize("name,backend", _all_backends(), ids=_IDS)
def test_device_kind_is_declared(name, backend):
    assert backend.device_kind, f"{name} opens no device kind"
    assert isinstance(backend.device_kind, str)
    assert backend.summary, f"{name} has no summary for the device listing"


def test_reference_is_the_only_reference_backend():
    """``is_reference`` drives dispatch's ordering, so exactly one sets it."""
    marked = [name for name, backend in _all_backends() if getattr(backend, "is_reference", False)]
    assert marked == ["reference"]


def test_reference_backend_is_always_available():
    """numpy is a base dependency, so the oracle is never missing."""
    backend = dict(_all_backends())["reference"]
    assert backend.available()


def test_registry_keys_match_backend_names():
    for key, entry in registry.BACKENDS.items():
        assert key == entry.name, f"registry key {key!r} != entry name {entry.name!r}"
        assert entry.module.endswith(key), f"{key}: module {entry.module} does not look like its package"


def _declared_but_absent() -> list[str]:
    """Backends whose runtime the probe cannot find on this host."""
    return [name for name, backend in _all_backends() if not backend.probe()]


def test_fake_backend_override_is_named_and_not_global(monkeypatch):
    """``POCKETLLM_FAKE_BACKEND`` makes exactly the named backend loadable.

    It exists so the conformance harness can run a stub's *declaration* on a host
    without its runtime, which is why it has to be checked: a bug here would
    either leave the harness silently skipping everything, or -- worse -- mark a
    backend available that nothing can open, turning a skip into a failure at the
    point of use.  The override is per-name; it must not flip the whole table.
    """
    absent = _declared_but_absent()
    if not absent:
        pytest.skip("every backend's runtime is present on this host; nothing to fake")
    target = absent[0]

    monkeypatch.setenv("POCKETLLM_FAKE_BACKEND", target)
    by_name = dict(_all_backends())

    assert by_name[target].available(), f"{target} was named but is still reported unavailable"
    for other in absent:
        if other != target:
            assert not by_name[other].available(), (
                f"naming {target} also made {other} available; the override is not per-name"
            )


def test_fake_backend_override_does_not_rewrite_the_declaration(monkeypatch):
    """Faking availability changes ``available`` and nothing else.

    The harness relies on this: it runs the *declared* ops of a stub whose runtime
    is absent, so the tables it reads must be the stub's own, unchanged.  A
    backend that grew ops when faked would be a backend that lies about what it
    can do, which is the one thing a declaration may never do.
    """
    absent = _declared_but_absent()
    if not absent:
        pytest.skip("every backend's runtime is present on this host; nothing to fake")
    target = absent[0]
    backend = dict(_all_backends())[target]

    before = {cap.op: (cap.dtypes, cap.quants) for cap in backend.capabilities()}
    monkeypatch.setenv("POCKETLLM_FAKE_BACKEND", target)
    after = {cap.op: (cap.dtypes, cap.quants) for cap in backend.capabilities()}

    assert before == after, f"faking {target} changed its declared capabilities"
    assert backend.graph() == backend.graph_capability