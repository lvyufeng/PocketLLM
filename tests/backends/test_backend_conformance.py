"""Every declared op, on every backend that claims it, against the reference.

This is the test that keeps a declaration honest.  A backend that lists an op it
cannot run fails here; a backend whose numerics drift from the reference fails
here; and a backend with no runtime, or one whose session is still a stub, skips
-- visibly, with the stub's own message, never silently.

The parameterization is *per declared capability*, not per backend, so a stub
that declares half the vocabulary is checked on exactly the half it declared.
Adding an op to a backend's table is therefore what enrols it here; there is no
second list to keep in step.
"""

from __future__ import annotations

import pytest

from pocketllm.backends import registry
from pocketllm.kernels.device import Device
from pocketllm.kernels.errors import BackendNotImplementedError, KernelError
from pocketllm.kernels.registry import OPS
from pocketllm.quant.formats import FORMATS

from .conftest import compare, read, sample_args


def _cases():
    """One ``(backend, op, quant)`` case per declared capability."""
    for backend in registry.available_backends():
        for cap in backend.capabilities():
            if cap.op not in OPS.names():
                raise AssertionError(f"{backend.name} declares {cap.op!r}, which the ABI does not define")
            if not cap.quants:
                yield backend.name, cap.op, None
                continue
            for quant in sorted(cap.quants, key=lambda q: q.name):
                yield backend.name, cap.op, quant.name


_CASES = list(_cases())
_CASES = [c for c in _CASES if c[0] != "reference"]


def _open(backend):
    return backend.open(Device(backend.device_kind))


@pytest.mark.parametrize(
    "backend_name,op,quant",
    [pytest.param(n, o, q, id=f"{n}-{o}{'-' + q if q else ''}") for n, o, q in _CASES],
)
def test_op_matches_reference(backend_name, op, quant, reference_session, backends):
    backend = next(b for b in backends if b.name == backend_name)
    if quant is not None and quant not in FORMATS:
        pytest.skip(f"{quant} is declared by the ABI but no decoder exists in pocketllm.quant")

    session = _open(backend)
    if type(session).__name__ == "UnimplementedSession":
        pytest.skip(session.describe)

    args, attrs = sample_args(op, session, quant=quant)
    ref_args, ref_attrs = sample_args(op, reference_session, quant=quant)

    try:
        (got,) = session.run(op, args, attrs=attrs)
    except BackendNotImplementedError as exc:
        pytest.skip(str(exc))
    except KernelError as exc:
        pytest.fail(f"{backend_name} declares {op!r} but refused it: {exc}")

    (want,) = reference_session.run(op, ref_args, attrs=ref_attrs)
    compare(op, read(reference_session, want), read(session, got), quant=quant)


def test_every_backend_was_exercised(backends):
    """The parameterization covers every loadable backend, so nothing is invisible."""
    names = {name for name, _, _ in _CASES}
    expected = {b.name for b in backends if b.name != "reference"}
    assert names == expected


def test_stub_sessions_refuse_by_name(backends):
    """A stub session names its missing runtime; it does not AttributeError."""
    from pocketllm.backends.base import UnimplementedSession

    checked = 0
    for backend in backends:
        session = backend.open(Device(backend.device_kind))
        if not isinstance(session, UnimplementedSession):
            continue
        checked += 1
        with pytest.raises(BackendNotImplementedError) as excinfo:
            session.run("argmax", [])
        assert backend.name in str(excinfo.value)
        # A stub must also refuse the memory calls, not just the op call: the
        # engine allocates before it runs, so a stub that only refused `run`
        # would fail with an AttributeError on the way in.
        with pytest.raises(BackendNotImplementedError):
            session.alloc(16)
        assert session.compile_graph(None) is None
        assert session.capture(None) is None
        session.flush()
        session.close()
    assert checked, "no stub backend was reachable; the refusal path is untested"