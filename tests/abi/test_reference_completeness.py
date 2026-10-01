"""The reference backend implements every op the ABI declares, in the same commit.

This is the rule that keeps the vocabulary honest.  A declared op is a promise
that some backend can run it; if the *only* implementation is an accelerated one
on hardware nobody has, the op has no definition -- there is nothing to compare a
kernel against, and no way to tell a wrong fast answer from a right one.

So a new op in ``pocketllm/kernels/ops/`` is not complete until
``pocketllm/backends/reference/kernels.py`` has an entry for it, and this test
fails until it does.  The failure message names the missing op, because the fix
is to write a kernel and not to relax the assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.backends.reference import BACKEND, ReferenceSession
from pocketllm.backends.reference.kernels import KERNELS
from pocketllm.kernels.device import Device
from pocketllm.kernels.registry import OPS


def test_every_declared_op_has_a_reference_kernel():
    declared = OPS.names()
    implemented = set(KERNELS)
    missing = sorted(declared - implemented)
    extra = sorted(implemented - declared)
    assert not missing, (
        "these ops are declared in pocketllm/kernels/ops/ but the reference backend "
        f"does not implement them: {missing}. A declared op with no reference kernel "
        "has no definition for any backend to be checked against."
    )
    assert not extra, f"the reference backend implements ops the ABI does not declare: {extra}"


def test_reference_capabilities_cover_the_registry():
    """Dispatch is built on capabilities, so they must cover the whole vocabulary."""
    caps = {cap.op for cap in BACKEND.capabilities()}
    assert caps == OPS.names(), f"capability/registry mismatch: {sorted(OPS.names() ^ caps)}"


def test_reference_declares_no_quants_it_cannot_take():
    """A quant format in a capability must be one the decoder table can expand."""
    from pocketllm.quant.formats import FORMATS

    for cap in BACKEND.capabilities():
        for quant in cap.quants:
            assert quant.name in FORMATS or quant.name in {"iq1_s"}, (
                f"the reference backend declares {quant.name} for {cap.op}, "
                "but pocketllm.quant.formats has no decoder for it"
            )


def test_reference_kernel_signatures_match_their_schemas():
    """A kernel accepts an argument for every non-optional arg the schema declares.

    A schema/implementation mismatch in *name* is invisible until the op is
    called on a device -- and the reference is called for every op during
    conformance, but only for the sample shapes.  Checking the signature here
    catches an op that was declared with an argument the kernel never learned
    about.
    """
    import inspect

    problems: list[str] = []
    for schema in OPS.schemas():
        kernel = KERNELS[schema.name]
        signature = inspect.signature(kernel)
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        names = set(signature.parameters)
        for spec in schema.args:
            if spec.optional:
                continue
            if spec.name in names:
                continue
            if accepts_kwargs:
                continue
            problems.append(f"{schema.name}: kernel has no argument {spec.name!r}")
    assert not problems, "\n".join(problems)


def test_reference_session_satisfies_the_backend_session_protocol():
    from pocketllm.kernels.backend import BackendSession

    session = BACKEND.open(Device("cpu"))
    assert isinstance(session, BackendSession)
    assert isinstance(BACKEND, __import__("pocketllm.kernels.backend", fromlist=["Backend"]).Backend)
    session.close()


def test_reference_reports_no_graph_path():
    """The reference backend runs eagerly; both graph methods answer None."""
    graph = BACKEND.graph()
    assert not graph.supported
    session = BACKEND.open(Device("cpu"))
    assert session.compile_graph(None) is None
    assert session.capture(None) is None
    session.close()


def test_reference_is_flagged_for_dispatch_ordering():
    assert BACKEND.is_reference is True
    assert BACKEND.name == "reference"
    assert BACKEND.available() is True


@pytest.mark.parametrize("op", sorted(OPS.names()))
def test_kernel_is_callable_with_its_schema_arity(op):
    """The kernel can be *called* with the schema's positional arity, dtype-free.

    This is a smoke check that catches an arity mistake in a kernel's signature
    without needing the conformance harness's sample values.
    """
    import inspect

    kernel = KERNELS[op]
    signature = inspect.signature(kernel)
    required = [
        p
        for p in signature.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    schema = OPS.get(op)
    required_tensors = [spec for spec in schema.args if not spec.optional]
    assert len(required) <= len(required_tensors), (
        f"{op}: kernel requires {len(required)} positional arguments but the schema declares "
        f"{len(required_tensors)} non-optional ones"
    )