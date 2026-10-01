"""Dispatch: which backend runs an op, and why not, stated as data.

Resolution reads declarations only -- schemas and capabilities -- never a
device, which is what makes it testable on a host with no accelerator.  The
tests below are mostly about the *refusal* path, because "why was this call
rejected" was the expensive question in the old tree and ``explain`` exists to
answer it.
"""

from __future__ import annotations

import pytest

from pocketllm.kernels import (
    Capability,
    Device,
    DType,
    GraphCapability,
    GraphMode,
    NoBackendError,
    QuantFormat,
    TensorDesc,
    quant_format,
)
from pocketllm.kernels.backend import CompileSpec
from pocketllm.kernels.dispatch import Dispatcher
from pocketllm.kernels.tensor import Tensor


class _Backend:
    """A declaration-only backend: enough to resolve against, nothing to run."""

    def __init__(
        self,
        name: str,
        device_kind: str,
        caps: tuple[Capability, ...],
        *,
        is_reference: bool = False,
        available: bool = True,
    ) -> None:
        self.name = name
        self.device_kind = device_kind
        self.version = "test"
        self.is_reference = is_reference
        self._caps = caps
        self._available = available

    def available(self) -> bool:
        return self._available

    def capabilities(self) -> tuple[Capability, ...]:
        return self._caps

    def graph(self) -> GraphCapability:
        return GraphCapability()

    def compile_spec(self) -> CompileSpec | None:
        return None

    def open(self, device, *, options=None):  # pragma: no cover - never opened here
        raise NotImplementedError


def _tensor(desc_shape, dtype=DType.F32, quant: QuantFormat | None = None) -> Tensor:
    from pocketllm.kernels.buffer import DeviceBuffer

    # A desc carries exactly one of dtype / quant, so asking for a quantized
    # tensor means *not* also passing the default element type.
    desc = TensorDesc(desc_shape, dtype=None if quant is not None else dtype, quant=quant)
    return Tensor(desc, DeviceBuffer(device=Device("cpu"), nbytes=desc.nbytes))


def _cpu_backend(supported: frozenset[str], *, dtypes=frozenset({DType.F32}), quants=frozenset()) -> _Backend:
    return _Backend(
        "cpu",
        "cpu",
        tuple(Capability(op=name, dtypes=dtypes, quants=quants) for name in sorted(supported)),
    )


def _reference() -> _Backend:
    # The reference backend implements every op, in f32, over every quant format.
    from pocketllm.kernels import OPS, QUANT_FORMATS

    return _Backend(
        "reference",
        "cpu",
        tuple(
            Capability(op=name, dtypes=frozenset({DType.F32}), quants=frozenset(QUANT_FORMATS.values()))
            for name in sorted(OPS.names())
        ),
        is_reference=True,
    )


def test_dispatch_picks_the_device_backend_over_reference() -> None:
    cpu = _cpu_backend(frozenset({"add"}))
    dispatcher = Dispatcher([cpu, _reference()])
    x = _tensor((4, 4))
    chosen = dispatcher.resolve("add", [x, x], Device("cpu"))
    assert chosen.backend.name == "cpu"
    assert chosen.is_reference is False


def test_reference_is_a_candidate_on_an_unimplemented_device() -> None:
    """A wrong device yields a correct-but-slow answer, not 'no backend'."""
    dispatcher = Dispatcher([_cpu_backend(frozenset({"add"})), _reference()])
    x = _tensor((4, 4))
    chosen = dispatcher.resolve("add", [x, x], Device("qnn"))
    assert chosen.backend.name == "reference"
    assert chosen.is_reference is True


def test_reference_fallback_can_be_forbidden() -> None:
    """`serve` forbids it: a 29B model on numpy is a ten-minute first token."""
    dispatcher = Dispatcher([_cpu_backend(frozenset({"gemm"})), _reference()], allow_reference_fallback=False)
    x = _tensor((4, 4))
    with pytest.raises(NoBackendError):
        dispatcher.resolve("add", [x, x], Device("cpu"))


def test_capability_domain_is_enforced_by_dtype() -> None:
    f16_only = _Backend("f16only", "cpu", (Capability(op="add", dtypes=frozenset({DType.F16})),))
    dispatcher = Dispatcher([f16_only], allow_reference_fallback=False)
    f32_pair = [_tensor((2, 2), dtype=DType.F32)] * 2
    with pytest.raises(NoBackendError):
        dispatcher.resolve("add", f32_pair, Device("cpu"))
    f16_pair = [_tensor((2, 2), dtype=DType.F16)] * 2
    assert dispatcher.resolve("add", f16_pair, Device("cpu")).backend.name == "f16only"


def test_capability_domain_is_enforced_by_quant_format() -> None:
    only_iq4 = _Backend(
        "iq4only",
        "cpu",
        (Capability(op="gemm_quant", dtypes=frozenset({DType.F32}), quants=frozenset({quant_format("iq4_nl")})),),
    )
    dispatcher = Dispatcher([only_iq4], allow_reference_fallback=False)
    x = _tensor((4, 256), dtype=DType.F32)
    w_iq4 = _tensor((8, 144), quant=quant_format("iq4_nl"))
    assert dispatcher.resolve("gemm_quant", [x, w_iq4], Device("cpu")).backend.name == "iq4only"

    w_q4k = _tensor((8, 144), quant=quant_format("q4_k"))
    with pytest.raises(NoBackendError):
        dispatcher.resolve("gemm_quant", [x, w_q4k], Device("cpu"))


def test_accepts_predicate_can_narrow_beyond_the_schema() -> None:
    decode_only = _Backend(
        "decode",
        "cpu",
        (Capability(op="gemm", dtypes=frozenset({DType.F32}), accepts=lambda args, attrs: args[0].desc.shape[0] == 1),),
    )
    dispatcher = Dispatcher([decode_only], allow_reference_fallback=False)
    wide = [_tensor((4, 8)), _tensor((8, 8))]
    with pytest.raises(NoBackendError):
        dispatcher.resolve("gemm", wide, Device("cpu"))
    narrow = [_tensor((1, 8)), _tensor((8, 8))]
    assert dispatcher.resolve("gemm", narrow, Device("cpu")).backend.name == "decode"


def test_unavailable_backend_is_skipped_with_a_reason() -> None:
    missing = _Backend("cuda", "cuda", (Capability(op="add", dtypes=frozenset({DType.F32})),), available=False)
    dispatcher = Dispatcher([missing, _reference()])
    x = _tensor((2, 2))
    resolution = dispatcher.explain("add", [x, x], Device("cuda"))
    assert resolution.chosen is not None and resolution.chosen.backend.name == "reference"
    assert ("cuda", "not available on this host") in resolution.rejected


def test_explain_names_the_missing_dtype() -> None:
    f16_only = _Backend("f16only", "cpu", (Capability(op="add", dtypes=frozenset({DType.F16})),))
    dispatcher = Dispatcher([f16_only], allow_reference_fallback=False)
    resolution = dispatcher.explain("add", [_tensor((2, 2))] * 2, Device("cpu"))
    assert resolution.chosen is None
    assert "f32" in resolution.reason()


def test_explain_names_the_missing_quant_format() -> None:
    only_iq4 = _Backend(
        "iq4only",
        "cpu",
        (Capability(op="gemm_quant", dtypes=frozenset({DType.F32}), quants=frozenset({quant_format("iq4_nl")})),),
    )
    dispatcher = Dispatcher([only_iq4], allow_reference_fallback=False)
    resolution = dispatcher.explain(
        "gemm_quant",
        [_tensor((4, 256)), _tensor((8, 144), quant=quant_format("q6_k"))],
        Device("cpu"),
    )
    assert "q6_k" in resolution.reason()


def test_backend_that_declares_nothing_is_reported_as_such() -> None:
    nothing = _Backend("empty", "cpu", ())
    dispatcher = Dispatcher([nothing], allow_reference_fallback=False)
    resolution = dispatcher.explain("add", [_tensor((2, 2))] * 2, Device("cpu"))
    assert ("empty", "does not declare 'add'") in resolution.rejected


def test_preference_orders_equal_candidates() -> None:
    a = _cpu_backend(frozenset({"add"}))
    b = _Backend("b", "cpu", (Capability(op="add", dtypes=frozenset({DType.F32})),))
    x = _tensor((2, 2))
    assert Dispatcher([a, b], preference=("b", "cpu")).resolve("add", [x, x], Device("cpu")).backend.name == "b"


def test_lower_rank_wins_among_equals() -> None:
    slow = _Backend("slow", "cpu", (Capability(op="add", dtypes=frozenset({DType.F32}), rank=900),))
    fast = _Backend("fast", "cpu", (Capability(op="add", dtypes=frozenset({DType.F32}), rank=10),))
    x = _tensor((2, 2))
    assert Dispatcher([slow, fast]).resolve("add", [x, x], Device("cpu")).backend.name == "fast"


def test_unknown_op_is_refused() -> None:
    from pocketllm.kernels.errors import OpNotDeclaredError

    dispatcher = Dispatcher([_reference()])
    with pytest.raises(OpNotDeclaredError):
        dispatcher.resolve("no_such_op", [], Device("cpu"))


def test_auxiliary_index_arguments_are_not_part_of_the_domain() -> None:
    """An op's index argument must not narrow its dtype domain.

    ``embedding`` takes ``tokens: i32`` beside a float table, and ``attention``
    takes ``positions: i32`` beside f32 queries.  A dispatcher that folded every
    tensor argument into the domain would demand that every backend declare i32,
    and every one would fail -- which is exactly what happened the first time a
    real graph was executed.  The schema's ``ArgSpec.dtype`` marks those
    arguments, and dispatch must read it.
    """
    cpu = _cpu_backend(frozenset({"embedding"}), dtypes=frozenset({DType.F32}))
    dispatcher = Dispatcher([cpu], allow_reference_fallback=False)

    table = _tensor((32, 8))
    tokens = _tensor((4,), dtype=DType.I32)
    resolved = dispatcher.resolve("embedding", [tokens, table], Device("cpu"))
    assert resolved.backend.name == "cpu"


def test_an_unsupported_data_dtype_is_still_refused() -> None:
    """The exemption above must be the marker's doing, not a blanket pass.

    A control for the test that follows it: an op whose arguments are all
    *unmarked* data still has its dtype domain enforced, so removing the marker
    would have broken this one rather than silently relaxing everything.
    """
    cpu = _cpu_backend(frozenset({"add"}), dtypes=frozenset({DType.F32}))
    dispatcher = Dispatcher([cpu], allow_reference_fallback=False)
    with pytest.raises(NoBackendError, match="no dtype i64"):
        dispatcher.resolve("add", [_tensor((4, 4), dtype=DType.I64)] * 2, Device("cpu"))