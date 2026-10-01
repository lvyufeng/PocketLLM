"""The reference backend, driven through the ABI, checked against the schema.

The conformance harness compares a backend to the reference; this file checks
the reference itself.  Two things are being asserted, and they are different:

* **The ABI works end to end.**  Every call goes through ``session.run``, which
  means through the schema's shape inference, the session's dtype marshalling,
  the kernel, and the output buffer -- so a mistake in any of those layers shows
  up here rather than on a device.
* **The reference agrees with its own semantics.**  Where the computation can be
  written out independently (a matmul, an RMS norm), the test does so; where it
  cannot (attention's masking), it asserts a property the semantics implies.

The harness's ability to *fail* is checked too: a deliberately wrong
implementation must not slip through :func:`~tests.backends.conftest.compare`.
A comparison that cannot fail is worse than no comparison, because it reads like
coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.backends.reference import BACKEND
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType, QUANT_FORMATS
from pocketllm.kernels.errors import KernelError
from pocketllm.kernels.registry import OPS
from pocketllm.kernels.tensor import Tensor, TensorDesc

from .conftest import compare, host_tensor, read, sample_args

_REFERENCE_OPS = sorted(OPS.names())


@pytest.fixture(scope="module")
def session():
    s = BACKEND.open(Device("cpu"))
    yield s
    s.close()


@pytest.mark.parametrize("op", _REFERENCE_OPS)
def test_op_runs_through_the_abi(op, session):
    """Every declared op runs, and its result matches the declared description."""
    args, attrs = sample_args(op, session)
    descs = OPS.get(op).infer(args, attrs)
    results = session.run(op, args, attrs=attrs)
    assert len(results) == len(descs)
    for tensor, desc in zip(results, descs):
        assert tensor.desc.shape == desc.shape, f"{op}: returned {tensor.desc.shape}, declared {desc.shape}"
        assert tensor.desc.dtype == desc.dtype, f"{op}: returned {tensor.desc.dtype}, declared {desc.dtype}"
        assert np.isfinite(read(session, tensor)).all(), f"{op}: non-finite output"


@pytest.mark.parametrize("op", _REFERENCE_OPS)
def test_op_accepts_a_preallocated_output(op, session):
    """``out=`` writes into the caller's tensor and returns that same tensor.

    This is the path a captured region uses -- the output buffer is fixed when
    the region is recorded -- so it is a contract, not an optimisation.
    """
    args, attrs = sample_args(op, session)
    descs = OPS.get(op).infer(args, attrs)
    supplied = [Tensor(desc, session.alloc(desc.nbytes, align=1)) for desc in descs]
    results = session.run(op, args, out=supplied, attrs=attrs)
    for got, want in zip(results, supplied):
        assert got is want, f"{op}: run(out=...) returned a fresh tensor instead of the supplied one"

    # And the supplied path must agree with the allocating one.
    fresh = session.run(op, args, attrs=attrs)
    for a, b in zip(supplied, fresh):
        compare(op, read(session, b), read(session, a), quant=None if not b.desc.is_quantized else "any")


def test_gemm_matches_numpy_matmul(session):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 256)).astype(np.float32)
    w = rng.standard_normal((5, 256)).astype(np.float32)
    bias = rng.standard_normal(5).astype(np.float32)
    (y,) = session.run("gemm", [host_tensor(session, x), host_tensor(session, w), host_tensor(session, bias)])
    np.testing.assert_allclose(read(session, y), x @ w.T + bias, rtol=1e-4)


def test_gemm_quant_matches_decoded_matmul(session):
    from pocketllm.quant import formats

    rng = np.random.default_rng(5)
    name = "iq4_xs"
    fmt = QUANT_FORMATS[name]
    desc = TensorDesc((5, 256), quant=fmt)
    blocks = rng.integers(0, 256, size=(5, 1, fmt.block_bytes), dtype=np.uint8)
    blocks[..., 0:2] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    buffer = session.alloc(desc.nbytes, align=1)
    buffer.host_view()[:] = blocks.tobytes()

    x = rng.standard_normal((3, 256)).astype(np.float32)
    (y,) = session.run("gemm_quant", [host_tensor(session, x), Tensor(desc, buffer)])
    expected = x @ formats.dequantize_row(name, blocks, 256).T
    np.testing.assert_allclose(read(session, y), expected, rtol=1e-4, atol=1e-5)


def test_attention_output_is_a_convex_combination(session):
    """Every output row is a weighted average of visible ``v`` rows, with weights summing to one.

    A property, not a re-derivation: whatever the mask does, the result must lie
    in the convex hull of the value vectors it attended to.  That catches an
    unnormalized softmax, a wrong scale applied after the softmax, and a value
    matrix read with the wrong head -- none of which a shape check would see.
    """
    rng = np.random.default_rng(6)
    q = rng.standard_normal((2, 2, 4)).astype(np.float32)
    k = rng.standard_normal((8, 2, 4)).astype(np.float32)
    v = rng.standard_normal((8, 2, 4)).astype(np.float32)
    positions = np.array([1, 5], dtype=np.int32)
    (out,) = session.run(
        "attention",
        [host_tensor(session, q), host_tensor(session, k), host_tensor(session, v), host_tensor(session, positions)],
        attrs={"softmax_scale": 0.5, "causal": True, "num_kv_heads": 2},
    )
    values = read(session, out)
    for i, end in enumerate((2, 6)):
        for h in range(2):
            span = v[:end, h, :]
            lo = np.minimum.reduce(span, axis=0) - 1e-5
            hi = np.maximum.reduce(span, axis=0) + 1e-5
            assert np.all(values[i, h] >= lo) and np.all(values[i, h] <= hi), (
                "attention output left the hull of its visible values"
            )


def test_rope_applies_the_tables_it_is_given(session):
    """With cos=1, sin=0 the rotation is the identity; with cos=0, sin=1 it is a quarter turn.

    Using unit tables rather than random ones is what makes the *layout* claim
    checkable: the two layouts permute which elements are paired, and at a
    quarter turn that permutation is directly visible in the output.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 2, 8)).astype(np.float32)
    positions = np.array([1, 4], dtype=np.int32)
    half = x.shape[-1] // 2

    identity_cos = np.ones((8, half), np.float32)
    zero_sin = np.zeros((8, half), np.float32)
    (out,) = session.run(
        "rope",
        [host_tensor(session, x), host_tensor(session, positions),
         host_tensor(session, identity_cos), host_tensor(session, zero_sin)],
        attrs={"layout": "split", "theta_base": 10000.0, "scaling": None},
    )
    np.testing.assert_allclose(read(session, out), x, rtol=1e-6)

    zero_cos = np.zeros((8, half), np.float32)
    one_sin = np.ones((8, half), np.float32)
    for layout in ("split", "interleaved"):
        (out,) = session.run(
            "rope",
            [host_tensor(session, x), host_tensor(session, positions),
             host_tensor(session, zero_cos), host_tensor(session, one_sin)],
            attrs={"layout": layout, "theta_base": 10000.0, "scaling": None},
        )
        got = read(session, out)
        if layout == "split":
            # (a, b) -> (-b, a) across the half boundary.
            np.testing.assert_allclose(got[..., :half], -x[..., half:], rtol=1e-6)
            np.testing.assert_allclose(got[..., half:], x[..., :half], rtol=1e-6)
        else:
            # (a, b) -> (-b, a) on adjacent pairs.
            np.testing.assert_allclose(got[..., 0::2], -x[..., 1::2], rtol=1e-6)
            np.testing.assert_allclose(got[..., 1::2], x[..., 0::2], rtol=1e-6)


def test_softmax_rows_sum_to_one(session):
    rng = np.random.default_rng(8)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    (out,) = session.run("softmax", [host_tensor(session, x)], attrs={"axis": -1})
    np.testing.assert_allclose(read(session, out).sum(-1), 1.0, rtol=1e-5)


def test_softmax_is_stable_for_large_logits(session):
    """A logit of 1e4 must not overflow, which is the whole point of the shift."""
    x = np.array([[1e4, 1e4 - 1.0, -1e4]], dtype=np.float32)
    (out,) = session.run("softmax", [host_tensor(session, x)], attrs={"axis": -1})
    values = read(session, out)
    assert np.isfinite(values).all()
    assert values[0, 0] > values[0, 1] > values[0, 2]


def test_topk_sample_respects_top_k(session):
    logits = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
    # A uniform of 0.999 must still land inside the top-2, never on token 2 or 3.
    for u in (0.0, 0.25, 0.75, 0.999):
        (tok,) = session.run(
            "topk_sample",
            [host_tensor(session, logits), host_tensor(session, np.array(u, np.float32))],
            attrs={"top_k": 2, "top_p": 1.0, "min_p": 0.0},
        )
        assert int(read(session, tok)) in (0, 1), f"uniform={u} drew outside the top-2"


def test_argmax_ties_take_the_lowest_index(session):
    (tok,) = session.run("argmax", [host_tensor(session, np.array([1.0, 3.0, 3.0], np.float32))])
    assert int(read(session, tok)) == 1


def test_temperature_refuses_a_non_positive_value(session):
    with pytest.raises(ValueError):
        session.run(
            "logits_temperature",
            [host_tensor(session, np.ones(4, np.float32))],
            attrs={"temperature": 0.0},
        )


def test_dtype_round_trip_through_the_session(session):
    """f16, bf16 and i32 survive a to_device/to_host round trip."""
    for values, dtype in (
        (np.array([1.5, -2.25, 0.0], np.float16), DType.F16),
        (np.array([1.5, -2.25, 0.0], np.float32), DType.F32),
        (np.array([1, -2, 300], np.int32), DType.I32),
    ):
        desc = TensorDesc(values.shape, dtype=dtype)
        tensor = session.to_device(memoryview(np.ascontiguousarray(values)).cast("B"), desc)
        got = read(session, tensor)
        np.testing.assert_array_equal(got, values)


def test_bf16_is_carried_as_raw_uint16_and_widened_exactly(session):
    """bfloat16 is the top half of a float32, so widening it must be lossless.

    numpy has no bfloat16, so the session stores one as ``uint16`` and shifts on
    the way out.  The test builds the packed form by hand -- ``float32 >> 16`` --
    and checks the session hands back exactly those values, which is the claim
    the dtype bridge rests on.
    """
    original = np.array([1.0, 2.5, -3.75, 0.15625], np.float32)
    packed = (original.view(np.uint32) >> 16).astype(np.uint16)
    desc = TensorDesc(original.shape, dtype=DType.BF16)
    tensor = session.to_device(memoryview(np.ascontiguousarray(packed)).cast("B"), desc)
    assert tensor.desc.nbytes == packed.nbytes
    got = read(session, tensor)
    assert got.dtype == np.float32
    # Widening is exact: every one of these values is representable in bf16.
    np.testing.assert_array_equal(got, original)


def test_bf16_narrowing_rounds_to_nearest(session):
    """Storing an f32 result into bf16 rounds, rather than truncating.

    The difference matters for a round trip: truncation biases every result
    toward zero, and a backend that truncated where the hardware rounds would
    drift from the reference by more than a tolerance should forgive.
    """
    from pocketllm.backends.reference.dtypes import read as read_dtype
    from pocketllm.backends.reference.dtypes import write as write_dtype

    # bf16 keeps 8 mantissa bits, so at 1.0 the spacing is 2**-7.  Three cases pin
    # the behaviour: below half a step rounds down, between half and a full step
    # rounds up (this is where truncation would differ), and an exact tie rounds
    # to even -- which at 1.0 means back down.
    values = np.array(
        [1.0 + 2.0**-9, 1.0 + 3.0 * 2.0**-9, 1.0 + 2.0**-8],
        np.float32,
    )
    target = np.zeros(values.size * 2, dtype=np.uint8)  # bf16 is two bytes wide
    write_dtype(target, values, DType.BF16)
    got = read_dtype(target, DType.BF16, values.shape)
    np.testing.assert_array_equal(got, np.array([1.0, 1.0 + 2.0**-7, 1.0], np.float32))

    # The middle value is the one that proves this is rounding and not a shift:
    # truncating the same bits lands on 1.0 instead of 1.0078125.
    truncated = ((values.view(np.uint32) >> 16) << 16).view(np.float32)
    assert truncated[1] == 1.0
    assert got[1] == 1.0 + 2.0**-7


def test_alloc_is_aligned_and_zeroed(session):
    for align in (1, 16, 64, 256):
        buffer = session.alloc(100, align=align)
        assert buffer.nbytes == 100
        assert buffer.address() % align == 0, f"align={align}: address {buffer.address()} is not aligned"
        assert bytes(buffer.host_view()) == bytes(100), "a fresh allocation must be zeroed"


def test_subview_shares_storage_and_checks_bounds(session):
    buffer = session.alloc(64, align=1)
    view = buffer.host_view()
    view[0:4] = b"\x01\x02\x03\x04"
    sub = buffer.subview(2, 4)
    assert bytes(sub.host_view()) == b"\x03\x04\x00\x00"
    sub.host_view()[0] = 9
    assert bytes(view[2:3]) == b"\x09", "a subview must alias the parent, not copy it"
    with pytest.raises(ValueError):
        buffer.subview(60, 8)


def test_freeing_a_foreign_buffer_is_a_no_op(session):
    from pocketllm.kernels.buffer import HostBuffer

    session.free(HostBuffer(np.zeros(4, np.uint8)))  # must not raise


def test_unknown_op_is_refused_by_name(session):
    from pocketllm.kernels.errors import OpNotDeclaredError

    with pytest.raises(OpNotDeclaredError) as excinfo:
        session.run("no_such_op", [])
    assert "no_such_op" in str(excinfo.value)


def test_run_after_close_is_refused(session):
    from pocketllm.backends.reference import BACKEND as _BACKEND

    fresh = _BACKEND.open(Device("cpu"))
    fresh.close()
    with pytest.raises(RuntimeError):
        fresh.run("argmax", [host_tensor(fresh, np.ones(4, np.float32))])


def test_a_supplied_output_of_the_wrong_shape_is_refused(session):
    args, attrs = sample_args("gemm", session)
    descs = OPS.get("gemm").infer(args, attrs)
    wrong = Tensor(TensorDesc((1, 1), dtype=DType.F32), session.alloc(4, align=1))
    with pytest.raises(KernelError):
        session.run("gemm", args, out=[wrong], attrs=attrs)


def test_the_harness_can_actually_fail():
    """A wrong result must not pass :func:`compare`.

    The harness is the only thing standing between "this backend works" and "this
    backend says it works", so its ability to reject is itself tested.  Without
    this, a comparison loosened to ``atol=inf`` would look like a passing suite.
    """
    expected = np.arange(4, dtype=np.float32)
    compare("add", expected, expected.copy())  # the honest case passes
    with pytest.raises(AssertionError):
        compare("add", expected, expected + 1.0)
    with pytest.raises(AssertionError):
        compare("add", expected, expected.reshape(2, 2))
    # An exact op must reject a difference a float tolerance would forgive.
    compare("argmax", np.array(3, np.int32), np.array(3, np.int32))
    with pytest.raises(AssertionError):
        compare("argmax", np.array(3, np.int32), np.array(4, np.int32))