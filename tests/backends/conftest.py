"""The conformance harness: one canonical call per op, run against every backend.

A backend's declaration is a claim, and this file is how the claim is checked.
For each op the registry declares, :func:`sample_args` builds a small call with
known-good shapes and finite values; the harness then runs it through a backend
and compares the result against the reference backend's, which is the normative
answer by construction.

Three things that are deliberate and worth reading before changing:

* **The harness speaks only the ABI.**  Tensors are built through
  ``BackendSession.to_device`` and read back through ``to_host``, never through a
  backend-specific convenience.  Otherwise the harness would only work on
  backends that happen to have copied the reference session's helpers, which is
  the opposite of what a conformance test is for.
* **A skip is not a pass.**  A backend whose runtime is absent skips, and a
  backend whose session is a stub raises ``BackendNotImplementedError``, which
  the harness turns into a skip carrying the stub's own message.  Neither enters
  ``tests/baseline_failures.txt``; the baseline is a set of *node ids*, and a
  skip has no failing node to record.
* **The reference backend is the oracle, not a peer.**  It is excluded from the
  comparison loop (it would be comparing it with itself); every capability it
  declares is checked for *completeness* separately, in
  ``tests/abi/test_reference_completeness.py``.

The samples are small on purpose -- a conformance run should be seconds, not
minutes, and every one of these ops is shape-polymorphic, so a tiny call
exercises the same code path a large one does.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.backends import registry
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType, QUANT_FORMATS
from pocketllm.kernels.registry import OPS
from pocketllm.kernels.tensor import Tensor, TensorDesc

#: Small, finite dimensions.  The wide axis is 256 so a quantized weight's
#: blocks divide evenly, which is what a real checkpoint guarantees too.
_HIDDEN = 256
_HEADS = 2
_HEAD_DIM = 4
_VOCAB = 16
_TOKENS = 2
_EXPERTS = 3
_CAPACITY = 8

_NP_OF: dict[DType, np.dtype] = {
    DType.F32: np.dtype("<f4"),
    DType.F16: np.dtype("<f2"),
    DType.I8: np.dtype("<i1"),
    DType.I16: np.dtype("<i2"),
    DType.I32: np.dtype("<i4"),
    DType.I64: np.dtype("<i8"),
    DType.U8: np.dtype("<u1"),
}

_NP_TO_ABI = {np.dtype(t): d for d, t in _NP_OF.items()}


# -- ABI-only tensor plumbing -------------------------------------------------


def host_tensor(session, array) -> Tensor:
    """Upload a numpy array through the ABI and return the matching tensor.

    Two numpy subtleties are handled here because they turn a scalar argument
    into a shape error that looks like the backend's fault:

    * ``np.asarray`` materialises a Python float as a 0-d array, which is what
      an op with a ``shape=()`` argument wants (``topk_sample``'s uniform).
    * ``np.ascontiguousarray`` *promotes* a 0-d array to shape ``(1,)`` -- in
      numpy 1.26 at least -- so it is applied only when the array is not already
      contiguous.  A 0-d array always is.
    """
    array = np.asarray(array)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    try:
        dtype = _NP_TO_ABI[array.dtype]
    except KeyError as exc:
        raise NotImplementedError(f"the harness has no ABI dtype for numpy {array.dtype}") from exc
    desc = TensorDesc(tuple(array.shape), dtype=dtype)
    return session.to_device(memoryview(array).cast("B"), desc)


def read(session, tensor: Tensor) -> np.ndarray:
    """Read a tensor back through the ABI as a numpy array."""
    view = session.to_host(tensor)
    desc = tensor.desc
    if desc.dtype is DType.BF16:
        raw = np.frombuffer(view, dtype="<u2").astype(np.uint32)
        return (raw << 16).view(np.float32).reshape(desc.shape)
    try:
        np_dtype = _NP_OF[desc.dtype]
    except KeyError as exc:
        raise NotImplementedError(f"the harness cannot read {desc.dtype}") from exc
    return np.frombuffer(view, dtype=np_dtype).reshape(desc.shape)


def blocks_tensor(session, name: str, rows: int, row_elems: int) -> Tensor:
    """A quantized :class:`Tensor` on ``session`` holding real blocks for ``name``.

    Blocks are allocated and filled through the ABI (``alloc`` then
    ``host_view``), so a backend with no host-visible memory fails here rather
    than in a decoder.
    """
    desc = TensorDesc((rows, row_elems), quant=QUANT_FORMATS[name])
    blocks = _quant_blocks(name, rows, row_elems)
    buffer = session.alloc(desc.nbytes, align=1)
    view = buffer.host_view()
    if view is None:
        raise NotImplementedError(f"{session.device} has no host-visible memory to build a weight in")
    view[:] = blocks.tobytes()
    return Tensor(desc, buffer)


def _quant_blocks(name: str, rows: int, row_elems: int) -> np.ndarray:
    """Deterministic blocks for a format, with real fp16 scales where one exists.

    The scale bytes are what make a decode finite and non-trivial: random bytes
    frequently spell an inf or a NaN in fp16, and a comparison of NaNs is not a
    comparison.  Only the fp16 fields are overwritten; everything else stays
    random, so the gather and the nibble split are genuinely exercised.
    """
    fmt = QUANT_FORMATS[name]
    blocks_per_row = -(-row_elems // fmt.block_elems)
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    blocks = rng.integers(0, 256, size=(rows, blocks_per_row, fmt.block_bytes), dtype=np.uint8)

    # The formats whose first two bytes are a single fp16 scale.
    if name in {
        "iq4_nl", "q8_0", "q4_k", "q5_k", "q6_k", "iq4_xs",
        "iq2_xxs", "iq2_xs", "iq3_xxs", "q2_k", "q3_k",
    }:
        blocks[..., 0:2] = np.frombuffer(np.float16(0.25).tobytes(), dtype=np.uint8)
    # The formats with a *second* fp16 field at bytes 2:4 (a min, or the high
    # half of a scale); leaving that random can spell an inf.
    if name in {"q4_k", "q5_k", "q6_k"}:
        blocks[..., 2:4] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
    # IQ1_M hides its super-scale across four nibbles, so a zero block decodes to
    # zero rather than to something finite; see test_quant_decoders.py.
    if name == "iq1_m":
        blocks[..., 49] = 0x30
        blocks[..., 51] = 0xC0
    return blocks


# -- the samples --------------------------------------------------------------


def sample_args(op: str, session, quant: str | None = None) -> tuple[list, dict]:
    """A canonical call to ``op``: positional args and attrs.

    ``quant`` names the format a quantized weight should use; the harness
    parameterizes over each backend's declared formats, so a format that does not
    decode is caught by the backend that *claimed* it rather than by whichever
    one happened to be tested first.
    """
    rng = np.random.default_rng(20261001)
    t = lambda array: host_tensor(session, array)  # noqa: E731 - a local alias, not a def

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    if op == "gemm":
        return [t(rand(2, _HIDDEN)), t(rand(4, _HIDDEN))], {}
    if op == "gemm_quant":
        return [t(rand(2, _HIDDEN)), blocks_tensor(session, quant or "iq4_nl", 4, _HIDDEN)], {}
    if op == "attention":
        q = rand(_TOKENS, _HEADS, _HEAD_DIM)
        k = rand(_CAPACITY, _HEADS, _HEAD_DIM)
        v = rand(_CAPACITY, _HEADS, _HEAD_DIM)
        positions = np.array([1, 4], dtype=np.int32)
        return [t(q), t(k), t(v), t(positions)], {"softmax_scale": 0.5, "causal": True, "num_kv_heads": _HEADS}
    if op == "rms_norm":
        return [t(rand(_TOKENS, _HIDDEN)), t(np.ones(_HIDDEN, np.float32))], {"eps": 1e-6}
    if op == "layer_norm":
        return [
            t(rand(_TOKENS, _HIDDEN)),
            t(np.ones(_HIDDEN, np.float32)),
            t(np.zeros(_HIDDEN, np.float32)),
        ], {"eps": 1e-5}
    if op in {"silu_mul", "add", "mul"}:
        return [t(rand(_TOKENS, _HIDDEN)), t(rand(_TOKENS, _HIDDEN))], {}
    if op == "softmax":
        return [t(rand(_TOKENS, _VOCAB))], {"axis": -1}
    if op == "rope":
        cos = np.cos(rng.standard_normal((_CAPACITY, _HEAD_DIM // 2))).astype(np.float32)
        sin = np.sin(rng.standard_normal((_CAPACITY, _HEAD_DIM // 2))).astype(np.float32)
        positions = np.array([0, 3], dtype=np.int32)
        return [t(rand(_TOKENS, _HEADS, _HEAD_DIM)), t(positions), t(cos), t(sin)], {
            "layout": "split",
            "theta_base": 10000.0,
            "scaling": None,
        }
    if op == "embedding":
        return [t(np.array([3, 1], dtype=np.int32)), t(rand(_VOCAB, _HIDDEN))], {}
    if op == "moe_ffn":
        return [
            t(rand(1, _HIDDEN)),
            t(np.array([[0, 2]], dtype=np.int32)),
            t(np.array([[0.6, 0.4]], dtype=np.float32)),
            t(rand(_EXPERTS, 2 * _HIDDEN, _HIDDEN)),
            t(rand(_EXPERTS, _HIDDEN, _HIDDEN)),
        ], {"top_k": 2, "norm_topk_prob": True, "swiglu": True}
    if op == "logits_temperature":
        return [t(rand(_VOCAB))], {"temperature": 0.7}
    if op == "argmax":
        return [t(rand(_VOCAB))], {}
    if op == "topk_sample":
        return [t(rand(_VOCAB)), t(np.array(0.5, np.float32))], {"top_k": 8, "top_p": 0.9, "min_p": 0.0}
    if op == "cache_append":
        cache = np.zeros((_CAPACITY, _HEADS, _HEAD_DIM), np.float32)
        values = rand(_TOKENS, _HEADS, _HEAD_DIM)
        positions = np.array([2, 5], dtype=np.int32)
        return [t(cache), t(values), t(positions)], {}
    if op == "cache_truncate":
        return [t(rand(_CAPACITY, _HEADS, _HEAD_DIM))], {"length": 3}
    raise KeyError(f"the conformance harness has no sample for {op!r}")


#: Ops whose result is a token id or an index: compared exactly, not by
#: tolerance, because a value one off is a different answer, not a rounding.
_EXACT = frozenset({"argmax", "topk_sample"})


def compare(op: str, expected: np.ndarray, actual: np.ndarray, quant: str | None = None) -> None:
    """Assert a backend's result matches the reference's, to the op's tolerance."""
    exp = np.asarray(expected)
    got = np.asarray(actual)
    assert got.shape == exp.shape, f"{op}: shape {got.shape} != reference {exp.shape}"
    if op in _EXACT:
        np.testing.assert_array_equal(got, exp, err_msg=f"{op}: exact result differs")
        return
    if quant is not None:
        # A quantized product accumulates a decoded fp16 scale over 256 terms, so
        # it gets a wider band than a dense op.
        np.testing.assert_allclose(got, exp, rtol=5e-2, atol=5e-3, err_msg=f"{op}/{quant}: differs from reference")
    else:
        np.testing.assert_allclose(got, exp, rtol=2e-3, atol=1e-4, err_msg=f"{op}: differs from reference")


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="session")
def reference_session():
    """One open reference session, shared: it allocates small tensors only.

    Session-scoped is safe because the session holds no mutable model state --
    each call allocates its own output -- and opening one per test would dominate
    the run time.
    """
    session = registry.get("reference").open(Device("cpu"))
    yield session
    session.close()


@pytest.fixture(scope="session")
def backends():
    return registry.available_backends()


@pytest.fixture(scope="session")
def declared_ops():
    return OPS.names()