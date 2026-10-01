"""The op vocabulary: what is declared, and that its declarations are coherent.

A schema is the single statement of what an op means.  These tests check the
statement is complete enough to be useful -- every tensor argument's shape is
either constrained or explicitly wildcarded, every return can be inferred, and
two declarations of the same name cannot disagree.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pocketllm.kernels import OPS, TensorDesc, DType, OpRegistry, ShapeError
from pocketllm.kernels.errors import OpNotDeclaredError
from pocketllm.kernels.ops import declare_all
from pocketllm.kernels.schema import Kind

#: The core vocabulary.  A new op is a deliberate act, and this list is where it
#: gets recorded; the reference-completeness test is what keeps a new entry from
#: shipping without an implementation.
EXPECTED_OPS = frozenset(
    {
        "gemm",
        "gemm_quant",
        "attention",
        "moe_ffn",
        "rope",
        "rms_norm",
        "layer_norm",
        "silu_mul",
        "add",
        "mul",
        "softmax",
        "embedding",
        "logits_temperature",
        "argmax",
        "topk_sample",
        "cache_append",
        "cache_truncate",
    }
)


def test_vocabulary_is_exactly_what_is_expected() -> None:
    assert OPS.names() == EXPECTED_OPS


def test_registry_is_frozen_after_declaration() -> None:
    with pytest.raises(RuntimeError):
        declare_all()


def test_unknown_op_is_refused_by_name() -> None:
    with pytest.raises(OpNotDeclaredError):
        OPS.get("no_such_op")


def test_extend_adds_outside_the_frozen_core() -> None:
    registry = OpRegistry()
    core = OPS.get("add")
    registry.extend(core)
    assert registry.names() == {"add"}


def test_same_name_different_schema_is_refused() -> None:
    registry = OpRegistry()
    original = OPS.get("add")
    registry.declare(original)
    with pytest.raises(ValueError):
        registry.declare(replace(original, semantics="a different statement of the same op"))


def test_same_name_same_schema_is_idempotent() -> None:
    registry = OpRegistry()
    schema = OPS.get("add")
    assert registry.declare(schema) is schema
    assert registry.declare(schema) is schema
    assert registry.names() == {"add"}


@pytest.mark.parametrize("schema", list(OPS.schemas()), ids=lambda s: s.name)
def test_every_schema_is_self_consistent(schema) -> None:
    assert schema.name
    assert schema.returns, f"{schema.name} declares no return"
    assert schema.shape_rule is not None, f"{schema.name} has no shape rule"
    assert schema.semantics, f"{schema.name} states no semantics"

    seen: set[str] = set()
    for spec in schema.declarations():
        assert spec.name not in seen, f"{schema.name} declares {spec.name!r} twice"
        seen.add(spec.name)
        if spec.kind is Kind.TENSOR and spec.shape is not None:
            assert all(dim for dim in spec.shape), f"{schema.name}.{spec.name} has an empty dim name"


def test_optional_and_variadic_are_consistent() -> None:
    for schema in OPS.schemas():
        for spec in schema.declarations():
            if spec.variadic:
                assert spec.optional, f"{schema.name}.{spec.name} is variadic but not optional"


def test_gemm_infers_output_shape_and_type() -> None:
    gemm = OPS.get("gemm")
    x = TensorDesc((5, 64), dtype=DType.F16)
    w = TensorDesc((32, 64), dtype=DType.F16)
    (y,) = gemm.infer([x, w])
    assert y.shape == (5, 32)
    assert y.dtype is DType.F16


def test_gemm_propagates_the_first_float_input() -> None:
    gemm = OPS.get("gemm")
    x = TensorDesc((2, 8), dtype=DType.BF16)
    w = TensorDesc((4, 8), dtype=DType.F32)
    (y,) = gemm.infer([x, w])
    assert y.dtype is DType.BF16


def test_gemm_rejects_disagreeing_k() -> None:
    gemm = OPS.get("gemm")
    with pytest.raises(ShapeError):
        gemm.infer([TensorDesc((5, 64), dtype=DType.F32), TensorDesc((32, 63), dtype=DType.F32)])


def test_optional_argument_may_be_omitted_or_none() -> None:
    gemm = OPS.get("gemm")
    x = TensorDesc((5, 64), dtype=DType.F32)
    w = TensorDesc((32, 64), dtype=DType.F32)
    assert gemm.infer([x, w])[0].shape == (5, 32)
    assert gemm.infer([x, w, None])[0].shape == (5, 32)


def test_missing_required_argument_is_refused() -> None:
    gemm = OPS.get("gemm")
    with pytest.raises(ShapeError):
        gemm.infer([TensorDesc((5, 64), dtype=DType.F32), None])


def test_return_shape_may_not_use_the_wildcard() -> None:
    """A return that named ``*`` would have no shape to allocate."""
    from pocketllm.kernels.schema import ArgSpec, OpSchema

    bad = OpSchema(
        name="bad",
        args=(ArgSpec("x", Kind.TENSOR, shape=("*",)),),
        returns=(ArgSpec("y", Kind.TENSOR, shape=("*",)),),
        dtypes=frozenset({DType.F32}),
        shape_rule=lambda shapes, attrs: [("*",)],
    )
    with pytest.raises(ShapeError):
        bad.infer([TensorDesc((4,), dtype=DType.F32)])