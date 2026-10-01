"""Op schemas: how one op's arguments, types and shapes are declared once.

A backend does not re-describe the shape of a GEMM.  It declares *that it
implements* ``gemm_quant`` and over what domain, and the schema below is the
single statement of what ``gemm_quant`` means -- its argument names, which of
them are tensors, what dtypes and quant formats it admits, and how an output
shape is inferred from its inputs.

Shape inference is written as a small vocabulary of symbolic dimensions.  In a
schema, a tensor argument whose ``shape=`` is ``("m", "k")`` says the first
dimension is named ``m`` and the second ``k``; two arguments that both say
``k`` must agree at call time.  ``infer`` resolves the names and returns the
output descriptors, so the rule is stated once and every backend gets it.

Nothing here executes and nothing here imports numpy or a device runtime: a
schema is data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .dtypes import DType, QuantFormat
from .errors import ShapeError
from .tensor import TensorDesc, elem_count

__all__ = ["Kind", "ArgSpec", "OpSchema"]


class Kind(Enum):
    """What an argument is."""

    TENSOR = "tensor"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"


@dataclass(frozen=True, slots=True)
class ArgSpec:
    """One argument or return of an op."""

    name: str
    kind: Kind
    optional: bool = False
    variadic: bool = False
    #: For a TENSOR argument: the symbolic shape it must satisfy, one name per
    #: dimension.  ``"*"`` means "any extent, unconstrained"; a repeated name
    #: constrains two arguments to agree.  ``None`` means the shape is not
    #: constrained by the schema.
    shape: tuple[str, ...] | None = None
    #: For a TENSOR argument: a human label for the role its type plays, when the
    #: accepted types are the op's ``dtypes`` but only some positions take a
    #: quantized operand (e.g. "weight").  Documentation, not enforcement.
    role: str | None = None
    #: For a RETURN: an explicit element type, when it differs from the op's
    #: input dtype (an index or token output is I32 even though the op reads f32).
    dtype: DType | None = None

    def __post_init__(self) -> None:
        if self.variadic and not self.optional:
            object.__setattr__(self, "optional", True)


@dataclass(frozen=True, slots=True)
class OpSchema:
    """One op, declared once.  The whole ABI vocabulary is the set of these."""

    name: str
    args: tuple[ArgSpec, ...]
    returns: tuple[ArgSpec, ...]
    dtypes: frozenset[DType] = frozenset()
    quants: frozenset[QuantFormat] = frozenset()
    #: Keyword attributes, declared so dispatch and backends can key on them.
    attrs: tuple[str, ...] = ()
    #: Output shapes, given the resolved input shapes and the attrs.
    shape_rule: Callable[[Mapping[str, tuple[int, ...]], Mapping[str, Any]], Sequence[tuple[int, ...]]] | None = None
    #: A plain-language statement of what the op computes.  The reference
    #: backend is the executable form of this; the two are expected to agree.
    semantics: str = ""

    def declarations(self) -> tuple[ArgSpec, ...]:
        return self.args + self.returns

    def infer(
        self,
        args: Sequence[Any],
        attrs: Mapping[str, Any] | None = None,
    ) -> tuple[TensorDesc, ...]:
        """Resolve the output descriptors, checking input shapes as a side effect.

        ``args`` is the op's positional arguments in declaration order; a tensor
        argument is a :class:`~pocketllm.kernels.tensor.Tensor` (or a desc), a
        scalar is a Python value.  Raises :class:`ShapeError` on a mismatch, so
        the caller learns *which* dimension disagreed rather than only that the
        call failed.
        """
        attrs = dict(attrs or {})
        binding: dict[str, int] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        first_dtype: DType | None = None
        for spec, value in zip(self.args, args):
            if spec.kind is Kind.TENSOR:
                if value is None:
                    if not spec.optional:
                        raise ShapeError(f"{self.name}: required argument {spec.name!r} is missing")
                    continue
                desc = _as_desc(value)
                shapes[spec.name] = desc.shape
                if first_dtype is None and desc.dtype is not None and desc.dtype.is_float:
                    first_dtype = desc.dtype
                _bind_shape(spec, desc.shape, binding)
        if self.shape_rule is None:
            raise ShapeError(f"{self.name}: no shape rule is declared for its returns")
        out_shapes = list(self.shape_rule(shapes, attrs))
        if len(out_shapes) != len(self.returns):
            raise ShapeError(
                f"{self.name}: shape rule produced {len(out_shapes)} outputs, "
                f"declared {len(self.returns)}"
            )
        out: list[TensorDesc] = []
        for spec, shape in zip(self.returns, out_shapes):
            resolved = tuple(_resolve_dim(dim, binding) for dim in shape)
            out.append(TensorDesc(resolved, dtype=_return_dtype(self, spec, first_dtype), quant=None))
        return tuple(out)


def _as_desc(value: Any) -> TensorDesc:
    if isinstance(value, TensorDesc):
        return value
    desc = getattr(value, "desc", None)
    if isinstance(desc, TensorDesc):
        return desc
    raise ShapeError(f"expected a tensor descriptor, got {type(value).__name__}")


def _bind_shape(spec: ArgSpec, shape: tuple[int, ...], binding: dict[str, int]) -> None:
    if spec.shape is None:
        return
    if len(spec.shape) != len(shape):
        raise ShapeError(f"{spec.name}: expected {len(spec.shape)} dims, got {len(shape)}")
    for name, extent in zip(spec.shape, shape):
        if name == "*":
            continue
        known = binding.get(name)
        if known is None:
            binding[name] = extent
        elif known != extent:
            raise ShapeError(f"{spec.name}: dimension {name!r} is {extent}, but was bound to {known}")


def _resolve_dim(dim: str | int, binding: Mapping[str, int]) -> int:
    if isinstance(dim, int):
        return dim
    if dim == "*":
        raise ShapeError("a return shape may not use the unconstrained dimension '*'")
    try:
        return binding[dim]
    except KeyError as exc:
        raise ShapeError(f"return shape names unknown dimension {dim!r}") from exc


def _return_dtype(schema: OpSchema, spec: ArgSpec, first_dtype: DType | None) -> DType:
    """A return's element type.

    An explicit ``ArgSpec.dtype`` wins -- an index or token output is I32 even
    though the op reads f32.  Otherwise the op is float-in/float-out, so the
    *first* float input's dtype propagates (f16 activations produce an f16
    result).  The f32 fallback is the reference backend's working type.
    """
    if spec.dtype is not None:
        return spec.dtype
    if first_dtype is not None and first_dtype.is_float:
        return first_dtype
    return DType.F32