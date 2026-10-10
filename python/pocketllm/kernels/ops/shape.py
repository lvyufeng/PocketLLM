"""Shape ops: views that change how a value is addressed, not what it holds.

A graph is not only arithmetic.  Two of the shapes a transformer needs are not
the shapes an op *produces*: a per-head norm reduces over the last axis while the
projection that feeds it is 2-D, and a sampler reads one row of a ``(tokens,
vocab)`` logits matrix.  Neither is a computation -- the bytes do not move in
value -- but both have to be *said*, because every downstream schema infers its
output shape from the shapes of its inputs.

``reshape`` is that statement, and it is deliberately the only one in v1.  A
``permute`` or a ``slice`` would each be an argument of their own; the one that
unblocks a real transformer is this one, and the rest arrive when something needs
them.

The op is *logical*: a backend may realize it as a view or as a copy.  A discrete
device often must copy, and its kernel is free to, because nothing in the
semantics promises that the input and the output share storage.
"""

from __future__ import annotations

from ..dtypes import DType
from ..errors import ShapeError
from ..schema import ArgSpec, Kind, OpSchema
from ..tensor import elem_count

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})

#: A placeholder for the one dimension the caller lets the op infer.  Only one
#: entry of a shape may be it, and it is resolved against the element count.
_INFER = -1


def _reshaped_shape(shape: tuple[int, ...], attrs) -> tuple[int, ...]:
    """Resolve ``attrs["shape"]`` against ``shape``'s element count.

    One dimension may be ``-1``: it takes whatever is left, which is how a
    caller writes "whatever this hands me" without spelling out a product.  More
    than one is ambiguous and is refused rather than resolved to something the
    caller did not mean.
    """
    target = attrs.get("shape")
    if target is None:
        raise ShapeError("reshape: the 'shape' attribute is required")
    out = tuple(int(dim) for dim in target)

    count = elem_count(shape)
    infer_positions = [index for index, dim in enumerate(out) if dim == _INFER]
    if len(infer_positions) > 1:
        raise ShapeError(f"reshape: {out} has more than one inferred dimension (-1)")
    if not infer_positions:
        given = elem_count(out)
        if given != count:
            raise ShapeError(f"reshape: {shape} holds {count} elements, but {out} holds {given}")
        return out

    known = elem_count(tuple(dim for dim in out if dim != _INFER))
    if known == 0 or count % known != 0:
        raise ShapeError(f"reshape: {shape} cannot be reshaped to {out}; {count} is not divisible by {known}")
    index = infer_positions[0]
    resolved = out[:index] + (count // known,) + out[index + 1 :]
    return resolved


RESHAPE = OpSchema(
    name="reshape",
    args=(ArgSpec("x", Kind.TENSOR, shape=None),),
    returns=(ArgSpec("out", Kind.TENSOR, shape=None),),
    dtypes=_FLOAT,
    attrs=("shape",),
    # The output's rank is the target's, which the schema's symbolic vocabulary
    # cannot express -- it names dimensions, it does not build a new arity.  The
    # rule therefore returns a tuple of the *resolved* extents, which ``infer``
    # accepts as literal dimensions.
    shape_rule=lambda shapes, attrs: [_reshaped_shape(shapes["x"], attrs)],
    semantics="the same elements, addressed with the shape given in `shape` (one -1 infers its dimension)",
)

SCHEMAS = (RESHAPE,)