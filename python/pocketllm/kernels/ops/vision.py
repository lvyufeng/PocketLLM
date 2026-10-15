"""The vision tower's front end: the patch embedding.

The tower starts by cutting an image into patches and projecting each one, and
for a checkpoint whose patch size equals its stride -- which is every ViT this
tree targets -- that projection *is* a matrix product over gathered patches.  It
is declared as an op of its own rather than left as an im2col in the graph for
the reason the ABI exists: a caller should name the operation it wants, and a
gather inlined in the model file would put the patch layout in the model *and*
in every backend's kernel, with nothing tying the two copies together.

The operand order is ggml's, deliberately.  GGUF stores the kernel as
``(kernel, kernel, in_ch, out_ch)`` with the output channel outermost, and that
flattened -- ``out_ch`` rows of ``in_ch * kernel * kernel`` -- is exactly the
row-major ``(out_ch, k)`` weight `gemm` contracts against.  Reading it that way
means nothing is transposed on load, and it makes ``ggml_conv_2d`` a
like-for-like oracle for this op instead of a comparison that first has to agree
on a layout.
"""

from __future__ import annotations

from ..dtypes import DType
from ..errors import ShapeError
from ..schema import ArgSpec, Kind, OpSchema

_FLOAT = frozenset({DType.F32, DType.F16, DType.BF16})


def _square_conv_output(ins: int, kernel: int, stride: int, pad: int) -> int:
    """``ggml_calc_conv_output_size`` for a square kernel at unit dilation."""
    return (ins + 2 * pad - kernel) // stride + 1


def _conv2d_shape(shapes, attrs):
    in_ch, height, width = shapes["x"]
    out_ch, k = shapes["w"]
    kernel = int(attrs["kernel"])
    stride = int(attrs["stride"])
    pad = int(attrs["pad"])
    if stride <= 0:
        raise ShapeError("conv2d: stride must be positive")
    if k != in_ch * kernel * kernel:
        raise ShapeError(
            f"conv2d: w has {k} columns, but x's {in_ch} channels and a "
            f"{kernel}x{kernel} kernel need {in_ch * kernel * kernel}"
        )
    out_h = _square_conv_output(height, kernel, stride, pad)
    out_w = _square_conv_output(width, kernel, stride, pad)
    if out_h <= 0 or out_w <= 0:
        raise ShapeError(
            f"conv2d: a {kernel}x{kernel} kernel at stride {stride} with padding {pad} "
            f"does not fit a {height}x{width} image"
        )
    return [(out_h * out_w, out_ch)]


CONV2D = OpSchema(
    name="conv2d",
    args=(
        ArgSpec("x", Kind.TENSOR, shape=("in_ch", "height", "width")),
        ArgSpec("w", Kind.TENSOR, shape=("out_ch", "k"), role="weight"),
        ArgSpec("bias", Kind.TENSOR, shape=("out_ch",), optional=True),
    ),
    returns=(ArgSpec("y", Kind.TENSOR, shape=("patches", "out_ch")),),
    dtypes=_FLOAT,
    attrs=("kernel", "stride", "pad"),
    shape_rule=_conv2d_shape,
    semantics=(
        "y[p, j] = sum_{ic, ky, kx} x[ic, oy*stride + ky - pad, ox*stride + kx - pad] "
        "* w[j, ic * kernel^2 + ky * kernel + kx] + bias[j], "
        "for the patch p = oy * out_w + ox"
    ),
)

SCHEMAS = (CONV2D,)
