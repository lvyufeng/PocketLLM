"""The patch embedding's index order, against the library the weights came from.

`test_op_conformance.py` checks the C kernel against the numpy reference, which
is the right question for the arithmetic and blind to the layout: a gather index
written the same wrong way in both is one opinion stated twice.  A
non-overlapping convolution is *nothing but* layout -- the `k` axis order, the
input's channel and row strides, and which output row each patch lands in -- so
it gets a second authority.  That authority is `ggml_conv_2d`, the function
llama.cpp builds `v.patch_embd` with, driven on the tensor's own bytes by
:mod:`ggml_conv`.

Two cases.  The first is synthetic and float32 on both sides, so a difference
could only be index order.  The second is the tower's real patch embedding --
the checkpoint's `v.patch_embd.weight` read out of the mmproj verbatim, at its
own `16x16x3 into 1024` geometry -- driven through the engine's `conv2d` with
`pocketllm-opcheck`.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ggml_conv import compile_tool, conv2d as ggml_conv2d, tool_path
from llama_oracle import LLAMA_LIB
from pocketllm.backends.reference import kernels as ref
from pocketllm.loader.gguf.reader import GGUFReader
from test_op_conformance import DENSE_RTOL, SEED, needs_tool, run, write_request

#: The tower's own `mmproj`, beside the text checkpoint.  Its presence is what
#: the second case needs; the first runs without it.
MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-qwen3vl-2b-f16.gguf")

pytestmark = [
    needs_tool,
    pytest.mark.skipif(not LLAMA_LIB.is_file(), reason=f"no llama.cpp build at {LLAMA_LIB}"),
]


@pytest.fixture(scope="module")
def ggml_tool() -> None:
    """Compile the instrument once, if the checkout has not already built it."""
    if not tool_path().is_file():
        compile_tool()


def _engine(image, weight_rows, *, kernel: int, stride: int, pad: int) -> np.ndarray:
    """The C kernel's answer, through `pocketllm-opcheck` on the CPU."""
    request = write_request(
        "conv2d",
        {"x": image, "w": weight_rows},
        {"kernel": kernel, "stride": stride, "pad": pad},
    )
    answer = run(request, "cpu")
    return np.array(answer["values"], dtype=np.float64).reshape(answer["out_shape"])


def _assert_agrees(got: np.ndarray, want: np.ndarray, *, note: str) -> None:
    """A relative bound on the *error*, against the spread of the answer.

    `DENSE_RTOL` is the tree's bound for two float32 computations of the same
    expression, and it is wide enough for the one real difference in the second
    case below: ggml's `conv_2d` narrows its im2col to the weight's own type, so
    an f16 weight narrows the image to f16 on the way in while this engine keeps
    it f32.  That is a fifth-digit effect on the image, where an index error is a
    whole pixel's worth of weight.
    """
    assert got.shape == want.shape, f"{note}: got {got.shape}, expected {want.shape}"
    spread = float(np.max(np.abs(want))) or 1.0
    worst = float(np.max(np.abs(got - want.astype(np.float64))))
    assert worst <= DENSE_RTOL * spread, (
        f"{note}: max |difference| = {worst} over a scale of {spread}"
    )


def test_the_engine_matches_ggml_on_a_synthetic_weight(ggml_tool) -> None:
    """An odd kernel at an odd stride with padding, float32 on both sides.

    Nothing here rounds: both sides read the same float32 weight and the same
    float32 image, and ggml's im2col is float32 because the weight is, so any
    difference is an index.  `kernel = 5` at `stride = 2` with `pad = 1` makes
    the patch grid (9, 13) -- not square, so a kernel that swapped the two
    spatial axes parts from ggml here and nowhere else.

    The tree's own reference is compared to ggml as well, so an in-tree mistake
    shows up as a three-way disagreement rather than as two agreeing wrong
    answers.
    """
    rng = np.random.default_rng(SEED)
    image = rng.uniform(0.0, 1.0, size=(3, 20, 28)).astype(np.float32)
    weight = rng.standard_normal((6, 3, 5, 5), dtype=np.float32)
    kernel, stride, pad = 5, 2, 1
    rows = weight.reshape(6, -1)

    want = ggml_conv2d(image, weight, kernel=kernel, stride=stride, pad=pad)
    assert want.shape == (9, 13, 6)
    flat = want.reshape(-1, want.shape[2])

    _assert_agrees(
        _engine(image, rows, kernel=kernel, stride=stride, pad=pad),
        flat,
        note="the C kernel against ggml_conv_2d",
    )
    _assert_agrees(
        ref.conv2d(image, rows, kernel=kernel, stride=stride, pad=pad),
        flat,
        note="the numpy reference against ggml_conv_2d",
    )


def test_the_towers_patch_embedding_matches_ggml(ggml_tool) -> None:
    """The real tensor, at the tower's own `16x16x3 into 1024` geometry.

    The weight is the checkpoint's own f16 bytes, read verbatim: the layout in
    the file is exactly what is being checked, so decoding it here first would
    test a reconstruction rather than the file.  The image is the tower's real
    `768x768`, because the patch grid is where an index error would live and at
    768 it is the grid the model actually sees.
    """
    if not MMPROJ.is_file():
        pytest.skip(f"no mmproj at {MMPROJ}")
    info = GGUFReader(str(MMPROJ)).read().tensors_by_name["v.patch_embd.weight"]
    with open(MMPROJ, "rb") as handle:
        handle.seek(info.absolute_offset)
        raw = handle.read(info.nbytes)
    # GGUF declares (kw, kh, in_ch, out_ch) with ne0 fastest; numpy indexes the
    # same memory with the last axis fastest, which is (out_ch, in_ch, kh, kw).
    weight = np.frombuffer(raw, dtype="<f2").reshape(tuple(reversed(info.dimensions)))
    assert weight.shape == (1024, 3, 16, 16), weight.shape

    rng = np.random.default_rng(SEED)
    image = rng.uniform(0.0, 1.0, size=(3, 768, 768)).astype(np.float32)
    kernel, stride, pad = 16, 16, 0

    want = ggml_conv2d(image, weight, kernel=kernel, stride=stride, pad=pad)
    assert want.shape == (48, 48, 1024)

    _assert_agrees(
        _engine(image, weight.astype(np.float32).reshape(1024, -1), kernel=kernel, stride=stride,
                pad=pad),
        want.reshape(-1, 1024),
        note="the tower's patch embedding against ggml_conv_2d",
    )
