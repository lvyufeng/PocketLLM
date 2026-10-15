"""llama.cpp's own 2-D convolution, as the outside authority for the layout.

`test_op_conformance.py` checks the C kernel against the numpy reference, and
that is the right question for the arithmetic.  It cannot see a layout error the
two agree on: the same gather index written the same wrong way in both is one
opinion stated twice.  A non-overlapping convolution is *nothing but* layout --
the `k` axis order, the input's channel and row strides, and which output row
each patch lands in -- so it is the op that most needs a second authority.

That authority is `ggml_conv_2d`: the function llama.cpp builds the tower's
`v.patch_embd` with, driven on the tensor's own bytes by
:file:`ggml_conv.c`.  Compiled here rather than by CMake, the way
:mod:`ggml_dequant` is, so a llama.cpp checkout stays a dependency of the test
that needs it and not of `libpocketllm.so`.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

from llama_oracle import LLAMA_BUILD, LLAMA_CPP, repository_root


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "ggml-conv"


def compile_tool() -> None:
    """Build the convolution instrument against llama.cpp's own ggml headers."""
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-o",
            str(output),
            str(pathlib.Path(__file__).resolve().parent / "ggml_conv.c"),
            f"-I{LLAMA_CPP / 'ggml' / 'include'}",
            f"-L{LLAMA_BUILD}",
            "-lggml",
            "-lggml-base",
            "-lggml-cpu",
            f"-Wl,-rpath,{LLAMA_BUILD}",
        ],
        check=True,
        capture_output=True,
    )


def conv2d(image: np.ndarray, weight: np.ndarray, *, kernel: int, stride: int, pad: int) -> np.ndarray:
    """``ggml_conv_2d``'s answer, as ``(out_h, out_w, out_ch)``.

    ``image`` is ``(in_ch, height, width)`` float32 with ``width`` fastest, and
    ``weight`` is the GGUF payload in storage order -- ``(out_ch, in_ch, kernel,
    kernel)`` as numpy indexes it, which is ``kernel x kernel x in_ch x out_ch``
    with the output channel outermost as ggml's ``ne`` indexes it.  Either width
    works: ggml reads an f16 tensor's two-byte elements and a float32 one's four.

    The tool writes ggml's own element order, ``out_w`` fastest and then
    ``out_h`` and ``out_ch`` -- which, because numpy's C order has the *last*
    axis fastest, is numpy's ``(out_ch, out_h, out_w)``.  This reads it that way
    and transposes to the image-major ``(out_h, out_w, out_ch)`` the engine's op
    uses, so the transpose is stated once here and every caller compares like
    with like.
    """
    image32 = np.ascontiguousarray(image, dtype=np.float32)
    in_ch, height, width = image32.shape
    payload = np.ascontiguousarray(weight)
    if payload.ndim != 4 or payload.shape[1:] != (in_ch, kernel, kernel):
        raise ValueError(
            f"the weight is {payload.shape}, not (out_ch, in_ch={in_ch}, {kernel}, {kernel})"
        )
    if payload.dtype == np.float16:
        weight_type = "f16"
    elif payload.dtype == np.float32:
        weight_type = "f32"
    else:
        raise ValueError(f"the tool does not read a {payload.dtype} weight")
    if (height + 2 * pad - kernel) < 0 or (width + 2 * pad - kernel) < 0:
        raise ValueError("the kernel does not fit the image at this padding")
    out_h = (height + 2 * pad - kernel) // stride + 1
    out_w = (width + 2 * pad - kernel) // stride + 1
    with tempfile.TemporaryDirectory() as tmp:
        image_path = pathlib.Path(tmp) / "image.bin"
        weight_path = pathlib.Path(tmp) / "weight.bin"
        image_path.write_bytes(image32.tobytes())
        weight_path.write_bytes(payload.tobytes())
        result = subprocess.run(
            [
                str(tool_path()),
                str(image_path),
                str(in_ch),
                str(height),
                str(width),
                str(weight_path),
                weight_type,
                str(kernel),
                str(stride),
                str(pad),
            ],
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "ggml_conv_2d failed: " + result.stderr.decode(errors="replace").strip()
        )
    out_ch = payload.shape[0]
    flat = np.frombuffer(result.stdout, dtype="<f4")
    if flat.size != out_h * out_w * out_ch:
        raise RuntimeError(f"ggml returned {flat.size} values, expected {out_h * out_w * out_ch}")
    return flat.reshape(out_ch, out_h, out_w).transpose(1, 2, 0)
