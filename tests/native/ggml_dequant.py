"""llama.cpp's own block decoder, as the second authority for the codebooks.

``pocketllm.quant`` is the tree's decoder: it reads ggml's tables out of the
vendored header and applies the layout with numpy. For an *arithmetic* format
that is checkable by re-deriving it, but the codebook formats are lookups, and a
table read with the wrong stride, bias or sign layout produces plausible weights
-- which is the failure mode that has no signature in this tree, because the
loader, the reference backend and every test would agree with each other while
all of them are wrong.

So the comparison has to be with something outside the tree: llama.cpp wrote
these files and its own ``dequantize_row_*`` reads them back.
``ggml_get_type_traits(...)->to_float`` *is* that decoder, and
:file:`ggml_dequant.c` calls it, compiled here against llama.cpp's headers the
way :mod:`llama_oracle` compiles the logits oracle. The same comparison, run by
hand, is how this tree's `q3_K` scale reading was found to decode four-bit scales
as if they were six and so lose two thirds of the range.

Nothing here is llama.cpp-specific state: one tool, one call per block range.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

from llama_oracle import LLAMA_BUILD, LLAMA_CPP, repository_root


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "ggml-dequant"


def compile_tool() -> None:
    """Build the block decoder against llama.cpp's own ggml headers."""
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-o",
            str(output),
            str(pathlib.Path(__file__).resolve().parent / "ggml_dequant.c"),
            f"-I{LLAMA_CPP / 'ggml' / 'include'}",
            f"-L{LLAMA_BUILD}",
            "-lggml",
            "-lggml-base",
            f"-Wl,-rpath,{LLAMA_BUILD}",
        ],
        check=True,
        capture_output=True,
    )


def decode(type_id: int, blocks: bytes) -> np.ndarray:
    """``blocks`` -- consecutive GGML blocks of ``type_id`` -- as float32.

    A flat ``(n_blocks * block_elems,)`` array, from llama.cpp's decoder.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
        handle.write(blocks)
        handle.flush()
        result = subprocess.run(
            [str(tool_path()), str(int(type_id)), handle.name], capture_output=True
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama.cpp's decoder for GGML type {type_id} failed: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return np.frombuffer(result.stdout, dtype="<f4")
