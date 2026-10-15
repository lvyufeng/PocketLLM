"""`ggml_rope_multi` in the vision mode, as the oracle for the tower's rotary.

The tower's rotary is the one piece of it this engine derives from reading
llama.cpp rather than from a tensor's shape: four positions per token, two of them
used (a patch's row and its column), a half-width rotation, and a frequency ramp
that restarts at each section boundary.  The tower-level comparison cannot check
it -- the reference's own f16 activations are a wider difference than a wrong
angle would be at the magnitudes involved.

So it is checked against the library, the way the patch order was:
:file:`ggml_rope_vision.c` calls `ggml_rope_multi` with `GGML_ROPE_TYPE_VISION`
and the positions clip.cpp builds, and the test drives this engine's own `rope` op
over a matching table.  Compiled here rather than by CMake, like the other
llama.cpp instruments in this directory.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

from llama_oracle import LLAMA_BUILD, LLAMA_CPP, repository_root


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "ggml-rope-vision"


def compile_tool() -> None:
    """Build the rotary instrument against llama.cpp's own ggml headers."""
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-o",
            str(output),
            str(pathlib.Path(__file__).resolve().parent / "ggml_rope_vision.c"),
            f"-I{LLAMA_CPP / 'ggml' / 'include'}",
            f"-L{LLAMA_BUILD}",
            "-lggml",
            "-lggml-base",
            "-lggml-cpu",
            f"-Wl,-rpath,{LLAMA_BUILD}",
            "-lm",
        ],
        check=True,
        capture_output=True,
    )


def rotate(x: np.ndarray, *, grid: int, merge: int) -> np.ndarray:
    """`ggml_rope_multi`'s answer for a `grid`-square patch grid.

    ``x`` is ``(grid**2, n_heads, head_dim)`` float32 with the head width fastest,
    which is both ggml's own order and the order this engine's `rope` op takes.
    """
    data = np.ascontiguousarray(x, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"x is {data.shape}, not (tokens, heads, head_dim)")
    n_pos, n_heads, head_dim = (int(v) for v in data.shape)
    if n_pos != grid * grid:
        raise ValueError(f"x has {n_pos} tokens and a {grid}-square grid has {grid * grid}")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "x.bin"
        path.write_bytes(data.tobytes())
        result = subprocess.run(
            [str(tool_path()), str(path), str(grid), str(merge), str(n_heads), str(head_dim)],
            capture_output=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "ggml_rope_multi failed: " + result.stderr.decode(errors="replace").strip()
        )
    flat = np.frombuffer(result.stdout, dtype="<f4")
    if flat.size != data.size:
        raise RuntimeError(f"ggml returned {flat.size} values, expected {data.size}")
    return flat.reshape(data.shape)
