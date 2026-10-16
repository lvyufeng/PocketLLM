"""`libmtmd` driving a whole vision-language eval, as the oracle for the composition.

The tower's own oracle (:mod:`mtmd_embed`) stops at the image's features and
`test_image_span.py` stops at one batch of them.  This is the rest of the path --
the text before the span, the span itself, the text after -- and it is the only
comparison that can say whether an image *reaches* the model.

**The reference is `mtmd_helper_eval_chunks`, and that is the point of the
instrument.**  An earlier version of `mtmd_vlm.c` drove the three chunks by hand
through the public API and got the span's multi-section positions wrong: llama.cpp
takes `pos[i + section * n_tokens]` for an *image* batch and `pos[section]` shared
across the batch for a *text* one, and hand-rolling the first as the second is a
plausible mistake that produces logits which merely look wrong.  The helper is the
library's own answer, so this instrument asks it rather than reimplementing it.

Compiled here rather than by CMake, like the other llama.cpp instruments in this
directory.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

import numpy as np

from llama_oracle import LLAMA_BUILD, LLAMA_CPP, repository_root


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "mtmd-vlm"


def compile_tool() -> None:
    """Build the whole-path instrument against llama.cpp's own mtmd headers."""
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-o",
            str(output),
            str(pathlib.Path(__file__).resolve().parent / "mtmd_vlm.c"),
            f"-I{LLAMA_CPP / 'include'}",
            f"-I{LLAMA_CPP / 'ggml' / 'include'}",
            f"-I{LLAMA_CPP / 'tools' / 'mtmd'}",
            f"-L{LLAMA_BUILD}",
            "-lmtmd",
            "-lllama",
            "-lggml",
            "-lggml-base",
            f"-Wl,-rpath,{LLAMA_BUILD}",
        ],
        check=True,
        capture_output=True,
    )


def run(mmproj: pathlib.Path, text_model: pathlib.Path, image: np.ndarray, prompt: str) -> dict:
    """The reference's answer for one image and one prompt.

    Returns the parsed JSON the instrument writes: `n_vocab`, `n_tokens` and the
    last position's `logits`, plus the `n_pos` the span consumed, which the
    instrument reports on stderr and this reads back into the same dict because it
    is part of the contract a caller has to reproduce.
    """
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"the image is {rgb.shape}, not (height, width, 3)")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "image.rgb"
        path.write_bytes(rgb.tobytes())
        result = subprocess.run(
            [
                str(tool_path()),
                str(mmproj),
                str(text_model),
                str(path),
                str(rgb.shape[1]),
                str(rgb.shape[0]),
                prompt,
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"mtmd-vlm failed: {result.stdout}{result.stderr}")
    parsed = json.loads(result.stdout)
    parsed["logits"] = np.array(parsed["logits"], dtype=np.float64)
    return parsed


def n_tokens(reference: dict) -> int:
    """How many tokens the prompt is, image included."""
    return int(reference["n_tokens"])


def n_pos(reference: dict) -> int:
    """How many *positions* the image span consumes, which is not its token count.

    A `qwen3vl` span is 576 tokens carrying 24 positions -- `max(nx, ny)` of them --
    so the text after an image continues from a position its token index does not
    reach.  That gap is the thing a composition gets wrong.
    """
    return int(reference["n_pos"])