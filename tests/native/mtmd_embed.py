"""llama.cpp's vision tower, as the oracle for the tower this engine grows.

The text half of a `qwen3vl` checkpoint already has an oracle in
:mod:`llama_oracle` -- llama.cpp's logits for a token sequence.  The tower needs
its own, because it is a *different graph*: a separate GGUF with its own
architecture, its own preprocessing contract, and an output that is not a
distribution over tokens but a block of image features.  `libmtmd` is that
oracle: `mtmd_encode` runs the tower, and `mtmd_get_output_embd` hands back what
it produced.

Compiled here rather than by CMake, like the three other llama.cpp instruments
in this directory, so a checkout of that project stays a dependency of the test
that needs it.

**The image is handed over as raw bytes.**  `mtmd_bitmap_init` takes an RGB
buffer, so the caller authors exactly the pixels the tower sees and no image
decoder stands between the two implementations.  What the library *does* do to
them -- the resize and the normalization into `(pixel - mean) / std` -- is part
of the contract being tested, and it happens inside `mtmd_tokenize`.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

from llama_oracle import LLAMA_BUILD, LLAMA_CPP, repository_root


def tool_path() -> pathlib.Path:
    return repository_root() / "build" / "mtmd-embed"


def compile_tool() -> None:
    """Build the tower oracle against llama.cpp's own mtmd headers."""
    output = tool_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-o",
            str(output),
            str(pathlib.Path(__file__).resolve().parent / "mtmd_embed.c"),
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


def embed(
    mmproj: pathlib.Path,
    text_model: pathlib.Path,
    image: np.ndarray,
    *,
    n_embd_out: int,
) -> tuple[np.ndarray, int]:
    """``(embeddings, n_tokens)`` for ``image``, from llama.cpp's tower.

    ``image`` is ``(height, width, 3)`` uint8 in the RGB order the bitmap takes.
    ``n_embd_out`` is the number of floats one token carries -- see
    :file:`mtmd_embed.c`, which explains why the caller has to state it.
    """
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"the image is {rgb.shape}, not (height, width, 3)")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    with tempfile.TemporaryDirectory() as tmp:
        rgb_path = pathlib.Path(tmp) / "image.rgb"
        out_path = pathlib.Path(tmp) / "embeddings.bin"
        rgb_path.write_bytes(rgb.tobytes())
        result = subprocess.run(
            [
                str(tool_path()),
                str(mmproj),
                str(text_model),
                str(rgb_path),
                str(width),
                str(height),
                str(int(n_embd_out)),
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "mtmd_encode failed: " + result.stderr.decode(errors="replace").strip()
            )
        reported: dict[str, int] = {}
        for line in result.stdout.decode(errors="replace").splitlines():
            key, _, value = line.partition(" ")
            if value.strip().isdigit():
                reported[key] = int(value)
        payload = out_path.read_bytes()
    if "n_tokens" not in reported or "n_embd_out" not in reported:
        raise RuntimeError(f"the oracle printed no shape: {result.stdout!r}")
    n_tokens = reported["n_tokens"]
    if reported["n_embd_out"] != int(n_embd_out):
        raise RuntimeError(f"the oracle ran at width {reported['n_embd_out']}, not {n_embd_out}")
    expected = n_tokens * int(n_embd_out)
    flat = np.frombuffer(payload, dtype="<f4")
    if flat.size != expected:
        raise RuntimeError(f"the oracle wrote {flat.size} floats, expected {expected}")
    return flat.reshape(n_tokens, int(n_embd_out)), n_tokens
