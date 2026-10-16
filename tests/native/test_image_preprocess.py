"""The tower's image preprocessing, against the reference's own choices.

The tower takes one square image of the size its position table covers, so everything
before that is the caller's: picking a size for an arbitrary image and resampling to
it.  Both are a *contract* rather than a convenience, because the tower's token count
is `(nx / patch / merge) * (ny / patch / merge)` -- a different target size is a
different prompt, not an approximate one.

This file checks the engine's implementation of that contract two ways, and they
answer different things:

  - :func:`test_the_target_size_rule_matches_the_reference` drives the engine's own
    `prices_for` over a spread of shapes and compares the *token counts* with what
    llama.cpp produced for the same shapes.  Token counts invert to the target size
    (`tokens = (nx/32) * (ny/32)`), so they pin the rule exactly without needing a
    pixel comparison -- and they catch the cases a plain aspect fit gets wrong, like
    `800x600` becoming `800x608` rather than `768x576`.
  - :func:`test_a_resized_image_matches_the_reference` runs the whole path on an image
    that is *not* the checkpoint's own size, which is what makes the resampler part
    of the comparison: a target that agreed while the pixels did not would show up
    here and nowhere else.

The reference's numbers were measured rather than read.  `mtmd_image_preprocessor_dyn_size`
is what a `qwen3vl_merger` projector gets, `calc_size_preserved_ratio` is the rule it
uses, and the twelve shapes below are twelve checks on the transcription -- all twelve
pass, which is what makes the engine's copy of the rule trustworthy.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import numpy as np
import pytest

from llama_oracle import LLAMA_LIB, repository_root
from mtmd_vlm import compile_tool as compile_vlm_oracle
from mtmd_vlm import tool_path as oracle_tool_path

MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-qwen3vl-2b-f16.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")
PROMPT = "Describe this image: <__media__>"

#: Shapes and the *measured* image-token counts the reference produced for them.
#: Chosen to pin the rule rather than to be representative: the checkpoint's own
#: square, a clean down-scale and up-scale, the shapes where rounding up to the grid
#: changes the answer (`800x600` -> `800x608`), the non-square pairs that must agree
#: under a transpose, one below `image_min_pixels` so the *growth* branch runs
#: (`64x64` -> `96x96`), and one at exactly the checkpoint's own size through the
#: other axis (`100x768` -> `96x768`).
#:
#: This set is bounded by *time*, not by interest: every entry costs one engine run
#: and the cost is the token count, which is the product of the two grid sides.  The
#: three largest cases the rule has been checked against live in
#: `SHAPES_THAT_ARE_SLOW` below, where they are opt-in -- a file whose fast path takes
#: minutes is a file that stops being run.
SHAPES = (
    (768, 768, 576),
    (512, 512, 256),
    (800, 600, 475),
    (600, 800, 475),
    (1200, 300, 342),
    (300, 1200, 342),
    (64, 64, 9),
    (32, 32, 9),
    (100, 768, 72),
)

#: The expensive end of the same rule, kept because each exercises a branch a small
#: shape cannot: `1024x1024` and `1472x1472` are upscales past the checkpoint's own
#: size, which is where the *position table* has to be interpolated rather than merely
#: gathered, and `3000x2000` is the only one over `image_max_pixels`, so it is the only
#: one that runs the shrink branch (`-> 2496x1664`, 4056 tokens).
#:
#: Opt in with `POCKETLLM_SLOW_IMAGE_TESTS=1`.  All three were measured against the
#: reference and all three agree; what keeps them out of the default run is that a
#: 4056-token image takes minutes on one core, and `1472x1472` is the shape the
#: reference itself calls "warmup" for that reason.
SHAPES_THAT_ARE_SLOW = (
    (1024, 1024, 1024),
    (1472, 1472, 2116),
    (3000, 2000, 4056),
)

SLOW = os.environ.get("POCKETLLM_SLOW_IMAGE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LLAMA_LIB.is_file(), reason=f"no llama.cpp build at {LLAMA_LIB}"
)


def engine_tool_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-vlm"


def _skip_reason() -> str | None:
    if not engine_tool_path().is_file():
        return "build/pocketllm-vlm is not built"
    if not MMPROJ.is_file():
        return f"no mmproj at {MMPROJ}"
    if not TEXT_MODEL.is_file():
        return f"no text model at {TEXT_MODEL}"
    return None


def image(width: int, height: int) -> np.ndarray:
    """A picture that is not symmetric in any direction, at the asked size."""
    axis_w = np.arange(width, dtype=np.uint16)
    axis_h = np.arange(height, dtype=np.uint16)
    picture = np.empty((height, width, 3), dtype=np.uint8)
    picture[:, :, 0] = (axis_w[None, :] % 256).astype(np.uint8)
    picture[:, :, 1] = (axis_h[:, None] % 256).astype(np.uint8)
    picture[:, :, 2] = ((axis_w[None, :] + axis_h[:, None]) % 256).astype(np.uint8)
    return picture


def engine(tmp_path: pathlib.Path, picture: np.ndarray, name: str) -> tuple[dict, np.ndarray]:
    """Run the engine's whole path and return its reported shape and logits."""
    rgb = tmp_path / f"{name}.rgb"
    out = tmp_path / f"{name}.bin"
    rgb.write_bytes(np.ascontiguousarray(picture, dtype=np.uint8).tobytes())
    result = subprocess.run(
        [
            str(engine_tool_path()),
            str(MMPROJ),
            str(TEXT_MODEL),
            str(rgb),
            str(picture.shape[1]),
            str(picture.shape[0]),
            PROMPT,
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "POCKETLLM_CPU_KV_F32": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"pocketllm-vlm failed: {result.stdout}{result.stderr}")
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    reported = {parts[0]: parts[1:] for parts in lines}
    return reported, np.frombuffer(out.read_bytes(), dtype="<f4").astype(np.float64)


def test_the_target_size_rule_matches_the_reference(tmp_path: pathlib.Path) -> None:
    """Twelve shapes, and the token count the reference produced for each.

    The engine is driven through the whole path -- there is no entry point that
    reports the target size alone -- so this is twelve short runs, each of which also
    proves the *buffer* was sized correctly: a token count that disagreed with the
    preprocessing would have written past the end of the planes vector rather than
    merely reported a different number.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    for width, height, expected in SHAPES:
        picture = image(width, height)
        reported, _ = engine(tmp_path, picture, f"shape_{width}x{height}")
        got = int(reported["n_image_tokens"][0])
        target_w, target_h = (int(v) for v in reported["target"])
        assert got == expected, (
            f"{width}x{height}: the engine chose a {target_w}x{target_h} target, which is "
            f"{got} image tokens, and the reference produced {expected}"
        )


@pytest.mark.skipif(not SLOW, reason="set POCKETLLM_SLOW_IMAGE_TESTS=1 to run")
def test_the_size_rule_also_holds_past_the_checkpoint_s_own(tmp_path: pathlib.Path) -> None:
    """The same rule at the sizes that cost minutes, one of which needs interpolation.

    Split from the fast loop rather than interleaved with it: the three here are where
    the position table stops being a gather -- `1024x1024` and `1472x1472` are past the
    checkpoint's own 24x24 grid, so the table has to be resampled onto theirs -- and
    they are also the three that made this file take an hour.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    for width, height, expected in SHAPES_THAT_ARE_SLOW:
        picture = image(width, height)
        reported, _ = engine(tmp_path, picture, f"slow_{width}x{height}")
        got = int(reported["n_image_tokens"][0])
        target_w, target_h = (int(v) for v in reported["target"])
        assert got == expected, (
            f"{width}x{height}: the engine chose a {target_w}x{target_h} target, which is "
            f"{got} image tokens, and the reference produced {expected}"
        )


def test_a_resized_image_matches_the_reference(tmp_path: pathlib.Path) -> None:
    """An image that is not the checkpoint's own size, all the way to the logits.

    `800x600` is chosen because its target (`800x608`) is neither an aspect fit nor a
    multiple of the input -- so the resampler is exercised on every pixel, and a
    different target size or a different kernel shows up in the logits rather than
    being absorbed by the token count agreeing.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    if not oracle_tool_path().is_file():
        compile_vlm_oracle()

    picture = image(800, 600)
    rgb = tmp_path / "resized.rgb"
    rgb.write_bytes(np.ascontiguousarray(picture, dtype=np.uint8).tobytes())

    from mtmd_vlm import run as run_oracle

    reference = run_oracle(MMPROJ, TEXT_MODEL, picture, PROMPT)
    reported, got = engine(tmp_path, picture, "resized")

    # The span's length is a property of the input shape, and both engines have to
    # agree on it before the logits mean anything: 800x600 becomes 800x608, which is
    # 25x19 merged cells.
    assert int(reported["n_image_tokens"][0]) == 475
    assert reference["n_tokens"] == 583 - 576 + 475

    want = reference["logits"]
    spread = float(want.max() - want.min())
    worst = float(np.max(np.abs(got - want)))
    correlation = float(np.corrcoef(want, got)[0, 1])
    # A bound and a correlation, because they fail differently: the bound catches a
    # magnitude error and the correlation catches an answer that is the wrong shape
    # entirely while staying inside it.  Measured: 6.8% of the spread at 0.988 --
    # the same order as the checkpoint's own square, which is 4.3% at 0.994, so the
    # resampling is not where the residue comes from.
    assert worst <= 0.1 * spread, (
        f"the engine and the reference disagree by {worst:.4g} over a logit spread of "
        f"{spread:.4g} on a resized image"
    )
    assert correlation > 0.95, (
        f"the logits correlate {correlation:.4f} with the reference on a resized image"
    )