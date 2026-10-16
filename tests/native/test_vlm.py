"""The whole vision-language path: a prompt with one image, and the next token.

`test_vision_tower.py` stops at the tower's features and `test_image_span.py` at
one batch of them.  This is the composition -- the text before the span, the span
itself, the text after -- against llama.cpp driving the same three chunks through
`mtmd_helper_eval_chunks` (see :mod:`mtmd_vlm`).  It is the only comparison that
can say whether an image *reaches* the model: a tower whose features are right and
a text graph that injects them in the wrong place passes both narrower tests and
fails this one.

## Why the first case is determinism rather than a bound

This path was written once and left unlanded because it did not agree with the
reference, and the disagreement turned out to be two separate things.  One of them
was a real bug and the cheapest of all to have caught: the token-batch-with-
positions entry point built the rotary table and **skipped the token gather**, so
the activation kept whatever the previous call had left in it.  The symptom was not
a wrong number but an *unstable* one -- the same command line produced the right
logits three runs in four and zeros the fourth, because the buffer it read was
freshly allocated and sometimes happened to be clean.

That is the failure the first case exists for, and it is worth saying why the two
narrower cases could not see it.  `test_image_span.py` drives the *embeddings*
branch, which is a different arm of the same `if`; and one call cannot show that a
buffer was never written when the buffer starts out zeroed.  Two calls compared to
each other can, and that needs no oracle, no geometry and no tolerance -- which is
what makes it the right shape for this bug rather than a tighter bound on an
existing case.

## And the second case is where the composition actually stands

The other half of the disagreement is **not** a bug and is still open.  Feeding the
engine llama.cpp's *own* image planes -- `--planes`, which takes the tower out of
the picture -- still disagrees with the reference, so what is missing is in how the
three calls compose.

Four candidates have since been eliminated by measurement rather than by argument,
and each elimination is worth more than the bound it replaces:

  - **The mask.**  Flipping the span's causal flag moves the engine by ~28% of the
    spread and the *wrong way* (the shipped causal arm agrees better), against a
    ~70% gap.  The engine's causal arm is also entry-for-entry identical to the
    reference's `(y, x)`-lexicographic rule on this grid.
  - **The cell count.**  The span does append 576 cache rows -- an append of one
    row would fit a 512-position window and it does not; the boundary is between
    512 (fails) and 576 (succeeds).  So the engine's own 576-row append is the
    right layout and the "it should share one slot" reading was wrong.
  - **The positions.**  The span's four components are element-wise identical to
    `mtmd_helper_image_get_decoder_pos`, including the row/column order and the
    `span_start` base.
  - **The tower.**  `--planes` removes it and the gap barely moves.
  - **The rotary itself.**  The engine's per-token table was recovered angle by
    angle and compared with `ggml_rope_multi`'s: every pair agrees to 6e-8, which is
    float32 rounding.  Same section assignment, same ramp, same base.  (Getting
    there took two attempts, and the first reported 199% -- a comparison measured
    against a misaligned input, which is the failure mode this file's own history
    warns about.  The angles are recoverable without ggml exposing its cache: feed
    the rotation a vector that is 1 in both members of every pair, and
    `atan2(out[p+half] - out[p], out[p+half] + out[p])` is the angle.)

What is left, and it is a *lead rather than a diagnosis*: the span batch's
activation **diverges with depth**.  Dumping its last row per block gives 3.5e3 at
block 0, 2.0e4 at block 12, 3.7e5 at block 20 and 6.2e6 at block 27, where the
reference's final hidden for the same batch is 1.8e4 -- a factor of 340, growing
exponentially from about block 13.  The plain token path does not do that (it
reaches 7.3e3 by block 7, the same order as the reference).

So the fault is in something only the span batch does, and the next probe is the
one that has not been run: the span batch's **Q, K and V as the engine actually
produces them**, post-projection and post-rotary, against llama.cpp's for the same
block -- rather than the table the rotary is run from, which is now known good.
The 576-row append interacting with the attention window is the other candidate,
and both are decided by the same comparison.

The case below asserts the facts that *are* settled (the token flow, and the span's
position count) and records that the logits do not yet match, rather than pretending
to a bound the path does not meet.  When the composition lands this is the file
whose assertions tighten, and `test_the_composition_is_what_is_missing` is the one
that turns into a logits comparison.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import numpy as np
import pytest

from llama_oracle import LLAMA_LIB, repository_root
from mtmd_vlm import compile_tool as compile_vlm_oracle
from mtmd_vlm import n_pos, n_tokens, run as run_oracle, tool_path as oracle_tool_path

#: The two files a vision checkpoint is.
MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-qwen3vl-2b-f16.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")

#: One image and one question around it.  The marker is `mtmd_default_marker()`,
#: and the text either side of it is what makes the two engines' token sequences
#: the same sequence rather than merely the same length.
PROMPT = "Describe this image: <__media__>"

pytestmark = pytest.mark.skipif(
    not LLAMA_LIB.is_file(), reason=f"no llama.cpp build at {LLAMA_LIB}"
)


def engine_tool_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-vlm"


def _image() -> np.ndarray:
    """The 768x768 picture the tower's own case uses, for the same reasons."""
    side = 768
    axis = np.arange(side, dtype=np.uint16)
    picture = np.empty((side, side, 3), dtype=np.uint8)
    picture[:, :, 0] = (axis[None, :] % 256).astype(np.uint8)
    picture[:, :, 1] = (axis[:, None] % 256).astype(np.uint8)
    picture[:, :, 2] = ((axis[None, :] + axis[:, None]) % 256).astype(np.uint8)
    return picture


def _skip_reason() -> str | None:
    if not engine_tool_path().is_file():
        return "build/pocketllm-vlm is not built"
    if not MMPROJ.is_file():
        return f"no mmproj at {MMPROJ}"
    if not TEXT_MODEL.is_file():
        return f"no text model at {TEXT_MODEL}"
    return None


def _engine(tmp_path: pathlib.Path, picture: np.ndarray, *, name: str,
            planes: pathlib.Path | None = None) -> tuple[np.ndarray, dict]:
    """Run the engine's whole path and return `(logits, reported shape)`.

    The shape lines (`n_tokens`, `n_positions`, `n_vocab`, `argmax`) are what the
    tool prints on *stdout*; its stderr carries the token ids, which is where a
    reader looks when something is wrong rather than what a case asserts on.

    The span's attention needs an f32 cache on the CPU, which is what the
    environment flag sets -- the non-causal arm refuses an f16 one rather than
    widening it, and `POCKETLLM_CPU_KV_F32` is how a CPU session asks.
    """
    rgb = tmp_path / f"{name}.rgb"
    out = tmp_path / f"{name}.bin"
    rgb.write_bytes(np.ascontiguousarray(picture, dtype=np.uint8).tobytes())
    command = [
        str(engine_tool_path()),
        str(MMPROJ),
        str(TEXT_MODEL),
        str(rgb),
        str(picture.shape[1]),
        str(picture.shape[0]),
        PROMPT,
        str(out),
    ]
    if planes is not None:
        command += ["--planes", str(planes)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "POCKETLLM_CPU_KV_F32": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"pocketllm-vlm failed: {result.stdout}{result.stderr}")
    reported = dict(line.split() for line in result.stdout.splitlines() if line.strip())
    logits = np.frombuffer(out.read_bytes(), dtype="<f4").astype(np.float64)
    return logits, reported


def test_the_path_is_deterministic(tmp_path: pathlib.Path) -> None:
    """The same call twice is the same logits.

    An equality rather than a tolerance, and not against an oracle: every input on
    this path is deterministic once it is read, so any difference between two runs
    is this engine reading memory it did not write.  That is exactly what happened
    -- see the module docstring -- and it is invisible to every other case in this
    directory.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    picture = _image()

    first, _ = _engine(tmp_path, picture, name="first")
    second, _ = _engine(tmp_path, picture, name="second")
    assert np.array_equal(first, second), (
        "the same call produced different logits, so something on this path reads "
        f"memory it did not write ({int((first != second).sum())} of {first.size} values differ)"
    )


def test_the_composition_carries_the_reference_layout(tmp_path: pathlib.Path) -> None:
    """The token flow and the span's position count, against `libmtmd`.

    The two facts that were settled before any logit was: how many tokens the
    prompt is, and how many *positions* the image span consumes.  They earn a case
    of their own because a composition error moves them first -- a span that took
    the wrong number of cache rows, or a batch after it that continued from the
    wrong place, changes these before it changes a logit.

    The second number is the interesting one: a `qwen3vl` span is 576 tokens
    carrying **24** positions (`max(nx, ny)` of them), so the text after an image
    continues from a position its token index never reaches.  An engine that
    conflated the two would still tokenize this prompt correctly and still produce
    plausible logits.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    if not oracle_tool_path().is_file():
        compile_vlm_oracle()

    picture = _image()
    reference = run_oracle(MMPROJ, TEXT_MODEL, picture, PROMPT)
    _, reported = _engine(tmp_path, picture, name="layout")

    assert int(reported["n_tokens"]) == n_tokens(reference), (
        f"the engine walked {reported.get('n_tokens')} tokens and the reference "
        f"{n_tokens(reference)}"
    )
    # The number this case exists for: 576 tokens, 24 positions.  An engine that
    # used the token count as the position count would sail past it.
    assert n_pos(reference) == 24, (
        f"the reference's span consumed {n_pos(reference)} positions, and a 576-token span "
        "of a 24x24 patch grid is 24 of them"
    )
    # Six text tokens before the span and one `<|vision_end|>` after it, and the
    # span between them counted in *positions*: the engine's last position is the
    # one number that says it kept the two apart.
    assert int(reported["n_positions"]) == 6 + n_pos(reference) + 1, (
        f"the engine's last position is {reported.get('n_positions')}, and six text tokens "
        f"plus {n_pos(reference)} span positions plus one after it is 31"
    )


@pytest.mark.xfail(
    strict=False,
    reason="the composition of the three calls does not match llama.cpp yet; the tower is "
    "exonerated by --planes, and this case is the one that turns green when the rest lands",
)
def test_the_composition_is_what_is_missing(tmp_path: pathlib.Path) -> None:
    """The tower is not the disagreement, and this records how far off the rest is.

    Feeding the engine llama.cpp's own image planes takes the tower out of the
    picture, and the logits still do not match the reference -- so whatever is
    missing is in how the three calls compose.  Measured with the tower replaced by
    the reference's own features: **70%** of the logit spread, correlation 0.13.

    **Marked expected-to-fail rather than deleted or loosened**, and the
    distinction matters.  A bound loose enough to pass here would accept an
    unrelated answer, which is exactly what a correlation of 0.13 is; deleting it
    would throw away the measurement; and recording it in
    `tests/baseline_failures.txt` would make it a permanent part of the tree's
    definition of normal, which a path that is meant to work is not.  `strict=False`
    means the case stops being an xfail the moment the composition lands, without
    anyone having to remember to come back for it.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    if not oracle_tool_path().is_file():
        compile_vlm_oracle()

    picture = _image()
    reference = run_oracle(MMPROJ, TEXT_MODEL, picture, PROMPT)
    want = reference["logits"]

    # The engine's *own* tower features, written where `--planes` can read them.
    #
    # **What this bisect does and does not rule out, stated carefully because an
    # earlier revision of this comment overstated it.**  `pocketllm-vision` runs the
    # tower, so the features come from this engine and not from llama.cpp's; they
    # are close to the reference's (the tower's own case measures them at 2.7e-2 of
    # the spread) but they are not the reference's.  So this removes the tower's
    # *graph* from the comparison -- a second run of it, with different planes, no
    # longer changes the answer -- and it does **not** remove the tower's output
    # error.  Feeding llama.cpp's own features is a different experiment, and it was
    # run by hand: the gap barely moved (0.794 against 0.754), which is what makes
    # the tower's error not the explanation.
    planes, _ = _engine_planes(tmp_path, picture)
    got, _ = _engine(tmp_path, picture, name="bisect", planes=planes)

    spread = float(want.max() - want.min())
    worst = float(np.max(np.abs(got - want)))
    assert worst <= 0.05 * spread, (
        f"the engine and the reference disagree by {worst:.4g} over a logit spread of "
        f"{spread:.4g} ({worst / spread:.1%}), correlation {np.corrcoef(want, got)[0, 1]:.3f}, "
        "with the tower's graph taken out of the path -- so what remains is in how the "
        "three calls compose"
    )


def _engine_planes(tmp_path: pathlib.Path, picture: np.ndarray) -> tuple[pathlib.Path, str]:
    """The tower's planes for `picture`, written through `pocketllm-vision`.

    The tool that already owns that dump, rather than a second way to produce the
    same numbers: `--planes` takes exactly the file `pocketllm-vision` writes.
    """
    rgb = tmp_path / "planes.rgb"
    out = tmp_path / "planes.bin"
    rgb.write_bytes(np.ascontiguousarray(picture, dtype=np.uint8).tobytes())
    result = subprocess.run(
        [
            str(repository_root() / "build" / "pocketllm-vision"),
            str(MMPROJ),
            str(rgb),
            str(picture.shape[1]),
            str(picture.shape[0]),
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pocketllm-vision failed: {result.stdout}{result.stderr}")
    return out, result.stdout