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

## And the second case is the one that closed it

It spent four commits as an `xfail` recording a 70%-of-spread disagreement, and the
cause turned out to be an **engine bug with nothing to do with images**:
`ensure_kv` released the KV slabs and reallocated them empty on growth, so the six
`<|vision_start|>` tokens written before the span were zeros in the span's key set.
A vision prompt crosses the initial 256-position capacity *inside its image span*,
which is why this path found it and no other test did.  Carrying the rows across the
growth took this case from correlation 0.103 to 0.994.

Thirteen candidates were eliminated by measurement before that, and they are recorded
here because each is a place the next reader does not have to look:

  - **The mask.**  Flipping the span's causal flag moves the engine by ~28% of the
    spread and the *wrong way*, and the engine's causal arm is entry-for-entry
    identical to the reference's `(y, x)`-lexicographic rule on this grid.
  - **The key set and visibility.**  A per-row set comparison: 0 of 576 span rows
    differ, 166176/166176 inside the span and 169632/169632 with the text before it.
  - **The cell count.**  The span does append 576 cache rows -- `n_ctx` 512 fails and
    576 succeeds -- so the engine's layout is right and "it should share one slot"
    was wrong.
  - **The positions.**  Element-wise identical to
    `mtmd_helper_image_get_decoder_pos`, row/column order and base included.
  - **The rotary table.**  Every pair agrees with `ggml_rope_multi` to 6e-8, angles
    recovered pair by pair.  (This one took two attempts and the first reported 199%
    -- a comparison whose two sides were not driven from the same input, which is the
    failure mode this file has now hit three times.  The angles are recoverable
    without ggml exposing its cache: feed the rotation a vector that is 1 in both
    members of every pair and `atan2(out[p+half] - out[p], out[p+half] + out[p])` is
    the angle.)
  - **The tower.**  Feeding llama.cpp's own `mtmd_encode` features moves the gap from
    0.75 to 0.79 -- no closer.
  - **The span's inputs.**  The tensor-by-tensor comparison that finally pointed at
    the answer: every input to block 0's attention -- embeddings, post-norm, Q, K
    after the rotary, V -- agreed to float32 noise while the attention *output* was
    86% of its own scale off.  Inputs right and output wrong is what reading zeroed
    keys looks like; the elimination was of the hypothesis that any of those tensors
    was wrong, and it is what made the search look at memory instead of arithmetic.

The two cases below are therefore an assertion each: the path is deterministic, and
it agrees with llama.cpp's own composition over the whole prompt.
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
from test_op_conformance import QUANTIZED_RTOL

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


def test_the_whole_path_matches_llama_cpp(tmp_path: pathlib.Path) -> None:
    """The composition, against llama.cpp's own.

    This case spent four commits as an `xfail` recording a 70%-of-spread
    disagreement, and it is the case that found the bug: the engine's `ensure_kv`
    released the KV slabs and reallocated them **empty** on growth, so the six
    `<|vision_start|>` tokens written before the image were zeros in the span's key
    set -- a vision prompt crosses the initial 256-position capacity inside its image
    span.  Carrying the rows across the growth took this from correlation 0.103 to
    0.994.

    It is deliberately kept as a whole-path case rather than replaced by tighter
    per-op ones, because the failure it caught was invisible to every narrower test:
    the tower, the span's positions, its mask, its key set and every input to block
    0's attention were all measured correct while this disagreed.

    The features come from this engine's own tower (`--planes`); what that isolates
    and what it does not is spelled out below.
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
    # `QUANTIZED_RTOL` is the tree's bound for this kind of comparison -- a quantized
    # llama.cpp against this engine's f32 path -- and the measured value is 4.3% of the
    # spread at correlation 0.994.  Left at the named bound rather than tightened
    # around the measurement: the residue here is the two engines' arithmetic, which
    # moves with the checkpoint and the card, and a bound fitted to one run of one
    # checkpoint fails the next one for a reason that is not a regression.
    assert worst <= QUANTIZED_RTOL * spread, (
        f"the engine and the reference disagree by {worst:.4g} over a logit spread of "
        f"{spread:.4g} ({worst / spread:.1%}), correlation {np.corrcoef(want, got)[0, 1]:.3f}"
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