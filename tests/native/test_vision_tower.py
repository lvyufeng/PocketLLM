"""The vision tower, against llama.cpp's own tower.

`test_forward.py` does this for the text graph and this is the other half of the
same model: a `qwen3vl` checkpoint's `mmproj` is a second GGUF with its own
architecture, its own preprocessing contract and an output that is a block of
image features rather than a distribution over tokens.  `libmtmd` is its oracle
(:mod:`mtmd_embed`) and `src/tools/vision.cpp` is the engine's side of the
comparison.

## The image is authored here, and so is the reason

The tower's preprocessing -- the resize and the normalization -- is llama.cpp's,
and reproducing the whole of it is not what this test is for.  What it does
instead is hand *both* sides the same raw 768x768 RGB buffer: at that size the
preprocessor's resize is the identity (both sides confirm it by producing 576
tokens), so the only transform left is `(value / 255 - mean) / std`, which
`Qwen3VlVision::encode` performs with the checkpoint's own statistics.  A resize
would be a *second* algorithm to agree about; there is none to agree about here.

## The comparison is between two widths, and that is the oracle's doing

llama.cpp builds this tower with the file's **f16** weights, and ggml's f16-weight
product narrows its float32 *activation* to f16 on the way in.  So the reference
rounds every matmul's input -- four per block, twenty-four blocks -- where this
engine keeps f32 throughout, and the difference that accumulates is the reference
being **less** precise rather than this engine being wrong.

That is a measured claim and not a comfortable one.  `tests/native/ggml_conv.py`
and its neighbours established the layout questions separately (the patch order,
the position table's raster, the rotary) and the loudest of them is exact: an
independent probe of `ggml_rope_multi` with `GGML_ROPE_TYPE_VISION`
(:mod:`ggml_rope_vision`, driven by the rotary case at the bottom of this file)
agrees with this engine's per-token rotary table to 1.2e-7, which is float32
rounding and not a coincidence.  What is left is precision, and the shape of what
is left is the shape of precision: across a plane of 1.18 million values the *median* element
sits 5e-5 of the plane's spread from the reference and the 99th percentile 1.2e-3,
while a handful of elements in the tail reach 2.7e-2 -- each one a place where a
sum nearly cancels and a relative bound stops meaning anything.

So the test asserts two things, and the tight one is the robust statistic:

  * :data:`TYPICAL_RTOL` on the 99th percentile, which is what a structural error
    -- a swapped axis, a mis-scaled angle, a wrong half -- fails immediately;
  * :data:`WORST_RTOL` on the single worst element, as a guard that cannot be
    defeated by a plane whose tail happens to be quiet.  This one is deliberately
    loose and deliberately last: a maximum over a million noisy elements is not a
    stable number, and a bound tight enough to mean something there would fail on
    the reference's own rounding.

The depth profile is the other half of the evidence.  The first deepstack tap is
read off block 5 and agrees with the reference to 4.7e-4 of its spread -- f16
epsilon, five blocks in -- so the front end, the rotary, the attention and five
blocks are all *right*, and what grows from there is the reference's rounding
accumulating through the remaining nineteen.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pytest

from llama_oracle import LLAMA_LIB, repository_root
from ggml_rope_vision import compile_tool as compile_rotary_oracle
from ggml_rope_vision import rotate, tool_path as rotary_tool_path
from mtmd_embed import compile_tool as compile_oracle
from mtmd_embed import embed, tool_path as oracle_tool_path
from test_op_conformance import opcheck_path, run, write_request

#: The tower's own file, beside the text checkpoint the text half is tested with.
MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-qwen3vl-2b-f16.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")

#: How far the engine's 99th-percentile element may sit from llama.cpp's, as a
#: fraction of the plane's spread.  Measured at 1.2e-3 (the base plane, the
#: widest of the four) and 4.5e-4 (the first tap); the headroom covers the
#: reference's own run-to-run stability rather than this engine's.
TYPICAL_RTOL = 5e-3

#: How far the single worst element may sit, for the reason the module docstring
#: gives: a maximum over 1.18 million values is not a stable statistic.  Measured
#: at 2.7e-2.
WORST_RTOL = 5e-2

pytestmark = pytest.mark.skipif(
    not LLAMA_LIB.is_file(), reason=f"no llama.cpp build at {LLAMA_LIB}"
)


def engine_tool_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-vision"


def _missing() -> str | None:
    if not engine_tool_path().is_file():
        return "build/pocketllm-vision is not built"
    if not MMPROJ.is_file():
        return f"no mmproj at {MMPROJ}"
    if not TEXT_MODEL.is_file():
        return f"no text model at {TEXT_MODEL}"
    return None


def image() -> np.ndarray:
    """A 768x768 RGB image that is not symmetric in any direction.

    The channel ramps run along different axes -- red along the row, green down
    the column, blue along both -- so a transpose of either spatial axis, a
    swapped channel, or a token order that is merely rotated all move a value
    somewhere else in the image rather than onto itself.
    """
    side = 768
    axis = np.arange(side, dtype=np.uint16)
    picture = np.empty((side, side, 3), dtype=np.uint8)
    picture[:, :, 0] = (axis[None, :] % 256).astype(np.uint8)
    picture[:, :, 1] = (axis[:, None] % 256).astype(np.uint8)
    picture[:, :, 2] = ((axis[None, :] + axis[:, None]) % 256).astype(np.uint8)
    return picture


def _engine(picture: np.ndarray, tmp: pathlib.Path) -> np.ndarray:
    """The engine's planes, as ``(n_planes, n_tokens, projection_dim)``."""
    rgb = pathlib.Path(tmp) / "image.rgb"
    out = pathlib.Path(tmp) / "features.bin"
    rgb.write_bytes(np.ascontiguousarray(picture, dtype=np.uint8).tobytes())
    result = subprocess.run(
        [
            str(engine_tool_path()),
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
    reported = dict(line.split() for line in result.stdout.splitlines() if line.strip())
    planes = int(reported["n_planes"])
    tokens = int(reported["n_tokens"])
    dim = int(reported["projection_dim"])
    flat = np.frombuffer(out.read_bytes(), dtype="<f4")
    if flat.size != planes * tokens * dim:
        raise RuntimeError(f"the engine wrote {flat.size} floats, expected {planes*tokens*dim}")
    return flat.reshape(planes, tokens, dim)


def _assert_plane(got: np.ndarray, want: np.ndarray, *, plane: int) -> None:
    """One plane against the oracle's, at the two bounds the module documents."""
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    spread = float(np.max(np.abs(want)))
    if spread == 0.0:
        raise AssertionError(f"plane {plane} is all zeros in the reference")
    ninetieth = float(np.percentile(difference, 99.0))
    worst = float(np.max(difference))
    assert ninetieth <= TYPICAL_RTOL * spread, (
        f"plane {plane}: the 99th-percentile difference is {ninetieth} over a scale of {spread}, "
        f"which is {ninetieth / spread:.3g} of it -- past the {TYPICAL_RTOL} a structural error "
        "would blow through"
    )
    assert worst <= WORST_RTOL * spread, (
        f"plane {plane}: the worst element differs by {worst} over a scale of {spread}"
    )


def test_the_tower_matches_llama_cpp(tmp_path: pathlib.Path) -> None:
    reason = _missing()
    if reason is not None:
        pytest.skip(reason)
    if not oracle_tool_path().is_file():
        compile_oracle()

    picture = image()
    got = _engine(picture, tmp_path)

    # llama.cpp's tower emits one plane per token-group concatenated along the
    # feature axis, which is the transpose of this engine's plane-major layout;
    # slicing the reference is that transpose, and nothing else.
    oracle, n_tokens = embed(MMPROJ, TEXT_MODEL, picture, n_embd_out=got.shape[0] * got.shape[2])
    assert oracle.shape == (got.shape[1], got.shape[0] * got.shape[2]), (
        f"the oracle produced {oracle.shape} and the engine {got.shape}"
    )
    assert n_tokens == got.shape[1]

    for plane in range(got.shape[0]):
        want = oracle[:, plane * got.shape[2] : (plane + 1) * got.shape[2]]
        _assert_plane(got[plane], want, plane=plane)


# --------------------------------------------------------------------------
# The rotary, against `ggml_rope_multi` rather than against the tower.
# --------------------------------------------------------------------------

#: A small grid with the tower's real head geometry: `head_dim` 64 makes the
#: rotation 32 pairs wide and puts the section boundaries at 16 and 32, which is
#: the arithmetic being checked; the grid only has to be big enough to have more
#: than one merged cell and more than one row of them.
ROTARY_GRID = 8
ROTARY_MERGE = 2
ROTARY_HEADS = 2
ROTARY_HEAD_DIM = 64

#: How far the two rotations may differ.  This is two float32 computations of the
#: same expression over a 64-wide vector, so the bound is rounding and nothing
#: else -- measured at 1.2e-7, and set an order wider only so a compiler's
#: reassociation of the pair's two products cannot fail the test.
ROTARY_RTOL = 1e-6


def _rotary_tables() -> tuple[np.ndarray, np.ndarray]:
    """This engine's rotary table, as ``(cos, sin)`` of ``(n_pos, head_dim/2)``.

    A second statement of `Qwen3VlVision::build_rope_tables`, written from the
    same reference loop, so that what the test drives the engine's `rope` op with
    is the table the model would build -- and so that a change to one without the
    other is a failure rather than a silent divergence.

    The arithmetic is float32 throughout, in the same places llama.cpp's is: the
    angle is accumulated by multiplication rather than recomputed as a power.
    """
    n_pos = ROTARY_GRID * ROTARY_GRID
    merged_side = ROTARY_GRID // ROTARY_MERGE
    half = ROTARY_HEAD_DIM // 2
    section = ROTARY_HEAD_DIM // 4
    scale = np.float32(10000.0 ** (-2.0 / half))
    cos_table = np.zeros((n_pos, half), dtype=np.float32)
    sin_table = np.zeros((n_pos, half), dtype=np.float32)
    for token in range(n_pos):
        within = token % (ROTARY_MERGE * ROTARY_MERGE)
        cell = token // (ROTARY_MERGE * ROTARY_MERGE)
        row = ROTARY_MERGE * (cell // merged_side) + within // ROTARY_MERGE
        column = ROTARY_MERGE * (cell % merged_side) + within % ROTARY_MERGE
        theta_row = np.float32(row)
        theta_column = np.float32(column)
        for pair in range(half):
            if pair == section:
                theta_column = np.float32(column)
            theta = theta_row if pair < section else theta_column
            cos_table[token, pair] = np.cos(theta)
            sin_table[token, pair] = np.sin(theta)
            theta_row = np.float32(theta_row * scale)
            theta_column = np.float32(theta_column * scale)
    return cos_table, sin_table


def test_the_rotary_matches_ggml() -> None:
    """The tower's rotary, exactly, against the library that defines it.

    The only case here that can be this tight, and the only piece of the tower
    that had to be *derived* -- so it is the one that gets an exact check.  A
    wrong section boundary, a missing ramp restart, a swapped row and column, or
    the interleaved pairing instead of the split-half one all fail it by more than
    float32 rounding.
    """
    if not LLAMA_LIB.is_file():
        pytest.skip(f"no llama.cpp build at {LLAMA_LIB}")
    if not opcheck_path().is_file():
        pytest.skip("build/pocketllm-opcheck is not built")
    if not rotary_tool_path().is_file():
        compile_rotary_oracle()

    rng = np.random.default_rng(20261016)
    n_pos = ROTARY_GRID * ROTARY_GRID
    x = rng.standard_normal((n_pos, ROTARY_HEADS, ROTARY_HEAD_DIM), dtype=np.float32)
    want = rotate(x, grid=ROTARY_GRID, merge=ROTARY_MERGE)
    cos_table, sin_table = _rotary_tables()

    # `positions` is the token index because the table already holds one row per
    # token; the op's `split` layout is the split-half pairing, which is the one
    # this tower uses.  (The text models pass a table indexed by position and a
    # `start_pos`; here the table *is* the per-token data.)
    request = write_request(
        "rope",
        {
            "x": x,
            "positions": np.arange(n_pos, dtype=np.int32),
            "cos": cos_table,
            "sin": sin_table,
        },
        {"start_pos": 0, "layout": "split"},
    )
    answer = run(request, "cpu")
    got = np.array(answer["values"], dtype=np.float64).reshape(want.shape)

    spread = float(np.max(np.abs(want))) or 1.0
    worst = float(np.max(np.abs(got - want.astype(np.float64))))
    assert worst <= ROTARY_RTOL * spread, (
        f"the engine's rotary differs from ggml_rope_multi by {worst} over a scale of {spread}"
    )
