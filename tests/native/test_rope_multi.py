"""The text graph's multi-section rotary, against llama.cpp's own.

A `qwen3vl` text graph is a multi-section rotary model: the rotating pairs are
partitioned into sections -- `rope.dimension_sections`, `[24, 20, 20, 0]` here --
and a pair takes its angle from *its section's* position.  A text token passes its
own index in all four, which is why the ordinary one-dimensional rotation this
engine has always had is the same thing for text, and why the 8/8 walk says
nothing about the sections.  An image patch passes a row and a column, and that is
the entire reason the machinery exists.

So this is a case the text half's own tests cannot be: it feeds a span whose
sections *disagree* and compares the logits against llama.cpp's, which is the same
input through `ggml_rope_multi`.  Nothing else about the input is new -- the
deepstack levels are zeroed, because they have their own test and would only add
their noise here -- so what is left is the rotary.

The positions are the ones a real image span carries: every token's *first*
component is its own index in the sequence, and a patch's second and third are its
row and its column.  That first component is not decoration -- llama.cpp's cache
is keyed by position, so a token's first component *is* the row of the cache it
writes, and a span whose patches shared one would have them overwrite each other.
Keeping it equal to the index is therefore both what the convention is and what
makes the two engines comparable at all: this one keys its cache by the token's
place in the batch, and the two agree exactly when the position says the same
thing.  Both engines are handed the same integers through the same file, so what
is compared is the rotary and not two conventions about slots.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from llama_oracle import LLAMA_LIB
from pocketllm.loader.gguf.reader import GGUFReader
from pocketllm.loader.gguf.tensor_reader import GGUFTensorDataReader
from test_deepstack import (
    CHECKPOINT,
    QUANTIZED_RTOL,
    _engine_logits,
    _oracle_logits,
    engine_tool_path,
)

#: Two text tokens, a 2x2 patch block, and a text token past them -- seven in all.
#: The last token is text, so the logits are read from a position whose span is
#: the whole sequence, and it is the only token whose row is a real embedding of
#: the token it stands for (the fake patch rows are embeddings of a *token*, not
#: image features, which the tower's own test covers).
TEXT_BEFORE = [785, 6722]
PATCH_GRID = [(0, 0), (0, 1), (1, 0), (1, 1)]
TEXT_AFTER = [374]

#: A patch row that is not a real embedding of anything: the same id repeated, so
#: that a case which mixed the rows up would still be caught by the positions.
PATCH_ID = 785


def _positions() -> np.ndarray:
    """`(n_tokens, 4)` int32, the four sections consecutive within a token.

    Section order is llama.cpp's `(t, y, x, z)`.  A text token repeats its index in
    all four, which is the same rotation as the one-dimensional table the token
    path uses; a patch carries its index as the first component, its row and its
    column next, and repeats the column in the fourth slot -- unused because this
    file gives that section no pairs, which is exactly the sort of thing a test
    should not rely on silently: were it to gain pairs, this case would start
    measuring them.
    """
    rows: list[list[int]] = []
    for index in range(len(TEXT_BEFORE)):
        rows.append([index, index, index, index])
    for y, x in PATCH_GRID:
        at = len(rows)
        rows.append([at, y, x, x])
    for _ in TEXT_AFTER:
        at = len(rows)
        rows.append([at, at, at, at])
    return np.array(rows, dtype=np.int32)


def test_the_multi_section_rotary_matches_llama_cpp(tmp_path: pathlib.Path) -> None:
    if not LLAMA_LIB.is_file():
        pytest.skip(f"no llama.cpp build at {LLAMA_LIB}")
    if not CHECKPOINT.is_file():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    if not engine_tool_path().is_file():
        pytest.skip("build/pocketllm-forward-embd is not built")

    metadata = GGUFReader(str(CHECKPOINT)).read().metadata
    n_embd = int(metadata["qwen3vl.embedding_length"])
    n_levels = int(metadata["qwen3vl.n_deepstack_layers"])

    positions = _positions()
    n_tokens = positions.shape[0]
    ids = TEXT_BEFORE + [PATCH_ID] * len(PATCH_GRID) + TEXT_AFTER
    with GGUFTensorDataReader(str(CHECKPOINT)) as reader:
        table = reader.read_tensor("token_embd.weight").astype(np.float32)
    base = table[ids]

    levels = [np.zeros((n_tokens, n_embd), dtype=np.float32) for _ in range(n_levels)]
    planes = np.stack([base, *levels])  # this engine: (n_planes, n_tokens, n_embd)
    rows = np.concatenate([base, *levels], axis=1)  # llama.cpp: (n_tokens, n_embd_inp)

    positions_path = tmp_path / "positions.bin"
    positions_path.write_bytes(np.ascontiguousarray(positions, dtype=np.int32).tobytes())

    got = _engine_logits(planes, tmp_path, positions=positions_path)
    want = _oracle_logits(rows, tmp_path, positions=positions_path)
    assert got.shape == want.shape, f"got {got.shape} logits, expected {want.shape}"

    spread = float(want.max() - want.min()) or 1.0
    worst = float(np.max(np.abs(got - want)))
    assert worst <= QUANTIZED_RTOL * spread, (
        f"the multi-section rotary differs from llama.cpp by {worst} over a spread of {spread}"
    )
