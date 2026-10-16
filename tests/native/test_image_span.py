"""An image span, as the text graph will be asked to run it.

Three things about an image span are not what a text prompt is, and this is the
case that has all three at once: its tokens carry a *position per rotary section*
(the token's index, then the patch's row and column), they attend **non-causally**
-- every patch to every patch -- and they arrive as an *embeddings* batch rather
than as tokens, because their rows are the vision tower's features.

`test_rope_multi.py` isolates the first of those and `test_deepstack.py` the
third; this one is the only place the mask is exercised.

**The batch is the span, and that is the reference's shape rather than a
convenience.**  llama.cpp exposes non-causal attention as a *context* flag
(`llama_set_causal_attn`) that mtmd sets around the decode of one image chunk, so
"non-causal" means "this batch" and not "this range inside a batch".  A caller
that drives a prompt chunk, an image chunk and a prompt chunk as three calls --
which is how `mtmd` works and how the engine's `forward_embeddings` is meant to be
driven -- is expressing exactly that, and the cache carries the prompt before the
span into its window.

**What this case pins is the arm, not the architecture.**  `mtmd_decode_use_non_causal`
answers true for the Gemma projectors only, so a `qwen3vl` span is decoded
*causally* in the reference; the flag here is forced on for **both** engines, which
is what makes this a comparison of two implementations of the same arm rather than
of what one checkpoint does.  (The arm is not idle for that: it is what the vision
tower's own attention uses.)

Both caches are f32: the span's arm refuses an f16 one rather than widening it, and
the oracle is pinned to f32 for the reason `llama_oracle.py` gives, so the two
sides are compared at the same width and what is left is the checkpoint's own
weight quantization.

That residue is larger here than for a prompt -- 9.0% of the logit spread against
the token path's 2.9% -- which is why the bound is :data:`QUANTIZED_RTOL` rather
than a tighter one, and why the module says what the bound is *for*: a wrong
convention.  Measured on this same input, the four combinations of the two
engines' mask are **60%** and **109%** where they disagree and **3.7%** where they
agree causally, so the case separates "the mask is wrong" from "the numbers are
quantized" by more than an order of magnitude.  The argmax is not asserted: these
rows stand in for image features rather than being them, and the top two tokens of
such an input are not a margin apart.
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

#: A 3x3 merged patch grid, which is the walk one image is cut into at a size a
#: test can afford.  Small, but every token's row and column differ from every
#: other's, so a span that mixed two of them up would move a value rather than
#: land on an equal one.
GRID = 3


def _span_positions() -> np.ndarray:
    """`(n_tokens, 4)` int32 for the span, in llama.cpp's `(t, y, x, z)` order.

    The first component is the token's index, which is both what the convention
    is and what keeps the two engines comparable: llama.cpp keys its cache by
    position, so that component *is* the cache row a token writes.

    **The fourth component is zero, and that is measured rather than chosen.**
    `mtmd_helper_image_get_decoder_pos` -- the authority both engines are being
    compared through -- emits `(t, y, x, 0)` for every span token, and
    `llm_graph_input_pos::set_input` zeroes the fourth section for a text token
    too.  This case used to write the column into it, which is what the engine did
    as well: with both sides agreeing on a wrong convention the comparison passed,
    and it took a third opinion to see it.  The fourth section is not inert either
    -- on a checkpoint whose declared sections end before the head does, the
    remaining pairs fall through to it -- so a shared mistake there is not
    self-cancelling, it is two implementations of the same different model.
    """
    rows: list[list[int]] = []
    for y in range(GRID):
        for x in range(GRID):
            rows.append([len(rows), y, x, 0])
    return np.array(rows, dtype=np.int32)


def test_the_image_span_matches_llama_cpp(tmp_path: pathlib.Path) -> None:
    if not LLAMA_LIB.is_file():
        pytest.skip(f"no llama.cpp build at {LLAMA_LIB}")
    if not CHECKPOINT.is_file():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    if not engine_tool_path().is_file():
        pytest.skip("build/pocketllm-forward-embd is not built")

    metadata = GGUFReader(str(CHECKPOINT)).read().metadata
    n_embd = int(metadata["qwen3vl.embedding_length"])
    n_levels = int(metadata["qwen3vl.n_deepstack_layers"])

    positions = _span_positions()
    n_tokens = positions.shape[0]
    # A real embedding row per token, and a distinct one each: the span's rows are
    # the tower's features in practice, but what matters here is that no two are
    # equal -- a case whose tokens shared a row would agree with a span that had
    # them in the wrong order.
    ids = [785 + index for index in range(n_tokens)]
    with GGUFTensorDataReader(str(CHECKPOINT)) as reader:
        table = reader.read_tensor("token_embd.weight").astype(np.float32)
    base = table[ids]

    levels = [np.zeros((n_tokens, n_embd), dtype=np.float32) for _ in range(n_levels)]
    planes = np.stack([base, *levels])  # this engine: (n_planes, n_tokens, n_embd)
    rows = np.concatenate([base, *levels], axis=1)  # llama.cpp: (n_tokens, n_embd_inp)

    positions_path = tmp_path / "positions.bin"
    positions_path.write_bytes(np.ascontiguousarray(positions, dtype=np.int32).tobytes())

    got = _engine_logits(planes, tmp_path, positions=positions_path, non_causal=True, kv_f32=True)
    want = _oracle_logits(rows, tmp_path, positions=positions_path, non_causal=True)
    assert got.shape == want.shape, f"got {got.shape} logits, expected {want.shape}"

    spread = float(want.max() - want.min()) or 1.0
    worst = float(np.max(np.abs(got - want)))
    assert worst <= QUANTIZED_RTOL * spread, (
        f"the image span differs from llama.cpp by {worst} over a spread of {spread}"
    )
