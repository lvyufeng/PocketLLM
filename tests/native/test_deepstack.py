"""The text graph's input width, against llama.cpp's own embeddings path.

A `qwen3vl` checkpoint's text graph is not fed tokens-only.  It carries
`n_deepstack_layers` extra feature levels, so its input row is
`n_embd * (1 + n_deepstack_layers)` wide: the first `n_embd` is the embedding the
first block reads, and each remaining level is added to a token's residual after
one of the first `n_deepstack_layers` blocks.  That is how a vision tower's
merged features reach the middle of the text model -- and it is the half of the
image path that has nothing to do with images.

So this case has no image and no tower in it.  The planes are synthetic, because
what is under test is the graph's *input width* and where the additions land, not
where the numbers came from: a case that needed the tower would fail for the
tower's reasons too.  `tests/native/test_vision_tower.py` is the other half.

**The two engines disagree about the layout and only about the layout.**  This
engine's `forward_embeddings` takes plane-major data -- `n_planes` blocks of
`n_tokens * n_embd` -- because that is what `Qwen3VlVision::encode` produces and
nothing has to be unpacked on the way in.  llama.cpp's input row interleaves the
levels per token.  The test writes the same numbers both ways: `np.stack` is one
and `np.concatenate` is the other, and a mistake in either is a comparison
against different data rather than a silent pass.

**The base plane is a real prompt and the levels are not.**  Feeding an *image's*
planes would make this case fail for the tower's reasons as well as the graph's,
and the tower has its own test; what the input has to be is *representative* --
so the base rows are the checkpoint's own embedding table for a real prompt, and
the levels are synthetic vectors at a scale that makes them matter.  That is what
the case is for: the levels are the only part of the input this graph does
something *new* with.

**Two cases, and the second is the first one's control.**  With the levels zeroed
the input is exactly what the token path would build, so the comparison is the
token path's own agreement on this checkpoint -- 2.9% of the logit spread, which
is the same figure `test_quantized_forward.py` sees on a prompt.  With the levels
in place it is 4.8%: **the injection is inside the noise the token path already
has**, which is the strongest statement the comparison can make.  Both sides run
an f32 cache (`kv_f32`, see :func:`_engine_logits`), so what is left between them
is the weight quantization rather than two cache widths as well.

The argmax is asserted only on the zero-level case: with the levels in place the
top two tokens of a five-token prompt are not guaranteed to be a margin apart, and
a flip there would be the checkpoint's near-tie rather than this graph's
arithmetic.

Run on `cpu` only.  The device dimension of this graph is `test_forward.py`'s
(cpu and the card, same checkpoint, same weights), the one kernel this path adds
to that graph is `add`, and the conformance suite covers it per-op on both
devices; what is new here is which rows are fed and where an addition lands.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

from llama_oracle import LLAMA_LIB, compile_tool, repository_root, tool_path as oracle_tool_path
from pocketllm.loader.gguf.reader import GGUFReader
from pocketllm.loader.gguf.tensor_reader import GGUFTensorDataReader

CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")

#: How far this engine may sit from llama.cpp on this checkpoint, as a fraction of
#: the logit spread -- and it is deliberately **not** `test_quantized_forward.py`'s
#: 5e-2, which is the bound for a *prompt*.  The measurements say why: the
#: comparison is across widths (llama.cpp quantizes activations to `q8_K` blocks
#: for a `q4_K` weight, where every kernel here contracts in f32), and the inputs
#: in this file are not sentences the checkpoint was tuned on.  Engine against
#: oracle: **2.9%** for the zero-level control -- which is the token path's own
#: agreement, and the figure `test_quantized_forward.py` sees -- **4.8%** with the
#: synthetic levels in place, and **9.0%** for an image span.
#:
#: What the bound has to catch is not that band but a *wrong* convention, and those
#: were measured too: **24%** for the block form of the multi-section rotary
#: (against the interleaved one this architecture uses) and **60-109%** for the
#: four combinations of the image span's mask.  An order below the smallest of
#: those is the room this needs.
QUANTIZED_RTOL = 1e-1

#: A real prompt, so that the base plane is what the graph is normally fed.  It is
#: `test_forward.py`'s, which is a sentence the checkpoint has an opinion about.
PROMPT = [785, 6722, 315, 9625, 374]

#: The scale of the synthetic levels, in the neighbourhood of a text embedding
#: (the table's rows average 0.024 in magnitude, and the tower's planes are much
#: larger than that).  Large enough that the injection moves the logits, which is
#: the point; fixed and seeded so a failure is reproducible.
LEVEL_SCALE = 3.0
SEED = 20261016


def engine_tool_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-forward-embd"


def _oracle_logits(rows: np.ndarray, tmp: pathlib.Path, positions: pathlib.Path | None = None,
                   non_causal: bool = False) -> np.ndarray:
    """llama.cpp's logits for an embeddings batch, one row per token.

    `positions`, when given, is the file `oracle.cpp --positions` takes: four
    sections consecutive within a token.  Without it every section gets the
    token's own index, which is what a text batch means.

    `non_causal` is the image span's attention: every token sees every token of
    the batch.  The oracle asks for it with `llama_set_causal_attn`, the same
    context-level flag `tools/mtmd/mtmd-helper.cpp` sets around an image chunk.
    """
    if not oracle_tool_path().is_file():
        compile_tool()
    path = tmp / "embeddings.bin"
    path.write_bytes(np.ascontiguousarray(rows, dtype=np.float32).tobytes())
    command = [str(oracle_tool_path()), str(CHECKPOINT), "--embeddings", str(path)]
    # The f32 cache `llama_oracle.py` pins the token path to, and for its reason:
    # llama.cpp's f16 cache is not self-consistent between its prefill and its
    # decode, so there is no single "llama.cpp on f16" to compare against.
    command += ["--kv-type", "f32"]
    if positions is not None:
        command += ["--positions", str(positions)]
    if non_causal:
        command += ["--non-causal"]
    result = subprocess.run(
        command, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"the oracle refused the embeddings batch: {result.stderr.strip()}")
    return np.array(json.loads(result.stdout)["logits"], dtype=np.float64)


def _engine_logits(planes: np.ndarray, tmp: pathlib.Path, positions: pathlib.Path | None = None,
                   non_causal: bool = False, kv_f32: bool = False) -> np.ndarray:
    """This engine's logits for the same numbers, in its own plane-major layout.

    `kv_f32` asks the CPU backend for an f32 cache through its own environment
    knob.  It is needed for a non-causal batch -- that arm refuses an f16 cache
    rather than widening it -- and it is the same width the oracle is pinned to,
    so a comparison that sets both is comparing like with like rather than across
    two cache widths.
    """
    path = tmp / "planes.bin"
    out = tmp / "logits.bin"
    path.write_bytes(np.ascontiguousarray(planes, dtype=np.float32).tobytes())
    command = [str(engine_tool_path()), str(CHECKPOINT), str(path), str(out)]
    if positions is not None:
        command += ["--positions", str(positions)]
    if non_causal:
        command += ["--non-causal"]
    env = dict(os.environ)
    if kv_f32:
        env["POCKETLLM_CPU_KV_F32"] = "1"
    result = subprocess.run(
        command, capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"pocketllm-forward-embd failed: {result.stderr.strip()}")
    return np.fromfile(out, dtype="<f4").astype(np.float64)


def _geometry() -> tuple[int, int, np.ndarray]:
    """``(n_embd, n_levels, the prompt's embedding rows)``."""
    metadata = GGUFReader(str(CHECKPOINT)).read().metadata
    n_embd = int(metadata["qwen3vl.embedding_length"])
    n_levels = int(metadata["qwen3vl.n_deepstack_layers"])
    with GGUFTensorDataReader(str(CHECKPOINT)) as reader:
        table = reader.read_tensor("token_embd.weight").astype(np.float32)
    return n_embd, n_levels, table[PROMPT]


def _both(n_embd: int, base: np.ndarray, levels: list[np.ndarray], tmp: pathlib.Path) -> tuple:
    """The engine's logits and llama.cpp's, from the same numbers in each layout."""
    planes = np.stack([base, *levels])  # this engine: (n_planes, n_tokens, n_embd)
    rows = np.concatenate([base, *levels], axis=1)  # llama.cpp: (n_tokens, n_embd_inp)
    # The same numbers, and the only difference between the two layouts is the
    # order of the plane axis -- stated here so that a mistake in either is a
    # comparison against different data rather than a silent pass.
    assert np.array_equal(planes.transpose(1, 0, 2).reshape(rows.shape), rows)
    # An f32 cache on both sides, so the comparison is the weight quantization and
    # not two cache widths as well.
    return _engine_logits(planes, tmp, kv_f32=True), _oracle_logits(rows, tmp)


@pytest.mark.parametrize("with_levels", [False, True], ids=["zero-levels", "synthetic-levels"])
def test_the_input_width_matches_llama_cpp(with_levels: bool, tmp_path: pathlib.Path) -> None:
    if not LLAMA_LIB.is_file():
        pytest.skip(f"no llama.cpp build at {LLAMA_LIB}")
    if not CHECKPOINT.is_file():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")
    if not engine_tool_path().is_file():
        pytest.skip("build/pocketllm-forward-embd is not built")

    n_embd, n_levels, base = _geometry()
    if n_levels <= 0:
        pytest.skip(f"{CHECKPOINT} carries no deepstack levels, so there is nothing to inject")

    if with_levels:
        rng = np.random.default_rng(SEED)
        levels = [
            rng.standard_normal((len(PROMPT), n_embd), dtype=np.float32) * LEVEL_SCALE
            for _ in range(n_levels)
        ]
    else:
        # The control: the input a token batch would build, since llama.cpp pads a
        # token's embedding with zeros and the injection then adds nothing.  Its
        # agreement is the *token path's*, which is what makes the other case's
        # number readable.
        levels = [np.zeros((len(PROMPT), n_embd), dtype=np.float32) for _ in range(n_levels)]

    got, want = _both(n_embd, base, levels, tmp_path)
    assert got.shape == want.shape, f"got {got.shape} logits, expected {want.shape}"

    spread = float(want.max() - want.min()) or 1.0
    worst = float(np.max(np.abs(got - want)))
    assert worst <= QUANTIZED_RTOL * spread, (
        f"the input width differs from llama.cpp by {worst} over a spread of {spread} "
        f"(levels {'synthetic' if with_levels else 'zeroed'})"
    )
    if not with_levels:
        # Only here: with the levels in place the top two tokens of a five-token
        # prompt are not guaranteed to be a margin apart, and a flip there would
        # be the checkpoint's near-tie rather than this graph's arithmetic.
        assert int(np.argmax(got)) == int(np.argmax(want)), "the argmax differs"
