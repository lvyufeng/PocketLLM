"""The architecture names beyond `qwen3`: `llama` and `qwen3vl`.

`llama` is the same dense transformer shape as `qwen3` -- RMSNorm, grouped-query
attention, SwiGLU -- and this engine reads one end to end: MiniCPM5-1B, 24
layers, GQA 16:2, an untied head, **no per-head QK norm**, a `llama-bpe`
vocabulary and a rope base of 5e6. Everything about it is either stated by the
file (its metadata prefix, which tensors are present) or one of the two
behavioural differences the graph now carries:

* the QK norm is optional -- absent here, bound when present;
* **the rotary embedding is the interleaved convention** (`x[2i]` against
  `x[2i+1]`), where Qwen's is split-half. Same angles, different pairs, and the
  wrong one is a different model rather than a rounding: measured before the
  interleaved kernel existed, this same walk was 39% of the logit range away
  from llama.cpp on `cpu`, with the argmax parting at the third token. With it,
  the walk is 2.96% / 3.17% (`cpu` / `cuda`) and every token agrees.

The prompt is `"Count: 1, 2, 3, ..., 11,"` -- written out as ids -- because its
continuation is a *sequence* rather than one obvious token, so a walk that
matches is evidence and a walk that degenerates into a loop is not. Its numbers
also span one and two digits, which is where this vocabulary's pre-tokenizer
shows: `10` and `11` are single tokens here.

`qwen3vl` is the third name, and it is the *text* half of a vision model
(Qwen3-VL-2B): Qwen3's tensors again, a **tied** head, and two hparams of its
own -- `n_deepstack_layers` and a four-element `rope.dimension_sections`, which
is what its multi-section rotary is built from. For text that layout collapses:
the sections are `[24, 20, 20, 0]` for a 128-wide head, summing to `head_dim / 2`
so every pair rotates, and a text token's three section positions are all its own
position -- so every pair takes angle `p * theta^(-2i/d)` whichever section it
belongs to. That is the engine's existing 1-D rope, and the walk below is what
says so rather than the argument: 8/8 on `cpu` and on the card, 3.06% / 3.02% of
the logit range. (With images the three positions differ, which is the half that
is not wired yet.)

Everything skips without the library, the checkpoint or llama.cpp. A skip is not
a pass.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm import native

from llama_oracle import LLAMA_LIB, compile_tool, logits as oracle_logits, tool_path

#: `llama`: openbmb's own `Q4_K_M`, quantized by its publisher rather than here.
MINICPM5 = pathlib.Path("/mnt/data1/models/minicpm5-1b-q4_k_m.gguf")
#: `qwen3vl`: the text half of Qwen's 2B vision model, their own `Q4_K_M`.
QWEN3VL = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")

#: `"Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,"` for the MiniCPM5 vocabulary. The
#: ids are written out rather than tokenized by the engine so a tokenizer
#: regression cannot read as a forward-pass one.
MINICPM5_PROMPT = [
    14686, 47, 242, 38, 33, 242, 39, 33, 242, 40, 33, 242, 41, 33, 242, 42, 33, 242, 43, 33, 242,
    44, 33, 242, 45, 33, 242, 46, 33, 242, 557, 33, 242, 682, 33,
]
#: `" 12, 13, 14, 15,"`: what both engines continue it with.
MINICPM5_CONTINUATION = [242, 627, 33, 242, 732, 33, 242, 739, 33, 242, 701, 33]

#: `"The capital of France is"` for the Qwen vocabulary, and the eight greedy
#: tokens both engines follow it with: `"Paris, and the capital of Germany is"`.
QWEN3VL_PROMPT = [785, 6722, 315, 9625, 374]
QWEN3VL_CONTINUATION = [12095, 11, 323, 279, 6722, 315, 9856, 374]

#: One entry per (name, checkpoint, prompt, continuation).
CASES = (
    pytest.param("llama", MINICPM5, MINICPM5_PROMPT, MINICPM5_CONTINUATION, id="llama"),
    pytest.param("qwen3vl", QWEN3VL, QWEN3VL_PROMPT, QWEN3VL_CONTINUATION, id="qwen3vl"),
)

#: How far the logits may move from llama.cpp's over the walk. Measured 2.96 /
#: 3.17% (`llama`, cpu / card) and 3.06 / 3.02% (`qwen3vl`), against the 6%
#: allowed -- the same order as the `q4_k_m` comparisons in
#: `test_quantized_forward.py`, which is the point: another architecture name
#: should not cost accuracy.
LOGIT_BOUND = 6e-2

#: The attention convention and cache width each backend is compared against,
#: as `test_quantized_forward.py` states them: the CPU kernel is the
#: full-softmax reduction, the card's is the block-reduced one, and the oracle is
#: asked for the f32 cache because llama.cpp's f16 one answers its own prefill and
#: decode paths differently.
ORACLE_FA = {"cpu": "off", "cuda": "on"}
ORACLE_KV = {"cpu": "f32", "cuda": "f32"}


def _needs_llama() -> str | None:
    if not LLAMA_LIB.is_file():
        return f"no llama.cpp build at {LLAMA_LIB}"
    return None


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")


@pytest.fixture(scope="session", autouse=True)
def oracle_tool() -> None:
    """Compile `src/tools/oracle.cpp` if it is not already built."""
    if _needs_llama() is None and not tool_path().is_file():
        compile_tool()


def _cuda_probe() -> str | None:
    """Why `cuda` cannot be used here, or None if it can -- the probe
    `test_quantized_forward.py` uses, on this file's checkpoint."""
    if not MINICPM5.is_file():
        return "no checkpoint"
    try:
        native.Engine.open(str(MINICPM5.with_suffix(".does-not-exist.gguf")), "cuda")
    except native.EngineUnavailable as exc:
        message = str(exc)
        return message if "cuda" in message and "not in this build" in message else None
    return "the cuda probe unexpectedly succeeded"


_CUDA_REASON = _cuda_probe()
DEVICES = [
    pytest.param(name, id=name)
    if name != "cuda" or _CUDA_REASON is None
    else pytest.param(name, marks=pytest.mark.skipif(True, reason=_CUDA_REASON), id=name)
    for name in ("cpu", "cuda")
]


@pytest.mark.parametrize("name,checkpoint,prompt,continuation", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_the_greedy_walk_matches_a_fresh_llama_cpp_prefill(
    name: str, checkpoint: pathlib.Path, prompt: list[int], continuation: list[int], device: str
) -> None:
    """The walk, context-equal, on both backends, for each extra architecture."""
    if not checkpoint.is_file():
        pytest.skip(f"no checkpoint at {checkpoint}")
    if _needs_llama() is not None:
        pytest.skip(_needs_llama() or "")

    ours: list[int] = []
    theirs: list[int] = []
    context = list(prompt)
    worst_relative = 0.0
    with native.Engine.open(str(checkpoint), device) as engine:
        for step in range(len(continuation)):
            got = engine.forward(context) if step == 0 else engine.forward([ours[-1]])
            want = oracle_logits(
                str(checkpoint), context, flash_attn=ORACLE_FA[device], kv_type=ORACLE_KV[device]
            )
            spread = max(want) - min(want)
            worst_relative = max(
                worst_relative, max(abs(a - b) for a, b in zip(got, want)) / spread
            )
            ours.append(engine.argmax(got))
            theirs.append(want.index(max(want)))
            context = context + [ours[-1]]

    assert ours == theirs, (
        f"{name} on {device}: the walk diverged at step "
        f"{next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)}: ours {ours}, "
        f"llama {theirs}"
    )
    assert worst_relative <= LOGIT_BOUND, (
        f"{name} on {device}: the logits moved {worst_relative:.2%} of the range from llama.cpp's"
    )
    # Not a degenerate walk: both engines count, and a loop of one token would
    # satisfy the comparison above while saying nothing about the weights.
    assert ours == continuation, f"{name} on {device}: the walk is not the expected sequence: {ours}"
