"""The width ladder's packed rungs, on files llama.cpp wrote.

`src/quant/blocks.h` decoded two formats -- `q4_k` and `q6_k` -- while every
document that says what this project *is* names the ladder "Q4 -> Q2 -> IQ2 ->
IQ1 -> ternary" as the thing the engine does when a checkpoint does not fit. The
two ends of that sentence did not meet: a `q2_k`, `q3_k`, `q5_k`, `iq2_xxs` or
`iq2_xs` file was refused at load with the type's name in the message, and a
format that cannot be loaded cannot be the answer to "it does not fit". This file
is the evidence for the five formats that closed the two rungs the engine now
decodes beyond its first.

**Two kinds of format, two kinds of mistake.** The k-quants are arithmetic: a
weight is a nibble multiplied by a scale, and reading the wrong nibble moves the
value. The IQ2 formats are *codebooks*: a weight is an index into a table of
small integer magnitudes with a sign applied on top, so a table read with the
wrong stride, an index split wrongly across two bytes, or a sign bit taken from
the wrong place all still produce numbers that look like weights. The first kind
of error a graph exposes by answering nonsense; the second it does not, which is
why `test_codebook_decoders.py` compares both decoders against llama.cpp's own
over real block bytes, and why that comparison is the stronger half of this
file's evidence.

**Why llama.cpp, and not the tree's own oracle.** `python/pocketllm/quant/
k_quants.py` decodes the same layouts, and `test_op_conformance.py` compares the
kernels against it block by block. That comparison is exact in its arithmetic
and blind in one direction: where the two read the same bit layout the same
*wrong* way, every case passes while both are wrong together. There is exactly
one authority in this tree that shares no code with either -- llama.cpp, which
wrote the format and quantized these files -- so the check that can fail is
"does the engine agree with the program that produced the checkpoint".

It found a real defect while this work was being done: the Python decoder read
`q3_K`'s sixteen six-bit scales as sixteen *four*-bit ones against a `-8` bias,
which decodes the first two-thirds of a block's scale range and none of the
rest, and measured 105% relative error against the f16 weights where llama.cpp's
own decode of the same blocks measures 16%. `dequantize_q3_k` was fixed in the
same change that added the C decoder, and `test_op_conformance.py`'s `q3_k`
cases are what hold the two to each other now.

**A fresh prefill at every step, not one growing context.** llama.cpp's own
one-token decode and its own batched prefill disagree on the `q3_k` file: asked
to continue once from the prompt it picks token `220` at the third step, and
asked for a fresh prefill of the same context it picks `279` -- which is what
the engine picks. Comparing against the single-call form would therefore score
the engine against a path llama.cpp itself does not agree with. So every step
here asks the oracle for the logits of the whole context, which is the question
with one answer, and the engine's one-token decode is what is being asked to
match it.

**The prompt is chosen so the answer is not a coin.** `1, 2, 3, 4, 5,` is a
sequence a 0.6B model is confident about, and confidence is what makes the walk
an argument: on the K-quant files llama.cpp's own top-2 margin never falls below
8.9% of the logit range, so a token that agreed by accident is not available.
The generic "The capital of France is" prompt is deliberately *not* used: there
the margins fall to 1% and the engines' arithmetic difference -- the engine
multiplies an exact float32 activation by the weights it decoded, llama.cpp
quantizes its activation to int8 first -- is wide enough to carry the token
away. That is a statement about two bits per weight, and this file reports it
rather than hiding it: one step of the `iq2_xxs` file has llama.cpp's own margin
at 0.28% of the range against a 2.3% difference between the engines, so nothing
decides that step and the two walks part -- ours continues "6, 7, 8, 9",
llama.cpp's goes "5, 7, 8, 9". `LOGIT_BOUND` below is what states it, and the
walk is allowed to differ from llama.cpp's only where it holds.

**What the tolerance is for.** The walk is the binder, and `LOGIT_BOUND` is a
coarse guard -- measured per format, with the reason the rows differ -- against a
decode or a wiring that is wrong *everywhere*. Pinning the decoders themselves is
`test_codebook_decoders.py`'s job, and it does it block for block.

The card is run as well as the CPU, which is where the *other* half of the
decoder wiring is: `quant_block_accumulate` in `src/kernel/cuda/backend.cu` grew
the same generic arm, and a format whose device path silently read a `q2_K`
block with the `q6_K` decoder would be fast, plausible and wrong.

Everything skips without the library, the checkpoints or llama.cpp. A skip is
not a pass.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm import native

from llama_oracle import LLAMA_LIB, compile_tool, logits as oracle_logits, tool_path

#: One file per format, all `llama-quantize` from the f16 checkpoint this tree's
#: other tests use. The K-quants need one command each:
#:
#:   llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-q2_k.gguf Q2_K
#:   llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-q3_k.gguf Q3_K
#:   llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-q5_k.gguf Q5_K
#:
#: The IQ2 files need an importance matrix first, and llama.cpp refuses the type
#: without one:
#:
#:   llama-imatrix -m qwen3-0.6b-f16.gguf -f calib.txt -o qwen3-0.6b.imatrix
#:   llama-quantize --imatrix qwen3-0.6b.imatrix qwen3-0.6b-f16.gguf \
#:       qwen3-0.6b-iq2_xxs.gguf IQ2_XXS     # and IQ2_XS likewise
#:
#: Not generated by the suite, for the same reason `test_quantized_forward.py`'s
#: checkpoint is not: quantizing is a ten-second CPU job and a few hundred
#: megabytes of artifact, and a suite that built its own would be comparing a
#: file against itself. A "plain" `Q3_K` file is not uniform either -- llama.cpp
#: promotes `ffn_down` and friends -- which is the point: one file exercises
#: three formats of the rung at once.
CHECKPOINTS = {
    "q2_k": pathlib.Path("/mnt/data1/models/qwen3-0.6b-q2_k.gguf"),
    "q3_k": pathlib.Path("/mnt/data1/models/qwen3-0.6b-q3_k.gguf"),
    "q5_k": pathlib.Path("/mnt/data1/models/qwen3-0.6b-q5_k.gguf"),
    "iq2_xxs": pathlib.Path("/mnt/data1/models/qwen3-0.6b-iq2_xxs.gguf"),
    "iq2_xs": pathlib.Path("/mnt/data1/models/qwen3-0.6b-iq2_xs.gguf"),
}

#: `"1, 2, 3, 4, 5,"`, as ids -- written out rather than tokenized by the engine
#: so a tokenizer regression cannot read as a forward-pass one, the same rule
#: `test_quantized_forward.py` states for its prompt.
PROMPT = [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11]

#: The expected continuation, `", 6, 7, 8, 9, 10,"`. Asserted as well as the
#: agreement with the oracle: two engines that both answered "the" twelve times
#: would satisfy the comparison and say nothing about the weights.
CONTINUATION = [220, 21, 11, 220, 22, 11, 220, 23, 11, 220, 24, 11]

STEPS = len(CONTINUATION)

#: How far the logits may move from llama.cpp's, per format, as a fraction of the
#: logit range.  Measured worst over the twelve steps, CPU / card:
#:
#:   q2_k    5.26 / 6.04      q3_k    5.17 / 3.41      q5_k    4.40 / 3.06
#:   iq2_xs  6.30 / 4.60      iq2_xxs 26.0 / 24.4
#:
#: The four two-to-five-bit rows sit together around five percent, and the gap
#: they measure is the one the module docstring names: the engine multiplies the
#: weights it decodes by a float32 activation, and llama.cpp quantizes its
#: activation to int8 first.  `iq2_xxs` is four times wider, and the driver is
#: the same one at its extreme -- 2.06 bits per weight, so a 0.4% activation
#: rounding maps onto a much larger fraction of a much coarser product.
#:
#: This is a guard against a decode or a wiring that is wrong *everywhere*, not
#: the comparison itself: `test_codebook_decoders.py` is what pins these decoders
#: against llama.cpp's own, bit for bit, on real blocks.
LOGIT_BOUND = {
    "q2_k": 1.0e-1,
    "q3_k": 1.0e-1,
    "q5_k": 1.0e-1,
    "iq2_xs": 1.0e-1,
    "iq2_xxs": 4.0e-1,
}

#: The attention convention each backend is compared against, as
#: `test_quantized_forward.py` states it: the CPU kernel is the full-softmax
#: reduction and the card's is the block-reduced one, and each is compared to
#: the llama.cpp mode it lands on.
ORACLE_FA = {"cpu": "off", "cuda": "on"}

#: The oracle's cache width.  f32 on both, because llama.cpp's f16 cache answers
#: its own prefill and one-token decode paths differently -- there is no single
#: f16 sequence to match.
ORACLE_KV = {"cpu": "f32", "cuda": "f32"}


def _needs_llama() -> str | None:
    if not LLAMA_LIB.is_file():
        return f"no llama.cpp build at {LLAMA_LIB}"
    return None


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_llama = pytest.mark.skipif(_needs_llama() is not None, reason=_needs_llama() or "")


@pytest.fixture(scope="session", autouse=True)
def oracle_tool() -> None:
    """Compile `src/tools/oracle.cpp` if it is not already built."""
    if not tool_path().is_file():
        compile_tool()


def _cuda_probe() -> str | None:
    """Why `cuda` cannot be used here, or None if it can.

    The same probe `test_quantized_forward.py` uses: open a path that does not
    exist and read which refusal comes back -- device first, file second, as
    `pocketllm_open` documents. It costs no checkpoint load.
    """
    if not native.is_available():
        return "libpocketllm.so is not built"
    probe = next(iter(CHECKPOINTS.values())).with_suffix(".does-not-exist.gguf")
    try:
        native.Engine.open(str(probe), "cuda")
    except native.EngineUnavailable as exc:
        message = str(exc)
        return message if "cuda" in message and "not in this build" in message else None
    return "the cuda probe unexpectedly succeeded"


_CUDA_REASON = _cuda_probe()

#: Both backends, with `cuda` skipped by name when the library has no card half
#: rather than dropped, so the set of what is *not* covered stays legible.
DEVICES = [
    pytest.param(name, id=name)
    if name != "cuda" or _CUDA_REASON is None
    else pytest.param(name, marks=pytest.mark.skipif(True, reason=_CUDA_REASON), id=name)
    for name in ("cpu", "cuda")
]


@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
@pytest.mark.parametrize("device", DEVICES)
def test_the_greedy_walk_matches_a_fresh_llama_cpp_prefill(name: str, device: str) -> None:
    """Twelve greedy tokens on a real file of the format, context-equal.

    Every step runs all twenty-eight layers of packed weights plus the packed
    embedding gather, and the comparison at each step is a discrete choice from
    the logits of the *same* context on both sides. A weight read from the wrong
    bit field -- the failure mode a decoder has -- changes it; so does a scale
    read with the wrong bias, which is the specific defect the `q3_k` decoder in
    this tree had.
    """
    checkpoint = CHECKPOINTS[name]
    if not checkpoint.is_file():
        pytest.skip(f"no {name} checkpoint at {checkpoint} (llama-quantize the f16 one)")
    if _needs_llama() is not None:
        pytest.skip(_needs_llama() or "")

    ours: list[int] = []
    theirs: list[int] = []
    undecided: list[int] = []
    context = list(PROMPT)
    worst_relative = 0.0
    with native.Engine.open(str(checkpoint), device) as engine:
        for step in range(STEPS):
            got = engine.forward(context) if step == 0 else engine.forward([ours[-1]])
            want = oracle_logits(
                str(checkpoint),
                context,
                flash_attn=ORACLE_FA[device],
                kv_type=ORACLE_KV[device],
            )
            spread = max(want) - min(want)
            worst_relative = max(
                worst_relative, max(abs(a - b) for a, b in zip(got, want)) / spread
            )
            ours.append(engine.argmax(got))
            theirs.append(want.index(max(want)))
            # A step whose argmax differs has to be one llama.cpp itself leaves
            # open: our token must sit inside the same band of llama.cpp's own
            # logits that the bound above describes.  That is the honest form of
            # "the engines' arithmetic differs by more than the gap did" -- and a
            # token chosen anywhere else in llama.cpp's ranking fails here, which
            # is what a wrong decode or a wrong block size would do.
            if ours[-1] != theirs[-1] and (
                max(want) - want[ours[-1]] <= LOGIT_BOUND[name] * spread
            ):
                undecided.append(step)
            context = context + [ours[-1]]

    disagreed = [step for step, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    assert sorted(undecided) == disagreed, (
        f"{name}: the walk diverged at step "
        f"{next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)} with the two tokens "
        f"more than {LOGIT_BOUND[name]:.0%} of the range apart in llama.cpp's own logits: "
        f"ours {ours}, llama {theirs}"
    )
    assert worst_relative <= LOGIT_BOUND[name], (
        f"{name}: the logits moved {worst_relative:.2%} of the range from llama.cpp's"
    )
    # The counting continuation, asserted *where llama.cpp's own walk is it*.
    # This is the assertion that a degenerate walk cannot pass -- twelve copies of
    # one token would match the other two checks trivially -- and it is bounded by
    # the oracle's own behaviour on purpose. `iq2_xxs` is the case that makes the
    # bound necessary: at 2.06 bits per weight llama.cpp itself leaves the pattern
    # at step 1 (it goes 5, 7, 8 where the sequence asks for 5, 6, 7) and never
    # returns to it, and a walk that has already parted follows its own context.
    oracle_on_pattern = [step for step in range(STEPS) if theirs[step] == CONTINUATION[step]]
    assert oracle_on_pattern, f"{name}: llama.cpp's walk is never the counting continuation"
    for step in oracle_on_pattern:
        assert ours[step] == CONTINUATION[step], (
            f"{name}: step {step} is the continuation for llama.cpp but {ours[step]} for us: {ours}"
        )
