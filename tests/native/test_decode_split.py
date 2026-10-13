"""The k-split decode kernel, on the checkpoint whose shapes engage it.

`gemm_quant` dispatches the split only at `m == 1`, only when `k` holds a
multiple of eight 256-weight super-blocks, and only while the pre-split grid is
under half a wave.  Reading the dispatch is not the same as covering it, and
"the token-identity gate passes" and "the split ran" are statements about
different things, so this file holds both arms of the kill switch on a
checkpoint that reaches the kernel.

Which tests reach it is worth being exact about, because it is not the
majority of them:

  * the 0.6B *f16* tests never enter `gemm_quant` -- those weights are f16 --
    so the split is not on their path at all;
  * `qwen3-0.6b-q4_k_m` **does** engage it: `attn_output` is `k = 2048` (eight
    super-blocks) with `n = 1024`, a 32-block grid, one split GEMM per layer
    per decode step.  That is what moves that checkpoint's decode logits
    between the two arms below, and it is why its greedy-sequence gate against
    llama.cpp (`test_quantized_forward.py`) is a real check of this kernel and
    not a vacuous one;
  * the four digest-gate shapes (`m = 64`, `96`, `192`, `512`) are prefill:
    they take the row kernel, which the split does not touch.

What this file adds is the checkpoint whose decode is *mostly* split.
`Qwen3-1.7B-Q4_K_M` has `k = 2048` on q/k/v/o and gate/up and `k = 6144` on
`ffn_down` (q6_K, repacked), so every decode GEMM except the excluded head
dispatches the split, and both packed branches -- q4_K and repacked q6_K -- run
on it.

**The split is fp32 reassociation, so there is no bit-equality to assert.**
What it has instead is the class of the change: the same tokens, and a logits
delta that is last-place rounding.  :data:`SPLIT_RTOL` is the bound on the
second -- two orders of magnitude above the 1.5e-5 measured over the first
eight steps, and still sixteen times below the 1.6e-2 the int8-activation
decode path measures on the same probe.  The tokens have no tolerance: sixteen
greedy steps must agree between the two arms, and the 0.6B's against
llama.cpp is what `test_quantized_forward.py` holds.

The 1.7B is deliberately *not* compared against llama.cpp here.  Its greedy
sequence parts from llama.cpp's at step 5 under both arms -- and on the
pre-split binary -- at a top-2 margin of 0.0036 over a logit spread of 30.8;
at that context llama.cpp's own batch and incremental paths disagree with each
other as well (a 10-token re-prefill picks the token this engine picks, its
one-token decode picks the other), so there is no single llama.cpp sequence to
compare against.  That is a pre-existing property of this checkpoint's card
path and not this kernel's subject.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from pocketllm import native

CHECKPOINT = pathlib.Path("/mnt/data1/models/Qwen3-1.7B-Q4_K_M.gguf")

#: `"The capital of France is"`, as ids -- the same prompt the other native
#: tests use, and ids rather than text for the same reason: a tokenizer
#: regression must not read as a kernel regression.
PROMPT = [785, 6722, 315, 9625, 374]

#: Steps of greedy decode per arm.  Sixteen is what the 0.6B gate uses, and on
#: this checkpoint the two arms agree on all of them.
STEPS = 16

#: How far the split's logits may sit from the sequential kernel's, relative to
#: the logits' rms.  Measured at 1.5e-5 over the first eight steps on both
#: 1.7B and 0.6B; the bound carries two orders of headroom over that and stays
#: well under the 1.6e-2 that reassociation-free *int8* activation quantizing
#: measures on the same probe.
SPLIT_RTOL = 1e-3


def _cuda_reason() -> str | None:
    """Why `cuda` cannot be used here, or None if it can.

    The probe `test_quantized_forward.py` uses: open a path that does not
    exist and read which of the two refusals comes back, so a build without
    the CUDA backend skips instead of failing in `Engine.open`.
    """
    if not native.is_available():
        return "libpocketllm.so is not built"
    try:
        native.Engine.open(str(CHECKPOINT.with_suffix(".does-not-exist.gguf")), "cuda")
    except native.EngineUnavailable as exc:
        message = str(exc)
        return message if "cuda" in message and "not in this build" in message else None
    return "the cuda probe unexpectedly succeeded"


needs_cuda = pytest.mark.skipif(_cuda_reason() is not None, reason=_cuda_reason() or "")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)

pytestmark = [needs_cuda, needs_checkpoint]


def _decode() -> dict:
    """One greedy decode under whatever the environment says about the split.

    The switch is read when the backend is constructed and `Engine.open`
    constructs one per call, so a single process can run both arms honestly.
    """
    step_logits = []
    tokens = []
    with native.Engine.open(str(CHECKPOINT), "cuda") as engine:
        logits = engine.forward(PROMPT)
        prefill = [float(value) for value in logits]
        token = engine.argmax(logits)
        for step in range(STEPS):
            tokens.append(token)
            if step + 1 < STEPS:
                logits = engine.forward([token])
                step_logits.append([float(value) for value in logits])
                token = engine.argmax(logits)
    return {"prefill": prefill, "steps": step_logits, "tokens": tokens}


@pytest.fixture(scope="module")
def arms() -> dict:
    """Both arms of `POCKETLLM_CUDA_DECODE_SPLIT`, decoded once.

    Module-scoped because each arm costs a 1.3 GB checkpoint load, and both
    tests below read the same pair of runs.  The environment is restored before
    the fixture returns, so no later test inherits the off state.
    """
    saved = os.environ.pop("POCKETLLM_CUDA_DECODE_SPLIT", None)
    try:
        on = _decode()
        os.environ["POCKETLLM_CUDA_DECODE_SPLIT"] = "0"
        off = _decode()
    finally:
        os.environ.pop("POCKETLLM_CUDA_DECODE_SPLIT", None)
        if saved is not None:
            os.environ["POCKETLLM_CUDA_DECODE_SPLIT"] = saved
    return {"on": on, "off": off}


def _worst_relative(reference: list[float], got: list[float]) -> float:
    """max |delta| over rms(reference) -- the scale-free form of the drift."""
    worst = max(abs(a - b) for a, b in zip(reference, got))
    rms = (sum(value * value for value in reference) / len(reference)) ** 0.5
    return worst / rms


def test_the_split_decode_generates_the_tokens_the_sequential_kernel_generates(arms: dict) -> None:
    """Sixteen greedy steps, both arms, same tokens.

    This is the sharp check and the one with no tolerance in it.  Every step
    runs the split on q/k/v/o and gate/up and on the repacked q6_K `ffn_down`,
    so a slice boundary computed wrong -- an off-by-one in `b0`, a partial
    index transposed -- changes a logit and the argmax follows.  A rounding
    difference is what the other test bounds; a token difference here is a bug.
    """
    assert arms["on"]["tokens"] == arms["off"]["tokens"], (
        f"the arms diverged at step "
        f"{next(i for i, (a, b) in enumerate(zip(arms['on']['tokens'], arms['off']['tokens'])) if a != b)}"
    )
    assert len(arms["on"]["tokens"]) == STEPS


def test_the_split_decode_moves_the_logits_by_rounding_alone(arms: dict) -> None:
    """The delta class, per decode step, and the batch the split must not touch.

    Two statements in one place because they are the same dispatcher: at
    `m == 1` the logits may move by last-place rounding, and at `m > 1` --
    the five-token prompt, which takes the same kernel branch but does not
    satisfy `m == 1` -- they must not move at all.
    """
    assert arms["on"]["prefill"] == arms["off"]["prefill"], (
        "the prompt's logits moved: the split is reaching a batch it must not"
    )
    for step, (reference, got) in enumerate(zip(arms["off"]["steps"], arms["on"]["steps"])):
        relative = _worst_relative(reference, got)
        assert relative <= SPLIT_RTOL, f"step {step}: max|d|/rms = {relative}"


def test_the_split_is_the_path_the_switch_turns_off(arms: dict) -> None:
    """The two arms must not be the same bits.

    Without this, the two tests above pass on a build where the dispatcher
    never fires -- which is exactly what the five-shape coverage sentence in
    the module docstring is about.  The split reassociates, so if it ran the
    decode logits differ; if this fails, the kernel is not being reached and
    the token gate is passing vacuously.  Check the dispatch conditions before
    relaxing it.
    """
    moved = [step for step, (a, b) in enumerate(zip(arms["on"]["steps"], arms["off"]["steps"])) if a != b]
    assert moved, "no decode step differs between the arms: the split did not run"