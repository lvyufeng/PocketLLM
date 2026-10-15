"""The near-miss architecture: what already works, and the one piece that does not.

`llama` is the same dense transformer shape as `qwen3` -- RMSNorm, grouped-query
attention, SwiGLU, and no per-head QK norm to bind -- and the engine already
reads one: its hyperparameters (under its own prefix), its weights, its untied
head, and its vocabulary, whose pre-tokenizer is one alternation away from
Qwen's and is checked against llama.cpp in `test_tokenizer.py` (MiniCPM5-1B is
the checkpoint this file uses, and the second arm of that corpus).

What it cannot do yet is *run* one, because the rotary embedding is the other
convention. llama.cpp hands `LLM_ARCH_LLAMA` the interleaved rotation --
`(x[2i], x[2i+1])` -- and `(x[i], x[i + d/2])` for Qwen (`llama-model.cpp`'s
`get_rope_type`), and `src/kernel/kernels.cpp` implements the second only. The
difference is not a relabelling: the *partner* of an element differs, so a
`llama` file pushed through the Qwen kernel rotates every head by the wrong
angles and answers with text that reads like a bad model rather than a bug.

**What is testable now is the refusal**, and it is tested below: it has to name
the missing piece rather than say "no graph", and it is checked where this
repository already checks refusal text, through `pocketllm-run` (the C ABI
carries a message for `open` and a code for `forward`, so a Python-side assertion
would only see `-2`).

**The evidence that everything else about the graph is right was measured by
hand and is recorded here to be re-run** when the interleaved kernel lands.
The experiment is the position-0 control: at a one-token prompt every angle is
zero, so the two conventions are the same identity rotation and the engines
should agree exactly there -- and they do, to 0.93% of the logit range
(argmax equal). From position 1 on the rotation stops being the identity, and
the same comparison is 12.2% at two tokens, 18.3% at three and 21.0% at four,
with the argmax parting at three. Because 2.3% of logits separate the engines
when *nothing* is rotated, the rest of the graph -- GQA 16:2, the absent QK
norm, the untied head, RMSNorm, SwiGLU, the 5e6 rope base -- is reading this file
correctly; what is left is the rotation. Re-run it with `llama_oracle.logits`
against `Engine.forward` on `[14686]` and `[14686, 47, 242, 38]` once the
`llama` arm of `arch_refusal` is gone, and expect those numbers to invert.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from pocketllm import native

#: A 2026 checkpoint that declares `llama`: 24 layers, 1B, GQA 16:2, an untied
#: head, no QK norm, and `llama-bpe` as its pre-tokenizer. `llama-quantize`d by
#: its publisher, not here.
CHECKPOINT = pathlib.Path("/mnt/data1/models/minicpm5-1b-q4_k_m.gguf")

#: `"Count"` and `"Count:"`, then `" 1"` and `","` -- ids this checkpoint's own
#: vocabulary gives them, written out rather than tokenized by the engine so a
#: tokenizer regression cannot read as a forward-pass one.
ONE_TOKEN = [14686]


def _run_tool() -> pathlib.Path:
    return native._repository_root() / "build" / "pocketllm-run"

needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_run = pytest.mark.skipif(not _run_tool().is_file(), reason="pocketllm-run is not built")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)

pytestmark = [needs_engine, needs_run, needs_checkpoint]


def test_a_llama_checkpoint_is_refused_with_the_reason() -> None:
    """The refusal is a sentence about rope, not "no graph"."""
    result = subprocess.run(
        [str(_run_tool()), str(CHECKPOINT), "--tokens", "14686", "--steps", "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "a `llama` checkpoint must not run yet"
    output = result.stdout + result.stderr
    assert "interleaved" in output and "rotary" in output, output
    # And that it is a *reason* and not the bare "this build has no graph for it"
    # the session used to raise, which left a user with a name and no next step.
    assert "cannot run: its rotary embedding" in output, output
