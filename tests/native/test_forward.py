"""The Qwen3 forward pass, against llama.cpp's own logits and tokens.

The graph is the third thing in this tree with no Python oracle -- after the
GGUF reader, which has one, and the tokenizer, which has this same one.  What
`src/model/qwen3.cpp` implements is a specific architecture's specific layer,
and the only authority on what that layer computes is the implementation the
checkpoint was converted for.

Three tests, in increasing strength:

* the logits of a batched prefill match within the ABI's dense tolerance;
* the same logits arrive whether the prompt is one batch or several, which is
  the property the KV cache exists to have and which nothing else here checks;
* the greedy token sequence matches for two dozen steps, which is the check
  that survives a small numerical difference and would catch an error large
  enough to change an answer.

The second is the one that found a real bug: a KV cache with no layer dimension
is *invisible* to the first and third tests as written, because a single batched
prefill writes each layer's slots immediately before reading them.  It is a
separate test now, and it is the reason the cache is indexed
`[layer][position][head]`.

Everything skips without the library, the checkpoint or llama.cpp.  A skip is
not a pass.
"""

from __future__ import annotations

import pathlib

import pytest

from pocketllm import native

from llama_oracle import LLAMA_LIB, compile_tool, generated, logits as oracle_logits, tool_path

CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3-0.6b-f16.gguf")

#: The ABI's tolerance for a dense product, from the plan.  The measured
#: agreement is one to two orders tighter than this -- 0.018 absolute on logits
#: that span 30, so about 6e-4 relative -- and the headroom is there because
#: llama.cpp's own kernels reassociate their sums and the exact figure moves
#: with which of its SIMD paths this host takes.
DENSE_RTOL = 2e-3

#: `"The capital of France is"`, tokenized with `parse_special` on.  Written as
#: ids rather than as the text so that a tokenizer regression cannot be
#: mistaken for a forward-pass one; the tokenizer has its own tests.
PROMPT = [785, 6722, 315, 9625, 374]


def _needs_llama() -> str | None:
    if not LLAMA_LIB.is_file():
        return f"no llama.cpp build at {LLAMA_LIB}"
    return None


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)
needs_llama = pytest.mark.skipif(_needs_llama() is not None, reason=_needs_llama() or "")

pytestmark = [needs_engine, needs_checkpoint, needs_llama]


@pytest.fixture(scope="session", autouse=True)
def oracle_tool() -> None:
    """Compile `src/tools/oracle.cpp` if it is not already built."""
    if not tool_path().is_file():
        compile_tool()


def _max_abs_difference(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b), f"logit vectors differ in length: {len(a)} vs {len(b)}"
    return max(abs(x - y) for x, y in zip(a, b))


@pytest.mark.parametrize(
    "tokens",
    [
        [785],
        [785, 6722],
        PROMPT,
        [9707, 11, 1879, 13, 576, 6722],
    ],
)
def test_the_logits_match_llama_cpp(tokens: list[int]) -> None:
    """A batched prefill's last-position logits, elementwise.

    Every position is compared, not the argmax alone: two implementations can
    agree on which token is largest and disagree everywhere else, and that
    difference is what turns into a different token three steps later.
    """
    expected = oracle_logits(str(CHECKPOINT), tokens)
    with native.Engine.open(str(CHECKPOINT)) as engine:
        got = engine.forward(tokens)

    assert len(got) == len(expected)
    # A relative bound on the whole vector: the logits span about 30, and an
    # absolute bound that suits the largest of them would be far too loose for
    # the small ones, which are the ones softmax actually differentiates.
    spread = max(expected) - min(expected)
    worst = _max_abs_difference(got, expected)
    assert worst <= DENSE_RTOL * spread, f"max |ours - llama| = {worst} over a spread of {spread}"

    assert got.index(max(got)) == expected.index(max(expected)), "the argmax differs"


def test_the_incremental_path_agrees_with_the_batched_one() -> None:
    """The same prompt, fed as one batch and as several calls, is the same.

    This is the KV cache's whole contract, and it is not implied by the test
    above: that one exercises a single prefill, and a cache that is not indexed
    per layer passes it.  Each layer's slots happen to hold the right values
    during a prefill because the layer writes them immediately before reading
    them -- so the corruption only appears on a *second* call, when the slots
    hold whatever the last layer left there.

    Comparing against llama.cpp would also catch it, but only for a decode
    longer than one step; comparing the two paths against *each other* is
    sharper, because it isolates the cache from every other source of
    difference between the two implementations.
    """
    with native.Engine.open(str(CHECKPOINT)) as engine:
        batched = engine.forward(PROMPT)

    splits = []
    for cut in (1, 2, 3, 4):
        with native.Engine.open(str(CHECKPOINT)) as engine:
            engine.forward(PROMPT[:cut])
            splits.append(engine.forward(PROMPT[cut:]))

    for cut, incremental in zip((1, 2, 3, 4), splits):
        worst = _max_abs_difference(batched, incremental)
        assert worst == 0.0, f"splitting the prompt at {cut} changed the logits by {worst}"


def test_the_greedy_sequence_matches_llama_cpp() -> None:
    """Twenty-four greedy tokens are the same ones llama.cpp produces.

    This is a weaker check per step than the logits comparison and a stronger
    one overall: it is what actually has to hold for the engine to be usable,
    and it is not implied by a logits test that passes at a tolerance, because
    a near-tie can resolve the other way.  It also runs the decode loop -- one
    token at a time through the incremental path -- where a broken cache shows
    up as a sequence that starts right and degenerates.

    `llama-cli` is deliberately not the oracle here.  It is a chat client in
    this build: given a prompt it wraps it in the checkpoint's chat template and
    answers a different question than the engine was asked.  The low-level API
    is driven through `src/tools/oracle.cpp` instead.
    """
    steps = 24
    expected = generated(str(CHECKPOINT), PROMPT, steps)

    with native.Engine.open(str(CHECKPOINT)) as engine:
        ours = []
        token = engine.argmax(engine.forward(PROMPT))
        for step in range(steps):
            ours.append(token)
            # One token per call, which is the path a real decode takes.  The
            # position comes from the session, so the second argument is only
            # here because `Engine.forward` sends the whole prompt each time.
            if step + 1 < steps:
                token = engine.argmax(engine.forward([token]))

    assert ours == expected, f"diverged at step {next(i for i, (a, b) in enumerate(zip(ours, expected)) if a != b)}"