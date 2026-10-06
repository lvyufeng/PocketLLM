"""The Qwen3 forward pass, against llama.cpp's own logits and tokens.

The graph is the third thing in this tree with no Python oracle -- after the
GGUF reader, which has one, and the tokenizer, which has this same one.  What
`src/model/qwen3.cpp` implements is a specific architecture's specific layer,
and the only authority on what that layer computes is the implementation the
checkpoint was converted for.

Four tests, in increasing strength:

* the logits of a batched prefill match within the ABI's dense tolerance;
* the same logits arrive whether the prompt is one batch or several, which is
  the property the KV cache exists to have and which nothing else here checks;
* the greedy token sequence matches for two dozen steps, which is the check
  that survives a small numerical difference and would catch an error large
  enough to change an answer;
* the two backends agree with each other, which is the weaker statement of the
  first three on `cuda` and the one that makes the CPU a usable oracle for a
  device kernel later.

The second is the one that found a real bug: a KV cache with no layer dimension
is *invisible* to the first and third tests as written, because a single batched
prefill writes each layer's slots immediately before reading them.  It is a
separate test now, and it is the reason the cache is indexed
`[layer][position][head]`.

The first three run on every backend, parametrized: a device backend that agrees
with the CPU but not with llama.cpp would mean the CPU is the one that is wrong,
and running the same comparison against the authority is what tells the two
cases apart.  `cuda` skips when the library was built without it.

Everything skips without the library, the checkpoint or llama.cpp.  A skip is
not a pass.
"""

from __future__ import annotations

import os
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

#: How far the two backends may drift from each other.  Tighter than
#: :data:`DENSE_RTOL` by three orders because this compares two implementations
#: of the *same* algorithm's association order, with no third party's kernel
#: choices in between -- the measured figure is around 8e-5 absolute on logits
#: spanning 30, so roughly 3e-6 relative for the batched prefill and 1e-5 for
#: the longer incremental case.  A bound at 2e-4 is a real check rather than one
#: that would accept a rearranged sum.
BACKEND_RTOL = 2e-4

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


def _cuda_probe() -> str | None:
    """Why `cuda` cannot be used here, or None if it can.

    Asked by opening a path that does not exist, which sounds like a trick and is
    the cheapest honest probe available.  ``pocketllm_open`` resolves the device
    *before* it maps the file -- the device is the argument the caller passed and
    the one they can act on -- so a build without CUDA answers with the backend
    it has, and a build with it answers with the missing file.  The two messages
    are different and neither costs a 1.5 GB load, which opening the real
    checkpoint just to look at the error would.

    This does lean on that ordering, and it is the ordering
    ``test_abi_smoke.py::test_an_unimplemented_backend_is_named_in_the_error``
    checks directly.  If it were reverted this would report CUDA as present and
    the tests below would fail noisily rather than skip -- which is the right way
    round for a probe to be wrong.
    """
    if not native.is_available():
        return "libpocketllm.so is not built"
    try:
        native.Engine.open(str(CHECKPOINT.with_suffix(".does-not-exist.gguf")), "cuda")
    except native.EngineUnavailable as exc:
        message = str(exc)
        # A build without the backend names it and says what it does provide.
        return message if "cuda" in message and "not in this build" in message else None
    # Opening a nonexistent file cannot succeed; reaching here means the loader
    # changed under this probe.
    return "the cuda probe unexpectedly succeeded"


#: The backends the graph runs on here.  `cpu` is always present; `cuda` is
#: dropped with the reason the probe gave, so a skip says why.
BACKENDS = ["cpu"]
_cuda_reason = _cuda_probe() if CHECKPOINT.is_file() else "no checkpoint"
if _cuda_reason is None:
    BACKENDS.append("cuda")


def _device_param(name: str) -> pytest.ParameterSet:
    """One backend as a parametrized case, marked to skip when it is absent."""
    if name in BACKENDS:
        return pytest.param(name, id=name)
    return pytest.param(name, marks=pytest.mark.skipif(True, reason=_cuda_reason or ""), id=name)


@pytest.mark.parametrize("device", [_device_param(name) for name in ("cpu", "cuda")])
@pytest.mark.parametrize(
    "tokens",
    [
        [785],
        [785, 6722],
        PROMPT,
        [9707, 11, 1879, 13, 576, 6722],
    ],
)
def test_the_logits_match_llama_cpp(device: str, tokens: list[int]) -> None:
    """A batched prefill's last-position logits, elementwise.

    Every position is compared, not the argmax alone: two implementations can
    agree on which token is largest and disagree everywhere else, and that
    difference is what turns into a different token three steps later.

    Run on every backend.  A device backend that matches the CPU but not the
    authority is a device backend built on a wrong CPU, and it is only by asking
    both the same question that the two cases can be told apart.
    """
    expected = oracle_logits(str(CHECKPOINT), tokens)
    with native.Engine.open(str(CHECKPOINT), device) as engine:
        got = engine.forward(tokens)

    assert len(got) == len(expected)
    # A relative bound on the whole vector: the logits span about 30, and an
    # absolute bound that suits the largest of them would be far too loose for
    # the small ones, which are the ones softmax actually differentiates.
    spread = max(expected) - min(expected)
    worst = _max_abs_difference(got, expected)
    assert worst <= DENSE_RTOL * spread, f"max |ours - llama| = {worst} over a spread of {spread}"

    assert got.index(max(got)) == expected.index(max(expected)), "the argmax differs"


@pytest.mark.parametrize("device", [_device_param(name) for name in ("cpu", "cuda")])
def test_the_incremental_path_agrees_with_the_batched_one(device: str) -> None:
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

    On a device this is also the test that catches a scratch buffer shared
    between concurrent work: the two paths use different call shapes for the
    same attention, so a score row that one of them sized or strided wrongly
    shows up as the batched answer and the incremental one disagreeing.
    """
    with native.Engine.open(str(CHECKPOINT), device) as engine:
        batched = engine.forward(PROMPT)

    splits = []
    for cut in (1, 2, 3, 4):
        with native.Engine.open(str(CHECKPOINT), device) as engine:
            engine.forward(PROMPT[:cut])
            splits.append(engine.forward(PROMPT[cut:]))

    for cut, incremental in zip((1, 2, 3, 4), splits):
        worst = _max_abs_difference(batched, incremental)
        assert worst == 0.0, f"splitting the prompt at {cut} changed the logits by {worst}"


@pytest.mark.parametrize("device", [_device_param(name) for name in ("cpu", "cuda")])
def test_the_greedy_sequence_matches_llama_cpp(device: str) -> None:
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

    with native.Engine.open(str(CHECKPOINT), device) as engine:
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


@pytest.mark.parametrize(
    "tokens",
    [
        [785],
        [785, 6722],
        PROMPT,
        [9707, 11, 1879, 13, 576, 6722],
    ],
)
def test_the_backends_agree_with_each_other(tokens: list[int]) -> None:
    """The CPU and the card produce the same logits for the same tokens.

    This is the check the first three already make on `cuda` -- they compare it
    to llama.cpp, and the CPU does too, so agreement between the backends is a
    consequence.  It is stated separately because it is what makes the CPU
    backend usable as *the* oracle for a kernel that has no third party to
    compare against: a quantized GEMM has no llama.cpp equivalent to diff, so
    the question it will be asked is exactly this one.

    The bound is :data:`BACKEND_RTOL` and not :data:`DENSE_RTOL`, and the
    difference is the point: llama.cpp reassociates its sums in ways neither
    backend does, so its tolerance has to be loose enough to absorb that. Two
    backends running the same association order have no such excuse, and a bound
    three orders tighter is a check that a rearranged sum would fail.
    """
    if "cuda" not in BACKENDS:
        pytest.skip(_cuda_reason or "cuda is not available")

    # The two backends now keep their caches at *different* widths on purpose --
    # f16 on the host and f32 on the card (see `preferred_kv_dtype`) -- and this
    # test is about the kernels, not about that trade.  So the host is put on
    # the card's basis for this one comparison, which is what the environment
    # switch exists for.  Left at the shipped widths the bound below would be
    # measuring a rounded cache and failing at 3e-4 of the spread, which is the
    # rounding rather than a disagreement about arithmetic.
    os.environ["POCKETLLM_CPU_KV_F32"] = "1"
    try:
        with native.Engine.open(str(CHECKPOINT), "cpu") as engine:
            host = engine.forward(tokens)
    finally:
        del os.environ["POCKETLLM_CPU_KV_F32"]
    with native.Engine.open(str(CHECKPOINT), "cuda") as engine:
        card = engine.forward(tokens)

    spread = max(host) - min(host)
    worst = _max_abs_difference(host, card)
    assert worst <= BACKEND_RTOL * spread, f"max |cpu - cuda| = {worst} over a spread of {spread}"
    assert host.index(max(host)) == card.index(max(card)), "the two backends disagree on the argmax"