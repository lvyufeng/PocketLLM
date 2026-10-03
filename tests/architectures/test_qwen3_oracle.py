"""Is the Qwen3 graph the model?  Two implementations, one tiny checkpoint.

``tests/architectures/test_qwen3.py`` checks the graph against a numpy
transcription of ``src/model/qwen3.cpp``, and that comparison is real: it is what
catches a wrong QK-norm axis or a rope applied after the cache write.  But the
transcription was itself written from the same reading of the C source, so where
that reading is wrong the two agree -- and a graph that is a faithful
transcription of the wrong thing still passes.  This file is the comparison that
does not have that hole, because its second implementation is somebody else's.

Two or three implementations of the *same* weights, then:

* the **C engine** in ``build/libpocketllm.so``, through the ``ctypes`` bridge;
* the **Python graph** on the reference backend, driven by the decode loop;
* **llama.cpp**, which is the authority the C engine is already checked against
  in ``tests/native/test_forward.py``.

The checkpoint they all read is written here rather than shipped.  A real
1.5 GB Qwen3-0.6B cannot be committed, does not exist in CI, and cannot be built
with hand-checked geometry; a two-layer model with an eight-wide head can.  Both
of this tree's readers parse it with the code they use on a real checkpoint, so
the loaders are exercised rather than stubbed -- and a bug in one of them would
show up in the logits here rather than only on the machine that has the big file.

**The Python half always runs.**  The reference backend is numpy, so the graph's
logits are computed on any host; the C engine and llama.cpp each sit behind their
own skip.  The pairs say different things:

* *C vs llama.cpp* re-establishes the authority claim on this file, so that a
  disagreement below has a known-correct side to land on;
* *Python vs C* is the claim this file exists for -- the graph against the engine
  it transcribes;
* *Python vs llama.cpp* is the same claim made without trusting the C engine, and
  it is the only one of the three that can fail while the other two pass.

The prompt is two ASCII bytes, chosen so that **the tokenizers are not part of
what is being measured**: the writer's vocabulary pairs each byte with one from
the opposite half of the alphabet, so no ASCII two-letter prompt has a merge to
take and every implementation produces the two ids the bytes name.  Tokenizing
``"ab"`` gives ``[97, 98]``, and the test below pins that against the engine's
own tokenizer rather than assuming it.  (That was a real failure, not a
hypothetical: the first version of the vocabulary paired neighbouring bytes, and
``"ab"`` tokenized to the single id 353 while the comparison silently measured
one token instead of two.)

A skip is not a pass.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import pytest

from pocketllm import native
from pocketllm.architectures.qwen3 import Qwen3Config, build as build_qwen3
from pocketllm.backends.reference import BACKEND
from pocketllm.engine import Executor
from pocketllm.engine.decode import Decoder
from pocketllm.kernels.device import Device
from pocketllm.loader.gguf.writer import write_gguf, qwen3_metadata, qwen2_tokenizer_metadata

#: The oracle lives in ``tests/native/``, where it is shared with
#: ``test_forward.py``, ``test_cli_run.py`` and the tokenizer's tests -- including
#: the *build* of ``src/tools/oracle.cpp``.  Importing it rather than calling the
#: tool directly means this file is runnable on its own: the tool is compiled if
#: it is missing instead of the test skipping for a reason the suite already
#: knows how to fix.  ``tests/`` is not a package, so the directory goes on the
#: path rather than being imported relatively.
_NATIVE = pathlib.Path(__file__).resolve().parents[1] / "native"
if str(_NATIVE) not in sys.path:
    sys.path.insert(0, str(_NATIVE))

from llama_oracle import LLAMA_LIB, compile_tool, logits as oracle_logits, tool_path  # noqa: E402

#: The conformance band from ``tests/backends/conftest.py``: what the ABI allows
#: a *dense* implementation to differ by.  Right for the two implementations that
#: share an association order -- the C engine and the Python graph agree to
#: 3.0e-7 on these logits, three orders inside it.
DENSE_RTOL = 2e-3
DENSE_ATOL = 1e-4

#: The band for a comparison *against llama.cpp*, which is a different thing and
#: needs more room.  ``test_forward.py`` can use the flat pair above because its
#: logits span 30; these span 1.4, and absolute error does not shrink with the
#: logits -- the measured disagreement is 1.3e-4 absolute, or 2.2% of one, which
#: is llama.cpp's own kernel reassociation and not anybody's bug.  This file's
#: geometry is deliberately small, so the band has to be too.
ORACLE_RTOL = 2e-2
ORACLE_ATOL = 5e-4

#: A tiny Qwen3 the two trees can both load: heads != kv_heads so the grouping is
#: exercised, ``head_dim`` larger than 2 so the rotary split has interior
#: elements, and ``rows=1`` because that is the width a decode step runs at and
#: the only width at which the graph and the C engine compute the same thing (the
#: graph projects all ``rows`` rows, the engine only the last).
TINY = Qwen3Config(
    hidden=32, layers=2, heads=4, kv_heads=2, head_dim=8, ff=48, vocab=512, context=16, rows=1
)

#: The prompt, as ids and as the text that must produce them.  ``"ab"`` cannot
#: reach a merge under the writer's vocabulary, so the ids are the bytes.
PROMPT_IDS = [97, 98]
PROMPT_TEXT = "ab"

def _needs_llama() -> str | None:
    """Mirrors ``tests/native/test_forward.py``: the *library* is what is missing.

    A missing tool is not a skip -- :func:`oracle_tool` compiles it -- so an
    absent llama.cpp checkout is the only reason not to run.
    """
    if not LLAMA_LIB.is_file():
        return f"no llama.cpp build at {LLAMA_LIB}"
    return None


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_llama = pytest.mark.skipif(_needs_llama() is not None, reason=_needs_llama() or "")


@pytest.fixture(scope="module", autouse=True)
def oracle_tool() -> None:
    """Compile ``src/tools/oracle.cpp`` if it is not already built."""
    if not tool_path().is_file():
        compile_tool()


# -- the checkpoint -----------------------------------------------------------


def _weights(cfg: Qwen3Config, seed: int = 11) -> dict[str, np.ndarray]:
    """Deterministic weights, one draw per tensor, in a real checkpoint's shapes.

    The scales are chosen so the logits land in an ordinary range rather than
    saturating or vanishing -- an all-zero or all-equal logit vector would make
    every comparison below pass for the wrong reason, and ``_compare`` refuses it.
    """
    rng = np.random.default_rng(seed)

    def matrix(rows: int, cols: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    def vector(n: int) -> np.ndarray:
        return (1.0 + rng.standard_normal((n,)) * 0.1).astype(np.float32)

    out = {
        "token_embd.weight": matrix(cfg.vocab, cfg.hidden, 0.3),
        "output.weight": matrix(cfg.vocab, cfg.hidden),
        "output_norm.weight": vector(cfg.hidden),
    }
    for layer in range(cfg.layers):
        base = f"blk.{layer}."
        out.update(
            {
                base + "attn_norm.weight": vector(cfg.hidden),
                base + "ffn_norm.weight": vector(cfg.hidden),
                base + "attn_q_norm.weight": vector(cfg.head_dim),
                base + "attn_k_norm.weight": vector(cfg.head_dim),
                base + "attn_q.weight": matrix(cfg.q_width, cfg.hidden),
                base + "attn_k.weight": matrix(cfg.kv_width, cfg.hidden),
                base + "attn_v.weight": matrix(cfg.kv_width, cfg.hidden),
                base + "attn_output.weight": matrix(cfg.hidden, cfg.q_width),
                base + "ffn_gate.weight": matrix(cfg.ff, cfg.hidden),
                base + "ffn_up.weight": matrix(cfg.ff, cfg.hidden),
                base + "ffn_down.weight": matrix(cfg.hidden, cfg.ff),
            }
        )
    return out


def _rope_tables(cfg: Qwen3Config) -> tuple[np.ndarray, np.ndarray]:
    """The tables the graph takes as inputs, computed the way the C engine does.

    ``build_rope_table`` in ``src/model/qwen3.cpp`` uses split-half pairing at
    angle ``position * theta^(-2i/d)``, and the graph declares ``layout="split"``
    for the same reason.  Computed here rather than read from the engine because
    the *table* is an input on the Python side and a weight on the C side, and a
    disagreement between the two would otherwise hide inside the comparison.
    """
    half = cfg.head_dim // 2
    angle = np.arange(cfg.context)[:, None] * cfg.rope_theta ** (
        -2.0 * np.arange(half) / cfg.head_dim
    )
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _write(cfg: Qwen3Config, weights: dict[str, np.ndarray], path: pathlib.Path) -> pathlib.Path:
    metadata = qwen3_metadata(
        hidden=cfg.hidden,
        layers=cfg.layers,
        heads=cfg.heads,
        kv_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        ff=cfg.ff,
        vocab=cfg.vocab,
        context=cfg.context,
        rms_eps=cfg.rms_eps,
        rope_theta=cfg.rope_theta,
    )
    metadata.update(qwen2_tokenizer_metadata(cfg.vocab))
    return pathlib.Path(write_gguf(str(path), weights, metadata))


# -- the implementations ------------------------------------------------------


def _c_logits(checkpoint: pathlib.Path, tokens: list[int]) -> np.ndarray:
    with native.Engine.open(str(checkpoint), "cpu") as engine:
        return np.asarray(engine.forward(tokens, vocab=TINY.vocab), dtype=np.float32)


def _python_logits(
    weights: dict[str, np.ndarray], tokens: list[int], *, rows: int = 1
) -> np.ndarray:
    """Run the graph and return the last row's logits.

    ``rows=1`` goes through the decode loop rather than a bare ``Executor.run``,
    on purpose: this is the production path -- one token per step with a live
    cache -- and running it here means the comparison also carries the claim that
    the loop's cache handling is what makes a sequence agree with the C engine's
    batched forward.  A cache written one step late or in the wrong slot would
    show up as a logits difference, which is the failure this file is best placed
    to catch.

    ``rows=2`` cannot use the loop (``Qwen3Config.rows`` is baked into the
    graph's reshapes) and is run directly with an empty cache, which is the
    batched-prefill shape the C engine takes.
    """
    spec = build_qwen3(TINY if rows == 1 else dataclasses.replace(TINY, rows=rows))
    cos, sin = _rope_tables(TINY)

    if rows == 1:
        decoder = Decoder(spec, device=Device("cpu"))
        try:
            session = decoder.handle
            decoder.bind(
                {name: session.tensor(np.ascontiguousarray(weights[name])) for name in spec.weight_values}
            )
            decoder.bind(
                {"rope_cos": session.tensor(cos), "rope_sin": session.tensor(sin)}
            )
            generation = decoder.start()
            logits = None
            for token in tokens:
                logits = generation.step(token)
            return np.asarray(session.array(logits)).copy().reshape(-1)
        finally:
            decoder.close()

    session = BACKEND.open(Device("cpu"))
    try:
        binding: dict[str, np.ndarray] = dict(weights)
        binding.update(
            tokens=np.array(tokens, np.int32),
            positions=np.arange(len(tokens), dtype=np.int32),
            rope_cos=cos,
            rope_sin=sin,
        )
        for layer in range(TINY.layers):
            for kind in ("k", "v"):
                binding[f"blk.{layer}.{kind}_cache"] = np.zeros(
                    (TINY.context, TINY.kv_heads, TINY.head_dim), np.float32
                )
        inputs = {
            name: session.tensor(np.ascontiguousarray(binding[name]))
            for name in spec.graph.input_names
        }
        out = Executor(session).run(spec.graph, inputs)
        return np.asarray(session.array(out["logits"])).copy()[-1].reshape(-1)
    finally:
        session.close()


def _llama_logits(checkpoint: pathlib.Path, tokens: list[int]) -> np.ndarray:
    return np.asarray(oracle_logits(str(checkpoint), tokens), dtype=np.float32)


# -- the comparisons ----------------------------------------------------------


def _compare(
    got: np.ndarray, want: np.ndarray, *, label: str, rtol: float, atol: float
) -> None:
    """Compare two logit vectors, refusing a comparison that could pass vacuously.

    ``want`` all-equal or all-zero would satisfy any tolerance, so the guard is
    asserted before the comparison rather than after: a test whose second
    implementation silently produced nothing is a test that passes and proves
    nothing.
    """
    assert got.shape == want.shape == (TINY.vocab,), f"{label}: {got.shape} vs {want.shape}"
    assert np.all(np.isfinite(got)), f"{label}: the first implementation is not finite"
    assert np.all(np.isfinite(want)), f"{label}: the second implementation is not finite"
    assert float(np.ptp(want)) > 0.1, (
        f"{label}: the reference logits span only {float(np.ptp(want)):g}; a flat vector "
        "makes this comparison vacuous"
    )
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"{label}: logits differ")


@pytest.fixture(scope="module")
def case(tmp_path_factory) -> dict:
    """The checkpoint, its weights, and the Python graph's logits for the prompt.

    Session-scoped because writing the file and running the graph is the
    expensive part and neither depends on which comparison is being made.  The C
    engine's logits are *not* computed here: doing so would make every test in
    the module -- including the two that need no engine -- error on a host
    without the library rather than skip.
    """
    weights = _weights(TINY)
    path = _write(TINY, weights, tmp_path_factory.mktemp("oracle") / "tiny-qwen3.gguf")
    return {"path": path, "weights": weights, "python": _python_logits(weights, PROMPT_IDS)}


def test_the_graphs_logits_are_not_flat(case) -> None:
    """The anti-vacuity guard, on its own so a failure names itself.

    Every comparison below needs the logits to carry information, and a graph
    that read a zeroed embedding -- or a weight binding that silently missed --
    would produce a constant vector that satisfies any tolerance.
    """
    logits = case["python"]
    assert np.all(np.isfinite(logits)), "the graph's logits are not finite"
    assert float(np.ptp(logits)) > 0.1, f"the graph's logits span {float(np.ptp(logits)):g}"


def test_the_graph_decodes_the_prompt_one_token_at_a_time(case) -> None:
    """``rows=1`` steps and one ``rows=2`` call, on the same weights.

    The decode loop feeds the prompt one token per step, and the C engine feeds
    it as one batch; the graph at ``rows=2`` is the shape that makes the two
    comparable.  Requiring the last row to match is what makes "one step per
    token" a legitimate way to prefill rather than an approximation that happens
    to be close -- and this half needs no engine, so it runs everywhere.
    """
    batched = _python_logits(case["weights"], PROMPT_IDS, rows=2)
    _compare(batched, case["python"], label="rows=2 vs rows=1", rtol=DENSE_RTOL, atol=DENSE_ATOL)


@needs_engine
def test_the_writer_produces_a_checkpoint_the_tokenizer_can_read(case) -> None:
    """The claim the comparisons rest on, checked against the checkpoint's own tokenizer.

    A vocabulary whose merges could fire on ``"ab"`` would make the prompt a
    tokenizer test and the comparison below would be measuring the wrong thing --
    so the ids are pinned against the engine's tokenizer rather than assumed.
    """
    with native.Engine.open(str(case["path"]), "cpu") as engine:
        assert engine.encode(PROMPT_TEXT, add_special=False) == PROMPT_IDS
        # The checkpoint's own `add_bos_token` is false, so this is the same
        # answer for an independent reason rather than a coincidence.
        assert engine.encode(PROMPT_TEXT) == PROMPT_IDS


@needs_engine
def test_the_python_graph_agrees_with_the_c_engine(case) -> None:
    """The claim this file exists for.

    Two independent implementations of ``Qwen3Model::forward`` -- one in C++ over
    its own kernel interface, one as a graph of ABI ops on numpy -- reading the
    same file.  The measured agreement is 3.0e-7 absolute on logits spanning 1.4,
    some 300 times inside the band, because neither has to reassociate a sum the
    other made.
    """
    _compare(
        case["python"], _c_logits(case["path"], PROMPT_IDS),
        label="python vs engine", rtol=DENSE_RTOL, atol=DENSE_ATOL,
    )


@needs_llama
def test_both_implementations_agree_with_llama_cpp(case) -> None:
    """The authority check, and the one that can fail while the others pass.

    ``test_the_python_graph_agrees_with_the_c_engine`` says the graph is what the
    C engine computes.  It does not say the C engine is right -- if the graph
    transcribed a bug faithfully, that test passes and the model is wrong.  This
    one compares each side against llama.cpp, the only implementation here not
    written from this tree's reading of the architecture.

    The C engine is checked against llama.cpp on a real checkpoint in
    ``tests/native/test_forward.py``; doing it here as well is what lets a
    failure below be attributed, because a disagreement needs a known-correct
    side to land on.
    """
    llama = _llama_logits(case["path"], PROMPT_IDS)
    _compare(case["python"], llama, label="python vs llama.cpp", rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
    if native.is_available():
        _compare(
            _c_logits(case["path"], PROMPT_IDS), llama,
            label="engine vs llama.cpp", rtol=ORACLE_RTOL, atol=ORACLE_ATOL,
        )
    # The answer rather than the number, which is the claim that survives a
    # small numerical difference -- and it is a *decode*, so a cache written one
    # step late fails here while the prefill comparison above still passes.
    assert int(np.argmax(case["python"])) == int(np.argmax(llama))
    if native.is_available():
        # The engine's own `argmax`, so the tie rule under test is the one it
        # ships rather than numpy's.
        assert native.Engine.argmax(list(_c_logits(case["path"], PROMPT_IDS))) == int(np.argmax(llama))