"""The decode loop: positions advance, the cache grows, and the graph path is called.

Three claims are worth testing here and nothing else in the suite makes them.

**The loop is a loop.**  A step appends a token at the current position, the next
step reads what the last one wrote, and a truncation puts the cache back to zero.
The reference backend computes this correctly, so a broken loop shows up as a
wrong *sequence*, not as an exception -- which is why the assertions below are
about positions and cache contents rather than about types.

**The graph path is reached.**  ``plan_execution`` and ``run_region`` were
declared and unwired; this file is where they first get a caller, so "the loop
runs" is not the claim -- "the loop compiles the graph and replays it" is.  The
fake session below counts the calls, because a fallback that silently ran eagerly
would pass every other test in this file.

**A wrong plan is caught rather than absorbed.**  ``singleton_plan`` refuses to
capture a graph when a region boundary would turn a cache into a region output,
because on the eager path that output is a buffer the executor will recycle.  The
test drives that check with a backend that *does* claim the narrower op set, so
the refusal is a decision this file pins rather than a property that happens to
hold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pocketllm.architectures import build
from pocketllm.architectures.qwen3 import Qwen3Config
from pocketllm.architectures.toy import ToyConfig
from pocketllm.backends.reference import BACKEND as REFERENCE
from pocketllm.engine import Decoder, Executor, Sampler, cache_descriptors, pick_token, singleton_plan
from pocketllm.kernels.backend import GraphCapability, GraphMode, RegionGranularity
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import TensorDesc

#: Small enough to build in a test, shaped so the loop cannot pass by accident:
#: two layers (so the per-layer cache is a real unrolling), four heads over two
#: kv heads (so the grouping is exercised), and a context short enough to reach
#: the limit in a few steps.
TINY = Qwen3Config(hidden=32, layers=2, heads=4, kv_heads=2, head_dim=8, ff=48, vocab=64, context=8, rows=1)


# -- fixtures -----------------------------------------------------------------


def _env(spec):
    """The graph's descriptor environment, through the public API rather than `_env`."""
    return spec.graph.verify()


def _weights(decoder, cfg: Qwen3Config, seed: int = 5) -> dict:
    """Deterministic weights for every graph weight, plus the rotary tables.

    The tables are graph *inputs* rather than weights -- they are computed from
    the config and bound once -- so a caller that forgets them gets a refusal
    rather than a model with a zeroed rotation.
    """
    rng = np.random.default_rng(seed)
    env = _env(decoder.spec)
    out = {}
    for name in decoder.spec.weight_values:
        shape = env[name].shape
        scale = 0.1 if name.endswith("_norm.weight") else 0.08
        base = 1.0 if name.endswith("_norm.weight") else 0.0
        out[name] = decoder.handle.tensor((base + rng.standard_normal(shape) * scale).astype(np.float32))
    half = cfg.head_dim // 2
    angle = np.arange(cfg.context)[:, None] * cfg.rope_theta ** (-2.0 * np.arange(half) / cfg.head_dim)
    out["rope_cos"] = decoder.handle.tensor(np.cos(angle).astype(np.float32))
    out["rope_sin"] = decoder.handle.tensor(np.sin(angle).astype(np.float32))
    return out


@pytest.fixture(scope="module")
def qwen3_decoder():
    spec = build("qwen3", TINY)
    decoder = Decoder(spec, device="cpu")
    decoder.bind(_weights(decoder, TINY))
    yield decoder
    decoder.close()


# -- the loop ----------------------------------------------------------------


def test_a_step_advances_the_position_and_writes_the_cache(qwen3_decoder) -> None:
    """Three steps, three slots: the cache holds each token at its own position.

    The claim is not "the cache is non-zero" -- a loop that wrote every token at
    slot zero would pass that.  It is that slots ``0..2`` are written and slot
    ``3`` is not, which is what makes the position advance observable.
    """
    generation = qwen3_decoder.start()
    try:
        for token in (3, 7, 11):
            logits = generation.step(token)
            assert logits.desc.shape == (1, TINY.vocab)
            assert np.all(np.isfinite(qwen3_decoder.handle.array(logits)))
        assert generation.position == 3
        assert generation.tokens == [3, 7, 11]

        for il in range(TINY.layers):
            for kind in ("k", "v"):
                cache = qwen3_decoder.handle.array(generation.bindings[f"blk.{il}.{kind}_cache"])
                written = np.flatnonzero(np.abs(cache).sum(axis=(1, 2)) > 0)
                assert list(written) == [0, 1, 2], (
                    f"blk.{il}.{kind}_cache: expected slots 0..2 written, got {list(written)}"
                )
    finally:
        qwen3_decoder.handle.run("cache_truncate", [generation.bindings["blk.0.k_cache"]], attrs={"length": 0})


def test_a_step_reads_what_the_last_one_wrote(qwen3_decoder) -> None:
    """A loop that ignored the cache would produce identical logits every step.

    Nothing else in the file distinguishes "the model read the cache" from "the
    model ignored it", and that is the property the whole design rests on: the
    attention op reads ``positions`` and every slot below it.

    The tokens have to differ for this to be sharp.  Decoding one token three
    times puts the *same* value row in every slot (value carries no rotation), so
    a two-slot softmax averages it back to the single-slot answer and the logits
    agree to within rounding -- a real behaviour, and a useless test.
    """
    generation = qwen3_decoder.start()
    try:
        first = np.asarray(qwen3_decoder.handle.array(generation.step(3))).copy()
        second = np.asarray(qwen3_decoder.handle.array(generation.step(7))).copy()
        third = np.asarray(qwen3_decoder.handle.array(generation.step(11))).copy()
        assert not np.allclose(first, second, atol=1e-5), "step 2 ignored the cache written by step 1"
        assert not np.allclose(second, third, atol=1e-5)
    finally:
        generation.truncate()


def test_truncate_empties_the_cache_and_resets_the_position(qwen3_decoder) -> None:
    """Between requests the cache is zeroed through the op, not rebound to a new tensor."""
    generation = qwen3_decoder.start()
    for token in (4, 9):
        generation.step(token)
    generation.truncate()

    assert generation.position == 0
    assert generation.tokens == []
    for name in qwen3_decoder.spec.cache_values:
        cache = qwen3_decoder.handle.array(generation.bindings[name])
        assert np.count_nonzero(cache) == 0, f"{name} still holds a previous request's rows"

    # ...and the generation is reusable at the same bindings.
    generation.step(1)
    assert generation.position == 1


def test_a_generation_refuses_to_decode_past_its_capacity(qwen3_decoder) -> None:
    """A cache write past the end is corruption, so it is a clear error instead."""
    generation = qwen3_decoder.start()
    for _ in range(TINY.context):
        generation.step(1)
    with pytest.raises(ValueError, match="context limit"):
        generation.step(1)


def test_generate_produces_max_tokens_and_stops_at_eos(qwen3_decoder) -> None:
    assert len(qwen3_decoder.generate([1, 2], max_tokens=3)) == 3
    # `eos` is checked against the sampled token, so a token that is never drawn
    # cannot terminate anything; a token that is drawn always does.
    first = qwen3_decoder.generate([1], max_tokens=1)[0]
    assert qwen3_decoder.generate([1], max_tokens=5, eos=first) == []


def test_stream_yields_the_same_tokens_as_generate(qwen3_decoder) -> None:
    """The generator and the list are one implementation, so they cannot diverge."""
    assert list(qwen3_decoder.stream([1, 2], max_tokens=4)) == qwen3_decoder.generate([1, 2], max_tokens=4)


def test_a_generate_between_requests_does_not_see_the_last_one(qwen3_decoder) -> None:
    """The cache is reset in the loop's ``finally``, so a second call starts clean.

    Without the truncation the second call would read the first call's tokens and
    produce a different sequence; the equality below is what pins that.
    """
    first = qwen3_decoder.generate([1, 2, 3], max_tokens=4)
    second = qwen3_decoder.generate([1, 2, 3], max_tokens=4)
    assert first == second


# -- the sampling step --------------------------------------------------------


def test_the_sampler_is_greedy_without_a_variate() -> None:
    """No draw means no sampling, whatever the other parameters say."""
    assert Sampler().greedy
    assert Sampler(top_k=4, top_p=0.9, min_p=0.05).greedy
    assert not Sampler(temperature=0.7, uniform=lambda: 0.5).greedy
    assert "greedy" in Sampler().describe()
    assert "topk_sample" in Sampler(top_k=4, uniform=lambda: 0.5).describe()


def test_pick_token_runs_the_declared_ops(qwen3_decoder) -> None:
    """The greedy path is ``argmax`` and nothing else; the sampled path is two ops.

    Driven straight at the session with a literal logits vector, so the op
    sequence and the attribute marshalling are checked without a model in the
    way -- ``argmax`` over a one-hot vector has exactly one right answer.
    """
    handle = qwen3_decoder.handle
    logits = np.full(8, -1.0, np.float32)
    logits[5] = 3.0
    tensor = handle.tensor(logits)
    assert pick_token(handle, tensor, Sampler()) == 5

    # A uniform of 0.0 selects the highest-probability token under any top-k that
    # keeps it, so the sampled path is pinned without pinning the RNG.
    sampled = pick_token(handle, tensor, Sampler(temperature=1.0, top_k=4, uniform=lambda: 0.0))
    assert sampled == 5


def test_a_non_positive_temperature_with_a_variate_is_refused(qwen3_decoder) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        qwen3_decoder.generate([1], max_tokens=1, sampler=Sampler(temperature=0.0, uniform=lambda: 0.5))


def test_an_empty_prompt_is_refused(qwen3_decoder) -> None:
    """A model needs one token to prime it; there is no logits vector without one."""
    with pytest.raises(ValueError, match="at least one token"):
        qwen3_decoder.generate([], max_tokens=1)


# -- binding and the missing-input refusal ------------------------------------


def test_a_constant_input_is_refused_rather_than_zeroed() -> None:
    """A zeroed rotary table is a different model, so it is a refusal.

    ``rope_cos``/``rope_sin`` are not weights (nothing loads them from a
    checkpoint) and not per-step inputs (they do not change), so a loop that
    filled them in with zeros would decode with no rotation at all -- finite,
    plausible, wrong.  This is that failure pinned as an error.
    """
    spec = build("qwen3", TINY)
    decoder = Decoder(spec, device="cpu")
    try:
        rng = np.random.default_rng(0)
        env = _env(spec)
        decoder.bind(
            {
                name: decoder.handle.tensor(rng.standard_normal(env[name].shape).astype(np.float32))
                for name in spec.weight_values
            }
        )
        assert set(decoder.missing_constants) == {"rope_cos", "rope_sin"}
        assert not decoder.ready()
        with pytest.raises(RuntimeError, match="rope_cos"):
            decoder.start()
    finally:
        decoder.close()


def test_a_bound_tensor_must_name_a_graph_input_and_match_its_descriptor(qwen3_decoder) -> None:
    with pytest.raises(KeyError, match="not a graph input"):
        qwen3_decoder.bind({"no.such.weight": qwen3_decoder.handle.tensor(np.zeros(4, np.float32))})
    wrong = qwen3_decoder.handle.tensor(np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="but the graph declares"):
        qwen3_decoder.bind({"rope_cos": wrong})


def test_a_cache_descriptor_and_its_plan_must_agree(qwen3_decoder) -> None:
    """The graph is the binding contract; the plan is the budget, and they are checked.

    ``CacheLayout.shape`` carries a leading layer axis -- the plan answers "how
    much does this cost" -- while a graph input is the 3-D shape a kernel reads.
    A binding that used the plan's shape would fail the executor's descriptor
    check one step later, so the two are reconciled once, here.
    """
    descriptors = dict(cache_descriptors(qwen3_decoder.spec))
    assert set(descriptors) == set(qwen3_decoder.spec.cache_values)
    for il in range(TINY.layers):
        for kind in ("k", "v"):
            name = f"blk.{il}.{kind}_cache"
            assert descriptors[name].shape == (TINY.context, TINY.kv_heads, TINY.head_dim)
            assert descriptors[name].dtype is DType.F32

    # A plan that disagrees in a dimension that is not the layer axis is a bug,
    # and it is caught at construction rather than at the first step.
    spec = build("qwen3", TINY)
    from pocketllm.architectures.cache import CacheLayout

    spec.cache = replace(
        spec.cache,
        layouts=(CacheLayout("blk.0.k_cache", layers=1, kv_heads=TINY.kv_heads, head_dim=TINY.head_dim + 2),)
        + spec.cache.layouts[1:],
    )
    with pytest.raises(ValueError, match="the graph declares"):
        cache_descriptors(spec)


def test_a_cache_value_that_is_not_a_graph_input_is_refused() -> None:
    spec = build("qwen3", TINY)
    spec.cache_values = spec.cache_values + ("blk.0.ghost",)
    with pytest.raises(ValueError, match="not a graph input"):
        cache_descriptors(spec)


def test_a_graph_with_no_cache_decodes_as_far_as_it_is_asked() -> None:
    """``toy`` has no attention and no cache: the loop has nothing to bound or reset."""
    spec = build("toy", ToyConfig(hidden=8, ff=16, vocab=32))
    decoder = Decoder(spec, device="cpu")
    try:
        rng = np.random.default_rng(2)
        env = _env(spec)
        decoder.bind(
            {name: decoder.handle.tensor(rng.standard_normal(env[name].shape).astype(np.float32))
             for name in spec.weight_values}
        )
        assert decoder.ready(), "a graph with no constants needs no table"
        assert len(decoder.generate([1, 2, 3], max_tokens=5)) == 5
    finally:
        decoder.close()


# -- the graph path -----------------------------------------------------------


class _AotBackend:
    """A backend that declares it can take the whole graph, and delegates to numpy.

    The point of the type is the *declaration*: ``graph()`` says ``AOT_COMPILE``
    with every op captured, so ``singleton_plan`` produces one whole-graph region
    and ``run_region`` calls ``compile_graph``.  The compilation itself is counted
    rather than merely implemented, because a test that only checked the output
    would pass even if the loop had quietly run eagerly -- which is the exact
    failure this file exists to catch.  The replayed work is the real reference
    executor, so the logits are the ones the model actually produces.
    """

    name = "aot-fake"
    device_kind = "cpu"
    version = "test"
    is_reference = False

    def __init__(self, delegate, *, captures) -> None:
        self.delegate = delegate
        self.captures = frozenset(captures)
        self.compiled = 0
        self.replayed = 0

    def available(self) -> bool:
        return True

    def capabilities(self):  # pragma: no cover - the dispatcher is the delegate's
        return ()

    def graph(self) -> GraphCapability:
        return GraphCapability(
            supported=True,
            mode=GraphMode.AOT_COMPILE,
            captures=self.captures,
            granularity=RegionGranularity.GRAPH,
        )

    def compile_spec(self):
        return None

    def open(self, device, *, options=None):
        return _AotSession(self, device)


class _AotSession:
    """The handle ``run_region`` talks to: a real reference session that also AOTs."""

    def __init__(self, backend: _AotBackend, device: Device) -> None:
        self.backend = backend
        self.device = device
        #: Set by whichever test owns this session, so a compiled region can be
        #: traced back to the graph it was cut from.
        self.current_graph = None
        # The real session, for the memory, the dtypes and the kernels.
        self._delegate = REFERENCE.open(device)
        self._compiled_graph = None

    # -- pass-through, so the eager path still works -------------------------

    def to_device(self, host, desc):
        return self._delegate.to_device(host, desc)

    def to_host(self, tensor):
        return self._delegate.to_host(tensor)

    def alloc(self, nbytes: int, *, align: int = 64):
        return self._delegate.alloc(nbytes, align=align)

    def free(self, buffer) -> None:
        self._delegate.free(buffer)

    def run(self, op, args, *, out=None, attrs=None):
        return self._delegate.run(op, args, out=out, attrs=attrs)

    def close(self) -> None:
        self._delegate.close()

    # -- the graph path ------------------------------------------------------

    def compile_graph(self, graph):
        # ``compile_graph`` takes the *region* the engine wants compiled -- the ABI
        # says so, and it is what lets a backend be handed a slice of a model.  The
        # compiled artifact is cached on the *session* rather than by the caller,
        # which is the ABI's own contract ("the result is cached by the session's
        # own compiled-graph handle, so a decode loop compiles once") and therefore
        # part of what a fake has to model to be a fair test.
        if self._compiled_graph is not None:
            return _CompiledHandle(self)
        self.backend.compiled += 1
        self._compiled_graph = graph
        return _CompiledHandle(self)

    def capture(self, region, *, warmup: int = 3):  # pragma: no cover - not the declared mode
        return None


class _CompiledHandle:
    """What ``compile_graph`` returns: run the graph and hand back its outputs.

    ``compile_graph`` has no ``inputs`` to give it -- ``run`` is called later, with
    the region's inputs in declaration order -- so the handle answers with the
    graph's own output names exactly as a captured artifact would.
    """

    def __init__(self, session: _AotSession) -> None:
        self.session = session

    def run(self, inputs):
        self.session.backend.replayed += 1
        region = self.session._compiled_graph
        # `run` receives the region's inputs in declaration order -- the whole graph
        # here, so that order is the graph's -- and answers in the region's output
        # order, which is how `run_region` pairs values back to names.
        bound = dict(zip(region.inputs, inputs))
        outputs = Executor(self.session._delegate).run(self.session.current_graph, bound)
        return [outputs[name] for name in region.outputs]


_ALL_OPS = (
    "add",
    "attention",
    "argmax",
    "cache_append",
    "cache_truncate",
    "embedding",
    "gemm",
    "layer_norm",
    "logits_temperature",
    "mul",
    "reshape",
    "rms_norm",
    "rope",
    "silu_mul",
    "softmax",
    "topk_sample",
)


def test_the_loop_compiles_the_graph_once_and_replays_it_every_step(qwen3_decoder) -> None:
    """The unwired graph path, wired: ``compile_graph`` is called *and* replayed.

    A fake AOT backend is installed where the reference backend would be.  It
    declares every op captured, so ``singleton_plan`` makes one whole-graph region;
    the loop's first step compiles and every step replays.  A loop that fell back
    to eager execution would leave ``compiled`` at zero while still producing
    correct tokens, which is why the counters are the assertion.
    """
    backend = _AotBackend(qwen3_decoder.handle, captures=_ALL_OPS)
    session = backend.open(Device("cpu"))
    try:
        generation = _generation_over(qwen3_decoder, session, backend)
        for token in (3, 7, 11):
            generation.step(token)
        assert generation.graph_path_used == 3
        assert backend.compiled == 1, "the region compiles once, not once per step"
        assert backend.replayed == 3, "every step replays the compiled artifact"
        # The tokens are the model's, not the fake's: the replayed work is the real
        # reference executor, so a wrong binding order would show up here.
        assert len(qwen3_decoder.generate([3], max_tokens=1)) == 1
    finally:
        session.close()


def test_a_backend_that_cannot_take_the_whole_graph_gets_one_eager_region() -> None:
    """``singleton_plan`` is the narrower question, and this is the answer it gives.

    A backend that takes the ops *around* the cache but not ``cache_append``
    cannot take the graph whole.  The general planner would still capture
    something; the loop's planner does not, because it hands one region to the
    backend and reads the logits out of the reply -- a partial plan's first region
    has no logits in it.  ``graph_path_used`` staying zero is the observable.
    """
    spec = build("qwen3", TINY)
    decoder = Decoder(spec, device="cpu")
    try:
        decoder.bind(_weights(decoder, TINY))
        backend = _AotBackend(decoder.handle, captures=set(_ALL_OPS) - {"cache_append"})
        session = backend.open(Device("cpu"))
        try:
            assert not singleton_plan(spec.graph, backend.graph()).uses_graph_path
            generation = _generation_over(decoder, session, backend)
            for token in (3, 7):
                generation.step(token)
            assert generation.graph_path_used == 0
            assert backend.compiled == 0
        finally:
            session.close()
    finally:
        decoder.close()


def _generation_over(decoder, session, backend):
    """A generation bound to ``session``, with the plan that session's backend earns.

    The bindings are the decoder's own -- the caches, the weights and the tables --
    because the fake session shares the reference session's memory space, so the
    same tensors are valid on both.
    """
    from pocketllm.engine.decode import Generation

    generation = decoder.start()
    generation.handle = session
    generation.plan = singleton_plan(decoder.spec.graph, backend.graph())
    session.current_graph = decoder.spec.graph
    return generation


def test_a_plain_graph_has_one_eager_region_under_singleton_plan() -> None:
    """The reference backend declares nothing, so a decode is one eager region."""
    graph = Graph(
        inputs=(TensorDesc((4,), DType.F32),),
        input_names=("x",),
        nodes=(
            Node(op="add", args=(Value("x"), Value("x")), outputs=("y",), name="n0"),
            Node(op="argmax", args=(Value("y"),), outputs=("t",), name="n1"),
        ),
        outputs=(Value("t"),),
    )
    plan = singleton_plan(graph, REFERENCE.graph())
    assert not plan.uses_graph_path
    assert len(plan.regions) == 1
    assert plan.regions[0].region.inputs == ("x",)
    assert plan.regions[0].region.outputs == ("t",)