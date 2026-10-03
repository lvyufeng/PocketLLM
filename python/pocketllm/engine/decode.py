"""Generation: the loop that makes a graph a model.

Everything up to here runs a graph *once*.  :class:`~pocketllm.engine.executor.Executor`
walks the nodes and stops; a :class:`~pocketllm.architectures.ir.ModelSpec` describes
one call.  What turns that into a model is the part nothing in this tree had:
append a token, advance a position, sample, feed the token back -- and stop when
something says stop.

**Why this is a module and not a method on ``Executor``.**  The executor's own
docstring says it is "the always-works path": a graph in, a value out, no
assumption that the backend can capture anything.  A decode loop is the opposite
kind of code -- it holds a cache across steps, it will hand a region to a
backend's graph path when one is offered, and it owns sampling policy.  Folding
that into the executor would make every one-shot run pay for a loop it does not
use.  So the loop lives here and *uses* the executor, eagerly or through
:func:`~pocketllm.engine.captured.run_region`.

**Where the graph path gets its first production caller.**  ``plan_execution`` and
``run_region`` were declared, tested and unwired: nothing in the tree called them.
A decode step is exactly the region shape they were written for -- a fixed
sequence of ops replayed at a new position with the same shapes -- so the wiring
belongs here, behind a check on what the backend declares.  A backend that
declares no graph path takes the eager branch and is *correct* rather than
degraded; that is the reference backend today, and it is the reason the eager
branch is not a fallback.

One consequence of ``run_region``'s contract shapes this file more than anything
else: it runs *one* region, so the region it is given has to be one that produces
the logits.  ``plan_execution`` is free to split a graph into several captured
runs with eager nodes between them -- good for a one-shot run, useless here,
because replaying the first region alone returns some middle value and not the
answer.  :func:`singleton_plan` is what turns "can this backend take regions?"
into the question a loop can act on: "can it take *this whole graph*?"  The answer
is one whole-graph region or one eager region, and no third case.

**The cache is updated in place, and the bindings are not rebound.**  A step binds
``tokens`` and ``positions`` and leaves every cache input bound to the tensor the
caller supplied.  That is not an optimisation: ``cache_append``'s declared
semantics are ``cache[positions[i]] = values[i]``, "an in-place write, returned for
composition", so the buffer the caller handed in *is* the updated cache.  The
reference implementation writes through it and the C engine's per-layer slab write
does the same, and the conformance harness is what keeps a third backend to it.

**Sampling is the host's, and the RNG with it.**  The engine takes a uniform
variate and holds no generator of its own -- the same split the C engine uses,
where ``run.cpp`` carries a ``std::mt19937_64`` and the library carries none.  So
:class:`Sampler` is *given* how to draw; with no draw supplied it decodes greedily
through ``argmax``, which needs no variate at all.

**A generation owns its cache, not the decoder.**  :meth:`Decoder.start` allocates
a fresh cache per generation, so two requests never see each other's tokens while
the weights stay bound.  The cost is a reallocation per request; the alternative is
a cache whose length a caller has to remember to reset, which is the bug this
avoids.
"""

from __future__ import annotations

import numpy as np

from typing import Any, Callable, Iterator, Mapping, Sequence

from pocketllm.architectures.ir import ModelSpec
from pocketllm.kernels.backend import GraphCapability
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.graph import Graph, GraphRegion
from pocketllm.kernels.tensor import Tensor, TensorDesc

from .captured import run_region
from .executor import Executor
from .planner import ExecutionPlan
from .session import EngineSession, SessionPolicy

__all__ = [
    "Sampler",
    "Generation",
    "Decoder",
    "STEP_INPUTS",
    "cache_descriptors",
    "pick_token",
    "singleton_plan",
]

#: The graph inputs a decode step replaces every token.  Everything else that is
#: neither a weight nor a cache value is a *constant* the caller binds once -- a
#: rotary table, a mask -- and it has to be named here rather than guessed,
#: because a loop that silently zeroed a rotary table would compute a model
#: nobody asked for: finite, plausible, and wrong.
STEP_INPUTS: tuple[str, ...] = ("tokens", "positions")

#: A numpy dtype for every ABI element type, so a helper can allocate through
#: ``to_device`` rather than through a backend's own convenience method (which is
#: the reference session's, and not part of the ABI).
_NP_OF: dict[str, str] = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "uint16",  # no numpy bfloat16; the bits are the descriptor's business
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


class Sampler:
    """How a step picks a token from logits.

    Greedy is the default and needs nothing: ``argmax`` is the engine's own op and
    takes no variate.  A non-greedy sampler is *given* a ``uniform`` callable --
    the host's RNG -- because the engine deliberately has none; see the module
    docstring for why that is the C engine's split too.

    Both declared sampling paths are reachable from here, and going through the
    ops rather than through numpy means a backend that implements sampling on the
    device uses its own kernel without this loop knowing.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        uniform: Callable[[], float] | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.uniform = uniform

    @property
    def greedy(self) -> bool:
        """No variate means no sampling -- and the op that needs one is not called.

        The rule keys on the variate and not on the parameters on purpose.  A
        caller that sets ``top_k`` but supplies no draw has asked for a sampled
        token without a source of randomness; falling back to ``topk_sample``'s
        own default variate would return a fixed token and call it sampling, which
        is worse than being greedy and saying so.
        """
        return self.uniform is None

    def describe(self) -> str:
        if self.greedy:
            return "greedy (argmax)"
        bits = [f"temperature={self.temperature}"]
        if self.top_k:
            bits.append(f"top_k={self.top_k}")
        if self.top_p < 1.0:
            bits.append(f"top_p={self.top_p}")
        if self.min_p > 0.0:
            bits.append(f"min_p={self.min_p}")
        return "topk_sample (" + ", ".join(bits) + ")"


def singleton_plan(graph: Graph, capability: GraphCapability) -> ExecutionPlan:
    """The plan a decode loop can use: one whole-graph region, or one eager region.

    ``plan_execution`` is free to split a graph into many regions, and for a
    one-shot run that is what it should do.  A decode loop cannot use those splits
    -- it hands one region to the backend and reads the logits out of what comes
    back, and a partial plan's first region does not contain them -- so it asks
    the narrower question instead: can this backend take the whole graph at once?

    The check is :meth:`GraphCapability.admits`, which is the backend's own
    declaration, so this restricts the general plan rather than forming a second
    opinion about what a backend can do.

    The decline branch does **not** delegate to :func:`plan_execution`.  A backend
    that supports a capture path at all would come back from that with a plan that
    captures some regions and runs others eagerly -- which is the thing a loop
    cannot use, and would have quietly reported ``uses_graph_path`` as true.  The
    answer here is one eager region over the whole graph, or nothing.
    """
    from .planner import PlanRegion

    region = GraphRegion.whole(graph)
    if capability.admits(region):
        return ExecutionPlan(regions=(PlanRegion(region=region, captured=True),), capability=capability)
    if not graph.nodes:
        return ExecutionPlan(regions=(), capability=capability)

    reason = (
        "the backend cannot take the whole graph at once"
        if capability.supported
        else "the backend declares no graph path"
    )
    eager = PlanRegion(region=region, captured=False, reason=reason)
    return ExecutionPlan(regions=(eager,), capability=capability)


def cache_descriptors(spec: ModelSpec) -> tuple[tuple[str, TensorDesc], ...]:
    """The descriptors the *graph* declares for the cache values, checked against the plan.

    The graph is the contract a step is bound against, so its descriptors are what
    a loop allocates from.  The :class:`~pocketllm.architectures.cache.CachePlan` is
    an allocation *plan* -- for a whole-model cache it carries a leading layer axis
    that a graph input does not, because the plan answers "how much does this cost"
    and the graph answers "what do I read".  The two describe the same bytes, and
    :func:`_plan_agrees` is what keeps them from drifting into a cache the plan
    budgets for and the graph refuses.
    """
    env = spec.graph.verify()
    plan = (
        {name: layout for layout, (name, _) in zip(spec.cache.layouts, spec.cache.specs())}
        if spec.cache.layouts
        else {}
    )
    out: list[tuple[str, TensorDesc]] = []
    for name in spec.cache_values:
        try:
            desc = env[name]
        except KeyError as exc:
            raise ValueError(
                f"{spec.name}: cache value {name!r} is not a graph input, so no step can bind it"
            ) from exc
        if name in plan:
            _plan_agrees(name, desc, plan[name].desc(spec.cache.default_capacity))
        out.append((name, desc))
    return tuple(out)


def _plan_agrees(name: str, declared: TensorDesc, planned: TensorDesc) -> None:
    """The plan may prepend one layer axis; nothing else may differ."""
    if declared == planned:
        return
    if planned.shape[1:] == declared.shape and planned.dtype == declared.dtype:
        return
    raise ValueError(
        f"cache {name!r}: the graph declares {declared} but the plan allocates {planned}; "
        "one of the two is wrong and a decode step cannot reconcile them"
    )


def pick_token(session, logits: Tensor, sampler: Sampler) -> int | None:
    """One token from a step's logits, through the declared sampling ops.

    The sampling ops are declared over a single row (``shape=("vocab",)``) while a
    graph may produce several, so a reshape narrows to the last row.  There is no
    ``slice``; at the width a decode loop builds at -- one row -- the reshape is
    the identity and this branch never fires.

    A module-level function rather than a method so it can be driven directly
    against a session, which is how the op sequence and the attribute marshalling
    are checked without a model in the way.
    """
    vocab = logits.desc.shape[-1]
    if logits.desc.shape != (vocab,):
        (logits,) = session.run("reshape", [logits], attrs={"shape": (int(vocab),)})
    if sampler.greedy:
        (token,) = session.run("argmax", [logits])
        return _read_i32(session, token)
    (scaled,) = session.run("logits_temperature", [logits], attrs={"temperature": sampler.temperature})
    assert sampler.uniform is not None  # `greedy` above is exactly "no variate"
    (token,) = session.run(
        "topk_sample",
        [scaled, _f32(session, float(sampler.uniform()))],
        attrs={"top_k": sampler.top_k, "top_p": sampler.top_p, "min_p": sampler.min_p},
    )
    return _read_i32(session, token)


class Generation:
    """One in-flight generation: its cache, its position, and what it produced.

    The cache lives here rather than in a model object because this is the thing
    that spans steps.  ``bindings`` is the running value environment -- every graph
    input's current tensor -- so a step means "replace ``tokens`` and
    ``positions``, run, keep everything else bound where it was".
    """

    def __init__(
        self,
        spec: ModelSpec,
        handle: Any,
        executor: Executor,
        bindings: dict[str, Tensor],
        plan: ExecutionPlan,
    ) -> None:
        self.spec = spec
        #: The backend *handle* (a ``ReferenceSession``, a ``QnnBackendSession``),
        #: not the :class:`EngineSession` that selected it: ``run_region`` speaks to
        #: the handle.
        self.handle = handle
        self.executor = executor
        self.bindings = bindings
        self.plan = plan
        self.position = 0
        self.tokens: list[int] = []
        self.graph_path_used = 0

    @property
    def capacity(self) -> int | None:
        """The context limit, or ``None`` for a graph that declares no cache.

        A model with no attention has nothing to bound, and reporting ``0`` for it
        would refuse the first step with a context-limit error about a cache that
        does not exist.
        """
        return self.spec.cache.default_capacity or None

    @property
    def logits_name(self) -> str:
        return self.spec.graph.outputs[0].name

    def step(self, token: int) -> Tensor:
        """Append ``token`` at the current position and return the logits.

        Refusing at the context limit rather than wrapping is deliberate: a cache
        write at a slot past the end is a device fault on a card and silent
        corruption on the host, and a clear error is worth more than either.

        ``tokens`` and ``positions`` are only rebound when the graph declares
        them, so a graph with no position input is driven by the token alone
        rather than failing the executor's extra-input check.
        """
        capacity = self.capacity
        if capacity is not None and self.position >= capacity:
            raise ValueError(
                f"{self.spec.name} is at its context limit ({capacity} positions); build the spec "
                "with a larger capacity rather than decoding past the end"
            )
        if "tokens" in self.bindings:
            self.bindings["tokens"] = _i32(self.handle, [int(token)])
        if "positions" in self.bindings:
            self.bindings["positions"] = _i32(self.handle, [self.position])
        logits = self._run()
        self.tokens.append(int(token))
        self.position += 1
        return logits

    def _run(self) -> Tensor:
        """Run one step, through the graph path when the backend declares one.

        The graph path is only usable when the plan is a single captured region
        *and* that region produces the logits -- the condition
        :func:`singleton_plan` guarantees and a caller who supplies their own plan
        has to meet.  Anything else runs eagerly rather than replaying a region
        whose result is not the answer.
        """
        if self.plan.uses_graph_path and len(self.plan.regions) == 1:
            region = self.plan.regions[0].region
            if self.plan.regions[0].captured and self.logits_name in region.outputs:
                outcome = run_region(self.handle, region, self.bindings)
                if outcome is not None:
                    self.graph_path_used += 1
                    return outcome.outputs[self.logits_name]
        return self.executor.run(self.spec.graph, self.bindings)[self.logits_name]

    def truncate(self) -> None:
        """Forget the cache and reset the position, for a new request.

        Every cache value is truncated through the declared op rather than by
        rebinding a fresh zero tensor, so a backend with a paged or
        device-resident cache forgets it its own way instead of being handed a
        host buffer it does not recognise.  The op writes in place, so the binding
        is left exactly where it was.
        """
        for name in self.spec.cache_values:
            self.handle.run("cache_truncate", [self.bindings[name]], attrs={"length": 0})
        self.position = 0
        self.tokens.clear()

    def summary(self) -> str:
        where = f"{self.graph_path_used} steps via the graph path" if self.plan.uses_graph_path else "eager"
        limit = "unbounded" if self.capacity is None else str(self.capacity)
        return f"{len(self.tokens)} tokens at position {self.position}/{limit} ({where})"


class Decoder:
    """A spec, a device and a set of weights: everything a decode needs but the prompt.

    Weights are bound once, in :meth:`bind`; a cache is allocated once *per
    generation*, which is what :meth:`start` hands out.

    ```python
    decoder = Decoder(spec, device="cpu")
    decoder.bind(weights)                 # the loader's job in a real run
    tokens = decoder.generate([101, 202], max_tokens=8)
    ```
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: Device | str | int | None = None,
        policy: SessionPolicy | None = None,
        options: Mapping[str, Any] | None = None,
        weights: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.spec = spec
        self.engine = EngineSession.open(device, policy=policy, options=options)
        self.handle = self.engine.session
        self.executor = self.engine.executor()
        self._env: dict[str, TensorDesc] = spec.graph.verify()
        #: Each cache value's *declared* descriptor, checked against the plan once
        #: at construction rather than re-checked on every generation.
        self._cache = cache_descriptors(spec)
        self._weights: dict[str, Tensor] = {}
        # The plan is computed from the backend's *declarations*, before any device
        # call, so a backend with a graph path gets one whole-graph region and one
        # without gets a single eager region and no branch at run time.
        self.plan = singleton_plan(spec.graph, self.engine.backend.graph())
        if weights is not None:
            self.bind(weights)

    # -- weights -------------------------------------------------------------

    def bind(self, tensors: Mapping[str, Tensor]) -> "Decoder":
        """Hold these tensors under the names the graph knows, checking each one.

        A name that is not a graph input, or a descriptor that disagrees with the
        graph's, is refused here rather than three ops into a kernel with an error
        naming a value nobody recognises -- the executor's own argument, applied
        one layer up so the message can name the checkpoint.
        """
        for name, tensor in tensors.items():
            desc = self._env.get(name)
            if desc is None:
                raise KeyError(
                    f"{name!r} is not a graph input of {self.spec.name}; it takes {list(self.spec.graph.input_names)}"
                )
            if tensor.desc != desc:
                raise ValueError(f"tensor {name!r} has descriptor {tensor.desc}, but the graph declares {desc}")
        self._weights.update(tensors)
        return self

    @property
    def missing_weights(self) -> tuple[str, ...]:
        return tuple(name for name in self.spec.weight_values if name not in self._weights)

    @property
    def missing_constants(self) -> tuple[str, ...]:
        """Non-weight, non-cache inputs nobody has bound: the rotary tables, usually.

        These are not the checkpoint's to supply -- a rotary table is computed from
        the config -- but they are the model's, not the request's, so a decode loop
        cannot invent them either.  Reporting them is the difference between "bind
        the tables" and a cache full of zeros.
        """
        cache = {name for name, _ in self._cache}
        return tuple(
            name
            for name in self.spec.graph.input_names
            if name not in cache and name not in STEP_INPUTS and name not in self._weights
        )

    @property
    def missing_inputs(self) -> tuple[str, ...]:
        return self.missing_weights + self.missing_constants

    def ready(self) -> bool:
        return not self.missing_inputs

    # -- generations ---------------------------------------------------------

    def start(self) -> Generation:
        """A fresh generation: an empty cache, every input bound, position zero.

        Every graph input is bound here so a step only replaces ``tokens`` and
        ``positions``.  A constant the caller has not supplied is refused rather
        than zeroed: a zeroed rotary table turns the model into a different one,
        and a decode loop that did that quietly would be worse than useless.  The
        only inputs filled in without being asked for are the cache values, which
        a generation owns, and the per-step inputs, which default to a zeroed row
        of the right width for a caller that reaches a step without setting them.
        """
        missing = self.missing_inputs
        if missing:
            raise RuntimeError(
                f"{self.spec.name} cannot decode: {len(missing)} input(s) unbound -- {list(missing)}. "
                "The checkpoint's weights and every non-per-step constant (a rotary table, for "
                "instance) must be bound with `bind()` first."
            )
        bindings: dict[str, Tensor] = dict(self._weights)
        for name, desc in self._cache:
            bindings[name] = _zeros(self.handle, desc)
        for name, desc in zip(self.spec.graph.input_names, self.spec.graph.inputs):
            if name in bindings:
                continue
            bindings[name] = _default_input(self.handle, desc)
        return Generation(self.spec, self.handle, self.executor, bindings, self.plan)

    def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int = 16,
        sampler: Sampler | None = None,
        eos: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Decode ``prompt``, then at most ``max_tokens`` more, greedily by default.

        The prompt is fed one token per step rather than as one wide call because
        the graph's width is fixed at build time (``Qwen3Config.rows``) and a
        decode step is one row.  That is a property of the ABI's literal
        ``reshape`` rather than a choice made here; a backend that can prefill
        wider wants a second spec at that width and a branch at this call, which
        is the AOT split the design already anticipates.
        """
        return list(self._iterate(prompt, max_tokens=max_tokens, sampler=sampler, eos=eos, on_token=on_token))

    def stream(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int = 16,
        sampler: Sampler | None = None,
        eos: int | None = None,
    ) -> Iterator[int]:
        """``generate`` as a generator: one token at a time, for a serving loop."""
        yield from self._iterate(prompt, max_tokens=max_tokens, sampler=sampler, eos=eos, on_token=None)

    def _iterate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        sampler: Sampler | None,
        eos: int | None,
        on_token: Callable[[int], None] | None,
    ) -> Iterator[int]:
        if not prompt:
            raise ValueError("a generation needs at least one token to prime the model")
        sampler = sampler or Sampler()
        if not sampler.greedy and sampler.temperature <= 0.0:
            raise ValueError("a sampling temperature must be positive; use greedy for no randomness")
        generation = self.start()
        try:
            logits: Tensor | None = None
            for token in prompt:
                logits = generation.step(token)
            for _ in range(int(max_tokens)):
                token = self._pick(logits, sampler)
                if token is None or token == eos:
                    return
                if on_token is not None:
                    on_token(token)
                yield token
                logits = generation.step(token)
        finally:
            # The cache is the session's largest buffer and a generation is over
            # the moment its last token is handed back.  `truncate` rather than a
            # free, so a backend whose cache is device-resident learns about it
            # through the op it already implements.
            generation.truncate()

    def _pick(self, logits: Tensor, sampler: Sampler) -> int | None:
        return pick_token(self.handle, logits, sampler)

    # -- lifetime ------------------------------------------------------------

    def close(self) -> None:
        self._weights.clear()
        self.executor.close()
        self.engine.close()

    def __enter__(self) -> "Decoder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def describe(self) -> str:
        state = "ready" if self.ready() else f"{len(self.missing_inputs)} inputs unbound"
        return f"{self.spec.describe()} [{state}] on {self.engine.describe()}; plan: {self.plan.summary()}"


# -- ABI-only tensor helpers ---------------------------------------------------
#
# These go through `to_device` rather than through a session's own `tensor()`
# convenience, which the reference backend has and the ABI does not.  A decode loop
# that only worked on backends that happened to copy that helper would be a loop
# that could not run on a phone.


def _zeros(session, desc: TensorDesc) -> Tensor:
    return _host(session, np.zeros(desc.shape, _np_dtype(desc)), desc)


def _default_input(session, desc: TensorDesc) -> Tensor:
    """A zeroed tensor for a per-step input, so a generation starts bindable.

    Only :data:`STEP_INPUTS` ever reaches this: ``tokens`` and ``positions`` are
    replaced before the first run, and a graph with a different width or no
    position input would otherwise fail the executor's coverage check before its
    first step.  Every other unbound input is a refusal in :meth:`Decoder.start`.
    """
    count = int(np.prod(desc.shape)) if desc.shape else 1
    values: np.ndarray = np.zeros(count, _np_dtype(desc))
    return _host(session, values.reshape(desc.shape), desc)


def _i32(session, values: Sequence[int]) -> Tensor:
    array = np.asarray(list(values), dtype=np.int32)
    return _host(session, array, TensorDesc(tuple(array.shape), dtype=DType.I32))


def _f32(session, value: float) -> Tensor:
    # A 0-d array, which is what the schema's `shape=()` uniform wants.
    array = np.asarray(value, dtype=np.float32)
    return _host(session, array, TensorDesc((), dtype=DType.F32))


def _host(session, array: np.ndarray, desc: TensorDesc) -> Tensor:
    raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    return session.to_device(memoryview(raw).cast("B"), desc)


def _np_dtype(desc: TensorDesc) -> str:
    assert desc.dtype is not None
    try:
        return _NP_OF[desc.dtype.value]
    except KeyError as exc:  # pragma: no cover - the ABI's dtypes are enumerated
        raise NotImplementedError(f"no numpy dtype for {desc.dtype}") from exc


def _read_i32(session, tensor: Tensor) -> int:
    view = np.frombuffer(session.to_host(tensor), dtype=np.int32)
    return int(view.reshape(-1)[0])