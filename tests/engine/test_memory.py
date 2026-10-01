"""The arena and the liveness plan: the two halves of one policy.

The plan is not a derivation of what the executor will do -- it is a
*measurement* of the same arena the executor drives, against a session that
counts instead of allocating.  That is worth testing as a fact rather than a
claim, which is what :func:`test_the_plan_predicts_what_the_executor_does` does:
it plans a graph and then runs it, and the numbers have to be the same.
"""

from __future__ import annotations

import pytest

from pocketllm.engine.memory import BufferArena, CountingSession, plan_memory, value_liveness
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.graph import Graph, Node, Value
from pocketllm.kernels.tensor import TensorDesc


def _desc(nbytes: int) -> TensorDesc:
    """A 2-d f32 descriptor of exactly ``nbytes`` -- ``add`` requires two dims."""
    if nbytes % 8 or nbytes < 8:
        raise ValueError("use a multiple of 8 bytes so the shape stays two-dimensional")
    return TensorDesc((nbytes // 8, 2), dtype=DType.F32)


def _chain(sizes):
    """``a0 = add(x, x); a1 = add(a0, a0); ...`` with one input ``x``.

    Built as ``(env, nodes, outputs)`` rather than as a :class:`Graph`, because a
    chain of ``add`` preserves shape and so cannot produce the *varied* sizes the
    planner's behaviour is about.  ``plan_memory`` and ``value_liveness`` consume
    an environment and a node order directly and never re-run a schema, so a
    hand-built environment is the honest input -- and it lets each value be the
    size the test means it to be.
    """
    env = {"x": _desc(sizes[0])}
    nodes = []
    previous = Value("x")
    for index, nbytes in enumerate(sizes):
        env[f"a{index}"] = _desc(nbytes)
        nodes.append(Node(op="add", args=(previous, previous), outputs=(f"a{index}",), name=f"n{index}"))
        previous = Value(f"a{index}")
    return env, tuple(nodes), (previous.name,)


def _chain_graph(sizes):
    """The same chain as a real :class:`Graph`, for tests that need ``verify``."""
    env, nodes, outputs = _chain(sizes)
    return Graph(
        inputs=(env["x"],),
        input_names=("x",),
        nodes=nodes,
        outputs=tuple(Value(name) for name in outputs),
        name="chain",
    )


def test_liveness_records_the_last_reader() -> None:
    env, nodes, _ = _chain([8, 8, 8])
    live = value_liveness(env, nodes)

    assert live["x"].born == -1, "a graph input is born before the first node"
    assert live["x"].dies == 0
    assert live["a0"].born == 0 and live["a0"].dies == 1
    assert live["a2"].born == 2 and live["a2"].dies == 2


def test_a_value_that_is_born_and_never_read_dies_at_birth() -> None:
    """A scratch value nothing reads must not look immortal.

    Otherwise the release pass never frees it and its buffer is held for the rest
    of the run.  The chain here ends at ``a0``: node 0 produces it, nothing reads
    it, so it dies at position 0.
    """
    env = {"x": _desc(8), "a0": _desc(8)}
    nodes = (Node(op="add", args=(Value("x"), Value("x")), outputs=("a0",), name="n0"),)
    live = value_liveness(env, nodes)
    assert live["a0"].born == 0
    assert live["a0"].dies == 0, "a value nothing reads dies where it is made"


def test_a_value_produced_by_no_node_is_treated_as_an_input() -> None:
    """``value_liveness`` is not handed the input list, and does not need it.

    A value in the environment that no node produces *is* a caller-supplied
    input -- there is no third case in a verified graph -- so it is born at -1
    and never freed, because its buffer belongs to the caller.
    """
    env, nodes, _ = _chain([8, 8])
    env = dict(env)
    env["supplied"] = _desc(8)
    live = value_liveness(env, nodes)

    assert live["supplied"].born == -1
    assert not live["supplied"].frees_after(99), "a caller's buffer is never handed back"


def test_plan_reports_two_activations_not_one_per_value() -> None:
    """A hand-off needs the old value and the new one at once; four values cost two.

    The peak is not one buffer, and that is not a defect: node ``i`` must read
    ``a(i-1)`` while writing ``a(i)``, so any allocator holds two activations
    across the boundary.  What the liveness pass buys is that it is *two* rather
    than one per value -- the scratch of a long graph does not accumulate.
    """
    env, nodes, outputs = _chain([64, 64, 64])
    plan = plan_memory(env, nodes, outputs=outputs)

    assert plan.peak_live_bytes == 128, "a live value plus the value replacing it"
    assert plan.allocations == 2, "the third value recycles the first's buffer"
    assert plan.reuses == 1
    assert plan.no_reuse_bytes == 256, "x, a0, a1, a2 at 64 B each"
    assert plan.saved_bytes == 128


def test_plan_does_not_grow_with_the_number_of_values() -> None:
    """The property that matters on a deep model: scratch is flat, not linear."""
    two, nodes_two, out_two = _chain([64, 64])
    six, nodes_six, out_six = _chain([64, 64, 64, 64, 64, 64])
    assert plan_memory(two, nodes_two, outputs=out_two).peak_live_bytes == (
        plan_memory(six, nodes_six, outputs=out_six).peak_live_bytes
    )


def test_plan_grows_the_peak_with_the_widest_value() -> None:
    env, nodes, outputs = _chain([8, 16, 32, 64])
    plan = plan_memory(env, nodes, outputs=outputs)
    assert plan.peak_live_bytes >= 64, "the widest value must be resident at once"
    assert plan.peak_live_bytes <= sum(entry.nbytes for entry in plan.values)


def test_plan_keeps_the_named_outputs_resident() -> None:
    """A graph output the caller keeps is not freed when its last reader runs."""
    env, nodes, outputs = _chain([64, 64])
    kept = plan_memory(env, nodes, outputs=outputs)
    dropped = plan_memory(env, nodes, outputs=())

    assert kept.peak_live_bytes >= 64
    assert dropped.peak_live_bytes <= kept.peak_live_bytes


def test_plan_for_value_names_what_it_does_not_have() -> None:
    env, nodes, outputs = _chain([8, 8])
    plan = plan_memory(env, nodes, outputs=outputs)
    assert plan.for_value("x").nbytes == 8
    with pytest.raises(KeyError, match="nope"):
        plan.for_value("nope")


def test_the_chain_graph_verifies() -> None:
    """The :class:`Graph` form of the helper is a real graph, not just data."""
    graph = _chain_graph([8, 8, 8])
    env = graph.verify()
    assert env["a2"].shape == (1, 2)


def test_the_plan_predicts_what_the_executor_does() -> None:
    """The planner and the executor drive the same arena in the same order.

    Not a shape assertion -- the exact numbers.  If these ever diverge, the plan
    printed by a tool is describing a different execution than the one that
    happens, which is worse than not printing one.
    """
    import numpy as np

    from pocketllm.architectures.toy import ToyConfig, build
    from pocketllm.backends.reference import BACKEND
    from pocketllm.engine.executor import Executor

    spec = build(ToyConfig(hidden=8, ff=16, vocab=32))
    graph = spec.graph
    env = graph.verify()
    planned = plan_memory(env, graph.nodes, outputs=("y",))

    session = BACKEND.open(Device("cpu"))
    try:
        executor = Executor(session)
        inputs = {}
        for name in graph.input_names:
            desc = env[name]
            array = np.zeros(desc.shape, np.float32) if desc.dtype.is_float else np.zeros(desc.shape, np.int32)
            inputs[name] = session.tensor(array)
        run = executor.run(graph, inputs, trace=True)
    finally:
        session.close()

    assert run.trace.peak_bytes == planned.peak_live_bytes
    assert run.trace.allocations == planned.allocations
    assert run.trace.reuses == planned.reuses


# -- the arena on its own ----------------------------------------------------


def test_arena_reuses_only_the_same_size() -> None:
    """Exact-size reuse; a larger free buffer is not silently loaned out.

    A subview of a bigger buffer would alias a value another part of the plan
    still considers live, and the plan is the thing that has to be right.
    """
    arena = BufferArena(CountingSession())
    arena.acquire(16)          # the only 16-byte buffer, never released
    big = arena.acquire(32)
    arena.release(big)

    assert arena.acquire(32) is big, "the same size must come back"
    assert arena.allocations == 2, "the 32-byte request was served from the pool"
    # The 16-byte buffer was not handed to the 32-byte request, and the released
    # 32-byte buffer was not handed to a later 16-byte one.
    assert arena.acquire(16).nbytes == 16
    arena.close()


def test_arena_ignores_a_buffer_it_never_handed_out() -> None:
    """Releasing a foreign buffer is a no-op: the executor's cleanup cannot know."""
    from pocketllm.kernels.buffer import DeviceBuffer

    arena = BufferArena(CountingSession())
    stranger = DeviceBuffer(device=Device("cpu"), nbytes=16)
    arena.release(stranger)  # must not raise, must not corrupt the pool
    assert arena.live_bytes == 0
    assert arena.acquire(16).nbytes == 16
    arena.close()


def test_arena_reports_when_it_is_empty() -> None:
    arena = BufferArena(CountingSession())
    buffer = arena.acquire(64)
    assert arena.live_bytes == 64
    arena.release(buffer)
    assert arena.live_bytes == 0
    arena.close()


def test_counting_session_counts() -> None:
    session = CountingSession()
    buffers = [session.alloc(8) for _ in range(3)]
    assert session.allocated == 3
    for buffer in buffers:
        session.free(buffer)
    assert session.freed == 3