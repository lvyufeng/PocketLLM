"""The model IR and the toy architecture: the builder's contract, exercised.

A builder that has never produced a graph an executor accepted is a design, not
code -- which is why one architecture ships.  The tests here drive the builder
through the same paths a real model would use: a weight declared, a node appended,
a descriptor inferred, a graph verified, a spec built, and finally *run*.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.architectures import ARCHITECTURES, GraphBuilder, ModelSpec, WeightTable, build, get, names
from pocketllm.architectures.cache import CacheLayout, CachePlan, uniform_cache
from pocketllm.architectures.ir import WeightSpec
from pocketllm.architectures.toy import ToyConfig
from pocketllm.backends.reference import BACKEND
from pocketllm.engine import Executor
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import TensorDesc


def _desc(*shape):
    return TensorDesc(tuple(shape), DType.F32)


# -- the builder -------------------------------------------------------------


def test_a_builder_infers_the_output_descriptor_from_the_schema() -> None:
    b = GraphBuilder("t")
    x = b.input("x", _desc(2, 4))
    w = b.weight("w", _desc(3, 4))
    y = b.one("gemm", x, w, outputs="y")
    env = b.build().graph.verify()
    assert env["y"].shape == (2, 3), "the schema's shape rule, not a guess"


def test_the_builder_refuses_a_duplicate_name() -> None:
    b = GraphBuilder("t")
    b.input("x", _desc(2, 2))
    with pytest.raises(ValueError, match="already declared"):
        b.input("x", _desc(2, 2))


def test_the_builder_refuses_the_wrong_number_of_output_names() -> None:
    b = GraphBuilder("t")
    x = b.input("x", _desc(2, 2))
    with pytest.raises(ValueError, match="returns 1 values"):
        b.op("add", [x, x], outputs=["a", "b"])


def test_the_builder_refuses_an_undeclared_argument() -> None:
    from pocketllm.kernels.graph import Value

    b = GraphBuilder("t")
    with pytest.raises(KeyError, match="ghost"):
        b.one("add", Value("ghost"), Value("ghost"), outputs="a")


def test_labels_are_unique_even_when_the_tag_repeats() -> None:
    """A model has a hundred layers; two of them are both called `attn.q`."""
    b = GraphBuilder("t")
    x = b.input("x", _desc(2, 2))
    first = b.one("add", x, x, outputs="a", tag="residual")
    second = b.one("add", first, first, outputs="b", tag="residual")
    labels = [node.label() for node in b.build().graph.nodes]
    assert len(set(labels)) == 2
    assert labels[0] == "residual"
    assert labels[1].startswith("residual.")


def test_weights_are_separated_from_request_inputs() -> None:
    """What the loader fills, and what a caller supplies, are different sets."""
    spec = build("toy", ToyConfig())
    assert spec.weight_values == ("embedding", "norm.weight", "ffn.gate", "ffn.up", "ffn.down")
    assert "tokens" in spec.graph.input_names
    assert "tokens" not in spec.weight_values


def test_a_weight_table_refuses_two_descriptors_for_one_name() -> None:
    table = WeightTable()
    table.add(WeightSpec("w", _desc(2, 2)))
    with pytest.raises(ValueError, match="declared twice"):
        table.add(WeightSpec("w", _desc(3, 3)))
    # The same spec twice is idempotent, not an error: a builder may re-declare.
    table.add(WeightSpec("w", _desc(2, 2)))
    assert len(table) == 1


def test_a_weight_table_reports_total_bytes() -> None:
    table = WeightTable([WeightSpec("a", _desc(2, 4)), WeightSpec("b", _desc(4, 4))])
    assert table.total_bytes() == 32 + 64
    with pytest.raises(KeyError, match="nope"):
        table.get("nope")


def test_a_spec_refuses_an_output_no_node_produces() -> None:
    from pocketllm.kernels.graph import Value

    b = GraphBuilder("t")
    x = b.input("x", _desc(2, 2))
    b.output(x)
    with pytest.raises(KeyError, match="ghost"):
        b.output(Value("ghost"))


def test_build_can_skip_verification() -> None:
    """A caller assembling a deliberately incomplete graph can ask not to verify."""
    b = GraphBuilder("t")
    b.input("x", _desc(2, 2))
    spec = b.build(verify=False)
    assert isinstance(spec, ModelSpec)
    assert not spec.verified


# -- the toy -----------------------------------------------------------------


def test_the_toy_runs_on_the_reference_backend() -> None:
    """The scaffold's acceptance test: a built graph executes end to end.

    This is the claim the whole architectures package rests on -- a builder that
    produces an ordinary graph an ordinary executor can run.  If it fails,
    something in the executor-visible contract moved.
    """
    spec = build("toy", ToyConfig(hidden=8, ff=16, vocab=32))
    session = BACKEND.open(Device("cpu"))
    try:
        executor = Executor(session)
        env = spec.graph.verify()
        inputs = {}
        for name in spec.graph.input_names:
            desc = env[name]
            array = np.zeros(desc.shape, np.int32) if desc.dtype is DType.I32 else np.ones(desc.shape, np.float32)
            inputs[name] = session.tensor(array)
        out = executor.run(spec.graph, inputs)
        y = np.asarray(session.array(out["y"]))
        assert y.shape == (1, 8)
        assert np.isfinite(y).all()
        assert not np.allclose(y, 0.0), "all-zero output means a weight was not read"
    finally:
        session.close()


def test_the_toy_is_deterministic_and_seedable() -> None:
    spec = build("toy", ToyConfig())
    other = build("toy", ToyConfig())
    assert spec.graph.nodes == other.graph.nodes
    assert [n.label() for n in spec.graph.nodes] == [n.label() for n in other.graph.nodes]


def test_the_toy_declares_no_cache() -> None:
    """No attention means no KV cache: the plan is empty, not absent."""
    spec = build("toy", ToyConfig())
    assert spec.cache_values == ()
    assert spec.cache.layouts == ()
    assert spec.cache.bytes_for() == 0 if spec.cache.default_capacity else True


# -- the registry ------------------------------------------------------------


def test_the_registry_builds_a_named_architecture() -> None:
    spec = build("toy")
    assert isinstance(spec, ModelSpec)
    with pytest.raises(KeyError, match="no architecture named"):
        get("xing4_0")


def test_the_registry_names_what_it_has() -> None:
    assert "toy" in names()
    assert "toy" in ARCHITECTURES
    assert ARCHITECTURES["toy"].summary, "every entry carries a line for the listing"


# -- the cache plan ----------------------------------------------------------


def test_a_uniform_cache_has_one_shape_per_tensor() -> None:
    plan = uniform_cache(layers=4, kv_heads=2, head_dim=8, default_capacity=16)
    specs = dict(plan.specs())
    assert specs["k_cache"].shape == (4, 16, 2, 8)
    assert specs["v_cache"].shape == (4, 16, 2, 8)
    assert plan.bytes_for() == 2 * 4 * 16 * 2 * 8 * 4


def test_the_cache_grows_with_the_context() -> None:
    """The context limit is a deployment choice, so the plan takes it as an argument."""
    plan = uniform_cache(layers=4, kv_heads=2, head_dim=8, default_capacity=16)
    assert plan.bytes_for(32) == 2 * plan.bytes_for(16)


def test_a_cache_plan_needs_a_capacity() -> None:
    plan = CachePlan(layouts=(CacheLayout("k", layers=1, kv_heads=1, head_dim=4),), default_capacity=0)
    with pytest.raises(ValueError, match="no cache capacity"):
        plan.specs()


def test_a_cache_layout_refuses_a_zero_capacity() -> None:
    layout = CacheLayout("k", layers=1, kv_heads=1, head_dim=4)
    with pytest.raises(ValueError, match="capacity"):
        layout.shape(0)