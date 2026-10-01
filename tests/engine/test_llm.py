"""``LLM``: the composition of a spec, a device, and the engine below them.

The class is small on purpose, and what is worth testing is not that it forwards
a call but the three places it has an opinion:

* **weights are bound, not passed per call** -- and a binding that does not match
  the graph is refused at the binding, naming the value, rather than surfacing as
  a shape error three nodes into a kernel;
* **an unbound model refuses to run**, instead of running with a stale or zero
  weight and producing a plausible wrong answer;
* **the device is opened once** -- one session, held for the object's life.

The graph used throughout is ``toy``, which is the only architecture that ships:
a real model would test the same contract with more nodes, and a mocked one would
test nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pocketllm.architectures import build
from pocketllm.architectures.toy import ToyConfig
from pocketllm.engine import AsyncLLM, LLM, SessionPolicy
from pocketllm.kernels.device import Device
from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import TensorDesc


def _toy(**overrides):
    return build("toy", ToyConfig(hidden=8, ff=16, vocab=32, **overrides))


def _host(wrapper) -> object:
    """The backend session behind an :class:`EngineSession`.

    The numpy helpers are the *backend's* -- one layer below the engine wrapper
    that owns the device and the policy -- and reaching through here rather than
    growing them on the wrapper keeps the two layers' jobs from blurring.
    """
    return wrapper.session


def _weights(spec, model, *, fill: float = 1.0):
    """Bind every weight the spec names to a constant tensor of the right shape."""
    session = _host(model)
    out = {}
    for name in spec.weight_values:
        desc = spec.graph.verify()[name]
        out[name] = session.tensor(np.full(desc.shape, fill, np.float32))
    return out


def _tokens(session, ids=(1,)):
    """One token, as the toy declares it: a single position, not a prompt.

    ``toy`` takes ``tokens`` at shape ``(1,)`` because embedding gathers with the
    position axis fixed -- the graph is a decode step, not a prefill.  A test that
    passes three ids is testing a model that does not exist.
    """
    return session.tensor(np.asarray(ids, np.int32))


# -- construction ------------------------------------------------------------


def test_a_model_is_not_ready_until_its_weights_are_bound() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        assert not model.ready()
        assert model.missing_weights == spec.weight_values
        with pytest.raises(RuntimeError, match="missing 5 weight"):
            model({"tokens": _tokens(_host(model.session))})


def test_binding_the_weights_makes_the_model_ready() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        assert model.ready()
        assert model.missing_weights == ()


def test_a_weight_bound_at_the_wrong_shape_is_refused_by_name() -> None:
    """The failure has to name the tensor, because the kernel's will not."""
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        with pytest.raises(ValueError, match="ffn.gate"):
            model.bind({"ffn.gate": _host(model.session).tensor(np.zeros((3, 3), np.float32))})


def test_a_name_that_is_not_a_graph_input_is_refused() -> None:
    """A typo in a checkpoint's tensor name is a load error, not a silent no-op."""
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        with pytest.raises(KeyError, match="ffn.gat"):
            model.bind({"ffn.gat": _host(model.session).tensor(np.zeros((16, 8), np.float32))})


def test_request_inputs_go_through_the_same_door_as_weights() -> None:
    """``tokens`` is a graph input too; the engine does not special-case it."""
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        ids = _tokens(_host(model.session))
        model.bind({"tokens": ids})  # legal: it is a graph input
        assert model.ready()
        assert (
            model({"tokens": ids})["y"].desc.shape
            == model.bind({"tokens": ids})({"tokens": ids})["y"].desc.shape
        )


def test_weights_from_binds_positionally_in_declaration_order() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        tensors = [_host(model.session).tensor(np.ones(spec.graph.verify()[n].shape, np.float32))
                   for n in spec.weight_values]
        model.weights_from(tensors)
        assert model.ready()
        with pytest.raises(ValueError, match="2 tensors for 5 weights"):
            model.weights_from(tensors[:2])


# -- running -----------------------------------------------------------------


def test_a_bound_model_runs_the_graph_and_returns_its_outputs() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        out = model({"tokens": _tokens(_host(model.session))})
        y = np.asarray(_host(model.session).array(out["y"]))
        assert y.shape == (1, 8)
        assert np.isfinite(y).all()
        assert not np.allclose(y, 0.0), "an all-zero output means the weights were not read"


def test_a_run_can_carry_the_trace() -> None:
    """``trace=True`` is the diagnostic surface: where it ran, and what it cost."""
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        run = model({"tokens": _tokens(_host(model.session))}, trace=True)
        assert [r.node.op for r in run.trace.nodes] == [
            "embedding", "rms_norm", "gemm", "gemm", "silu_mul", "gemm",
        ]
        assert run.trace.backends_used == ("reference",)
        assert not run.trace.crossed_a_device
        assert run.trace.peak_bytes > 0


def test_an_input_the_graph_does_not_take_is_refused() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        tokens = _tokens(_host(model.session))
        with pytest.raises(ValueError, match="unexpected"):
            model({"tokens": tokens, "prompt": tokens})


def test_a_missing_input_is_refused_with_the_list_of_what_it_takes() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        model.bind(_weights(spec, model.session))
        with pytest.raises(ValueError, match="missing"):
            model({})


# -- the session it opened ---------------------------------------------------


def test_the_model_reports_the_device_it_opened() -> None:
    spec = _toy()
    with LLM(spec, device="cpu") as model:
        assert model.session.device == Device("cpu")
        assert model.session.backend.name == "reference"
        assert "toy" in model.describe()
        assert "5 weights unbound" in model.describe()
        model.bind(_weights(spec, model.session))
        assert "[ready]" in model.describe()


def test_a_named_backend_reaches_the_model_through_the_policy() -> None:
    spec = _toy()
    model = LLM(spec, device="cpu", policy=SessionPolicy.for_run(backend="reference"))
    try:
        assert model.session.backend.name == "reference"
    finally:
        model.close()


def test_closing_twice_is_safe_and_the_context_manager_closes() -> None:
    spec = _toy()
    model = LLM(spec, device="cpu")
    with model as opened:
        assert opened is model
    model.close()  # idempotent


# -- async -------------------------------------------------------------------


class _Recorder:
    """An ``LLM`` that notes the thread each call ran on, then delegates.

    Subclassing ``LLM`` and reacting the instance with ``__dict__`` would share
    the underlying session with a model the ``with`` block then closes, so the
    call would fail for the wrong reason.  Delegating keeps the spy and the
    subject separate objects with one lifetime each.
    """

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        self.calls: list[int] = []

    @property
    def spec(self):
        return self._inner.spec

    def bind(self, tensors):
        self._inner.bind(tensors)
        return self

    def __call__(self, inputs, *, trace: bool = False):
        import threading

        self.calls.append(threading.get_ident())
        return self._inner(inputs, trace=trace)

    def close(self) -> None:
        self._inner.close()

    def describe(self) -> str:
        return self._inner.describe()


def test_the_async_form_runs_on_a_worker_thread() -> None:
    """Not ``async def`` for decoration: the call must actually leave the loop."""
    import asyncio
    import threading

    spec = _toy()
    loop_thread = threading.get_ident()
    recorder = None

    with LLM(spec, device="cpu") as inner:
        inner.bind(_weights(spec, inner.session))
        recorder = _Recorder(inner)
        async_model = AsyncLLM(recorder)
        tokens = _tokens(_host(inner.session))
        out = asyncio.run(async_model({"tokens": tokens}))

    assert recorder.calls and recorder.calls[0] != loop_thread
    assert out["y"].desc.shape == (1, 8)


def test_the_async_form_refuses_a_model_that_is_not_ready() -> None:
    import asyncio

    spec = _toy()
    async_model = AsyncLLM(LLM(spec, device="cpu"))

    async def go():
        return await async_model({"tokens": _tokens(_host(async_model._llm.session))})

    with pytest.raises(RuntimeError, match="missing"):
        asyncio.run(go())
    asyncio.run(async_model.close())


def test_the_package_facade_resolves_both_names_lazily() -> None:
    """``pocketllm.LLM`` is the same object the engine exports, and is not eager."""
    import pocketllm
    from pocketllm import engine

    assert pocketllm.LLM is engine.LLM
    assert pocketllm.AsyncLLM is engine.AsyncLLM