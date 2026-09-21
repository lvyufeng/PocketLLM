"""The V4.1 generation loop's contract with the model it records graphs on.

`tests/test_v41_backend.py` covers what the *serving adapter* does with a `Generation`, including
that it hands the driver back when the request finishes. What is here is the loop's own half of that
contract, and the case the adapter cannot create: a run that is *left* rather than finished. The
serving adapter's cancellation and stop-string paths both unwind out of `on_token`, so the loop is
abandoned mid-step with a driver it will never return to anybody.

`_decode_graphs` says what leaving it installed costs -- `Block.forward` hands every forward to
`block.decode_graph`, whose sink the first real pass sized at one row, so the next prompt dies on
`_put`'s `copy_` with "output with shape [1, 1, 4, 5120] doesn't match the broadcast shape
[1, 1364, 4, 5120]". These tests pin that an unwound run installs nothing: on both the raising
callback and the failure inside the step.

`Pos` and the model are the only things faked. The loop, the pick, the capture ordering and the
release are the released code.
"""

from __future__ import annotations

import torch

from src.models.deepseek_v4_1 import generate as generate_module
from src.models.deepseek_v4_1 import graphs as graphs_module
from src.models.deepseek_v4_1.generate import generate

DIM = 8
PROMPT = [1, 2, 3]


class FakeGraphs:
    """Stands in for `graphs.DecodeGraphs`, and only records what the loop does to it.

    The real one captures CUDA graphs, which needs a card; what the loop's contract with it is --
    built once, installed on the model, released if the loop is left early -- needs none.
    """

    built: list["FakeGraphs"] = []

    def __init__(self, model, layer_ids=None, warmup=None) -> None:
        self.model = model
        self.released = 0
        FakeGraphs.built.append(self)

    def capture_pass(self, forward) -> list:
        """One forward recorded. The real pass snapshots and restores the caches around it; the
        loop does not read its return value, so nothing here has to survive it."""
        forward()
        return []

    def release(self) -> None:
        self.released += 1
        self.model.decode_graph_installed = False


class FakeBackbone:
    """A backbone with just the surface the loop touches: a call, a reset, a limit, and a cache."""

    def __init__(self, *, max_seq_len=64) -> None:
        self.max_seq_len = max_seq_len
        self.decode_graph_installed = False
        # `_cache_device` reads the position's card off a buffer, so the name is the interface.
        self._buffers = {"layers.0.window_kv_cache": torch.zeros(1)}

    def named_buffers(self):
        return list(self._buffers.items())

    def reset_state(self, batch: int) -> None:
        self.reset_batch = batch

    def __call__(self, tokens, position, chunk=None):
        width = tokens.shape[-1]
        # One row of logits per token: the prefill's last row is the first pick, and the loop's own
        # count is `len(result.tokens)` against `max_new_tokens`, so the shape only has to line up.
        logits = torch.zeros(1 if width == 1 else width, DIM)
        logits[:, 0] = 1.0
        return None, logits, None


def _loop(monkeypatch):
    FakeGraphs.built = []
    monkeypatch.setattr(graphs_module, "DecodeGraphs", FakeGraphs)
    return FakeBackbone()


def test_an_unwinding_callback_leaves_the_model_without_its_graphs(monkeypatch) -> None:
    """The serving adapter raises out of `on_token` to end a request. Nobody gets the driver."""
    back = _loop(monkeypatch)
    seen: list[int] = []

    def stop_at_the_second_token(token, _logits):
        # The first call is the prefill's pick, which is before any graph exists; the second is the
        # first token the replay produced, which is the one a cancelling caller would be watching.
        seen.append(token)
        if len(seen) == 2:
            raise RuntimeError("cancelled")

    try:
        generate(back, PROMPT, max_new_tokens=8, graphs=True, on_token=stop_at_the_second_token)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the callback's failure did not reach the caller")

    driver = FakeGraphs.built[0]
    assert driver.released == 1, "an abandoned run left its recording installed on the blocks"


def test_a_step_that_fails_leaves_the_model_without_its_graphs(monkeypatch) -> None:
    """The same for a failure inside the replay: the loop never reaches a `return`."""
    back = _loop(monkeypatch)
    calls = {"n": 0}
    real = back.__class__.__call__

    def failing(self, tokens, position, chunk=None):
        # The prefill and the capture pass are allowed through; the first replayed step is not.
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("the replay died")
        return real(self, tokens, position, chunk)

    monkeypatch.setattr(back.__class__, "__call__", failing)

    try:
        generate(back, PROMPT, max_new_tokens=8, graphs=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the step's failure did not reach the caller")

    assert FakeGraphs.built[0].released == 1


def test_a_generation_that_finishes_hands_its_graphs_to_the_caller(monkeypatch) -> None:
    """The other half of the contract: a run that returns keeps them installed, so the caller --
    which is the adapter, in a service -- is the one that releases. `Generation.driver` says so."""
    back = _loop(monkeypatch)

    result = generate(back, PROMPT, max_new_tokens=2, graphs=True)

    driver = FakeGraphs.built[0]
    assert result.driver is driver
    assert driver.released == 0, "the loop released a driver it also handed back"


def test_a_generation_that_stops_on_the_prefill_never_builds_a_graph(monkeypatch) -> None:
    """The first token comes off the prefill's logits, so an `eos` there ends the run before any
    capture -- and `Generation.driver` documents `None` for exactly this case."""
    back = _loop(monkeypatch)

    result = generate(back, PROMPT, max_new_tokens=4, graphs=True, eos_token_id=0)

    assert FakeGraphs.built == []
    assert result.driver is None
    assert result.tokens == [0]
    assert result.stopped == "eos"


def test_the_eager_loop_builds_no_graph_at_all(monkeypatch) -> None:
    """`graphs=False` is the loop `_decode` runs, and it must not reach for a card to put a
    position on: this backbone's cache buffer is on the host."""
    back = _loop(monkeypatch)

    result = generate(back, PROMPT, max_new_tokens=2)

    assert FakeGraphs.built == []
    assert result.driver is None
    assert result.tokens == [0, 0]
    assert result.stopped == "length"
