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
release are the released code, and so is the prefix store: `PrefixCache` needs no card either, so the
tests below drive the loop against the real one. What that buys is the assertion a fake could not
make -- that the key `_prefill` stores under and the key it looks up are the same key -- and what it
costs is a `min_tokens` small enough for `PROMPT`'s three tokens to clear.
"""

from __future__ import annotations

import torch

from src.models.deepseek_v4_1 import generate as generate_module
from src.models.deepseek_v4_1 import graphs as graphs_module
from src.models.deepseek_v4_1 import prefix_cache as prefix_cache_module
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
    """A backbone with the surface the loop touches: a call, a reset, a cache, and the two prefix
    hooks `LoadedBackbone` answers with the Engram hash slices that ride beside the tree's buffers."""

    def __init__(self, *, max_seq_len=64) -> None:
        self.max_seq_len = max_seq_len
        self.decode_graph_installed = False
        # `_cache_device` reads the position's card off a buffer, so the name is the interface.
        self._buffers = {"layers.0.window_kv_cache": torch.zeros(1)}
        # `(tokens, position, chunk)` per forward, in order: how a test says which forwards a prompt
        # cost, and the whole of the claim a stored prefix makes.
        self.calls: list[tuple[list[int], int, object]] = []
        # The Engram slices the store handed back, and the tree's buffer as it stood when the second
        # half of a restore ran -- see `restore_prefix`.
        self.restored: list[torch.Tensor] = []
        self.buffer_at_restore: list[float] = []

    def named_buffers(self):
        return list(self._buffers.items())

    def reset_state(self, batch: int) -> None:
        self.reset_batch = batch
        # The real one zeroes the position tables. A marker no forward writes is what lets a test
        # tell a buffer a restore reached from one the reset overwrote after it.
        self._buffers["layers.0.window_kv_cache"].fill_(-1.0)

    def snapshot_prefix(self, limit: int):
        """The hash slice a prefill keeps, one row per position.

        `None` is the other answer a backbone gives here -- a config with no Engram layers, which is
        `LoadedBackbone.snapshot_prefix`'s documented case and `tests/test_models_deepseek_v4_1_loader.py`'s
        subject -- and a prefill that gets it stores the tree's buffers alone.
        """
        return torch.arange(int(limit), dtype=torch.float32).reshape(1, -1)

    def restore_prefix(self, saved) -> None:
        """Recorded rather than applied: what a test reads here is the *tree's* buffer.

        `_restore` writes the tree's buffers first and reaches this second, and `_prefill` resets
        before either, so a buffer already holding the snapshot when this runs is the ordering
        `_prefill` documents. One still holding the reset's marker means the reset came after the
        restore, which is a hit whose first new token reads caches that are half the last request's.
        """
        self.restored.append(saved)
        self.buffer_at_restore.append(float(self._buffers["layers.0.window_kv_cache"].item()))

    def __call__(self, tokens, position, chunk=None):
        ids = [int(token) for token in tokens.reshape(-1).tolist()]
        self.calls.append((ids, position, chunk))
        # `position + width` is a value no other call produces, so a buffer that ends up holding it
        # is a buffer this forward -- and not a restore -- wrote.
        self._buffers["layers.0.window_kv_cache"].fill_(float(position + len(ids)))
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


def _cache(head_tokens: int = 0) -> prefix_cache_module.PrefixCache:
    """A real store over a prompt short enough to reason about.

    `min_tokens=1` because the store's default floor is a whole hash block -- 64 tokens, the length
    the chain can key in one step -- and `PROMPT` is three; the floor's own behavior is
    `tests/test_models_deepseek_v4_1_prefix_cache.py`'s. The budget is a megabyte because a toy
    payload is a handful of bytes and the eviction walk is not what these tests are about.
    """
    return prefix_cache_module.PrefixCache(
        budget_bytes=1 << 20, max_seq_len=64, min_tokens=1, head_tokens=head_tokens
    )


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


def test_a_cold_prompt_is_forwarded_whole_and_a_repeat_of_it_is_not(monkeypatch) -> None:
    """The two ends of the store's contract: a miss costs exactly what it always cost, and a prompt
    that *is* a stored prefix costs no prompt forward at all.

    The trailing `([0], 3, None)` in both lists is the loop's own step, not a prompt: `_decode` picks
    a token and then forwards it to have a distribution for the next one, whether or not the pick's
    row came out of a forward. What the repeat is missing is the three-token call.
    """
    back = _loop(monkeypatch)
    cache = _cache()

    first = generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)
    assert back.calls == [(PROMPT, 0, None), ([0], 3, None)]
    assert first.cached_tokens == 0, "a cold prompt reported reuse it did not have"
    assert cache.lengths() == [3], "the prefill did not keep the state it just built"

    back.calls.clear()
    second = generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)
    assert back.calls == [([0], 3, None)], "the repeat forwarded the prompt again"
    assert second.cached_tokens == 3
    assert second.tokens == [0]
    assert second.prompt_tokens == 3


def test_a_longer_prompt_resumes_from_the_stored_end_and_forwards_only_the_tail(monkeypatch) -> None:
    """The conversational case: turn N's prompt is turn N-1's plus more, so the shared part is the
    length the store was keyed at and the forward starts there rather than at zero.

    `position=3` is the load-bearing half of the assertion. A tail forwarded at zero would run the
    prefill bodies over one token and produce a row for the wrong position entirely.
    """
    back = _loop(monkeypatch)
    cache = _cache()
    generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)

    back.calls.clear()
    longer = PROMPT + [9]
    result = generate(back, longer, max_new_tokens=1, prefix_cache=cache)

    assert back.calls == [([9], 3, None), ([0], 4, None)]
    assert result.cached_tokens == 3
    assert result.prompt_tokens == 4


def test_the_head_anchor_costs_one_chunk_on_a_miss_and_resumes_the_next_prompt(
    monkeypatch,
) -> None:
    """A head anchor is a snapshot at a fixed length, so the cold prefill has to *end* a chunk there.

    That boundary is the anchor's whole cost and it is paid once, on the miss: the second prompt below
    shares only the first two tokens -- a different render of the same header, which is what the
    anchor is for -- and it resumes at 2 rather than forwarding both. Note that it does not resume at
    3: the store holds that length too, but for this prompt it is a different key.
    """
    back = _loop(monkeypatch)
    cache = _cache(head_tokens=2)

    cold = generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)
    assert back.calls == [([1, 2], 0, None), ([3], 2, None), ([0], 3, None)]
    assert cold.cached_tokens == 0
    assert cache.lengths() == [3, 2]

    back.calls.clear()
    other = generate(back, [1, 2, 99], max_new_tokens=1, prefix_cache=cache)

    assert back.calls == [([99], 2, None), ([0], 3, None)]
    assert other.cached_tokens == 2
    assert other.prompt_tokens == 3


def test_a_resume_restores_the_tree_and_the_hash_slice_after_the_reset(monkeypatch) -> None:
    """What a hit puts back, and in what order: the reset, then the tree's buffers, then the Engram
    slice that is not one of them.

    The buffer is read inside `restore_prefix`, which `_restore` reaches second, so `3.0` is the
    snapshot having already landed -- `-1.0` would be the reset running after the restore, and `99.0`
    the restore never running at all. The cold run's value comes from the prefill's own forward at
    position 0, so the two numbers agreeing is the round trip and not a coincidence.

    The hash slice is the other half: `snapshot_rows` cannot carry `EngramHashIds.cache` because it is
    not a registered buffer, so it travels under `HASH_CACHE` and is `restore_prefix`'s to put back.
    A continuation's first token reads the previous `max_ngram_size - 1` positions through it.
    """
    back = _loop(monkeypatch)
    cache = _cache()

    generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)
    assert back.buffer_at_restore == [], "a cold prefill restored something"

    generate(back, PROMPT, max_new_tokens=1, prefix_cache=cache)

    assert back.buffer_at_restore == [3.0], "the restore did not precede the hash slice"
    assert len(back.restored) == 1
    assert torch.equal(back.restored[0], torch.arange(3, dtype=torch.float32).reshape(1, -1)), (
        "the store handed back a hash slice that is not the one the prefill kept"
    )
