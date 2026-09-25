"""Is a restored prefix the state the same prefix would have built from scratch?

A prefix store is only sound if a *resume* is the same forward as a cold prefill cut at the same
place, and for this model that claim has two halves that are not the same kind of claim.

The first is exact and is about the *copy*: a snapshot carries a global layer's leading rows and a
windowed layer's whole ring, and putting them back has to reproduce those bytes and the layer's write
head. That is asserted with `torch.equal`, because a copy that is a rounding away from its source is
a bug and not a tolerance.

The second is not exact and the reason is worth writing down rather than hiding behind a loose
number: `tests/test_models_mimo_v2_device_attention.py::test_a_chunked_prefill_is_a_one_shot_prefill`
holds a chunked prefill to `atol=1e-4` against the one-shot, because the attention's gemms change
shape with a chunk and the float sums reassociate. A resume *is* a chunk boundary at the stored
length -- generally not a multiple of the chunk the cold path would have used -- so the tokens past
it get the continuation's chunking and not the one-shot's. The tests here therefore compare against a
cold prefill at a relative tolerance, and the claim they are pinning is that the state is right: a
wrong key, a wrong slot or a stale row moves the answer by orders more than the rearrangement does.
The test that separates the two is `test_the_tail_a_snapshot_does_not_carry_is_never_read`, which
poisons the rows a snapshot leaves behind and demands the continuation come out *bit-identical* --
the only way that can hold is if the tail was never read at all.

The other thing here is that the write head is state. A restore that put the buffers back and left
the head where the previous request stopped would answer a continuation out of a span the snapshot
never described, and the failure would look like a model bug several tokens later rather than like a
cache bug. `set_written` is the only writer and the snapshot carries it per layer.

The last section is the adapter: that the store is built, reported in `capabilities`, counted in
`/metrics`, and that a repeat's reuse reaches a client as `usage.prompt_tokens_details.cached_tokens`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams  # noqa: E402
from pocketllm.backends.mimo_backend import MimoBackend  # noqa: E402
from src.models.mimo_v2.device_attention import MimoV2KVCache  # noqa: E402
from src.models.mimo_v2.device_model import MimoV2DeviceModel  # noqa: E402
from src.models.mimo_v2.generate import generate  # noqa: E402
from src.models.mimo_v2.prefix_cache import (  # noqa: E402
    WRITTEN,
    PrefixCache,
    geometry_tag,
    layer_key,
    layer_value,
    restore_rows,
    snapshot_rows,
)
from tests.test_mimo_serving import FakeTokenizer  # noqa: E402
from tests.test_models_mimo_v2_device_model import (  # noqa: E402
    DEVICE,
    SyntheticCheckpoint,
    SyntheticSource,
    synthetic_tensors,
    tiny_config,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

#: The miniature's two layers are one of each family -- layer 0 global, layer 1 windowed -- which is
#: the whole reason the snapshot rule has two branches in it. The window is four slots, so a prompt
#: of a dozen tokens wraps it and the ring's live rows are not its leading ones.
WINDOW = 4
CAPACITY = 64

#: Twelve tokens, all inside the miniature's vocabulary of 64.
PROMPT = [3, 17, 42, 5, 8, 11, 2, 9, 4, 6, 7, 1]

#: What a resume's chunking is allowed to cost the answer, as a fraction of the answer's own peak.
#: The miniature's logits run to a few units and the reassociation over two layers is at the 1e-6
#: level; the bound is two orders above what a measurement shows and four below what a wrong slot
#: produces, which is the gap the number has to sit in rather than be tight to.
ROUNDING = 1e-3


def stack(*, layers=None, capacity: int = CAPACITY, **kwargs) -> tuple[MimoV2DeviceModel, object]:
    """A miniature model on the card and a cache sized for it.

    `chunk_rows=0` is the whole expert share in one band: without it `mlp` dispatches a multi-row
    call to a chunk arena that does not exist, and every prefill here is multi-row. `deal='id'` is
    the deal a chunk needs, for the reason `tests/test_models_mimo_v2_prefill.py` records.
    """
    config = tiny_config(layers=2)
    source = SyntheticSource()
    checkpoint = SyntheticCheckpoint(config, synthetic_tensors(config, source))
    model = MimoV2DeviceModel(
        checkpoint,
        device=DEVICE,
        dtype=torch.float32,
        layers=layers,
        expert_source=source,
        pin=False,
        deal="id",
        chunk_rows=0,
        **kwargs,
    )
    return model, model.cache(capacity)


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The largest disagreement between two rows, as a fraction of the expected row's own peak."""
    return float((actual - expected).abs().max() / expected.abs().max())


def cold(model, ids: list[int], *, chunk: int, capacity: int = CAPACITY) -> torch.Tensor:
    """A prompt forward-passed from zero, which is what a miss is and what a resume is measured on."""
    cache = model.cache(capacity)
    return model.prefill(ids, cache=cache, chunk=chunk)


def resumed(
    model,
    ids: list[int],
    cut: int,
    *,
    chunk: int,
    capacity: int = CAPACITY,
    poison: bool = False,
) -> torch.Tensor:
    """Prefill to `cut`, snapshot, wipe, restore, and forward the rest at its own position.

    The wipe between the halves is not a test artefact: `generate` resets the cache first for the
    same reason, and what a restore has to survive is a cache whose rows above the stored prefix hold
    the *previous* request's tokens. `poison` is that failure made deliberate -- the global layer's
    tail is filled with a value no forward of this prompt could have produced -- so an arm that moves
    is an arm that read a row the snapshot did not carry.
    """
    cache = model.cache(capacity)
    model.prefill(ids[:cut], cache=cache, chunk=cut)
    saved = snapshot_rows(cache)

    cache.reset()
    if poison:
        for layer in cache.layers:
            key, value = cache.key_buffer(layer), cache.value_buffer(layer)
            key[:, cut:].fill_(float("nan"))
            value[:, cut:].fill_(float("nan"))
    restore_rows(cache, saved)
    assert all(cache.written(layer) == cut for layer in cache.layers), "the write head did not go back"
    return model.prefill(ids[cut:], cache=cache, chunk=chunk, start_pos=cut)


# ---------------------------------------------------------------------------
# What a snapshot carries
# ---------------------------------------------------------------------------


@needs_cuda
def test_a_snapshot_cuts_a_global_layer_to_its_prefix_and_stores_a_ring_whole():
    """The two branches, read off the miniature's own geometry rather than off a byte count.

    Twelve tokens into a four-slot ring is a wrapped ring: its live positions are slots 0..3 holding
    positions 8..11, so there is no leading run of rows to cut and the whole buffer goes. The global
    layer is the opposite -- positions 0..11 in rows 0..11 and nothing above them -- and that cut is
    the whole reason the store is worth having at 262144, where the uncut buffer is 1.51 GiB a rank.
    """
    model, cache = stack()
    model.prefill(PROMPT, cache=cache, chunk=len(PROMPT))
    saved = snapshot_rows(cache)

    assert set(saved) == {
        layer_key(0), layer_value(0), layer_key(1), layer_value(1), WRITTEN,
    }
    # Layer 0 is the global one: four key heads, cut to the twelve positions a prefix wrote.
    assert saved[layer_key(0)].shape == (4, len(PROMPT), 16)
    assert saved[layer_value(0)].shape == (4, len(PROMPT), 8)
    # Layer 1 is the ring: eight heads over its four slots, whatever the prompt is.
    assert saved[layer_key(1)].shape == (8, WINDOW, 16)
    assert saved[layer_value(1)].shape == (8, WINDOW, 8)
    assert torch.equal(saved[WRITTEN], torch.tensor([len(PROMPT), len(PROMPT)], dtype=torch.int64))
    # And every one of them is on the host, because the store outlives the request while the card is
    # holding the cache, the expert band and whatever a deployment asked to keep resident.
    assert all(value.device.type == "cpu" for value in saved.values())


@needs_cuda
def test_a_snapshot_is_the_bytes_that_are_there_and_a_restore_is_the_bytes_back():
    """The copy half, exactly: what a restore puts in the buffers is what the snapshot took out.

    `torch.equal` and not a tolerance, because this comparison has no arithmetic on either side. It
    is not a tautology either -- the guarantee is that the *cut* is the state, and a snapshot that cut
    a windowed layer's ring to `min(written, slots)` rows, or a global layer's to the ring's width,
    would satisfy every shape assertion in the test above and fail here.
    """
    model, cache = stack()
    model.prefill(PROMPT, cache=cache, chunk=len(PROMPT))
    saved = snapshot_rows(cache)

    cache.reset()
    for layer in cache.layers:
        cache.key_buffer(layer).zero_()
        cache.value_buffer(layer).zero_()
    restore_rows(cache, saved)

    for layer in cache.layers:
        rows = saved[layer_key(layer)].shape[1]
        assert torch.equal(cache.key_buffer(layer)[:, :rows], saved[layer_key(layer)].to(DEVICE))
        assert torch.equal(cache.value_buffer(layer)[:, :rows], saved[layer_value(layer)].to(DEVICE))
    assert [cache.written(layer) for layer in cache.layers] == [len(PROMPT), len(PROMPT)]


@needs_cuda
def test_the_tail_a_snapshot_does_not_carry_is_never_read():
    """A poisoned tail, and a continuation that comes out bit-identical anyway.

    A snapshot of a global layer carries its first `p` rows and leaves the rest holding the previous
    request's tokens -- that is the price of not zeroing a gigabyte, and the argument for why it is
    free is that every read is bounded by the layer's write head, which the snapshot restores. The
    arm here is that argument turned into a measurement: fill the tail with NaN, restore, continue,
    and require the same bits as an unpoisoned arm. A NaN that reached the attention would not be a
    small difference; it would be a row of NaN, and the equality below is what says it never got
    there.
    """
    ids = PROMPT + [12, 13, 14, 15]
    cut = len(PROMPT)

    clean = resumed(stack()[0], ids, cut, chunk=len(ids) - cut)
    poisoned = resumed(stack()[0], ids, cut, chunk=len(ids) - cut, poison=True)
    assert torch.equal(clean, poisoned), "the tail was read"
    assert not bool(torch.isnan(poisoned).any())


@needs_cuda
@pytest.mark.parametrize("cut", (1, 3, 4, 7, 11))
def test_a_resumed_prefill_is_the_cold_prefill_cut_at_the_stored_length(cut: int):
    """Every split point, including the ones a chunking would not have chosen.

    `cut=4` is the window, so a restore lands on the ring's own boundary; the odd ones cut a chunk in
    half, which is the case a resume actually is -- a chat turn's stored length is not a multiple of
    the chunk width. The tolerance is the chunked-prefill tolerance and not a looser one: the state
    is either the state or it is not, and the rearrangement of the sums is the only thing allowed to
    move the answer.
    """
    model, _ = stack()
    whole = cold(model, PROMPT, chunk=5)
    part = resumed(model, PROMPT, cut, chunk=5)
    moved = relative(part, whole)
    assert moved < ROUNDING, f"cut {cut}: the row moved by {moved:.3e} of its own peak"


@needs_cuda
def test_a_resume_past_the_window_is_not_a_slice_of_the_ring():
    """A prompt deeper than the window, resumed: the ring has wrapped and the restore has to know it.

    The window here is four slots and the cut is eleven positions in, so the positions a continuation
    reads are eight through ten -- slots 0 to 2 of the ring -- and the positions four through seven
    that a reader might have expected to find there are gone. A store that kept the ring's *leading*
    `min(written, slots)` rows would pass every test above and answer this one out of the wrong
    window, which is the kind of wrong that produces fluent text.
    """
    model, _ = stack()
    whole = cold(model, PROMPT, chunk=6)
    part = resumed(model, PROMPT, 11, chunk=6)
    moved = relative(part, whole)
    assert moved < ROUNDING, f"the row moved by {moved:.3e} of its own peak"


@needs_cuda
def test_a_resume_continues_a_prompt_that_wrapped_its_window_further():
    """Two flushes of the ring rather than one, so the ring's phase is not the snapshot's own."""
    ids = PROMPT + PROMPT
    model, _ = stack()
    whole = cold(model, ids, chunk=8, capacity=len(ids) + 2)
    part = resumed(model, ids, len(PROMPT), chunk=8, capacity=len(ids) + 2)
    moved = relative(part, whole)
    assert moved < ROUNDING, f"the row moved by {moved:.3e} of its own peak"


@needs_cuda
def test_the_geometry_is_mixed_into_every_key():
    """A chain is only meaningful inside the layout it was taken in: the rank's head count, the
    context, and the world."""
    model, cache = stack()
    single = geometry_tag(cache, 1, CAPACITY)
    assert single != geometry_tag(cache, 4, CAPACITY)
    assert single != geometry_tag(cache, 1, CAPACITY * 2)

    # A split attention gives every buffer a share of the heads, so a store taken with the whole
    # attention on every rank cannot be read at four -- and the tag is what refuses it rather than
    # the shapes, which would only fail once something was already being copied. Built as a cache
    # rather than as a whole model: the split is a property of the buffers, and the buffers are what
    # the tag reads.
    split = MimoV2KVCache(model.config, CAPACITY, [0, 1], device=DEVICE, shards=4, shard=0)
    assert split.key_buffer(0).shape[0] < cache.key_buffer(0).shape[0]
    assert geometry_tag(split, 4, CAPACITY) != geometry_tag(cache, 4, CAPACITY)


@needs_cuda
def test_every_rank_of_a_split_attends_over_the_same_number_of_heads():
    """Why a per-rank store agrees without a collective: the split is even, so the budget bites alike.

    Nothing is broadcast to keep four ranks restoring the same prefix -- the argument is that the key
    is the prompt's tokens, the shape options are the launcher's, and the *size of a snapshot* is the
    same number on every rank, so the eviction order is the same sequence everywhere. That last clause
    is the one with an `if` in it, and this is the `if`: a split that left one rank a head more than
    another would make its entries bigger, its budget evict sooner, and the ranks would then resume
    different requests -- which, given that a layer's `all_reduce` is entered once per forward, is a
    hang rather than a wrong answer.
    """
    model, _ = stack()
    caches = [
        MimoV2KVCache(model.config, CAPACITY, [0, 1], device=DEVICE, shards=4, shard=rank)
        for rank in range(4)
    ]
    per_rank = [
        sum(tensor.numel() * tensor.element_size() for tensor in snapshot_rows(cache).values())
        for cache in caches
    ]
    assert len(set(per_rank)) == 1, per_rank
    # And the shapes that decide it: the windowed family has twice the key heads of the global one,
    # so a quarter of it is not a quarter of the same number.
    assert {cache.key_buffer(1).shape[0] for cache in caches} == {2}


# ---------------------------------------------------------------------------
# The loop the store plugs into
# ---------------------------------------------------------------------------


@needs_cuda
def test_an_exact_repeat_forwards_nothing_and_leaves_the_store_alone(monkeypatch):
    """The prompt *is* the stored prefix: a restore and a sample, and no forward at all.

    The row the first new token comes from was computed when the anchor was taken and rides with the
    entry, which is what makes this legal; the cheap-looking alternative, restoring the state and
    forwarding the last token again at its own position, re-enters the ring's own recurrence for a
    position the restored state has already counted. `Entry.logits` is the whole of why that rule is
    not needed, and this test is that it is actually taken: the second request's prefill is counted
    and has to be zero.
    """
    model, cache = stack()
    store = PrefixCache(budget_bytes=1 << 20, max_seq_len=CAPACITY, min_tokens=4)
    first = generate(model, PROMPT, max_new_tokens=4, cache=cache, prefix_cache=store, chunk=6)

    calls: list[tuple[int, int]] = []
    original = model.prefill

    def counting(ids, **kwargs):
        calls.append((len(ids), int(kwargs.get("start_pos", 0))))
        return original(ids, **kwargs)

    monkeypatch.setattr(model, "prefill", counting)
    second = generate(model, PROMPT, max_new_tokens=4, cache=cache, prefix_cache=store, chunk=6)

    assert calls == [], f"an exact repeat forwarded {calls}"
    assert second.cached_tokens == len(PROMPT) and second.prompt_tokens == len(PROMPT)
    assert second.tokens == first.tokens


@needs_cuda
def test_a_longer_prompt_forwards_only_the_remainder(monkeypatch):
    """The conversational case: the next turn is the previous prompt plus the answer plus a question.

    What is asserted is the *shape* of the second request's forward -- the ids it was given and the
    position they go in at -- because that is the store's whole contract with the loop. The position
    is the half that is easy to get wrong: a remainder forwarded from zero is the model reading a
    prompt it was never given, which produces an answer rather than an error.
    """
    model, cache = stack()
    store = PrefixCache(budget_bytes=1 << 20, max_seq_len=CAPACITY, min_tokens=4)
    generate(model, PROMPT, max_new_tokens=2, cache=cache, prefix_cache=store, chunk=6)

    calls: list[tuple[int, int]] = []
    original = model.prefill

    def counting(ids, **kwargs):
        calls.append((len([int(token) for token in ids]), int(kwargs.get("start_pos", 0))))
        return original(ids, **kwargs)

    monkeypatch.setattr(model, "prefill", counting)
    longer = PROMPT + [9, 9]
    second = generate(model, longer, max_new_tokens=2, cache=cache, prefix_cache=store, chunk=6)

    assert calls == [(2, len(PROMPT))], calls
    assert second.cached_tokens == len(PROMPT)


@needs_cuda
def test_the_head_anchor_serves_a_shared_header(monkeypatch):
    """Two conversations under one rendered header: the head hits where the end cannot.

    The header is eight tokens and the two prompts diverge after it, so the end anchors are useless
    to each other -- which is the case the anchor exists for, and the case a chat service is in when
    a client sends the same system message twice with different questions.
    """
    head = 8
    model, cache = stack()
    store = PrefixCache(budget_bytes=1 << 20, max_seq_len=CAPACITY, min_tokens=4, head_tokens=head)
    first = PROMPT[:head] + [20, 21, 22, 23]
    second = PROMPT[:head] + [30, 31, 32, 33]
    generate(model, first, max_new_tokens=2, cache=cache, prefix_cache=store, chunk=6)

    calls: list[tuple[int, int]] = []
    original = model.prefill

    def counting(ids, **kwargs):
        calls.append((len([int(token) for token in ids]), int(kwargs.get("start_pos", 0))))
        return original(ids, **kwargs)

    monkeypatch.setattr(model, "prefill", counting)
    other = generate(model, second, max_new_tokens=2, cache=cache, prefix_cache=store, chunk=6)

    assert other.cached_tokens == head
    # The header is stored on the way past -- one forward of it, which is the chunk boundary the
    # anchor costs a cold prefill -- and the prompt is stored at its end. A hit at the head resumes.
    assert calls and calls[0] == (len(second) - head, head)
    # And the anchor's own row is the row the prompt carries there, so a request of exactly the
    # header's length is a lookup and a sample rather than a forward that recomputes a row the
    # anchor already produced.
    exact = generate(model, PROMPT[:head], max_new_tokens=1, cache=cache, prefix_cache=store, chunk=6)
    assert exact.cached_tokens == head


@needs_cuda
def test_a_prompt_that_is_not_in_the_store_is_the_cold_prefill_it_always_was():
    """A miss is not a code path of its own: the same prompt, the same row, with the store in place."""
    model, cache = stack()
    store = PrefixCache(budget_bytes=1 << 16, max_seq_len=CAPACITY, min_tokens=4)
    plain = generate(model, PROMPT, max_new_tokens=3, cache=cache)
    with_store = generate(model, PROMPT, max_new_tokens=3, cache=cache, prefix_cache=store, chunk=6)
    assert with_store.cached_tokens == 0
    assert with_store.tokens == plain.tokens


@needs_cuda
def test_an_entry_that_does_not_fit_the_budget_is_dropped_and_not_reported():
    """A store with no room is a store that stores nothing, and the loop goes on serving."""
    model, cache = stack()
    store = PrefixCache(budget_bytes=1, max_seq_len=CAPACITY, min_tokens=4)
    generation = generate(model, PROMPT, max_new_tokens=2, cache=cache, prefix_cache=store, chunk=6)
    assert generation.cached_tokens == 0
    assert len(store) == 0 and store.bytes == 0
    assert store.stats()["misses"] == 0, "a store with nothing in it was never asked"


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


#: The adapter's prompt, and the reason it is longer than the one above: the store's floor is a whole
#: hash block by default -- it is what an entry's fixed tables cost whatever its length, and a store
#: of eight tokens would spend them on a ring -- so a request the served path would actually cache is
#: one of at least sixty-four. Everything else here runs the store at a floor of four so a test can
#: work on a prompt short enough to read.
LONG_PROMPT = [(index * 29 + 7) % 63 for index in range(80)]


def adapter(*, max_model_len: int = 2 * CAPACITY, **options) -> tuple[MimoBackend, object]:
    """A MiMo adapter whose model is the miniature, so the request path runs for real.

    The model is injected rather than loaded, which is what keeps the 149.81 GiB expert bank out of a
    unit test; everything below the injection point -- the cache, the store, the loop -- is the
    served code.
    """
    model, _ = stack()
    args = EngineArgs(
        model="a-mimo-checkpoint",
        backend="mimo",
        max_model_len=max_model_len,
        backend_options=dict(options),
    )
    return MimoBackend(args, loader=lambda _args, _options: model, tokenizer=FakeTokenizer()), model


def ask(backend: MimoBackend, ids: list[int], budget: int = 2) -> object:
    request = GenerationRequest(
        prompt_tokens=list(ids), sampling_params=SamplingParams(max_tokens=budget)
    )
    return backend.generate([request])[0]


@needs_cuda
def test_the_adapter_builds_the_store_reports_it_and_prices_a_repeat():
    """The wiring a client can see: the switch, the metric, and the usage field.

    All three are the parity bar with the V4.1 adapter, and each one is a thing a deployment decides
    on: `supports_prefix_caching` is what a load balancer reads, the `prefix_cache_*` series is what
    an operator watches to know whether the budget is the right size, and `cached_tokens` is what a
    client is billed on.
    """
    backend, model = adapter()
    backend.prepare()
    capabilities = backend.capabilities
    assert capabilities.supports_prefix_caching is True
    assert capabilities.supports_batch is False
    assert capabilities.details["prefix_cache_bytes"] == 4 << 30
    assert capabilities.details["prefix_cache_head_tokens"] == 1024

    cold = ask(backend, LONG_PROMPT)
    assert cold.usage.cached_tokens == 0
    assert cold.usage.prompt_tokens == len(LONG_PROMPT)
    assert "prompt_tokens_details" not in cold.usage.as_dict()

    warm = ask(backend, LONG_PROMPT)
    assert warm.usage.cached_tokens == len(LONG_PROMPT)
    assert warm.usage.as_dict()["prompt_tokens_details"] == {"cached_tokens": len(LONG_PROMPT)}
    assert warm.token_ids == cold.token_ids, "the reuse changed the answer"

    metrics = backend.metrics()
    assert metrics["prefix_cache_hits_total"] == 1
    assert metrics["prefix_cache_reused_tokens_total"] == len(LONG_PROMPT)
    assert metrics["prefix_cache_entries"] >= 1
    assert metrics["prefix_cache_budget_bytes"] > 0
    assert 0 < metrics["prefix_cache_bytes"] <= metrics["prefix_cache_budget_bytes"]


@needs_cuda
def test_the_adapter_leaves_the_store_off_when_the_launcher_says_so():
    """`--enable-prefix-caching` is the switch, and a disabled store is not a store that misses."""
    backend, _ = adapter(prefix_cache_bytes=0)
    backend.prepare()
    assert backend.capabilities.supports_prefix_caching is False
    ask(backend, LONG_PROMPT)
    ask(backend, LONG_PROMPT)
    assert backend.metrics().get("prefix_cache_hits_total") is None


@needs_cuda
def test_the_adapter_reports_no_store_over_a_cache_it_cannot_snapshot():
    """A stand-in cache is not a cache, and the run says so rather than reporting false misses.

    This is the case every test in `tests/test_mimo_serving.py` is in: a scripted model whose cache
    holds a capacity, a reset and a memory figure, and no buffers. A store built over it would
    *work* -- the walk it does would find nothing and every request would miss -- and a deployment
    reading `supports_prefix_caching` would size a budget for a cache that never fills.
    """
    from tests.test_mimo_serving import ScriptedModel

    args = EngineArgs(
        model="a-mimo-checkpoint", backend="mimo", max_model_len=2 * CAPACITY
    )
    backend = MimoBackend(
        args, loader=lambda _args, _options: ScriptedModel(), tokenizer=FakeTokenizer()
    )
    backend.prepare()
    assert backend.capabilities.supports_prefix_caching is False
    assert "describes no state" in backend.capabilities.details["prefix_cache"]


@needs_cuda
def test_the_adapter_takes_a_byte_budget_with_a_suffix():
    """`4g` and `4294967296` are one budget, and a negative one is a refusal at startup."""
    from pocketllm.api import ConfigurationError

    suffixed, _ = adapter(prefix_cache_bytes="1m")
    suffixed.prepare()
    plain, _ = adapter(prefix_cache_bytes=1 << 20)
    plain.prepare()
    assert suffixed.capabilities.details["prefix_cache_bytes"] == 1 << 20
    assert plain.capabilities.details["prefix_cache_bytes"] == 1 << 20

    with pytest.raises(ConfigurationError, match="must not be negative"):
        adapter(prefix_cache_bytes="-1")

