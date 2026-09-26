"""Xing4.0-29B-A4B's serving path: the adapter, the loop and the prefix store.

Nothing here reads the 17.84 GiB checkpoint.  The model is a scripted stand-in
that answers whatever the test tells it to, because what these tests are about is
the adapter's own contract -- which checkpoint it claims, which options it
refuses, what it stores between requests and what a cancel does -- and a test
that needed a 2080 Ti to check a capabilitity flag would be a test that never
runs in CI.  The three places where the real model matters are covered where they
belong: the parity tests in `tests/test_xing4_0_moe.py` and
`tests/test_xing4_0_hyper_connection.py`, and the served request recorded in
`docs/models/xing4.0-29b-a4b.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    SamplingParams,
    UnsupportedFeatureError,
)
from pocketllm.backends import factory
from pocketllm.backends.xing4_backend import (
    DEFAULT_PREFILL_CHUNK,
    Xing4Backend,
    _Options,
    resolve_paths,
)
from src.models.xing4_0.generate import Generation, generate, sample_token
from src.models.xing4_0.prefix_cache import LatentPrefixCache, restore, snapshot

# ---------------------------------------------------------------------------- stand-ins

VOCAB = 32
LAYERS = 2
WIDTH = 4


class FakeLatent:
    def __init__(self, capacity: int) -> None:
        self.latent = torch.zeros((1, capacity, WIDTH), dtype=torch.float16)
        self.capacity = int(capacity)
        self.length = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.latent.numel()) * self.latent.element_size()


class ScriptedModel:
    """A trunk that answers what the script says and records what it was asked.

    ``forward`` writes the row's own token id into the cache so that a restored
    prefix is observable: a run that resumed forwards only the remainder, and the
    rows it did not forward are the ones a store handed it.
    """

    def __init__(self, scripted=(11, 12, 13, 14)) -> None:
        self.scripted = list(scripted)
        self.blocks = [object()] * LAYERS
        # The released head count, because the adapter derives its prefill chunk
        # from it and the context it was given.
        self.params = SimpleNamespace(n_heads=32)
        self.forwards: list[tuple[list[int], int]] = []
        self.nbytes = 1234
        self._cursor = 0

    def make_cache(self, capacity: int, *, batch: int = 1):
        return [FakeLatent(capacity) for _ in range(LAYERS)]

    def _row(self, token: int) -> torch.Tensor:
        row = torch.full((1, VOCAB), -10.0)
        row[0, int(token) % VOCAB] = 10.0
        return row

    def forward(self, input_ids, *, cache=None, start_pos=0, absorbed=True):
        ids = [int(token) for token in input_ids]
        self.forwards.append((ids, int(start_pos)))
        if cache is not None:
            for index, layer in enumerate(cache):
                end = int(start_pos) + len(ids)
                layer.latent[0, int(start_pos) : end, 0] = torch.tensor(
                    [float(token) for token in ids], dtype=layer.latent.dtype
                )
                layer.length = max(layer.length, end)
        if len(ids) > 1:
            # A prefill hands back the row after the last prompt token, which is
            # the script's first entry; the cursor then points at the second.
            self._cursor = 1
            return self._row(self.scripted[0])
        row = self._row(self.scripted[min(self._cursor, len(self.scripted) - 1)])
        self._cursor += 1
        return row


class FakeTokenizer:
    def __init__(self, eos_token_id=2) -> None:
        self.eos_token_id = eos_token_id
        self.pieces = {0: "", 1: "", 2: "<|im_end|>", 3: "", 11: "a", 12: "b", 13: "c", 14: "d"}
        self.encoding: dict[str, list[int]] = {}

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(self.encoding.get(str(text), [7, 8, 9]))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return f"{body}<|assistant|>"

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(
            "" if skip_special_tokens and int(t) in {0, 1, 2} else self.pieces.get(int(t), "?")
            for t in ids
        )


def backend(*, model=None, tokenizer=None, **options) -> Xing4Backend:
    model = model if model is not None else ScriptedModel()
    tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
    args = EngineArgs(
        model="a-xing4-checkpoint",
        backend="xing4",
        max_model_len=64,
        backend_options={"gguf": "/nowhere/xing4_0-29b-IQ4_NL.gguf", **options},
    )
    instance = Xing4Backend(args, loader=lambda _path, _options: model, tokenizer=tokenizer)
    instance.prepare()
    return instance


def request(*, request_id="r1", messages=(("user", "hi"),), **params) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt="ignored when messages are present",
        metadata={"messages": [{"role": role, "content": text} for role, text in messages]},
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0, **params),
    )


# ---------------------------------------------------------------------------- the store


def test_the_store_restores_exactly_what_a_prefill_left() -> None:
    """A resume is not a recomputation: the rows are the rows, byte for byte."""
    cache = [FakeLatent(32) for _ in range(LAYERS)]
    for index, layer in enumerate(cache):
        layer.latent[0, :6] = torch.arange(6).reshape(1, 6, 1) + index
        layer.length = 6
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=32, n_layers=LAYERS)
    store.store([1, 2, 3, 4, 5, 6], snapshot(cache, 6))

    target = [FakeLatent(32) for _ in range(LAYERS)]
    assert store.longest([1, 2, 3, 4, 5, 6, 7, 8]) == 6
    restore(target, store.materialise([1, 2, 3, 4, 5, 6, 7, 8], 6), 6)
    for source, restored in zip(cache, target):
        assert restored.length == 6
        assert torch.equal(source.latent[:, :6], restored.latent[:, :6])


def test_the_store_matches_only_at_the_prompt_s_first_token() -> None:
    """A shared *suffix* is not a shared prefix and cannot be resumed from."""
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=32, n_layers=1)
    cache = [FakeLatent(32)]
    cache[0].latent[0, :4] = 1.0
    cache[0].length = 4
    store.store([9, 8, 7, 6], snapshot(cache, 4))
    assert store.longest([9, 8, 7, 6]) == 4
    assert store.longest([9, 8, 7, 6, 5]) == 4
    assert store.longest([5, 9, 8, 7, 6]) == 0
    assert store.longest([8, 7, 6]) == 0


def test_the_store_evicts_to_its_budget_and_keeps_the_newest() -> None:
    """The budget is bytes of cache state, and the least recently used goes first."""
    per_entry = LAYERS * 4 * WIDTH * 2  # four rows a layer, fp16
    store = LatentPrefixCache(budget_bytes=per_entry * 2, capacity=32, n_layers=LAYERS)
    for key in ([1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]):
        cache = [FakeLatent(32) for _ in range(LAYERS)]
        for layer in cache:
            layer.latent[0, :4] = 0.5
            layer.length = 4
        store.store(key, snapshot(cache, 4))
    stats = store.stats()
    assert stats["entries"] == 2
    assert stats["bytes"] <= store.budget_bytes
    assert store.longest([1, 1, 1, 1]) == 0, "the oldest was evicted"
    assert store.longest([2, 2, 2, 2]) == 4 and store.longest([3, 3, 3, 3]) == 4


def test_a_prefix_that_could_not_be_stored_is_not_claimed() -> None:
    """A row longer than the capacity would restore past the end of the cache."""
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=4, n_layers=1)
    cache = [FakeLatent(32)]
    cache[0].latent[0, :8] = 1.0
    store.store([1, 2, 3, 4, 5, 6, 7, 8], snapshot(cache, 8))
    assert store.stats()["entries"] == 0


# ---------------------------------------------------------------------------- the loop


def test_the_loop_stops_at_the_end_of_turn_token() -> None:
    model = ScriptedModel(scripted=(11, 12, 2, 14))
    cache = model.make_cache(32)
    result = generate(model, [1, 5], max_new_tokens=8, eos_token_id=2, cache=cache)
    # The end-of-turn token is in the list: it is what the loop stopped *on*, and
    # the text the client sees has it removed by `skip_special_tokens`, not by the
    # loop forgetting it.
    assert result.tokens == [11, 12, 2]
    assert result.stopped == "eos"


def test_the_loop_stops_at_its_budget() -> None:
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    result = generate(model, [1, 5], max_new_tokens=3, eos_token_id=99, cache=cache)
    assert result.tokens == [11, 12, 13]
    assert result.stopped == "length"


def test_the_loop_stops_when_the_predicate_says_so() -> None:
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    seen = []

    def stop() -> bool:
        seen.append(len(seen))
        return len(seen) > 1

    result = generate(model, [1, 5], max_new_tokens=8, eos_token_id=99, cache=cache, on_step=stop)
    assert result.stopped == "cancel"
    assert result.tokens == [11], "the check is before the step, so one token got out"


def test_a_decode_step_sits_at_the_position_after_the_prompt() -> None:
    """The loop's positions are arithmetic on the prompt's length, not a counter."""
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    generate(model, [1, 5, 6], max_new_tokens=3, eos_token_id=99, cache=cache)
    assert model.forwards == [([1, 5, 6], 0), ([11], 3), ([12], 4)]


def test_a_chunked_prompt_is_forwarded_in_chunks_and_masked_at_the_seam() -> None:
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    generate(model, list(range(1, 11)), max_new_tokens=1, eos_token_id=99, cache=cache, chunk=4)
    assert model.forwards[:3] == [([1, 2, 3, 4], 0), ([5, 6, 7, 8], 4), ([9, 10], 8)]


def test_the_loop_resumes_from_the_store_and_forwards_only_the_remainder() -> None:
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=64, n_layers=LAYERS)

    generate(model, [1, 2, 3, 4, 5, 6], max_new_tokens=1, eos_token_id=99, cache=cache,
             prefix_cache=store)
    model.forwards.clear()
    result = generate(model, [1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=1, eos_token_id=99,
                      cache=cache, prefix_cache=store)
    assert result.cached_tokens == 6
    assert model.forwards[0] == ([7, 8], 6), "the six reused tokens were not forwarded"


def test_a_resumed_prompt_forwards_its_remainder_at_the_right_positions() -> None:
    """A resume is a *shortened* prefill, not a restart: positions stay absolute.

    Nothing is renumbered.  The six reused tokens occupied positions 0 to 5, so
    the remainder starts at 6 and the decode after it starts at 8 -- a run that
    restarted its positions at zero would attend to the wrong rows and would
    still produce a plausible token, which is why this is asserted on the calls
    rather than on the answer.
    """
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=64, n_layers=LAYERS)
    generate(model, [3, 1, 4, 1, 5, 6], max_new_tokens=1, eos_token_id=99, cache=cache,
             prefix_cache=store)
    model.forwards.clear()
    generate(model, [3, 1, 4, 1, 5, 6, 7, 8], max_new_tokens=2, eos_token_id=99, cache=cache,
             prefix_cache=store)
    assert model.forwards == [([7, 8], 6), ([11], 8)]


def test_a_second_request_on_a_shared_cache_does_not_see_the_first_one() -> None:
    """The bug this guards, which a served run found: the cache is not reset for you.

    One cache serves every request, and `append` only ever *raises* its length, so
    a prompt shorter than its predecessor's inherits every row past its own end and
    attends to a conversation that has moved on.  The fake writes each token into
    the cache as it goes, so what this asserts is that the second request's rows
    are its own.
    """
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(64)
    generate(model, [1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=1, eos_token_id=99, cache=cache)
    assert cache[0].length == 8
    generate(model, [21, 22], max_new_tokens=1, eos_token_id=99, cache=cache)
    assert cache[0].length == 2, "the first request's six trailing rows are still a context"
    assert cache[0].latent[0, :2, 0].tolist() == [21.0, 22.0]


def test_a_prompt_the_store_has_never_seen_is_forwarded_whole() -> None:
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    cache = model.make_cache(32)
    store = LatentPrefixCache(budget_bytes=1 << 20, capacity=64, n_layers=LAYERS)
    generate(model, [1, 2, 3], max_new_tokens=1, eos_token_id=99, cache=cache, prefix_cache=store)
    model.forwards.clear()
    result = generate(model, [4, 5, 6], max_new_tokens=1, eos_token_id=99, cache=cache,
                      prefix_cache=store)
    assert result.cached_tokens == 0
    assert model.forwards[0] == ([4, 5, 6], 0)


def test_sampling_is_greedy_unless_a_temperature_asks_otherwise() -> None:
    row = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
    assert sample_token(row) == 1
    assert sample_token(row, temperature=0.0) == 1
    # A draw at a nonzero temperature is seeded, so the same request is the same
    # answer twice.
    a = sample_token(row, temperature=1.0, generator=_seeded(7))
    b = sample_token(row, temperature=1.0, generator=_seeded(7))
    assert a == b


def _seeded(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_top_k_and_top_p_narrow_the_distribution() -> None:
    row = torch.tensor([[10.0, 9.0, -5.0, -5.0]])
    assert sample_token(row, temperature=1.0, top_k=2, generator=_seeded(1)) in {0, 1}
    assert sample_token(row, temperature=1.0, top_p=0.5, generator=_seeded(1)) == 0


def test_a_prompt_with_no_tokens_is_refused() -> None:
    model = ScriptedModel()
    with pytest.raises(ValueError):
        generate(model, [], max_new_tokens=1, cache=model.make_cache(8))


# ---------------------------------------------------------------------------- the adapter


def test_the_adapter_serves_a_request_and_names_its_numbers() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.encoding["<|user|>hi<|assistant|>"] = [5, 6, 7, 8]
    adapter = backend(tokenizer=tokenizer)
    result = adapter.generate([request()])[0]
    assert result.text == "abcd"
    assert result.finish_reason == "length"
    assert result.usage.prompt_tokens == 4
    assert result.timings.prefill_seconds >= 0.0
    assert adapter.capabilities.supports_prefix_caching is True
    assert adapter.capabilities.supports_batch is False


def test_the_adapter_reuses_a_prefix_across_two_requests() -> None:
    """The second turn of a chat resends its history and does not pay for it twice.

    What the answer *is* is the store's own test -- the rows it hands back are the
    rows a prefill left, byte for byte, and an answer built from them is the
    answer.  What this asserts is what the adapter is responsible for: that the
    reuse happened, that it is counted, and that the second request was told
    about it.
    """
    tokenizer = FakeTokenizer()
    tokenizer.encoding["<|user|>hi<|assistant|>"] = [5, 6, 7, 8]
    adapter = backend(tokenizer=tokenizer)
    first = adapter.generate([request(request_id="a")])[0]
    second = adapter.generate([request(request_id="b")])[0]
    assert first.usage.cached_tokens == 0
    assert second.usage.cached_tokens == 4
    assert second.finish_reason == first.finish_reason
    assert adapter.metrics()["prefix_cache_hits_total"] == 1
    assert adapter.metrics()["prefix_cache_reused_tokens_total"] == 4


def test_the_adapter_streams_text_and_ends_with_a_reason() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.encoding["<|user|>hi<|assistant|>"] = [5]
    adapter = backend(tokenizer=tokenizer)
    events = list(adapter.stream(request(request_id="s")))
    text = "".join(event.text or "" for event in events)
    assert text == "abcd"
    assert events[-1].finish_reason == "length"
    assert events[-1].usage.completion_tokens == 4


def test_a_stream_cuts_at_a_stop_string_and_holds_back_a_partial_one() -> None:
    """What is sent is what is before the marker, and nothing of the marker.

    The holdback is what makes that true on a stream: ``BE`` could still turn out
    to be the start of ``BETA``, so it waits for the token that decides it, and
    the token that decides it is the one that is cut.
    """
    tokenizer = FakeTokenizer()
    tokenizer.encoding["<|user|>hi<|assistant|>"] = [5]
    tokenizer.pieces[11], tokenizer.pieces[12] = "hello, BE", "TA more"
    adapter = backend(model=ScriptedModel(scripted=(11, 12, 13, 14)), tokenizer=tokenizer)
    events = list(adapter.stream(request(request_id="t", stop=["BETA"])))
    assert "".join(event.text or "" for event in events) == "hello, "
    assert events[-1].finish_reason == "stop"


def test_a_cancelled_request_stops_and_does_not_leave_the_table() -> None:
    adapter = backend()
    adapter._begin_request("gone")
    assert adapter.cancel("gone") is True
    assert adapter.cancel("never-seen") is False
    adapter._clear_request("gone")
    assert adapter.active_request_count() == 0


def test_the_adapter_reports_what_it_holds_between_requests() -> None:
    adapter = backend()
    metrics = adapter.metrics()
    assert metrics["xing4_resident_bytes"] == 1234
    assert metrics["xing4_context_positions"] == 64
    assert metrics["xing4_kv_cache_bytes"] > 0
    assert metrics["prefix_cache_budget_bytes"] > 0


def test_the_chunk_is_derived_from_the_context_the_launcher_asked_for() -> None:
    """A long context gets a narrow chunk, because the score path is `chunk x tokens`."""
    from pocketllm.backends.xing4_backend import _chunk_for, _prefill_peak_bytes

    room = 1 << 30
    assert _chunk_for(2048, 32, room) == 1536
    assert _chunk_for(8192, 32, room) == 384
    assert _chunk_for(32768, 32, room) == 128
    assert _chunk_for(262144, 32, room) == 128, "floored, because a narrower chunk makes no progress"
    assert _chunk_for(0, 32) == DEFAULT_PREFILL_CHUNK
    # The room the adapter actually computes on a 2080 Ti with this checkpoint
    # resident: 3.40 GiB free minus the reserve.
    from pocketllm.backends.xing4_backend import PREFILL_DEVICE_RESERVE

    device_room = int(3.40 * 2**30) - PREFILL_DEVICE_RESERVE
    assert _chunk_for(2048, 32, device_room) == DEFAULT_PREFILL_CHUNK
    assert _chunk_for(32768, 32, device_room) == 256
    # Whatever room it was given, what it picked fits in it -- which is the
    # property an earlier version of this function failed: it budgeted four bytes
    # a score element where the peak is eight, so it answered 256 at a
    # 32768-token context and a 1 GiB room, and 256 does not fit 1 GiB there.
    for context in (2048, 8192, 32768):
        for given in (1 << 29, 1 << 30, 3 << 30):
            picked = _chunk_for(context, 32, given)
            assert _prefill_peak_bytes(picked, 32, context) <= given or picked == 128, (
                f"chunk {picked} at {context} tokens needs "
                f"{_prefill_peak_bytes(picked, 32, context) / 2**20:.0f} MiB of a {given / 2**20:.0f} MiB room"
            )


def test_a_context_whose_narrowest_chunk_does_not_fit_is_refused() -> None:
    """Refused at load, not left to the allocator inside the first long prompt.

    The KV cache and the prefill's score path want the same few GiB, so past a
    context that depends on the card there is no chunk the loop can run at all.
    The adapter knows the card's free memory, so it can say so in a line rather
    than OOM twenty minutes into a request -- which is what a `--max-model-len`
    beyond the hardware used to do.
    """

    class Pinned(Xing4Backend):
        """A backend whose room is fixed, because the test is not on the card."""

        def _prefill_room(self) -> int:
            return 256 << 20

    model = ScriptedModel()
    args = EngineArgs(
        model="a-xing4-checkpoint",
        backend="xing4",
        max_model_len=32768,
        backend_options={"gguf": "/nowhere/xing4_0-29b-IQ4_NL.gguf"},
    )
    instance = Pinned(args, loader=lambda _path, _options: model, tokenizer=FakeTokenizer())
    with pytest.raises(ConfigurationError, match="max-model-len 32768"):
        instance.prepare()
    # And the same context on a card with room is not refused, so the check is
    # about the room and not about the number.
    roomy = Xing4Backend(args, loader=lambda _path, _options: model, tokenizer=FakeTokenizer())
    roomy.prepare()
    assert "peak" in roomy.capabilities.details["prefill"]


def test_the_prefill_peak_is_the_two_measured_terms() -> None:
    """The formula against the probe, so a change to either constant is visible.

    Measured on one 2080 Ti at an 8192-token context: 281.5, 552.4, 1095.2 and
    2183.3 MiB for chunks of 128, 256, 512 and 1024.  Within a few percent, which
    is what a budget needs.
    """
    from pocketllm.backends.xing4_backend import _prefill_peak_bytes

    measured = {128: 281.5, 256: 552.4, 512: 1095.2, 1024: 2183.3}
    for chunk, mib in measured.items():
        got = _prefill_peak_bytes(chunk, 32, 8192) / 2**20
        assert abs(got - mib) / mib < 0.05, f"chunk {chunk}: predicted {got:.1f} MiB, measured {mib} MiB"


def test_a_named_chunk_is_not_second_guessed() -> None:
    assert _Options.from_args(
        EngineArgs(model="m", backend="xing4", backend_options={"prefill_chunk": 64})
    ).prefill_chunk == 64
    assert _Options.from_args(EngineArgs(model="m", backend="xing4")).prefill_chunk is None


def test_prefix_caching_off_is_a_zero_budget() -> None:
    args = EngineArgs(
        model="a-xing4-checkpoint",
        backend="xing4",
        enable_prefix_caching=False,
        backend_options={"prefix_cache_bytes": 1 << 30},
    )
    assert _Options.from_args(args).prefix_cache_bytes == 0
    adapter = backend(prefix_cache_bytes=1 << 30)
    assert adapter.capabilities.supports_prefix_caching is True


def test_an_unknown_backend_option_is_refused() -> None:
    args = EngineArgs(
        model="a-xing4-checkpoint", backend="xing4", backend_options={"nonsense": 1}
    )
    with pytest.raises(ConfigurationError):
        _Options.from_args(args)


def test_the_options_a_native_launch_always_carries_are_accepted() -> None:
    args = EngineArgs(
        model="a-xing4-checkpoint",
        backend="xing4",
        backend_options={"engine_kind": "auto", "pd_mode": "scheduler", "nccl_id_path": "/tmp/x"},
    )
    options = _Options.from_args(args)
    assert options.prefill_chunk is None, "unnamed means derived at load, from the context"
    assert options.prefix_cache_bytes > 0


# ---------------------------------------------------------------------------- selection


def _write(directory: Path, **config) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_the_factory_claims_a_xing4_config(tmp_path: Path) -> None:
    directory = _write(tmp_path / "release", model_type="xing4_0", architectures=["Xing4_0ForCausalLM"])
    assert factory.select_backend(EngineArgs(model=str(directory))) == "xing4"


def test_the_factory_claims_the_quantized_dir_beside_a_release(tmp_path: Path) -> None:
    """A directory of GGUF files is selected by the header inside them."""
    import numpy as np

    (tmp_path / "release").mkdir()
    _write(tmp_path / "release", model_type="xing4_0")
    (tmp_path / "release" / "tokenization_xing4_0.py").write_text("", encoding="utf-8")
    quantized = tmp_path / "release-GGUF"
    quantized.mkdir()
    _fake_gguf(quantized / "xing4_0-29b-IQ4_NL.gguf")

    args = EngineArgs(model=str(quantized))
    assert factory.select_backend(args) == "xing4"


def test_a_directory_named_like_a_gguf_but_holding_another_architecture_is_not_claimed(
    tmp_path: Path,
) -> None:
    quantized = tmp_path / "other-GGUF"
    quantized.mkdir()
    _fake_gguf(quantized / "other.gguf", architecture="llama")
    selected = factory.select_backend(EngineArgs(model=str(quantized)))
    assert selected != "xing4"


def test_an_explicit_xing4_backend_refuses_another_checkpoint(tmp_path: Path) -> None:
    directory = _write(tmp_path / "other", model_type="qwen3", architectures=["Qwen3"])
    with pytest.raises(UnsupportedFeatureError):
        factory.select_backend(EngineArgs(model=str(directory), backend="xing4"))


def test_the_paths_come_from_the_release_beside_the_weights(tmp_path: Path) -> None:
    weights = tmp_path / "weights-GGUF"
    weights.mkdir()
    (weights / "xing4_0-29b-IQ4_NL.gguf").write_bytes(b"")
    release = _write(tmp_path / "weights", model_type="xing4_0")
    (release / "tokenization_xing4_0.py").write_text("", encoding="utf-8")
    gguf, directory = resolve_paths(str(weights), _Options(), None)
    assert gguf.endswith("xing4_0-29b-IQ4_NL.gguf")
    assert directory == str(release)


def test_a_sibling_with_a_config_but_not_the_vocabulary_is_not_the_release(tmp_path: Path) -> None:
    """The bug this guards: another checkpoint's config.json chosen for its alphabet."""
    weights = tmp_path / "weights-GGUF"
    weights.mkdir()
    (weights / "xing4_0-29b-IQ4_NL.gguf").write_bytes(b"")
    _write(tmp_path / "aaa-another-model", model_type="llama")
    release = _write(tmp_path / "weights", model_type="xing4_0")
    (release / "tokenization_xing4_0.py").write_text("", encoding="utf-8")
    assert resolve_paths(str(weights), _Options(), None)[1] == str(release)


def test_weights_with_no_release_beside_them_are_refused_with_the_path(tmp_path: Path) -> None:
    weights = tmp_path / "lonely"
    weights.mkdir()
    (weights / "xing4_0-29b-IQ4_NL.gguf").write_bytes(b"")
    with pytest.raises(ConfigurationError) as caught:
        resolve_paths(str(weights), _Options(), None)
    assert "tokenizer-path" in str(caught.value)


def test_an_explicit_tokenizer_directory_wins(tmp_path: Path) -> None:
    weights = tmp_path / "weights-GGUF"
    weights.mkdir()
    (weights / "xing4_0-29b-IQ4_NL.gguf").write_bytes(b"")
    _write(tmp_path / "weights", model_type="xing4_0")
    elsewhere = tmp_path / "served"
    _write(elsewhere, model_type="xing4_0")
    assert resolve_paths(str(weights), _Options(tokenizer=str(elsewhere)), None)[1] == str(elsewhere)


def test_two_ggufs_in_one_directory_are_refused_rather_than_guessed(tmp_path: Path) -> None:
    weights = tmp_path / "weights-GGUF"
    weights.mkdir()
    (weights / "a.gguf").write_bytes(b"")
    (weights / "b.gguf").write_bytes(b"")
    with pytest.raises(ConfigurationError):
        resolve_paths(str(weights), _Options(), None)


def _fake_gguf(path: Path, *, architecture: str = "xing4_0") -> None:
    """A GGUF header and nothing else: enough for `general.architecture` to be read."""
    import struct

    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    metadata = struct.pack("<Q", 1) + string("general.architecture") + struct.pack("<I", 8) + string(
        architecture
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + metadata
    path.write_bytes(header + struct.pack("<Q", 0))
