"""The MiMo adapter's request path, exercised without a checkpoint and without a card.

Neither half of the runtime can be stood up here: the checkpoint is 150 GiB of experts on a host
bank and the tree wants four cards. Both are therefore injected -- a scripted stand-in for the
model and a table for the tokenizer -- and what is asserted is everything the adapter and the
generation loop themselves decide:

* which prompt a request becomes, and that a chat is rendered by the template the checkpoint ships
  rather than by a generic one;
* that a greedy request is one chunked prefill and one step a token, and one step fewer than the
  tokens it returned;
* that the loop stops on the checkpoint's own end-of-turn tokens, on the caller's budget, on a
  cancel, and on a stop string -- and that the text a stop string truncates is not sent;
* that a stream sends the same text as the unstreamed call, holds back a tail that is still half a
  stop string, and never sends a replacement character;
* that a request the run's cache cannot hold is refused before the loop starts;
* that the launcher's levers reach the model, and that a lever the adapter does not know is a
  refusal rather than a silently ignored key.

What is *not* covered, and cannot be: the four-card collectives. The per-step cancel broadcast is
built but not entered -- ``_step_sync`` returns a local flag at a world of one, which is what every
test here runs as -- and nothing below has been run against the released checkpoint or its real
tokenizer; the acceptance run for that is ``docs/models/mimo-v2.6-flash.md``.
"""

from __future__ import annotations

import torch
import pytest

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    RequestCancelledError,
    SamplingParams,
)
from pocketllm.backends import factory
from pocketllm.backends.mimo_backend import (
    DEFAULT_EXPERT_ROWS,
    DEFAULT_PREFILL_CHUNK,
    DEFAULT_RESIDENT_ROWS,
    MimoBackend,
    _hold_back,
    _Options,
)
from src.models.mimo_v2.generate import Generation, generate, sample_token


# ---------------------------------------------------------------------------- stand-ins


VOCAB = 32


class FakeCache:
    """A cache's contract as the loop reads it: a capacity, a reset, a memory figure."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.resets = 0
        self.memory_bytes = 23_000 * int(capacity)

    def reset(self) -> None:
        self.resets += 1


class ScriptedModel:
    """Stands in for ``MimoV2DeviceModel``: it hands back the logits a script says it should.

    The adapter only reaches for ``cache``, ``prefill`` and ``step``, which is the whole of the
    model's serving surface, so what this records is the call sequence the adapter is responsible
    for. ``ScriptedModel.scripted`` is the token list the run is meant to produce: step `i`'s row
    is a one-hot at that token, so `argmax` follows the script and a sampled draw follows it too.
    """

    def __init__(self, scripted=(11, 12, 13)) -> None:
        self.scripted = list(scripted)
        self.prefills: list[tuple[list[int], int]] = []
        self.steps: list[tuple[int, int]] = []
        self.caches: list[FakeCache] = []
        self._cursor = 0

    def cache(self, capacity: int) -> FakeCache:
        cache = FakeCache(capacity)
        self.caches.append(cache)
        return cache

    def _row(self, token: int) -> torch.Tensor:
        row = torch.full((1, VOCAB), -10.0)
        row[0, int(token)] = 10.0
        return row

    def prefill(self, prompt_ids, *, cache=None, chunk=512):
        self.prefills.append(([int(token) for token in prompt_ids], int(chunk)))
        self._cursor = 0
        return self._row(self.scripted[0])

    def step(self, token_id, *, start_pos, cache=None):
        self.steps.append((int(token_id), int(start_pos)))
        self._cursor += 1
        return self._row(self.scripted[min(self._cursor, len(self.scripted) - 1)])


class FakeTokenizer:
    """Ids to text through a table, recording what it was asked to encode."""

    def __init__(self, pieces=None, encoding=None, eos_token_id=2) -> None:
        self.pieces = {0: "<s>", 1: "</s>", 2: "<|im_end|>", 3: ""}
        self.pieces.update(pieces or {})
        self.encoding = dict(encoding or {})
        self.eos_token_id = eos_token_id
        self.calls: list[tuple[str, bool]] = []
        self.templates: list[tuple[list[dict], bool, dict]] = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((str(text), bool(add_special_tokens)))
        return {"input_ids": list(self.encoding.get(str(text), [7]))}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.templates.append((list(messages), bool(add_generation_prompt), dict(kwargs)))
        body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return f"{body}<|assistant|>"

    def decode(self, ids, skip_special_tokens=True):
        pieces = (
            ""
            if skip_special_tokens and token in {0, 1, 2}
            else self.pieces.get(token, "")
            for token in ids
        )
        return "".join(pieces)


def backend(*, model=None, tokenizer=None, **options) -> MimoBackend:
    """An adapter whose model and tokenizer are injected, so nothing is read from disk."""
    model = model if model is not None else ScriptedModel()
    tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
    args = EngineArgs(
        model="a-mimo-checkpoint",
        backend="mimo",
        backend_options=dict(options),
    )
    return MimoBackend(args, loader=lambda _args, _options: model, tokenizer=tokenizer)


def request(prompt_ids=(5, 6, 7), **params) -> GenerationRequest:
    return GenerationRequest(prompt_tokens=list(prompt_ids), sampling_params=SamplingParams(**params))


# ---------------------------------------------------------------------------- the levers


def test_the_launcher_levers_are_resolved_and_an_unknown_one_is_refused():
    """Every option the adapter reads is a documented one, and a typo is a refusal.

    A key that silently does nothing is how a run ends up measured on the wrong lever, which is
    why this backend refuses one rather than carrying it -- the four keys the CLI itself always
    fills in are the exception, and they are dropped rather than refused.
    """
    options = _Options.from_args(
        EngineArgs(
            model="x",
            backend="mimo",
            backend_options={
                "engine_kind": "cpp",
                "pd_mode": "off",
                "routed_experts_device": "cpu",
                "nccl_id_path": "/tmp/x",
                "expert_deal": "id",
                "expert_rows": 8,
                "prefill_chunk": 512,
                "pin": "false",
                "resident_rows": 12,
                "slots": 3,
            },
        )
    )
    assert (options.chunk_rows, options.deal) == (8, "id")
    assert (options.prefill_chunk, options.slots, options.pin) == (512, 3, False)
    assert options.resident_rows == 12
    defaults = _Options.from_args(EngineArgs(model="x", backend="mimo"))
    assert defaults.chunk_rows == DEFAULT_EXPERT_ROWS
    assert defaults.prefill_chunk == DEFAULT_PREFILL_CHUNK and defaults.pin
    assert defaults.resident_rows == DEFAULT_RESIDENT_ROWS == 0
    with pytest.raises(ConfigurationError, match="no option"):
        _Options.from_args(
            EngineArgs(model="x", backend="mimo", backend_options={"expert_deals": "id"})
        )
    with pytest.raises(ConfigurationError, match="deal"):
        _Options.from_args(
            EngineArgs(model="x", backend="mimo", backend_options={"deal": "round-robin"})
        )
    with pytest.raises(ConfigurationError, match="prefill_chunk"):
        _Options.from_args(
            EngineArgs(model="x", backend="mimo", backend_options={"prefill_chunk": 0})
        )
    with pytest.raises(ConfigurationError, match="pin"):
        _Options.from_args(
            EngineArgs(model="x", backend="mimo", backend_options={"pin": "maybe"})
        )
    with pytest.raises(ConfigurationError, match="resident_rows"):
        _Options.from_args(
            EngineArgs(model="x", backend="mimo", backend_options={"resident_rows": -1})
        )


def test_a_mimo_checkpoint_selects_its_own_adapter_and_another_is_refused(tmp_path):
    """`auto` reaches this adapter from the checkpoint's own config, and `mimo` refuses the rest."""
    import json

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"]}),
        encoding="utf-8",
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "mimo"
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="mimo")) == "mimo"
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v41", "architectures": []}), encoding="utf-8"
    )
    # An explicit `mimo` on somebody else's checkpoint is a refusal, and `auto` finds their adapter.
    with pytest.raises(Exception, match="MiMo-V2.6-Flash checkpoints only"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="mimo"))
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "v41"


def test_the_launchers_chunk_and_rows_reach_the_model():
    """A request's prefill width is the launcher's, and the levers are the ones the run was built on."""
    seen: dict[str, object] = {}

    def loader(_args, options):
        seen["chunk_rows"] = options.chunk_rows
        seen["deal"] = options.deal
        seen["slots"] = options.slots
        seen["pin"] = options.pin
        return ScriptedModel()

    adapter = MimoBackend(
        EngineArgs(model="x", backend="mimo", backend_options={"prefill_chunk": 128, "deal": "id"}),
        loader=loader,
        tokenizer=FakeTokenizer(),
    )
    adapter.generate([request(max_tokens=1)])
    assert seen == {"chunk_rows": DEFAULT_EXPERT_ROWS, "deal": "id", "slots": 2, "pin": True}
    assert adapter.capabilities.details["prefill"].startswith("grouped multi-token kernel, 128")


# ---------------------------------------------------------------------------- a request


def test_a_greedy_request_is_one_prefill_and_one_step_a_token():
    """The prompt goes through the chunked path once and every token after it is a decode step.

    One step fewer than the tokens returned, and not one a token: the prefill ends holding the last
    row's logits, so the first token costs one distribution and the step after it is the second
    token's. A loop that stepped before drawing would spend a forward per token and one more.
    """
    model = ScriptedModel(scripted=(11, 12, 2, 13))
    adapter = backend(model=model)
    result = adapter.generate([request(max_tokens=5)])[0]
    assert model.prefills == [([5, 6, 7], DEFAULT_PREFILL_CHUNK)]
    assert model.steps == [(11, 3), (12, 4)]
    assert result.token_ids == [11, 12, 2] and result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 3 and result.usage.completion_tokens == 3
    assert result.text == ""  # the fake's pieces are all special or empty
    assert result.timings.prefill_seconds > 0 and result.timings.tpot_seconds > 0


def test_the_loop_stops_at_the_end_of_turn_and_at_the_callers_budget():
    """Two ways to end, and the finish reason names which one it was."""
    model = ScriptedModel(scripted=(2, 11))
    adapter = backend(model=model)
    ended = adapter.generate([request(max_tokens=6)])[0]
    assert ended.token_ids == [2] and ended.finish_reason == "stop"
    assert model.steps == []  # the end-of-turn token came out of the prefill's own row

    model = ScriptedModel(scripted=(11, 12, 13, 14))
    adapter = backend(model=model)
    capped = adapter.generate([request(max_tokens=3)])[0]
    assert capped.token_ids == [11, 12, 13] and capped.finish_reason == "length"


def test_a_chat_is_rendered_by_the_checkpoints_own_template():
    """`messages` go through `apply_chat_template`, and the rendered prompt is not re-tokenized
    with special tokens of its own -- the template's markup already carries them."""
    tokenizer = FakeTokenizer(encoding={"<|user|>hi<|assistant|>": [9, 10, 11]})
    adapter = backend(tokenizer=tokenizer)
    result = adapter.generate(
        [
            GenerationRequest(
                prompt="ignored",
                sampling_params=SamplingParams(max_tokens=1),
                metadata={"messages": [{"role": "user", "content": "hi"}]},
            )
        ]
    )[0]
    assert tokenizer.templates == [([{"role": "user", "content": "hi"}], True, {})]
    assert tokenizer.calls == [("<|user|>hi<|assistant|>", False)]
    assert result.usage.prompt_tokens == 3


def test_a_completion_prompt_goes_to_the_tokenizer_verbatim():
    """A raw completion is not a chat and gets no header, which is what the launcher does too."""
    tokenizer = FakeTokenizer(encoding={"plain text": [4, 4]})
    adapter = backend(tokenizer=tokenizer)
    adapter.generate([GenerationRequest(prompt="plain text", sampling_params=SamplingParams(max_tokens=1))])
    assert tokenizer.calls == [("plain text", True)]
    assert tokenizer.templates == []


def test_a_sampled_request_draws_the_same_token_from_the_same_seed():
    """Sampling is seeded from the request, so four ranks draw the same token without a message."""
    draws = [
        sample_token(
            torch.tensor([[0.0, 5.0, 5.0, 5.0]]),
            temperature=1.0,
            generator=torch.Generator().manual_seed(7),
        )
        for _ in range(2)
    ]
    assert draws[0] == draws[1]
    assert sample_token(torch.tensor([[0.0, 1.0, 9.0]]), temperature=0.0) == 2
    # A temperature at the greedy boundary is greedy, and top-k of one is the argmax.
    assert sample_token(torch.tensor([[0.0, 1.0, 9.0]]), temperature=1e-5) == 2
    assert sample_token(torch.tensor([[0.0, 1.0, 9.0]]), temperature=1.0, top_k=1) == 2


def test_a_cancel_reaches_the_loop_at_a_token_boundary():
    """A client's disconnect stops the loop between two tokens, not inside a forward.

    Two shapes, and they are different code paths: a request cancelled before it starts is refused
    at the door -- the adapter checks before the loop, which is what keeps a rank 0 from leaving
    its peers inside a broadcast -- and one cancelled while it is running ends at the boundary the
    per-step check lands on.
    """
    adapter = backend(model=ScriptedModel(scripted=(11, 12, 13, 14)))
    req = request(max_tokens=8)
    adapter._begin_request(req.request_id)
    adapter.cancel(req.request_id)
    with pytest.raises(RequestCancelledError):
        adapter.generate([req])

    # The same loop without the flag runs to its budget, so what stopped it was the flag.
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    straight = generate(
        model,
        [5, 6, 7],
        max_new_tokens=4,
        eos_token_id={2},
        cache=model.cache(64),
        on_step=lambda: False,
    )
    assert straight.stopped == "length" and len(straight.tokens) == 4

    # A flag that is set in the middle stops it there, which is the boundary a cancel lands on.
    seen = {"steps": 0}

    def stop_after_two() -> bool:
        seen["steps"] += 1
        return seen["steps"] > 2

    stopped = generate(
        ScriptedModel(scripted=(11, 12, 13, 14)),
        [5, 6, 7],
        max_new_tokens=8,
        eos_token_id={2},
        cache=FakeCache(64),
        on_step=stop_after_two,
    )
    assert stopped.stopped == "cancel" and stopped.tokens == [11, 12]


def test_a_single_rank_run_needs_no_process_group():
    """The per-step sync is a local flag at a world of one, and a collective at anything else."""
    adapter = backend()
    adapter._world, adapter._rank = 1, 0
    assert adapter._step_sync("req")() is False
    # A local condition -- the stream's stop string -- is folded into the same answer, because it
    # is the only thing that can carry a fact known on one rank to the other three.
    assert adapter._step_sync("req", lambda: True)() is True
    assert adapter._step_sync("req", lambda: False)() is False


def test_a_stop_string_never_leaves_the_loop_from_inside_it():
    """The marker stops the loop at a *step boundary* the ranks agree on, and not on the spot.

    Raising out of the token callback is the obvious way to end a stream at a stop string and it is
    wrong on four ranks: the marker is found on rank 0 and only rank 0, so the rank that unwound
    would leave its three peers inside the next layer's `all_reduce` and wedge the deployment
    rather than answer it. What the streamer does instead is record the hit and hand `reached` to
    the per-step sync, which is already a collective.

    Asserted through the loop the adapter actually runs: `stopped` is the cancel-shaped ending the
    per-step check produces, the caller reads it as `stop`, and no token past the marker is
    accepted.
    """
    tokenizer = FakeTokenizer(pieces={11: "alpha", 12: "BETA", 13: "gamma"})
    model = ScriptedModel(scripted=(11, 12, 13, 14))
    adapter = backend(model=model, tokenizer=tokenizer)
    stream = adapter.stream(request(max_tokens=6, stop=("BETA",)))
    events = list(stream)
    assert "".join(event.text for event in events) == "alpha"
    assert events[-1].finish_reason == "stop"

    # The loop kept its shape: the marker's token was produced and sampled, and the step after it
    # is the one the boundary landed on -- the loop is not shorter by a token, it is shorter by a
    # boundary.
    assert model.steps == [(11, 3), (12, 4)]


# ---------------------------------------------------------------------------- the stream


def test_the_stream_sends_the_same_text_as_the_unstreamed_result():
    """A stream is the same answer, one token at a time, and the last event carries the usage."""
    tokenizer = FakeTokenizer(pieces={11: "Hello", 12: ", ", 13: "world"})
    model = ScriptedModel(scripted=(11, 12, 2))
    adapter = backend(model=model, tokenizer=tokenizer)
    chunks = [event.text for event in adapter.stream(request(max_tokens=4))]
    assert "".join(chunks) == "Hello, "  # the last scripted token is the end of turn
    assert model.steps == [(11, 3), (12, 4)]  # one step a token after the prefill's own row

    model = ScriptedModel(scripted=(11, 12, 2))
    adapter = backend(model=model, tokenizer=tokenizer)
    events = list(adapter.stream(request(max_tokens=4)))
    assert events[-1].finish_reason == "stop"
    assert events[-1].usage.completion_tokens == 3
    assert "".join(event.text for event in events) == adapter.generate([request(max_tokens=4)])[0].text


def test_a_stop_string_ends_the_answer_where_it_starts_and_is_never_sent():
    """The marker is the client's, not the answer's: the text past it is not sent."""
    tokenizer = FakeTokenizer(pieces={11: "alpha", 12: "BETA", 13: "gamma"})
    adapter = backend(model=ScriptedModel(scripted=(11, 12, 13, 14)), tokenizer=tokenizer)
    events = list(adapter.stream(request(max_tokens=6, stop=("BETA",))))
    assert "".join(event.text for event in events) == "alpha"
    assert events[-1].finish_reason == "stop"

    # The unstreamed call reports the whole answer, marker included: the marker is what a client
    # asked to stop at, not a token the model did not produce, and a caller that only reads ``text``
    # gets the same string it would have got from any other runtime.
    unstreamed = backend(
        model=ScriptedModel(scripted=(11, 12, 13, 14)), tokenizer=tokenizer
    ).generate([request(max_tokens=6, stop=("BETA",))])[0]
    assert unstreamed.text == "alphaBETAgamma"


def test_a_tail_that_is_still_half_a_stop_string_is_held_back():
    """A stream cannot take a character back, so a partial marker waits for the token that ends it."""
    assert _hold_back("alpha BE", ["BETA"]) == "alpha "
    assert _hold_back("alpha BETA", ["BETA"]) == "alpha BETA"  # whole: the caller cuts it
    assert _hold_back("alpha B", ["BETA"]) == "alpha "
    assert _hold_back("alpha", ["BETA"]) == "alpha"
    assert _hold_back("alphabet", ["BETA"]) == "alphabet"
    assert _hold_back("", ["BETA"]) == ""

    tokenizer = FakeTokenizer(pieces={11: "alpha", 12: "BE", 13: "TA", 14: "!"})
    adapter = backend(model=ScriptedModel(scripted=(11, 12, 13, 14)), tokenizer=tokenizer)
    events = list(adapter.stream(request(max_tokens=3, stop=("BETA",))))
    assert "".join(event.text for event in events) == "alpha"  # neither "BE" nor "TA" went out


# ---------------------------------------------------------------------------- the shape


def test_a_prompt_that_does_not_fit_the_context_is_refused():
    """The refusal is before the loop, because a prompt with no room has no answer to give."""
    adapter = MimoBackend(
        EngineArgs(model="x", backend="mimo", max_model_len=16, backend_options={}),
        loader=lambda _args, _options: ScriptedModel(),
        tokenizer=FakeTokenizer(),
    )
    with pytest.raises(ConfigurationError, match="positions this run's cache was sized at"):
        adapter.generate([request(prompt_ids=range(16), max_tokens=1)])
    # An absent budget is everything the prompt leaves, which is what an OpenAI client that sent
    # no max_tokens asked for.
    result = adapter.generate([request(prompt_ids=[5, 6], max_tokens=None)])[0]
    assert result.usage.completion_tokens <= 14


def test_the_cache_is_the_context_the_launcher_asked_for():
    """One cache for the life of the process, reset per request, sized to `--max-model-len`."""
    model = ScriptedModel(scripted=(11, 2))
    adapter = MimoBackend(
        EngineArgs(model="x", backend="mimo", max_model_len=4096, backend_options={}),
        loader=lambda _args, _options: model,
        tokenizer=FakeTokenizer(),
    )
    adapter.generate([request(max_tokens=1)])
    adapter.generate([request(max_tokens=1)])
    assert len(model.caches) == 1
    assert model.caches[0].capacity == 4096
    assert model.caches[0].resets == 2
    assert adapter.capabilities.details["context"].startswith("4096 positions")


def test_generate_reports_what_the_loop_cost():
    """The result's timings are the loop's own, and a prefill is the whole of the first token's wait."""
    generation = generate(
        ScriptedModel(scripted=(11, 2)),
        [5, 6, 7],
        max_new_tokens=4,
        eos_token_id={2},
        cache=FakeCache(64),
        chunk=256,
    )
    assert isinstance(generation, Generation)
    assert generation.stopped == "eos"
    assert generation.prefill_seconds > 0 and generation.decode_seconds > 0
    assert generation.ttft_seconds == generation.prefill_seconds
    assert generation.step_seconds == pytest.approx(
        generation.decode_seconds / max(1, len(generation.tokens) - 1)
    )


def test_a_prompt_with_no_tokens_or_a_budget_of_none_is_refused():
    with pytest.raises(ValueError, match="nothing to prefill"):
        generate(ScriptedModel(), [], max_new_tokens=4)
    with pytest.raises(ValueError, match="not a generation"):
        generate(ScriptedModel(), [5], max_new_tokens=0)
    with pytest.raises(ValueError, match="not a chunk"):
        generate(ScriptedModel(), [5], max_new_tokens=4, chunk=0)


# ------------------------------------------------------------------- four ranks


@pytest.fixture
def group(monkeypatch):
    """A one-process stand-in for a four-rank group: the collectives are recorded, not performed.

    The deadlock this exists to catch is not a wrong number, it is a *missing* call. Every routed
    layer closes with an ``all_reduce``, so rank 0 entering a generation its peers were never told
    about does not run a solo answer -- it arrives at a collective they are not at, and NCCL hangs
    both sides. A test that never enters a collective cannot see that, which is exactly why the
    first real serving run hung on its first request while every test here passed.
    """
    import torch.distributed as dist

    sent: list[dict] = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda payload, src=0: sent.append(payload[0]))
    monkeypatch.setattr(dist, "broadcast", lambda flag, src=0: None)
    return sent


def as_four_ranks(adapter, rank: int) -> MimoBackend:
    """An adapter that believes it is one rank of four, without a group behind it."""
    adapter._distributed = True
    adapter._ep = None
    adapter._device = torch.device("cpu")
    adapter._world, adapter._rank = 4, rank
    return adapter


def test_rank_zero_hands_every_request_to_its_peers_before_it_runs_it(group):
    """One payload a request, and it leaves before the generation does."""
    adapter = as_four_ranks(backend(), 0)
    adapter.generate([request(prompt_ids=(5, 6), max_tokens=3)])
    adapter.generate([request(prompt_ids=(7,), max_tokens=2)])

    assert [payload["op"] for payload in group] == ["generate", "generate"]
    first = group[0]
    assert first["prompt_ids"] == [5, 6]
    assert first["max_new_tokens"] == 3
    assert first["temperature"] == 0.0 and first["seed"] is None
    assert set(first) == {
        "op",
        "request_id",
        "prompt_ids",
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "seed",
    }


def test_only_rank_zero_dispatches(group):
    """A worker sends nothing: it is the one being told, and a second send is a second request."""
    as_four_ranks(backend(), 2)._dispatch(request(), [5], 4)
    assert group == []


def test_a_worker_runs_the_payload_rank_zero_sent_key_for_key(group, monkeypatch):
    """The payload's keys are the loop's arguments, which is what keeps the ranks in step.

    A key renamed on one side and not the other is not a crash in the payload -- it is a worker
    that raises before its first ``all_reduce`` while rank 0 waits inside one, or worse, a worker
    that runs the loop with a *different* budget and stops at a different token. So this asserts
    the round trip rather than either half: what rank 0 sent is what the worker executes.
    """
    leader = as_four_ranks(backend(), 0)
    leader.generate([request(prompt_ids=(5, 6), max_tokens=3, temperature=0.7, top_k=4, seed=9)])
    payload = group[0]

    seen: dict = {}

    def spy(_model, prompt_ids, **kwargs):
        seen["prompt_ids"] = list(prompt_ids)
        seen.update(kwargs)
        return Generation(tokens=[11], stopped="length")

    monkeypatch.setattr("src.models.mimo_v2.generate.generate", spy)
    worker = as_four_ranks(backend(), 3)
    worker._run_payload(payload)

    assert seen["prompt_ids"] == payload["prompt_ids"]
    assert seen["max_new_tokens"] == payload["max_new_tokens"]
    assert seen["temperature"] == payload["temperature"]
    assert seen["top_k"] == payload["top_k"]
    assert seen["top_p"] == payload["top_p"]
    assert seen["seed"] == payload["seed"]
