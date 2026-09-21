"""The V4.1 adapter's request path, exercised without a checkpoint and without a card.

Neither half of the runtime can be built on this host: ``load_backbone`` wants 476 GiB across four
cards, and ``generate`` wants the tree that loader returns. Both are therefore injected -- a
stand-in for the loaded backbone and a scripted stand-in for the loop -- and what is asserted here
is everything the adapter itself decides:

* which prompt a request becomes, and that it is the checkpoint's own ``encoding/encoding.py``
  rather than a jinja template (this checkpoint ships no ``chat_template`` at all);
* that a request the attention caches cannot hold is refused on rank 0 *before* the broadcast,
  because a rank that refuses after it leaves its peers in a collective nobody enters;
* that a stop string ends the generation with the text up to the marker rather than with the
  tokens the loop had already produced past it;
* that a thinking-mode stream is split as it arrives, each half diffed against what was already
  sent rather than re-sent;
* that the loop's own decode figure travels with the result, and that when the loop unwinds on a
  stop string -- where no ``Generation`` comes back at all -- the wall minus the first token's wait
  is what is left of it.

The encoder is written into a temporary checkpoint directory as a real file, because that is what
the adapter reads: the format belongs to the checkpoint, so the loader has to find
``encoding/encoding.py`` under the path it was pointed at, render through it, and fall back to the
tolerant split when the checkpoint's own parser will not accept the text. The tokenizer is a stub
for the same reason, and it pins two details that are easy to get wrong and expensive to get wrong
in production: ``add_special_tokens=False`` on a rendered prompt that already opens with its own
beginning-of-sentence token, and a raw completion prompt that goes to the tokenizer verbatim.

What is *not* covered here, and cannot be: no V4.1 checkpoint exists on this host, so nothing below
has been run against the released encoder or a real tokenizer, and the four-card collectives
(``_broadcast``, the per-step ``all_reduce``) are stubbed rather than entered.
"""

from __future__ import annotations

import threading

import pytest

from pocketllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    RequestCancelledError,
    SamplingParams,
    UnsupportedFeatureError,
)
from pocketllm.backends.v41_backend import V41Backend, _split_running


# ---------------------------------------------------------------------------- stand-ins

#: The encoder a V4.1 checkpoint ships, shrunk to the contract the adapter uses. It is strict in the
#: same three ways the released one is -- it wants the end-of-sentence token, it wants the thinking
#: block closed, and it asserts rather than returning an error -- so a test that makes it refuse is
#: making the real parser refuse.
FAKE_ENCODER = """\
def encode_messages(messages, thinking_mode="chat", context=None, reasoning_effort=None):
    body = "|".join(f"{m['role']}:{m['content']}" for m in messages)
    return f"<b>{body}</b>[{thinking_mode}][{reasoning_effort}]"


def parse_message_from_completion_text(text, thinking_mode):
    assert text.endswith("<e>"), f"the generation did not finish: {text!r}"
    body = text[: -len("<e>")]
    if thinking_mode == "thinking":
        assert "</think>" in body, f"the thinking block never closed: {body!r}"
        reasoning, content = body.split("</think>", 1)
    else:
        reasoning, content = "", body
    calls = []
    if "<tool>" in content:
        calls.append({"id": "call_1", "type": "function",
                      "function": {"name": "weather", "arguments": "{}"}})
    return {"role": "assistant", "reasoning_content": reasoning,
            "content": content, "tool_calls": calls}
"""


class FakeTokenizer:
    """Ids to text through a table, recording what it was asked to encode."""

    def __init__(self, pieces=None, encoding=None, eos_token_id=1) -> None:
        self.pieces = {0: "<s>", 1: "<e>"}
        self.pieces.update(pieces or {})
        self.encoding = dict(encoding or {})
        self.eos_token_id = eos_token_id
        self.special_ids = {0, 1}
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((str(text), bool(add_special_tokens)))
        return {"input_ids": list(self.encoding.get(str(text), [7]))}

    def decode(self, ids, skip_special_tokens=True):
        pieces = (
            "" if skip_special_tokens and token in self.special_ids else self.pieces.get(token, "")
            for token in ids
        )
        return "".join(pieces)


class FakeBuffer:
    """A named buffer: only ``device`` is read, so this is not a tensor."""

    def __init__(self, device) -> None:
        self.device = device


class FakeModel:
    def __init__(self, buffers) -> None:
        self._buffers = tuple(buffers)

    def named_buffers(self):
        return iter(self._buffers)


class FakeFront:
    """Stands in for ``LoadedBackbone``: the adapter only reaches for ``model.named_buffers``."""

    def __init__(self, buffers=()) -> None:
        self.model = FakeModel(buffers)


class ScriptedGenerate:
    """Stands in for ``src.models.deepseek_v4_1.generate.generate``.

    Records the call, then replays a fixed token list through the caller's hook -- which is the
    adapter's own per-step hook, so the stop-string and cancellation decisions under test are the
    real ones and not a re-implementation of them.
    """

    def __init__(
        self,
        tokens=(11, 12),
        *,
        stopped="eos",
        decode_seconds=0.4,
        before_token=None,
        failure=None,
        driver=None,
    ) -> None:
        self.tokens = list(tokens)
        self.stopped = stopped
        self.decode_seconds = decode_seconds
        self.before_token = before_token
        self.failure = failure
        self.driver = driver
        self.calls: list[dict] = []

    def __call__(
        self,
        front,
        prompt_ids,
        *,
        max_new_tokens=32,
        temperature=0.0,
        top_k=None,
        eos_token_id=None,
        seed=None,
        on_token=None,
        graphs=False,
        prefill_chunk=None,
    ):
        from src.models.deepseek_v4_1.generate import Generation

        self.calls.append({
            "front": front,
            "prompt_ids": list(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "eos_token_id": eos_token_id,
            "seed": seed,
            "graphs": graphs,
            "prefill_chunk": prefill_chunk,
        })
        if self.failure is not None:
            raise self.failure
        for index, token in enumerate(self.tokens):
            if index == 0 and self.before_token is not None:
                self.before_token()
            if on_token is not None:
                on_token(token, None)
        return Generation(
            tokens=list(self.tokens),
            prompt_tokens=len(list(prompt_ids)),
            stopped=self.stopped,
            driver=self.driver,
            decode_seconds=self.decode_seconds,
        )


@pytest.fixture
def loop(monkeypatch):
    """Install a scripted loop, and hand back an installer that keeps the stub reachable."""

    import src.models.deepseek_v4_1.generate as module

    def install(**kwargs) -> ScriptedGenerate:
        stub = ScriptedGenerate(**kwargs)
        monkeypatch.setattr(module, "generate", stub)
        return stub

    return install


# ---------------------------------------------------------------------------- helpers


def _args(model, **overrides) -> EngineArgs:
    base = dict(model=str(model), backend="v41")
    base.update(overrides)
    return EngineArgs(**base)


def _build(model, *, tokenizer=None, buffers=(), **overrides):
    """A backend with both injection points filled, so nothing is loaded and no card is touched."""
    tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
    front = FakeFront(buffers)
    backend = V41Backend(_args(model, **overrides), front=front, tokenizer=tokenizer)
    return backend, tokenizer


def _request(request_id="r1", **metadata) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt="user: hi",
        sampling_params=SamplingParams(max_tokens=4),
        metadata=dict(metadata),
    )


def _checkpoint(tmp_path, *, encoder: bool = True):
    """A checkpoint directory: the adapter reads the encoder out of it by path."""
    if encoder:
        directory = tmp_path / "encoding"
        directory.mkdir(exist_ok=True)
        (directory / "encoding.py").write_text(FAKE_ENCODER, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------- capabilities


def test_capabilities_describe_a_serialized_single_request_service(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path))
    capabilities = backend.capabilities

    assert capabilities.name == "v41"
    assert "deepseek_v41" in capabilities.models
    assert capabilities.model_formats == ("safetensors",)
    # One mutable KV state: a second concurrent request would decode against the first one's
    # positions, so the batch promise is refused rather than honoured partially.
    assert capabilities.supports_batch is False
    assert capabilities.supports_streaming is True
    assert capabilities.supports_cancellation is True
    assert capabilities.supports_prefix_caching is False
    assert capabilities.supports_logprobs is False
    assert "encoding/encoding.py" in capabilities.details["prompt_format"]


# ---------------------------------------------------------------------------- options


def test_the_launcher_options_this_backend_ignores_are_accepted(tmp_path):
    """A launch must not have to strip the flags it shares with the native adapter."""
    backend, _ = _build(
        _checkpoint(tmp_path),
        backend_options={
            "engine_kind": "auto",
            "routed_experts_device": "cuda",
            "pd_mode": 0,
            "nccl_id_path": "/tmp/nccl-id",
        },
    )
    assert backend.capabilities.name == "v41"


def test_an_unknown_backend_option_is_refused_with_the_known_set(tmp_path):
    with pytest.raises(ConfigurationError, match="does not recognise backend option"):
        _build(_checkpoint(tmp_path), backend_options={"expert_pool_size": 288})


@pytest.mark.parametrize(
    "options, message",
    [
        ({"expert_pool_rows": -1}, "must not be negative"),
        ({"expert_buffers": -1}, "must not be negative"),
        ({"expert_world": 0}, "must be >= 1"),
        ({"expert_deal": "random"}, "'sorted' or 'id'"),
        ({"prefill_chunk": 0}, "must be >= 1"),
        ({"threads": 0}, "must be >= 1"),
    ],
)
def test_a_backend_option_out_of_range_is_refused(tmp_path, options, message):
    with pytest.raises(ConfigurationError, match=message):
        _build(_checkpoint(tmp_path), backend_options=options)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model_format": "gguf"}, "safetensors shards only"),
        ({"dtype": "float16"}, "bfloat16"),
        ({"attention_window": 4096}, "attention_window"),
        ({"speculative_method": "mtp"}, "speculative decoding"),
        ({"max_batch_size": 2}, "max_batch_size must be 1"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
    ],
)
def test_options_the_runtime_cannot_honour_are_refused_at_construction(
    tmp_path, overrides, message
):
    """Before anything is loaded: a 476 GiB read is a bad place to discover a typo."""
    with pytest.raises(UnsupportedFeatureError, match=message):
        _build(_checkpoint(tmp_path), **overrides)


def test_the_backend_option_wins_over_the_cli_flag_for_the_prefill_chunk(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path), backend_options={"prefill_chunk": 4096})
    assert backend._prefill_chunk == 4096

    backend, _ = _build(_checkpoint(tmp_path), prefill_chunk_tokens=1024)
    assert backend._prefill_chunk == 1024

    backend, _ = _build(_checkpoint(tmp_path))
    assert backend._prefill_chunk is None


def test_an_expert_device_option_offsets_by_rank_only_when_one_process_drives_a_card(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path), backend_options={"device": "cuda:2"})
    assert backend._resolve_expert_device() == "cuda:2"

    backend, _ = _build(_checkpoint(tmp_path))
    assert backend._resolve_expert_device() is None

    # A sharded run wants where the split starts, not this rank's card: the loader adds the rank.
    backend, _ = _build(
        _checkpoint(tmp_path), tensor_parallel_size=4, backend_options={"device": "cuda:2"}
    )
    backend._world, backend._rank = 4, 2
    assert backend._resolve_expert_device() == "cuda:0"

    backend, _ = _build(
        _checkpoint(tmp_path), tensor_parallel_size=4, backend_options={"expert_device": "cuda:1"}
    )
    assert backend._resolve_expert_device() == "cuda:1"


def test_a_rank_whose_card_the_loader_arithmetic_cannot_name_is_refused(tmp_path):
    """The loader adds the rank to the split's first card, so a rank must not sit below it."""
    backend, _ = _build(
        _checkpoint(tmp_path), tensor_parallel_size=4, backend_options={"device": "cuda:1"}
    )
    backend._world, backend._rank = 1, 3
    with pytest.raises(ConfigurationError, match="one node"):
        backend._resolve_expert_device()


def test_a_sharded_tree_lands_on_the_rank_own_card(tmp_path):
    """A host tree under a sharded run has no backend to collect over: NCCL carries no CPU tensor."""
    backend, _ = _build(_checkpoint(tmp_path))
    assert backend._tree_device() is None

    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank, backend._local_rank = 4, 2, 2
    assert backend._tree_device() == "cuda:2"

    # An explicit device still wins, sharded or not.
    backend, _ = _build(_checkpoint(tmp_path), backend_options={"device": "cuda:3"})
    backend._world, backend._rank, backend._local_rank = 4, 2, 2
    assert backend._tree_device() == "cuda:3"


# ---------------------------------------------------------------------------- prompts


def test_prompt_tokens_bypass_the_tokenizer(tmp_path):
    backend, tokenizer = _build(_checkpoint(tmp_path))
    request = GenerationRequest(request_id="r1", prompt_tokens=[4, 5, 6])

    assert backend._tokenize(request) == [4, 5, 6]
    assert tokenizer.calls == []


def test_a_raw_completion_prompt_goes_to_the_tokenizer_verbatim(tmp_path):
    """``/v1/completions`` is not chat: no header, no encoder, which is what the launcher does too."""
    backend, tokenizer = _build(_checkpoint(tmp_path))
    tokenizer.encoding["raw completion"] = [21, 22]

    assert backend._tokenize(_request()) == [7]
    assert backend._tokenize(GenerationRequest(request_id="r2", prompt="raw completion")) == [21, 22]
    assert tokenizer.calls[-1] == ("raw completion", True)


def test_a_chat_prompt_is_rendered_by_the_checkpoints_own_encoder(tmp_path):
    backend, tokenizer = _build(_checkpoint(tmp_path))
    rendered = "<b>user:hi</b>[thinking][max]"
    tokenizer.encoding[rendered] = [9, 10]

    prompt_ids = backend._tokenize(
        _request(
            messages=[{"role": "user", "content": "hi"}],
            thinking_mode="thinking",
            reasoning_effort="max",
        )
    )

    assert prompt_ids == [9, 10]
    # The rendered prompt already opens with its own beginning-of-sentence token, so a second one
    # added by the tokenizer is a token the model was never trained on.
    assert tokenizer.calls == [(rendered, False)]


def test_a_reasoning_budget_is_translated_before_it_reaches_the_encoder(tmp_path):
    backend, tokenizer = _build(_checkpoint(tmp_path))
    rendered = "<b>user:hi</b>[thinking][70]"
    tokenizer.encoding[rendered] = [9]

    assert backend._tokenize(
        _request(messages=[{"role": "user", "content": "hi"}], thinking_mode="thinking",
                 reasoning_effort=70)
    ) == [9]


def test_an_effort_the_scale_does_not_have_is_a_configuration_error(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path))
    with pytest.raises(ConfigurationError, match="reasoning_effort"):
        backend._tokenize(
            _request(messages=[{"role": "user", "content": "hi"}], thinking_mode="thinking",
                     reasoning_effort="banana")
        )


def test_a_checkpoint_without_an_encoder_refuses_a_chat_request_and_says_what_to_do(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path, encoder=False))
    with pytest.raises(ConfigurationError, match="/v1/completions"):
        backend._tokenize(_request(messages=[{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------- the request


def test_the_loop_gets_the_requests_sampling_and_this_runs_levers(tmp_path, loop):
    stub = loop(tokens=[11])
    backend, _ = _build(
        _checkpoint(tmp_path), max_model_len=4096, backend_options={"decode_graphs": True}
    )
    request = GenerationRequest(
        request_id="r1",
        prompt_tokens=[4, 5],
        sampling_params=SamplingParams(max_tokens=3, temperature=0.7, top_k=40, seed=99),
    )

    backend.generate([request])

    call = stub.calls[0]
    assert call["prompt_ids"] == [4, 5]
    assert call["max_new_tokens"] == 3
    assert call["temperature"] == 0.7
    assert call["top_k"] == 40
    assert call["seed"] == 99
    assert call["eos_token_id"] == 1
    assert call["graphs"] is True
    assert call["front"] is backend._front


class FakeDriver:
    """Stands in for ``graphs.DecodeGraphs``: what the adapter does with it is hand it back."""

    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


def test_the_graphs_a_request_recorded_are_handed_back_when_it_ends(tmp_path, loop):
    """A recording is one request's, and `Block.forward` hands the *next* prompt to whatever is
    installed on the blocks -- whose sink is one row wide, because that is what a decode step is."""
    driver = FakeDriver()
    loop(tokens=[11, 12], driver=driver)
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64, backend_options={"decode_graphs": True})

    backend.generate([_request()])

    assert driver.released == 1


def test_an_eager_request_has_no_graphs_to_hand_back(tmp_path, loop):
    stub = loop(tokens=[11, 12])
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)

    backend.generate([_request()])

    assert stub.calls[0]["graphs"] is False
    assert stub.driver is None


def test_a_generation_is_reported_with_its_usage_and_the_loops_own_decode_time(tmp_path, loop):
    loop(tokens=[11, 12, 13], stopped="eos", decode_seconds=0.6)
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    request = GenerationRequest(request_id="r1", prompt_tokens=[4, 5], sampling_params=SamplingParams(max_tokens=3))

    result = backend.generate([request])[0]

    assert result.token_ids == [11, 12, 13]
    assert result.finish_reason == "stop"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (2, 3)
    # The loop's figure is the decode steps alone, so what is left of the wall is the prompt.
    assert result.timings.decode_seconds == 0.6
    assert result.timings.prefill_seconds >= 0.0
    assert result.timings.tpot_seconds == pytest.approx(0.2)
    assert result.metadata["stopped"] == "eos"
    assert backend.active_request_count() == 0


def test_hitting_max_tokens_is_a_length_finish_not_a_stop(tmp_path, loop):
    loop(tokens=[11, 12], stopped="length")
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)

    result = backend.generate(
        [GenerationRequest(request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=2))]
    )[0]

    assert result.finish_reason == "length"
    assert result.metadata["stopped"] == "length"


def test_a_thinking_generation_is_split_by_the_checkpoints_parser(tmp_path, loop):
    """The parser wins when it accepts the text: it is also what reads tool calls back."""
    tokenizer = FakeTokenizer(
        pieces={11: "why", 12: "</think>", 13: "answer", 14: "<tool>", 1: "<e>"}
    )
    loop(tokens=[11, 12, 13, 14, 1])
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=4),
        metadata={"thinking_mode": "thinking"},
    )

    result = backend.generate([request])[0]

    # The parse ran on the raw decode -- which still carries the end-of-sentence token, or the
    # parser would have refused -- and the answer excludes it by construction.
    assert result.text == "answer<tool>"
    assert result.metadata["reasoning_content"] == "why"
    assert result.metadata["tool_calls"][0]["function"]["name"] == "weather"


def test_a_generation_the_parser_refuses_falls_back_to_the_tolerant_split(tmp_path, loop):
    """A ``max_tokens`` cut is the ordinary case, so a refusal is not an error to report."""
    tokenizer = FakeTokenizer(pieces={11: "why", 12: "</think>", 13: "half"})
    loop(tokens=[11, 12, 13], stopped="length")
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=3),
        metadata={"thinking_mode": "thinking"},
    )

    result = backend.generate([request])[0]

    assert result.text == "half"
    assert result.metadata["reasoning_content"] == "why"
    assert "tool_calls" not in result.metadata


def test_a_stop_string_ends_the_generation_at_the_marker(tmp_path, loop):
    tokenizer = FakeTokenizer(pieces={11: "answer", 12: "STOP", 13: "trailing", 1: "<e>"})
    loop(tokens=[11, 12, 13])
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4],
        sampling_params=SamplingParams(max_tokens=3, stop=("STOP",)),
    )

    result = backend.generate([request])[0]

    # The text is the answer; the tokens past the marker are not, and the result does not claim
    # they are a completion the client should see.
    assert result.text == "answer"
    assert result.token_ids == [11, 12]
    assert result.usage.completion_tokens == 2
    assert result.finish_reason == "stop"
    assert result.metadata["stopped"] == "stop"
    # No `Generation` came back, so the loop's own figure is gone: the wall minus the first token's
    # wait is what is left of it, and the prompt's forward is that wait.
    assert result.timings.ttft_seconds == pytest.approx(result.timings.prefill_seconds)
    assert result.timings.decode_seconds == pytest.approx(
        result.timings.total_seconds - result.timings.ttft_seconds
    )


def test_a_stop_string_the_client_asked_for_is_matched_against_what_the_client_is_sent(
    tmp_path, loop
):
    """The running text is decoded, not the token's own piece: a byte-level piece can be half a
    character, so the match runs on the text a client would have received."""
    tokenizer = FakeTokenizer(pieces={11: "a", 12: "b", 13: "c"})
    stub = loop(tokens=[11, 12, 13])
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4],
        sampling_params=SamplingParams(max_tokens=3, stop=("abc",)),
    )

    result = backend.generate([request])[0]

    assert result.text == ""
    assert result.metadata["stopped"] == "stop"
    assert stub.calls  # the loop ran; the stop is a token-boundary unwind, not a refusal


# ---------------------------------------------------------------------------- cancellation


def test_a_request_cancelled_before_the_loop_never_reaches_it(tmp_path, loop):
    stub = loop(tokens=[11])
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    request = GenerationRequest(request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=2))

    backend._begin_request("r1")
    assert backend.cancel("r1") is True
    with pytest.raises(RequestCancelledError):
        backend.generate([request])

    assert stub.calls == []
    assert backend.active_request_count() == 0


def test_a_cancel_that_lands_mid_loop_unwinds_at_a_token_boundary(tmp_path, loop):
    """The hook is the only place every rank reaches once a token, so it is where this is decided."""
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    stub = loop(tokens=[11, 12, 13], before_token=lambda: backend.cancel("r1"))
    request = GenerationRequest(request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=3))

    with pytest.raises(RequestCancelledError, match="cancelled"):
        backend.generate([request])

    assert stub.calls  # the loop started, and unwound inside it
    assert backend.active_request_count() == 0


def test_cancelling_a_request_that_is_not_running_is_a_no_op(tmp_path):
    backend, _ = _build(_checkpoint(tmp_path))
    assert backend.cancel("never-seen") is False


# ---------------------------------------------------------------------------- length


def test_a_request_the_caches_cannot_hold_is_refused_before_any_rank_is_told(tmp_path, loop):
    stub = loop(tokens=[11])
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=16)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=list(range(10)), sampling_params=SamplingParams(max_tokens=100)
    )

    with pytest.raises(ConfigurationError, match="attention caches were sized at 16"):
        backend.generate([request])

    assert stub.calls == []


def test_a_refusal_on_rank_zero_never_reaches_the_broadcast(tmp_path, loop):
    """Rank 0 refusing *after* the broadcast is a peer blocked on a request that is not coming."""
    stub = loop(tokens=[11])
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=16, tensor_parallel_size=4)
    backend._world, backend._rank = 4, 0
    broadcast: list[object] = []
    backend._broadcast = broadcast.append
    request = GenerationRequest(
        request_id="r1", prompt_tokens=list(range(10)), sampling_params=SamplingParams(max_tokens=100)
    )

    with pytest.raises(ConfigurationError):
        backend.generate([request])

    assert broadcast == []
    assert stub.calls == []


def test_the_shard_hands_the_workers_the_same_request_it_runs_itself(tmp_path, loop, monkeypatch):
    import torch.distributed as dist

    # The per-step agreement is a real collective and there is no process group here. What this
    # test reads is the payload that travels to the workers, and the flag rank 0 sends is its own.
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)

    stub = loop(tokens=[11])
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64, tensor_parallel_size=4)
    backend._world, backend._rank = 4, 0
    broadcast: list[object] = []
    backend._broadcast = broadcast.append
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4, 5],
        sampling_params=SamplingParams(max_tokens=2),
        metadata={"thinking_mode": "thinking"},
    )

    backend.generate([request])

    assert len(broadcast) == 1
    payload = broadcast[0]
    assert payload["op"] == "generate"
    assert payload["prompt_ids"] == [4, 5]
    assert payload["max_new_tokens"] == 2
    # The reading of the answer travels with the prompt so a worker unwinds on the same one.
    assert payload["thinking_mode"] == "thinking"
    assert stub.calls  # and rank 0 ran it too, on the same payload


# ---------------------------------------------------------------------------- streaming


def _drain(backend, request):
    return list(backend.stream(request))


def test_a_thinking_stream_splits_reasoning_from_the_answer_as_it_arrives(tmp_path, loop):
    tokenizer = FakeTokenizer(pieces={11: "why", 12: "</think>", 13: "answer"})
    loop(tokens=[11, 12, 13], decode_seconds=0.3)
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=3),
        metadata={"thinking_mode": "thinking"},
    )

    events = _drain(backend, request)

    tokens = [event for event in events if event.token_id is not None]
    assert [event.token_id for event in tokens] == [11, 12, 13]
    # The reasoning goes out as it arrives rather than waiting for the answer, and each half is the
    # piece that half grew by -- not the whole text again.
    assert [event.metadata.get("reasoning_content", "") for event in tokens] == ["why", "", ""]
    assert "".join(event.text for event in tokens) == "answer"
    last = events[-1]
    assert last.finish_reason == "stop"
    assert last.usage.completion_tokens == 3
    assert backend.active_request_count() == 0


def test_a_chat_stream_sends_every_token_as_content(tmp_path, loop):
    tokenizer = FakeTokenizer(pieces={11: "hello ", 12: "there"})
    loop(tokens=[11, 12])
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=2)
    )

    events = _drain(backend, request)

    assert "".join(event.text for event in events if event.token_id is not None) == "hello there"
    assert all(not event.metadata for event in events if event.token_id is not None)


def test_a_stream_reports_a_failed_loop_in_band_rather_than_truncating(tmp_path, loop):
    """The HTTP layer turns this into an error event; swallowing it would end the stream as a
    short answer and a 200."""
    loop(failure=RuntimeError("the loop blew up"))
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    request = GenerationRequest(request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=2))

    with pytest.raises(RuntimeError, match="the loop blew up"):
        _drain(backend, request)

    assert backend.active_request_count() == 0


def test_a_stream_that_is_abandoned_stops_the_producer(tmp_path, loop):
    """A client that stops reading must not leave the loop holding the request lock forever."""
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    loop(tokens=list(range(11, 11 + 8)))
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=8)
    )

    stream = backend.stream(request)
    next(stream)
    stream.close()

    # The generator's own cleanup marks the request cancelled, which is what unwinds the producer
    # at its next token and lets the lock go.
    assert backend.active_request_count() == 0
    assert backend.cancel("r1") is False


# ---------------------------------------------------------------------------- multi-rank


def test_the_flag_lives_on_the_card_the_attention_caches_are_on(tmp_path):
    """Read the way ``generate._cache_device`` reads it: nothing promises this process's current
    device is the tree's."""
    buffers = [("layers.0.attention.window_kv_cache", FakeBuffer("cuda:3"))]
    backend, _ = _build(_checkpoint(tmp_path), buffers=buffers)
    assert backend._flag_device() == "cuda:3"

    backend, _ = _build(_checkpoint(tmp_path), buffers=[("layers.0.other", FakeBuffer("cuda:3"))])
    assert str(backend._flag_device()) == "cpu"


def test_close_hands_the_workers_a_shutdown_before_the_group_is_reaped(tmp_path):
    """Order is the whole point: after ``super().close()`` there is nobody left to receive it."""
    order: list[str] = []

    class FakeSupervisor:
        def cleanup(self):
            order.append("cleanup")

    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 0
    backend._supervisor = FakeSupervisor()

    def broadcast(payload):
        order.append("broadcast")
        backend._shutdown = payload

    backend._broadcast = broadcast
    backend.close()

    assert order == ["broadcast", "cleanup"]
    assert backend._shutdown == {"op": "shutdown"}


def test_a_worker_rank_has_no_one_to_broadcast_a_shutdown_to(tmp_path):
    sent: list[object] = []
    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 1
    backend._broadcast = sent.append

    backend.close()

    assert sent == []


def test_run_worker_refuses_the_single_process_and_rank_zero_cases(tmp_path, monkeypatch):
    backend, _ = _build(_checkpoint(tmp_path))
    with pytest.raises(UnsupportedFeatureError, match="single-process"):
        backend.run_worker()

    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 0
    monkeypatch.setattr(backend, "_init_distributed", lambda: None)
    with pytest.raises(UnsupportedFeatureError, match="must not be called on rank 0"):
        backend.run_worker()


def test_a_worker_serves_rank_zeros_requests_until_it_is_told_to_stop(tmp_path, monkeypatch):
    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 1
    monkeypatch.setattr(backend, "_init_distributed", lambda: None)
    payload = {"op": "generate", "request_id": "r1", "prompt_ids": [4]}
    mailbox = [payload, {"op": "shutdown"}]
    backend._broadcast = lambda _: mailbox.pop(0)
    served: list[tuple] = []
    monkeypatch.setattr(backend, "_run_payload", lambda *args: served.append(args))
    announced: list[bool] = []

    backend.run_worker(on_ready=lambda: announced.append(True))

    assert announced == [True]
    assert len(served) == 1
    assert served[0] == (payload, None, None)
    assert backend._closed is False


def test_a_worker_that_unwinds_with_rank_zero_does_not_desynchronize(tmp_path, monkeypatch):
    """Cancelled and stopped are both agreed per step, so neither is an error on a worker rank."""
    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 1
    monkeypatch.setattr(backend, "_init_distributed", lambda: None)
    mailbox = [{"op": "generate", "request_id": "r1"}, {"op": "shutdown"}]
    backend._broadcast = lambda _: mailbox.pop(0)
    calls: list[int] = []

    def run_payload(*_args):
        calls.append(1)
        raise RequestCancelledError("r1 was cancelled")

    monkeypatch.setattr(backend, "_run_payload", run_payload)

    backend.run_worker()

    assert calls == [1]
    assert mailbox == []


def test_the_worker_loop_reads_one_shape_of_message(tmp_path, monkeypatch):
    """A message that is not a request keeps the loop alive; one that is not a message ends it."""
    backend, _ = _build(_checkpoint(tmp_path), tensor_parallel_size=4)
    backend._world, backend._rank = 4, 1
    monkeypatch.setattr(backend, "_init_distributed", lambda: None)
    mailbox = [{"op": "other"}, None]
    backend._broadcast = lambda _: mailbox.pop(0)
    monkeypatch.setattr(backend, "_run_payload", lambda *a: pytest.fail("not a request"))

    backend.run_worker()

    assert mailbox == []


# ---------------------------------------------------------------------------- the split


@pytest.mark.parametrize(
    "text, thinking, expected",
    [
        ("answer", False, ("", "answer")),
        ("answer", True, ("answer", "")),
        ("why</think>answer", True, ("why", "answer")),
        ("why</think>", True, ("why", "")),
        ("</think>answer", True, ("", "answer")),
    ],
)
def test_the_running_split(text, thinking, expected):
    assert _split_running(text, thinking) == expected


def test_the_split_is_recomputed_and_not_tracked_across_tokens():
    """Relied on by the streaming diff: the marker's offset moves as the pieces under it settle."""
    assert _split_running("wh", True) == ("wh", "")
    assert _split_running("why</think>an", True) == ("why", "an")


def test_a_stream_and_a_generation_of_the_same_tokens_agree_on_the_answer(tmp_path, loop):
    """Two paths, one reading: the stream diff and the finished split must not disagree."""
    tokenizer = FakeTokenizer(pieces={11: "why", 12: "</think>", 13: "answer"})
    loop(tokens=[11, 12, 13])
    backend, _ = _build(_checkpoint(tmp_path), tokenizer=tokenizer, max_model_len=64)

    def request():
        return GenerationRequest(
            request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=3),
            metadata={"thinking_mode": "thinking"},
        )

    events = _drain(backend, request())
    streamed = "".join(event.text for event in events if event.token_id is not None)
    streamed_reasoning = "".join(
        event.metadata.get("reasoning_content", "") for event in events if event.token_id is not None
    )
    result = backend.generate([request()])[0]

    assert streamed == result.text == "answer"
    assert streamed_reasoning == result.metadata["reasoning_content"] == "why"


def test_the_streaming_producer_and_the_consumer_are_different_threads(tmp_path, loop):
    """The loop has to run somewhere the HTTP thread can still reach `cancel`."""
    backend, _ = _build(_checkpoint(tmp_path), max_model_len=64)
    seen: list[str] = []

    def record():
        seen.append(threading.current_thread().name)

    loop(tokens=[11], before_token=record)
    request = GenerationRequest(
        request_id="r1", prompt_tokens=[4], sampling_params=SamplingParams(max_tokens=1)
    )

    _drain(backend, request)

    assert seen == ["pocketllm-v41-stream"]
