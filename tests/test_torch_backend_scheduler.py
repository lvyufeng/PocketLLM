"""The Torch adapter reaches the same `BatchScheduler` the `cpp` backend drives.

`--backend torch` is what serves every checkpoint the four specific adapters do not claim, so this
is the last request lifecycle outside the shared scheduler and these tests are about closing it.

The engine is scripted rather than loaded: what is under test is the route, and the two arms are
driven by the same fake runtime, so anything that differs between them is the path and not the
model. The scheduler itself is the real one -- a fake of it would show that this adapter calls
something, not that the something accepts a Python engine and drives it a token at a time.
"""

from __future__ import annotations

import importlib

import pytest

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.torch_backend import TorchBackend

#: The tokens both arms produce. Three of them, so a decode step and a stop are both exercised.
TOKENS = [11, 12, 13]


@pytest.fixture(scope="module")
def native_module():
    """The C++ extension, which is what carries the scheduler binding."""
    try:
        return importlib.import_module("pocketllm_cpp")
    except ImportError as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"native pocketllm_cpp module is not built: {exc}")


class ScriptedTokenizer:
    #: Deliberately not one of `TOKENS`. A run that stops on the checkpoint's own end-of-text is a
    #: different test (the scheduler retires the row and the runtime's loop ends); here every script
    #: has to run to its budget so that the two arms are compared on the same three tokens.
    eos_token_id = 99

    def decode(self, token_ids) -> str:
        return "".join(f"<{token}>" for token in token_ids)

    def encode(self, text: str) -> list[int]:
        return [7, 8]


class ScriptedServingEngine:
    """The legacy runtime, with both of its shapes.

    `submit` answers the whole request and `submit_stream` answers one event per decode step, which
    is the difference the scheduler cares about: it drives one token per call and needs the second
    shape. The events are what the scheduler is fed, and they are what the streaming HTTP route is
    fed too -- the adapter reads the same ones, which is where the two routes' agreement comes from.
    """

    def __init__(self, tokens=None) -> None:
        self.tokens = list(TOKENS if tokens is None else tokens)
        self.payloads: list[dict] = []

    def submit(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return self._formatted(payload)

    def submit_stream(self, payload: dict):
        self.payloads.append(payload)
        for token in self.tokens:
            yield {"type": "token", "token_ids": [token]}
        yield self._done(payload, self.tokens)

    def _done(self, payload: dict, tokens) -> dict:
        prompt_ids = payload["_prompt_ids"]
        return {
            "type": "done",
            "completion_tokens": [list(tokens)],
            "prefill_time": 0.5,
            "decode_time": 0.25,
            "prefill_tokens": len(prompt_ids),
            "decode_tokens": max(0, len(tokens) - 1),
            "finish_reason": "stop",
        }

    def _formatted(self, payload: dict) -> dict:
        """The serial path's answer, built by the legacy path's own formatter.

        Computed here rather than in the adapter so the two arms are compared against a mapping
        this test produced, not against one the code under test happened to hand both of them.
        """
        from src.server.openai import _format_completion_result

        prompt_ids = payload["_prompt_ids"]
        return _format_completion_result(
            TOKENIZER,
            str(payload["thinking_mode"]),
            prompt_ids,
            list(self.tokens),
            0.5,
            0.25,
            len(prompt_ids),
            max(0, len(self.tokens) - 1),
            stop=payload.get("stop"),
            max_tokens=payload.get("max_tokens"),
        )

    def close(self) -> None:
        pass


TOKENIZER = ScriptedTokenizer()


def backend(**options) -> TorchBackend:
    """This runtime on the legacy path: the queue, and no scheduler."""
    return _backend(enable_batching=False, **options)


def batched_backend(engine=None, **options) -> TorchBackend:
    """This runtime serving through the scheduler `cpp` uses."""
    return _backend(enable_batching=True, engine=engine, **options)


def _backend(*, enable_batching: bool, engine=None, **options) -> TorchBackend:
    args = EngineArgs(
        model="a-torch-checkpoint",
        backend="torch",
        # Named by both arms, and different in kind: this is the bound the scheduler checks a
        # prompt against, and on the legacy path it only sizes the default budget.
        max_model_len=64,
        backend_options={"enable_batching": enable_batching, **options},
    )
    return TorchBackend(
        args,
        runtime={"tokenizer": TOKENIZER, "model_id": "scripted"},
        serving_engine=engine if engine is not None else ScriptedServingEngine(),
    )


def request(request_id: str = "r1", **metadata) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=[7, 8],
        # The scripted runtime produces exactly this many tokens, so the row ends on its budget
        # rather than on a step that never comes.
        sampling_params=SamplingParams(max_tokens=len(TOKENS)),
        metadata={"thinking_mode": "chat", **metadata},
    )


# ---------------------------------------------------------------- the route, and what it costs


def test_the_two_routes_declare_themselves_apart(native_module) -> None:
    """`supports_batch` and the detail both say which path is live, and say it the same way."""
    serial = backend()
    batched = batched_backend()
    try:
        assert serial.capabilities.supports_batch is False
        assert serial.capabilities.details["scheduler"] == "legacy serving queue"
        assert batched.capabilities.supports_batch is True
        assert "BatchScheduler" in batched.capabilities.details["scheduler"]
    finally:
        batched.close()


def test_asking_for_the_scheduler_without_a_context_serves_anyway(native_module) -> None:
    """The refusal is a warning and the serialized path, because it is the scheduler that cannot
    answer the question -- not this runtime, which serves fine without one."""
    args = EngineArgs(
        model="a-torch-checkpoint",
        backend="torch",
        backend_options={"enable_batching": True},
    )
    with pytest.warns(UserWarning, match="needs --max-model-len"):
        instance = TorchBackend(
            args,
            runtime={"tokenizer": TOKENIZER, "model_id": "scripted"},
            serving_engine=ScriptedServingEngine(),
        )
    assert instance.capabilities.supports_batch is False
    instance.close()


def test_the_scheduler_reads_the_runtime_s_own_declaration(native_module) -> None:
    """One row, and one because the runtime said so rather than because a constant kept them apart."""
    batched = batched_backend()
    try:
        caps = batched._scheduler.engine_caps()
        assert caps.max_slots == 1
        assert caps.continuous_batching is False
        assert caps.chunked_prefill is False
        assert caps.paged_kv is False
    finally:
        batched.close()


# ------------------------------------------------------------------------------- the answers


def test_the_batched_path_answers_what_the_serial_path_answers(native_module) -> None:
    """Turning the scheduler on changes the route, not the answer.

    The two arms are the same scripted runtime and the same request, so the only thing that can
    differ is which path ran -- and on this plane the answer is not the decoded tokens but the
    parsed assistant message, which is why the adapter routes both through the legacy formatter.
    """
    engine = ScriptedServingEngine()
    serial = backend(engine=ScriptedServingEngine())
    batched = batched_backend(engine=engine)
    try:
        one = serial.generate([request()])[0]
        two = batched.generate([request()])[0]

        assert two.text == one.text
        assert two.finish_reason == one.finish_reason
        assert two.usage.prompt_tokens == one.usage.prompt_tokens
        assert two.usage.completion_tokens == one.usage.completion_tokens
        # Not `token_ids`: this plane's serial path answers with the parsed message and carries no
        # ids at all, and the routed path has to answer the same way rather than more informatively.
        assert one.token_ids == two.token_ids == []
        # The split the scheduler keeps for its own gauges reaches the client rather than being
        # zeroed on the way through a result type this runtime did not build.
        assert two.timings.total_seconds > 0
    finally:
        batched.close()


def test_every_token_the_runtime_streams_is_one_the_scheduler_counted(native_module) -> None:
    """The events are the answer on both routes, so a token cannot go missing between them.

    The streaming HTTP route builds its own text from the same accumulated event ids (see
    `_StreamingDecoder.final_message`), so the count here is the count a streamed client gets. A
    scheduler fed fewer tokens than the runtime produced would answer a shorter completion and call
    it finished, which is the failure this pins.
    """
    batched = batched_backend()
    try:
        result = batched.generate([request()])[0]
        assert len(TOKENS) == result.usage.completion_tokens
        # And the row is retired rather than left running, which is the same reading the gauges
        # give: a scheduler that kept the slot occupied would report it in the next request.
        assert batched.metrics()["requests_running"] == 0.0
    finally:
        batched.close()


# -------------------------------------------------------------------------------- the gauges


def test_the_scheduler_gauges_are_published_only_where_a_scheduler_is(native_module) -> None:
    """The reading that separates a scheduler from a queue, and it has to be absent on the queue.

    Both paths answer the same request, so this series is the only in-process evidence that a
    request went through the scheduler rather than through the legacy queue. The serialized path
    exports none of it -- not as zero, which would read as "the scheduler is here and idle" about a
    process that has none.
    """
    batched = batched_backend()
    serial = backend()
    try:
        assert "requests_running" not in serial.metrics()

        gauges = batched.metrics()
        assert gauges["requests_running"] == 0.0
        assert gauges["requests_waiting"] == 0.0
        assert gauges["slots_free"] == 1.0
        # No block pool on this plane either, so the pool gauges are not published at all.
        assert not [key for key in gauges if key.startswith("kv_blocks")]
    finally:
        batched.close()
