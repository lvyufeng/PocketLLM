from __future__ import annotations

import asyncio

import pytest

from pocketllm import (
    BackendCapabilities,
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    LLM,
    SamplingParams,
    TokenEvent,
    Usage,
)
from pocketllm.backends.base import BackendBase
from pocketllm.backends.factory import create_backend


# The facade uses the factory in production; tests replace it with a small
# dependency-free backend so API behavior is testable without a checkpoint.


_original_create_backend = create_backend



class FakeBackend(BackendBase):
    def __init__(self):
        super().__init__()
        self._ready = True

    @property
    def capabilities(self):
        return BackendCapabilities(
            name="fake", supports_streaming=True, supports_cancellation=True,
        )

    def generate(self, requests):
        result = []
        for request in requests:
            self._begin_request(request.request_id)
            try:
                self._check_cancelled(request.request_id)
                result.append(GenerationResult(
                    request_id=request.request_id,
                    token_ids=[1, 2],
                    text=request.prompt or "",
                    usage=Usage(3, 2),
                ))
            finally:
                self._clear_request(request.request_id)
        return result

    def stream(self, request):
        self._begin_request(request.request_id)
        try:
            self._check_cancelled(request.request_id)
            yield TokenEvent(request.request_id, token_id=1, text="a")
            self._check_cancelled(request.request_id)
            yield TokenEvent(request.request_id, token_id=2, text="b", finish_reason="stop", usage=Usage(1, 2))
        finally:
            self._clear_request(request.request_id)


class InjectedLLM(LLM):
    def __init__(self):
        self.args = EngineArgs(model="fake", backend="torch")
        self._backend = FakeBackend()
        self._closed = False


def test_engine_args_and_sampling_aliases():
    args = EngineArgs(model="model", backend="torch", tensor_parallel_size=2, tensor_parallel_rank=1)
    assert args.checkpoint_dir == "model"
    params = SamplingParams.from_openai({"max_completion_tokens": 7, "stop": "END", "n": 2})
    assert params.max_tokens == 7
    assert params.stop == ("END",)
    assert params.n == 2


def test_invalid_public_options_fail_early():
    with pytest.raises(ConfigurationError):
        EngineArgs(model="x", backend="bad")
    with pytest.raises(ConfigurationError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ConfigurationError):
        GenerationRequest()


def test_fake_backend_sync_and_stream():
    llm = InjectedLLM()
    result = llm.generate(["hello"])[0]
    assert result.text == "hello"
    assert result.usage.as_dict() == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    events = list(llm.generate_stream("stream"))
    assert "".join(event.text for event in events) == "ab"
    assert events[-1].finish_reason == "stop"
    llm.close()
    with pytest.raises(RuntimeError):
        llm.generate("closed")


def test_cancel_requires_an_active_request():
    llm = InjectedLLM()
    assert llm.cancel("never-submitted") is False
    stream = llm.generate_stream("stream")
    first = next(stream)
    assert llm.cancel(first.request_id) is True
    llm.close()


def test_async_facade_with_fake_backend():
    from pocketllm.engine import AsyncLLM

    async def run():
        async_llm = AsyncLLM.__new__(AsyncLLM)
        async_llm._llm = InjectedLLM()
        from concurrent.futures import ThreadPoolExecutor
        async_llm._executor = ThreadPoolExecutor(max_workers=1)
        async_llm._closed = False
        result = (await async_llm.generate("hello"))[0]
        assert result.text == "hello"
        values = []
        async for event in async_llm.generate_stream("x"):
            values.append(event.text)
        assert values == ["a", "b"]
        await async_llm.close()

    asyncio.run(run())
