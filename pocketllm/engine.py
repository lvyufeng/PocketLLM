"""High-level synchronous and asynchronous PocketLLM facades."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from pocketllm.api import (
    BackendCapabilities,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    SamplingParams,
    TokenEvent,
)
from pocketllm.backends.factory import create_backend
from pocketllm.protocol import build_chat_request


_TOKEN_DONE = object()


def _next_event(iterator: Iterator[TokenEvent]) -> TokenEvent | object:
    try:
        return next(iterator)
    except StopIteration:
        return _TOKEN_DONE


class LLM:
    """A vLLM-like facade over one selected PocketLLM backend."""

    def __init__(self, args: EngineArgs | None = None, **kwargs: Any) -> None:
        self.args = args if args is not None else EngineArgs(**kwargs)
        if kwargs and args is not None:
            raise TypeError("pass either EngineArgs or keyword options, not both")
        self._backend = create_backend(self.args)
        self._closed = False

    @property
    def backend(self):
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.capabilities.name

    def capabilities(self) -> BackendCapabilities:
        return self._backend.capabilities

    def health(self) -> HealthStatus:
        return self._backend.health()

    @staticmethod
    def _requests(prompts: Sequence[str | Sequence[int]], params: SamplingParams) -> list[GenerationRequest]:
        requests = []
        for prompt in prompts:
            if isinstance(prompt, str):
                requests.append(GenerationRequest(prompt=prompt, sampling_params=params))
            else:
                requests.append(GenerationRequest(prompt_tokens=[int(token) for token in prompt], sampling_params=params))
        return requests

    def generate(
        self,
        prompts: str | Sequence[str | Sequence[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[GenerationResult]:
        if self._closed:
            raise RuntimeError("LLM is closed")
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()
        return self._backend.generate(self._requests(prompts, params))

    def generate_stream(
        self,
        prompt: str | Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[TokenEvent]:
        if self._closed:
            raise RuntimeError("LLM is closed")
        params = sampling_params or SamplingParams()
        request = self._requests([prompt], params)[0]
        return self._backend.stream(request)

    @staticmethod
    def _chat_body(
        messages: Sequence[Mapping[str, Any]],
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> dict[str, Any]:
        """Build an OpenAI-shaped body for the shared chat normalizer."""
        normalized_messages = [
            dict(message) if isinstance(message, Mapping) else message
            for message in messages
        ]
        body: dict[str, Any] = {"messages": normalized_messages}
        for name, value in (
            ("reasoning", reasoning),
            ("reasoning_effort", reasoning_effort),
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("response_format", response_format),
        ):
            if value is not None:
                body[name] = value
        return body

    def _chat_request(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams | None = None,
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> GenerationRequest:
        body = self._chat_body(
            messages,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )
        return build_chat_request(body, sampling_params, request_id=request_id)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams | None = None,
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> list[GenerationResult]:
        """Generate a response for normalized OpenAI-style chat messages."""
        if self._closed:
            raise RuntimeError("LLM is closed")
        request = self._chat_request(
            messages,
            sampling_params,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            request_id=request_id,
        )
        return self._backend.generate([request])

    def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams | None = None,
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> Iterator[TokenEvent]:
        """Stream token events for normalized OpenAI-style chat messages."""
        if self._closed:
            raise RuntimeError("LLM is closed")
        request = self._chat_request(
            messages,
            sampling_params,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            request_id=request_id,
        )
        return self._backend.stream(request)

    def cancel(self, request_id: str) -> bool:
        return self._backend.cancel(request_id)

    def close(self) -> None:
        if not self._closed:
            self._backend.close()
            self._closed = True

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AsyncLLM:
    """Async facade using a worker executor around the backend contract.

    This provides non-blocking application integration now.  It intentionally
    does not advertise true multi-request device batching; that remains a
    backend scheduler feature.
    """

    def __init__(self, args: EngineArgs | None = None, *, max_workers: int = 1, **kwargs: Any) -> None:
        self._llm = LLM(args, **kwargs)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="pocketllm-async"
        )
        self._closed = False

    @property
    def backend_name(self) -> str:
        return self._llm.backend_name

    def capabilities(self) -> BackendCapabilities:
        return self._llm.capabilities()

    def health(self) -> HealthStatus:
        return self._llm.health()

    async def generate(
        self,
        prompts: str | Sequence[str | Sequence[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[GenerationResult]:
        if self._closed:
            raise RuntimeError("AsyncLLM is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._llm.generate, prompts, sampling_params)

    async def generate_stream(
        self,
        prompt: str | Sequence[int],
        sampling_params: SamplingParams | None = None,
    ) -> AsyncIterator[TokenEvent]:
        if self._closed:
            raise RuntimeError("AsyncLLM is closed")
        loop = asyncio.get_running_loop()
        iterator = await loop.run_in_executor(
            self._executor,
            self._llm.generate_stream,
            prompt,
            sampling_params,
        )
        while True:
            event = await loop.run_in_executor(self._executor, _next_event, iterator)
            if event is _TOKEN_DONE:
                return
            yield event

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams | None = None,
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> list[GenerationResult]:
        if self._closed:
            raise RuntimeError("AsyncLLM is closed")
        loop = asyncio.get_running_loop()
        call = partial(
            self._llm.chat,
            messages,
            sampling_params,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            request_id=request_id,
        )
        return await loop.run_in_executor(self._executor, call)

    async def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        sampling_params: SamplingParams | None = None,
        *,
        reasoning: Any = None,
        reasoning_effort: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        request_id: str | None = None,
    ) -> AsyncIterator[TokenEvent]:
        if self._closed:
            raise RuntimeError("AsyncLLM is closed")
        loop = asyncio.get_running_loop()
        call = partial(
            self._llm.chat_stream,
            messages,
            sampling_params,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            request_id=request_id,
        )
        iterator = await loop.run_in_executor(self._executor, call)
        while True:
            event = await loop.run_in_executor(self._executor, _next_event, iterator)
            if event is _TOKEN_DONE:
                return
            yield event

    async def cancel(self, request_id: str) -> bool:
        if self._closed:
            return False
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._llm.cancel, request_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._llm.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    async def __aenter__(self) -> "AsyncLLM":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
