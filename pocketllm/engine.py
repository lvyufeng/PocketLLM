"""High-level synchronous and asynchronous PocketLLM facades."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
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
