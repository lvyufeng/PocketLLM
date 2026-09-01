"""Prompt rendering tests for the Torch adapter.

These inject a fake serving engine so no checkpoint, CUDA device, or real
runtime is required; only the payload the adapter builds is inspected.
"""

from __future__ import annotations

from pocketllm.api import EngineArgs, GenerationRequest, SamplingParams
from pocketllm.backends.torch_backend import TorchBackend
from src.encoding.dsv4 import encode_messages


class RecordingTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return [11, 12]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class RecordingServingEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def submit(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"content": "ok", "prompt_tokens": 2, "completion_tokens": 1, "finish_reason": "stop"}

    def submit_stream(self, payload: dict):
        self.payloads.append(payload)
        yield {"type": "token", "token_ids": [11]}
        yield {"type": "done", "prompt_tokens": 2, "completion_tokens": [[11]], "finish_reason": "stop"}

    def close(self) -> None:
        pass


def _backend(tokenizer: RecordingTokenizer) -> tuple[TorchBackend, RecordingServingEngine]:
    engine = RecordingServingEngine()
    backend = TorchBackend(
        EngineArgs(model="model", backend="torch"),
        runtime={"tokenizer": tokenizer, "model_id": "fake-model"},
        serving_engine=engine,
    )
    return backend, engine


def test_chat_metadata_is_rendered_with_the_deepseek_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-chat",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    backend.generate([request])

    # The adapter must reuse the runtime's own chat encoding rather than the
    # server's plain fallback text.
    assert tokenizer.encoded == [encode_messages(messages, thinking_mode="chat")]
    # The normalized messages are forwarded so the runtime can re-encode.
    assert engine.payloads[-1]["messages"] == messages
    assert engine.payloads[-1]["thinking_mode"] == "chat"
    backend.close()


def test_thinking_mode_and_effort_reach_the_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-thinking",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "thinking", "reasoning_effort": "max"},
    )

    backend.generate([request])

    assert tokenizer.encoded == [
        encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")
    ]
    assert engine.payloads[-1]["reasoning_effort"] == "max"
    backend.close()


def test_raw_prompts_are_encoded_unchanged():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt="raw completion prompt",
        request_id="req-raw",
        sampling_params=SamplingParams(max_tokens=1),
    )

    backend.generate([request])

    assert tokenizer.encoded == ["raw completion prompt"]
    assert engine.payloads[-1]["messages"] == [{"role": "user", "content": "raw completion prompt"}]
    backend.close()


def test_prompt_tokens_bypass_the_tokenizer():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt_tokens=[7, 8],
        request_id="req-tokens",
        sampling_params=SamplingParams(max_tokens=1),
    )

    backend.generate([request])

    assert tokenizer.encoded == []
    assert engine.payloads[-1]["_prompt_ids"] == [7, 8]
    backend.close()


def test_streaming_uses_the_same_prompt_rendering():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    list(backend.stream(request))

    assert tokenizer.encoded[0] == encode_messages(messages, thinking_mode="chat")
    backend.close()
