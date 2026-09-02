from __future__ import annotations

import copy

import pytest

from pocketllm.api import SamplingParams
from pocketllm.protocol import build_chat_request, render_fallback_prompt


def test_build_chat_request_accepts_openai_sampling_fields_without_explicit_params():
    request = build_chat_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0.4,
            "stop": "END",
        }
    )

    assert request.sampling_params.max_tokens == 5
    assert request.sampling_params.temperature == 0.4
    assert request.sampling_params.stop == ("END",)

def test_build_chat_request_preserves_normalized_chat_fields_and_sampling():
    body = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ],
        "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "function": {"name": "calculator"}}],
        "tool_choice": "required",
        "response_format": {"type": "json_object"},
    }

    request = build_chat_request(body, SamplingParams(max_tokens=7), request_id="chat-7")

    assert request.request_id == "chat-7"
    assert request.sampling_params.max_tokens == 7
    assert request.metadata["thinking_mode"] == "thinking"
    assert request.metadata["reasoning_effort"] == "high"
    assert request.metadata["response_format"] == {"type": "json_object"}
    assert request.metadata["tools"] == body["tools"]
    assert request.metadata["messages"][0]["tools"] == body["tools"]
    assert "must call at least one available tool" in request.metadata["messages"][-1]["content"]
    assert request.prompt == render_fallback_prompt(request.metadata["messages"])


def test_build_chat_request_does_not_mutate_nested_caller_data():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Weather?"}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
    }
    original = copy.deepcopy(body)

    request = build_chat_request(body, SamplingParams(max_tokens=2))

    assert body == original
    request.metadata["messages"][0]["content"] = "changed"
    request.metadata["tools"][0]["function"]["name"] = "changed"
    assert body == original


def test_build_chat_request_uses_body_id_and_normalizes_sampling_when_omitted():
    request = build_chat_request({
        "messages": [{"role": "user", "content": "hi"}],
        "request_id": "body-id",
        "max_completion_tokens": 3,
        "temperature": 0.25,
    })

    assert request.request_id == "body-id"
    assert request.sampling_params.max_tokens == 3
    assert request.sampling_params.temperature == 0.25


def test_build_chat_request_generates_id_when_not_supplied():
    request = build_chat_request(
        {"messages": [{"role": "user", "content": "hi"}]},
        SamplingParams(max_tokens=1),
    )

    assert request.request_id.startswith("req-")


def test_build_completion_request_preserves_raw_prompt():
    from pocketllm.protocol import build_completion_request

    request = build_completion_request({"prompt": "raw", "max_tokens": 2}, request_id="completion-id")

    assert request.prompt == "raw"
    assert request.request_id == "completion-id"
    assert request.metadata == {"thinking_mode": "chat", "stream_options": {}, "response_format": None}
    assert request.sampling_params.max_tokens == 2


def test_build_completion_request_does_not_mutate_body():
    from pocketllm.protocol import build_completion_request

    body = {"prompt": ["a", "b"], "response_format": {"type": "text"}}
    original = copy.deepcopy(body)
    build_completion_request(body)
    assert body == original


def test_sampling_params_are_copied_for_request_ownership():
    params = SamplingParams(max_tokens=2, extra={"nested": {"value": 1}})
    request = build_chat_request({"messages": [{"role": "user", "content": "hi"}]}, params)

    assert request.sampling_params is not params
    request.sampling_params.extra["nested"]["value"] = 2
    assert params.extra["nested"]["value"] == 1


@pytest.mark.parametrize(
    "messages",
    [[], ["not an object"], "not a message list"],
)
def test_build_chat_request_rejects_invalid_messages(messages):
    with pytest.raises(ValueError):
        build_chat_request(
            {"messages": messages},
            SamplingParams(max_tokens=1),
        )
