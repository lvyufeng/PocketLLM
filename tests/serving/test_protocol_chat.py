from __future__ import annotations

import pytest

from pocketllm.protocol import (
    ChatRequest,
    apply_stop_to_text,
    normalize_content,
    normalize_tool_calls,
    prepare_messages,
    render_fallback_prompt,
    thinking_config,
)


def test_normalize_content_flattens_text_blocks_and_marks_others():
    assert normalize_content("plain") == "plain"
    assert normalize_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert normalize_content([{"type": "image_url"}]) == "[Unsupported image_url]"
    assert normalize_content(None) == ""


def test_prepare_messages_attaches_tools_and_required_instruction():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        "tool_choice": "required",
    }

    messages = prepare_messages(body)

    # A system turn is inserted to carry tools, and the instruction lands on the
    # last instruction-bearing message.
    assert messages[0]["role"] == "system"
    assert messages[0]["tools"] == body["tools"]
    assert messages[-1]["content"].startswith("hi")
    assert "must call at least one available tool" in messages[-1]["content"]


def test_prepare_messages_rejects_unknown_named_tool():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": {"type": "function", "function": {"name": "missing"}},
    }

    with pytest.raises(ValueError, match="unknown tool"):
        prepare_messages(body)


def test_prepare_messages_requires_a_non_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        prepare_messages({"messages": []})
    with pytest.raises(ValueError, match="each message must be an object"):
        prepare_messages({"messages": ["hi"]})


def test_thinking_config_reads_reasoning_effort_from_either_field():
    assert thinking_config({}) == ("chat", None)
    assert thinking_config({"reasoning_effort": "high"}) == ("thinking", "high")
    assert thinking_config({"reasoning": {"effort": "max"}}) == ("thinking", "max")
    # An unrecognized effort still selects thinking mode but drops the value.
    assert thinking_config({"reasoning_effort": "turbo"}) == ("thinking", None)


def test_normalize_tool_calls_serializes_arguments_and_assigns_ids():
    calls = normalize_tool_calls([{"function": {"name": "weather", "arguments": {"city": "sh"}}}])

    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"] == {"name": "weather", "arguments": '{"city": "sh"}'}
    assert normalize_tool_calls([{"function": {}}]) == []


def test_apply_stop_to_text_truncates_at_the_earliest_match():
    assert apply_stop_to_text("abcdef", ["cd", "ef"]) == ("ab", True)
    assert apply_stop_to_text("abcdef", "zz") == ("abcdef", False)
    assert apply_stop_to_text("abcdef", None) == ("abcdef", False)


def test_chat_request_carries_normalized_messages_as_metadata():
    request = ChatRequest.from_body({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "reasoning_effort": "low",
    })

    metadata = request.metadata()
    assert metadata["thinking_mode"] == "thinking"
    assert metadata["reasoning_effort"] == "low"
    assert metadata["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_request_validates_completion_prompts():
    assert ChatRequest.from_body({"prompt": ["a", "b"]}, completion=True).prompt == "ab"
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        ChatRequest.from_body({"prompt": ""}, completion=True)
    with pytest.raises(ValueError, match="prompt list must contain only strings"):
        ChatRequest.from_body({"prompt": [1]}, completion=True)
    with pytest.raises(ValueError, match="stream_options must be an object"):
        ChatRequest.from_body({"prompt": "hi", "stream_options": 1}, completion=True)


def test_render_fallback_prompt_is_deterministic():
    text = render_fallback_prompt([
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
    ])

    assert text == "system: s\nuser: u"
