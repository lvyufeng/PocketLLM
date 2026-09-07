import json

from src.server.cpp_sidecar import ChatTemplateTemplater, detect_architecture


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [7, 8, 9] if kwargs["tokenize"] else "rendered prompt"


class NoThinkingArgumentTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
    ):
        self.calls.append((messages, add_generation_prompt, tokenize))
        return [4, 5] if tokenize else "fallback prompt"


def test_cpp_sidecar_detects_root_and_nested_architectures(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "DeepSeek_V4"}), encoding="utf-8"
    )
    assert detect_architecture(str(root)) == "deepseek_v4"

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "config.json").write_text(
        json.dumps({"text_config": {"model_type": "Qwen3_5_Text"}}),
        encoding="utf-8",
    )
    assert detect_architecture(str(nested)) == "qwen3_5"

    undeclared = tmp_path / "undeclared"
    undeclared.mkdir()
    (undeclared / "config.json").write_text("{}", encoding="utf-8")
    assert detect_architecture(str(undeclared)) == ""


def test_generic_sidecar_uses_checkpoint_chat_template_for_ids_and_text():
    tokenizer = RecordingTokenizer()
    templater = ChatTemplateTemplater(tokenizer)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "weather"}}]

    prompt, token_ids = templater.encode(
        {
            "messages": messages,
            "tools": tools,
            "thinking_mode": "thinking",
            "add_generation_prompt": True,
        }
    )

    assert prompt == "rendered prompt"
    assert token_ids == [7, 8, 9]
    assert len(tokenizer.calls) == 2
    for seen_messages, kwargs in tokenizer.calls:
        assert seen_messages == messages
        assert kwargs["tools"] == tools
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is True
    assert tokenizer.calls[0][1]["tokenize"] is True
    assert tokenizer.calls[1][1]["tokenize"] is False


def test_generic_sidecar_falls_back_when_template_has_no_thinking_argument():
    tokenizer = NoThinkingArgumentTokenizer()
    prompt, token_ids = ChatTemplateTemplater(tokenizer).encode(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_mode": "chat",
        }
    )

    assert prompt == "fallback prompt"
    assert token_ids == [4, 5]
    # Each render first tries enable_thinking, then retries without it.
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[0][2] is True
    assert tokenizer.calls[1][2] is False


def test_generic_sidecar_splits_reasoning_into_protocol_field():
    templater = ChatTemplateTemplater(RecordingTokenizer())

    parsed = templater.parse("work it out</think>final answer", "thinking")
    assert parsed == {
        "content": "final answer",
        "reasoning_content": "work it out",
        "tool_calls": [],
    }

    unfinished = templater.parse("still reasoning", "thinking")
    assert unfinished == {
        "content": "",
        "reasoning_content": "still reasoning",
        "tool_calls": [],
    }

    chat = templater.parse(" plain answer ", "chat")
    assert chat == {
        "content": " plain answer ",
        "reasoning_content": "",
        "tool_calls": [],
    }
