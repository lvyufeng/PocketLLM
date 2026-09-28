import json

from pocketllm.protocol.templating import (
    ChatTemplateTemplater,
    DeepSeekV4Templater,
    build_templater,
    detect_architecture,
)


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


class BatchEncodingTokenizer:
    """Reproduces what transformers 5.x returns for ``tokenize=True``.

    ``apply_chat_template(..., tokenize=True)`` returns a ``BatchEncoding`` when
    the tokenizer is a fast one, and a ``BatchEncoding`` iterates as its *keys*.
    """

    class BatchEncoding:
        def __init__(self, input_ids) -> None:
            self.input_ids = list(input_ids)
            self.attention_mask = [1] * len(self.input_ids)

        def __iter__(self):
            return iter(("input_ids", "attention_mask"))

        def __getitem__(self, key):
            return getattr(self, key)

    def apply_chat_template(self, messages, **kwargs):
        if kwargs["tokenize"]:
            return self.BatchEncoding([11, 12, 13])
        return "rendered prompt"


def _write_gguf(path, architecture: str) -> None:
    """A GGUF header and nothing else: enough for ``general.architecture`` to be read."""
    import struct

    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    metadata = struct.pack("<Q", 1) + string("general.architecture") + struct.pack("<I", 8) + string(
        architecture
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + metadata
    path.write_bytes(header + struct.pack("<Q", 0))


def test_an_architecture_is_read_from_a_config_file_or_a_gguf(tmp_path):
    """The two container types a checkpoint can arrive in, and the empty answer.

    ``detect_architecture`` used to read ``config.json`` only, which is right for the sidecar
    (the C++ registry handed it the answer) and wrong for the Python host, where a GGUF is found
    by scanning the model path. A GGUF declares the same fact under ``general.architecture``.
    """
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

    assert detect_architecture("") == ""
    assert detect_architecture(str(tmp_path / "missing")) == ""


def test_a_gguf_declares_its_architecture_in_its_own_metadata(tmp_path):
    gguf = tmp_path / "model.gguf"
    _write_gguf(gguf, "qwen3_5")
    assert detect_architecture(str(gguf)) == "qwen3_5"
    # A directory holding exactly one GGUF is the same checkpoint, and is how the Python host
    # resolves a model path, so the two readings must not disagree.
    assert detect_architecture(str(tmp_path)) == "qwen3_5"


def test_two_ggufs_in_one_directory_name_no_architecture(tmp_path):
    """No way to say which artifact was meant, so neither is guessed at."""
    _write_gguf(tmp_path / "a.gguf", "qwen3_5")
    _write_gguf(tmp_path / "b.gguf", "deepseek_v4")

    assert detect_architecture(str(tmp_path)) == ""


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


def test_generic_sidecar_reads_ids_from_a_batch_encoding():
    """A fast tokenizer's chat template returns a BatchEncoding, not a list.

    Iterating that for ids yields its keys, so ``int(token)`` raised
    ``invalid literal for int() with base 10: 'input_ids'`` - an HTTP 400 on
    every chat request, while the completions path (tokenized in C++) kept
    working.  The real shape is faked here rather than reached through a
    checkpoint, so the regression is covered without a model to load.
    """
    templater = ChatTemplateTemplater(BatchEncodingTokenizer())

    prompt, token_ids = templater.encode({"messages": [{"role": "user", "content": "hi"}]})

    assert token_ids == [11, 12, 13]
    assert prompt == "rendered prompt"


def test_generic_sidecar_decodes_replayed_tool_arguments_for_the_template():
    """A replayed assistant message has to reach the template as an object.

    Qwen's chat template walks `tool_call.function.arguments` with `|items`,
    while OpenAI specifies the same field as a JSON string.  A second turn
    replays the assistant message the server just returned, so passing the
    request through verbatim raises "Can only get item pairs from a mapping"
    before the model is ever reached.
    """
    tokenizer = RecordingTokenizer()
    messages = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_" + "0" * 24,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris", "days": 3}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_" + "0" * 24, "content": "{}"},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    ChatTemplateTemplater(tokenizer).encode({"messages": messages, "tools": tools})

    assert len(tokenizer.calls) == 2
    for seen_messages, _ in tokenizer.calls:
        assert seen_messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris",
            "days": 3,
        }
        assert seen_messages[2] == messages[2]
    # The request is not the template's to rewrite.
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris", "days": 3}'


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


QWEN_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}

QWEN_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)


def test_qwen_templater_reports_tool_calls_and_clears_the_content():
    templater = build_templater("qwen3_5", RecordingTokenizer())

    parsed = templater.parse(QWEN_CALL, "chat", [QWEN_TOOL])

    assert parsed["content"] == ""
    assert parsed["reasoning_content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    # The schema is what makes "3" an integer rather than the string the XML
    # spelling alone would be.
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}


def test_a_call_after_the_think_block_is_read_from_the_content():
    templater = build_templater("qwen3_5", RecordingTokenizer())

    parsed = templater.parse(f"weighing it up</think>{QWEN_CALL}", "thinking", [QWEN_TOOL])

    assert parsed["reasoning_content"] == "weighing it up"
    assert parsed["content"] == ""
    assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"


def test_an_architecture_with_no_known_syntax_leaves_the_call_in_the_content():
    """No parser is registered, so the XML is what the client sees.  Inventing a
    parse for a syntax nobody has read would be worse than showing the text."""
    templater = build_templater("llama", RecordingTokenizer())

    parsed = templater.parse(QWEN_CALL, "chat", [QWEN_TOOL])

    assert parsed["tool_calls"] == []
    assert parsed["content"] == QWEN_CALL


def test_a_truncated_call_is_left_in_the_content_rather_than_half_reported():
    templater = build_templater("qwen3_5", RecordingTokenizer())
    truncated = QWEN_CALL.split("</parameter>")[0]

    parsed = templater.parse(truncated, "chat", [QWEN_TOOL])

    assert parsed["tool_calls"] == []
    assert parsed["content"] == truncated


def test_build_templater_selects_a_templater_per_architecture():
    # The registry reports the architecture name; only qwen3_5 has a call syntax
    # this repository has read, and only deepseek_v4 has its own templater class.
    assert isinstance(
        build_templater("qwen3_5", RecordingTokenizer()), ChatTemplateTemplater
    )
    assert isinstance(build_templater("llama", RecordingTokenizer()), ChatTemplateTemplater)
    assert isinstance(
        build_templater("deepseek_v4", RecordingTokenizer()), DeepSeekV4Templater
    )


def test_every_templater_can_split_a_running_answer():
    """The split a *stream* needs, which is not the same reading as ``parse``.

    ``parse`` reads a whole grammar and refuses a generation that has not finished arriving;
    DeepSeek-V4's in particular raises on a missing EOS. A stream does not have a finished
    generation to hand it -- it has a prefix, on every token -- so it needs the marker split alone.
    Both templater classes therefore expose one, and it is the same one.
    """
    text = "weighing it up</think>the answer"

    assert ChatTemplateTemplater(RecordingTokenizer()).split_reasoning(text, "thinking") == (
        "weighing it up",
        "the answer",
    )
    assert DeepSeekV4Templater(RecordingTokenizer()).split_reasoning(text, "thinking") == (
        "weighing it up",
        "the answer",
    )
