"""Which chat template an architecture uses, and how its answer reads back.

A checkpoint declares a ``model_type`` and that selects a templater.  Every architecture this build
serves goes through the checkpoint's own Hugging Face chat template, which is what lets a runtime
serve a model this repository has no bespoke encoder for; the bespoke encoders that used to live
beside the Qwen and DeepSeek runtimes left with them.

The classes carry both directions.  A host may encode through
:func:`pocketllm.protocol.encode_chat_prompt`, or through the templater itself when it wants the
rendered text alongside the ids, and reads the answer back -- by ``parse`` when a generation is
finished, and by ``split_reasoning`` while one is still arriving.  That reading was once the only
place in the tree a generated text was split into ``content`` / ``reasoning_content`` /
``tool_calls``, so it lives in the shared protocol plane rather than in any one front end.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


def detect_architecture(ckpt: str) -> str:
    """The checkpoint's architecture, normalized the way the C++ registry does.

    Same rule as ``pocket::detect_architecture`` in ``core/model_registry.cpp``: a safetensors
    checkpoint declares ``model_type`` in ``config.json``, a GGUF declares
    ``general.architecture`` in its own metadata, and ``qwen3_5_text`` -- where the multimodal
    wrapper hides the text model's type -- and ``qwen35`` -- the GGUF export's spelling -- both
    fold onto ``qwen3_5``.  Returns ``""`` when nothing is declared, which selects the generic
    templater.

    One difference from the C++ rule, in the Python host's direction: a directory *containing* a
    GGUF is accepted as well as the file itself, because that is how this side resolves a model
    path (`gguf_checkpoint_file`), and the two answers disagreeing would mean the prompt and the
    weights came from different readings of the same argument.
    """
    text = str(ckpt or "")
    if not text:
        return ""
    if text.endswith(".gguf"):
        return _gguf_architecture(text)
    try:
        with open(os.path.join(text, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        config = None
    if isinstance(config, dict):
        declared = _canonical(config.get("model_type"), config.get("text_config"))
        if declared:
            return declared
    return _gguf_architecture(_contained_gguf(text))


def _contained_gguf(directory: str) -> str:
    """The single ``.gguf`` directly inside ``directory``, or ``""``.

    Only a sole candidate is accepted: with two artifacts in one directory there is no way to say
    which the caller meant, and guessing one would be silent.
    """
    try:
        names = sorted(
            name for name in os.listdir(directory) if name.endswith(".gguf")
        )
    except OSError:
        return ""
    if len(names) != 1:
        return ""
    return os.path.join(directory, names[0])


def _gguf_architecture(path: str) -> str:
    if not path:
        return ""
    try:
        from src.loader.gguf.bundle import read_gguf_bundle

        metadata = read_gguf_bundle(path).metadata
    except Exception:
        return ""
    return _canonical(metadata.get("general.architecture"), None)


def _canonical(model_type: Any, text_config: Any) -> str:
    if not model_type and isinstance(text_config, dict):
        model_type = text_config.get("model_type")
    model_type = str(model_type or "").lower()
    # Three spellings of one runtime, and ``qwen35`` is not a typo of the other two: it is what
    # the GGUF export declares, the way ``qwen3_5_text`` is what the multimodal wrapper declares.
    # ``canonical_qwen_architecture`` folds the same three (``core/qwen_config.cpp``), and the two
    # have to agree: this answer selects the tool-call parser, so a checkpoint reaching the Python
    # host under the GGUF's spelling would otherwise read a tool call differently from the same
    # checkpoint reached through the native registry.
    if model_type in ("qwen3_5_text", "qwen35"):
        return "qwen3_5"
    return model_type


def splice_tools(messages: list[dict[str, Any]], tools: Any) -> list[dict[str, Any]]:
    """Attach ``tools`` to the system/developer message, adding one if absent."""
    messages = list(messages)
    attach_idx = None
    for idx, msg in enumerate(messages):
        if msg.get("role") in {"system", "developer"}:
            attach_idx = idx
            break
    if attach_idx is None:
        messages.insert(0, {"role": "system", "content": ""})
        attach_idx = 0
    messages[attach_idx] = {**messages[attach_idx], "tools": tools}
    return messages


def token_id_list(encoded: Any) -> list[int]:
    """Normalize ``apply_chat_template(..., tokenize=True)`` to a list of ids.

    The return type is not stable across tokenizers and transformers versions: a plain list, or a
    ``BatchEncoding`` whose ``input_ids`` holds the ids.  A ``BatchEncoding`` iterates as its *keys*,
    so the naive ``[int(t) for t in encoded]`` raises ``invalid literal for int() with base 10:
    'input_ids'`` -- a failure that surfaced as an HTTP 400 on every chat request while the
    completions path, which tokenizes in C++, kept working.  Both shapes are accepted rather than
    pinning a transformers version.
    """
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is not None:
        encoded = input_ids
    return [int(token) for token in encoded]


def split_reasoning(text: str, thinking_mode: str) -> tuple[str, str]:
    """Split a generated text into ``(reasoning, content)`` on the ``</think>`` marker.

    A prompt built with ``enable_thinking`` ends *inside* the block, so the opening tag is normally
    part of the prompt and only the closing one is in the answer.  A generation that stopped before
    closing the block is all reasoning and no answer.

    Text-based rather than token-id-based, which is what makes it the *same* split on a stream and
    off one: a stream recomputes it on the running decode and diffs against what it already sent, so
    the two routes cannot disagree about where the answer starts.  The cost is the one case the
    marker straddles a token boundary -- a client is then sent its first characters as reasoning.
    """
    marker = "</think>"
    if marker in text:
        head, _, tail = text.partition(marker)
        return head.removeprefix("<think>"), tail
    if thinking_mode == "thinking":
        return text.removeprefix("<think>"), ""
    return "", text


class Templater(Protocol):
    """One architecture's chat prompt and answer reading."""

    def encode(self, req: dict[str, Any]) -> tuple[str, list[int]]: ...

    def split_reasoning(self, text: str, thinking_mode: str) -> tuple[str, str]: ...

    def parse(self, text: str, thinking_mode: str, tools: Any = None) -> dict[str, Any]: ...


def _tool_call_parsers() -> dict[str, Any]:
    """Which tool-call syntax each architecture emits, by registry name.

    An architecture absent from this table returns no tool calls and leaves the call syntax in the
    content, which is what the generic templater did for every model before this table existed:
    inventing a parse for a model whose syntax has not been read would drop or corrupt calls
    silently.

    Empty, and deliberately so: the only entry this table ever held was Qwen's, read by
    ``src.encoding.qwen_tool_calls``, and that encoder left with the Qwen runtimes.  The single
    architecture this build serves declares no tool-call syntax, so its calls stay in the content
    rather than being parsed by an encoder that is no longer here.
    """
    return {}


class ChatTemplateTemplater:
    """The checkpoint's own HF chat template, for models with no bespoke encoder.

    Deliberately narrower than the DeepSeek path.  ``reasoning_effort``, ``context`` and
    ``drop_thinking`` have no equivalent in a stock chat template and are ignored.  Tool calls are
    parsed only for an architecture whose emitted syntax is known (see ``_tool_call_parsers``);
    anywhere else they are left in the content, which keeps them visible instead of inventing a
    parse that would silently drop them.
    """

    def __init__(self, tokenizer, tool_call_parser=None) -> None:
        self._tokenizer = tokenizer
        self._tool_call_parser = tool_call_parser

    def _apply(self, messages, tools, add_generation_prompt, thinking_mode, tokenize):
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        if tools:
            kwargs["tools"] = tools
        # Qwen and several other templates gate the reasoning block on `enable_thinking`; templates
        # that do not take it raise TypeError, so fall back rather than refusing the request.
        try:
            return self._tokenizer.apply_chat_template(
                messages, enable_thinking=(thinking_mode == "thinking"), **kwargs
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, req: dict[str, Any]) -> tuple[str, list[int]]:
        from .prompt import template_messages

        messages = list(req.get("messages") or [])
        tools = req.get("tools")
        tools = tools if isinstance(tools, list) and tools else None
        add_generation_prompt = bool(req.get("add_generation_prompt", True))
        thinking_mode = req.get("thinking_mode", "chat")
        # A replayed assistant message carries its `tool_calls[].function.arguments` as the JSON
        # string OpenAI specifies, while Qwen's template iterates them as an object
        # (`arguments|items`).  The template gets the converted copy; the caller's list is left as
        # it was received.
        messages = template_messages(messages)
        # Tokenize through the template rather than re-encoding the rendered text: a template that
        # emits a BOS would otherwise get a second one from encode().
        token_ids = self._apply(messages, tools, add_generation_prompt, thinking_mode, True)
        prompt_text = self._apply(messages, tools, add_generation_prompt, thinking_mode, False)
        return str(prompt_text), token_id_list(token_ids)

    def split_reasoning(self, text: str, thinking_mode: str) -> tuple[str, str]:
        return split_reasoning(text, thinking_mode)

    def parse(self, text: str, thinking_mode: str, tools: Any = None) -> dict[str, Any]:
        reasoning, content = self.split_reasoning(text, thinking_mode)
        tool_calls: list[dict[str, Any]] = []
        if self._tool_call_parser is not None:
            # None means the completion carries no call this parser can read in full, so the text
            # stands as it was written.  That is the same answer as "this model's calls are not
            # parsed", and it is the one that cannot show a client a truncated call.
            parsed = self._tool_call_parser(content, tools)
            if parsed is not None:
                content, tool_calls = parsed
        return {"content": content, "reasoning_content": reasoning, "tool_calls": tool_calls}


def build_templater(architecture: str, tokenizer) -> ChatTemplateTemplater:
    """The templater an architecture name selects."""
    return ChatTemplateTemplater(tokenizer, _tool_call_parsers().get(architecture))
