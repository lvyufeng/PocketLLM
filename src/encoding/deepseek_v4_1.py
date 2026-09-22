"""DeepSeek-V4.1's prompt format, taken from the checkpoint that ships it.

V4.1 is not V4 with a renamed tokenizer. Three changes separate the two formats and every one of them
is visible in the rendered prompt: tool calls are wrapped in ``<｜DSML｜ calls>`` with a leading space
in the tag name, where V4 wrote ``<｜DSML｜tool_calls>``; reasoning effort became a numeric budget
rendered as ``Reasoning Effort: {budget} (range 1-100, ...)`` instead of a sentence; and a
mid-conversation message can be a system message via ``<｜System｜>``. A prompt built by
:mod:`src.encoding.deepseek_v4` is therefore *plausible* for this model and wrong, which is the
failure mode worth spending a module to avoid: it does not raise, it just answers a differently
formatted question.

Why this is a loader rather than a translation. The released checkpoint carries the format as a
Python module at ``<checkpoint>/encoding/encoding.py`` -- 979 self-contained lines that import
nothing from the inference tree -- and carries **no** ``chat_template`` in its tokenizer config, so
Hugging Face's ``apply_chat_template`` has nothing to apply. Serving this checkpoint means running
the code the checkpoint shipped, the same way serving a Qwen checkpoint means running the jinja
template *it* shipped. The file is loaded by path and imported under a private name so that two
checkpoints with different formats can be resident at once, and the module is the encoder's own: a
change to the format arrives with the next checkpoint rather than with the next release of this
package.

What is *not* taken from the checkpoint is the split of a completion into reasoning and content.
The released ``parse_message_from_completion_text`` asserts its way through the answer -- it requires
a closing ``</think>`` and a trailing end-of-sentence token, and asserts that the text is fully
consumed -- which is the right contract for grading a well-formed generation and the wrong one for a
server, where ``max_tokens`` truncation is ordinary. :func:`split_completion` implements the same
reading tolerantly and :func:`parse_message` prefers the checkpoint's parser, falling back to the
tolerant split when the output is not well formed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any


#: The directory and file the released checkpoint keeps its encoder in.
ENCODER_DIRNAME = "encoding"
ENCODER_FILENAME = "encoding.py"

#: The marker that closes the thinking block. Plain text in this format, not a special token, which
#: is why it survives `skip_special_tokens` and can be found in the user-visible text.
THINKING_END = "</think>"

#: The marker that opens a tool-call block. Also plain text, for the same reason. The checkpoint
#: composes it itself, as ``f"\n\n<{dsml_token}{tool_calls_block_name}"``, and its parser matches on
#: exactly this -- two newlines, then the tag, without the closing ``>`` -- so a reader that stops
#: here stops where the parser starts reading. Written out rather than read off the loaded encoder
#: because the readers below run on a text that may be truncated at any character, and a comparison
#: that cannot fail is worth more there than a name that cannot be wrong.
TOOL_CALLS_START = "\n\n<｜DSML｜ calls"

#: OpenAI's effort names mapped onto V4.1's 1-100 budget. Three of these are the checkpoint's own
#: aliases and are passed through under their own names; `minimal` and `medium` are this package's
#: reading of the public scale, and both keep the property that matters -- a client that asks for
#: less reasoning gets a smaller number than one that asks for more, and a client that asks for none
#: of this still gets a valid prefix rather than the checkpoint's assertion.
_EFFORT_BUDGETS: dict[str, int] = {
    "minimal": 25,
    "medium": 62,
}

#: The public names the checkpoint renders itself. Kept as names rather than numbers so the alias
#: table stays the checkpoint's to change.
_EFFORT_ALIASES = frozenset({"low", "high", "max"})

_LOCK = threading.Lock()
_MODULES: dict[str, ModuleType] = {}


class EncoderUnavailableError(RuntimeError):
    """The checkpoint does not carry the encoder this format needs."""


def encoder_path(checkpoint_dir: str) -> str | None:
    """The checkpoint's encoder, or ``None`` when it does not ship one."""
    path = os.path.join(str(checkpoint_dir), ENCODER_DIRNAME, ENCODER_FILENAME)
    return path if os.path.isfile(path) else None


def load_encoder(checkpoint_dir: str) -> ModuleType:
    """Import the checkpoint's encoder module, once per checkpoint directory.

    Loaded by path under a name derived from the directory rather than added to ``sys.path``: a
    checkpoint directory is data, and putting it on the import path would let any module inside it
    shadow a name the rest of the process imports.
    """
    path = encoder_path(checkpoint_dir)
    if path is None:
        raise EncoderUnavailableError(
            f"{checkpoint_dir} carries no {ENCODER_DIRNAME}/{ENCODER_FILENAME}; the DeepSeek-V4.1 "
            "prompt format is shipped by the checkpoint, and this backend will not guess at it"
        )
    key = os.path.realpath(path)
    with _LOCK:
        module = _MODULES.get(key)
        if module is not None:
            return module
        name = f"_deepseek_v4_1_encoding_{abs(hash(key)):x}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - a readable file always has one
            raise EncoderUnavailableError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # Registered before execution so a module that imports itself back by name finds the
        # partially built object rather than starting a second copy.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _MODULES[key] = module
        return module


def reasoning_effort(value: Any) -> int | str | None:
    """Translate a public effort name into something V4.1's renderer accepts.

    ``None`` stays ``None`` so the checkpoint applies its own default rather than this module
    restating it.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # a bool is an int, and never a budget
        raise ValueError("reasoning_effort must be an integer budget or one of low/high/max")
    if isinstance(value, int):
        if not 1 <= value <= 100:
            raise ValueError(f"reasoning_effort budget must be within 1-100, got {value}")
        return value
    name = str(value).lower()
    if name in _EFFORT_ALIASES:
        return name
    if name in _EFFORT_BUDGETS:
        return _EFFORT_BUDGETS[name]
    raise ValueError(
        f"reasoning_effort {value!r} is not one of the checkpoint's names "
        f"{sorted(_EFFORT_ALIASES)} or a budget in 1-100"
    )


def encode_messages(
    checkpoint_dir: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str = "chat",
    reasoning_effort_value: Any = None,
    tools: Any = None,
    context: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Render normalized chat messages into the prompt text V4.1 expects.

    ``tools`` is not passed to the encoder separately: the OpenAI tool schemas belong to the first
    system message, and the control plane has already attached them there (see
    :func:`pocketllm.protocol.chat.prepare_messages`). They are accepted and ignored here rather than
    silently dropped upstream, so a caller reading this signature does not go looking for a
    parameter that does not exist.
    """
    del tools
    encoder = load_encoder(checkpoint_dir)
    render = getattr(encoder, "encode_messages", None)
    if not callable(render):  # pragma: no cover - a checkpoint that ships the file ships the function
        raise EncoderUnavailableError(
            f"{checkpoint_dir}'s encoder has no encode_messages(); it is not a V4.1 encoder"
        )
    rendered = render(
        [dict(message) for message in messages],
        thinking_mode=str(thinking_mode),
        context=[dict(message) for message in (context or [])] or None,
        reasoning_effort=reasoning_effort(reasoning_effort_value),
    )
    if not isinstance(rendered, str):  # pragma: no cover - V4.1 returns text unless asked for media
        raise EncoderUnavailableError("the checkpoint encoder returned a non-text prompt")
    return rendered


def split_completion(text: str, *, thinking_mode: str = "chat") -> dict[str, Any]:
    """Split a completion into reasoning and answer, tolerating a truncated one.

    The checkpoint's own parser is the reference for this split and it is strict in three ways a
    server cannot be: it requires ``</think>`` when thinking, it requires the end-of-sentence token,
    and it asserts that nothing is left over. A generation cut by ``max_tokens`` violates all three
    and is a normal answer, so this reads the same format and keeps whatever arrived: everything
    before the marker is reasoning, everything after it is content, and a thinking-mode answer that
    never closed its block is all reasoning with no content -- which is what it is.
    """
    body = str(text or "")
    if str(thinking_mode) != "thinking":
        return {"reasoning_content": "", "content": body, "tool_calls": []}
    index = body.find(THINKING_END)
    if index < 0:
        return {"reasoning_content": body, "content": "", "tool_calls": []}
    return {
        "reasoning_content": body[:index],
        "content": body[index + len(THINKING_END):],
        "tool_calls": [],
    }


def cut_tool_calls(text: str) -> str:
    r"""``text`` up to the tool-call block that opens in it, and none of the block itself.

    A tool call is not part of the answer. The checkpoint's parser reads the block into OpenAI's
    ``tool_calls`` field and stops its reading of the content where the block opens, so the same
    characters cannot also be the answer -- and a client shown them is shown the format instead of
    the reply. This is that cut, made on a text that is still growing: a stream has to decide before
    it knows whether the generation will turn out to be well formed at all, and the parser, which
    insists that it is, cannot be asked.

    The tail is held back while it could still *become* the tag, which is what a running decode
    costs. ``\n\n<`` at the end of the text is three characters the answer may not contain and a
    stream cannot take back, so a trailing fragment of the tag is withheld with it; the next token
    either completes the tag -- and the cut was the right one -- or breaks it, and the held-back
    characters go out then. Both branches leave the cut where it was or move it forward, which is
    what lets a caller diff this against what it has already sent.
    """
    index = text.find(TOOL_CALLS_START)
    if index >= 0:
        return text[:index]
    # Longest first: the fragment that reaches furthest back into the tag is the one to hold.
    for size in range(min(len(TOOL_CALLS_START) - 1, len(text)), 0, -1):
        if TOOL_CALLS_START.startswith(text[-size:]):
            return text[: len(text) - size]
    return text


def parse_strict(checkpoint_dir: str, text: str, *, thinking_mode: str = "chat") -> dict[str, Any] | None:
    """The checkpoint's own parse of a completion, or ``None`` when it will not accept it.

    Returned separately from :func:`parse_message` because a caller that holds *two* readings of the
    same generation -- the raw one, which is what this parser needs, and the cleaned one a client
    sees -- has to choose which to fall back on itself.  ``None`` is not an error: it is the parser
    saying the text is not a well-formed message.
    """
    try:
        parser = getattr(load_encoder(checkpoint_dir), "parse_message_from_completion_text", None)
        if not callable(parser):
            return None
        parsed = parser(str(text), str(thinking_mode))
    except Exception:
        # The parser's contract is a well-formed message and it enforces that with assertions, so a
        # rejection arrives as whatever the assert happened to raise.  Every one of them means the
        # same thing here.
        return None
    if not isinstance(parsed, Mapping) or "content" not in parsed:
        return None
    return {
        "reasoning_content": str(parsed.get("reasoning_content") or ""),
        "content": str(parsed.get("content") or ""),
        "tool_calls": list(parsed.get("tool_calls") or ()),
    }


def parse_message(checkpoint_dir: str, text: str, *, thinking_mode: str = "chat") -> dict[str, Any]:
    """The structured assistant message for a completion, as the wire wants it.

    The checkpoint's parser wins when it accepts the text, because it is also the only thing that
    reads tool calls back into OpenAI format. Its contract is that the text is well formed, so a
    rejection is not an error to report -- it is the signal to fall back to :func:`split_completion`,
    which cannot fail and does not invent tool calls.
    """
    parsed = parse_strict(checkpoint_dir, text, thinking_mode=thinking_mode)
    return parsed if parsed is not None else split_completion(text, thinking_mode=thinking_mode)


__all__ = [
    "ENCODER_DIRNAME",
    "ENCODER_FILENAME",
    "EncoderUnavailableError",
    "THINKING_END",
    "TOOL_CALLS_START",
    "cut_tool_calls",
    "encode_messages",
    "encoder_path",
    "load_encoder",
    "parse_message",
    "parse_strict",
    "reasoning_effort",
    "split_completion",
]
