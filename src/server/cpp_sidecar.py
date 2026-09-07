"""Long-lived Python helper for the cpp_engine OpenAI server.

Reads JSON-line requests on stdin, writes JSON-line responses on stdout.

Chat templating is per-architecture, mirroring the C++ model registry: the
checkpoint declares a ``model_type`` and that selects a templater.  DeepSeek-V4
keeps ``src.encoding.deepseek_v4``, which renders DSML tool-call syntax,
thinking_mode and reasoning_effort.  Everything else goes through the
checkpoint's own HF chat template, which is what makes the C++ server able to
serve a model this repo has no bespoke encoder for.

Protocol (one JSON object per stdin/stdout line):

  request {"op": "encode", "messages": [...], "thinking_mode": "chat"|"thinking",
           "reasoning_effort": "low"|"medium"|"high"|null,
           "add_generation_prompt": true,
           "context": [...] (optional), "drop_thinking": true}
  reply   {"ok": true, "prompt_text": "...", "token_ids": [...]}

  request {"op": "parse", "text": "...", "thinking_mode": "chat"|"thinking"}
  reply   {"ok": true, "content": "...", "reasoning": "...", "tool_calls": [...]}

  request {"op": "ping"}
  reply   {"ok": true, "pong": true}

Errors: {"ok": false, "err": "..."}.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

from transformers import AutoTokenizer

# Make sure src/ is importable when invoked from arbitrary CWDs.
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _configure_stdio() -> None:
    """Use UTF-8 for the JSON-line protocol regardless of the parent locale."""
    # The C++ parent emits raw multi-byte JSON for non-ASCII content (Chinese
    # prompts, tool-call XML, etc.). Kept out of module import so the templaters
    # remain unit-testable under pytest's captured stdin/stdout wrappers.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8")


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    _emit({"ok": False, "err": msg})


def detect_architecture(ckpt: str) -> str:
    """The checkpoint's architecture, normalized the way the C++ registry does.

    Same rule as ``pocket::detect_architecture`` in core/model_registry.cpp,
    including folding ``qwen3_5_text`` (where the multimodal wrapper hides the
    text model's type) onto ``qwen3_5``.  Returns "" when nothing is declared,
    which selects the generic templater.
    """
    try:
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    model_type = config.get("model_type")
    if not model_type:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            model_type = text_config.get("model_type")
    model_type = str(model_type or "").lower()
    return "qwen3_5" if model_type == "qwen3_5_text" else model_type


def _splice_tools(messages: list[dict[str, Any]], tools: Any) -> list[dict[str, Any]]:
    """Attach `tools` to the system/developer message, adding one if absent."""
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


class DeepSeekV4Templater:
    """DSML chat template, thinking modes and tool-call parsing for DeepSeek-V4."""

    def __init__(self, tokenizer) -> None:
        from src.encoding.deepseek_v4 import (
            encode_messages,
            eos_token,
            parse_message_from_completion_text,
        )

        self._tokenizer = tokenizer
        self._encode_messages = encode_messages
        self._eos_token = eos_token
        self._parse = parse_message_from_completion_text

    def encode(self, req: dict[str, Any]) -> tuple[str, list[int]]:
        messages = list(req.get("messages") or [])
        tools = req.get("tools")
        if isinstance(tools, list) and tools:
            messages = _splice_tools(messages, tools)
        prompt_text = self._encode_messages(
            messages,
            thinking_mode=req.get("thinking_mode", "chat"),
            context=req.get("context"),
            drop_thinking=bool(req.get("drop_thinking", True)),
            add_default_bos_token=bool(req.get("add_generation_prompt", True)),
            reasoning_effort=req.get("reasoning_effort"),
        )
        return prompt_text, list(self._tokenizer.encode(prompt_text))

    def parse(self, text: str, thinking_mode: str) -> dict[str, Any]:
        # parse_message_from_completion_text requires the EOS token to be
        # present.  The C++ engine emits raw decoded text without re-inserting
        # the EOS string (it stops on the EOS token id), so append it if missing.
        if not text.endswith(self._eos_token):
            text = text + self._eos_token
        return self._parse(text, thinking_mode)


class ChatTemplateTemplater:
    """The checkpoint's own HF chat template, for models with no bespoke encoder.

    Deliberately narrower than the DeepSeek path.  ``reasoning_effort``,
    ``context`` and ``drop_thinking`` have no equivalent in a stock chat
    template and are ignored, and tool *calls* in the completion are left in the
    content rather than parsed: their syntax is per-model, and returning an empty
    tool_calls list keeps them visible instead of inventing a parse that would
    silently drop them.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def _apply(self, messages, tools, add_generation_prompt, thinking_mode, tokenize):
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        if tools:
            kwargs["tools"] = tools
        # Qwen and several other templates gate the reasoning block on
        # `enable_thinking`; templates that do not take it raise TypeError, so
        # fall back rather than refusing the request.
        try:
            return self._tokenizer.apply_chat_template(
                messages, enable_thinking=(thinking_mode == "thinking"), **kwargs
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, req: dict[str, Any]) -> tuple[str, list[int]]:
        messages = list(req.get("messages") or [])
        tools = req.get("tools")
        tools = tools if isinstance(tools, list) and tools else None
        add_generation_prompt = bool(req.get("add_generation_prompt", True))
        thinking_mode = req.get("thinking_mode", "chat")
        # Tokenize through the template rather than re-encoding the rendered
        # text: a template that emits a BOS would otherwise get a second one
        # from encode().
        token_ids = self._apply(messages, tools, add_generation_prompt, thinking_mode, True)
        prompt_text = self._apply(messages, tools, add_generation_prompt, thinking_mode, False)
        return str(prompt_text), [int(t) for t in token_ids]

    def parse(self, text: str, thinking_mode: str) -> dict[str, Any]:
        reasoning = ""
        content = text
        marker = "</think>"
        if marker in text:
            head, _, tail = text.partition(marker)
            # A prompt built with enable_thinking ends inside the block, so the
            # opening tag is usually part of the prompt rather than the answer.
            reasoning = head.removeprefix("<think>")
            content = tail
        elif thinking_mode == "thinking":
            # Generation stopped before closing the block; all of it is reasoning.
            reasoning = text.removeprefix("<think>")
            content = ""
        return {"content": content, "reasoning_content": reasoning, "tool_calls": []}


def build_templater(architecture: str, tokenizer):
    if architecture == "deepseek_v4":
        return DeepSeekV4Templater(tokenizer)
    return ChatTemplateTemplater(tokenizer)


def _handle_encode(templater, req: dict[str, Any]) -> None:
    prompt_text, token_ids = templater.encode(req)
    _emit({"ok": True, "prompt_text": prompt_text, "token_ids": token_ids})


def _handle_parse(templater, req: dict[str, Any]) -> None:
    parsed = templater.parse(req.get("text", ""), req.get("thinking_mode", "chat"))
    _emit({"ok": True, **parsed})


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (used to load the HF tokenizer)")
    parser.add_argument("--tokenizer-path", default=None, help="Optional override for tokenizer directory")
    parser.add_argument(
        "--architecture",
        default=None,
        help="Architecture already detected by the native model registry",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_path or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # The C++ server passes the answer from pocket::detect_architecture so the
    # two halves cannot select different models. Detection here is only the
    # standalone-script fallback used by tests and manual protocol probes.
    architecture = args.architecture or detect_architecture(args.ckpt)
    templater = build_templater(architecture, tokenizer)
    _emit({
        "ok": True,
        "ready": True,
        "architecture": architecture,
        "templater": type(templater).__name__,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 1,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as ex:
            _err(f"json decode failed: {ex}")
            continue
        op = req.get("op")
        try:
            if op == "encode":
                _handle_encode(templater, req)
            elif op == "parse":
                _handle_parse(templater, req)
            elif op == "ping":
                _emit({"ok": True, "pong": True})
            elif op == "shutdown":
                _emit({"ok": True, "bye": True})
                return 0
            else:
                _err(f"unknown op: {op!r}")
        except Exception as ex:  # noqa: BLE001
            _err(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
