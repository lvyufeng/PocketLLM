"""Long-lived Python helper for the cpp_engine OpenAI server.

Reads JSON-line requests on stdin, writes JSON-line responses on stdout.

The templating itself lives in :mod:`pocketllm.protocol.templating`, because it is not this
transport's business and because the Python host needs the same answer: a checkpoint's architecture
selects a chat template, and that template's output has to be read back into content, reasoning and
tool calls the same way whichever front end asked. This module is the pipe between the C++ engine
and that implementation, and nothing else.

Protocol (one JSON object per stdin/stdout line):

  request {"op": "encode", "messages": [...], "thinking_mode": "chat"|"thinking",
           "reasoning_effort": "low"|"medium"|"high"|null,
           "add_generation_prompt": true,
           "context": [...] (optional), "drop_thinking": true}
  reply   {"ok": true, "prompt_text": "...", "token_ids": [...]}

  request {"op": "tokenize", "prompt": "..."}
  reply   {"ok": true, "token_ids": [...]}

  request {"op": "parse", "text": "...", "thinking_mode": "chat"|"thinking",
           "tools": [...] (optional, the same definitions the request carried)}
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

from pocketllm.protocol.templating import (  # noqa: E402  (needs _REPO_ROOT on sys.path)
    build_templater,
    detect_architecture,
)


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


def _handle_encode(templater, req: dict[str, Any]) -> None:
    prompt_text, token_ids = templater.encode(req)
    _emit({"ok": True, "prompt_text": prompt_text, "token_ids": token_ids})


def _handle_tokenize(tokenizer, req: dict[str, Any]) -> None:
    """Tokenize raw text without applying chat template."""
    prompt = req.get("prompt", "")
    if not isinstance(prompt, str):
        _err("prompt must be a string")
        return
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    _emit({"ok": True, "token_ids": token_ids})


def _handle_parse(templater, req: dict[str, Any]) -> None:
    parsed = templater.parse(
        req.get("text", ""), req.get("thinking_mode", "chat"), req.get("tools")
    )
    _emit({"ok": True, **parsed})


def load_tokenizer(ckpt: str, tokenizer_path: str | None = None):
    """The checkpoint's vocabulary, however that checkpoint carries it.

    A safetensors export ships ``tokenizer.json`` beside its config and
    ``AutoTokenizer`` assembles the pair. A GGUF ships the same facts -- the
    vocabulary, the merges, the special-token ids and the chat template -- in its
    own header, and the released ternary artifact is one such file with nothing
    beside it, so there is no directory for ``transformers`` to read and asking
    it for one fails before the engine is ever reached. Both branches return the
    same class, which is what lets the rest of this module stay unaware of which
    container it is serving. An explicit ``tokenizer_path`` still wins: a caller
    who names one wants that one.
    """

    if tokenizer_path:
        return AutoTokenizer.from_pretrained(tokenizer_path)
    # Imported here rather than at module scope: the sidecar is also run against
    # safetensors checkpoints, where neither the adapter package nor the GGUF
    # reader is needed, and a failure to import either should not be able to
    # stop the path that does not use them.
    from pocketllm.backends.cpp_backend import gguf_checkpoint_file

    gguf = gguf_checkpoint_file(ckpt)
    if gguf:
        from src.encoding.gguf_tokenizer import build_gguf_hf_tokenizer

        tokenizer, _metadata = build_gguf_hf_tokenizer(gguf)
        return tokenizer
    return AutoTokenizer.from_pretrained(ckpt)


def main() -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the model checkpoint the vocabulary is read from",
    )
    parser.add_argument("--tokenizer-path", default=None, help="Optional override for tokenizer directory")
    parser.add_argument(
        "--architecture",
        default=None,
        help="Architecture already detected by the native model registry",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.ckpt, args.tokenizer_path)
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
            elif op == "tokenize":
                _handle_tokenize(tokenizer, req)
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
