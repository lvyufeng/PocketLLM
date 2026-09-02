"""Backend-neutral request/response protocol helpers.

This subpackage holds pure normalization logic shared by the legacy DeepSeek
server and the unified PocketLLM server.  It imports no Torch, CUDA, or native
module, so both control planes can depend on it.
"""

from .chat import (
    ChatRequest,
    apply_stop_to_text,
    normalize_content,
    normalize_tool_calls,
    prepare_messages,
    render_fallback_prompt,
    stop_strings,
    thinking_config,
    tool_choice_instruction,
    tool_names,
)
from .prompt import encode_chat_prompt
from .requests import build_chat_request, build_completion_request

__all__ = [
    "ChatRequest",
    "build_chat_request",
    "build_completion_request",
    "apply_stop_to_text",
    "encode_chat_prompt",
    "normalize_content",
    "normalize_tool_calls",
    "prepare_messages",
    "render_fallback_prompt",
    "stop_strings",
    "thinking_config",
    "tool_choice_instruction",
    "tool_names",
]
