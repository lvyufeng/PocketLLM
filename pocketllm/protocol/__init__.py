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

__all__ = [
    "ChatRequest",
    "apply_stop_to_text",
    "normalize_content",
    "normalize_tool_calls",
    "prepare_messages",
    "render_fallback_prompt",
    "stop_strings",
    "thinking_config",
    "tool_choice_instruction",
    "tool_names",
]
