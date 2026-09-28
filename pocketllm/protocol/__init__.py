"""Backend-neutral request/response protocol helpers.

This subpackage holds normalization, prompt encoding and answer reading that no device backend
owns, so every host reaches one implementation: request bodies in, prompt ids and structured
assistant messages out.  It imports no Torch, CUDA, or native module.

It was shared by two control planes -- the legacy DeepSeek server and the unified one -- until the
former was retired to a single runtime module; see
[#447](https://github.com/lvyufeng/PocketLLM/issues/447).
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
from .prompt import encode_chat_prompt, template_messages
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
    "template_messages",
    "thinking_config",
    "tool_choice_instruction",
    "tool_names",
]
