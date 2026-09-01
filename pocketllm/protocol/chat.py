"""Chat/completion request normalization shared by both control planes.

The helpers here are pure functions over JSON-like values.  They hold the
semantics the legacy DeepSeek server validated over time: multimodal content
flattening, tool attachment and ``tool_choice`` instructions, reasoning mode
detection, tool-call normalization, and stop-string truncation.

``src.server.openai`` re-exports these so there is one implementation rather
than a simplified copy in the unified server.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

_REASONING_EFFORTS = {None, "minimal", "low", "medium", "high", "max"}
_TOOL_ATTACH_ROLES = {"system", "developer"}
_INSTRUCTION_ROLES = {"user", "developer", "system"}


def normalize_content(content: Any) -> str:
    """Flatten OpenAI content blocks into the plain text the runtimes accept."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(f"[Unsupported {block.get('type', 'content')}]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def tool_choice_instruction(tool_choice: Any, tools: Any) -> tuple[bool, str | None]:
    """Return whether to attach tools and the instruction implied by the choice."""
    if tool_choice is None or tool_choice == "auto":
        return True, None
    if tool_choice == "none":
        return False, "Do not call any tools. Answer directly."
    if tool_choice == "required":
        return True, "You must call at least one available tool if the user request can be answered with a tool."
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function" or not isinstance(tool_choice.get("function"), dict):
            raise ValueError("tool_choice object must be {type:'function', function:{name:'...'}}")
        name = tool_choice["function"].get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.function.name must be a non-empty string")
        names = tool_names(tools)
        if names and name not in names:
            raise ValueError(f"tool_choice references unknown tool: {name}")
        return True, f"You must call the tool named {name}. Do not call any other tool."
    raise ValueError("tool_choice must be 'auto', 'none', 'required', or a function choice object")


def prepare_messages(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize ``messages`` together with tools and response format."""
    raw_messages = body.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be an object")
        copied = dict(msg)
        role = copied.get("role")
        if "content" in copied and role != "tool":
            copied["content"] = normalize_content(copied["content"])
        elif role == "tool" and isinstance(copied.get("content"), list):
            copied["content"] = normalize_content(copied.get("content"))
        messages.append(copied)
    tools = body.get("tools")
    attach_tools, instruction = tool_choice_instruction(body.get("tool_choice"), tools)
    tool_attach_idx = None
    for idx, msg in enumerate(messages):
        if msg.get("role") in _TOOL_ATTACH_ROLES:
            tool_attach_idx = idx
            break
    if tools is not None and attach_tools and tool_attach_idx is None:
        messages.insert(0, {"role": "system", "content": ""})
        tool_attach_idx = 0
    last_user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in _INSTRUCTION_ROLES:
            last_user_idx = idx
            break
    if tools is not None and attach_tools and tool_attach_idx is not None:
        messages[tool_attach_idx]["tools"] = tools
    if last_user_idx is not None:
        if body.get("response_format") is not None:
            messages[last_user_idx]["response_format"] = body["response_format"]
        if instruction:
            content = messages[last_user_idx].get("content") or ""
            messages[last_user_idx]["content"] = f"{content}\n\n{instruction}" if content else instruction
    return messages


def thinking_config(body: Mapping[str, Any]) -> tuple[str, str | None]:
    """Resolve ``reasoning``/``reasoning_effort`` into a thinking mode and effort."""
    reasoning = body.get("reasoning")
    effort = body.get("reasoning_effort")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        effort = reasoning.get("effort")
    if reasoning is None and effort is None:
        return "chat", None
    if effort not in _REASONING_EFFORTS:
        effort = None
    return "thinking", effort


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Coerce parsed tool calls into the OpenAI wire shape."""
    normalized: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return normalized
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = function.get("name") or tool_call.get("name")
        arguments = function.get("arguments", tool_call.get("arguments", "{}"))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if not name:
            continue
        normalized.append({
            "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": str(name), "arguments": arguments},
        })
    return normalized


def stop_strings(stop: Any) -> list[str]:
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, (list, tuple)):
        return [item for item in stop if isinstance(item, str) and item]
    return []


def apply_stop_to_text(text: str, stop: Any) -> tuple[str, bool]:
    """Truncate ``text`` at the earliest stop string.

    Returns the possibly truncated text and whether a stop string matched, so
    callers can set ``finish_reason`` without re-scanning.
    """
    earliest: int | None = None
    for item in stop_strings(stop):
        position = text.find(item)
        if position >= 0 and (earliest is None or position < earliest):
            earliest = position
    if earliest is None:
        return text, False
    return text[:earliest], True


def render_fallback_prompt(messages: Any) -> str:
    """Render messages when no model-specific chat template is available.

    Backends whose tokenizer or runtime owns a real chat template should never
    reach this.  It exists so fake backends and non-DeepSeek checkpoints still
    receive a deterministic prompt instead of an error.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role", "user"))
        parts.append(f"{role}: {normalize_content(message.get('content', ''))}")
    return "\n".join(parts)


@dataclass(slots=True)
class ChatRequest:
    """Normalized chat/completion body fields with cross-backend meaning."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    prompt: str | None = None
    thinking_mode: str = "chat"
    reasoning_effort: str | None = None
    tools: Any = None
    tool_choice: Any = None
    response_format: Any = None
    stream_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: Mapping[str, Any], *, completion: bool = False) -> "ChatRequest":
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            raise ValueError("stream_options must be an object")
        if completion:
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                if not prompt or not all(isinstance(item, str) for item in prompt):
                    raise ValueError("prompt list must contain only strings")
                prompt = "".join(prompt)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string")
            return cls(
                prompt=prompt,
                response_format=body.get("response_format"),
                stream_options=dict(stream_options),
            )
        messages = prepare_messages(body)
        mode, effort = thinking_config(body)
        return cls(
            messages=messages,
            thinking_mode=mode,
            reasoning_effort=effort,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            response_format=body.get("response_format"),
            stream_options=dict(stream_options),
        )

    def metadata(self) -> dict[str, Any]:
        """Return backend-visible request metadata.

        Only values with a stable meaning across backends are included; no
        tensors, device handles, or runtime objects.
        """
        data: dict[str, Any] = {
            "thinking_mode": self.thinking_mode,
            "stream_options": dict(self.stream_options),
            "response_format": self.response_format,
        }
        if self.messages:
            data["messages"] = [dict(message) for message in self.messages]
        if self.reasoning_effort is not None:
            data["reasoning_effort"] = self.reasoning_effort
        if self.tools is not None:
            data["tools"] = self.tools
        if self.tool_choice is not None:
            data["tool_choice"] = self.tool_choice
        return data
