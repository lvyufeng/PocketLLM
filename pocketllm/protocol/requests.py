"""Backend-neutral request construction helpers.

These helpers bridge OpenAI-shaped inputs to the public generation types. They
are shared by the HTTP server and the library facade so request normalization,
metadata, and prompt fallbacks do not drift between entry points.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pocketllm.api import GenerationRequest, SamplingParams

from .chat import ChatRequest, render_fallback_prompt


_DEFAULT_REQUEST_ID_PREFIX = "req-"


def _request_id(value: Any) -> str:
    return str(value or f"{_DEFAULT_REQUEST_ID_PREFIX}{uuid.uuid4().hex}")


def _copy_sampling_params(params: SamplingParams) -> SamplingParams:
    """Keep request-owned mutable sampling fields independent of callers."""
    return copy.deepcopy(params)


def _copy_body(body: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError(f"{kind} body must be an object")
    return copy.deepcopy(dict(body))


def _resolve_sampling_params(
    body: Mapping[str, Any], sampling_params: SamplingParams | None,
) -> SamplingParams:
    """Copy caller params, or normalize OpenAI sampling fields from the body.

    ``response_format`` may arrive either on :class:`SamplingParams` or in the
    body, so both entry points produce the same request. A body value is the
    more specific per-call input and wins.
    """
    if sampling_params is None:
        return SamplingParams.from_openai(body)
    params = _copy_sampling_params(sampling_params)
    if body.get("response_format") is not None:
        params = replace(params, response_format=copy.deepcopy(body["response_format"]))
    return params


def build_chat_request(
    body: Mapping[str, Any],
    sampling_params: SamplingParams | None = None,
    *,
    request_id: str | None = None,
) -> GenerationRequest:
    """Build a generation request from an OpenAI chat-completion body.

    If sampling parameters are omitted, OpenAI sampling fields in ``body`` are
    normalized with :meth:`SamplingParams.from_openai`. The body is deep-copied
    before normalization so caller-owned messages and tools are not mutated.
    """
    copied = _copy_body(body, "chat")
    chat = ChatRequest.from_body(copied)
    metadata = copy.deepcopy(chat.metadata())
    return GenerationRequest(
        prompt=render_fallback_prompt(metadata["messages"]),
        sampling_params=_resolve_sampling_params(copied, sampling_params),
        request_id=_request_id(request_id or copied.get("request_id")),
        metadata=metadata,
    )


def build_completion_request(
    body: Mapping[str, Any],
    sampling_params: SamplingParams | None = None,
    *,
    request_id: str | None = None,
) -> GenerationRequest:
    """Build a text-completion request without chat normalization."""
    copied = _copy_body(body, "completion")
    chat = ChatRequest.from_body(copied, completion=True)
    return GenerationRequest(
        prompt=chat.prompt,
        sampling_params=_resolve_sampling_params(copied, sampling_params),
        request_id=_request_id(request_id or copied.get("request_id")),
        metadata=copy.deepcopy(chat.metadata()),
    )
