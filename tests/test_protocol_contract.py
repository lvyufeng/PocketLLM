"""The request-field contract: what this server refuses, and what it says.

Ported from ``cpp_engine/tests/test_openai_request_fields.cpp``, which checked the same table on
every C++ build and goes away with the native front end. The cases are the same because the policy
is the same -- a field whose value would have changed the answer is refused by name rather than
answered as if it had not been sent -- and the split into shape and capability cases is the one the
port introduced: shape holds on every runtime, capability is one adapter's answer.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from pocketllm.protocol.contract import (
    CHAT,
    COMPLETIONS,
    MAX_CHOICES,
    MAX_LOGPROB_ALTERNATIVES,
    ServedFields,
    audit,
    audit_shape,
    is_stop_shape,
)


#: A runtime that applies every field, for the cases that are about shape alone. Every other case
#: below names the fields its runtime serves, so a refusal is attributed to the half that caused it.
EVERYTHING = ServedFields(**{field.name: True for field in fields(ServedFields)})


def shape(body: dict, endpoint: str = CHAT):
    return audit_shape(body, endpoint=endpoint)


def capability(body: dict, *, endpoint: str = CHAT, **serves):
    return audit(body, endpoint=endpoint, serves=ServedFields(**serves))


def refusal(result) -> tuple[str, str]:
    """A refusal as ``(field, message)``, asserting it is one."""
    assert result is not None, "expected a refusal"
    return result.field, result.message


# ---------------------------------------------------------------------------------------------
# Shapes: the same on every runtime
# ---------------------------------------------------------------------------------------------


def test_a_body_of_defaults_is_accepted() -> None:
    assert shape({
        "model": "local", "messages": [], "temperature": 0.0, "top_p": 1.0, "top_k": 20,
        "seed": 7, "max_tokens": 64, "stream": False, "n": 1, "stop": [],
        "frequency_penalty": 0, "presence_penalty": 0, "logprobs": False,
        "top_logprobs": 0, "logit_bias": {}, "user": "u", "store": False,
        "metadata": {}, "service_tier": "auto", "tool_choice": "auto",
        "parallel_tool_calls": True,
    }) is None
    assert shape({
        "prompt": "hi", "max_tokens": 64, "n": 1, "best_of": 1, "stop": None,
        "logprobs": None, "echo": False, "suffix": "", "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }, COMPLETIONS) is None
    # Explicit nulls are how several SDKs spell "not set".
    assert shape({"n": None, "stop": None, "logprobs": None, "top_logprobs": None,
                  "logit_bias": None, "tool_choice": None, "parallel_tool_calls": None}) is None
    assert shape({"best_of": None, "echo": None, "suffix": None}, COMPLETIONS) is None


def test_choices_are_a_whole_number_in_range() -> None:
    for value in (1, 2, 3, MAX_CHOICES, 2.0):
        assert shape({"n": value}) is None, value
    for value in (0, -1, 1.5, "3", True, MAX_CHOICES + 1):
        field, _ = refusal(shape({"n": value}))
        assert field == "n", value
    # The refusal says how many were asked for and names the ceiling it applies.
    result = shape({"n": 999})
    assert result is not None
    assert result.requested == "999"
    assert str(MAX_CHOICES) in result.message


def test_a_stop_sequence_is_a_string_or_a_list_of_strings() -> None:
    for value in ("\n\n", ["USER:"], ["a", "b"], "", [], ["", ""]):
        assert shape({"stop": value}) is None, value
    for value in (5, True, {"a": 1}, ["a", 5], [["a"]]):
        field, _ = refusal(shape({"stop": value}))
        assert field == "stop", value
    assert is_stop_shape("x") and is_stop_shape(["x"]) and not is_stop_shape(5)
    # The refusal shows the value, so a caller sending a number sees which one.
    result = shape({"stop": 5})
    assert result is not None and result.requested == "5"


def test_log_probabilities_keep_each_endpoints_spelling() -> None:
    for value in (False, True):
        assert shape({"logprobs": value}) is None, value
    assert shape({"logprobs": True, "top_logprobs": 5}) is None
    assert shape({"logprobs": True, "top_logprobs": 0}) is None
    # A count is the other endpoint's spelling of the field, so it is not a boolean here.
    for value in (1, "true", []):
        field, _ = refusal(shape({"logprobs": value}))
        assert field == "logprobs", value

    for value in (0, 5, MAX_LOGPROB_ALTERNATIVES, None):
        assert shape({"logprobs": value}, COMPLETIONS) is None, value
    for value in (-1, 2.5, True, MAX_LOGPROB_ALTERNATIVES + 1, "5"):
        field, _ = refusal(shape({"logprobs": value}, COMPLETIONS))
        assert field == "logprobs", value

    # `top_logprobs` is a chat field, and on its own it is inert only at 0.
    field, _ = refusal(shape({"top_logprobs": 5}, COMPLETIONS))
    assert field == "top_logprobs"
    assert shape({"top_logprobs": None}, COMPLETIONS) is None
    field, _ = refusal(shape({"top_logprobs": 5}))
    assert field == "top_logprobs"
    for value in (-1, "5"):
        field, _ = refusal(shape({"top_logprobs": value}))
        assert field == "top_logprobs", value
    field, _ = refusal(shape({"logprobs": False, "top_logprobs": 5}))
    assert field == "top_logprobs"
    assert shape({"top_logprobs": 0}) is None
    assert shape({"logprobs": False, "top_logprobs": 0}) is None

    # The ceiling is named on both endpoints' spellings of it.
    for body, endpoint in (
        ({"logprobs": True, "top_logprobs": 64}, CHAT),
        ({"logprobs": 64}, COMPLETIONS),
    ):
        result = shape(body, endpoint)
        assert result is not None
        assert str(MAX_LOGPROB_ALTERNATIVES) in result.message


def test_stream_options_are_refused_only_where_they_lose_something() -> None:
    field, _ = refusal(shape({"stream": True, "stream_options": {"include_usage": True}}))
    assert field == "stream_options.include_usage"
    assert shape({"stream": True, "stream_options": {"include_usage": False}}) is None
    assert shape({"stream": True, "stream_options": {}}) is None
    # Without streaming there is no chunk stream to add usage to, and a non-streaming response
    # already carries "usage" -- which is the whole thing the option asks for.
    assert shape({"stream_options": {"include_usage": True}}) is None
    field, _ = refusal(
        shape({"stream": True, "stream_options": {"include_usage": True}}, COMPLETIONS)
    )
    assert field == "stream_options.include_usage"


def test_completions_only_fields_are_not_audited_on_the_chat_endpoint() -> None:
    """Which endpoint a field belongs to decides whether it is looked at at all.

    ``best_of``, ``echo`` and ``suffix`` are /v1/completions fields with no chat counterpart, so a
    chat body carrying one is not refused for it; the completions endpoint refuses each of them,
    because it is the endpoint where the value would have changed the answer.
    """
    for body in ({"suffix": "\nEND"}, {"echo": True}, {"best_of": 4}):
        assert shape(body, CHAT) is None, body
        field, _ = refusal(capability(body, endpoint=COMPLETIONS))
        assert field in body, body
    # A large logit_bias is summarised by size rather than dumped into the message.
    result = capability({"logit_bias": {"1": 1, "2": 2, "3": 3}})
    assert result is not None and result.requested == "{3 keys}"


# ---------------------------------------------------------------------------------------------
# Capability: one runtime's answer
# ---------------------------------------------------------------------------------------------


def test_a_runtime_that_applies_nothing_refuses_what_would_change_the_answer() -> None:
    field, _ = refusal(capability({"n": 2}))
    assert field == "n"
    field, _ = refusal(capability({"stop": ["USER:"]}))
    assert field == "stop"
    field, _ = refusal(capability({"logprobs": True}))
    assert field == "logprobs"
    field, _ = refusal(capability({"logprobs": 5}, endpoint=COMPLETIONS))
    assert field == "logprobs"
    field, _ = refusal(capability({"frequency_penalty": 0.5}))
    assert field == "frequency_penalty"
    field, _ = refusal(capability({"presence_penalty": -2}))
    assert field == "presence_penalty"
    field, _ = refusal(capability({"repetition_penalty": 1.1}))
    assert field == "repetition_penalty"
    field, _ = refusal(capability({"min_p": 0.1}))
    assert field == "min_p"
    field, _ = refusal(capability({"logit_bias": {"50256": -100}}))
    assert field == "logit_bias"
    field, _ = refusal(capability({"response_format": {"type": "json_object"}}))
    assert field == "response_format"
    field, _ = refusal(capability({"parallel_tool_calls": False}))
    assert field == "parallel_tool_calls"
    field, _ = refusal(capability({"best_of": 2}, endpoint=COMPLETIONS))
    assert field == "best_of"
    field, _ = refusal(capability({"suffix": "\nEND"}, endpoint=COMPLETIONS))
    assert field == "suffix"
    field, _ = refusal(capability({"echo": True}, endpoint=COMPLETIONS))
    assert field == "echo"

    # The values that name what this server does anyway cost nobody anything.
    assert capability({"n": 1}) is None
    assert capability({"stop": []}) is None
    assert capability({"logprobs": False}) is None
    assert capability({"frequency_penalty": 0, "presence_penalty": 0.0}) is None
    assert capability({"repetition_penalty": 1.0}) is None
    assert capability({"min_p": 0.0}) is None
    assert capability({"logit_bias": {}}) is None
    assert capability({"response_format": {"type": "text"}}) is None
    assert capability({"parallel_tool_calls": True}) is None
    assert capability({"best_of": 1}, endpoint=COMPLETIONS) is None
    assert capability({"suffix": ""}, endpoint=COMPLETIONS) is None
    assert capability({"echo": False}, endpoint=COMPLETIONS) is None


def test_a_runtime_that_applies_a_field_is_not_asked_about_its_capability() -> None:
    served = dict(choices=True, stop=True, logprobs=True, penalties=True, repetition_penalty=True,
                  min_p=True, logit_bias=True, structured_outputs=True, echo=True, suffix=True,
                  best_of=True, parallel_tool_calls=True)
    assert capability({"n": 4}, **served) is None
    assert capability({"stop": ["USER:"]}, **served) is None
    assert capability({"logprobs": True, "top_logprobs": 3}, **served) is None
    assert capability({"logprobs": 5}, endpoint=COMPLETIONS, **served) is None
    assert capability({"frequency_penalty": 0.5}, **served) is None
    assert capability({"response_format": {"type": "json_object"}}, **served) is None
    assert capability({"logit_bias": {"1": 1}}, **served) is None
    assert capability({"min_p": 0.1}, **served) is None
    # ... but the shapes are checked either way, which is what keeps a served field from being a
    # field nothing validates.
    field, _ = refusal(capability({"n": 0}, **served))
    assert field == "n"
    field, _ = refusal(capability({"stop": 5}, **served))
    assert field == "stop"
    field, _ = refusal(capability({"logprobs": True, "top_logprobs": 64}, **served))
    assert field == "top_logprobs"


def test_a_streamed_ranking_is_a_separate_capability_from_logging() -> None:
    """A chunk carries the text of its token, so a ranking would have to travel beside it.

    Answering a streaming request for log probabilities with a stream that carries none produces
    exactly the response this module exists to prevent: one a client cannot tell from a request that
    asked for nothing.
    """
    served = dict(logprobs=True)
    field, _ = refusal(capability({"stream": True, "logprobs": True}, **served))
    assert field == "logprobs"
    field, _ = refusal(
        capability({"stream": True, "logprobs": 5}, endpoint=COMPLETIONS, **served)
    )
    assert field == "logprobs"
    # 0 is a real request here, not the "off" value false is on chat.
    field, _ = refusal(
        capability({"stream": True, "logprobs": 0}, endpoint=COMPLETIONS, **served)
    )
    assert field == "logprobs"

    assert capability({"stream": True, "logprobs": False}, **served) is None
    assert capability({"stream": True}, **served) is None
    assert capability({"stream": True, "logprobs": None}, endpoint=COMPLETIONS, **served) is None
    assert capability({"logprobs": True}, **served) is None
    assert capability({"logprobs": 5}, endpoint=COMPLETIONS, **served) is None
    # The refusal is about the pair, not about either field.
    assert capability(
        {"stream": True, "logprobs": True}, logprobs=True, streaming_logprobs=True
    ) is None


def test_every_refusal_carries_the_field_the_value_and_a_remedy() -> None:
    """Four parts, every time: a caller who cannot tell "dropped" from "wrong spelling" is stuck."""
    for body, endpoint, serves in (
        ({"n": 2}, CHAT, {}),
        ({"stop": ["x"]}, CHAT, {}),
        ({"logprobs": True}, CHAT, {}),
        ({"response_format": {"type": "json_object"}}, CHAT, {}),
        ({"echo": True}, COMPLETIONS, {}),
    ):
        result = audit(body, endpoint=endpoint, serves=ServedFields(**serves))
        assert result is not None
        assert result.field in result.message
        assert result.requested
        assert len(result.message) > 40
        assert "not supported by this server" in result.message


def test_an_unknown_endpoint_name_is_not_silently_treated_as_chat() -> None:
    """The endpoint decides which spelling of two fields is correct, so it is not a free choice.

    Anything that is not :data:`CHAT` reading as completions would have made a mistyped endpoint
    accept ``logprobs: true``, which is the wrong shape there and a real request rather than an
    absent one.
    """
    for endpoint in ("completions-but-misspelled", "Chat", ""):
        with pytest.raises(ValueError, match="endpoint"):
            audit({"n": 2}, endpoint=endpoint, serves=ServedFields(choices=True))
        with pytest.raises(ValueError, match="endpoint"):
            audit_shape({"n": 2}, endpoint=endpoint)
