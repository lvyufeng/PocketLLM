"""Reading Qwen's XML tool-call syntax back into OpenAI tool calls.

The parser is the only place the server turns generated text into a
``tool_calls`` array, so the cases here are the ones that decide what a client
is shown: a call whose arguments are typed from the schema, a call with no
schema at all, and the malformed or truncated completions that must yield no
call rather than a half-read one.
"""

import json
import re

from src.encoding.qwen_tool_calls import parse

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "integer"},
                "days": {"type": "number"},
                "metric": {"type": "boolean"},
                "note": {"type": ["string", "null"]},
                "options": {"type": "object"},
            },
        },
    },
}


def call(name, *params):
    """Render the XML the Qwen chat template emits for one call."""
    body = f"<tool_call>\n<function={name}>\n"
    for key, value in params:
        body += f"<parameter={key}>\n{value}\n</parameter>\n"
    body += "</function>\n</tool_call>"
    return body


def arguments(tool_calls, index=0):
    return json.loads(tool_calls[index]["function"]["arguments"])


def test_arguments_take_their_type_from_the_declared_schema():
    text = call(
        "get_weather",
        ("city", "Paris"),
        ("zip", "94107"),
        ("days", "3.0"),
        ("metric", "True"),
        ("note", "null"),
        ("options", '{"a": [1, 2]}'),
    )

    content, tool_calls = parse(text, [WEATHER_TOOL])

    assert content == ""
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert arguments(tool_calls) == {
        "city": "Paris",
        # Integer, number and boolean come back as the JSON types the schema
        # promised, and an integral float is written as an integer.
        "zip": 94107,
        "days": 3,
        "metric": True,
        "note": None,
        "options": {"a": [1, 2]},
    }


def test_a_call_without_a_schema_keeps_every_argument_as_text():
    """Without a declaration there is nothing to coerce against, and guessing
    would rewrite values such as a zero-padded id or a leading ``+``."""
    text = call("get_weather", ("zip", "01234"), ("metric", "True"))

    content, tool_calls = parse(text, None)

    assert content == ""
    assert arguments(tool_calls) == {"zip": "01234", "metric": "True"}


def test_a_value_that_does_not_parse_as_its_declared_type_is_left_alone():
    text = call("get_weather", ("zip", "not a number"), ("metric", "yes"))
    _, tool_calls = parse(text, [WEATHER_TOOL])

    # "yes" is a boolean-ish word the template never emits, and an integer field
    # holding prose is not an integer; both survive as the text that was written.
    assert arguments(tool_calls) == {"zip": "not a number", "metric": False}


def test_a_value_whose_declared_type_is_unknown_stays_text():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"v": {"type": "date"}}},
        },
    }
    _, tool_calls = parse(call("f", ("v", "2026-09-15")), [tool])

    assert arguments(tool_calls) == {"v": "2026-09-15"}


def test_a_bare_function_object_is_accepted_as_a_tool_definition():
    """A client may send the function without the OpenAI envelope, and the
    template forwards whichever form it was given."""
    bare = WEATHER_TOOL["function"]
    _, tool_calls = parse(call("get_weather", ("zip", "94107")), [bare])

    assert arguments(tool_calls) == {"zip": 94107}


def test_a_zero_argument_call_produces_an_empty_argument_object():
    _, tool_calls = parse(call("list_files"), [WEATHER_TOOL])

    assert tool_calls[0]["function"] == {"name": "list_files", "arguments": "{}"}


def test_a_python_literal_container_is_read_like_json():
    """Qwen's own reference path falls back to a Python literal, so
    ``{'a': 1}`` -- single quotes, no double quotes anywhere -- is a common
    near-miss from the model rather than a different value."""
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"v": {"type": "array"}}},
        },
    }
    _, tool_calls = parse(call("f", ("v", "{'a': [1, 2]}")), [tool])

    assert arguments(tool_calls) == {"v": {"a": [1, 2]}}


def test_two_calls_in_one_completion_are_both_reported_in_order():
    text = call("get_weather", ("city", "Paris")) + call("get_time", ("zone", "UTC"))

    content, tool_calls = parse(text, [WEATHER_TOOL])

    assert content == ""
    assert [tc["function"]["name"] for tc in tool_calls] == ["get_weather", "get_time"]
    assert arguments(tool_calls, 1) == {"zone": "UTC"}


def test_text_around_the_calls_stays_in_the_content():
    text = f"I'll check both.\n{call('get_weather', ('city', 'Paris'))}\nDone."

    content, tool_calls = parse(text, [WEATHER_TOOL])

    assert content == "I'll check both.\n\nDone."
    assert len(tool_calls) == 1


def test_each_call_gets_its_own_id_in_the_protocol_shape():
    text = call("get_weather", ("city", "Paris")) + call("get_weather", ("city", "Rome"))
    _, tool_calls = parse(text, [WEATHER_TOOL])

    ids = [tc["id"] for tc in tool_calls]
    assert len(set(ids)) == 2
    assert all(re.fullmatch(r"call_[0-9a-f]{24}", i) for i in ids)


def test_a_completion_with_no_call_is_not_a_tool_call():
    assert parse("It is 21 degrees in Paris.", [WEATHER_TOOL]) is None
    assert parse("", [WEATHER_TOOL]) is None


def test_a_truncated_call_yields_nothing_rather_than_the_arguments_so_far():
    """The token budget can cut a call in half.  The parameters that arrived
    look complete, which is exactly why they must not be reported."""
    truncated = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    assert parse(truncated, [WEATHER_TOOL]) is None

    # The closing tag is present but the function body after it is not, so the
    # element itself is never closed.
    assert parse("<tool_call>\n<function=f>\n</function>", [WEATHER_TOOL]) is None


def test_a_malformed_call_yields_nothing_and_leaves_the_text_alone():
    assert parse("<tool_call>\nno function here\n</tool_call>", [WEATHER_TOOL]) is None
    # A parameter that is opened and never closed.
    assert parse(
        "<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>",
        [WEATHER_TOOL],
    ) is None
    # A second value for one name: reporting either would drop the other.
    assert parse(call("f", ("a", "1"), ("a", "2")), [WEATHER_TOOL]) is None
    # Junk after the closing tag of the function, inside the element.
    assert parse(
        "<tool_call>\n<function=f>\n</function>trailing\n</tool_call>", [WEATHER_TOOL]
    ) is None


def test_prose_between_two_calls_yields_nothing():
    """The template tells the model not to write anything between calls.  A
    completion that does has an order the caller would not see, so neither call
    is reported."""
    text = call("a") + "\nactually, wait\n" + call("b")

    assert parse(text, [WEATHER_TOOL]) is None


def test_a_multiline_parameter_value_keeps_its_newlines():
    tool = {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {"body": {"type": "string"}}},
        },
    }
    _, tool_calls = parse(call("write_file", ("body", "line one\nline two")), [tool])

    assert arguments(tool_calls) == {"body": "line one\nline two"}


def test_the_template_omitting_a_newline_still_parses():
    """Whitespace between the elements is not part of the syntax; a model that
    writes the whole call on one line means the same call."""
    text = "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"

    content, tool_calls = parse(text, [WEATHER_TOOL])

    assert content == ""
    assert arguments(tool_calls) == {"city": "Paris"}
