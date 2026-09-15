"""Tool-call syntax used by Qwen's chat template family.

The checkpoint's own chat template renders a tool call as XML rather than as
JSON:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

The shape carries no type information, so ``<parameter=zip>`` holding ``12345``
is indistinguishable from an integer unless the tool schema is consulted.  The
schema is therefore an argument to :func:`parse` rather than something this
module guesses from the text: with no declaration a value stays a string, which
is what the template does when it renders a string argument, and a value the
schema calls an integer becomes one.

Parsing is all-or-nothing.  A completion that opens a ``<tool_call>`` this module
cannot read in full -- truncated at the token budget, malformed, or holding two
calls separated by prose -- yields no calls at all, and the caller is expected to
leave the text in the content field rather than show a client a half-read call
whose arguments look complete.  Nothing here raises on model output; every
failure is a return value.
"""

from __future__ import annotations

import ast
import json
import uuid
from typing import Any, Optional

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
FUNCTION_START = "<function="
FUNCTION_END = "</function>"
PARAMETER_START = "<parameter="
PARAMETER_END = "</parameter>"

# Declared JSON Schema types, grouped by how a value read back out of the XML
# should be interpreted.  A type that appears in none of these is left as the raw
# string, which is the only reading that cannot corrupt it.
_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})
_INTEGER_TYPES = frozenset(
    {"integer", "int", "int32", "int64", "uint", "long", "short", "unsigned"}
)
_NUMBER_TYPES = frozenset({"number", "float", "double", "num", "decimal"})
_BOOLEAN_TYPES = frozenset({"boolean", "bool", "binary"})
_STRUCTURED_PREFIXES = ("object", "array", "arr", "dict", "list", "sequence", "map")

_WHITESPACE = " \t\r\n"


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in _WHITESPACE:
        index += 1
    return index


def _tool_schemas(tools: Any) -> dict[str, dict[str, Any]]:
    """Map tool name to its declared parameter properties.

    Accepts both the OpenAI envelope (``{"type": "function", "function": {...}}``)
    and a bare function object, because a client is free to send either and the
    template passes both through.  A tool with no usable schema is still recorded,
    as an empty property map, so a call to it is parsed with every argument left a
    string rather than dropped.
    """
    schemas: dict[str, dict[str, Any]] = {}
    if not isinstance(tools, list):
        return schemas
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        properties: dict[str, Any] = {}
        if isinstance(parameters, dict) and isinstance(parameters.get("properties"), dict):
            properties = parameters["properties"]
        schemas[name] = properties
    return schemas


def _declared_type(schemas: dict[str, dict[str, Any]], tool: str, param: str) -> Optional[str]:
    schema = schemas.get(tool, {}).get(param)
    if not isinstance(schema, dict):
        return None
    declared = schema.get("type")
    if isinstance(declared, list):
        # A union such as ["string", "null"]: the first entry that is not "null"
        # is the one the model was asked to emit.
        declared = next((item for item in declared if item != "null"), None)
    return declared if isinstance(declared, str) else None


def _coerce(raw: str, declared: Optional[str]) -> Any:
    """Read one parameter body as the type the schema declares for it.

    The fallback at every step is the raw string: when the declared type is
    unknown, or the value does not parse as the type it claims, the text is
    returned unchanged rather than replaced by a guess.  ``null`` is the one
    spelling read the same way whatever the declared type is, because a model has
    no other way to say "absent" inside this syntax.
    """
    text = raw.strip()
    if text == "null":
        return None
    kind = (declared or "").lower()
    if kind in _STRING_TYPES or not kind:
        return raw
    if kind in _BOOLEAN_TYPES:
        return text.lower() == "true"
    if kind in _INTEGER_TYPES:
        try:
            return int(text)
        except ValueError:
            return raw
    if kind in _NUMBER_TYPES:
        try:
            number = float(text)
        except ValueError:
            return raw
        # A schema that says "number" is satisfied by an integer, and writing 2.0
        # where the model emitted 2 makes the JSON the caller sees differ from the
        # JSON it would have written itself.
        return int(number) if number.is_integer() else number
    if kind.startswith(_STRUCTURED_PREFIXES):
        return _structured(text, raw)
    # An unrecognised type name.  Guessing between string and JSON here is how a
    # value like "01234" or "{not json}" gets silently rewritten.
    return raw


def _structured(text: str, raw: str) -> Any:
    """Parse a container value the schema declares.

    The model was asked for JSON and usually produces it, but Qwen's own
    reference path also accepts Python literals -- ``{'a': 1}`` with single
    quotes is a common miss -- so that is tried too.
    """
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    return raw


def _parse_call(body: str) -> Optional[tuple[str, list[tuple[str, str]]]]:
    """Parse the inside of one ``<tool_call>`` element.

    The template emits ``\\n<function=NAME>\\n<parameter=A>\\nVALUE\\n</parameter>
    \\n</function>\\n``.  Whitespace between the elements is not treated as
    significant -- a model that omits a newline still means the same call, and
    failing on that would cost the caller the whole parse -- but every tag has to
    be the one this syntax defines, in order, and every parameter has to be
    closed.  Returns ``(name, [(param, value), ...])`` or ``None``.
    """
    index = _skip_space(body, 0)
    if not body.startswith(FUNCTION_START, index):
        return None
    index += len(FUNCTION_START)
    name_end = body.find(">", index)
    if name_end < 0:
        return None
    name = body[index:name_end].strip()
    if not name or "<" in name:
        return None
    index = name_end + 1

    params: list[tuple[str, str]] = []
    seen: set[str] = set()
    while True:
        index = _skip_space(body, index)
        if body.startswith(FUNCTION_END, index):
            index += len(FUNCTION_END)
            break
        if not body.startswith(PARAMETER_START, index):
            return None
        index += len(PARAMETER_START)
        name_end = body.find(">", index)
        if name_end < 0:
            return None
        param = body[index:name_end].strip()
        if not param or "<" in param:
            return None
        if param in seen:
            # Two values for one name cannot both be reported, and picking one
            # would drop the other silently.
            return None
        seen.add(param)
        index = name_end + 1
        close = body.find(PARAMETER_END, index)
        if close < 0:
            return None
        value = body[index:close]
        # Exactly the newline the template puts after the opening tag and before
        # the closing one; anything else is part of the value, which may span
        # lines.
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        params.append((param, value))
        index = close + len(PARAMETER_END)

    if _skip_space(body, index) != len(body):
        return None
    return name, params


def parse(text: str, tools: Any = None) -> Optional[tuple[str, list[dict[str, Any]]]]:
    """Split a completion into content and OpenAI-format tool calls.

    Returns ``None`` when the text carries no tool call, and also when it carries
    one that cannot be read in full -- the two cases a caller handles the same
    way, by leaving the text alone.  Otherwise returns ``(content, tool_calls)``,
    the calls in the shape the OpenAI API defines with ``arguments`` included as
    a JSON string.
    """
    opening = text.find(TOOL_CALL_START)
    if opening < 0:
        return None

    schemas = _tool_schemas(tools)
    tool_calls: list[dict[str, Any]] = []
    remaining = text
    index = opening
    while True:
        index = _skip_space(remaining, index)
        if not remaining.startswith(TOOL_CALL_START, index):
            break
        body_start = index + len(TOOL_CALL_START)
        body_end = remaining.find(TOOL_CALL_END, body_start)
        if body_end < 0:
            # Truncated mid-call: the closing tag never arrived, so the arguments
            # on hand are a prefix of what the model meant to write.
            return None
        call = _parse_call(remaining[body_start:body_end])
        if call is None:
            return None
        name, params = call
        arguments = {
            param: _coerce(value, _declared_type(schemas, name, param))
            for param, value in params
        }
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
        remaining = remaining[:index] + remaining[body_end + len(TOOL_CALL_END) :]

    if TOOL_CALL_START in remaining:
        # Text between two calls.  The template tells the model not to write any,
        # and reporting the calls on either side of it would put an answer the
        # caller never saw in a fixed order.
        return None
    if not tool_calls:
        return None
    return remaining.strip(), tool_calls
