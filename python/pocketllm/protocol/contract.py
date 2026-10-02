"""Which request fields this front end answers, and what it says when it cannot.

An OpenAI request body is wider than any one runtime. A field whose value would change the
generated text, and which the selected runtime does not apply, has three possible treatments:
honour it, refuse it by name, or accept it and generate something else. Only the first two are
answers a client can act on, and landing in the third by accident is what this module exists to
prevent -- ``n=3`` answered with one choice, ``logprobs=true`` answered with no ranking, both with a
200 and nothing to say anything was dropped.

How the two mature servers draw the line is what fixes the shape of this module. vLLM validates the
body in pydantic on the request model -- types, ranges, and the cross-field rules like
``top_logprobs`` requiring ``logprobs`` -- and lets anything well-formed through to the engine,
which is where the capability questions are settled (``_verify_args`` rejecting ``min_p`` under
speculative decoding, for one). SGLang does the same thing from ``to_sampling_params``: the request
model declares every field, the scheduler applies what it can. Neither asks the HTTP layer whether
the *model* supports a field.

So this module is split the same way, and the split is the point:

* :func:`audit` answers what only the body can answer -- shapes, ranges, and the rules that relate
  two fields to each other. It is endpoint-aware because the two endpoints genuinely disagree:
  ``logprobs`` is a boolean on chat and a count on /v1/completions, where even 0 asks for something.
  This half needs to know nothing about engines and is the same for every runtime.
* :class:`ServedFields` is the capability half, and it is a *declaration* rather than a fixed table
  because it is per runtime: the torch adapter applies ``frequency_penalty`` and the C++ one has no
  such term in its sampler. A runtime states what its answer actually applies, and :func:`audit`
  refuses the rest by name -- which is the treatment vLLM gives ``suffix`` and SGLang gives nothing,
  except that here losing the field is reported instead of silent.

The policy itself -- which fields are refusable and what each refusal says -- was the native C++
front end's, in ``cpp_engine/core/openai_request_fields.cpp`` and ``check_sampling_supported`` in
``cpp_engine/engine/openai_server.cpp``, both deleted with that front end. It is ported rather than
re-decided: the unified front end serves the same engine, so deleting that binary without carrying
the contract over would have traded an honest 400 for a silent ignore on every request naming a
field the engine cannot apply. What the port changes is *where the answer comes from*: those files
asked one engine, and this asks whichever runtime is running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

__all__ = [
    "CHAT",
    "COMPLETIONS",
    "MAX_CHOICES",
    "MAX_LOGPROB_ALTERNATIVES",
    "FieldRefusal",
    "ServedFields",
    "StructuredOutput",
    "audit",
    "audit_shape",
    "is_stop_shape",
    "structured_output_spec",
]


#: The chat-completions endpoint.
CHAT = "chat"
#: The text-completions endpoint, which spells two shared fields differently from chat.
COMPLETIONS = "completions"

#: The largest ``n`` accepted. A request for choices is served by running one generation request per
#: choice, so the field is bounded by a fixed server limit rather than by anything an engine
#: declares: the limit keeps one request from filling the queue with choices that will not be reached
#: before its deadline, and it is set far above what an ordinary client asks for. It is the shape
#: half of the field and holds whatever the capability half says; see :attr:`ServedFields.choices`.
MAX_CHOICES = 128

#: The largest number of ranked alternatives per position, from either endpoint's spelling of it.
#: Each alternative is ranked for every generated position, so the count bounds a per-token cost. 20
#: is the documented chat ceiling; the server does not accept past it rather than truncating.
MAX_LOGPROB_ALTERNATIVES = 20


@dataclass(frozen=True, slots=True)
class FieldRefusal:
    """One refused field: its name, what arrived, and the sentence a client reads.

    ``field`` is separate from ``message`` because it goes into OpenAI's ``param`` slot, which is
    what lets a client act on the refusal without parsing prose.
    """

    field: str
    requested: str
    message: str

    @classmethod
    def build(cls, field: str, requested: Any, actual: str, remedy: str) -> "FieldRefusal":
        """The refusal, with all four parts every one of them carries.

        The field, the value that arrived, what this server does instead, and what the caller can do
        about it -- a caller who cannot tell whether the field was dropped, ignored, or needs a
        different spelling has to read the source to find out, which is the cost this shape avoids.
        """
        rendered = _render(requested)
        return cls(
            field=field,
            requested=rendered,
            message=f'"{field}" = {rendered} is not supported by this server: {actual} {remedy}',
        )


@dataclass(frozen=True, slots=True)
class ServedFields:
    """Which request fields a runtime's answer actually applies.

    Every field defaults to ``False``, which is the safe direction: a runtime that has not said it
    honours a field gets a refusal rather than a request generated as if the field were absent. A
    runtime opts in to what it does.

    ``structured_outputs`` is a capability rather than a field name because that is what it is --
    ``response_format`` is the field, and holding the model to the schema is what the engine either
    can or cannot do.
    """

    #: ``n``: more than one choice.
    choices: bool = False
    #: ``stop``: client sequences matched against the decoded text.
    stop: bool = False
    #: ``logprobs`` / ``top_logprobs`` on a non-streaming response.
    logprobs: bool = False
    #: ``logprobs`` on a streaming response, which needs a ranking beside every chunk's token.
    streaming_logprobs: bool = False
    #: ``frequency_penalty`` and ``presence_penalty``.
    penalties: bool = False
    #: ``repetition_penalty``.
    repetition_penalty: bool = False
    #: ``min_p``.
    min_p: bool = False
    #: ``logit_bias``.
    logit_bias: bool = False
    #: ``response_format``, held to by constrained decoding.
    structured_outputs: bool = False
    #: ``echo`` on /v1/completions: the prompt echoed into the response text.
    echo: bool = False
    #: ``suffix`` on /v1/completions: text appended after the completion.
    suffix: bool = False
    #: ``best_of`` on /v1/completions: candidate sampling with a likelihood comparison.
    best_of: bool = False
    #: ``parallel_tool_calls=false``: the number of calls the model emits is limited.
    parallel_tool_calls: bool = False


#: A runtime that applies every field, so an audit against it asks only about shape.
#:
#: Built from the dataclass rather than written out, because a field added above and forgotten here
#: would be one the shape pass silently stopped checking -- and the shape pass is where the checks
#: that hold on every runtime live.
_EVERYTHING = ServedFields(**{f.name: True for f in fields(ServedFields)})


def audit_shape(body: Mapping[str, Any], *, endpoint: str = CHAT) -> FieldRefusal | None:
    """The first field in ``body`` whose *shape* is wrong, or ``None``.

    The half of :func:`audit` that is the same on every runtime, and it is a separate function
    because the two halves are run by different callers: the host checks shape before it dispatches,
    and the backend checks capability, which needs a runtime to ask. Calling :func:`audit` from the
    host with nothing to say about the runtime would not check shape alone -- it would refuse every
    capability field in the table, which is the opposite of what a shape check is for.
    """
    return audit(body, endpoint=endpoint, serves=_EVERYTHING)


def _absent(value: Any) -> bool:
    return value is None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _default_number(value: Any, expected: float) -> bool:
    return _absent(value) or (_is_number(value) and math.isclose(float(value), expected, abs_tol=1e-9))


def _default_bool(value: Any, expected: bool) -> bool:
    return _absent(value) or (isinstance(value, bool) and value is expected)


def _whole_number(value: Any) -> bool:
    """A count, as opposed to a measure: 2 is one, 2.5 is not.

    Kept separate from the range checks below, because folding it in would report 2.5 as out of
    range and point the caller at the wrong half of the mistake.
    """
    return _is_number(value) and float(value) == math.floor(float(value))


def _number(value: float) -> str:
    if float(value) == math.floor(float(value)) and abs(float(value)) < 1e15:
        return str(int(value))
    return f"{float(value):g}"


def _render(value: Any, budget: int = 80) -> str:
    """The refused value as a message can show it.

    Scalars render as themselves, and a short list of scalars renders in full -- ``stop`` is nearly
    always a few strings, and seeing them is the difference between a useful refusal and a guessing
    game. Anything else is summarised by size, because the caller acts on the field name and a
    ``logit_bias`` can hold hundreds of entries.
    """
    if _absent(value):
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_number(value):
        return _number(float(value))
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                return f"[{len(value)} {'item' if len(value) == 1 else 'items'}]"
            parts.append(_render(item, 0))
            if sum(len(part) for part in parts) > budget:
                return f"[{len(value)} {'item' if len(value) == 1 else 'items'}]"
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, Mapping):
        count = len(value)
        return f"{{{count} {'key' if count == 1 else 'keys'}}}"
    return str(value)


def is_stop_shape(value: Any) -> bool:
    """Whether ``stop`` has the documented shape: a string, or a list of strings."""
    if isinstance(value, str):
        return True
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, str) for item in value)


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    """What ``response_format`` asked for, reduced to what a constrained decode is built from.

    Three shapes are accepted and two of them mean work. ``{"type": "text"}`` is the shape that asks
    for nothing, and it is the value an OpenAI client sends by default, so it is read as a kind
    rather than refused for having no constraint behind it. ``{"type": "json_object"}`` accepts any
    JSON object. ``{"type": "json_schema"}`` carries the object schema to hold the answer to, and
    that schema is the only thing here an engine cannot derive for itself.
    """

    #: ``"text"``, ``"json_object"`` or ``"json_schema"``.
    kind: str
    #: The object schema under ``json_schema.schema``, present only for :attr:`kind` ``json_schema``.
    schema: Mapping[str, Any] | None = None

    @property
    def constrains(self) -> bool:
        """Whether this asks the engine to hold the answer to anything.

        The distinction the adapter branches on: a ``text`` request is generated on whatever path the
        runtime would have used anyway, and only the other two need a token constraint. Keeping it
        a property of the spec rather than a ``kind != "text"`` at every call site is what stops the
        two from disagreeing when a fourth kind is added.
        """
        return self.kind != "text"


def structured_output_spec(response_format: Any) -> StructuredOutput | FieldRefusal:
    """``response_format`` as :class:`StructuredOutput`, or the refusal saying why it is not one.

    The native front end's ``parse_response_format``, minus the two halves that are not about the
    value: whether the running engine can constrain at all is the capability half
    (:attr:`ServedFields.structured_outputs`), and whether the schema is one the engine's validator
    supports is only knowable where the validator is. What stays here is the reading of the field,
    which is the same on every runtime and is what both the host's audit and the ``cpp`` adapter need
    -- read once, so a schema the audit passed is the schema the adapter builds from.
    """
    if not isinstance(response_format, Mapping):
        return FieldRefusal.build(
            "response_format", response_format,
            "response_format is an object with a type, and this value is not an object.",
            'Send {"type": "text"}, {"type": "json_object"}, or a json_schema response_format.',
        )

    kind = response_format.get("type")
    if not isinstance(kind, str):
        return FieldRefusal.build(
            "response_format", response_format,
            "response_format needs a string type, and this value has none.",
            'Send {"type": "text"}, {"type": "json_object"}, or a json_schema response_format.',
        )
    if kind == "text":
        return StructuredOutput(kind="text")
    if kind == "json_object":
        return StructuredOutput(kind="json_object")
    if kind != "json_schema":
        return FieldRefusal.build(
            "response_format", response_format,
            f"response_format type must be 'text', 'json_object' or 'json_schema', not {kind!r}.",
            'Send {"type": "text"}, {"type": "json_object"}, or a json_schema response_format.',
        )

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        return FieldRefusal.build(
            "response_format", response_format,
            "a json_schema response_format needs a json_schema object carrying the schema.",
            'Send {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}.',
        )
    schema = json_schema.get("schema")
    if not isinstance(schema, Mapping):
        return FieldRefusal.build(
            "response_format", json_schema,
            "a json_schema response_format needs an object schema under json_schema.schema.",
            'Send {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}.',
        )
    return StructuredOutput(kind="json_schema", schema=schema)


def audit(  # noqa: C901 - one field per block, in the order the refusal table documents
    body: Mapping[str, Any],
    *,
    endpoint: str = CHAT,
    serves: ServedFields = ServedFields(),
) -> FieldRefusal | None:
    """The first field in ``body`` this server cannot serve, or ``None``.

    Each block below refuses in two steps, and the order between them is the whole design: first
    whether the value has a shape this server can act on at all, then whether the runtime applies
    it. A body that fails both is refused for its shape, because that is the mistake the caller has
    to fix first -- changing ``"stop": 5`` into a list is not the same act as removing a list this
    server will not match.

    A value that names what this server does anyway -- ``n`` of 1, ``logprobs`` false, an empty
    ``stop`` list, penalties of zero -- is accepted, so a client that spells out the defaults is not
    punished for it.
    """
    if endpoint not in {CHAT, COMPLETIONS}:
        raise ValueError(
            f"unknown endpoint {endpoint!r}: the endpoint decides which spelling of "
            f'"logprobs" and "top_logprobs" is correct, so it is not optional'
        )
    chat = endpoint == CHAT

    # How many completions come back. Served by running one request per choice, so what is left to
    # refuse is a count that cannot be served: zero, a fraction, a non-number, or a count past the
    # ceiling -- a request for choices beyond it would put that many requests in the queue, which is
    # a way to make one request cost the server an unbounded amount of memory.
    value = body.get("n")
    if not _absent(value):
        if not _whole_number(value):
            return FieldRefusal.build(
                "n", value,
                "the number of choices is a whole number and this value is not one.",
                'Send "n" as an integer, or omit it for one choice.',
            )
        if float(value) < 1.0 or float(value) > float(MAX_CHOICES):
            return FieldRefusal.build(
                "n", value,
                f"this server generates one request per choice and accepts at most {MAX_CHOICES} "
                "choices in a single request.",
                f'Lower "n" to {MAX_CHOICES} or less; a client that needs more can send the request '
                "again.",
            )
        if not serves.choices and float(value) > 1.0:
            return FieldRefusal.build(
                "n", value,
                "this runtime generates one choice per request.",
                'Remove "n", or run a runtime that serves several choices.',
            )

    # Client stop sequences. A sequence is matched against the decoded text rather than against
    # token ids, so only a value of the documented shape can be one at all -- accepting a number or
    # a nested array would turn a client mistake into a request that silently never stops.
    value = body.get("stop")
    if not _absent(value):
        if not is_stop_shape(value):
            return FieldRefusal.build(
                "stop", value,
                "a stop sequence is a string, or a list of strings, and this value is neither.",
                'Send "stop" as a string or an array of strings.',
            )
        if not serves.stop and list(value) != []:
            return FieldRefusal.build(
                "stop", value,
                "this runtime returns the generated text whole, with no stop sequence matched "
                "against it.",
                'Remove "stop", and truncate the answer yourself.',
            )

    # Per-token log probabilities. The two endpoints spell the same request differently and the
    # difference is not cosmetic: chat takes a boolean and puts the number of ranked alternatives in
    # "top_logprobs", while /v1/completions takes one count in "logprobs" where even 0 includes the
    # sampled token's own probability. A boolean is meaningless on the second and a count on the
    # first, so each is checked against the endpoint that defines it.
    logprobs = body.get("logprobs")
    top_logprobs = body.get("top_logprobs")
    if chat:
        if not _absent(logprobs) and not isinstance(logprobs, bool):
            return FieldRefusal.build(
                "logprobs", logprobs,
                "the chat endpoint takes a boolean: true asks for the sampled token's own log "
                'probability, plus any alternatives named in "top_logprobs".',
                'Send "logprobs" as true or false, or omit it.',
            )
    elif not _absent(logprobs) and (
        not _whole_number(logprobs)
        or float(logprobs) < 0.0
        or float(logprobs) > float(MAX_LOGPROB_ALTERNATIVES)
    ):
        return FieldRefusal.build(
            "logprobs", logprobs,
            f"the number of alternatives to rank per position is a whole number from 0 to "
            f"{MAX_LOGPROB_ALTERNATIVES}.",
            f'Lower "logprobs" to {MAX_LOGPROB_ALTERNATIVES} or less; 0 reports the sampled '
            "token's own probability and no alternatives.",
        )

    if chat:
        if not _absent(top_logprobs) and (
            not _whole_number(top_logprobs)
            or float(top_logprobs) < 0.0
            or float(top_logprobs) > float(MAX_LOGPROB_ALTERNATIVES)
        ):
            return FieldRefusal.build(
                "top_logprobs", top_logprobs,
                f"the number of alternatives to rank per position is a whole number from 0 to "
                f"{MAX_LOGPROB_ALTERNATIVES}.",
                f'Lower "top_logprobs" to {MAX_LOGPROB_ALTERNATIVES} or less; 0 reports the '
                "sampled token's own probability and no alternatives.",
            )
        # Alternatives are reported only for a request that asked for log probabilities, so naming a
        # count here asks for something this request never turns on -- and answering with no
        # "logprobs" object at all reads as a server that ignored the field. A count of 0 is inert
        # on its own and accepted.
        if (
            _whole_number(top_logprobs)
            and float(top_logprobs) > 0.0
            and not (isinstance(logprobs, bool) and logprobs)
        ):
            return FieldRefusal.build(
                "top_logprobs", top_logprobs,
                "alternatives are reported only for a request that asks for log probabilities, and "
                'this one\'s "logprobs" is absent or false.',
                'Set "logprobs" to true, or remove "top_logprobs".',
            )
    elif not _absent(top_logprobs):
        return FieldRefusal.build(
            "top_logprobs", top_logprobs,
            'the completions endpoint names the number of alternatives in "logprobs" itself.',
            'Put the count in "logprobs" and remove "top_logprobs".',
        )

    # Whether the request asked for a ranking, in the spelling its endpoint uses: false is off on
    # chat, while /v1/completions spells the same thing as the count 0. Both were validated above,
    # so this reads them rather than re-deriving them.
    asked = (
        (isinstance(logprobs, bool) and logprobs)
        if chat
        else _whole_number(logprobs)
    )
    if asked and not serves.logprobs:
        return FieldRefusal.build(
            "logprobs", logprobs,
            "this runtime does not report per-token log probabilities.",
            'Remove "logprobs", or run a runtime that ranks them.',
        )

    streaming = body.get("stream") is True
    if asked and streaming and not serves.streaming_logprobs:
        return FieldRefusal.build(
            "logprobs", logprobs,
            "log probabilities are reported on a non-streaming response, and a streamed chunk "
            "carries the text of its token with no ranking beside it.",
            'Remove "stream", or remove "logprobs".',
        )

    # stream_options.include_usage is refused only when it would change the response: a
    # non-streaming completion already carries "usage", which is the whole thing the option asks
    # for, so that request is answered with exactly what it asked for and is left alone.
    if streaming:
        stream_options = body.get("stream_options")
        if isinstance(stream_options, Mapping) and not _default_bool(
            stream_options.get("include_usage"), False
        ):
            return FieldRefusal.build(
                "stream_options.include_usage", stream_options.get("include_usage"),
                "a streaming response is a sequence of delta chunks followed by \"[DONE]\", and "
                "none of them carries a \"usage\" object.",
                'Remove "stream_options", or set "include_usage" to false; usage is reported on '
                "non-streaming requests.",
            )

    # Repetition controls. A sampler with no such term generates as if the penalty were its default,
    # which can be the opposite of what the caller asked for when the penalty was there to suppress
    # a loop.
    if not serves.penalties:
        for field in ("frequency_penalty", "presence_penalty"):
            value = body.get(field)
            if not _default_number(value, 0.0):
                return FieldRefusal.build(
                    field, value,
                    "this sampler has no repetition or presence term, so the request is generated "
                    "as if the penalty were 0.",
                    f'Remove "{field}", or set it to 0.',
                )
    if not serves.repetition_penalty:
        value = body.get("repetition_penalty")
        if not _default_number(value, 1.0):
            return FieldRefusal.build(
                "repetition_penalty", value,
                "this sampler has no repetition term, so the request is generated as if the penalty "
                "were 1.",
                'Remove "repetition_penalty", or set it to 1.',
            )
    if not serves.min_p:
        value = body.get("min_p")
        if not _default_number(value, 0.0):
            return FieldRefusal.build(
                "min_p", value,
                "this sampler has no min-p truncation stage, so the request is generated with "
                "whatever nucleus the other controls leave.",
                'Remove "min_p", or set it to 0.',
            )

    # Per-token bias. Dropping it silently changes the distribution for exactly the tokens the
    # caller cared about most.
    if not serves.logit_bias:
        value = body.get("logit_bias")
        if not _absent(value) and not (isinstance(value, Mapping) and not value):
            return FieldRefusal.build(
                "logit_bias", value,
                "no per-token bias is applied, so every biased token is sampled at its unmodified "
                "probability.",
                'Remove "logit_bias".',
            )

    # Structured outputs. The value is read the same way on every runtime, so the shape is checked
    # here and only the capability is checked below -- a runtime with no constrained decoding still
    # needs to be told that `{"type": "json_object"}` is a shape it can never accept, rather than
    # being handed one the host thought was fine.
    value = body.get("response_format")
    if not _absent(value):
        spec = structured_output_spec(value)
        if isinstance(spec, FieldRefusal):
            return spec
        if spec.constrains and not serves.structured_outputs:
            return FieldRefusal.build(
                "response_format", value,
                "this runtime applies no constrained decoding, so the answer is unconstrained text.",
                'Remove "response_format", or run a runtime with structured outputs.',
            )

    if not chat:
        # Candidate sampling with a likelihood comparison, and text appended after the completion.
        if not serves.best_of:
            value = body.get("best_of")
            if not _default_number(value, 1.0):
                return FieldRefusal.build(
                    "best_of", value,
                    "one candidate is generated per request, and no second candidate is sampled to "
                    "compare it against.",
                    'Remove "best_of", or set it to 1.',
                )
        if not serves.suffix:
            value = body.get("suffix")
            if not _absent(value) and not (isinstance(value, str) and not value):
                return FieldRefusal.build(
                    "suffix", value,
                    "the completion is returned on its own, with no suffix text appended after it.",
                    'Remove "suffix".',
                )
        if not serves.echo:
            value = body.get("echo")
            if not _default_bool(value, False):
                return FieldRefusal.build(
                    "echo", value,
                    'the response "text" holds only the generated continuation, never the prompt.',
                    'Remove "echo", or set it to false.',
                )
        return None

    # Tool use. The definitions reach the chat template either way, and the selection policy is
    # applied by the shared request builder rather than by the runtime -- "none" drops the
    # definitions and "required" or a named function becomes an instruction in the prompt, which
    # every runtime that encodes through that builder gets. So the only tool field left here is the
    # one nothing limits: how many calls the model emits.
    if not serves.parallel_tool_calls:
        value = body.get("parallel_tool_calls")
        if not _default_bool(value, True):
            return FieldRefusal.build(
                "parallel_tool_calls", value,
                "the number of tool calls the model emits is not limited.",
                'Remove "parallel_tool_calls", and keep the first call yourself if only one is '
                "acceptable.",
            )
    return None
