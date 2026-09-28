"""Constrained decoding, driven through the binding the serving path uses.

`response_format` reaches the engine as a token constraint, and the constraint is a mask computed
over the vocabulary *piece by piece*. Three things can be wrong with that and only one of them is
visible from the adapter's side: that the wrong factory was chosen for a shape, that the constraint
was built over a vocabulary the engine does not sample from, or that the mask itself admits text
the schema forbids. The adapter's tests cover the first; this module covers the other two, which is
why it exists rather than being folded into them.

The vocabulary is written here rather than read from a checkpoint, and it is deliberately not a
byte-level BPE vocabulary of the size a real model has. What it has to be is *spelled the way a real
one is*: `Ġ` for a space is the whole reason a mask cannot be built anywhere but over the engine's
own pieces, and a synthetic vocabulary that spelled a space `" "` would let a broken implementation
pass. It is also why these tests run with no checkpoint and no card -- the thing under test is a
predicate over a vocabulary, and the vocabulary is the input.
"""

from __future__ import annotations

import json

import pytest

pocketllm_cpp = pytest.importorskip("pocketllm_cpp")

#: A vocabulary in the shape `tokenizer.json` has, small enough that every piece can be named in a
#: test and complete enough that a JSON object can be written out of it. `Ġ` is the byte-level BPE
#: spelling of a space, and it is here so that a mask built from the raw entries -- rather than from
#: `decode_piece` -- is a failure this file can see.
VOCABULARY = {
    "{": 0,
    "}": 1,
    '"': 2,
    ":": 3,
    "true": 4,
    "a": 5,
    "hello": 6,
    "Ġ": 7,
    "city": 8,
    ",": 9,
    "Paris": 10,
}

#: The schema the tests below write an answer to: one required string property, and nothing else
#: allowed, so every token that is not part of `{"city": "..."}` is a token the mask must refuse.
CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


@pytest.fixture()
def tokenizer(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    # `ensure_ascii=False` because the pieces are read as bytes: a released `tokenizer.json` stores
    # a non-ASCII piece as itself, and letting `json.dump` escape it would build a vocabulary no
    # checkpoint has and test a spelling nothing produces.
    (checkpoint / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": VOCABULARY, "merges": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return pocketllm_cpp.Tokenizer(str(checkpoint))


def _ids(tokenizer) -> dict[str, int]:
    ids = {piece: tokenizer.token_id(piece) for piece in VOCABULARY}
    missing = sorted(piece for piece, value in ids.items() if value < 0)
    assert not missing, f"the vocabulary did not load these pieces: {missing}"
    return ids


def test_a_piece_is_what_the_tokenizer_emits_and_not_what_the_file_stores(tokenizer) -> None:
    """`Ġ` is a space, and that is the whole reason the vocabulary has to cross the binding.

    A mask built from the raw vocabulary entries would compare `Ġ` against the answer text, refuse
    every space, and refuse every token that continues a word -- so a constrained answer would come
    back empty, or as one unbroken string, with nothing anywhere saying why.
    """
    ids = _ids(tokenizer)
    assert tokenizer.decode_piece(ids["Ġ"]) == " "
    assert tokenizer.decode_piece(ids["city"]) == "city"
    assert tokenizer.decode_piece(ids['"']) == '"'
    # The raw entry is still reachable and still says `Ġ`, which is the fact being contrasted.
    assert tokenizer.token_id("Ġ") >= 0


def test_a_schema_is_met_one_token_at_a_time_as_the_answer_is_written(tokenizer) -> None:
    """The constraint's whole interface: may this token be taken, and is the answer done.

    Both answers are needed by the sampler at different moments -- the mask decides what it may draw
    from, and completion is what ends the request -- so a constraint that only answered the first
    would generate a correct object and then keep going.
    """
    ids = _ids(tokenizer)
    constraint = pocketllm_cpp.make_json_schema_constraint(tokenizer, json.dumps(CITY_SCHEMA))

    assert not constraint.is_complete()
    # Every piece of `{"city": "Paris"}` and the closing brace last: the object names a property the
    # schema does not allow is refused, and a space inside the string is a `Ġ` piece rather than a
    # literal one -- which is the tokenizer's spelling and not an accident of this vocabulary.
    pieces = ("{", '"', "city", '"', ":", "Ġ", '"', "Paris", '"', "}")
    for piece in pieces[:-1]:
        assert constraint.accept_token(ids[piece]), f"the mask refused {piece!r} mid-answer"
        assert not constraint.is_complete(), f"{piece!r} closed an object that is still open"
    assert constraint.accept_token(ids[pieces[-1]])
    assert constraint.is_complete(), "a satisfied schema did not report completion"


def test_a_token_the_schema_forbids_is_refused_without_moving_the_constraint(tokenizer) -> None:
    """Refused, and the answer so far survives it.

    The sampler asks about a token and then either takes it or does not; a refusal that also advanced
    the state would corrupt every subsequent question, and the failure would look like a constraint
    that admits the wrong tokens rather than one that was never allowed to say no.
    """
    ids = _ids(tokenizer)
    constraint = pocketllm_cpp.make_json_schema_constraint(tokenizer, json.dumps(CITY_SCHEMA))
    for piece in ("{", '"', "city", '"', ":"):
        assert constraint.accept_token(ids[piece])

    # The value position of a `city` property takes a string. Prose is not one, and neither is a
    # key: this is the mask doing the work rather than the model being asked nicely.
    assert not constraint.accept_token(ids["hello"])
    assert not constraint.accept_token(ids["city"])
    assert not constraint.accept_token(ids["}"])
    # ... and the object is still open, so the answer continues from where it was.
    assert constraint.accept_token(ids["Ġ"])
    assert constraint.accept_token(ids['"'])
    assert constraint.accept_token(ids["Paris"])
    assert constraint.accept_token(ids['"'])
    assert constraint.accept_token(ids["}"])
    assert constraint.is_complete()


def test_a_json_object_constraint_takes_any_object_and_still_refuses_prose(tokenizer) -> None:
    """`{"type": "json_object"}` derives its grammar rather than being given one, so the two
    factories are not interchangeable: this one has no `city` to insist on, and it is the only one
    that can be built for a request that supplied no schema."""
    ids = _ids(tokenizer)
    constraint = pocketllm_cpp.make_json_object_constraint(tokenizer)

    assert not constraint.accept_token(ids["hello"])
    assert not constraint.accept_token(ids["a"])
    assert constraint.accept_token(ids["{"])
    for piece in ('"', "hello", '"', ":", "Ġ", "true"):
        assert constraint.accept_token(ids[piece]), f"an object may carry any property, not {piece!r}"
    assert constraint.accept_token(ids["}"]), "the object could not be closed"
    assert constraint.is_complete()


def test_reset_returns_the_constraint_to_its_initial_state(tokenizer) -> None:
    """A constraint is reused only by a caller that resets it, and the state it carries is the answer
    so far -- so a reset that left any of it behind would let the next answer start mid-object."""
    ids = _ids(tokenizer)
    constraint = pocketllm_cpp.make_json_object_constraint(tokenizer)
    for piece in ("{", '"', "a", '"', ":", "true", "}"):
        assert constraint.accept_token(ids[piece])
    assert constraint.is_complete()

    constraint.reset()
    assert not constraint.is_complete()
    # The first token of an answer is `{` again, and a value is not: proof the state really went back.
    assert not constraint.accept_token(ids["true"])
    assert constraint.accept_token(ids["{"])


def test_a_schema_keyword_the_validator_does_not_implement_is_refused_at_construction(
    tokenizer,
) -> None:
    """Refused rather than ignored, which is the difference between a constrained answer and one
    that looks constrained.

    A `pattern` silently dropped is a mask that admits every string where the caller asked for a
    digit-only one, and nothing in the response would say so. The refusal happens where the schema
    is read, so it reaches the client as a 400 on the request and not as an answer that happens to
    be wrong.
    """
    for keyword, value in (("pattern", "^[0-9]+$"), ("anyOf", [{"type": "string"}]), ("not", {})):
        schema = {"type": "object", "properties": {"city": {"type": "string", keyword: value}}}
        with pytest.raises(Exception, match=f"'{keyword}' is not supported"):
            pocketllm_cpp.make_json_schema_constraint(tokenizer, json.dumps(schema))


def test_a_schema_that_is_not_an_object_is_refused(tokenizer) -> None:
    """The root of a JSON Schema is an object, and a JSON value of any other shape is not one."""
    with pytest.raises(Exception, match="root must be an object"):
        pocketllm_cpp.make_json_schema_constraint(tokenizer, json.dumps([{"type": "object"}]))
