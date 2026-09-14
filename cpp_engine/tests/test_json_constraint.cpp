#include "json_constraint.hpp"
#include "json_lite.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace pocket;

void test_basic_object() {
    JsonConstraint constraint = JsonConstraint::json_object();
    assert(constraint.accept("{"));
    assert(constraint.accept("\"key\""));
    assert(constraint.accept(":"));
    assert(constraint.accept("\"value\""));
    assert(constraint.accept("}"));
    assert(constraint.is_complete());
}

void test_reject_invalid_syntax() {
    JsonConstraint comma = JsonConstraint::json_object();
    assert(comma.accept("{"));
    assert(!comma.accept(","));

    JsonConstraint missing_colon = JsonConstraint::json_object();
    assert(missing_colon.accept("{\"key\""));
    assert(!missing_colon.accept("1"));

    JsonConstraint trailing_comma = JsonConstraint::json_object();
    assert(trailing_comma.accept("{\"a\":1,"));
    assert(!trailing_comma.accept("}"));

    JsonConstraint leading_zero = JsonConstraint::json_object();
    assert(leading_zero.accept("{\"a\":"));
    assert(!leading_zero.accept("01"));

    JsonConstraint bad_literal = JsonConstraint::json_object();
    assert(bad_literal.accept("{\"a\":"));
    assert(!bad_literal.accept("truX"));
}

void test_nested_values() {
    JsonConstraint constraint = JsonConstraint::json_object();
    assert(constraint.accept("{\"outer\":{\"inner\":123}}"));
    assert(constraint.is_complete());

    JsonConstraint arrays = JsonConstraint::json_object();
    assert(arrays.accept("{\"arr\":[1,2,{\"ok\":true}]}"));
    assert(arrays.is_complete());
}

void test_literals_and_escapes() {
    JsonConstraint constraint = JsonConstraint::json_object();
    assert(constraint.accept("{\"t\":true,\"f\":false,\"n\":null}"));
    assert(constraint.is_complete());

    JsonConstraint escaped = JsonConstraint::json_object();
    assert(escaped.accept("{\"line\\n\":\"quote: \\\"ok\\\"\"}"));
    assert(escaped.is_complete());

    JsonConstraint invalid_escape = JsonConstraint::json_object();
    assert(invalid_escape.accept("{\"a\":\""));
    assert(!invalid_escape.accept("\\x"));
}

void test_can_accept_is_speculative() {
    JsonConstraint constraint = JsonConstraint::json_object();
    assert(constraint.accept("{\"key\":"));
    assert(constraint.can_accept("\"value\"}"));
    assert(constraint.can_accept("123}"));
    assert(constraint.can_accept("true}"));
    assert(!constraint.is_complete());
    assert(constraint.accept("\"value\"}"));
    assert(constraint.is_complete());
}

void test_schema_required_and_types() {
    const JsonValue schema = parse_json(R"({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    })");

    JsonConstraint valid = JsonConstraint::from_schema(schema);
    assert(valid.accept(R"({"name":"Alice","age":42})"));
    assert(valid.is_complete());

    JsonConstraint missing = JsonConstraint::from_schema(schema);
    assert(missing.accept(R"({"name":"Alice")"));
    assert(!missing.accept("}"));

    JsonConstraint wrong_type = JsonConstraint::from_schema(schema);
    assert(wrong_type.accept(R"({"name":"Alice","age":)"));
    assert(!wrong_type.accept("1.5}"));
}

void test_schema_enum_and_bounds() {
    const JsonValue enum_schema = parse_json(R"({
        "type": "object",
        "properties": {"color": {"enum": ["red", "green", "blue"]}}
    })");

    JsonConstraint valid = JsonConstraint::from_schema(enum_schema);
    assert(valid.accept(R"({"color":"red"})"));
    assert(valid.is_complete());

    JsonConstraint invalid = JsonConstraint::from_schema(enum_schema);
    assert(invalid.accept(R"({"color":)"));
    assert(!invalid.accept("\"yellow\"}"));

    const JsonValue array_schema = parse_json(R"({
        "type": "object",
        "properties": {"items": {
            "type": "array", "minItems": 2, "maxItems": 3,
            "items": {"type": "integer"}
        }}
    })");
    JsonConstraint too_short = JsonConstraint::from_schema(array_schema);
    assert(too_short.accept(R"({"items":[1)"));
    assert(!too_short.accept("]}"));

    JsonConstraint valid_array = JsonConstraint::from_schema(array_schema);
    assert(valid_array.accept(R"({"items":[1,2]})"));
    assert(valid_array.is_complete());
}

void test_schema_additional_properties_and_refs() {
    const JsonValue schema = parse_json(R"({
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "additionalProperties": false
    })");
    JsonConstraint constraint = JsonConstraint::from_schema(schema);
    assert(constraint.accept(R"({"id":1})"));
    assert(constraint.is_complete());

    JsonConstraint extra = JsonConstraint::from_schema(schema);
    assert(extra.accept(R"({"unknown":)"));
    assert(!extra.accept("1}"));

    const JsonValue ref_schema = parse_json(R"({
        "$defs": {"positive": {"type": "integer", "minimum": 1}},
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/positive"}}
    })");
    JsonConstraint ref = JsonConstraint::from_schema(ref_schema);
    assert(ref.accept(R"({"value":2})"));
    assert(ref.is_complete());

    JsonConstraint ref_invalid = JsonConstraint::from_schema(ref_schema);
    assert(ref_invalid.accept(R"({"value":0)"));
    assert(!ref_invalid.accept("}"));
}

void test_unsupported_keywords_throw() {
    for (const char* keyword : {"anyOf", "oneOf", "allOf", "not", "pattern",
                                "multipleOf"}) {
        const std::string schema_text = std::string("{\"") + keyword + "\": []}";
        bool threw = false;
        try {
            JsonConstraint::from_schema(parse_json(schema_text));
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    bool remote_ref_threw = false;
    try {
        JsonConstraint::from_schema(parse_json(R"({"$ref":"https://example.com/schema"})"));
    } catch (const std::runtime_error&) {
        remote_ref_threw = true;
    }
    assert(remote_ref_threw);
}

void test_completion_and_whitespace() {
    JsonConstraint constraint = JsonConstraint::json_object();
    assert(!constraint.is_complete());
    assert(constraint.accept(" { \"a\": 1 } \n"));
    assert(constraint.is_complete());
    assert(constraint.accept(" \t"));
    assert(!constraint.accept("x"));

    constraint.reset();
    assert(!constraint.is_complete());
    assert(constraint.accept("{}"));
    assert(constraint.is_complete());
}

int main() {
    test_basic_object();
    test_reject_invalid_syntax();
    test_nested_values();
    test_literals_and_escapes();
    test_can_accept_is_speculative();
    test_schema_required_and_types();
    test_schema_enum_and_bounds();
    test_schema_additional_properties_and_refs();
    test_unsupported_keywords_throw();
    test_completion_and_whitespace();
    std::cout << "All JSON constraint tests passed\n";
    return 0;
}
