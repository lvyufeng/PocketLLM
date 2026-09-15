#pragma once

#include "json_lite.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace pocket {

// Incremental JSON validator used by structured generation. The validator accepts
// valid prefixes, so it can be queried for every vocabulary piece before sampling.
// A speculative query never changes the state; accept() commits only when the
// complete piece is valid.
class JsonConstraint {
public:
    // Accept any JSON object, without imposing a property schema.
    static JsonConstraint json_object();

    // Accept values described by a JSON Schema. Unsupported keywords are rejected
    // at construction time instead of being silently ignored.
    static JsonConstraint from_schema(const JsonValue& schema);

    // Return whether appending text would preserve a valid JSON prefix.
    bool can_accept(std::string_view text) const;

    // Append text atomically. On failure the validator state is unchanged.
    bool accept(std::string_view text);

    // True only after the top-level value has been closed and schema checks pass.
    bool is_complete() const;

    void reset();

private:
    enum class ContextKind { Object, Array };
    enum class ContextState {
        ObjectExpectKeyOrEnd,
        ObjectExpectColon,
        ObjectExpectValue,
        ObjectAfterValue,
        ArrayExpectValueOrEnd,
        ArrayAfterValue,
    };

    struct Frame {
        ContextKind kind;
        ContextState state;
        std::shared_ptr<const JsonValue> schema;
        std::unordered_set<std::string> keys_seen;
        std::string pending_key;
        int item_count = 0;
        // Distinguishes an empty container from a container immediately after a
        // comma, where a closing delimiter would be a trailing-comma error.
        bool allow_end = true;

        Frame(ContextKind kind_in, std::shared_ptr<const JsonValue> schema_in,
              ContextState state_in)
            : kind(kind_in), state(state_in), schema(std::move(schema_in)) {}
    };

    enum class ActiveKind { None, String, Number, Literal };
    enum class NumberState {
        Minus,
        Zero,
        Integer,
        Dot,
        Fraction,
        Exponent,
        ExponentSign,
        ExponentDigits,
    };

    struct Active {
        ActiveKind kind = ActiveKind::None;
        bool is_key = false;
        std::shared_ptr<const JsonValue> schema;
        std::string text;
        std::string decoded;
        std::string literal_target;
        size_t literal_pos = 0;
        NumberState number_state = NumberState::Zero;
        int unicode_digits = 0;
        uint32_t unicode_value = 0;
        bool escape = false;
    };

    std::vector<Frame> stack_;
    std::shared_ptr<const JsonValue> root_schema_;
    bool root_object_only_ = false;
    bool root_started_ = false;
    bool complete_ = false;
    Active active_;

    JsonConstraint() = default;

    bool accept_impl(std::string_view text);
    bool feed_char(char c);
    bool start_value(char c, std::shared_ptr<const JsonValue> schema);
    bool start_string(std::shared_ptr<const JsonValue> schema, bool is_key);
    bool start_number(char first, std::shared_ptr<const JsonValue> schema);
    bool start_literal(char first, std::shared_ptr<const JsonValue> schema);
    bool finish_active_value();
    bool finish_number();
    bool close_object();
    bool close_array();
    bool complete_container_value(std::shared_ptr<const JsonValue> schema);

    std::shared_ptr<const JsonValue> normalize_schema(
        std::shared_ptr<const JsonValue> schema) const;
    std::shared_ptr<const JsonValue> schema_for_object_value(const Frame& frame,
                                                              const std::string& key) const;
    bool object_value_allowed(const Frame& frame, const std::string& key) const;
    std::shared_ptr<const JsonValue> schema_for_array_item(const Frame& frame) const;
    bool schema_allows_start(const JsonValue& schema, char c) const;
    bool validate_scalar(const Active& value) const;
    bool validate_schema_keywords(const JsonValue& schema,
                                  const Active& value) const;
    bool type_matches(const JsonValue& schema, ActiveKind kind,
                      const std::string& literal_target,
                      const std::string& number_text) const;
    void check_unsupported_keywords(const JsonValue& schema) const;
    const JsonValue* resolve_ref(const std::string& ref) const;

    static bool is_whitespace(char c);
    static bool is_delimiter(char c);
    static bool is_hex_digit(char c);
    static int hex_value(char c);
    static bool is_digit(char c);
    static bool number_state_terminal(NumberState state);
    static void append_utf8(std::string& out, uint32_t codepoint);
};

}  // namespace pocket
