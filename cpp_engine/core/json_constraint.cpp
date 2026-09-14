#include "json_constraint.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pocket {

namespace {

bool value_is_scalar(const JsonValue& value) {
    return value.is_null() || value.is_bool() || value.is_number() ||
           value.is_string();
}

bool number_from_text(const std::string& text, double* value) {
    if (text.empty()) return false;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    *value = parsed;
    return true;
}

}  // namespace

bool JsonConstraint::is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool JsonConstraint::is_delimiter(char c) {
    return is_whitespace(c) || c == ',' || c == '}' || c == ']';
}

bool JsonConstraint::is_hex_digit(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

int JsonConstraint::hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool JsonConstraint::is_digit(char c) {
    return c >= '0' && c <= '9';
}

bool JsonConstraint::number_state_terminal(NumberState state) {
    return state == NumberState::Zero || state == NumberState::Integer ||
           state == NumberState::Fraction || state == NumberState::ExponentDigits;
}

void JsonConstraint::append_utf8(std::string& out, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

JsonConstraint JsonConstraint::json_object() {
    JsonConstraint result;
    result.root_object_only_ = true;
    return result;
}

JsonConstraint JsonConstraint::from_schema(const JsonValue& schema) {
    if (!schema.is_object()) {
        throw std::runtime_error("JSON Schema root must be an object");
    }
    JsonConstraint result;
    result.root_schema_ = std::make_shared<JsonValue>(schema);
    result.check_unsupported_keywords(schema);
    return result;
}

bool JsonConstraint::can_accept(std::string_view text) const {
    JsonConstraint copy = *this;
    return copy.accept_impl(text);
}

bool JsonConstraint::accept(std::string_view text) {
    JsonConstraint copy = *this;
    if (!copy.accept_impl(text)) return false;
    *this = std::move(copy);
    return true;
}

bool JsonConstraint::accept_impl(std::string_view text) {
    for (char c : text) {
        if (!feed_char(c)) return false;
    }
    return true;
}

bool JsonConstraint::feed_char(char c) {
    if (complete_) {
        return is_whitespace(c);
    }

    if (active_.kind != ActiveKind::None) {
        if (active_.kind == ActiveKind::String) {
            if (active_.unicode_digits > 0) {
                if (!is_hex_digit(c)) return false;
                active_.unicode_value = (active_.unicode_value << 4) |
                    static_cast<uint32_t>(hex_value(c));
                --active_.unicode_digits;
                if (active_.unicode_digits == 0) {
                    // Reject surrogate halves rather than emitting invalid UTF-8.
                    if (active_.unicode_value >= 0xd800 &&
                        active_.unicode_value <= 0xdfff) {
                        return false;
                    }
                    append_utf8(active_.decoded, active_.unicode_value);
                    active_.unicode_value = 0;
                }
                return true;
            }
            if (active_.escape) {
                active_.escape = false;
                switch (c) {
                    case '"': active_.decoded.push_back('"'); return true;
                    case '\\': active_.decoded.push_back('\\'); return true;
                    case '/': active_.decoded.push_back('/'); return true;
                    case 'b': active_.decoded.push_back('\b'); return true;
                    case 'f': active_.decoded.push_back('\f'); return true;
                    case 'n': active_.decoded.push_back('\n'); return true;
                    case 'r': active_.decoded.push_back('\r'); return true;
                    case 't': active_.decoded.push_back('\t'); return true;
                    case 'u':
                        active_.unicode_digits = 4;
                        active_.unicode_value = 0;
                        return true;
                    default: return false;
                }
            }
            if (c == '"') return finish_active_value();
            if (c == '\\') {
                active_.escape = true;
                return true;
            }
            if (static_cast<unsigned char>(c) < 0x20) return false;
            active_.decoded.push_back(c);
            return true;
        }

        if (active_.kind == ActiveKind::Literal) {
            if (active_.literal_pos >= active_.literal_target.size()) {
                return feed_char(c);
            }
            if (c != active_.literal_target[active_.literal_pos]) return false;
            ++active_.literal_pos;
            if (active_.literal_pos == active_.literal_target.size()) {
                return finish_active_value();
            }
            return true;
        }

        // Number parsing is the only scalar that cannot be closed until its
        // delimiter is seen. A delimiter is then fed to the parent state.
        if (active_.number_state == NumberState::Minus) {
            if (c == '0') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Zero;
                return true;
            }
            if (c >= '1' && c <= '9') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Integer;
                return true;
            }
            return false;
        }
        if (active_.number_state == NumberState::Zero) {
            if (c == '.') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Dot;
                return true;
            }
            if (c == 'e' || c == 'E') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Exponent;
                return true;
            }
            if (is_delimiter(c)) {
                if (!finish_number()) return false;
                return feed_char(c);
            }
            return false;
        }
        if (active_.number_state == NumberState::Integer) {
            if (is_digit(c)) {
                active_.text.push_back(c);
                return true;
            }
            if (c == '.') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Dot;
                return true;
            }
            if (c == 'e' || c == 'E') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Exponent;
                return true;
            }
            if (is_delimiter(c)) {
                if (!finish_number()) return false;
                return feed_char(c);
            }
            return false;
        }
        if (active_.number_state == NumberState::Dot) {
            if (!is_digit(c)) return false;
            active_.text.push_back(c);
            active_.number_state = NumberState::Fraction;
            return true;
        }
        if (active_.number_state == NumberState::Fraction) {
            if (is_digit(c)) {
                active_.text.push_back(c);
                return true;
            }
            if (c == 'e' || c == 'E') {
                active_.text.push_back(c);
                active_.number_state = NumberState::Exponent;
                return true;
            }
            if (is_delimiter(c)) {
                if (!finish_number()) return false;
                return feed_char(c);
            }
            return false;
        }
        if (active_.number_state == NumberState::Exponent) {
            if (c == '+' || c == '-') {
                active_.text.push_back(c);
                active_.number_state = NumberState::ExponentSign;
                return true;
            }
            if (is_digit(c)) {
                active_.text.push_back(c);
                active_.number_state = NumberState::ExponentDigits;
                return true;
            }
            return false;
        }
        if (active_.number_state == NumberState::ExponentSign) {
            if (!is_digit(c)) return false;
            active_.text.push_back(c);
            active_.number_state = NumberState::ExponentDigits;
            return true;
        }
        if (active_.number_state == NumberState::ExponentDigits) {
            if (is_digit(c)) {
                active_.text.push_back(c);
                return true;
            }
            if (is_delimiter(c)) {
                if (!finish_number()) return false;
                return feed_char(c);
            }
            return false;
        }
        return false;
    }

    if (stack_.empty()) {
        if (!root_started_) {
            if (is_whitespace(c)) return true;
            return start_value(c, root_schema_);
        }
        // A root container is closed only through close_object/close_array, and
        // a root scalar is completed by finish_active_value.
        return false;
    }

    Frame& frame = stack_.back();
    if (frame.kind == ContextKind::Object) {
        switch (frame.state) {
            case ContextState::ObjectExpectKeyOrEnd:
                if (is_whitespace(c)) return true;
                if (c == '"') return start_string(nullptr, true);
                if (c == '}') return frame.allow_end && close_object();
                return false;
            case ContextState::ObjectExpectColon:
                if (is_whitespace(c)) return true;
                if (c != ':') return false;
                frame.state = ContextState::ObjectExpectValue;
                return true;
            case ContextState::ObjectExpectValue:
                if (is_whitespace(c)) return true;
                if (!object_value_allowed(frame, frame.pending_key)) return false;
                return start_value(c, schema_for_object_value(
                    frame, frame.pending_key));
            case ContextState::ObjectAfterValue:
                if (is_whitespace(c)) return true;
                if (c == ',') {
                    frame.state = ContextState::ObjectExpectKeyOrEnd;
                    frame.allow_end = false;
                    return true;
                }
                if (c == '}') return close_object();
                return false;
            case ContextState::ArrayExpectValueOrEnd:
            case ContextState::ArrayAfterValue:
                return false;
        }
    } else {
        switch (frame.state) {
            case ContextState::ArrayExpectValueOrEnd:
                if (is_whitespace(c)) return true;
                if (c == ']') return frame.allow_end && close_array();
                return start_value(c, schema_for_array_item(frame));
            case ContextState::ArrayAfterValue:
                if (is_whitespace(c)) return true;
                if (c == ',') {
                    frame.state = ContextState::ArrayExpectValueOrEnd;
                    frame.allow_end = false;
                    return true;
                }
                if (c == ']') return close_array();
                return false;
            case ContextState::ObjectExpectKeyOrEnd:
            case ContextState::ObjectExpectColon:
            case ContextState::ObjectExpectValue:
            case ContextState::ObjectAfterValue:
                return false;
        }
    }
    return false;
}

bool JsonConstraint::start_value(
    char c, std::shared_ptr<const JsonValue> schema) {
    schema = normalize_schema(std::move(schema));
    if (schema && !schema_allows_start(*schema, c)) return false;

    const bool is_root = stack_.empty() && !root_started_;
    if (is_root) {
        if (root_object_only_ && c != '{') return false;
        root_started_ = true;
    }

    if (c == '{') {
        stack_.emplace_back(ContextKind::Object, schema,
                            ContextState::ObjectExpectKeyOrEnd);
        return true;
    }
    if (c == '[') {
        stack_.emplace_back(ContextKind::Array, schema,
                            ContextState::ArrayExpectValueOrEnd);
        return true;
    }
    if (c == '"') return start_string(std::move(schema), false);
    if (c == '-' || (c >= '0' && c <= '9')) {
        return start_number(c, std::move(schema));
    }
    if (c == 't' || c == 'f' || c == 'n') {
        return start_literal(c, std::move(schema));
    }
    return false;
}

bool JsonConstraint::start_string(std::shared_ptr<const JsonValue> schema,
                                  bool is_key) {
    active_ = Active{};
    active_.kind = ActiveKind::String;
    active_.is_key = is_key;
    active_.schema = normalize_schema(std::move(schema));
    return true;
}

bool JsonConstraint::start_number(char first,
                                  std::shared_ptr<const JsonValue> schema) {
    active_ = Active{};
    active_.kind = ActiveKind::Number;
    active_.schema = normalize_schema(std::move(schema));
    active_.text.push_back(first);
    if (first == '-') {
        active_.number_state = NumberState::Minus;
    } else if (first == '0') {
        active_.number_state = NumberState::Zero;
    } else {
        active_.number_state = NumberState::Integer;
    }
    return true;
}

bool JsonConstraint::start_literal(char first,
                                   std::shared_ptr<const JsonValue> schema) {
    active_ = Active{};
    active_.kind = ActiveKind::Literal;
    active_.schema = normalize_schema(std::move(schema));
    active_.literal_target = first == 't' ? "true" :
                             first == 'f' ? "false" : "null";
    active_.literal_pos = 1;
    return true;
}

bool JsonConstraint::finish_number() {
    if (active_.kind != ActiveKind::Number ||
        !number_state_terminal(active_.number_state)) {
        return false;
    }
    return finish_active_value();
}

bool JsonConstraint::finish_active_value() {
    if (active_.kind == ActiveKind::None) return false;
    const Active value = active_;

    if (value.is_key) {
        if (stack_.empty() || stack_.back().kind != ContextKind::Object ||
            stack_.back().state != ContextState::ObjectExpectKeyOrEnd) {
            return false;
        }
        Frame& object = stack_.back();
        object.pending_key = value.decoded;
        object.keys_seen.insert(value.decoded);
        object.state = ContextState::ObjectExpectColon;
        active_ = Active{};
        return true;
    }

    if (!validate_scalar(value)) return false;
    active_ = Active{};
    if (stack_.empty()) {
        complete_ = true;
        return true;
    }
    return complete_container_value(value.schema);
}

bool JsonConstraint::close_object() {
    if (stack_.empty() || stack_.back().kind != ContextKind::Object) {
        return false;
    }
    Frame frame = stack_.back();
    if (frame.state == ContextState::ObjectExpectValue ||
        frame.state == ContextState::ObjectExpectColon ||
        (frame.state == ContextState::ObjectExpectKeyOrEnd && !frame.allow_end)) {
        return false;
    }
    if (frame.state != ContextState::ObjectExpectKeyOrEnd &&
        frame.state != ContextState::ObjectAfterValue) {
        return false;
    }

    const auto schema = normalize_schema(frame.schema);
    if (schema && schema->is_object()) {
        if (const JsonValue* required = object_get(schema->object(), "required")) {
            if (!required->is_array()) return false;
            for (const JsonValue& item : required->array()) {
                if (!item.is_string() ||
                    frame.keys_seen.find(item.string()) == frame.keys_seen.end()) {
                    return false;
                }
            }
        }
    }

    stack_.pop_back();
    return complete_container_value(schema);
}

bool JsonConstraint::close_array() {
    if (stack_.empty() || stack_.back().kind != ContextKind::Array) {
        return false;
    }
    Frame frame = stack_.back();
    if (frame.state == ContextState::ArrayExpectValueOrEnd && !frame.allow_end) {
        return false;
    }
    if (frame.state != ContextState::ArrayExpectValueOrEnd &&
        frame.state != ContextState::ArrayAfterValue) {
        return false;
    }

    const auto schema = normalize_schema(frame.schema);
    if (schema && schema->is_object()) {
        if (const JsonValue* min_items = object_get(schema->object(), "minItems")) {
            if (!min_items->is_number() ||
                frame.item_count < static_cast<int>(min_items->number())) {
                return false;
            }
        }
        if (const JsonValue* max_items = object_get(schema->object(), "maxItems")) {
            if (!max_items->is_number() ||
                frame.item_count > static_cast<int>(max_items->number())) {
                return false;
            }
        }
    }

    stack_.pop_back();
    return complete_container_value(schema);
}

bool JsonConstraint::complete_container_value(
    std::shared_ptr<const JsonValue> /*schema*/) {
    if (stack_.empty()) {
        complete_ = true;
        return true;
    }
    Frame& parent = stack_.back();
    if (parent.kind == ContextKind::Object) {
        if (parent.state != ContextState::ObjectExpectValue) return false;
        parent.state = ContextState::ObjectAfterValue;
        return true;
    }
    if (parent.state != ContextState::ArrayExpectValueOrEnd) return false;
    ++parent.item_count;
    parent.state = ContextState::ArrayAfterValue;
    return true;
}

std::shared_ptr<const JsonValue> JsonConstraint::normalize_schema(
    std::shared_ptr<const JsonValue> schema) const {
    for (int depth = 0; schema && depth < 32; ++depth) {
        if (!schema->is_object()) return schema;
        const JsonValue* ref = object_get(schema->object(), "$ref");
        if (!ref) return schema;
        if (!ref->is_string()) return nullptr;
        const JsonValue* resolved = resolve_ref(ref->string());
        if (!resolved) {
            throw std::runtime_error("unresolved local JSON Schema reference: " +
                                     ref->string());
        }
        schema = std::make_shared<JsonValue>(*resolved);
    }
    if (schema) throw std::runtime_error("too many nested JSON Schema $ref entries");
    return schema;
}

bool JsonConstraint::object_value_allowed(const Frame& frame,
                                          const std::string& key) const {
    const auto schema = normalize_schema(frame.schema);
    if (!schema || !schema->is_object()) return true;
    const JsonObject& object = schema->object();
    if (const JsonValue* properties = object_get(object, "properties")) {
        if (!properties->is_object()) return false;
        if (object_get(properties->object(), key) != nullptr) return true;
    }
    if (const JsonValue* additional = object_get(object, "additionalProperties")) {
        return !additional->is_bool() || additional->boolean() ||
               additional->is_object();
    }
    return true;
}

std::shared_ptr<const JsonValue> JsonConstraint::schema_for_object_value(
    const Frame& frame, const std::string& key) const {
    const auto schema = normalize_schema(frame.schema);
    if (!schema || !schema->is_object()) return nullptr;
    const JsonObject& object = schema->object();
    if (const JsonValue* properties = object_get(object, "properties")) {
        if (!properties->is_object()) return nullptr;
        if (const JsonValue* property = object_get(properties->object(), key)) {
            return normalize_schema(std::make_shared<JsonValue>(*property));
        }
    }
    if (const JsonValue* additional = object_get(object, "additionalProperties")) {
        if (additional->is_bool() && !additional->boolean()) return nullptr;
        if (additional->is_object()) {
            return normalize_schema(std::make_shared<JsonValue>(*additional));
        }
    }
    return nullptr;
}

std::shared_ptr<const JsonValue> JsonConstraint::schema_for_array_item(
    const Frame& frame) const {
    const auto schema = normalize_schema(frame.schema);
    if (!schema || !schema->is_object()) return nullptr;
    if (const JsonValue* items = object_get(schema->object(), "items")) {
        if (items->is_object()) {
            return normalize_schema(std::make_shared<JsonValue>(*items));
        }
    }
    return nullptr;
}

bool JsonConstraint::schema_allows_start(const JsonValue& schema, char c) const {
    if (!schema.is_object()) return true;
    const JsonValue* type = object_get(schema.object(), "type");
    if (!type) return true;

    auto matches = [c](const std::string& name) {
        if (name == "object") return c == '{';
        if (name == "array") return c == '[';
        if (name == "string") return c == '"';
        if (name == "number" || name == "integer") {
            return c == '-' || (c >= '0' && c <= '9');
        }
        if (name == "boolean") return c == 't' || c == 'f';
        if (name == "null") return c == 'n';
        return false;
    };
    if (type->is_string()) return matches(type->string());
    if (type->is_array()) {
        for (const JsonValue& value : type->array()) {
            if (value.is_string() && matches(value.string())) return true;
        }
        return false;
    }
    return false;
}

bool JsonConstraint::type_matches(
    const JsonValue& schema, ActiveKind kind,
    const std::string& literal_target,
    const std::string& number_text) const {
    const JsonValue* type = object_get(schema.object(), "type");
    if (!type) return true;
    auto matches = [&](const std::string& name) {
        if (name == "string") return kind == ActiveKind::String;
        if (name == "number") return kind == ActiveKind::Number;
        if (name == "integer") {
            if (kind != ActiveKind::Number) return false;
            double value = 0.0;
            return number_from_text(number_text, &value) &&
                   value == std::floor(value);
        }
        if (name == "boolean") {
            return kind == ActiveKind::Literal && literal_target != "null";
        }
        if (name == "null") {
            return kind == ActiveKind::Literal && literal_target == "null";
        }
        return false;
    };
    if (type->is_string()) return matches(type->string());
    if (type->is_array()) {
        for (const JsonValue& value : type->array()) {
            if (value.is_string() && matches(value.string())) return true;
        }
        return false;
    }
    return false;
}

bool JsonConstraint::validate_schema_keywords(
    const JsonValue& schema, const Active& value) const {
    if (!type_matches(schema, value.kind, value.literal_target, value.text)) return false;

    auto matches_literal = [&](const JsonValue& candidate) {
        if (value.kind == ActiveKind::String) {
            return candidate.is_string() && candidate.string() == value.decoded;
        }
        if (value.kind == ActiveKind::Number) {
            double actual = 0.0;
            return candidate.is_number() && number_from_text(value.text, &actual) &&
                   std::fabs(actual - candidate.number()) <=
                       std::numeric_limits<double>::epsilon() *
                           std::max(1.0, std::fabs(actual));
        }
        if (value.literal_target == "null") return candidate.is_null();
        return candidate.is_bool() &&
               ((value.literal_target == "true") == candidate.boolean());
    };

    if (const JsonValue* enum_values = object_get(schema.object(), "enum")) {
        if (!enum_values->is_array()) return false;
        bool found = false;
        for (const JsonValue& candidate : enum_values->array()) {
            if (matches_literal(candidate)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    if (const JsonValue* constant = object_get(schema.object(), "const")) {
        if (!matches_literal(*constant)) return false;
    }

    if (value.kind == ActiveKind::String) {
        if (const JsonValue* min_length = object_get(schema.object(), "minLength")) {
            if (!min_length->is_number() ||
                value.decoded.size() < static_cast<size_t>(min_length->number())) {
                return false;
            }
        }
        if (const JsonValue* max_length = object_get(schema.object(), "maxLength")) {
            if (!max_length->is_number() ||
                value.decoded.size() > static_cast<size_t>(max_length->number())) {
                return false;
            }
        }
    }
    if (value.kind == ActiveKind::Number) {
        double number = 0.0;
        if (!number_from_text(value.text, &number)) return false;
        if (const JsonValue* minimum = object_get(schema.object(), "minimum")) {
            if (!minimum->is_number() || number < minimum->number()) return false;
        }
        if (const JsonValue* maximum = object_get(schema.object(), "maximum")) {
            if (!maximum->is_number() || number > maximum->number()) return false;
        }
    }
    return true;
}

bool JsonConstraint::validate_scalar(const Active& value) const {
    if (!value.schema) return true;
    if (!value.schema->is_object()) return false;
    return validate_schema_keywords(*value.schema, value);
}

void JsonConstraint::check_unsupported_keywords(const JsonValue& schema) const {
    if (!schema.is_object()) {
        throw std::runtime_error("JSON Schema entries must be objects");
    }
    const JsonObject& object = schema.object();
    static constexpr const char* kUnsupported[] = {
        "anyOf", "oneOf", "allOf", "not", "pattern", "multipleOf",
    };
    for (const char* keyword : kUnsupported) {
        if (object_get(object, keyword) != nullptr) {
            throw std::runtime_error(std::string("JSON Schema keyword '") +
                                     keyword + "' is not supported");
        }
    }

    if (const JsonValue* ref = object_get(object, "$ref")) {
        if (!ref->is_string() || ref->string().compare(0, 8, "#/$defs/") != 0) {
            throw std::runtime_error("only local #/$defs/ JSON Schema references are supported");
        }
        if (resolve_ref(ref->string()) == nullptr) {
            throw std::runtime_error("unresolved local JSON Schema reference: " +
                                     ref->string());
        }
    }
    if (const JsonValue* enum_values = object_get(object, "enum")) {
        if (!enum_values->is_array()) {
            throw std::runtime_error("JSON Schema enum must be an array");
        }
        for (const JsonValue& value : enum_values->array()) {
            if (!value_is_scalar(value)) {
                throw std::runtime_error(
                    "object and array values in JSON Schema enum are not supported");
            }
        }
    }
    if (const JsonValue* constant = object_get(object, "const")) {
        if (!value_is_scalar(*constant)) {
            throw std::runtime_error(
                "object and array values in JSON Schema const are not supported");
        }
    }

    if (const JsonValue* properties = object_get(object, "properties")) {
        if (!properties->is_object()) {
            throw std::runtime_error("JSON Schema properties must be an object");
        }
        for (const auto& [name, property] : properties->object()) {
            check_unsupported_keywords(property);
        }
    }
    if (const JsonValue* items = object_get(object, "items")) {
        check_unsupported_keywords(*items);
    }
    if (const JsonValue* defs = object_get(object, "$defs")) {
        if (!defs->is_object()) {
            throw std::runtime_error("JSON Schema $defs must be an object");
        }
        for (const auto& [name, definition] : defs->object()) {
            check_unsupported_keywords(definition);
        }
    }
}

const JsonValue* JsonConstraint::resolve_ref(const std::string& ref) const {
    if (!root_schema_ || ref.compare(0, 8, "#/$defs/") != 0) return nullptr;
    const std::string name = ref.substr(8);
    if (name.empty() || name.find('/') != std::string::npos ||
        !root_schema_->is_object()) {
        return nullptr;
    }
    const JsonValue* defs = object_get(root_schema_->object(), "$defs");
    if (!defs || !defs->is_object()) return nullptr;
    return object_get(defs->object(), name);
}

bool JsonConstraint::is_complete() const {
    if (complete_) return true;
    // A root number has no closing delimiter. It is complete when its lexical
    // state is terminal; enclosing objects/arrays still require their delimiter.
    return root_started_ && stack_.empty() && active_.kind == ActiveKind::Number &&
           number_state_terminal(active_.number_state);
}

void JsonConstraint::reset() {
    stack_.clear();
    active_ = Active{};
    root_started_ = false;
    complete_ = false;
}

}  // namespace pocket
