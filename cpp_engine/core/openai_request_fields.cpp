#include "openai_request_fields.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>

#include "openai_stop_strings.hpp"

namespace pocket {

namespace {

bool absent_or_null(const JsonValue* value) {
    return value == nullptr || value->is_null();
}

bool is_default_number(const JsonValue* value, double expected) {
    return absent_or_null(value) ||
           (value->is_number() && std::fabs(value->number() - expected) < 1.0e-9);
}

bool is_default_bool(const JsonValue* value, bool expected) {
    return absent_or_null(value) ||
           (value->is_bool() && value->boolean() == expected);
}

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += "\"";
    return out;
}

std::string number_to_string(double d) {
    char buf[32];
    if (d == std::floor(d) && std::fabs(d) < 1.0e15) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(d));
    } else {
        std::snprintf(buf, sizeof(buf), "%g", d);
    }
    return buf;
}

// The refused value as it appears in the message. An array renders its entries
// when they are all scalars -- `stop` is nearly always a short list of strings,
// and seeing them is the difference between a useful refusal and a guessing
// game -- and falls back to a size summary otherwise, since a logit_bias can
// hold hundreds of entries and the field name is what the caller has to act on.
std::string render_value(const JsonValue& value, std::size_t budget) {
    if (value.is_null()) return "null";
    if (value.is_bool()) return value.boolean() ? "true" : "false";
    if (value.is_number()) return number_to_string(value.number());
    if (value.is_string()) return quote(value.string());
    if (value.is_array()) {
        const JsonArray& items = value.array();
        std::string rendered = "[";
        bool scalar_only = true;
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (!items[i].is_string() && !items[i].is_number() && !items[i].is_bool()) {
                scalar_only = false;
                break;
            }
            if (i > 0) rendered += ", ";
            rendered += render_value(items[i], 0);
            if (rendered.size() > budget) {
                scalar_only = false;
                break;
            }
        }
        if (scalar_only) return rendered + "]";
        std::ostringstream os;
        os << "[" << items.size() << (items.size() == 1 ? " item]" : " items]");
        return os.str();
    }
    std::ostringstream os;
    os << "{" << value.object().size()
       << (value.object().size() == 1 ? " key}" : " keys}");
    return os.str();
}

std::string render_value(const JsonValue& value) {
    return render_value(value, 80);
}

// Builds the refusal: the field name, the value that was asked for, what this
// server does instead, and what the caller can do about it. All four parts are
// present in every message because a caller who cannot tell whether the field
// was dropped, is being ignored, or needs a different spelling has to read the
// source to find out.
RequestFieldCheck refuse(const std::string& field, const JsonValue& requested,
                         const std::string& actual, const std::string& remedy) {
    RequestFieldCheck check;
    check.ok = false;
    check.field = field;
    check.requested = render_value(requested);
    std::ostringstream os;
    os << "\"" << field << "\" = " << check.requested
       << " is not supported by this server: " << actual << " " << remedy;
    check.message = os.str();
    return check;
}

}  // namespace

RequestFieldCheck check_request_fields(const JsonObject& body, OpenAiEndpoint endpoint) {
    const bool chat = endpoint == OpenAiEndpoint::ChatCompletions;

    RequestFieldCheck check;

    // How many completions come back. Every response emitter in the server
    // writes a single-element `choices` array with a hardcoded index 0, so a
    // request for more gets one and no indication that the rest are missing.
    const JsonValue* value = object_get(body, "n");
    if (!is_default_number(value, 1.0)) {
        return refuse("n", *value,
                      "\"choices\" always holds exactly one entry and its index "
                      "is always 0.",
                      "Remove \"n\", or set it to 1 and read the single choice.");
    }

    // Client stop sequences are implemented -- matched over the decoded text,
    // see openai_stop_strings.hpp -- so only a value of the wrong *shape* is a
    // refusal here. Accepting a number or a nested array would turn a client
    // mistake into a request that silently never stops.
    value = object_get(body, "stop");
    if (!absent_or_null(value) && !is_stop_shape(*value)) {
        return refuse("stop", *value,
                      "a stop sequence is a string, or a list of strings, and "
                      "this value is neither.",
                      "Send \"stop\" as a string or an array of strings.");
    }

    // Per-token log probabilities.
    value = object_get(body, "logprobs");
    if (!absent_or_null(value)) {
        // Chat takes a boolean, and false -- the default -- asks for nothing.
        // /v1/completions takes a count, where even 0 asks for the sampled
        // token's logprob, so no value is inert on that endpoint.
        const bool inert = chat && value->is_bool() && !value->boolean();
        if (!inert) {
            return refuse("logprobs", *value,
                          "no \"logprobs\" object is returned on any choice.",
                          chat ? "Remove \"logprobs\", or set it to false."
                               : "Remove \"logprobs\".");
        }
    }
    // A count of alternatives to rank per position. Only 0 -- the default -- asks
    // for nothing; a negative count is outside the documented range rather than a
    // larger default, so it is refused along with everything above zero.
    value = object_get(body, "top_logprobs");
    if (!absent_or_null(value) && !(value->is_number() && value->number() == 0.0)) {
        return refuse("top_logprobs", *value,
                      "there are no per-token logprobs, so there is no set of "
                      "alternatives to rank.",
                      "Remove \"top_logprobs\".");
    }

    // Repetition controls. The sampler has neither term, so a request naming
    // one is generated as if it were 0 -- which can be the opposite of what the
    // caller asked for when the penalty was there to suppress a loop.
    for (const char* field : {"frequency_penalty", "presence_penalty"}) {
        value = object_get(body, field);
        if (!is_default_number(value, 0.0)) {
            return refuse(field, *value,
                          "this sampler has no repetition or presence term, so "
                          "the request is generated as if the penalty were 0.",
                          std::string("Remove \"") + field +
                              "\", or set it to 0.");
        }
    }

    // Per-token bias. Silently dropping it changes the distribution for exactly
    // the tokens the caller cared about most.
    value = object_get(body, "logit_bias");
    if (!absent_or_null(value)) {
        const bool inert = value->is_object() && value->object().empty();
        if (!inert) {
            return refuse("logit_bias", *value,
                          "no per-token bias is applied, so every biased token "
                          "is sampled at its unmodified probability.",
                          "Remove \"logit_bias\".");
        }
    }

    // stream_options.include_usage. Rejected only when it would change the
    // response: a non-streaming completion already carries "usage", which is
    // the whole thing the option asks for, so only a streaming request is
    // actually losing anything.
    const JsonValue* stream_value = object_get(body, "stream");
    const bool streaming = stream_value != nullptr && stream_value->is_bool() &&
                           stream_value->boolean();
    value = object_get(body, "stream_options");
    if (streaming && value != nullptr && value->is_object()) {
        const JsonValue* include_usage = object_get(value->object(), "include_usage");
        if (!is_default_bool(include_usage, false)) {
            return refuse("stream_options.include_usage", *include_usage,
                          "a streaming response is a sequence of delta chunks "
                          "followed by \"[DONE]\", and none of them carries a "
                          "\"usage\" object.",
                          "Remove \"stream_options\", or set \"include_usage\" "
                          "to false; usage is reported on non-streaming "
                          "requests.");
        }
    }

    if (!chat) {
        // Multiple candidates scored by their likelihood. Only reachable on
        // /v1/completions, and rejected at any value above 1.
        value = object_get(body, "best_of");
        if (!is_default_number(value, 1.0)) {
            return refuse("best_of", *value,
                          "one candidate is generated per request, and no "
                          "second candidate is sampled to compare it against.",
                          "Remove \"best_of\", or set it to 1.");
        }

        // Text appended after the completion / the prompt echoed into it.
        value = object_get(body, "suffix");
        if (!absent_or_null(value) &&
            !(value->is_string() && value->string().empty())) {
            return refuse("suffix", *value,
                          "the completion is returned on its own and no suffix "
                          "text is appended after it.",
                          "Remove \"suffix\".");
        }
        value = object_get(body, "echo");
        if (!is_default_bool(value, false)) {
            return refuse("echo", *value,
                          "the response \"text\" holds only the generated "
                          "continuation, never the prompt.",
                          "Remove \"echo\", or set it to false.");
        }
    } else {
        // Tool use. The definitions themselves are forwarded to the chat
        // template, but the selection policy is not applied.
        value = object_get(body, "tool_choice");
        if (!absent_or_null(value) &&
            !(value->is_string() && value->string() == "auto")) {
            return refuse("tool_choice", *value,
                          "tool definitions reach the chat template, but the "
                          "model is not constrained to call a tool, to skip "
                          "them, or to call one particular function, so the "
                          "policy in this field has no effect.",
                          "Remove \"tool_choice\" (the default is \"auto\") and "
                          "decide what to do with the returned \"tool_calls\" "
                          "yourself.");
        }
        value = object_get(body, "parallel_tool_calls");
        if (!is_default_bool(value, true)) {
            return refuse("parallel_tool_calls", *value,
                          "the number of tool calls the model emits is not "
                          "limited.",
                          "Remove \"parallel_tool_calls\", and keep the first "
                          "call yourself if only one is acceptable.");
        }
    }

    // Accepted and deliberately inert, none of which can change the generated
    // text: "user", "store", "metadata", "service_tier", and "model" -- the
    // server serves exactly one model and echoes its configured name back in
    // every response, so a mismatched "model" would be a routing request to a
    // server that has nothing to route to. Rejecting these would break clients
    // over nothing, which is the opposite failure from the one this function
    // exists to prevent.
    return check;
}

int effective_max_tokens(const JsonObject& body, int fallback) {
    int chosen = fallback;
    // Iterated in precedence order: OpenAI deprecated "max_tokens" in favour of
    // "max_completion_tokens" and applies the latter when a request carries
    // both, so reading it second is what makes it win.
    for (const char* field : {"max_tokens", "max_completion_tokens"}) {
        const JsonValue* value = object_get(body, field);
        if (value == nullptr || !value->is_number() || value->number() <= 0.0) continue;
        // Clamped rather than cast: a budget far above INT_MAX is not a real
        // request, and converting it would be undefined behaviour. The engine
        // applies its own context bound on top of whatever comes out here.
        chosen = value->number() >= 2147483647.0 ? 2147483647
                                                 : static_cast<int>(value->number());
    }
    return chosen;
}

}  // namespace pocket
