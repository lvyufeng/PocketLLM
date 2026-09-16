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

// A count field, as opposed to a measure: 2 is one, 2.5 is not. Folding this into
// the range checks below would report 2.5 as out of range, which points the
// caller at the wrong half of the mistake.
bool is_whole_number(const JsonValue* value) {
    return !absent_or_null(value) && value->is_number() &&
           value->number() == std::floor(value->number());
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

    // Read here rather than where "stream_options" needs it, because whether the
    // response is streamed decides whether asking for log probabilities can be
    // served at all, and that question is settled with the logprobs block below.
    const JsonValue* stream_value = object_get(body, "stream");
    const bool streaming = stream_value != nullptr && stream_value->is_bool() &&
                           stream_value->boolean();

    RequestFieldCheck check;

    // How many completions come back. The server submits one scheduler request
    // per choice, so any count the caller names is produced -- but a request for
    // choices beyond kMaxChoices would put that many requests in the queue,
    // which is a way to make one request cost the server an unbounded amount of
    // memory, so the ceiling is refused rather than quietly honoured.
    const JsonValue* value = object_get(body, "n");
    if (!absent_or_null(value)) {
        if (!value->is_number() || value->number() != std::floor(value->number())) {
            return refuse("n", *value,
                          "the number of choices is a whole number and this "
                          "value is not one.",
                          "Send \"n\" as an integer, or omit it for one choice.");
        }
        if (value->number() < 1.0 || value->number() > static_cast<double>(kMaxChoices)) {
            std::ostringstream os;
            os << "this server generates one request per choice and accepts at "
               << "most " << kMaxChoices << " choices in a single request.";
            return refuse("n", *value, os.str(),
                          "Lower \"n\" to " + std::to_string(kMaxChoices) +
                              " or less; a client that needs more can send the "
                              "request again.");
        }
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

    // Per-token log probabilities. Both fields are implemented, so what is left
    // to refuse is a value this server cannot act on.
    //
    // The two endpoints spell the same request differently and the difference is
    // not cosmetic: chat takes `logprobs` as a boolean and puts the number of
    // ranked alternatives in `top_logprobs`, while /v1/completions takes one
    // count in `logprobs` where even 0 includes the sampled token's own
    // probability. A boolean is meaningless on the second and a count on the
    // first, so each is checked against the endpoint that defines it.
    const JsonValue* logprobs_value = object_get(body, "logprobs");
    const JsonValue* top_logprobs_value = object_get(body, "top_logprobs");
    if (chat) {
        if (!absent_or_null(logprobs_value) && !logprobs_value->is_bool()) {
            return refuse("logprobs", *logprobs_value,
                          "the chat endpoint takes a boolean: true asks for the "
                          "sampled token's own log probability, plus any "
                          "alternatives named in \"top_logprobs\".",
                          "Send \"logprobs\" as true or false, or omit it.");
        }
    } else if (!absent_or_null(logprobs_value)) {
        // The count is what bounds the work: every alternative is ranked per
        // generated position, so an unbounded count would ask the sampler for a
        // ranking wider than it keeps and the response for a list longer than a
        // caller can use.
        if (!is_whole_number(logprobs_value) || logprobs_value->number() < 0.0 ||
            logprobs_value->number() > static_cast<double>(kMaxLogprobAlternatives)) {
            std::ostringstream os;
            os << "the number of alternatives to rank per position is a whole "
               << "number from 0 to " << kMaxLogprobAlternatives << ".";
            return refuse("logprobs", *logprobs_value, os.str(),
                          "Lower \"logprobs\" to " +
                              std::to_string(kMaxLogprobAlternatives) +
                              " or less; 0 reports the sampled token's own "
                              "probability and no alternatives.");
        }
    }
    if (chat && !absent_or_null(top_logprobs_value)) {
        if (!is_whole_number(top_logprobs_value) ||
            top_logprobs_value->number() < 0.0 ||
            top_logprobs_value->number() >
                static_cast<double>(kMaxLogprobAlternatives)) {
            std::ostringstream os;
            os << "the number of alternatives to rank per position is a whole "
               << "number from 0 to " << kMaxLogprobAlternatives << ".";
            return refuse("top_logprobs", *top_logprobs_value, os.str(),
                          "Lower \"top_logprobs\" to " +
                              std::to_string(kMaxLogprobAlternatives) +
                              " or less; 0 reports the sampled token's own "
                              "probability and no alternatives.");
        }
        // Alternatives exist only for a request that asked for log
        // probabilities, so naming a count is a request for something this one
        // never turns on -- and answering it with no "logprobs" object at all
        // reads as a server that ignored the field. A count of 0 asks for no
        // alternatives, which is what this request produces anyway, so it is
        // inert on its own and accepted.
        if (top_logprobs_value->number() > 0.0 &&
            !(logprobs_value != nullptr && logprobs_value->is_bool() &&
              logprobs_value->boolean())) {
            return refuse("top_logprobs", *top_logprobs_value,
                          "alternatives are reported only for a request that "
                          "asks for log probabilities, and this one's "
                          "\"logprobs\" is absent or false.",
                          "Set \"logprobs\" to true, or remove \"top_logprobs\".");
        }
    } else if (!chat && !absent_or_null(top_logprobs_value)) {
        return refuse("top_logprobs", *top_logprobs_value,
                      "the completions endpoint names the number of "
                      "alternatives in \"logprobs\" itself.",
                      "Put the count in \"logprobs\" and remove "
                      "\"top_logprobs\".");
    }

    // Log probabilities are reported on a non-streaming response only. A chunk
    // carries the text of the token it delivers, so a ranking for that token
    // would have to travel beside it -- a different wire shape from the one this
    // server streams. Answering anyway would produce a stream that is
    // indistinguishable from one whose request asked for no ranking at all,
    // which is the failure this whole function exists to prevent.
    //
    // What counts as asking differs by endpoint: false is off on chat, while
    // /v1/completions spells the same thing as the count 0, where 0 is already a
    // real request for the sampled token's own probability. The block above
    // validated both spellings, so this reads them rather than re-deriving them.
    if (streaming) {
        const bool asked = chat
            ? (logprobs_value != nullptr && logprobs_value->is_bool() &&
               logprobs_value->boolean())
            : is_whole_number(logprobs_value);
        if (asked) {
            return refuse("logprobs", *logprobs_value,
                          "log probabilities are reported on a non-streaming "
                          "response, and a streamed chunk carries the text of "
                          "its token with no ranking beside it.",
                          "Remove \"stream\", or remove \"logprobs\".");
        }
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

int requested_choices(const JsonObject& body) {
    const JsonValue* value = object_get(body, "n");
    if (value == nullptr || !value->is_number()) return 1;
    const double requested = value->number();
    // The same test the audit applies, so a caller that audited first gets its
    // own value back and a caller that did not gets a request that generates one
    // choice rather than a truncated count.
    if (requested != std::floor(requested) || requested < 1.0 ||
        requested > static_cast<double>(kMaxChoices)) {
        return 1;
    }
    return static_cast<int>(requested);
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
