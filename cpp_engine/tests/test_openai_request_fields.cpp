// Unit tests for the request-field audit that the native OpenAI server runs
// before it accepts a body.
//
// The audit is pure JSON inspection, so this test links pocket_core, needs no
// checkpoint or device, and runs in under a second. That is the point: the
// server's field contract is checked on every build rather than only in the
// acceptance runs that need four GPUs and a 27B checkpoint.
//
// Plain `assert` is deliberately not used. Release builds define NDEBUG, which
// would compile every assertion in this file out and report success without
// running anything.

#include "openai_request_fields.hpp"

#include <iostream>
#include <string>

using namespace pocket;

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    std::cout << "FAIL line " << line << ": " << expression << "\n";
    ++g_failures;
}

#define CHECK(condition) check((condition), #condition, __LINE__)

// A refusal has to do three things at once: report failure, name the field in
// the machine-readable slot, and mention both the field and the refused value
// in the message a human reads.
void check_refused(const RequestFieldCheck& result, const std::string& field, int line) {
    if (result.ok) {
        std::cout << "FAIL line " << line << ": expected a refusal of \"" << field
                  << "\" but the request was accepted\n";
        ++g_failures;
        return;
    }
    if (result.field != field) {
        std::cout << "FAIL line " << line << ": refused \"" << result.field
                  << "\" instead of \"" << field << "\"\n";
        ++g_failures;
    }
    if (result.message.find(field) == std::string::npos) {
        std::cout << "FAIL line " << line << ": message does not name \"" << field
                  << "\": " << result.message << "\n";
        ++g_failures;
    }
    if (result.requested.empty()) {
        std::cout << "FAIL line " << line << ": refusal of \"" << field
                  << "\" does not record the requested value\n";
        ++g_failures;
    }
    if (result.message.size() < 40) {
        std::cout << "FAIL line " << line << ": message for \"" << field
                  << "\" explains nothing: " << result.message << "\n";
        ++g_failures;
    }
}

#define REFUSED(call, field) check_refused((call), (field), __LINE__)

JsonObject object_of(const std::string& json) { return parse_json(json).object(); }

RequestFieldCheck audit_chat(const std::string& json) {
    return check_request_fields(object_of(json), OpenAiEndpoint::ChatCompletions);
}

RequestFieldCheck audit_completions(const std::string& json) {
    return check_request_fields(object_of(json), OpenAiEndpoint::Completions);
}

// Every field a client may send at its documented default has to be accepted,
// including the ones this server ignores on purpose: a body that asks for
// nothing unusual must not be punished for naming it.
void test_accepts_defaults() {
    CHECK(audit_chat(
        R"({"model":"local","messages":[],"temperature":0.0,"top_p":1.0,"top_k":20,)"
        R"("seed":7,"max_tokens":64,"stream":false,"n":1,"stop":[],)"
        R"("frequency_penalty":0,"presence_penalty":0,"logprobs":false,)"
        R"("top_logprobs":0,"logit_bias":{},"user":"u","store":false,)"
        R"("metadata":{},"service_tier":"auto","tool_choice":"auto",)"
        R"("parallel_tool_calls":true})")
              .ok);

    CHECK(audit_completions(
        R"({"model":"local","prompt":"hi","max_tokens":64,"n":1,"best_of":1,)"
        R"("stop":null,"logprobs":null,"echo":false,"suffix":"",)"
        R"("frequency_penalty":0.0,"presence_penalty":0.0})")
              .ok);

    // Explicit nulls are how several SDKs spell "not set".
    CHECK(audit_chat(
        R"({"n":null,"stop":null,"logprobs":null,"top_logprobs":null,)"
        R"("logit_bias":null,"tool_choice":null,"parallel_tool_calls":null})")
              .ok);
    CHECK(audit_completions(R"({"best_of":null,"echo":null,"suffix":null})").ok);
}

// `n` moved from refused to implemented: the server runs one scheduler request
// per choice and answers with one entry per choice, so what is left to refuse is
// a count it cannot serve -- zero, a fraction, a non-number, or a count past the
// request ceiling.
void test_choices() {
    CHECK(audit_chat(R"({"n":1})").ok);
    CHECK(audit_chat(R"({"n":2})").ok);
    CHECK(audit_chat(R"({"n":3})").ok);
    CHECK(audit_chat(R"({"n":128})").ok);
    // A count written as a float is still a whole number, and SDKs send it that
    // way when the field came from a float-typed variable.
    CHECK(audit_chat(R"({"n":2.0})").ok);
    CHECK(audit_chat(R"({"n":null})").ok);
    CHECK(audit_completions(R"({"n":2})").ok);

    REFUSED(audit_chat(R"({"n":0})"), "n");
    REFUSED(audit_chat(R"({"n":-1})"), "n");
    REFUSED(audit_chat(R"({"n":1.5})"), "n");
    REFUSED(audit_chat(R"({"n":"3"})"), "n");
    REFUSED(audit_chat(R"({"n":true})"), "n");
    REFUSED(audit_completions(R"({"n":129})"), "n");

    // The refusal has to say how many were asked for, and to name the ceiling it
    // applies rather than leaving the caller to find it.
    CHECK(audit_chat(R"({"n":999})").requested == "999");
    CHECK(audit_chat(R"({"n":999})").message.find(std::to_string(kMaxChoices)) !=
          std::string::npos);

    CHECK(requested_choices(object_of(R"({"n":3})")) == 3);
    CHECK(requested_choices(object_of(R"({"n":1})")) == 1);
    CHECK(requested_choices(object_of("{}")) == 1);
    // A body the audit would have refused generates one choice rather than a
    // truncated count.
    CHECK(requested_choices(object_of(R"({"n":2.5})")) == 1);
    CHECK(requested_choices(object_of(R"({"n":0})")) == 1);
    CHECK(requested_choices(object_of(R"({"n":999})")) == 1);
    CHECK(requested_choices(object_of(R"({"n":"3"})")) == 1);
}

// `stop` moved from refused to implemented, so what is left to refuse is a
// value of the wrong shape: a number, a nested array, or a list with a non-string
// entry. Every well-formed value -- including the empty ones -- is accepted,
// because a sequence that is present is now actually matched against the text.
void test_rejects_stop() {
    CHECK(audit_chat(R"({"stop":"\n\n"})").ok);
    CHECK(audit_chat(R"({"stop":["USER:"]})").ok);
    CHECK(audit_chat(R"({"stop":["a","b"]})").ok);
    CHECK(audit_chat(R"({"stop":""})").ok);
    CHECK(audit_chat(R"({"stop":[]})").ok);
    CHECK(audit_chat(R"({"stop":["",""]})").ok);

    REFUSED(audit_chat(R"({"stop":5})"), "stop");
    REFUSED(audit_chat(R"({"stop":true})"), "stop");
    REFUSED(audit_chat(R"({"stop":{"a":1}})"), "stop");
    REFUSED(audit_chat(R"({"stop":["a",5]})"), "stop");
    REFUSED(audit_chat(R"({"stop":[["a"]]})"), "stop");
    REFUSED(audit_completions(R"({"stop":5})"), "stop");

    // The refusal names the field and shows what arrived, so a caller sending a
    // number sees which value was rejected rather than only which field.
    CHECK(audit_chat(R"({"stop":5})").requested == "5");
}

// `logprobs` and `top_logprobs` moved from refused to implemented, so what is
// left to refuse is a value this server cannot act on. The two endpoints spell
// the same request differently -- chat takes a boolean plus a count in
// "top_logprobs", /v1/completions takes one count in "logprobs" -- so each value
// is checked against the endpoint that defines it.
void test_logprobs_shape() {
    CHECK(audit_chat(R"({"logprobs":false})").ok);
    CHECK(audit_chat(R"({"logprobs":true})").ok);
    CHECK(audit_chat(R"({"logprobs":true,"top_logprobs":5})").ok);
    CHECK(audit_chat(R"({"logprobs":true,"top_logprobs":0})").ok);
    // A count is the other endpoint's spelling of the field, so it is not a
    // boolean here.
    REFUSED(audit_chat(R"({"logprobs":1})"), "logprobs");
    REFUSED(audit_chat(R"({"logprobs":"true"})"), "logprobs");
    REFUSED(audit_chat(R"({"logprobs":[]})"), "logprobs");

    // /v1/completions takes a count, where 0 already asks for the sampled
    // token's own logprob, so every value in range asks for something.
    CHECK(audit_completions(R"({"logprobs":0})").ok);
    CHECK(audit_completions(R"({"logprobs":5})").ok);
    CHECK(audit_completions(R"({"logprobs":20})").ok);
    CHECK(audit_completions(R"({"logprobs":null})").ok);
    REFUSED(audit_completions(R"({"logprobs":-1})"), "logprobs");
    // 2.5 is a count that is not a count; the message has to say so rather than
    // report it as out of range.
    REFUSED(audit_completions(R"({"logprobs":2.5})"), "logprobs");
    REFUSED(audit_completions(R"({"logprobs":true})"), "logprobs");
    REFUSED(audit_completions(R"({"logprobs":21})"), "logprobs");
    REFUSED(audit_completions(R"({"logprobs":"5"})"), "logprobs");

    // "top_logprobs" is a chat field, and on its own it is inert only at 0.
    REFUSED(audit_completions(R"({"top_logprobs":5})"), "top_logprobs");
    CHECK(audit_completions(R"({"top_logprobs":null})").ok);
    REFUSED(audit_chat(R"({"top_logprobs":5})"), "top_logprobs");
    REFUSED(audit_chat(R"({"top_logprobs":-1})"), "top_logprobs");
    REFUSED(audit_chat(R"({"top_logprobs":"5"})"), "top_logprobs");
    REFUSED(audit_chat(R"({"logprobs":false,"top_logprobs":5})"), "top_logprobs");
    CHECK(audit_chat(R"({"top_logprobs":0})").ok);
    CHECK(audit_chat(R"({"logprobs":false,"top_logprobs":0})").ok);

    // The ceiling is named in the message rather than left for the caller to
    // find, on both endpoints' spellings of it.
    CHECK(audit_chat(R"({"logprobs":true,"top_logprobs":64})")
              .message.find(std::to_string(kMaxLogprobAlternatives)) !=
          std::string::npos);
    CHECK(audit_completions(R"({"logprobs":64})")
              .message.find(std::to_string(kMaxLogprobAlternatives)) !=
          std::string::npos);
}

// A streamed chunk carries the text of its token and no ranking beside it, so a
// request that asks for log probabilities on the streaming path is refused
// rather than answered with a stream indistinguishable from one whose request
// asked for none.
void test_rejects_streamed_logprobs() {
    REFUSED(audit_chat(R"({"stream":true,"logprobs":true})"), "logprobs");
    REFUSED(audit_chat(R"({"stream":true,"logprobs":true,"top_logprobs":3})"),
            "logprobs");
    REFUSED(audit_completions(R"({"stream":true,"logprobs":5})"), "logprobs");
    // 0 asks for the sampled token's own probability here, so it is not the
    // "off" value the way false is on chat.
    REFUSED(audit_completions(R"({"stream":true,"logprobs":0})"), "logprobs");

    CHECK(audit_chat(R"({"stream":true,"logprobs":false})").ok);
    CHECK(audit_chat(R"({"stream":true})").ok);
    CHECK(audit_completions(R"({"stream":true})").ok);
    CHECK(audit_completions(R"({"stream":true,"logprobs":null})").ok);
    // The refusal is about the pair, not about either field.
    CHECK(audit_chat(R"({"logprobs":true})").ok);
    CHECK(audit_completions(R"({"logprobs":5})").ok);
}

void test_rejects_penalties() {
    REFUSED(audit_chat(R"({"frequency_penalty":0.5})"), "frequency_penalty");
    REFUSED(audit_chat(R"({"presence_penalty":-2})"), "presence_penalty");
    REFUSED(audit_completions(R"({"frequency_penalty":1.5})"), "frequency_penalty");
    REFUSED(audit_chat(R"({"frequency_penalty":"0"})"), "frequency_penalty");
    CHECK(audit_chat(R"({"frequency_penalty":0,"presence_penalty":0.0})").ok);
}

void test_rejects_logit_bias() {
    REFUSED(audit_chat(R"({"logit_bias":{"50256":-100}})"), "logit_bias");
    REFUSED(audit_chat(R"({"logit_bias":[]})"), "logit_bias");
    CHECK(audit_chat(R"({"logit_bias":{}})").ok);

    // A large bias is summarised by size rather than dumped into the message.
    CHECK(audit_chat(R"({"logit_bias":{"1":1,"2":2,"3":3}})").requested == "{3 keys}");
}

void test_rejects_best_of() {
    REFUSED(audit_completions(R"({"best_of":2})"), "best_of");
    REFUSED(audit_completions(R"({"best_of":0})"), "best_of");
    CHECK(audit_completions(R"({"best_of":1})").ok);

    // "best_of" is a /v1/completions field; the chat endpoint has no such
    // parameter, so a body carrying one is not audited for it.
    CHECK(audit_chat(R"({"best_of":4})").ok);
}

void test_rejects_tool_policy() {
    REFUSED(audit_chat(R"({"tool_choice":"none"})"), "tool_choice");
    REFUSED(audit_chat(R"({"tool_choice":"required"})"), "tool_choice");
    REFUSED(audit_chat(R"({"tool_choice":{"type":"function"}})"), "tool_choice");
    CHECK(audit_chat(R"({"tool_choice":"auto"})").ok);

    REFUSED(audit_chat(R"({"parallel_tool_calls":false})"), "parallel_tool_calls");
    REFUSED(audit_chat(R"({"parallel_tool_calls":"false"})"), "parallel_tool_calls");
    CHECK(audit_chat(R"({"parallel_tool_calls":true})").ok);

    // "tool_choice" is a chat field; a completions body is not audited for it.
    CHECK(audit_completions(R"({"tool_choice":"required"})").ok);
}

void test_rejects_stream_options() {
    REFUSED(audit_chat(R"({"stream":true,"stream_options":{"include_usage":true}})"),
            "stream_options.include_usage");
    CHECK(audit_chat(R"({"stream":true,"stream_options":{"include_usage":false}})").ok);
    CHECK(audit_chat(R"({"stream":true,"stream_options":{}})").ok);

    // Without streaming there is no chunk stream to add usage to, and a
    // non-streaming response already carries "usage" -- which is the whole
    // thing the option asks for, so it is satisfied rather than ignored.
    CHECK(audit_chat(R"({"stream_options":{"include_usage":true}})").ok);
    REFUSED(audit_completions(R"({"stream":true,"stream_options":{"include_usage":true}})"),
            "stream_options.include_usage");
}

void test_rejects_completions_only_fields() {
    REFUSED(audit_completions(R"({"suffix":"\nEND"})"), "suffix");
    CHECK(audit_completions(R"({"suffix":""})").ok);
    REFUSED(audit_completions(R"({"echo":true})"), "echo");
    CHECK(audit_completions(R"({"echo":false})").ok);
    // Neither is a chat field.
    CHECK(audit_chat(R"({"suffix":"x","echo":true})").ok);
}

void test_effective_max_tokens() {
    const JsonObject empty = object_of("{}");
    CHECK(effective_max_tokens(empty, 128) == 128);

    CHECK(effective_max_tokens(object_of(R"({"max_tokens":16})"), 128) == 16);
    CHECK(effective_max_tokens(object_of(R"({"max_completion_tokens":32})"), 128) == 32);

    // OpenAI deprecated "max_tokens" in favour of "max_completion_tokens" and
    // applies the latter when a request carries both.
    CHECK(effective_max_tokens(
              object_of(R"({"max_tokens":16,"max_completion_tokens":32})"), 128) == 32);

    // Anything that cannot be a budget falls back, which is what the server did
    // before this helper existed.
    CHECK(effective_max_tokens(object_of(R"({"max_tokens":0})"), 128) == 128);
    CHECK(effective_max_tokens(object_of(R"({"max_tokens":-5})"), 128) == 128);
    CHECK(effective_max_tokens(object_of(R"({"max_tokens":"64"})"), 128) == 128);
    CHECK(effective_max_tokens(object_of(R"({"max_completion_tokens":null})"), 128) == 128);
    // A budget past INT_MAX is clamped instead of converted, which would be
    // undefined behaviour.
    CHECK(effective_max_tokens(object_of(R"({"max_tokens":1e30})"), 128) == 2147483647);
    // A usable "max_tokens" still applies when the newer field is unusable.
    CHECK(effective_max_tokens(
              object_of(R"({"max_tokens":16,"max_completion_tokens":0})"), 128) == 16);
}

}  // namespace

int main() {
    test_accepts_defaults();
    test_choices();
    test_rejects_stop();
    test_logprobs_shape();
    test_rejects_streamed_logprobs();
    test_rejects_penalties();
    test_rejects_logit_bias();
    test_rejects_best_of();
    test_rejects_tool_policy();
    test_rejects_stream_options();
    test_rejects_completions_only_fields();
    test_effective_max_tokens();

    if (g_failures != 0) {
        std::cout << g_failures << " request field test(s) failed\n";
        return 1;
    }
    std::cout << "All request field tests passed\n";
    return 0;
}
