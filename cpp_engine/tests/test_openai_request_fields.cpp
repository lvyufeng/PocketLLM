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

void test_rejects_n() {
    REFUSED(audit_chat(R"({"n":3})"), "n");
    REFUSED(audit_chat(R"({"n":0})"), "n");
    REFUSED(audit_chat(R"({"n":"3"})"), "n");
    REFUSED(audit_completions(R"({"n":2})"), "n");
    CHECK(audit_chat(R"({"n":1})").ok);

    // The refusal has to say how many were asked for.
    CHECK(audit_chat(R"({"n":3})").requested == "3");
}

void test_rejects_stop() {
    REFUSED(audit_chat(R"({"stop":"\n\n"})"), "stop");
    REFUSED(audit_chat(R"({"stop":["USER:"]})"), "stop");
    REFUSED(audit_chat(R"({"stop":["","END"]})"), "stop");
    REFUSED(audit_chat(R"({"stop":5})"), "stop");
    CHECK(audit_chat(R"({"stop":""})").ok);
    CHECK(audit_chat(R"({"stop":[]})").ok);
    CHECK(audit_chat(R"({"stop":["",""]})").ok);

    // A short list of strings renders its entries, so the caller can see which
    // sequence was refused without re-reading the request it just sent.
    CHECK(audit_chat(R"({"stop":["a","b"]})").requested == "[\"a\", \"b\"]");
}

void test_rejects_logprobs() {
    REFUSED(audit_chat(R"({"logprobs":true})"), "logprobs");
    REFUSED(audit_chat(R"({"logprobs":1})"), "logprobs");
    CHECK(audit_chat(R"({"logprobs":false})").ok);

    // /v1/completions takes a count, where 0 already asks for the sampled
    // token's logprob, so nothing about the field is inert there.
    REFUSED(audit_completions(R"({"logprobs":0})"), "logprobs");
    REFUSED(audit_completions(R"({"logprobs":5})"), "logprobs");
    CHECK(audit_completions(R"({"logprobs":null})").ok);

    REFUSED(audit_chat(R"({"top_logprobs":5})"), "top_logprobs");
    REFUSED(audit_chat(R"({"top_logprobs":-1})"), "top_logprobs");
    REFUSED(audit_chat(R"({"top_logprobs":"5"})"), "top_logprobs");
    CHECK(audit_chat(R"({"top_logprobs":0})").ok);
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
    test_rejects_n();
    test_rejects_stop();
    test_rejects_logprobs();
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
