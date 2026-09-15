#pragma once

#include <string>

#include "json_lite.hpp"

namespace pocket {

// Which endpoint a request body is being audited against. The two are not
// interchangeable: `prompt` is a no-op on chat and the whole conversation lives
// in `messages`, while `tool_choice` only exists on chat.
enum class OpenAiEndpoint {
    ChatCompletions,
    Completions,
};

// Outcome of auditing a request body against what this server actually does.
struct RequestFieldCheck {
    bool ok = true;
    // The offending field's name ("n", "stop", ...), empty when ok. Carried
    // separately so the HTTP layer can put it in the OpenAI `param` slot instead
    // of making the client parse it back out of the message.
    std::string field;
    // What the request asked for, rendered for the message ("3", "[2 items]").
    std::string requested;
    // Complete client-facing message naming the field, the requested value, what
    // this server does instead, and what to do about it.
    std::string message;
};

// Rejects every documented OpenAI request field this server would otherwise
// ignore, when the value the client sent would change the output. A field whose
// value names what this server does anyway -- n=1, logprobs=false, penalties of
// zero, an empty stop list -- is accepted, so clients that send the defaults
// explicitly are not punished for it.
//
// The alternative was the behaviour this replaces: `n=3` returning one choice
// and `logprobs=true` returning no logprobs, both with a 200 and no indication
// that anything was dropped. That is the same failure mode as a capability flag
// that reports batching it does not do -- the request looks configured and the
// output is something else.
//
// Fields that cannot change the generated text -- `user`, `store`, `metadata`,
// `service_tier`, `model` -- are accepted and inert on purpose; rejecting them
// would break clients over nothing. That test is applied field by field rather
// than by category, though: `parallel_tool_calls` reads like one of those, but
// a client sending false is asking for a limit this server does not enforce, so
// that value is refused. See docs/guides/pocketllm_api.md for the table.
RequestFieldCheck check_request_fields(const JsonObject& body, OpenAiEndpoint endpoint);

// The effective generation budget: OpenAI deprecated max_tokens in favour of
// max_completion_tokens and gives the latter precedence when both are present.
// Only a positive number counts, and `fallback` is used when neither field is
// usable, which is the same substitution the server made before this existed.
int effective_max_tokens(const JsonObject& body, int fallback);

}  // namespace pocket
