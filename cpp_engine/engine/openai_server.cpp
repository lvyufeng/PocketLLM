#include "openai_server.hpp"

#include "batch_scheduler.hpp"
#include "metrics.hpp"
#include "json_lite.hpp"
#include "openai_request_fields.hpp"
#include "openai_stop_strings.hpp"
#include "qwen_engine.hpp"
#include "token_constraint.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pocket {

namespace {

std::string make_request_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream os;
    os << "req_" << std::hex << dist(gen);
    return os.str();
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
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
    return out;
}

// Returns the JSON array text for `messages` from the request, preserving the
// caller-provided structure verbatim. json_lite doesn't ship a serializer, so
// we extract the raw substring from the original body using offsets we infer
// by re-rendering the value.
std::string extract_messages_json(const std::string& body) {
    // Trivially locate `"messages"` and bracket-match to find the array.
    size_t key = body.find("\"messages\"");
    if (key == std::string::npos) return "[]";
    size_t pos = body.find('[', key);
    if (pos == std::string::npos) return "[]";
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = pos; i < body.size(); ++i) {
        char c = body[i];
        if (esc) { esc = false; continue; }
        if (c == '\\' && in_str) { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') ++depth;
        else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0 && c == ']') return body.substr(pos, i - pos + 1);
        }
    }
    return "[]";
}

// Same as extract_messages_json but for the top-level `tools` array. Returns
// an empty string when no `tools` field is present.
std::string extract_tools_json(const std::string& body) {
    size_t key = body.find("\"tools\"");
    if (key == std::string::npos) return "";
    size_t pos = body.find('[', key);
    if (pos == std::string::npos) return "";
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = pos; i < body.size(); ++i) {
        char c = body[i];
        if (esc) { esc = false; continue; }
        if (c == '\\' && in_str) { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') ++depth;
        else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0 && c == ']') return body.substr(pos, i - pos + 1);
        }
    }
    return "";
}

double get_number(const JsonObject& obj, const std::string& key, double fallback) {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_number()) return v->number();
    return fallback;
}

bool get_bool(const JsonObject& obj, const std::string& key, bool fallback) {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_bool()) return v->boolean();
    return fallback;
}

std::string get_string(const JsonObject& obj, const std::string& key, const std::string& fallback = "") {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_string()) return v->string();
    return fallback;
}

// Truncate `s` to its longest prefix that is a complete UTF-8 sequence.
// Returns {complete_prefix, leftover_bytes}. Walks back at most 3 bytes from
// the end looking for a start byte; if the trailing sequence is complete the
// full string is returned, if incomplete the partial sequence is withheld.
std::pair<std::string, std::string> split_utf8_complete(const std::string& s) {
    if (s.empty()) return {"", ""};
    size_t i = s.size();
    for (int back = 0; back < 4 && i > 0; ++back) {
        unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0x80) == 0) {
            // ASCII byte at i-1; everything up to s.size() is complete.
            return { s, "" };
        }
        if ((c & 0xC0) == 0xC0) {
            // Start byte at i-1.
            size_t expected = 1;
            if ((c & 0xE0) == 0xC0) expected = 2;
            else if ((c & 0xF0) == 0xE0) expected = 3;
            else if ((c & 0xF8) == 0xF0) expected = 4;
            size_t available = s.size() - (i - 1);
            if (available >= expected) {
                // Trailing sequence is complete.
                return { s, "" };
            }
            // Incomplete; withhold the partial sequence starting at i-1.
            return { s.substr(0, i - 1), s.substr(i - 1) };
        }
        // Continuation byte; keep walking back.
        --i;
    }
    // No start byte found within the last 4 bytes; emit everything as-is.
    return { s, "" };
}

std::string render_choice_message(const std::string& content, const std::string& reasoning, const std::string& tool_calls_json) {
    std::ostringstream os;
    os << "{\"role\":\"assistant\"";
    os << ",\"content\":\"" << json_escape(content) << "\"";
    if (!reasoning.empty()) {
        os << ",\"reasoning_content\":\"" << json_escape(reasoning) << "\"";
    }
    if (!tool_calls_json.empty() && tool_calls_json != "[]") {
        os << ",\"tool_calls\":" << tool_calls_json;
    }
    os << "}";
    return os.str();
}

// Seeds for the choices of one request.
//
// Choice 0 keeps the seed the client named, so a pinned "seed" still describes
// the first choice, and the others are derived from it. Varying the seed is what
// makes the choices different answers instead of the same answer repeated, and
// it only means anything once sampling is stochastic: a greedy sampler ignores
// the seed entirely.
//
// splitmix64 rather than std::seed_seq with an mt19937: libstdc++'s seed_seq
// output is an implementation detail that may change between GCC versions, and a
// seed that depends on the toolchain would change the generated text from one
// build to the next. Plain 64-bit arithmetic does not.
uint64_t derived_choice_seed(uint64_t base, int choice) {
    uint64_t z = base + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(choice + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t choice_seed(uint64_t base, int choice, bool vary) {
    if (choice == 0 || !vary) return base;
    return derived_choice_seed(base, choice);
}

// Hand-off between the scheduler thread, which produces tokens, and the HTTP
// thread, which writes them to the socket.
//
// The scheduler's callbacks run inline in its schedule loop, so writing to a
// socket from them would let one slow client stall every other request in the
// batch. They only append here and signal; all socket I/O happens on the HTTP
// thread draining this. Held by shared_ptr because the callbacks outlive the
// handler's stack frame if the client disconnects early.
struct TokenStream {
    std::mutex m;
    std::condition_variable cv;
    std::deque<int> tokens;
    bool done = false;
    SchedulerGenerationResult result;

    enum class Next { Token, Done, Timeout };

    void push(int token) {
        {
            std::lock_guard<std::mutex> lk(m);
            tokens.push_back(token);
        }
        cv.notify_one();
    }

    void finish(const SchedulerGenerationResult& r) {
        {
            std::lock_guard<std::mutex> lk(m);
            result = r;
            done = true;
        }
        cv.notify_one();
    }

    // Queued tokens are drained before Done is reported: the completion
    // callback can land while tokens are still buffered, and dropping them
    // would truncate the answer.
    Next next(int* token, std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lk(m);
        while (tokens.empty() && !done) {
            if (cv.wait_until(lk, deadline) == std::cv_status::timeout &&
                tokens.empty() && !done) {
                return Next::Timeout;
            }
        }
        if (tokens.empty()) return Next::Done;
        *token = tokens.front();
        tokens.pop_front();
        return Next::Token;
    }

    // Non-blocking, for the emitters that drain several choices at once and must
    // not block on any one of them. False means "nothing queued right now", which
    // is not the same as finished: ask is_finished() for that, after draining.
    bool try_pop(int* token) {
        std::lock_guard<std::mutex> lk(m);
        if (tokens.empty()) return false;
        *token = tokens.front();
        tokens.pop_front();
        return true;
    }

    bool is_finished() {
        std::lock_guard<std::mutex> lk(m);
        return done;
    }
};

// Activity notifier shared by the choices of one request.
//
// A streaming request with several choices cannot wait on them one after
// another: the chunks of the later choices are already produced and would sit in
// their streams until the first choice finished, which is a batch download
// dressed up as a stream. So the choices share this counter instead -- every
// token and every completion bumps it -- and the HTTP thread drains whichever
// choices have something and then waits here for any of them to move.
struct ChoiceGroup {
    std::mutex m;
    std::condition_variable cv;
    uint64_t activity = 0;

    void bump() {
        {
            std::lock_guard<std::mutex> lk(m);
            ++activity;
        }
        cv.notify_all();
    }

    // Waits until the counter moves past `seen`, which `seen` is then advanced
    // to. False means the deadline passed with nothing moving.
    bool wait_for_activity(uint64_t& seen, std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lk(m);
        if (activity != seen) {
            seen = activity;
            return true;
        }
        if (cv.wait_until(lk, deadline) == std::cv_status::timeout && activity == seen) {
            return false;
        }
        seen = activity;
        return true;
    }
};

}  // namespace

struct OpenAIServer::Impl {
    InferenceEngine& engine;
    const Tokenizer& tok;
    PythonSidecar& sidecar;
    OpenAIServerConfig cfg;
    // Constructed here rather than handed in: how many completions run at once
    // is a serving decision, and the scheduler clamps it to what the engine
    // declared anyway.
    BatchScheduler sched;
    httplib::Server svr;
    std::atomic<bool> running{false};
    // Token id for "</think>", or -1 when the vocabulary has none. Looked up
    // once; the streaming path needs it per token.
    const int think_end_id;
    EngineMetrics metrics;

    // Request tracking for cancellation API
    struct TrackedRequest {
        // A request that asked for n choices runs as n scheduler requests, so
        // cancelling it has to cancel all of them rather than whichever one was
        // registered last.
        std::vector<uint64_t> scheduler_ids;
        std::string client_id;
        std::chrono::steady_clock::time_point start_time;
    };
    std::mutex tracked_requests_mutex_;
    std::unordered_map<std::string, TrackedRequest> tracked_requests_;

    Impl(InferenceEngine& e, const Tokenizer& t, PythonSidecar& s, const OpenAIServerConfig& c)
        : engine(e), tok(t), sidecar(s), cfg(c),
          sched(&e, c.max_batch_size > 0 ? c.max_batch_size : 1),
          think_end_id(t.token_id("</think>")) {
        sched.set_prefill_token_budget(cfg.prefill_token_budget);
        std::cerr << "[server] batch width " << sched.max_batch_size()
                  << " (engine declares max_slots=" << sched.engine_caps().max_slots
                  << ", continuous_batching="
                  << (sched.engine_caps().continuous_batching ? "yes" : "no")
                  << "), prefill budget " << sched.prefill_token_budget() << "\n";
    }

    void handle_health(httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    }

    void handle_ready(httplib::Response& res) {
        // The engine is initialized if we reach this point (constructor passed).
        // Return 200 to signal readiness for serving requests.
        res.set_content("{\"ready\":true}", "application/json");
    }

    void handle_alive(httplib::Response& res) {
        res.set_content("{\"alive\":true}", "application/json");
    }

    void handle_models(httplib::Response& res) {
        std::ostringstream os;
        os << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(cfg.model_name)
           << "\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
        res.set_content(os.str(), "application/json");
    }

    void handle_metrics(httplib::Response& res) {
        const auto stats = sched.get_stats();

        // Try to get Qwen-specific prefix cache stats
        const QwenPrefixCacheStats* cache_stats = nullptr;
        QwenPrefixCacheStats qwen_stats;
        QwenEngine* qwen = dynamic_cast<QwenEngine*>(&engine);
        if (qwen) {
            qwen_stats = qwen->prefix_cache_stats();
            cache_stats = &qwen_stats;
        }

        const std::string body = metrics.serialize_prometheus(
            stats.waiting_requests, stats.running_requests,
            stats.free_slots, stats.reserved_blocks, stats.total_blocks,
            stats.free_blocks, stats.cache_pinned_blocks, cache_stats);

        res.set_content(body, "text/plain; version=0.0.4");
    }

    void handle_cancel_request(const httplib::Request& req, httplib::Response& res) {
        const std::string request_id = req.path_params.at("id");

        std::vector<uint64_t> scheduler_ids;
        {
            std::lock_guard<std::mutex> lock(tracked_requests_mutex_);
            auto it = tracked_requests_.find(request_id);
            if (it == tracked_requests_.end()) {
                res.status = 404;
                std::ostringstream os;
                os << "{\"error\":{\"message\":\"Request not found or already completed\","
                   << "\"type\":\"invalid_request_error\"}}";
                res.set_content(os.str(), "application/json");
                return;
            }
            scheduler_ids = it->second.scheduler_ids;
        }

        bool cancelled = false;
        for (uint64_t id : scheduler_ids) {
            cancelled = sched.cancel_request(id) || cancelled;
        }

        std::ostringstream os;
        os << "{\"id\":\"" << json_escape(request_id) << "\""
           << ",\"object\":\"request.cancel\""
           << ",\"cancelled\":" << (cancelled ? "true" : "false") << "}";
        res.set_content(os.str(), "application/json");
    }

    // Rejects a request that names sampling this engine cannot vary per row
    // unless it happens to name exactly what the engine will do anyway. The
    // alternative -- accepting the field and sampling with something else --
    // returns output the caller has no way to know is not what it asked for.
    bool check_sampling_supported(const JsonObject& obj, std::string& err_out) const {
        const Capabilities& caps = sched.engine_caps();

        auto mismatch = [&](const char* key, double requested, double configured) {
            std::ostringstream os;
            os << "this engine's effective " << key << " is " << configured
               << " and it cannot apply " << key << "=" << requested
               << " to this request. Omit the field to accept the effective value, "
               << "or run an engine configuration that supports it.";
            err_out = os.str();
        };

        const JsonValue* v = object_get(obj, "temperature");
        if (!caps.per_request_sampling && v != nullptr && v->is_number() &&
            std::fabs(v->number() - caps.fixed_temperature) > 1.0e-5) {
            mismatch("temperature", v->number(), caps.fixed_temperature);
            return false;
        }
        const double effective_temperature = caps.per_request_sampling
            ? get_number(obj, "temperature", 1.0)
            : caps.fixed_temperature;
        const bool stochastic = effective_temperature > 1.0e-5;

        v = object_get(obj, "top_p");
        if (!caps.per_request_sampling && stochastic &&
            v != nullptr && v->is_number() &&
            std::fabs(v->number() - caps.fixed_top_p) > 1.0e-5) {
            mismatch("top_p", v->number(), caps.fixed_top_p);
            return false;
        }
        v = object_get(obj, "top_k");
        if (!caps.per_request_top_k && stochastic &&
            v != nullptr && v->is_number() &&
            static_cast<int>(v->number()) != caps.fixed_top_k) {
            mismatch("top_k", v->number(), caps.fixed_top_k);
            return false;
        }
        // Seed only changes anything once sampling is stochastic; under greedy
        // decoding every seed produces the same tokens, so naming one is not a
        // disagreement.
        v = object_get(obj, "seed");
        if (!caps.per_request_sampling && stochastic &&
            v != nullptr && v->is_number() &&
            static_cast<unsigned long long>(v->number()) != caps.fixed_seed) {
            mismatch("seed", v->number(), static_cast<double>(caps.fixed_seed));
            return false;
        }

        // Several choices are several sampling runs, and the one thing that
        // would collapse them into one run repeated is a sampling distribution
        // the engine fixes engine-wide: every choice would then be drawn from
        // that same distribution with the same seed, and n identical texts
        // would be handed back as n independent samples.
        //
        // Greedy sampling is deliberately not that case. "The same text n
        // times" is what a greedy request for n choices asks for, so it is
        // served; what is refused is claiming to have sampled when the engine
        // could not have varied anything.
        const int choices = requested_choices(obj);
        if (choices > 1 && stochastic && !caps.per_request_sampling) {
            std::ostringstream os;
            os << "\"n\" = " << choices << " is not supported by this engine: it "
               << "samples at engine-wide values (temperature "
               << caps.fixed_temperature << ") that it cannot vary per request, "
               << "so all " << choices << " choices would be the same text "
               << "presented as independent samples. Remove \"n\", or run an "
               << "engine configuration that samples per request.";
            err_out = os.str();
            return false;
        }
        return true;
    }

    bool parse_response_format(const JsonObject& obj,
                               std::shared_ptr<TokenConstraint>& constraint_out,
                               std::string& err_out) const {
        constraint_out.reset();
        const JsonValue* response_format = object_get(obj, "response_format");
        if (response_format == nullptr) return true;
        if (!sched.engine_caps().structured_outputs) {
            err_out = "this engine does not support structured outputs (response_format)";
            return false;
        }
        if (!response_format->is_object()) {
            err_out = "response_format must be an object";
            return false;
        }
        const JsonObject& format = response_format->object();
        const JsonValue* type_value = object_get(format, "type");
        if (type_value == nullptr || !type_value->is_string()) {
            err_out = "response_format requires a string type";
            return false;
        }
        const std::string& type = type_value->string();
        try {
            if (type == "text") {
                return true;
            }
            if (type == "json_object") {
                constraint_out = make_json_object_constraint(tok);
                return true;
            }
            if (type == "json_schema") {
                const JsonValue* json_schema = object_get(format, "json_schema");
                if (json_schema == nullptr || !json_schema->is_object()) {
                    err_out = "json_schema response_format requires a json_schema object";
                    return false;
                }
                const JsonValue* schema = object_get(json_schema->object(), "schema");
                if (schema == nullptr || !schema->is_object()) {
                    err_out = "json_schema response_format requires an object schema";
                    return false;
                }
                constraint_out = make_json_schema_constraint(tok, *schema);
                return true;
            }
        } catch (const std::exception& e) {
            err_out = std::string("invalid response_format: ") + e.what();
            return false;
        }
        err_out = "response_format type must be 'text', 'json_object', or 'json_schema'";
        return false;
    }

    bool encode_request(const std::string& body, const JsonObject& obj,
                        EncodeReply& out, std::string& thinking_mode_out,
                        int& max_tokens_out, bool& stream_out,
                        BatchSamplingParams& sp_out, std::string& request_id_out,
                        std::shared_ptr<TokenConstraint>& constraint_out,
                        std::string& err_out) {
        if (!check_sampling_supported(obj, err_out)) return false;

        if (!parse_response_format(obj, constraint_out, err_out)) return false;

        EncodeRequest enc;
        enc.messages_json = extract_messages_json(body);
        enc.tools_json = extract_tools_json(body);
        thinking_mode_out = get_string(obj, "thinking_mode", cfg.default_thinking_mode);
        enc.thinking_mode = thinking_mode_out;
        enc.reasoning_effort = get_string(obj, "reasoning_effort", "");
        enc.add_generation_prompt = get_bool(obj, "add_generation_prompt", true);
        enc.drop_thinking = get_bool(obj, "drop_thinking", true);

        // "max_completion_tokens" wins over the deprecated "max_tokens" when a
        // request carries both, which is OpenAI's rule; reading only
        // "max_tokens" silently handed such a request the server default.
        max_tokens_out = effective_max_tokens(obj, cfg.default_max_tokens);
        stream_out = get_bool(obj, "stream", false);

        // Extract optional request_id from client, or generate one
        request_id_out = get_string(obj, "request_id", "");
        if (request_id_out.empty()) {
            request_id_out = make_request_id();
        }

        // An engine that fixes its own sampling ignores these; check_sampling_
        // supported has already established the request agrees with them, so
        // filling them in either way keeps one code path.
        sp_out.temperature = static_cast<float>(get_number(obj, "temperature", 1.0));
        sp_out.top_p = static_cast<float>(get_number(obj, "top_p", 1.0));
        sp_out.top_k = static_cast<int>(get_number(obj, "top_k", 20));
        sp_out.seed = static_cast<unsigned long long>(get_number(obj, "seed", 0));
        sp_out.max_new_tokens = max_tokens_out;

        out = sidecar.encode(enc);
        if (!out.ok) {
            err_out = out.err;
            return false;
        }
        if (out.token_ids.empty()) {
            err_out = "encode produced no tokens";
            return false;
        }
        return true;
    }

    void emit_error(httplib::Response& res, int status, const std::string& msg) {
        std::ostringstream os;
        os << "{\"error\":{\"message\":\"" << json_escape(msg)
           << "\",\"type\":\"server_error\"}}";
        res.status = status;
        res.set_content(os.str(), "application/json");
    }

    // 400 in the OpenAI error shape, with `param` naming the field the caller
    // has to change so a client can act on it without parsing the prose. That
    // field is the difference between this and the generic shape emit_error
    // writes, and it is what makes a refusal machine-readable.
    void emit_request_error(httplib::Response& res, const std::string& msg,
                            const std::string& param) {
        std::ostringstream os;
        os << "{\"error\":{\"message\":\"" << json_escape(msg)
           << "\",\"type\":\"invalid_request_error\"";
        if (!param.empty()) {
            os << ",\"param\":\"" << json_escape(param) << "\"";
        }
        os << ",\"code\":null}}";
        res.status = 400;
        res.set_content(os.str(), "application/json");
    }

    // Audits the body against the request fields this server implements and, on
    // refusal, writes the 400 itself. Returns whether the handler should carry
    // on. Called before any generation work -- and before the sidecar is asked
    // to tokenize or encode -- so a request that cannot be honoured costs
    // nothing and is refused for the field it named rather than for whatever
    // the engine would have failed on later.
    bool audit_request_fields(httplib::Response& res, const JsonObject& obj,
                              OpenAiEndpoint endpoint) {
        const RequestFieldCheck check = check_request_fields(obj, endpoint);
        if (check.ok) return true;
        metrics.record_request_end(false, 0.0, 0.0, 0, 0);
        emit_request_error(res, check.message, check.field);
        return false;
    }

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        const auto request_start = std::chrono::steady_clock::now();
        metrics.record_request_start();

        const std::string& body = req.body;
        JsonValue jv;
        try { jv = parse_json(body); } catch (const std::exception& ex) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, std::string("invalid JSON: ") + ex.what());
            return;
        }
        if (!jv.is_object()) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "request body must be JSON object");
            return;
        }
        const auto& obj = jv.object();
        if (!audit_request_fields(res, obj, OpenAiEndpoint::ChatCompletions)) return;
        EncodeReply enc_reply;
        std::string thinking_mode;
        int max_tokens = 0;
        bool stream = false;
        BatchSamplingParams sp;
        std::string client_id;
        std::shared_ptr<TokenConstraint> constraint;
        std::string err;
        if (!encode_request(body, obj, enc_reply, thinking_mode, max_tokens, stream, sp, client_id, constraint, err)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, err);
            return;
        }

        if (cfg.log_requests) {
            std::cerr << "[server] request prompt_tokens=" << enc_reply.token_ids.size()
                      << " max_tokens=" << max_tokens << " stream=" << (stream ? 1 : 0)
                      << " thinking_mode=" << thinking_mode << "\n";
        }

        // No inflight mutex: concurrency is the scheduler's to decide, bounded
        // by what the engine declared. Serialising here would give a paged
        // batched engine the throughput of a single-session one.
        if (static_cast<int>(enc_reply.token_ids.size()) + max_tokens > engine.max_context()) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "prompt + max_tokens exceeds max_context");
            return;
        }

        // Parsed once per request: every emitter below applies the same list to
        // the text it is about to hand over.
        const std::vector<std::string> stops = parse_stop_sequences(obj);

        // One constraint object per choice: a TokenConstraint is stateful and
        // only the scheduler advances it, so two choices sharing one object
        // would let one choice's grammar state decide what the other may
        // generate. The audit has already vouched for the count.
        std::vector<std::shared_ptr<TokenConstraint>> constraints;
        if (!build_choice_constraints(obj, requested_choices(obj), std::move(constraint),
                                      constraints, err)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, err);
            return;
        }

        if (stream) {
            handle_stream(req, res, enc_reply, sp, constraints, stops, thinking_mode,
                          client_id, request_start);
        } else {
            handle_nonstream(res, enc_reply, sp, constraints, stops, thinking_mode,
                             client_id, request_start);
        }
    }

    void handle_completions(const httplib::Request& req, httplib::Response& res) {
        const auto request_start = std::chrono::steady_clock::now();
        metrics.record_request_start();

        const std::string& body = req.body;
        JsonValue jv;
        try { jv = parse_json(body); } catch (const std::exception& ex) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, std::string("invalid JSON: ") + ex.what());
            return;
        }
        if (!jv.is_object()) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "request body must be JSON object");
            return;
        }
        const auto& obj = jv.object();

        // Extract prompt (string or array of strings)
        std::string prompt_str;
        const JsonValue* prompt_val = object_get(obj, "prompt");
        if (prompt_val == nullptr) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "missing required field: prompt");
            return;
        }
        if (prompt_val->is_string()) {
            prompt_str = prompt_val->string();
        } else if (prompt_val->is_array()) {
            const auto& arr = prompt_val->array();
            if (arr.empty()) {
                metrics.record_request_end(false, 0.0, 0.0, 0, 0);
                emit_error(res, 400, "prompt array must not be empty");
                return;
            }
            // Only handle single-prompt case for now (n=1)
            if (!arr[0].is_string()) {
                metrics.record_request_end(false, 0.0, 0.0, 0, 0);
                emit_error(res, 400, "prompt array must contain strings");
                return;
            }
            prompt_str = arr[0].string();
        } else {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "prompt must be string or array of strings");
            return;
        }

        if (!audit_request_fields(res, obj, OpenAiEndpoint::Completions)) return;

        // Tokenize without chat template
        TokenizeRequest tok_req;
        tok_req.prompt = prompt_str;
        TokenizeReply tok_reply = sidecar.tokenize(tok_req);
        if (!tok_reply.ok) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "tokenization failed: " + tok_reply.err);
            return;
        }
        if (tok_reply.token_ids.empty()) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "tokenization produced no tokens");
            return;
        }

        // Extract generation parameters
        std::string err;
        if (!check_sampling_supported(obj, err)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, err);
            return;
        }
        std::shared_ptr<TokenConstraint> constraint;
        if (!parse_response_format(obj, constraint, err)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, err);
            return;
        }

        // Same precedence rule as /v1/chat/completions: "max_completion_tokens"
        // wins over the deprecated "max_tokens" when both are present.
        int max_tokens = effective_max_tokens(obj, cfg.default_max_tokens);
        bool stream = get_bool(obj, "stream", false);

        std::string client_id = get_string(obj, "request_id", "");
        if (client_id.empty()) {
            client_id = make_request_id();
        }

        BatchSamplingParams sp;
        sp.temperature = static_cast<float>(get_number(obj, "temperature", 1.0));
        sp.top_p = static_cast<float>(get_number(obj, "top_p", 1.0));
        sp.top_k = static_cast<int>(get_number(obj, "top_k", 20));
        sp.seed = static_cast<unsigned long long>(get_number(obj, "seed", 0));
        sp.max_new_tokens = max_tokens;

        if (static_cast<int>(tok_reply.token_ids.size()) + max_tokens > engine.max_context()) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, "prompt + max_tokens exceeds max_context");
            return;
        }

        if (cfg.log_requests) {
            std::cerr << "[server] /v1/completions prompt_tokens=" << tok_reply.token_ids.size()
                      << " max_tokens=" << max_tokens << " stream=" << (stream ? 1 : 0) << "\n";
        }

        // Reuse the same generation logic, but use "text_completion" format
        EncodeReply enc_reply;
        enc_reply.ok = true;
        enc_reply.token_ids = std::move(tok_reply.token_ids);
        enc_reply.prompt_text = prompt_str;

        // Parsed once per request: every emitter below applies the same list to
        // the text it is about to hand over.
        const std::vector<std::string> stops = parse_stop_sequences(obj);

        // One constraint object per choice, for the reason given in the chat
        // handler above.
        std::vector<std::shared_ptr<TokenConstraint>> constraints;
        if (!build_choice_constraints(obj, requested_choices(obj), std::move(constraint),
                                      constraints, err)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 400, err);
            return;
        }

        if (stream) {
            handle_completions_stream(enc_reply, sp, constraints, stops, client_id,
                                      request_start, res);
        } else {
            handle_completions_nonstream(enc_reply, sp, constraints, stops, client_id,
                                         request_start, res);
        }
    }

    // One in-flight choice: the scheduler request it runs as, and the hand-off
    // its tokens and its result arrive on. A choice is one scheduler request
    // because one request is one slot with one sampling configuration, and that
    // produces one sequence.
    struct ChoiceRun {
        uint64_t request_id = 0;
        std::shared_ptr<TokenStream> stream;
    };

    // Per-choice streaming state: the tokens seen so far, the bytes already
    // written, and how the choice ended. Every choice advances through these
    // independently -- one choice finishing does not end the response.
    struct ChoiceStream {
        std::vector<int> generated;
        std::size_t sent_offset = 0;  // bytes of decode_tokens(generated) already sent
        bool in_reasoning = false;    // chat only: the prompt ended inside <think>
        bool stop_matched = false;
        bool finished = false;        // its token stream has been drained
        bool timed_out = false;
        std::string finish_reason = "length";
        std::string error;
    };

    // What one completed choice produced: the text a client reads and the counts
    // the usage block is built from.
    struct ChoiceText {
        int index = 0;
        std::string finish_reason;
        std::string text;
        std::size_t completion_tokens = 0;
    };

    // One grammar per choice.
    //
    // A TokenConstraint is stateful: the scheduler advances it with the tokens it
    // accepts and resets it when it completes. Choices sharing one object would
    // let a choice that finished wipe the grammar out from under the ones still
    // generating, and would have them all accept each other's tokens.
    // parse_response_format is a pure function of the body, so running it once
    // per choice is what gives each choice its own machine; the first one comes
    // from the audit path, which already built it.
    bool build_choice_constraints(const JsonObject& obj, int choices,
                                  std::shared_ptr<TokenConstraint> first,
                                  std::vector<std::shared_ptr<TokenConstraint>>& out,
                                  std::string& err_out) {
        out.assign(static_cast<std::size_t>(choices), nullptr);
        if (first == nullptr) return true;  // no structured output was requested
        out[0] = std::move(first);
        for (int i = 1; i < choices; ++i) {
            if (!parse_response_format(obj, out[i], err_out)) return false;
            if (out[i] == nullptr) {
                err_out = "response_format produced no constraint";
                return false;
            }
        }
        return true;
    }

    // Submits one scheduler request per choice.
    //
    // `group` non-null additionally streams each choice's tokens onto its stream
    // as they are produced and bumps the group on every token and every
    // completion; that is what the streaming emitters drain. The non-streaming
    // emitters pass null and wait on the completion result alone.
    //
    // Returns false when the request could never be admitted. That is decided by
    // the prompt and the budget, which every choice shares, so it is one decision
    // for the group rather than one per choice -- and when it happens, nothing is
    // left submitted.
    bool submit_choices(const std::vector<int>& prompt_tokens,
                        const BatchSamplingParams& sp,
                        const std::vector<std::shared_ptr<TokenConstraint>>& constraints,
                        const std::shared_ptr<ChoiceGroup>& group,
                        std::vector<ChoiceRun>& out) {
        out.assign(constraints.size(), ChoiceRun{});
        // Distinct seeds are what makes the choices different answers rather than
        // one answer repeated. They only differ when the sampler reads them --
        // under greedy every seed produces the same tokens -- and the case where
        // the engine samples stochastically at values it cannot vary per request
        // is refused for n > 1 before it reaches here.
        const bool vary_seed = sp.temperature > 1.0e-5f;
        for (std::size_t i = 0; i < constraints.size(); ++i) {
            const std::shared_ptr<TokenStream> stream = std::make_shared<TokenStream>();
            BatchSamplingParams choice_sp = sp;
            choice_sp.seed = choice_seed(sp.seed, static_cast<int>(i), vary_seed);
            choice_sp.constraint = constraints[i].get();
            std::function<void(const SchedulerGenerationResult&)> on_finish =
                [stream, group](const SchedulerGenerationResult& r) {
                    stream->finish(r);
                    if (group) group->bump();
                };
            TokenCallback on_token;
            if (group) {
                on_token = [stream, group](uint64_t, int token) {
                    stream->push(token);
                    group->bump();
                };
            }
            const uint64_t id = sched.submit_request(prompt_tokens, choice_sp, on_finish,
                                                     on_token, constraints[i]);
            if (id == 0) {
                for (std::size_t j = 0; j < i; ++j) {
                    sched.cancel_request(out[j].request_id);
                }
                out.clear();
                return false;
            }
            out[i].request_id = id;
            out[i].stream = stream;
        }
        return true;
    }

    // Tracks every choice under the one client-visible id, so cancelling the
    // request cancels all of it.
    void track_request(const std::string& client_id, const std::vector<ChoiceRun>& runs) {
        std::vector<uint64_t> ids;
        ids.reserve(runs.size());
        for (const ChoiceRun& run : runs) ids.push_back(run.request_id);
        std::lock_guard<std::mutex> lock(tracked_requests_mutex_);
        tracked_requests_[client_id] = {std::move(ids), client_id,
                                        std::chrono::steady_clock::now()};
    }

    void untrack_request(const std::string& client_id) {
        std::lock_guard<std::mutex> lock(tracked_requests_mutex_);
        tracked_requests_.erase(client_id);
    }

    // Waits for every choice, records the request's metrics, and reduces the
    // outcomes to the answers that arrived. The two non-streaming emitters do the
    // same thing up to how one answer is rendered, which is how much of it is
    // shared here.
    //
    // Returns false -- having written the error response -- when no choice
    // produced anything. A choice that did not arrive is left out rather than
    // reported as an empty answer, and each entry carries the index of the choice
    // it is, so a caller can see which ones are missing.
    bool finish_choices(httplib::Response& res, const std::vector<ChoiceRun>& runs,
                        const std::vector<std::string>& stops,
                        const std::vector<int>& prompt_tokens,
                        const std::string& client_id,
                        std::chrono::steady_clock::time_point request_start,
                        std::vector<ChoiceText>& texts_out,
                        std::size_t& completion_tokens_out) {
        texts_out.clear();
        completion_tokens_out = 0;

        std::chrono::steady_clock::time_point ttft_time;
        bool ttft_recorded = false;
        std::string first_error;

        // The timeout is the request's, not the choice's: a group of choices gets
        // the one budget a single-choice request would have had, measured the
        // same way, from just after submission. A request wide enough to queue
        // some of its choices behind the batch can therefore come back short.
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(cfg.request_timeout_seconds);
        for (std::size_t i = 0; i < runs.size(); ++i) {
            int unused_token = 0;
            if (runs[i].stream->next(&unused_token, deadline) == TokenStream::Next::Timeout) {
                // Abandoning one choice of several leaves no result behind: the
                // callback owns the stream, so it stays valid until the scheduler
                // observes the cancellation, and a callback-mode request is not
                // duplicated into poll_result().
                sched.cancel_request(runs[i].request_id);
                continue;
            }
            if (!ttft_recorded) {
                ttft_time = std::chrono::steady_clock::now();
                ttft_recorded = true;
            }
            SchedulerGenerationResult out;
            {
                std::lock_guard<std::mutex> lk(runs[i].stream->m);
                out = runs[i].stream->result;
            }
            if (!out.error.empty()) {
                if (first_error.empty()) first_error = out.error;
                continue;
            }

            const std::vector<int> generated = strip_stop_token(
                out.generated_tokens, out.finish_reason, out.constraint_completed);
            // A client stop sequence ends the answer at its first byte. Matching
            // runs on the decoded text rather than on token ids because a
            // sequence is not one token; see openai_stop_strings.hpp. The engine
            // still ran to max_tokens -- the scheduler has no way to see a
            // text-level sequence -- but the completion the client reads ends at
            // the sequence, which is what "stop" means to a caller.
            const std::string decoded = tok.decode_tokens(generated);
            const StopScan scan = scan_stop_strings(decoded, stops);

            ChoiceText text;
            text.index = static_cast<int>(i);
            text.text = decoded.substr(0, scan.final_length);
            text.finish_reason = scan.matched ? "stop" : out.finish_reason;
            text.completion_tokens = generated.size();
            completion_tokens_out += generated.size();
            texts_out.push_back(std::move(text));
        }

        const auto request_end = std::chrono::steady_clock::now();
        const double duration =
            std::chrono::duration<double>(request_end - request_start).count();
        const double ttft = ttft_recorded
            ? std::chrono::duration<double>(ttft_time - request_start).count()
            : 0.0;

        if (texts_out.empty()) {
            metrics.record_request_end(false, duration, ttft, prompt_tokens.size(), 0);
            if (!first_error.empty()) {
                emit_error(res, 500, first_error);
            } else {
                emit_error(res, 504, "generation timed out");
            }
            return false;
        }

        // A short answer is reported rather than returned silently: the response
        // is a 200 with fewer choices than were asked for.
        if (cfg.log_requests && texts_out.size() != runs.size()) {
            std::cerr << "[server] request " << client_id << " returned "
                      << texts_out.size() << " of " << runs.size() << " choices\n";
        }
        metrics.record_request_end(true, duration, ttft, prompt_tokens.size(),
                                   completion_tokens_out);
        return true;
    }

    // A stop token is the last token of a sequence and is still counted and
    // stored by the engine, which needs it to keep the KV cache consistent with
    // what it returns. It is not part of the answer, so drop it before
    // detokenizing and before reporting completion_tokens. A structured-output
    // terminal token is also reported as "stop", but it is part of the JSON and
    // must be preserved.
    static std::vector<int> strip_stop_token(
        const std::vector<int>& tokens, const std::string& finish_reason,
        bool preserve_terminal_token = false) {
        if (preserve_terminal_token || finish_reason != "stop" || tokens.empty()) {
            return tokens;
        }
        return std::vector<int>(tokens.begin(), tokens.end() - 1);
    }

    void handle_completions_nonstream(
        const EncodeReply& enc, const BatchSamplingParams& sp,
        const std::vector<std::shared_ptr<TokenConstraint>>& constraints,
        const std::vector<std::string>& stops,
        const std::string& client_id,
        std::chrono::steady_clock::time_point request_start,
        httplib::Response& res) {
        std::vector<ChoiceRun> runs;
        if (!submit_choices(enc.token_ids, sp, constraints, nullptr, runs)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 503,
                       "request rejected: its worst-case KV footprint exceeds the "
                       "whole block pool, so it could never be admitted");
            return;
        }
        track_request(client_id, runs);

        std::vector<ChoiceText> texts;
        std::size_t completion_tokens = 0;
        const bool answered = finish_choices(res, runs, stops, enc.token_ids, client_id,
                                             request_start, texts, completion_tokens);
        untrack_request(client_id);
        if (!answered) return;

        std::ostringstream os;
        os << "{\"id\":\"" << json_escape(client_id) << "\""
           << ",\"object\":\"text_completion\""
           << ",\"created\":" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
           << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
           << ",\"choices\":[";
        for (std::size_t i = 0; i < texts.size(); ++i) {
            if (i > 0) os << ",";
            os << "{\"index\":" << texts[i].index
               << ",\"finish_reason\":\"" << texts[i].finish_reason << "\""
               << ",\"text\":\"" << json_escape(texts[i].text) << "\"}";
        }
        os << "],\"usage\":{\"prompt_tokens\":" << enc.token_ids.size()
           << ",\"completion_tokens\":" << completion_tokens
           << ",\"total_tokens\":" << (enc.token_ids.size() + completion_tokens) << "}}";
        res.set_content(os.str(), "application/json");
    }

    void handle_completions_stream(
        const EncodeReply& enc, const BatchSamplingParams& sp,
        const std::vector<std::shared_ptr<TokenConstraint>>& constraints,
        const std::vector<std::string>& stops,
        const std::string& client_id,
        std::chrono::steady_clock::time_point request_start,
        httplib::Response& res) {
        const long long created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [this, enc, sp, constraints, stops, client_id, created, request_start]
            (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {

            const std::size_t choices = constraints.size();
            std::chrono::steady_clock::time_point ttft_time;
            bool ttft_recorded = false;

            // Every chunk names the choice it carries, so a client reading
            // several answers out of one stream can tell them apart.
            auto send_chunk = [&](std::size_t index, const std::string& text,
                                  const char* finish_reason = nullptr) {
                std::ostringstream os;
                os << "{\"id\":\"" << json_escape(client_id) << "\",\"object\":\"text_completion\""
                   << ",\"created\":" << created << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
                   << ",\"choices\":[{\"index\":" << index
                   << ",\"text\":\"" << json_escape(text) << "\"";
                if (finish_reason != nullptr) {
                    os << ",\"finish_reason\":\"" << finish_reason << "\"";
                } else {
                    os << ",\"finish_reason\":null";
                }
                os << "}]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            };

            auto send_error = [&](const std::string& message) {
                std::ostringstream os;
                os << "{\"error\":{\"message\":\"" << json_escape(message)
                   << "\",\"type\":\"server_error\"}}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            };

            // A message about one choice of several says which one; a request
            // that asked for one choice keeps the request-level wording.
            auto describe = [&](std::size_t index, const std::string& message) {
                if (choices == 1) return message;
                return "choice " + std::to_string(index) + ": " + message;
            };

            const std::shared_ptr<ChoiceGroup> group = std::make_shared<ChoiceGroup>();
            std::vector<ChoiceRun> runs;
            if (!submit_choices(enc.token_ids, sp, constraints, group, runs)) {
                const auto request_end = std::chrono::steady_clock::now();
                const double duration = std::chrono::duration<double>(request_end - request_start).count();
                metrics.record_request_end(false, duration, 0.0, enc.token_ids.size(), 0);
                send_error("request rejected: its worst-case KV footprint exceeds "
                           "the whole block pool, so it could never be admitted");
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            }
            track_request(client_id, runs);

            std::vector<ChoiceStream> state(choices);
            int token_count = 0;

            auto emit_delta = [&](std::size_t index) {
                ChoiceStream& s = state[index];
                const std::string full = tok.decode_tokens(s.generated);
                // Deliverable-now: everything before a matched sequence, or
                // everything except a trailing partial sequence, which the next
                // token may complete. Text already sent cannot be taken back, so
                // the half-formed sequence is held rather than streamed and
                // retracted.
                const StopScan scan = scan_stop_strings(full, stops);
                if (scan.matched) s.stop_matched = true;
                if (scan.safe_length <= s.sent_offset) return;
                std::string candidate =
                    full.substr(s.sent_offset, scan.safe_length - s.sent_offset);
                auto [complete, leftover] = split_utf8_complete(candidate);
                if (!complete.empty()) {
                    s.sent_offset += complete.size();
                    send_chunk(index, complete);
                }
            };

            // Round-robin over the choices: drain whatever each one has queued,
            // then wait for any of them to move. Waiting on one choice at a time
            // would hold the others' chunks -- already produced -- until it
            // finished, which is a batch download wearing a stream's clothes.
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(cfg.request_timeout_seconds);
            bool timed_out = false;
            uint64_t seen = 0;
            while (true) {
                bool all_finished = true;
                for (std::size_t i = 0; i < choices; ++i) {
                    if (state[i].finished) continue;
                    int token = 0;
                    while (runs[i].stream->try_pop(&token)) {
                        if (!ttft_recorded) {
                            ttft_time = std::chrono::steady_clock::now();
                            ttft_recorded = true;
                        }
                        state[i].generated.push_back(token);
                        emit_delta(i);
                        ++token_count;
                    }
                    if (runs[i].stream->is_finished()) {
                        state[i].finished = true;
                        continue;
                    }
                    all_finished = false;
                }
                if (all_finished) break;
                if (!group->wait_for_activity(seen, deadline)) {
                    timed_out = true;
                    break;
                }
            }

            // Whatever arrived is taken, including anything the last token
            // queued before the deadline: those bytes are part of the answer.
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].finished) continue;
                int token = 0;
                while (runs[i].stream->try_pop(&token)) {
                    state[i].generated.push_back(token);
                    ++token_count;
                }
                if (runs[i].stream->is_finished()) {
                    state[i].finished = true;
                } else {
                    // The deadline belongs to the request, so a choice still
                    // running when the group stops waiting has timed out -- and
                    // one choice timing out does not discard the others.
                    state[i].timed_out = true;
                    state[i].finish_reason = "timeout";
                    sched.cancel_request(runs[i].request_id);
                }
            }

            const auto request_end = std::chrono::steady_clock::now();
            const double duration = std::chrono::duration<double>(request_end - request_start).count();
            const double ttft = ttft_recorded ? std::chrono::duration<double>(ttft_time - request_start).count() : 0.0;

            bool any_error = false;
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].timed_out) continue;
                std::lock_guard<std::mutex> lk(runs[i].stream->m);
                state[i].finish_reason = runs[i].stream->result.finish_reason;
                state[i].error = runs[i].stream->result.error;
                if (!state[i].error.empty()) any_error = true;
            }

            // Generation has ended, so a trailing partial sequence can no longer
            // complete: those bytes are part of the answer and are flushed rather
            // than withheld. A match still truncates.
            for (std::size_t i = 0; i < choices; ++i) {
                ChoiceStream& s = state[i];
                const std::string full = tok.decode_tokens(s.generated);
                const StopScan scan = scan_stop_strings(full, stops);
                if (scan.matched) s.stop_matched = true;
                const std::size_t deliverable = std::max(scan.final_length, s.sent_offset);
                if (deliverable > s.sent_offset) {
                    std::string tail = full.substr(s.sent_offset, deliverable - s.sent_offset);
                    s.sent_offset += tail.size();
                    send_chunk(i, tail);
                }
                // The scheduler saw no stop token, so it reports "length" even
                // when the answer it produced was cut short at a client sequence.
                if (s.stop_matched && s.finish_reason != "timeout") s.finish_reason = "stop";
            }

            const bool success = !timed_out && !any_error;
            metrics.record_request_end(success, duration, ttft, enc.token_ids.size(), token_count);
            untrack_request(client_id);

            if (timed_out) send_error("generation timed out");
            for (std::size_t i = 0; i < choices; ++i) {
                if (!state[i].error.empty()) send_error(describe(i, state[i].error));
            }
            // A choice that failed has no answer to terminate, so a request whose
            // choices all failed ends after the error events above -- which is
            // exactly how it ended before choices existed.
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].error.empty()) {
                    send_chunk(i, "", state[i].finish_reason.c_str());
                }
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }

    void handle_nonstream(httplib::Response& res, const EncodeReply& enc,
                          const BatchSamplingParams& sp,
                          const std::vector<std::shared_ptr<TokenConstraint>>& constraints,
                          const std::vector<std::string>& stops,
                          const std::string& thinking_mode,
                          const std::string& client_id,
                          std::chrono::steady_clock::time_point request_start) {
        std::vector<ChoiceRun> runs;
        if (!submit_choices(enc.token_ids, sp, constraints, nullptr, runs)) {
            metrics.record_request_end(false, 0.0, 0.0, 0, 0);
            emit_error(res, 503,
                       "request rejected: its worst-case KV footprint exceeds the "
                       "whole block pool, so it could never be admitted");
            return;
        }
        track_request(client_id, runs);

        std::vector<ChoiceText> texts;
        std::size_t completion_tokens = 0;
        const bool answered = finish_choices(res, runs, stops, enc.token_ids, client_id,
                                             request_start, texts, completion_tokens);
        untrack_request(client_id);
        if (!answered) return;

        std::ostringstream os;
        os << "{\"id\":\"" << json_escape(client_id) << "\""
           << ",\"object\":\"chat.completion\""
           << ",\"created\":" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
           << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
           << ",\"choices\":[";
        for (std::size_t i = 0; i < texts.size(); ++i) {
            // Truncated before the sidecar sees them: the parser splits the text
            // into content / reasoning / tool calls, and a stop sequence that
            // lands inside a tool call would otherwise be parsed as part of one.
            const ParsedMessage parsed = sidecar.parse(texts[i].text, thinking_mode);
            const std::string content = parsed.ok ? parsed.content : texts[i].text;
            const std::string reasoning = parsed.ok ? parsed.reasoning : std::string();
            const std::string tool_calls_json = parsed.ok ? parsed.tool_calls_json : "[]";
            if (i > 0) os << ",";
            os << "{\"index\":" << texts[i].index
               << ",\"finish_reason\":\"" << texts[i].finish_reason << "\""
               << ",\"message\":"
               << render_choice_message(content, reasoning, tool_calls_json) << "}";
        }
        os << "],\"usage\":{\"prompt_tokens\":" << enc.token_ids.size()
           << ",\"completion_tokens\":" << completion_tokens
           << ",\"total_tokens\":" << (enc.token_ids.size() + completion_tokens) << "}}";
        res.set_content(os.str(), "application/json");
    }

    void handle_stream(const httplib::Request& /*req*/, httplib::Response& res,
                       const EncodeReply& enc, const BatchSamplingParams& sp,
                       const std::vector<std::shared_ptr<TokenConstraint>>& constraints,
                       const std::vector<std::string>& stops,
                       const std::string& thinking_mode, const std::string& client_id,
                       std::chrono::steady_clock::time_point request_start) {
        const long long created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [this, enc, sp, constraints, stops, client_id, created, thinking_mode, request_start]
            (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {

            const std::size_t choices = constraints.size();
            std::chrono::steady_clock::time_point ttft_time;
            bool ttft_recorded = false;

            // Every chunk names the choice it carries, so a client reading
            // several answers out of one stream can tell them apart.
            auto send_chunk = [&](std::size_t index, const std::string& delta,
                                  const char* field, const char* finish_reason = nullptr) {
                std::ostringstream os;
                os << "{\"id\":\"" << json_escape(client_id) << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
                   << ",\"choices\":[{\"index\":" << index << ",\"delta\":{";
                if (!delta.empty()) {
                    os << "\"" << field << "\":\"" << json_escape(delta) << "\"";
                }
                os << "}";
                if (finish_reason != nullptr) {
                    os << ",\"finish_reason\":\"" << finish_reason << "\"";
                } else {
                    os << ",\"finish_reason\":null";
                }
                os << "}]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            };

            auto send_error = [&](const std::string& message) {
                std::ostringstream os;
                os << "{\"error\":{\"message\":\"" << json_escape(message)
                   << "\",\"type\":\"server_error\"}}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            };

            // A message about one choice of several says which one; a request
            // that asked for one choice keeps the request-level wording.
            auto describe = [&](std::size_t index, const std::string& message) {
                if (choices == 1) return message;
                return "choice " + std::to_string(index) + ": " + message;
            };

            // First chunk: one role marker per choice. They share a chunk because
            // they arrive at the same moment and a client that reads them in one
            // parse is no worse off than one that reads them in several.
            {
                std::ostringstream os;
                os << "{\"id\":\"" << json_escape(client_id) << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
                   << ",\"choices\":[";
                for (std::size_t i = 0; i < choices; ++i) {
                    if (i > 0) os << ",";
                    os << "{\"index\":" << i
                       << ",\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}";
                }
                os << "]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            }

            const std::shared_ptr<ChoiceGroup> group = std::make_shared<ChoiceGroup>();
            std::vector<ChoiceRun> runs;
            if (!submit_choices(enc.token_ids, sp, constraints, group, runs)) {
                const auto request_end = std::chrono::steady_clock::now();
                const double duration = std::chrono::duration<double>(request_end - request_start).count();
                metrics.record_request_end(false, duration, 0.0, enc.token_ids.size(), 0);
                send_error("request rejected: its worst-case KV footprint exceeds "
                           "the whole block pool, so it could never be admitted");
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            }
            track_request(client_id, runs);

            // Sliding-window prefix used to compute deltas + UTF-8 boundary buffer,
            // plus the per-choice reasoning state. In thinking mode the prompt ends
            // with <think>, so a completion starts inside the reasoning block and
            // switches to the answer at the first </think>. Splitting on the token
            // id rather than the decoded text keeps each stream's UTF-8 decoding
            // self-contained and cannot be fooled by a model that writes the
            // literal characters.
            std::vector<ChoiceStream> state(choices);
            for (ChoiceStream& s : state) {
                s.in_reasoning = (thinking_mode == "thinking");
            }
            int token_count = 0;

            auto emit_delta = [&](std::size_t index) {
                ChoiceStream& s = state[index];
                const std::string full = tok.decode_tokens(s.generated);
                // Stop sequences apply to the answer only. The reasoning block
                // is a separate field that ends on a token id, and truncating it
                // mid-block would drop the answer that follows.
                std::size_t deliverable = full.size();
                if (!s.in_reasoning) {
                    // Deliverable-now: everything before a matched sequence, or
                    // everything except a trailing partial sequence, which the
                    // next token may complete. Text already sent cannot be taken
                    // back, so the half-formed sequence is held rather than
                    // streamed and retracted.
                    const StopScan scan = scan_stop_strings(full, stops);
                    if (scan.matched) s.stop_matched = true;
                    deliverable = scan.safe_length;
                }
                if (deliverable <= s.sent_offset) return;
                std::string candidate = full.substr(s.sent_offset, deliverable - s.sent_offset);
                auto [complete, leftover] = split_utf8_complete(candidate);
                if (!complete.empty()) {
                    s.sent_offset += complete.size();
                    send_chunk(index, complete,
                               s.in_reasoning ? "reasoning_content" : "content");
                }
            };

            // Closing the reasoning block restarts the delta window: the marker
            // itself belongs to neither stream, and dropping the tokens before it
            // keeps decode_tokens() cheap on long generations.
            auto accept = [&](std::size_t index, int tok_id) {
                ChoiceStream& s = state[index];
                if (s.in_reasoning && tok_id == think_end_id) {
                    emit_delta(index);
                    s.generated.clear();
                    s.sent_offset = 0;
                    s.in_reasoning = false;
                    return;
                }
                s.generated.push_back(tok_id);
                emit_delta(index);
            };

            // Round-robin over the choices: drain whatever each one has queued,
            // then wait for any of them to move. Waiting on one choice at a time
            // would hold the others' chunks -- already produced -- until it
            // finished, which is a batch download wearing a stream's clothes.
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(cfg.request_timeout_seconds);
            bool timed_out = false;
            uint64_t seen = 0;
            while (true) {
                bool all_finished = true;
                for (std::size_t i = 0; i < choices; ++i) {
                    if (state[i].finished) continue;
                    int token = 0;
                    while (runs[i].stream->try_pop(&token)) {
                        if (!ttft_recorded) {
                            ttft_time = std::chrono::steady_clock::now();
                            ttft_recorded = true;
                        }
                        accept(i, token);
                        ++token_count;
                    }
                    if (runs[i].stream->is_finished()) {
                        state[i].finished = true;
                        continue;
                    }
                    all_finished = false;
                }
                if (all_finished) break;
                if (!group->wait_for_activity(seen, deadline)) {
                    timed_out = true;
                    break;
                }
            }

            // Whatever arrived is taken, including anything the last token
            // queued before the deadline: those bytes are part of the answer.
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].finished) continue;
                int token = 0;
                while (runs[i].stream->try_pop(&token)) {
                    accept(i, token);
                    ++token_count;
                }
                if (runs[i].stream->is_finished()) {
                    state[i].finished = true;
                } else {
                    // The deadline belongs to the request, so a choice still
                    // running when the group stops waiting has timed out -- and
                    // one choice timing out does not discard the others.
                    state[i].timed_out = true;
                    state[i].finish_reason = "timeout";
                    sched.cancel_request(runs[i].request_id);
                }
            }

            const auto request_end = std::chrono::steady_clock::now();
            const double duration = std::chrono::duration<double>(request_end - request_start).count();
            const double ttft = ttft_recorded ? std::chrono::duration<double>(ttft_time - request_start).count() : 0.0;

            bool any_error = false;
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].timed_out) continue;
                std::lock_guard<std::mutex> lk(runs[i].stream->m);
                state[i].finish_reason = runs[i].stream->result.finish_reason;
                state[i].error = runs[i].stream->result.error;
                if (!state[i].error.empty()) any_error = true;
                // Stop tokens are not queued by the scheduler, so they can never
                // contribute bytes to the stream even when a caller supplies a
                // custom, non-special stop id.
            }

            // Flush any remaining tail bytes (e.g. an isolated partial sequence
            // at EOS). In practice decode_tokens at terminal state produces
            // valid UTF-8, so this is usually empty.
            for (std::size_t i = 0; i < choices; ++i) {
                ChoiceStream& s = state[i];
                const std::string full = tok.decode_tokens(s.generated);
                std::size_t deliverable = full.size();
                if (!s.in_reasoning) {
                    // Generation has ended, so a trailing partial sequence can
                    // no longer complete: those bytes are part of the answer and
                    // are flushed rather than withheld. A match still truncates.
                    const StopScan scan = scan_stop_strings(full, stops);
                    if (scan.matched) s.stop_matched = true;
                    deliverable = std::max(scan.final_length, s.sent_offset);
                }
                if (deliverable > s.sent_offset) {
                    std::string tail = full.substr(s.sent_offset, deliverable - s.sent_offset);
                    s.sent_offset += tail.size();
                    send_chunk(i, tail, s.in_reasoning ? "reasoning_content" : "content");
                }
                // The scheduler saw no stop token, so it reports "length" even
                // when the answer it produced was cut short at a client sequence.
                if (s.stop_matched && s.finish_reason != "timeout") s.finish_reason = "stop";
            }

            const bool success = !timed_out && !any_error;
            metrics.record_request_end(success, duration, ttft, enc.token_ids.size(), token_count);
            untrack_request(client_id);

            if (timed_out) send_error("generation timed out");
            for (std::size_t i = 0; i < choices; ++i) {
                if (!state[i].error.empty()) send_error(describe(i, state[i].error));
            }
            // A choice that failed has no answer to terminate, so a request whose
            // choices all failed ends after the error events above -- which is
            // exactly how it ended before choices existed.
            for (std::size_t i = 0; i < choices; ++i) {
                if (state[i].error.empty()) {
                    send_chunk(i, "", "content", state[i].finish_reason.c_str());
                }
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }

    void run() {
        running = true;
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
        svr.Get("/ready", [this](const httplib::Request&, httplib::Response& res) { handle_ready(res); });
        svr.Get("/alive", [this](const httplib::Request&, httplib::Response& res) { handle_alive(res); });
        svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) { handle_models(res); });
        svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) { handle_metrics(res); });
        svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try { handle_chat_completions(req, res); }
            catch (const std::exception& ex) { emit_error(res, 500, ex.what()); }
        });
        svr.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try { handle_completions(req, res); }
            catch (const std::exception& ex) { emit_error(res, 500, ex.what()); }
        });
        svr.Delete("/v1/requests/:id", [this](const httplib::Request& req, httplib::Response& res) {
            try { handle_cancel_request(req, res); }
            catch (const std::exception& ex) { emit_error(res, 500, ex.what()); }
        });
        std::cerr << "[server] listening on " << cfg.host << ":" << cfg.port << "\n";
        svr.listen(cfg.host.c_str(), cfg.port);
        running = false;
    }

    void stop() {
        if (running) svr.stop();
    }
};

OpenAIServer::OpenAIServer(InferenceEngine& engine, const Tokenizer& tokenizer,
                           PythonSidecar& sidecar, const OpenAIServerConfig& cfg)
    : impl_(std::make_unique<Impl>(engine, tokenizer, sidecar, cfg)) {}
OpenAIServer::~OpenAIServer() = default;
void OpenAIServer::run() { impl_->run(); }
void OpenAIServer::stop() { impl_->stop(); }

}  // namespace pocket
