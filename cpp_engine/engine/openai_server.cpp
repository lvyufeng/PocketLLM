#include "openai_server.hpp"

#include "batch_scheduler.hpp"
#include "json_lite.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pocket {

namespace {

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

std::string make_request_id() {
    using clock = std::chrono::steady_clock;
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
    std::ostringstream os;
    os << "chatcmpl-cpp-" << std::hex << t;
    return os.str();
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

    void handle_models(httplib::Response& res) {
        std::ostringstream os;
        os << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(cfg.model_name)
           << "\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
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
        return true;
    }

    bool encode_request(const std::string& body, const JsonObject& obj,
                        EncodeReply& out, std::string& thinking_mode_out,
                        int& max_tokens_out, bool& stream_out,
                        BatchSamplingParams& sp_out, std::string& err_out) {
        if (!check_sampling_supported(obj, err_out)) return false;

        EncodeRequest enc;
        enc.messages_json = extract_messages_json(body);
        enc.tools_json = extract_tools_json(body);
        thinking_mode_out = get_string(obj, "thinking_mode", cfg.default_thinking_mode);
        enc.thinking_mode = thinking_mode_out;
        enc.reasoning_effort = get_string(obj, "reasoning_effort", "");
        enc.add_generation_prompt = get_bool(obj, "add_generation_prompt", true);
        enc.drop_thinking = get_bool(obj, "drop_thinking", true);

        max_tokens_out = static_cast<int>(get_number(obj, "max_tokens", cfg.default_max_tokens));
        if (max_tokens_out <= 0) max_tokens_out = cfg.default_max_tokens;
        stream_out = get_bool(obj, "stream", false);

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

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        JsonValue jv;
        try { jv = parse_json(body); } catch (const std::exception& ex) {
            emit_error(res, 400, std::string("invalid JSON: ") + ex.what());
            return;
        }
        if (!jv.is_object()) {
            emit_error(res, 400, "request body must be JSON object");
            return;
        }
        const auto& obj = jv.object();
        EncodeReply enc_reply;
        std::string thinking_mode;
        int max_tokens = 0;
        bool stream = false;
        BatchSamplingParams sp;
        std::string err;
        if (!encode_request(body, obj, enc_reply, thinking_mode, max_tokens, stream, sp, err)) {
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
            emit_error(res, 400, "prompt + max_tokens exceeds max_context");
            return;
        }

        if (stream) {
            handle_stream(req, res, enc_reply, sp, thinking_mode);
        } else {
            handle_nonstream(res, enc_reply, sp, thinking_mode);
        }
    }

    // A stop token is the last token of a sequence and is still counted and
    // stored by the engine, which needs it to keep the KV cache consistent with
    // what it returns. It is not part of the answer, so drop it before
    // detokenizing and before reporting completion_tokens.
    static std::vector<int> strip_stop_token(const std::vector<int>& tokens,
                                             const std::string& finish_reason) {
        if (finish_reason != "stop" || tokens.empty()) return tokens;
        return std::vector<int>(tokens.begin(), tokens.end() - 1);
    }

    void handle_nonstream(httplib::Response& res, const EncodeReply& enc,
                          const BatchSamplingParams& sp, const std::string& thinking_mode) {
        auto completion = std::make_shared<TokenStream>();
        const uint64_t request_id = sched.submit_request(
            enc.token_ids, sp,
            [completion](const SchedulerGenerationResult& r) { completion->finish(r); });
        if (request_id == 0) {
            emit_error(res, 503,
                       "request rejected: its worst-case KV footprint exceeds the "
                       "whole block pool, so it could never be admitted");
            return;
        }

        int unused_token = 0;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(cfg.request_timeout_seconds);
        if (completion->next(&unused_token, deadline) == TokenStream::Next::Timeout) {
            // The callback owns `completion`, so it remains valid until the
            // scheduler observes the cancellation. Because callback-mode
            // requests are not duplicated into poll_result(), abandoning this
            // HTTP response leaves no completed-result map entry behind.
            sched.cancel_request(request_id);
            emit_error(res, 504, "generation timed out");
            return;
        }

        SchedulerGenerationResult out;
        {
            std::lock_guard<std::mutex> lk(completion->m);
            out = completion->result;
        }

        if (!out.error.empty()) {
            emit_error(res, 500, out.error);
            return;
        }

        const std::vector<int> generated = strip_stop_token(out.generated_tokens, out.finish_reason);
        const std::string text = tok.decode_tokens(generated);
        const ParsedMessage parsed = sidecar.parse(text, thinking_mode);
        std::string content = parsed.ok ? parsed.content : text;
        std::string reasoning = parsed.ok ? parsed.reasoning : std::string();
        std::string tool_calls_json = parsed.ok ? parsed.tool_calls_json : "[]";

        std::ostringstream os;
        os << "{\"id\":\"" << make_request_id() << "\""
           << ",\"object\":\"chat.completion\""
           << ",\"created\":" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
           << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
           << ",\"choices\":[{\"index\":0,\"finish_reason\":\"" << out.finish_reason << "\""
           << ",\"message\":" << render_choice_message(content, reasoning, tool_calls_json) << "}]"
           << ",\"usage\":{\"prompt_tokens\":" << enc.token_ids.size()
           << ",\"completion_tokens\":" << generated.size()
           << ",\"total_tokens\":" << (enc.token_ids.size() + generated.size()) << "}}";
        res.set_content(os.str(), "application/json");
    }

    void handle_stream(const httplib::Request& /*req*/, httplib::Response& res,
                       const EncodeReply& enc, const BatchSamplingParams& sp,
                       const std::string& thinking_mode) {
        const std::string id = make_request_id();
        const long long created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [this, enc, sp, id, created, thinking_mode]
            (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {

            auto send_chunk = [&](const std::string& delta, const char* field,
                                  const char* finish_reason = nullptr) {
                std::ostringstream os;
                os << "{\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
                   << ",\"choices\":[{\"index\":0,\"delta\":{";
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

            // First chunk: role marker.
            {
                std::ostringstream os;
                os << "{\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << json_escape(cfg.model_name) << "\""
                   << ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            }

            auto stream = std::make_shared<TokenStream>();
            const uint64_t request_id = sched.submit_request(
                enc.token_ids, sp,
                [stream](const SchedulerGenerationResult& r) { stream->finish(r); },
                [stream](uint64_t, int token) { stream->push(token); });
            if (request_id == 0) {
                send_error("request rejected: its worst-case KV footprint exceeds "
                           "the whole block pool, so it could never be admitted");
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            }

            // Sliding-window prefix used to compute deltas + UTF-8 boundary buffer.
            std::vector<int> generated;
            size_t sent_offset = 0;  // bytes of decode_tokens(generated) already sent

            // In thinking mode the prompt ends with <think>, so the completion
            // starts inside the reasoning block and switches to the answer at the
            // first </think>. Splitting on the token id rather than the decoded
            // text keeps each stream's UTF-8 decoding self-contained and cannot be
            // fooled by a model that writes the literal characters.
            bool in_reasoning = (thinking_mode == "thinking");

            auto emit_delta = [&]() {
                const std::string full = tok.decode_tokens(generated);
                if (full.size() <= sent_offset) return;
                std::string candidate = full.substr(sent_offset);
                auto [complete, leftover] = split_utf8_complete(candidate);
                if (!complete.empty()) {
                    sent_offset += complete.size();
                    send_chunk(complete, in_reasoning ? "reasoning_content" : "content");
                }
            };

            // Closing the reasoning block restarts the delta window: the marker
            // itself belongs to neither stream, and dropping the tokens before it
            // keeps decode_tokens() cheap on long generations.
            auto accept = [&](int tok_id) {
                if (in_reasoning && tok_id == think_end_id) {
                    emit_delta();
                    generated.clear();
                    sent_offset = 0;
                    in_reasoning = false;
                    return;
                }
                generated.push_back(tok_id);
                emit_delta();
            };

            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(cfg.request_timeout_seconds);
            bool timed_out = false;
            while (true) {
                int token = 0;
                const TokenStream::Next next = stream->next(&token, deadline);
                if (next == TokenStream::Next::Token) {
                    accept(token);
                    continue;
                }
                if (next == TokenStream::Next::Timeout) {
                    timed_out = true;
                    sched.cancel_request(request_id);
                }
                break;
            }

            std::string finish_reason = "length";
            std::string generation_error;
            if (timed_out) {
                finish_reason = "timeout";
            } else {
                std::lock_guard<std::mutex> lk(stream->m);
                finish_reason = stream->result.finish_reason;
                generation_error = stream->result.error;
                // Stop tokens are not queued by the scheduler, so they can never
                // contribute bytes to the stream even when a caller supplies a
                // custom, non-special stop id.
            }

            // Flush any remaining tail bytes (e.g. an isolated partial sequence
            // at EOS). In practice decode_tokens at terminal state produces
            // valid UTF-8, so this is usually empty.
            {
                const std::string full = tok.decode_tokens(generated);
                if (full.size() > sent_offset) {
                    std::string tail = full.substr(sent_offset);
                    sent_offset += tail.size();
                    send_chunk(tail, in_reasoning ? "reasoning_content" : "content");
                }
            }

            if (timed_out) send_error("generation timed out");
            if (!generation_error.empty()) {
                send_error(generation_error);
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            }

            // Final chunk with finish_reason.
            send_chunk("", "content", finish_reason.c_str());
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }

    void run() {
        running = true;
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
        svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) { handle_models(res); });
        svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try { handle_chat_completions(req, res); }
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
