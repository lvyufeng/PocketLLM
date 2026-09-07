#pragma once

#include "inference_engine.hpp"
#include "python_sidecar.hpp"
#include "tokenizer.hpp"

#include <memory>
#include <string>

namespace pocket {

struct OpenAIServerConfig {
    int port = 8000;
    std::string host = "0.0.0.0";
    std::string default_thinking_mode = "chat";
    int default_max_tokens = 256;
    bool log_requests = true;
    // Reported by /v1/models and echoed in every completion.
    std::string model_name = "pocketllm";
    // Requests the scheduler may run concurrently. Clamped down to whatever the
    // engine declares in caps().max_slots, so the DeepSeek-V4 adapter still
    // serialises at 1 while a multi-slot engine runs the full width.
    int max_batch_size = 8;
    // Prompt tokens a request may advance per schedule iteration; 0 disables
    // chunking. Ignored by an engine that does not declare chunked prefill.
    int prefill_token_budget = 4096;
    // How long a single completion may take before the connection is failed.
    // Generous by default: a 65K prompt at TP4 spends seconds in prefill alone,
    // and a queued request also waits behind everything admitted before it.
    int request_timeout_seconds = 900;
};

// OpenAI-compatible HTTP server over any InferenceEngine.
//
// Requests are driven through BatchScheduler rather than by calling the engine
// directly, which is what makes concurrency the engine's decision instead of the
// server's: an engine declaring continuous batching runs several completions at
// once, and one declaring a single session has the scheduler serialise them.
// Either way the server issues no worker_command_* itself -- the scheduler's
// batched entry points announce each forward to the TP group.
//
// The tokenizer is the server's, not the engine's. Detokenizing a response is a
// serving concern, and requiring every model runtime to carry a Tokenizer would
// put one on engines that have no use for it.
class OpenAIServer {
public:
    OpenAIServer(InferenceEngine& engine, const Tokenizer& tokenizer,
                 PythonSidecar& sidecar, const OpenAIServerConfig& cfg);
    ~OpenAIServer();

    // Blocks until stop() is called or the listener fails.
    void run();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pocket
