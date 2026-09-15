#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pocket {

struct EncodeRequest {
    std::string messages_json;        // JSON array of {role, content, ...}
    std::string thinking_mode = "chat";
    std::string reasoning_effort;     // empty = none
    std::string tools_json;           // JSON array of tools or empty
    bool add_generation_prompt = true;
    bool drop_thinking = true;
};

struct TokenizeRequest {
    std::string prompt;              // Raw text to tokenize
};

struct EncodeReply {
    bool ok = false;
    std::string err;
    std::string prompt_text;
    std::vector<int> token_ids;
    // Echoed back from the request so the handler can hand the parser the same
    // array the chat template was given, without reading the body a second time.
    std::string tools_json;
};

struct TokenizeReply {
    bool ok = false;
    std::string err;
    std::vector<int> token_ids;
};

struct ParsedMessage {
    bool ok = false;
    std::string err;
    std::string content;
    std::string reasoning;
    std::string tool_calls_json;      // raw JSON array string, may be "[]"
};

// Thin C++ wrapper around a long-running Python helper. The helper selects the
// checkpoint's chat template, tokenizes requests, and parses generated text into
// structured fields. One sidecar instance per server process; calls are
// serialised via the internal mutex.
class PythonSidecar {
public:
    PythonSidecar(const std::string& python_bin,
                  const std::string& script_path,
                  const std::string& ckpt_dir,
                  const std::string& architecture = "");
    ~PythonSidecar();

    PythonSidecar(const PythonSidecar&) = delete;
    PythonSidecar& operator=(const PythonSidecar&) = delete;

    int eos_token_id() const { return eos_token_id_; }

    EncodeReply encode(const EncodeRequest& req);
    TokenizeReply tokenize(const TokenizeRequest& req);
    // `tools_json` is the same array the request carried, or empty.  It reaches
    // the parser because the tool-call syntaxes that carry no type information
    // of their own (Qwen's XML) can only be read back against the schema the
    // caller declared; a template whose syntax is self-describing ignores it.
    ParsedMessage parse(const std::string& text, const std::string& thinking_mode,
                        const std::string& tools_json = std::string());

private:
    std::string send_request(const std::string& json_line);
    void shutdown();

    int child_pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;
    int eos_token_id_ = 1;
    std::mutex mu_;
};

}  // namespace pocket
