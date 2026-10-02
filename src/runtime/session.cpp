#include "runtime/session.h"

#include <utility>

#include "runtime/status.h"

namespace pocketllm {

Session::Session(std::string gguf_path, std::string backend, std::unique_ptr<GgufReader> checkpoint,
                 std::unique_ptr<Tokenizer> tokenizer, std::unique_ptr<Qwen3Model> model)
    : gguf_path_(std::move(gguf_path)),
      backend_(std::move(backend)),
      checkpoint_(std::move(checkpoint)),
      tokenizer_(std::move(tokenizer)),
      model_(std::move(model)) {}

Session::~Session() = default;

std::unique_ptr<Session> Session::open(const std::string &gguf_path, const std::string &backend) {
  /* "cpu" is the only backend this build has.  An empty name means the caller
   * took the default, which the header documents as "cpu"; anything else is a
   * backend that exists in the Python declarations but has no C implementation
   * yet, and saying so is more useful than pretending the request was honoured. */
  std::string device = backend.empty() ? "cpu" : backend;
  if (device != "cpu") {
    throw Error("backend '" + device +
                "' has no implementation in this build; the C engine currently "
                "provides 'cpu' only");
  }

  /* The stat check is gone: `GgufReader` opens the file itself and reports a
   * missing one with the same errno message, so checking first would only be a
   * second place that can disagree about what "opens" means. */
  auto checkpoint = std::make_unique<GgufReader>(gguf_path);
  auto tokenizer = std::make_unique<Tokenizer>(*checkpoint);

  /* The model is optional in a way the reader and the tokenizer are not, and
   * the architecture, not the parse, is what decides: `GgufReader` reads any
   * well-formed GGUF, so a checkpoint of an architecture this build has no
   * graph for is a *successful* open that cannot run.  That is the state every
   * other architecture is in, and turning it into a failed open would make
   * `encode` -- which works fine -- unreachable for them. */
  std::unique_ptr<Qwen3Model> model;
  if (checkpoint->get_string("general.architecture", "") == "qwen3") {
    model = Qwen3Model::load(*checkpoint);
  }

  return std::unique_ptr<Session>(
      new Session(gguf_path, device, std::move(checkpoint), std::move(tokenizer), std::move(model)));
}

const float *Session::forward(const int32_t *tokens, int64_t n) {
  if (model_ == nullptr) {
    throw Error("this checkpoint is architecture '" +
                checkpoint_->get_string("general.architecture", "?") +
                "', which this build has no graph for; only 'qwen3' runs");
  }
  const float *logits = model_->forward(tokens, n, position_);
  position_ = model_->cache_length();
  return logits;
}

void Session::reset() {
  position_ = 0;
  if (model_ != nullptr) {
    model_->reset();
  }
}

}  // namespace pocketllm