#include "runtime/session.h"

#include "runtime/status.h"

namespace pocketllm {

Session::Session(std::string gguf_path, std::string backend, std::unique_ptr<GgufReader> checkpoint,
                 std::unique_ptr<Tokenizer> tokenizer)
    : gguf_path_(std::move(gguf_path)),
      backend_(std::move(backend)),
      checkpoint_(std::move(checkpoint)),
      tokenizer_(std::move(tokenizer)) {}

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
  return std::unique_ptr<Session>(
      new Session(gguf_path, device, std::move(checkpoint), std::move(tokenizer)));
}

void Session::reset() { position_ = 0; }

}  // namespace pocketllm