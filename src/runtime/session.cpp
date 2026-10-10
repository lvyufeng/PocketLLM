#include "runtime/session.h"

#include <utility>
#include <vector>

#include "kernel/backend.h"
#include "runtime/status.h"

namespace pocketllm {

Session::Session(std::string gguf_path, std::string backend_name,
                 std::unique_ptr<kernel::Backend> backend, std::unique_ptr<GgufReader> checkpoint,
                 std::unique_ptr<Tokenizer> tokenizer, std::unique_ptr<Qwen3Model> model)
    : gguf_path_(std::move(gguf_path)),
      backend_name_(std::move(backend_name)),
      backend_(std::move(backend)),
      checkpoint_(std::move(checkpoint)),
      tokenizer_(std::move(tokenizer)),
      model_(std::move(model)) {}

Session::~Session() = default;

std::unique_ptr<Session> Session::open(const std::string &gguf_path, const std::string &backend) {
  /* An empty name means the caller took the default, which the header documents
   * as "cpu".  Anything else is resolved against what this *build* provides.
   *
   * The check happens here, before the file is mapped, and it is not merely an
   * optimization. The device is the argument the *caller* passed, so it is the
   * one they can always act on; the checkpoint is a file, and a caller told
   * "that is not a GGUF file" has learned nothing about the request they got
   * wrong. Asking about the device first is what makes the error name the thing
   * the caller controls -- and it means a two-gigabyte mapping is not paid for
   * on a request that was never going to work. */
  std::string device = backend.empty() ? "cpu" : backend;
  kernel::require_backend(device);

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
   * `encode` -- which works fine -- unreachable for them.
   *
   * The backend is *created* only here and not before, so a checkpoint this
   * build has no graph for does not pay for a CUDA context. Creating it is a
   * heavier thing than asking whether it can be created, which is what the
   * check above already did. */
  std::unique_ptr<kernel::Backend> device_backend;
  std::unique_ptr<Qwen3Model> model;
  if (checkpoint->get_string("general.architecture", "") == "qwen3") {
    device_backend = kernel::make_backend(device);
    model = Qwen3Model::load(*checkpoint, *device_backend);
    /* The model has copied every tensor it needs into the backend's memory, and
     * the tokenizer its vocabulary and merges, so nothing reads the file again.
     * Release the mapping here rather than at session teardown: `MemAvailable`
     * is what the caller watches, and it is measured while the session is open,
     * not after it closes.  A checkpoint of an architecture this build cannot
     * run stays mapped, because binding never happened and `tensor_data` may
     * still be the only thing keeping the loaded bytes alive. */
    checkpoint->release_mapping();
  }

  return std::unique_ptr<Session>(new Session(gguf_path, device, std::move(device_backend),
                                              std::move(checkpoint), std::move(tokenizer),
                                              std::move(model)));
}

std::vector<float> Session::forward(const int32_t *tokens, int64_t n, int64_t *argmax_out) {
  if (model_ == nullptr) {
    throw Error("this checkpoint is architecture '" +
                checkpoint_->get_string("general.architecture", "?") +
                "', which this build has no graph for; only 'qwen3' runs");
  }
  const kernel::DeviceBuffer logits = model_->forward(tokens, n, position_);
  position_ = model_->cache_length();

  /* The logits come back in one transfer.  Not because they are wanted on the
   * host -- a decode wants one integer -- but because the ABI hands them to the
   * caller, and a caller that does not want them is not the one this call is
   * for.  `argmax_out`, when the caller asks, is filled from the device. */
  std::vector<float> host(static_cast<std::size_t>(model_->n_vocab()));
  backend_->copy_to_host(host.data(), logits, model_->n_vocab() * 4);

  if (argmax_out != nullptr) {
    /* A separate device round trip for the token, which is what a caller that
     * only wants the next id would do.  It costs four bytes against the four
     * hundred kilobytes just transferred, and it keeps the logits path and the
     * token path as two things a caller can choose between rather than one
     * thing with a flag. */
    kernel::DeviceBuffer index = backend_->allocate(8);
    backend_->argmax(logits, model_->n_vocab(), index);
    backend_->synchronize();
    backend_->copy_to_host(argmax_out, index, 8);
    backend_->release(index);
  }
  return host;
}

void Session::reset() {
  position_ = 0;
  if (model_ != nullptr) {
    model_->reset();
  }
}

}  // namespace pocketllm