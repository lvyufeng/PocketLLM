/* The engine session: one checkpoint, one device, one process.
 *
 * This is the C++ object behind the opaque `pocketllm_session*` in the header.
 *
 * What it owns, and why:
 *
 *   - the checkpoint handle, so `forward` does not re-open the file per call;
 *   - the *device*, because a session is bound to one and the model's weights
 *     live on it.  One process owns one device: there is no rank, no world size
 *     and no device list -- `EngineArgs` has none either, and for the same
 *     reason;
 *   - the KV cache, because it is stateful across calls and belongs to the
 *     session rather than to any one `forward`;
 *   - the position counter, which is the cache's length and has to advance in
 *     step with it;
 *   - the arena the activations are carved from, reused across calls.
 */

#ifndef POCKETLLM_RUNTIME_SESSION_H
#define POCKETLLM_RUNTIME_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/qwen3.h"
#include "tokenizer/bpe.h"

namespace pocketllm {

class Session {
 public:
  /* Open `gguf_path` on the named backend.  Throws `Error` with a message
   * written for the caller -- a missing file, a backend this build does not
   * provide, a checkpoint whose architecture this build cannot run -- which
   * `c_api.cpp` turns into the negative return the header documents. */
  static std::unique_ptr<Session> open(const std::string &gguf_path, const std::string &backend);

  ~Session();

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  const std::string &backend() const { return backend_name_; }

  /* Which device this session actually bound to, for diagnostics. */
  std::string device_description() const {
    return backend_ == nullptr ? "(none)" : backend_->describe();
  }

  /* The mapped checkpoint.  Held open for the session's lifetime: re-opening a
   * 1.5 GB file per `forward` would re-map it per call, and the tensors are read
   * straight out of the mapping by the loader. */
  const GgufReader &checkpoint() const { return *checkpoint_; }

  /* The tokenizer built from this checkpoint's metadata, or nullptr if the
   * vocabulary is one this build cannot read.  Built in `open` -- deliberately
   * there rather than on first use -- because a vocabulary this build cannot
   * tokenize is a checkpoint this build cannot run, and that belongs in the
   * open's error rather than in the first `encode`'s. */
  const Tokenizer *tokenizer() const { return tokenizer_.get(); }

  /* The bound model, or nullptr for a checkpoint whose architecture this build
   * does not implement.  Built in `open` for the same reason the tokenizer is.
   * nullptr is still possible -- a checkpoint with a vocabulary but an
   * unrecognised architecture is one `encode` can serve. */
  const Qwen3Model *model() const { return model_.get(); }

  /* Run `tokens` and return the logits of the last one, on the host.  When
   * `argmax_out` is not null it also receives the index of the largest logit,
   * computed on the device -- see the note there for why the caller chooses.
   *
   * Throws on a failure; `c_api.cpp` turns that into the header's return code. */
  std::vector<float> forward(const int32_t *tokens, int64_t n, int64_t *argmax_out);

  /* Drop the KV cache and rewind to position 0. */
  void reset();

 private:
  Session(std::string gguf_path, std::string backend_name,
          std::unique_ptr<kernel::Backend> backend, std::unique_ptr<GgufReader> checkpoint,
          std::unique_ptr<Tokenizer> tokenizer, std::unique_ptr<Qwen3Model> model);

  std::string gguf_path_;
  std::string backend_name_;
  std::unique_ptr<kernel::Backend> backend_;
  std::unique_ptr<GgufReader> checkpoint_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Qwen3Model> model_;

  /* The next position the model will write.  It is the KV cache's length, and
   * keeping the two in one object is what stops them drifting. */
  int64_t position_ = 0;
};

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_SESSION_H */