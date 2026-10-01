/* The engine session: one checkpoint, one device, one process.
 *
 * This is the C++ object behind the opaque `pocketllm_session*` in the header.
 * It is deliberately thin at first -- the GGUF reader, the tokenizer and the
 * model graph are attached to it as those land -- but the *shape* is fixed
 * now, because a session that owned the wrong things later is a rewrite of
 * every entry point rather than an addition to one.
 *
 * What it owns, and why:
 *
 *   - the checkpoint handle, so `forward` does not re-open the file per call;
 *   - the KV cache, because it is stateful across calls and belongs to the
 *     session rather than to any one `forward`;
 *   - the position counter, which is the cache's length and has to advance in
 *     step with it;
 *   - the arena the activations are carved from, reused across calls.
 *
 * What it does *not* own: the device.  One process owns one device, and the
 * session borrows it.  There is no rank, no world size and no device list --
 * `EngineArgs` has none either, and for the same reason.
 */

#ifndef POCKETLLM_RUNTIME_SESSION_H
#define POCKETLLM_RUNTIME_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "tokenizer/bpe.h"

namespace pocketllm {

class Session {
 public:
  /* Open `gguf_path` on the named backend.  Throws `Error` with a message
   * written for the caller -- a missing file, an unknown backend, a checkpoint
   * whose architecture this build cannot run -- which `c_api.cpp` turns into
   * the negative return the header documents. */
  static std::unique_ptr<Session> open(const std::string &gguf_path, const std::string &backend);

  ~Session();

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  const std::string &backend() const { return backend_; }

  /* The mapped checkpoint.  Held open for the session's lifetime: re-opening a
   * 1.5 GB file per `forward` would re-map it per call, and the tensors are read
   * straight out of the mapping by the kernels. */
  const GgufReader &checkpoint() const { return *checkpoint_; }

  /* The tokenizer built from this checkpoint's metadata, or nullptr if the
   * checkpoint's vocabulary is one this build cannot read.  It is built in
   * `open` -- deliberately there rather than on first use -- because a
   * vocabulary this build cannot tokenize is a checkpoint this build cannot
   * run, and that belongs in the open's error rather than in the first
   * `encode`'s.  The two callers below still check, because the pointer is
   * what they have. */
  const Tokenizer *tokenizer() const { return tokenizer_.get(); }

  /* Drop the KV cache and rewind to position 0. */
  void reset();

 private:
  Session(std::string gguf_path, std::string backend, std::unique_ptr<GgufReader> checkpoint,
          std::unique_ptr<Tokenizer> tokenizer);

  std::string gguf_path_;
  std::string backend_;
  std::unique_ptr<GgufReader> checkpoint_;
  std::unique_ptr<Tokenizer> tokenizer_;

  /* The next position the model will write.  It is the KV cache's length, and
   * keeping the two in one object is what stops them drifting. */
  int64_t position_ = 0;
};

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_SESSION_H */