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

  /* Drop the KV cache and rewind to position 0. */
  void reset();

 private:
  Session(std::string gguf_path, std::string backend);

  std::string gguf_path_;
  std::string backend_;

  /* The next position the model will write.  It is the KV cache's length, and
   * keeping the two in one object is what stops them drifting. */
  int64_t position_ = 0;
};

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_SESSION_H */