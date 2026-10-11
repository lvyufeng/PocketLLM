/* The `extern "C"` surface, and nothing else.
 *
 * Every function here is a three-line shape: validate the arguments the C
 * convention cannot (a NULL pointer the caller's compiler would have caught in
 * C++), delegate to the C++ object, and translate.  No exception is allowed to
 * escape, so each entry point is a `try`/`catch` pair around the delegate --
 * that is the entire reason this translation unit exists separately from the
 * classes it calls.
 *
 * The header is the documentation for the behaviour; this file is the mapping
 * from it onto C++.
 */

#include "pocketllm.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "kernel/kernels.h"
#include "runtime/session.h"
#include "runtime/status.h"

namespace {

/* The version string is built once and returned by pointer, so a caller may
 * hold it for the process lifetime without owning it -- which is what the
 * header promises.  `static` at function scope initializes on first call and is
 * thread-safe under C++11 and later. */
const std::string &abi_version_string() {
  static const std::string version =
      std::to_string(POCKETLLM_ABI_VERSION_MAJOR) + "." + std::to_string(POCKETLLM_ABI_VERSION_MINOR);
  return version;
}

/* The opaque handle is a `Session` wearing a different name.  Keeping the
 * reinterpret_cast here rather than inline at each entry point means the cast
 * appears once per direction, so a change to the handle's representation is a
 * one-place change. */
pocketllm::Session *as_session(pocketllm_session *handle) {
  return reinterpret_cast<pocketllm::Session *>(handle);
}

/* `encode` and `decode` take a `const` session in the header -- they do not
 * change the model or the cache -- so they need their own spelling of the same
 * cast.  Casting the constness away would be a lie the type system is entitled
 * to catch; this keeps it honest. */
const pocketllm::Session *as_session(const pocketllm_session *handle) {
  return reinterpret_cast<const pocketllm::Session *>(handle);
}

}  // namespace

extern "C" {

const char *pocketllm_abi_version(void) { return abi_version_string().c_str(); }

pocketllm_session *pocketllm_open(const char *gguf_path, const char *backend, char *err, size_t err_cap) {
  if (gguf_path == nullptr) {
    pocketllm::write_error(err, err_cap, std::string("pocketllm_open: gguf_path is NULL"));
    return nullptr;
  }
  try {
    std::string device = backend == nullptr ? "" : backend;
    auto session = pocketllm::Session::open(gguf_path, device);
    return reinterpret_cast<pocketllm_session *>(session.release());
  } catch (const std::exception &e) {
    pocketllm::write_error(err, err_cap, e);
    return nullptr;
  }
}

void pocketllm_close(pocketllm_session *session) {
  /* Deliberately no try/catch: `delete` on a well-formed object cannot throw,
   * and a destructor that did would have nothing useful to report anyway. */
  delete as_session(session);
}

int pocketllm_encode(const pocketllm_session *session, const char *text, int add_special, int parse_special,
                     int32_t *out, int cap) {
  if (session == nullptr || text == nullptr || cap < 0) {
    return -1;
  }
  try {
    const pocketllm::Tokenizer *tokenizer = as_session(session)->tokenizer();
    if (tokenizer == nullptr) {
      return -2;
    }
    /* `out == NULL && cap == 0` is the sizing call the header documents, and it
     * works because `data()` on an empty vector is never written to before the
     * size check. */
    const std::vector<int32_t> ids = tokenizer->encode(text, add_special != 0, parse_special != 0);
    if (out == nullptr) {
      return static_cast<int>(ids.size());
    }
    if (static_cast<int>(ids.size()) > cap) {
      /* The header promises the caller can size the buffer from a sizing call,
       * so a result that does not fit reports the size it needed and writes
       * nothing. */
      return static_cast<int>(ids.size());
    }
    std::memcpy(out, ids.data(), ids.size() * sizeof(int32_t));
    return static_cast<int>(ids.size());
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_decode(const pocketllm_session *session, const int32_t *ids, int n, char *out, int cap) {
  if (session == nullptr || ids == nullptr || out == nullptr || n < 0 || cap <= 0) {
    return -1;
  }
  try {
    const pocketllm::Tokenizer *tokenizer = as_session(session)->tokenizer();
    if (tokenizer == nullptr) {
      return -2;
    }
    const std::string text = tokenizer->decode(std::vector<int32_t>(ids, ids + n));
    /* The terminator has to fit, so the body is capped one short of `cap`.
     * Truncating mid-character is the caller's choice -- the header says the
     * result is NUL-terminated and says nothing about cutting on a character
     * boundary, and a caller streaming a partial answer wants the bytes. */
    const std::size_t room = static_cast<std::size_t>(cap) - 1;
    const std::size_t written = text.size() < room ? text.size() : room;
    std::memcpy(out, text.data(), written);
    out[written] = '\0';
    return static_cast<int>(written);
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_forward(pocketllm_session *session, const int32_t *tokens, int n, float *logits, int logits_cap) {
  if (session == nullptr || tokens == nullptr || n <= 0 || logits == nullptr || logits_cap <= 0) {
    return -1;
  }
  try {
    pocketllm::Session *self = as_session(session);
    const pocketllm::Qwen3Model *model = self->model();
    /* Asked before the work rather than after: the gather needs the vocabulary
     * size to bound its output, and a caller whose buffer is too small has
     * already lost by the time the logits exist. */
    if (model == nullptr) {
      return -2;
    }
    const int64_t vocab = model->n_vocab();
    if (vocab > logits_cap) {
      /* The header promises a negative return rather than a partial write, so
       * the caller can reissue with a larger buffer. */
      return -1;
    }
    const std::vector<float> values = self->forward(tokens, n, nullptr);
    std::memcpy(logits, values.data(), static_cast<std::size_t>(vocab) * sizeof(float));
    return static_cast<int>(vocab);
  } catch (const pocketllm::ContextLengthError &e) {
    /* A *recoverable* refusal, with its own return code: the request is too
     * long for the checkpoint's window, which is the caller's to fix, not the
     * engine's failure.  Reported before the generic catch so it does not
     * flatten into `-1` -- the whole reason the type exists.  `pocketllm_forward`
     * has no `err` buffer, so the code is the entire signal: a host that wants to
     * name the limit reads it from the same GGUF key the engine does. */
    return pocketllm::kErrContextLength;
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_forward_argmax(pocketllm_session *session, const int32_t *tokens, int n,
                             int32_t *out_token) {
  if (session == nullptr || tokens == nullptr || n <= 0 || out_token == nullptr) {
    return -1;
  }
  try {
    pocketllm::Session *self = as_session(session);
    if (self->model() == nullptr) {
      return -2;
    }
    /* The whole point: the id comes back from `Session::forward`'s device
     * argmax, and the vocabulary is never brought to the host.  The int64 the
     * device writes is narrowed to the int32 the header promises; the vocabulary
     * is far below INT32_MAX, so the cast is exact. */
    int64_t token = 0;
    self->forward(tokens, n, &token);
    *out_token = static_cast<int32_t>(token);
    return 0;
  } catch (const pocketllm::ContextLengthError &e) {
    return pocketllm::kErrContextLength;
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_forward_batch(pocketllm_session *session, const int32_t *tokens,
                            const int64_t *n_per_seq, int64_t n_seq, int32_t *out_tokens) {
  if (session == nullptr || tokens == nullptr || n_per_seq == nullptr || n_seq <= 0 ||
      out_tokens == nullptr) {
    return -1;
  }
  try {
    pocketllm::Session *self = as_session(session);
    if (self->model() == nullptr) {
      return -2;
    }
    /* One batched forward; `Session::forward_batch` reads each sequence's
     * greedy id from its own row's device argmax, so the vocabulary stays on
     * the device exactly as the single-sequence fast path keeps it. */
    self->forward_batch(tokens, n_per_seq, n_seq, out_tokens);
    return 0;
  } catch (const pocketllm::ContextLengthError &e) {
    return pocketllm::kErrContextLength;
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_reserve(pocketllm_session *session, int64_t positions, int64_t slots) {
  if (session == nullptr || positions <= 0 || slots <= 0) {
    return -1;
  }
  try {
    pocketllm::Session *self = as_session(session);
    if (self->model() == nullptr) {
      return -2;
    }
    self->reserve(positions, slots);
    return 0;
  } catch (const pocketllm::ContextLengthError &e) {
    return pocketllm::kErrContextLength;
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_reset(pocketllm_session *session) {
  if (session == nullptr) {
    return -1;
  }
  try {
    as_session(session)->reset();
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

int pocketllm_argmax(const float *logits, int n) {
  if (logits == nullptr || n <= 0) {
    return -1;
  }
  int best = 0;
  for (int i = 1; i < n; ++i) {
    /* Strictly greater, so ties resolve to the lowest index.  A greedy decoder
     * that broke ties the other way would still be correct, but the oracle and
     * `llama-cli` both take the first, and a mismatch there would look like a
     * numerical bug for a day. */
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

int pocketllm_temperature(const float *logits, float *out, int n, float temperature) {
  if (logits == nullptr || out == nullptr || n <= 0) {
    return -1;
  }
  try {
    pocketllm::kernel::logits_temperature(logits, out, n, temperature);
    return 0;
  } catch (const std::exception &) {
    /* The kernel's only refusal here is a non-positive temperature, which the
     * header documents as this function's refusal too.  Catching it rather than
     * re-checking the predicate keeps the rule in one place -- a caller that
     * passes 0.0 must get a negative return and not a vector of infinities. */
    return -1;
  }
}

int pocketllm_sample(const float *logits, int n, float uniform, int top_k, float top_p, float min_p) {
  if (logits == nullptr || n <= 0) {
    return -1;
  }
  try {
    /* The ranked index list the sampler needs, and the same allocation shape
     * `topk_sample` takes on the device side: it is per-call rather than
     * per-session because this function holds no state, and at this vocabulary
     * it is 1.2 MB against a 600 KB logit vector the caller already has.  A
     * decode loop that cared could hold it itself and call the kernel, which is
     * why the kernel takes it as a parameter rather than allocating. */
    std::vector<int64_t> order(static_cast<std::size_t>(n));
    int64_t token = 0;
    pocketllm::kernel::topk_sample(logits, n, uniform, top_k, top_p, min_p, order.data(), &token);
    return static_cast<int>(token);
  } catch (const std::exception &) {
    return -1;
  }
}

}  // extern "C"