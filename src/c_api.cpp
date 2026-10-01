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

int pocketllm_encode(const pocketllm_session *session, const char *text, int add_special, int32_t *out, int cap) {
  if (session == nullptr || text == nullptr) {
    return -1;
  }
  /* The tokenizer lands in the next step.  Failing with a distinct code now
   * keeps `forward`'s failure -- which is about the model, not the vocabulary --
   * distinguishable from it. */
  (void)add_special;
  (void)out;
  (void)cap;
  return -2;
}

int pocketllm_decode(const pocketllm_session *session, const int32_t *ids, int n, char *out, int cap) {
  if (session == nullptr || ids == nullptr || out == nullptr || n < 0 || cap <= 0) {
    return -1;
  }
  (void)ids;
  (void)n;
  out[0] = '\0';
  return 0;
}

int pocketllm_forward(pocketllm_session *session, const int32_t *tokens, int n, float *logits, int logits_cap) {
  if (session == nullptr || tokens == nullptr || n <= 0 || logits == nullptr || logits_cap <= 0) {
    return -1;
  }
  /* The graph walk lands after the reader and the tokenizer.  Until then this
   * is the one call that reports the missing piece by name, because it is the
   * one a caller reaches last. */
  return -3;
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

}  // extern "C"