#include "runtime/session.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstring>

#include "runtime/status.h"

namespace pocketllm {

namespace {

/* True if `path` names an existing regular file.  Doing this by `stat` rather
 * than by opening keeps the check cheap and keeps the reason the open fails
 * legible: "no such file" and "not a file" are different mistakes, and the
 * reader in the next step will fail with a third message for a file that opens
 * but is not a GGUF. */
bool is_regular_file(const std::string &path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace

Session::Session(std::string gguf_path, std::string backend)
    : gguf_path_(std::move(gguf_path)), backend_(std::move(backend)) {}

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

  if (!is_regular_file(gguf_path)) {
    throw Error("cannot open checkpoint '" + gguf_path + "': " + std::strerror(ENOENT));
  }

  return std::unique_ptr<Session>(new Session(gguf_path, device));
}

void Session::reset() { position_ = 0; }

}  // namespace pocketllm