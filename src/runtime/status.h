/* Errors at the ABI boundary.
 *
 * The engine is C++ and uses exceptions internally, for the same reason any
 * resource-owning C++ does: an error deep in the GGUF reader should not have to
 * be threaded back through every caller as a return code.  But no exception may
 * cross `extern "C"`: unwinding into a host language's frame is undefined
 * behaviour, and a `ctypes` frame is a host language's frame.
 *
 * So `c_api.cpp` wraps every entry point in `catch` and reports failure the way
 * the header documents -- a negative return and a message in the caller's
 * `err` buffer.  This file holds the two pieces that makes possible: the
 * message writer, and the exception type the engine throws when a message is
 * meant for the user rather than for a log.
 */

#ifndef POCKETLLM_RUNTIME_STATUS_H
#define POCKETLLM_RUNTIME_STATUS_H

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pocketllm {

/* An error whose message is written for whoever called the ABI.  Anything else
 * that escapes -- bad_alloc, a std::runtime_error from the STL -- is caught too
 * and reported by its `what()`, but this type is the one that carries a
 * deliberate message. */
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &what) : std::runtime_error(what) {}
};

/* Copy `message` into a caller-owned buffer, NUL-terminated and truncated
 * rather than overflowing.  A NULL buffer or a zero capacity is a no-op: the
 * header lets a caller that wants only the return value pass neither. */
void write_error(char *err, std::size_t err_cap, const std::string &message) noexcept;

/* Same, but for a message being built as it unwinds.  Helps the `catch` blocks
 * in `c_api.cpp` stay one line each. */
void write_error(char *err, std::size_t err_cap, const std::exception &e) noexcept;

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_STATUS_H */