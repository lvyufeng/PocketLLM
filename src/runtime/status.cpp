#include "runtime/status.h"

#include <cstring>

namespace pocketllm {

namespace {

/* The one place the truncation rule is implemented, so both overloads share it.
 * `cap` includes the terminator, matching `strncpy` and the header's wording. */
void copy_message(char *err, std::size_t cap, const char *message) noexcept {
  if (err == nullptr || cap == 0) {
    return;
  }
  std::size_t len = std::strlen(message);
  if (len > cap - 1) {
    len = cap - 1;
  }
  std::memcpy(err, message, len);
  err[len] = '\0';
}

}  // namespace

void write_error(char *err, std::size_t err_cap, const std::string &message) noexcept {
  copy_message(err, err_cap, message.c_str());
}

void write_error(char *err, std::size_t err_cap, const std::exception &e) noexcept {
  copy_message(err, err_cap, e.what());
}

}  // namespace pocketllm