/* Which backends this library was built with.
 *
 * The answer is a *build* fact and not a runtime one, which is the whole reason
 * this is a small compiled-in table rather than a probe. `pocketllm_open`
 * asking for `cuda` on a library built without CUDA must fail with "this build
 * has no CUDA backend"; a probe that looked for a device would instead find one
 * on a machine whose library cannot run it, and the failure would arrive later
 * as an unresolved symbol or a wrong answer.
 *
 * `POCKETLLM_WITH_CUDA` is defined by CMake when a CUDA toolchain was found.
 * The CUDA backend is compiled only then, so the declaration below is inside
 * the guard as well as the definition -- a reference to a class that does not
 * exist is a link error, which is the correct outcome but a worse message than
 * the one `backend_available` gives.
 */

#include "kernel/backend.h"

#include "runtime/status.h"

namespace pocketllm {
namespace kernel {

/* Defined in `cpu/backend.cpp`, and in `cuda/backend.cpp` / `ascend/backend.cpp`
 * when the corresponding backend is built. */
std::unique_ptr<Backend> make_cpu_backend();
#ifdef POCKETLLM_WITH_CUDA
std::unique_ptr<Backend> make_cuda_backend();
#endif
#ifdef POCKETLLM_WITH_ASCEND
std::unique_ptr<Backend> make_ascend_backend();
#endif

namespace {

bool known(const std::string &name) {
  if (name == "cpu") {
    return true;
  }
#ifdef POCKETLLM_WITH_CUDA
  if (name == "cuda") {
    return true;
  }
#endif
#ifdef POCKETLLM_WITH_ASCEND
  if (name == "ascend") {
    return true;
  }
#endif
  return false;
}

}  // namespace

bool backend_available(const std::string &name) { return known(name); }

void require_backend(const std::string &name) {
  if (known(name)) {
    return;
  }
  /* The refusal names the build, not the machine: `qnn`, `horizon` and
   * `ascend` are declared in the Python registry and have no C implementation
   * in any build, while `cuda` has one in some. Saying which of those two cases
   * this is saves the reader from checking. */
  std::string message = "backend '" + name + "' is not in this build; it provides 'cpu'";
#ifdef POCKETLLM_WITH_CUDA
  message += " and 'cuda'";
#endif
#ifdef POCKETLLM_WITH_ASCEND
  message += " and 'ascend'";
#endif
  throw Error(message);
}

std::unique_ptr<Backend> make_backend(const std::string &name) {
  /* Separate from `require_backend` so a caller can ask the question early --
   * before it has done anything expensive or irreversible -- and create the
   * backend later, when it knows it wants one. The two must give the same answer
   * to the same name, which is why they share `known`. */
  require_backend(name);
  if (name == "cpu") {
    return make_cpu_backend();
  }
#ifdef POCKETLLM_WITH_ASCEND
  if (name == "ascend") {
    /* Reached only for "ascend": `require_backend` threw for every other name. */
    return make_ascend_backend();
  }
#endif
#ifdef POCKETLLM_WITH_CUDA
  /* Reached only for "cuda": `require_backend` threw for every other name. */
  return make_cuda_backend();
#else
  /* Unreachable for the same reason, and written as a throw rather than a
   * fallback so that the day the table grows a name this function has no branch
   * for, the caller gets an error instead of a CUDA request quietly running on
   * the CPU -- which is the failure mode a silent default would create. */
  throw Error("backend '" + name + "' is listed but has no implementation");
#endif
}

}  // namespace kernel
}  // namespace pocketllm