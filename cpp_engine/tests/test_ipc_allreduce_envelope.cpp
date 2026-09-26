// Whether a call takes the hand-written Ascend collective or falls through to
// HCCL, and what the environment is allowed to say about it -- plus the one
// diagnostic that is a number rather than a switch, the interval between two host
// reads of the device arrival wait's status word.
//
// Host-only: `ascend_ipc_allreduce_f16_applies()` and
// `ascend_ipc_status_check_every()` read the environment and do arithmetic, so this
// needs no NPU, no checkpoint and no peer processes. Nothing here calls ACL even
// though the binary links the Ascend core, which is why it runs anywhere the
// backend builds.
//
// It exists because the switch became a default. While it was opt-in, a wrong
// reading of the variable cost a measurement; now the same mistake silently
// changes which collective a shipped server uses on every step, and which of two
// answers a token is computed from. The spellings are pinned individually rather
// than through one representative, because the trap this replaced was exactly a
// spelling -- `atoi` made `=true` mean "off".

#include "ipc_allreduce.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << "  " << message << ": " << (condition ? "PASS" : "FAIL") << "\n";
    if (!condition) ++failures;
}

void set_env(const char* name, const char* value) {
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
}

// A plane well inside the ceiling, so the switch is the only thing these cases
// vary. The ceiling itself gets its own group below.
bool applies(const char* switch_value, int world = 4, int count = 4096) {
    set_env("POCKET_ASCEND_IPC_ALLREDUCE", switch_value);
    return pocket::ascend_ipc_allreduce_f16_applies(world, count);
}

constexpr int kWorld = 4;
constexpr size_t kCeiling = 512 * 5120;

void test_switch_spellings() {
    std::cout << "the switch is an opt-out\n";

    // Unset is the shipped configuration and has to be on: this is the whole
    // point of the change, and it is the case an ordinary `pocketllm serve` hits.
    check(applies(nullptr), "unset: on");
    check(applies(""), "empty: on");

    // Anything that is not one of the five off spellings counts as on, including
    // the spellings that used to be misread.
    check(applies("1"), "1: on");
    check(applies("true"), "true: on");
    check(applies("on"), "on: on");
    check(applies("yes"), "yes: on");
    check(applies("TRUE"), "TRUE: on");
    check(applies(" 1"), "leading space: on");

    // The off spellings, which have to match `qwen_env_enabled_default` in
    // cpp_engine/engine/qwen_engine.cpp one for one.
    check(!applies("0"), "0: off");
    check(!applies("false"), "false: off");
    check(!applies("FALSE"), "FALSE: off");
    check(!applies("off"), "off: off");
    check(!applies("OFF"), "OFF: off");
}

void test_envelope() {
    std::cout << "the envelope still falls through to HCCL\n";

    // Back to unset, which is the shipped configuration: the group above leaves
    // whatever spelling it tested last behind.
    set_env("POCKET_ASCEND_IPC_ALLREDUCE", nullptr);

    check(!pocket::ascend_ipc_allreduce_f16_applies(1, 4096),
          "world 1: false, there is nothing to reduce across");
    check(!pocket::ascend_ipc_allreduce_f16_applies(0, 4096), "world 0: false");
    check(!pocket::ascend_ipc_allreduce_f16_applies(-1, 4096), "world -1: false");
    check(!pocket::ascend_ipc_allreduce_f16_applies(9, 4096),
          "world above the stamp-range limit: false");
    check(pocket::ascend_ipc_allreduce_f16_applies(2, 4096), "world 2: true");
    check(pocket::ascend_ipc_allreduce_f16_applies(8, 4096),
          "world 8, the largest this backend targets: true");

    check(!pocket::ascend_ipc_allreduce_f16_applies(kWorld, 0),
          "count 0: false");
    check(!pocket::ascend_ipc_allreduce_f16_applies(kWorld, -1),
          "count -1: false");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 1),
          "count 1, the rows=1 decode plane: true");

    // The ceiling is inclusive, and it is in elements, not rows or bytes. The
    // widest plane the engine issues here is `rows * 5120`, so this admits 512
    // rows and refuses 513.
    check(pocket::ascend_ipc_allreduce_f16_applies(
              kWorld, static_cast<int>(kCeiling)),
          "exactly at the default ceiling: true");
    check(!pocket::ascend_ipc_allreduce_f16_applies(
              kWorld, static_cast<int>(kCeiling) + 1),
          "one element above the default ceiling: false");

    // What the ceiling is for: a plane past the crossing loses to HCCL, so it must
    // not reach this path by default. 513 rows of 5120 is just past it; the
    // measured crossing is between 421 and 629 rows.
    check(!pocket::ascend_ipc_allreduce_f16_applies(kWorld, 629 * 5120),
          "629 rows, past the measured crossing: false");
}

void test_max_elements_override() {
    std::cout << "the size override moves the ceiling both ways\n";

    const std::string name = "POCKET_ASCEND_IPC_ALLREDUCE_MAX_ELEMENTS";
    set_env(name.c_str(), nullptr);

    // Lowering: a ceiling the default admits is refused once the override is
    // below it.
    set_env(name.c_str(), "4096");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4096),
          "override 4096, count 4096: true");
    check(!pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4097),
          "override 4096, count 4097: false");
    check(!pocket::ascend_ipc_allreduce_f16_applies(
              kWorld, static_cast<int>(kCeiling)),
          "override 4096, count at the default ceiling: false");

    // Raising. This is the direction that a second redundant `count >` test in the
    // predicate used to make write-only: the override could lower the ceiling and
    // never raise it, so a caller above the default stayed on HCCL however the
    // environment was set. Without that test it moves both ways.
    set_env(name.c_str(), "4194304");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4194304),
          "override 4194304, count at the raised ceiling: true");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 3000000),
          "override 4194304, count above the default ceiling: true");
    check(!pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4194305),
          "override 4194304, count one above it: false");

    // A non-positive override keeps the default rather than setting a ceiling of
    // zero. Both spellings reach the same branch: `strtol` reads the empty string
    // as 0 as well.
    set_env(name.c_str(), "0");
    check(pocket::ascend_ipc_allreduce_f16_applies(
              kWorld, static_cast<int>(kCeiling)),
          "override 0 keeps the default ceiling: true");
    check(!pocket::ascend_ipc_allreduce_f16_applies(
              kWorld, static_cast<int>(kCeiling) + 1),
          "override 0 is not a ceiling of zero: false");
    set_env(name.c_str(), "");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4096),
          "override empty keeps the default: true");
    set_env(name.c_str(), "-1");
    check(pocket::ascend_ipc_allreduce_f16_applies(kWorld, 4096),
          "override -1 keeps the default: true");

    set_env(name.c_str(), nullptr);
}

void test_status_check_interval() {
    std::cout << "the status-check interval is not a switch\n";

    const std::string name = "POCKET_ASCEND_IPC_ALLREDUCE_STATUS_EVERY";
    set_env(name.c_str(), nullptr);

    // The shipped interval, and the reason it is 128 rather than a round number: a
    // decode step is 129 collectives, so one check in 128 is one check a step.
    check(pocket::ascend_ipc_status_check_every() == 128, "unset: 128");
    set_env(name.c_str(), "");
    check(pocket::ascend_ipc_status_check_every() == 128, "empty: 128");
    set_env(name.c_str(), "128");
    check(pocket::ascend_ipc_status_check_every() == 128, "128: 128");

    // The case this group exists for. `0` is a reading, not an absence: it means
    // "never look at the word", which is the arm that prices the read and is wrong
    // by construction as a configuration. A parser that treated 0 as unset would
    // answer 128 and quietly turn the control arm into the shipped arm -- which is
    // exactly what `qwen_env_int` does to a QWEN_* knob set to 0, and why this one
    // is read with `strtoll`.
    set_env(name.c_str(), "0");
    check(pocket::ascend_ipc_status_check_every() == 0, "0: 0, not the default");

    set_env(name.c_str(), "1");
    check(pocket::ascend_ipc_status_check_every() == 1, "1: 1, a check per call");
    set_env(name.c_str(), "32");
    check(pocket::ascend_ipc_status_check_every() == 32,
          "32: 32, the interval this default replaced");

    // Negatives mean something to `strtoll` and nothing here. Falling back is the
    // safe direction of the two -- clamping to 0 would read a typo as "turn the
    // detector off".
    set_env(name.c_str(), "-1");
    check(pocket::ascend_ipc_status_check_every() == 128, "-1: the default");
    set_env(name.c_str(), "-9223372036854775808");
    check(pocket::ascend_ipc_status_check_every() == 128,
          "INT64_MIN: the default");

    // No digits at all. `strtoll` answers 0 for both of these, and 0 is the one
    // reading that disables the check, so a typo must not land on it.
    set_env(name.c_str(), "true");
    check(pocket::ascend_ipc_status_check_every() == 128, "true: the default");
    set_env(name.c_str(), "off");
    check(pocket::ascend_ipc_status_check_every() == 128,
          "off: the default, negatives and words are refused, not clamped");

    // Trailing text after the digits is read leniently, the way
    // `..._MAX_ELEMENTS` reads it.
    set_env(name.c_str(), "32x");
    check(pocket::ascend_ipc_status_check_every() == 32, "32x: 32");
    set_env(name.c_str(), " 64");
    check(pocket::ascend_ipc_status_check_every() == 64,
          "leading space: 64, `strtoll` skips it");

    // The knob is read per call rather than cached at first use, so the process
    // above has already driven nine readings. A cached reader would have answered
    // the first one -- 128 -- for all of them, and this last case is what says so.
    set_env(name.c_str(), "7");
    check(pocket::ascend_ipc_status_check_every() == 7,
          "7 after nine other readings: 7, nothing is cached");

    set_env(name.c_str(), nullptr);
    check(pocket::ascend_ipc_status_check_every() == 128, "back to unset: 128");
}

}  // namespace

int main() {
    // The ambient environment must not decide the outcome of a test that is about
    // the environment. Both variables are cleared first and each case sets the one
    // it is about; a run under the serving sweep's exports would otherwise fail
    // its unset case for a reason that has nothing to do with this code.
    set_env("POCKET_ASCEND_IPC_ALLREDUCE", nullptr);
    set_env("POCKET_ASCEND_IPC_ALLREDUCE_MAX_ELEMENTS", nullptr);
    set_env("POCKET_ASCEND_IPC_ALLREDUCE_STATUS_EVERY", nullptr);

    test_switch_spellings();
    test_envelope();
    test_max_elements_override();
    test_status_check_interval();

    set_env("POCKET_ASCEND_IPC_ALLREDUCE", nullptr);
    set_env("POCKET_ASCEND_IPC_ALLREDUCE_MAX_ELEMENTS", nullptr);
    set_env("POCKET_ASCEND_IPC_ALLREDUCE_STATUS_EVERY", nullptr);

    std::cout << (failures == 0 ? "ipc all-reduce envelope: ok\n"
                                : "ipc all-reduce envelope: FAILED\n");
    return failures == 0 ? 0 : 1;
}
