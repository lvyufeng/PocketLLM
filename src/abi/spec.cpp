#include "abi/spec.h"

#include <array>
#include <string>

namespace pocketllm {

namespace {

/* reader.py's GGML_TYPES, transcribed id for id.  The array is indexed by type
 * id and padded to the highest id the table carries (143, the fork-private
 * ptq1_0), which is why a lookup is a bounds check rather than a search: the
 * ids are dense enough that a table beats a map here, and the reader is on the
 * path to every checkpoint load.
 *
 * The sentinel marks a hole -- an id GGML defines that this table does not
 * carry.  4, 5, 31-141 are such holes in the upstream numbering. */
constexpr GgmlType kHole{"", 0, 0};

constexpr std::array<GgmlType, 144> kGgmlTypes = {{
    {"f32", 1, 4},          // 0
    {"f16", 1, 2},          // 1
    {"q4_0", 32, 18},       // 2
    {"q4_1", 32, 20},       // 3
    kHole,                  // 4
    kHole,                  // 5
    {"q5_0", 32, 22},       // 6
    {"q5_1", 32, 24},       // 7
    {"q8_0", 32, 34},       // 8
    {"q8_1", 32, 40},       // 9
    {"q2_k", 256, 84},      // 10
    {"q3_k", 256, 110},     // 11
    {"q4_k", 256, 144},     // 12
    {"q5_k", 256, 176},     // 13
    {"q6_k", 256, 210},     // 14
    {"q8_k", 256, 292},     // 15
    {"iq2_xxs", 256, 66},   // 16
    {"iq2_xs", 256, 74},    // 17
    {"iq3_xxs", 256, 98},   // 18
    {"iq1_s", 256, 50},     // 19
    {"iq4_nl", 32, 18},     // 20
    {"iq3_s", 256, 110},    // 21
    {"iq2_s", 256, 82},     // 22
    {"iq4_xs", 256, 136},   // 23
    {"i8", 1, 1},           // 24
    {"i16", 1, 2},          // 25
    {"i32", 1, 4},          // 26
    {"i64", 1, 8},          // 27
    {"f64", 1, 8},          // 28
    {"iq1_m", 256, 56},     // 29
    {"bf16", 1, 2},         // 30
}};

}  // namespace

GgmlType ggml_type_of(int type_id) {
  if (type_id < 0 || static_cast<std::size_t>(type_id) >= kGgmlTypes.size()) {
    /* The name has to outlive the call, so an unknown id gets a static buffer's
     * contents rather than a temporary.  It is written once per distinct id on
     * first use and never mutated after, which keeps it safe without a lock on
     * the read path. */
    static std::array<std::string, 64> unknown{};
    static std::array<bool, 64> filled{};
    if (type_id >= 0 && type_id < static_cast<int>(unknown.size())) {
      if (!filled[static_cast<std::size_t>(type_id)]) {
        unknown[static_cast<std::size_t>(type_id)] = "unknown_" + std::to_string(type_id);
        filled[static_cast<std::size_t>(type_id)] = true;
      }
      GgmlType result{unknown[static_cast<std::size_t>(type_id)].c_str(), 0, 0};
      return result;
    }
    return GgmlType{"unknown", 0, 0};
  }

  const GgmlType &entry = kGgmlTypes[static_cast<std::size_t>(type_id)];
  if (entry.name[0] == '\0') {
    return GgmlType{"unknown", 0, 0};
  }
  return entry;
}

}  // namespace pocketllm