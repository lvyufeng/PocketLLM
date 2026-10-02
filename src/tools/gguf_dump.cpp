/* Print a GGUF's directory as JSON, for comparison against the Python reader.
 *
 * This is an internal tool, not part of the ABI: it exists so the reader test
 * can diff the C parse against `pocketllm.loader.gguf`'s on a real checkpoint
 * without the engine having to expose its internals.  It links the reader
 * directly rather than going through `libpocketllm.so`, which keeps the ABI to
 * the eight functions the header declares.
 *
 * Tensors are printed sorted by name, because the two readers do not have to
 * agree on directory order to agree on the directory, and a test that failed on
 * ordering would be a test that hides a real mismatch in noise.
 *
 * Usage: pocketllm-gguf-dump <checkpoint.gguf>
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

#include "abi/spec.h"
#include "gguf/reader.h"
#include "runtime/status.h"

namespace {

/* JSON string escaping, minimal but correct for the bytes that appear in GGUF
 * metadata: names, tokenizer pieces and chat templates.  A chat template
 * contains newlines and quotes, so this is not optional. */
std::string json_escape(const std::string &in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

void print_value(const pocketllm::GgufValue &value, int type) {
  using pocketllm::GgufArray;
  if (const auto *as_int = std::get_if<int64_t>(&value)) {
    std::printf("%lld", static_cast<long long>(*as_int));
  } else if (const auto *as_double = std::get_if<double>(&value)) {
    std::printf("%.17g", *as_double);
  } else if (const auto *as_bool = std::get_if<bool>(&value)) {
    std::printf("%s", *as_bool ? "true" : "false");
  } else if (const auto *as_string = std::get_if<std::string>(&value)) {
    std::printf("\"%s\"", json_escape(*as_string).c_str());
  } else if (const auto *as_array = std::get_if<GgufArray>(&value)) {
    /* An array prints its item type and length rather than its contents: the
     * tokenizer vocabulary is 150k strings, and the reader test compares those
     * separately.  Length plus item type is the right granularity for a
     * directory dump -- it catches a misparsed length or type immediately and
     * costs nothing. */
    std::printf("{\"__array__\":true,\"item_type\":%d,\"length\":%llu}",
                as_array->value_type, static_cast<unsigned long long>(as_array->length));
    (void)type;
  } else {
    std::printf("null");
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <checkpoint.gguf>\n", argv[0]);
    return 2;
  }

  try {
    pocketllm::GgufReader reader(argv[1]);

    std::printf("{");
    std::printf("\"version\":%u,", reader.version());
    std::printf("\"tensor_count\":%llu,", static_cast<unsigned long long>(reader.tensor_count()));
    std::printf("\"metadata_count\":%llu,", static_cast<unsigned long long>(reader.metadata_count()));
    std::printf("\"alignment\":%llu,", static_cast<unsigned long long>(reader.alignment()));
    std::printf("\"data_start\":%llu,", static_cast<unsigned long long>(reader.data_start()));
    std::printf("\"size\":%llu,", static_cast<unsigned long long>(reader.size()));

    std::printf("\"metadata\":{");
    bool first = true;
    for (const auto &entry : reader.metadata()) {
      if (!first) {
        std::printf(",");
      }
      first = false;
      std::printf("\"%s\":", json_escape(entry.first).c_str());
      print_value(entry.second.value, entry.second.type);
    }
    std::printf("},");

    std::vector<const pocketllm::GgufTensorInfo *> sorted;
    sorted.reserve(reader.tensors().size());
    for (const auto &tensor : reader.tensors()) {
      sorted.push_back(&tensor);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const pocketllm::GgufTensorInfo *a, const pocketllm::GgufTensorInfo *b) {
                return a->name < b->name;
              });

    std::printf("\"tensors\":[");
    first = true;
    for (const pocketllm::GgufTensorInfo *tensor : sorted) {
      if (!first) {
        std::printf(",");
      }
      first = false;
      std::printf("{\"name\":\"%s\",\"type_id\":%d,\"type\":\"%s\",\"offset\":%llu,"
                  "\"absolute_offset\":%llu,\"nbytes\":%llu,\"size_known\":%s,\"dims\":[",
                  json_escape(tensor->name).c_str(), tensor->type_id,
                  pocketllm::ggml_type_of(tensor->type_id).name,
                  static_cast<unsigned long long>(tensor->offset),
                  static_cast<unsigned long long>(tensor->absolute_offset),
                  static_cast<unsigned long long>(tensor->nbytes),
                  tensor->size_known ? "true" : "false");
      for (std::size_t i = 0; i < tensor->dimensions.size(); ++i) {
        std::printf("%s%llu", i ? "," : "",
                    static_cast<unsigned long long>(tensor->dimensions[i]));
      }
      std::printf("]}");
    }
    std::printf("]}\n");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}