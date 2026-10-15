/* Run the text graph from caller-supplied input embeddings, and write the logits.
 *
 * This is the door a vision model's features come in through.  It is a tool
 * rather than an ABI call for the reason `pocketllm-vision` is: the ABI's surface
 * is the call a text-only host makes, and an image is not part of it yet.  What a
 * test needs and cannot get anywhere else is the *logits* of a forward whose
 * input embeddings are known -- so they can be compared against llama.cpp's own
 * embeddings path, which is the only other implementation of a deepstack.
 *
 * Usage:
 *   pocketllm-forward-embd <checkpoint.gguf> <planes.bin> <logits.bin>
 *                           [--positions FILE] [--device NAME]
 *
 * `planes.bin` is **plane-major**: `n_planes` blocks of `n_tokens * n_embd` f32
 * with plane 0 first.  That is `Qwen3VlVision::encode`'s layout and this model's,
 * and it is *not* the checkpoint's: llama.cpp's input row interleaves the levels
 * per token, so a caller holding that layout transposes the plane axis before
 * handing it over.  `n_tokens` is the file's size divided by a row's.
 *
 * `--positions` supplies the multi-section rotary's positions: `n_tokens * 4`
 * int32, the four sections consecutive within a token, (t, y, x, z) as llama.cpp
 * names them.  Without it every section is the token's own index, which is what a
 * text batch means; with it a token can carry a row and a column, which is what
 * an image patch does.
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/qwen3.h"
#include "runtime/status.h"

namespace {

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <checkpoint.gguf> <planes.bin> <logits.bin>\n"
               "       [--positions FILE] [--device cpu|cuda]\n",
               argv0);
  return 2;
}

std::vector<float> read_floats(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw pocketllm::Error("cannot open '" + path + "'");
  }
  std::vector<float> values;
  float chunk[8192];
  std::size_t read = 0;
  while ((read = std::fread(chunk, sizeof(float), 8192, file)) > 0) {
    values.insert(values.end(), chunk, chunk + read);
  }
  std::fclose(file);
  return values;
}

/* A whole file as int32.  Its own reader rather than a cast of `read_floats`:
 * those bytes *are* integers, and reading them as float32 first turns every small
 * one into a denormal near zero -- a wrong answer with no error, which is the
 * shape of mistake this file's readers exist to make impossible. */
std::vector<int32_t> read_int32(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw pocketllm::Error("cannot open '" + path + "'");
  }
  std::vector<int32_t> values;
  int32_t chunk[8192];
  std::size_t read = 0;
  while ((read = std::fread(chunk, sizeof(int32_t), 8192, file)) > 0) {
    values.insert(values.end(), chunk, chunk + read);
  }
  std::fclose(file);
  return values;
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> positional;
  std::string device = "cpu";
  std::string positions_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--device") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      device = argv[++i];
    } else if (argument == "--positions") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      positions_path = argv[++i];
    } else if (argument == "--help" || argument == "-h") {
      return usage(argv[0]);
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() != 3) {
    return usage(argv[0]);
  }

  try {
    pocketllm::GgufReader checkpoint(positional[0]);
    auto backend = pocketllm::kernel::make_backend(device);
    auto model = pocketllm::Qwen3Model::load(checkpoint, *backend);
    const int64_t row = model->n_embd_inp();
    const std::vector<float> planes = read_floats(positional[1]);
    if (row <= 0 || planes.empty() || planes.size() % static_cast<std::size_t>(row) != 0) {
      throw pocketllm::Error("'" + positional[1] + "' is " + std::to_string(planes.size()) +
                             " floats, which is not a whole number of rows of " +
                             std::to_string(row));
    }
    const int64_t n_tokens = static_cast<int64_t>(planes.size()) / row;
    std::vector<int64_t> positions;
    if (!positions_path.empty()) {
      const std::vector<int32_t> raw = read_int32(positions_path);
      if (static_cast<int64_t>(raw.size()) != n_tokens * 4) {
        throw pocketllm::Error("'" + positions_path + "' is " + std::to_string(raw.size()) +
                               " integers, and " + std::to_string(n_tokens) +
                               " tokens need four each");
      }
      positions.assign(raw.begin(), raw.end());
    }
    const pocketllm::kernel::DeviceBuffer logits_device =
        model->forward_embeddings(planes.data(), positions.empty() ? nullptr : positions.data(),
                                  n_tokens, /*start_pos=*/0);
    std::vector<float> logits(static_cast<std::size_t>(model->n_vocab()));
    backend->copy_to_host(logits.data(), logits_device, model->n_vocab() * 4);

    std::FILE *out = std::fopen(positional[2].c_str(), "wb");
    if (out == nullptr) {
      throw pocketllm::Error("cannot write '" + positional[2] + "'");
    }
    const std::size_t written = std::fwrite(logits.data(), sizeof(float), logits.size(), out);
    const bool short_write = written != logits.size();
    std::fclose(out);
    if (short_write) {
      throw pocketllm::Error("short write to '" + positional[2] + "'");
    }
    std::printf("n_tokens %lld\n", static_cast<long long>(n_tokens));
    std::printf("n_embd_inp %lld\n", static_cast<long long>(row));
    std::printf("n_vocab %lld\n", static_cast<long long>(model->n_vocab()));
    return 0;
  } catch (const pocketllm::Error &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
