/* Run a vision tower over one image and write its features as float32.
 *
 * Internal, like the other tools: it links the engine's objects directly rather
 * than going through the ABI, which is how a caller reaches a *piece* of a model
 * before the ABI has an opinion about images.  The tower's oracle is llama.cpp's
 * own `mtmd_encode`, driven by `tests/native/mtmd_embed.c`, and the test that
 * compares the two is `tests/native/test_vision_tower.py`.
 *
 * The image arrives as raw RGB bytes for the same reason the oracle's does:
 * whatever produced the pixels has to be able to say exactly what they were, and
 * an image decoder's own choices -- colour space, orientation, chroma rounding --
 * would be a second description of the input that this side would have to match.
 *
 * Usage:
 *   pocketllm-vision <mmproj.gguf> <rgb.bin> <width> <height> <out.bin>
 *                    [--device NAME]
 *
 * The output is `n_planes` planes of `n_tokens x projection_dim` float32 in file
 * order, and those three numbers go to stdout.  `Qwen3VlVision::encode` says why
 * the planes are separate rather than interleaved per token.
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/qwen3vl_vision.h"
#include "runtime/status.h"

namespace {

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <mmproj.gguf> <rgb.bin> <width> <height> <out.bin> "
               "[--device cpu|cuda]\n",
               argv0);
  return 2;
}

/* A whole file as bytes, or a message naming it.  The image is small enough
 * (megabytes) that streaming it would be a second way to get it wrong. */
std::vector<uint8_t> read_file(const std::string &path) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw pocketllm::Error("cannot open '" + path + "'");
  }
  std::vector<uint8_t> bytes;
  uint8_t chunk[65536];
  std::size_t read = 0;
  while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
    bytes.insert(bytes.end(), chunk, chunk + read);
  }
  std::fclose(file);
  return bytes;
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> positional;
  std::string device = "cpu";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--device") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      device = argv[++i];
    } else if (argument == "--help" || argument == "-h") {
      return usage(argv[0]);
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() != 5) {
    return usage(argv[0]);
  }

  try {
    auto reader = std::make_unique<pocketllm::GgufReader>(positional[0]);
    auto backend = pocketllm::kernel::make_backend(device);
    auto model = pocketllm::Qwen3VlVision::load(*reader, *backend);

    const int64_t width = std::atoll(positional[2].c_str());
    const int64_t height = std::atoll(positional[3].c_str());
    const std::vector<uint8_t> rgb = read_file(positional[1]);
    if (static_cast<int64_t>(rgb.size()) != width * height * 3) {
      std::fprintf(stderr, "'%s' is %zu bytes, and %lldx%lld RGB is %lld\n", positional[1].c_str(),
                   rgb.size(), static_cast<long long>(width), static_cast<long long>(height),
                   static_cast<long long>(width * height * 3));
      return 2;
    }

    /* Sized from the *input's* shape, not from the checkpoint's own: `encode` resizes
     * to the grid this shape needs, and a buffer sized for `n_tokens()` (the
     * checkpoint's square) is a write past the end for any other one.  The tool takes
     * any size for that reason; `tokens_for` is the same call the encoder makes. */
    const int64_t tokens = model->tokens_for(width, height);
    const int64_t plane = tokens * model->projection_dim();
    std::vector<float> features(static_cast<std::size_t>(model->n_planes() * plane));
    model->encode(rgb.data(), height, width, features.data());

    std::FILE *out = std::fopen(positional[4].c_str(), "wb");
    if (out == nullptr) {
      std::fprintf(stderr, "cannot write '%s'\n", positional[4].c_str());
      return 2;
    }
    const std::size_t written =
        std::fwrite(features.data(), sizeof(float), features.size(), out);
    const bool short_write = written != features.size();
    std::fclose(out);
    if (short_write) {
      std::fprintf(stderr, "short write to '%s'\n", positional[4].c_str());
      return 2;
    }

    std::printf("n_planes %lld\n", static_cast<long long>(model->n_planes()));
    std::printf("n_tokens %lld\n", static_cast<long long>(model->n_tokens()));
    std::printf("projection_dim %lld\n", static_cast<long long>(model->projection_dim()));
    return 0;
  } catch (const pocketllm::Error &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
