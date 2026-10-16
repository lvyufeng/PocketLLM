/* Run a whole vision-language eval: a prompt with one image, and the next token.
 *
 * Every piece of this path is already in the tree -- the tower, the text graph's
 * input width, its per-section rotary and its non-causal span -- and this is the
 * call that composes them, in the order `libmtmd` composes them:
 *
 *   1. the text before the image, as tokens, ending in `<|vision_start|>`;
 *   2. the image span, as an *embeddings* batch: the tower's planes, one position
 *      per rotary section per token, and non-causal attention;
 *   3. the text after it, starting with `<|vision_end|>`.
 *
 * Three calls rather than one, because the span's attention is a property of a
 * batch and not of a range inside one -- the same shape llama.cpp exposes as a
 * context flag -- and because that is how the reference drives it.  The two
 * engines are then two implementations of one contract, which is what makes the
 * comparison in `tests/native/test_vlm.py` mean something.
 *
 * Usage:
 *   pocketllm-vlm <mmproj.gguf> <text_model.gguf> <rgb.bin> <width> <height>
 *                 "<prompt>" <logits.bin> [--planes FILE] [--device NAME]
 *
 * `--planes` replaces the tower with a file: `n_planes` blocks of `n_tokens *
 * projection_dim` float32, plane-major, which is exactly what the tower writes.
 * It is a bisect instrument -- the engine fed llama.cpp's own features isolates
 * whether a disagreement is the tower's or the composition's -- and it is why the
 * image arguments are optional when it is given.
 *
 * The prompt holds the media marker where the image goes -- `mtmd_default_marker()`
 * in llama.cpp, which is the `<__media__>` below -- and this tool splits it there.
 * More than one marker is refused: the flow above is one image per call.
 *
 * The image is handed over as raw RGB bytes for the reason the tower's own oracle
 * gives, and at the checkpoint's own `image_size` square for the reason the tower
 * gives: the resize that would let it take another size is not implemented, and
 * this tool says so rather than resampling.
 *
 * On the CPU this needs `POCKETLLM_CPU_KV_F32=1`: the span is non-causal, and that
 * arm refuses an f16 cache rather than widening it.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/qwen3.h"
#include "model/qwen3vl_vision.h"
#include "runtime/status.h"
#include "tokenizer/bpe.h"

namespace {

/* llama.cpp's `mtmd_default_marker()`.  Spelled here rather than asked for,
 * because asking would mean linking `libmtmd` into the engine for a string. */
constexpr const char *kMarker = "<__media__>";

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <mmproj.gguf> <text_model.gguf> <rgb.bin> <width> <height> "
               "<prompt> <logits.bin> [--planes FILE] [--skip-vision] [--device cpu|cuda]\n"
               "  the image arguments are still required with --planes, and unused;\n"
               "  --skip-vision does not load the tower at all, which is the bisect for\n"
               "  whether a disagreement is the tower's or the composition's\n",
               argv0);
  return 2;
}

std::vector<uint8_t> read_bytes(const std::string &path) {
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
  std::string planes_path;
  bool skip_vision = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--device") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      device = argv[++i];
    } else if (argument == "--help" || argument == "-h") {
      return usage(argv[0]);
    } else if (argument == "--skip-vision") {
      skip_vision = true;
    } else if (argument == "--planes") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      planes_path = argv[++i];
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() != 7) {
    return usage(argv[0]);
  }

  try {
    pocketllm::GgufReader mmproj(positional[0]);
    pocketllm::GgufReader text_model(positional[1]);
    auto backend = pocketllm::kernel::make_backend(device);
    std::unique_ptr<pocketllm::Qwen3VlVision> vision_holder;
    if (!skip_vision) {
      vision_holder = pocketllm::Qwen3VlVision::load(mmproj, *backend);
    }
    pocketllm::Qwen3VlVision *vision = vision_holder.get();
    auto model = pocketllm::Qwen3Model::load(text_model, *backend);
    pocketllm::Tokenizer tokenizer(text_model);

    /* With `--skip-vision` the tower is not loaded at all, so the geometry it
     * would have stated is taken from the checkpoint's own metadata instead --
     * the same numbers, read the same way `Qwen3VlVision::load` reads them. */
    int64_t geom_deepstack = 0;
    if (const pocketllm::GgufValue *value = mmproj.find("clip.vision.is_deepstack_layers")) {
      if (const pocketllm::GgufArray *taps = std::get_if<pocketllm::GgufArray>(value)) {
        for (uint8_t tap : taps->bools) {
          if (tap != 0) {
            ++geom_deepstack;
          }
        }
      }
    }
    const int64_t geom_planes = vision != nullptr ? vision->n_planes() : 1 + geom_deepstack;
    const int64_t geom_side = vision != nullptr
                                  ? vision->merged_side()
                                  : mmproj.get_int("clip.vision.image_size", 0) /
                                        mmproj.get_int("clip.vision.patch_size", 1) /
                                        mmproj.get_int("clip.vision.spatial_merge_size", 1);
    const int64_t geom_tokens = geom_side * geom_side;
    const int64_t geom_dim = vision != nullptr
                                 ? vision->projection_dim()
                                 : mmproj.get_int("clip.vision.projection_dim", 0);
    const std::size_t plane_floats =
        static_cast<std::size_t>(geom_planes * geom_tokens * geom_dim);
    std::vector<float> planes;
    if (!planes_path.empty()) {
      /* The bisect path: features from a file instead of from the tower. */
      const std::vector<uint8_t> raw = read_bytes(planes_path);
      if (raw.size() != plane_floats * 4) {
        throw pocketllm::Error("'" + planes_path + "' is " + std::to_string(raw.size()) +
                               " bytes, and " + std::to_string(plane_floats) +
                               " floats were expected");
      }
      planes.resize(plane_floats);
      std::memcpy(planes.data(), raw.data(), raw.size());
    } else {
      const int64_t width = std::atoll(positional[3].c_str());
      const int64_t height = std::atoll(positional[4].c_str());
      const int64_t geom_image = mmproj.get_int("clip.vision.image_size", 0);
      if (width != geom_image || height != geom_image) {
        throw pocketllm::Error("this tower takes " + std::to_string(geom_image) + "x" +
                               std::to_string(geom_image) +
                               " images and this one is " + std::to_string(width) + "x" +
                               std::to_string(height) + "; resample it first");
      }
      const std::vector<uint8_t> rgb = read_bytes(positional[2]);
      if (static_cast<int64_t>(rgb.size()) != width * height * 3) {
        throw pocketllm::Error("'" + positional[2] + "' is not width * height * 3 bytes");
      }
      planes.resize(plane_floats);
      vision->encode(rgb.data(), height, width, planes.data());
    }

    const std::string prompt = positional[5];
    const std::size_t marker = prompt.find(kMarker);
    if (marker == std::string::npos || prompt.find(kMarker, marker + 1) != std::string::npos) {
      throw pocketllm::Error("the prompt needs exactly one '" + std::string(kMarker) + "', and this "
                             "tool's flow is one image per call");
    }
    /* The markers around the span are mtmd's for this projector
     * (`PROJECTOR_TYPE_QWEN3VL`), and they are what makes the two engines' token
     * sequences the same sequence rather than merely the same length. */
    const std::string before = prompt.substr(0, marker) + "<|vision_start|>";
    const std::string after = "<|vision_end|>" + prompt.substr(marker + std::strlen(kMarker));
    const std::vector<int32_t> tokens_before = tokenizer.encode(before, /*add_special=*/false,
                                                               /*parse_special=*/true);
    const std::vector<int32_t> tokens_after = tokenizer.encode(after, /*add_special=*/false,
                                                              /*parse_special=*/true);
    if (tokens_before.empty() || tokens_after.empty()) {
      throw pocketllm::Error("the prompt tokenized to nothing on one side of the image");
    }

    /* 1. The text before the image, at its own positions. */
    model->forward(tokens_before.data(), static_cast<int64_t>(tokens_before.size()),
                   /*start_pos=*/0);

    /* 2. The span.  Its positions are llama.cpp's for this projector: one
     * temporal position for the whole span, and each patch's row and column
     * offset by it -- `mtmd_image_tokens_get_decoder_pos`, as measured.  The
     * column is the fast axis, which is the same order the merge gives the
     * tokens. */
    const int64_t span = geom_tokens;
    const int64_t side = geom_side;
    const int64_t span_start = model->cache_length();
    std::vector<int64_t> span_positions(static_cast<std::size_t>(span) * 4);
    for (int64_t i = 0; i < span; ++i) {
      span_positions[static_cast<std::size_t>(i * 4 + 0)] = span_start;
      span_positions[static_cast<std::size_t>(i * 4 + 1)] = span_start + i / side;
      span_positions[static_cast<std::size_t>(i * 4 + 2)] = span_start + i % side;
      span_positions[static_cast<std::size_t>(i * 4 + 3)] = 0;
    }
    /* Causal, like the rest of the graph -- and that is a *measured* fact about
     * this architecture rather than an assumption.  llama.cpp's rule is
     * `mtmd_decode_use_non_causal`, which answers true only for the Gemma
     * projectors; a `qwen3vl_merger` span is decoded causally, so a span token
     * sees the tokens before it and not the ones after.  (The non-causal arm
     * exists and is what the tower's own attention uses -- and a projector that
     * needed it would ask for it here.) */
    model->forward_embeddings(planes.data(), span_positions.data(), span, span_start,
                             /*non_causal=*/false);

    /* 3. The text after it.  Its positions continue from the span's *positions*,
     * which are `side` of them and not `span`, while its cache rows continue from
     * the cache's own length -- the two numbers an image span pulls apart. */
    const int64_t after_base = span_start + side;
    std::vector<int64_t> after_positions(static_cast<std::size_t>(tokens_after.size()) * 4);
    for (std::size_t i = 0; i < tokens_after.size(); ++i) {
      for (int section = 0; section < 4; ++section) {
        after_positions[i * 4 + static_cast<std::size_t>(section)] = after_base + static_cast<int64_t>(i);
      }
    }
    const pocketllm::kernel::DeviceBuffer logits_device =
        model->forward_positions(tokens_after.data(), after_positions.data(),
                                 static_cast<int64_t>(tokens_after.size()), model->cache_length());

    std::vector<float> logits(static_cast<std::size_t>(model->n_vocab()));
    backend->copy_to_host(logits.data(), logits_device, model->n_vocab() * 4);
    std::FILE *out = std::fopen(positional[6].c_str(), "wb");
    if (out == nullptr) {
      throw pocketllm::Error("cannot write '" + positional[6] + "'");
    }
    const std::size_t written = std::fwrite(logits.data(), sizeof(float), logits.size(), out);
    const bool short_write = written != logits.size();
    std::fclose(out);
    if (short_write) {
      throw pocketllm::Error("short write to '" + positional[6] + "'");
    }

    int64_t best = 0;
    for (int64_t i = 1; i < model->n_vocab(); ++i) {
      if (logits[static_cast<std::size_t>(i)] > logits[static_cast<std::size_t>(best)]) {
        best = i;
      }
    }
    const int64_t total = static_cast<int64_t>(tokens_before.size()) + span +
                          static_cast<int64_t>(tokens_after.size());
    std::printf("n_tokens %lld\n", static_cast<long long>(total));
    std::printf("n_positions %lld\n", static_cast<long long>(after_base + static_cast<int64_t>(tokens_after.size())));
    std::printf("n_vocab %lld\n", static_cast<long long>(model->n_vocab()));
    std::printf("argmax %lld\n", static_cast<long long>(best));
    std::fprintf(stderr, "tokens_before:");
    for (const int32_t id : tokens_before) {
      std::fprintf(stderr, " %d", id);
    }
    std::fprintf(stderr, "\ntokens_after:");
    for (const int32_t id : tokens_after) {
      std::fprintf(stderr, " %d", id);
    }
    std::fprintf(stderr, "\n");
    return 0;
  } catch (const pocketllm::Error &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
