/* Run a whole audio encode: PCM in, the tower's features out.
 *
 * This is the audio side's equivalent of `pocketllm-vision`, and it exists for the same
 * two reasons: it is how the tower is compared against llama.cpp, and it is what a
 * caller without a C++ build can drive.
 *
 * What it composes is the front end and the tower, in the order the reference does:
 *
 *   1. the log-mel front end over the raw samples (`model/audio_frontend.h`), which
 *      picks its own window count;
 *   2. for each 800-frame window, the tower (`model/qwen3a_audio.h`).
 *
 * The windows are *independent:* a signal longer than one window is encoded as several
 * blocks, and the reference feeds each to the text model in turn rather than
 * concatenating them into one span.  A caller that flattened them would give the text
 * graph a sequence the reference never produced.
 *
 * Usage:
 *   pocketllm-audio <mmproj.gguf> <pcm.f32> <features.bin> [--sample-rate HZ]
 *
 * The PCM is raw float32, mono, little-endian, at the checkpoint's own rate -- 16 kHz
 * here.  Resampling is the caller's: `ffmpeg -ar 16000 -ac 1 -f f32le` is what the
 * tests use, and doing it here would put a resampler in the engine that the reference
 * does not have.
 *
 * The output is `(n_windows, tokens_per_window, projection_dim)` float32, windows
 * concatenated: `n_windows * tokens_for_frames(800) * projection_dim` floats, which the
 * tool prints so the caller does not have to work it out.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/audio_frontend.h"
#include "model/qwen3a_audio.h"
#include "runtime/asr_session.h"
#include "runtime/status.h"

namespace {

int usage(const char *argv0) {
  std::fprintf(stderr, "usage: %s <mmproj.gguf> <pcm.f32> <features.bin> [--device cpu|cuda]\n",
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
  std::string text_model;
  std::string prompt = "Transcribe the audio.";
  int64_t steps = 32;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--device") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      device = argv[++i];
    } else if (argument == "--text-model") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      text_model = argv[++i];
    } else if (argument == "--prompt") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      prompt = argv[++i];
    } else if (argument == "--steps") {
      if (i + 1 >= argc) {
        return usage(argv[0]);
      }
      steps = std::atoll(argv[++i]);
    } else if (argument == "--help" || argument == "-h") {
      return usage(argv[0]);
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() != 3) {
    return usage(argv[0]);
  }

  /* With a text model, this is the whole path: samples in, a transcript out.  Without
   * one it stops at the features, which is what a caller comparing towers wants. */
  if (!text_model.empty()) {
    try {
      auto session = pocketllm::AsrSession::open(positional[0], text_model, device);
      const std::vector<uint8_t> raw = read_bytes(positional[1]);
      std::vector<float> samples(raw.size() / sizeof(float));
      std::memcpy(samples.data(), raw.data(), raw.size());

      const pocketllm::AsrSession::Composition report =
          session->compose(prompt, samples.data(), static_cast<int64_t>(samples.size()));
      const pocketllm::Tokenizer *tokenizer = session->text().tokenizer();
      int32_t next = 0;
      for (std::size_t i = 0; i < report.logits.size(); ++i) {
        if (report.logits[i] > report.logits[static_cast<std::size_t>(next)]) {
          next = static_cast<int32_t>(i);
        }
      }
      std::string text;
      std::vector<int32_t> emitted{next};
      /* Greedy to `--steps`, which is the same decode loop `pocketllm-run` uses and
       * the same one the reference's CLI runs by default.  A vision-language comparison
       * stops at logits because one answer is the point; a transcript needs the tokens
       * after it. */
      for (int64_t step = 1; step < steps; ++step) {
        std::vector<float> logits_step = session->text().forward(&next, 1, nullptr);
        int32_t best = 0;
        for (std::size_t i = 1; i < logits_step.size(); ++i) {
          if (logits_step[i] > logits_step[static_cast<std::size_t>(best)]) {
            best = static_cast<int32_t>(i);
          }
        }
        if (tokenizer != nullptr && tokenizer->special().eos >= 0 && best == tokenizer->special().eos) {
          break;
        }
        next = best;
        emitted.push_back(next);
      }
      /* Decoded once at the end rather than per token: a byte-level BPE piece is not
       * necessarily a string on its own, and decoding each one separately is how a
       * multi-byte character becomes replacement characters. */
      if (tokenizer != nullptr) {
        text = tokenizer->decode(emitted);
      }
      std::printf("text %s\n", text.c_str());
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
      return 1;
    }
  }

  try {
    pocketllm::GgufReader mmproj(positional[0]);
    auto backend = pocketllm::kernel::make_backend(device);

    const std::string refusal = pocketllm::audio_arch_refusal(mmproj);
    if (!refusal.empty()) {
      throw pocketllm::Error("'" + positional[0] + "': " + refusal);
    }
    auto tower = pocketllm::Qwen3Audio::load(mmproj, *backend);

    const std::vector<uint8_t> raw = read_bytes(positional[1]);
    if (raw.size() % sizeof(float) != 0) {
      throw pocketllm::Error("'" + positional[1] + "' is not a whole number of float32 samples");
    }
    std::vector<float> samples(raw.size() / sizeof(float));
    std::memcpy(samples.data(), raw.data(), raw.size());
    if (samples.empty()) {
      throw pocketllm::Error("'" + positional[1] + "' holds no samples");
    }

    const pocketllm::AudioFrontendParams params = tower->frontend_params();
    const int64_t windows =
        pocketllm::audio_window_count(static_cast<int64_t>(samples.size()), params);
    if (windows <= 0) {
      throw pocketllm::Error("the signal is too short to make one window");
    }

    std::FILE *out = std::fopen(positional[2].c_str(), "wb");
    if (out == nullptr) {
      throw pocketllm::Error("cannot write '" + positional[2] + "'");
    }
    int64_t written = 0;
    int64_t first_tokens = 0;
    for (int64_t window = 0; window < windows; ++window) {
      /* The front end windows the *whole* signal, so each call is given every sample and
       * an index rather than a slice: slicing first would move every frame relative to
       * the reflection padding. */
      const pocketllm::AudioMel mel = pocketllm::audio_log_mel_window(
          samples.data(), static_cast<int64_t>(samples.size()), params, window);
      const int64_t tokens = tower->tokens_for_frames(mel.n_len);
      if (window == 0) {
        first_tokens = tokens;
      }
      std::vector<float> features(static_cast<std::size_t>(tokens * tower->projection_dim()));
      tower->encode(mel, features.data());
      const std::size_t count = std::fwrite(features.data(), sizeof(float), features.size(), out);
      if (static_cast<std::size_t>(count) != features.size()) {
        std::fclose(out);
        throw pocketllm::Error("short write to '" + positional[2] + "'");
      }
      written += static_cast<int64_t>(features.size());
    }
    std::fclose(out);

    std::printf("n_windows %lld\n", static_cast<long long>(windows));
    std::printf("n_frames %lld\n", static_cast<long long>(windows * 800));
    std::printf("tokens_per_window %lld\n", static_cast<long long>(first_tokens));
    std::printf("projection_dim %lld\n", static_cast<long long>(tower->projection_dim()));
    std::printf("n_floats %lld\n", static_cast<long long>(written));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}