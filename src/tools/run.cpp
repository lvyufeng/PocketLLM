/* Generate text with the engine, for watching it work.
 *
 * Internal, like the other two tools: it links the engine's pieces directly so
 * that it can print *inside* the loop -- the prompt tokens, the first logits'
 * top candidates, each token as it is chosen.  `llama-cli` can be diffed
 * against, but it cannot be interrupted half a token in to say which candidate
 * it was choosing between, and that is the question a wrong first token asks.
 *
 * Greedy by default, and the sampling flags are here so the loop that a user
 * actually runs can be watched too.  What greedy is *for* is the property that
 * is hard to get right -- that the distribution the model produces is the same
 * one llama.cpp's does -- and greedy makes that a single number per step
 * instead of a distribution comparison.  Sampling is not a different model,
 * only a different draw from it, so the two paths share everything up to the
 * choice of token.
 *
 * The RNG is the host's, here as in `cli.py`: the engine takes a uniform
 * variate and holds none of its own, so a `--seed` is this tool's
 * `std::mt19937_64` and not a property of the library.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "kernel/kernels.h"
#include "model/qwen3.h"
#include "runtime/status.h"
#include "tokenizer/bpe.h"

namespace {

/* `/mnt/data1/models/qwen3-0.6b-f16.gguf` is 1.5 GB and the model is bound
 * whole, so the tool wants the token count bounded before it starts rather than
 * after it has been running for a minute. */
constexpr int kDefaultSteps = 32;

void print_top(const float *logits, int64_t n, int k) {
  std::vector<int32_t> ids(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    ids[static_cast<std::size_t>(i)] = static_cast<int32_t>(i);
  }
  const int take = static_cast<int>(std::min<int64_t>(k, n));
  std::partial_sort(ids.begin(), ids.begin() + take, ids.end(), [&](int32_t a, int32_t b) {
    return logits[a] > logits[b];
  });
  for (int i = 0; i < take; ++i) {
    const int32_t id = ids[static_cast<std::size_t>(i)];
    std::fprintf(stderr, "    %6d  %10.4f\n", id, logits[id]);
  }
}

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <checkpoint.gguf> [--prompt TEXT] [--steps N] [--device cpu|cuda]\n"
               "       [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n"
               "       [--print-top K]   # list the first position's top candidates\n"
               "       %s <checkpoint.gguf> --tokens 1 2 3   # ids, bypassing the tokenizer\n",
               argv0, argv0);
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    return usage(argv[0]);
  }

  const std::string path = argv[1];
  std::string prompt = "The capital of France is";
  int steps = kDefaultSteps;
  /* The diagnostic list of top candidates, renamed from `--top` now that
   * `--top-k` is a sampling flag: two options an editor completes to `--top`
   * is a bug reported as "my sampling does nothing". */
  int print_top_n = 5;
  std::string device = "cpu";
  float temperature = 0.0F;
  int64_t top_k = 0;
  float top_p = 1.0F;
  float min_p = 0.0F;
  uint64_t seed = 0;
  bool have_seed = false;
  std::vector<int32_t> literal_tokens;
  bool have_literal = false;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    } else if (arg == "--print-top" && i + 1 < argc) {
      print_top_n = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    } else if (arg == "--top-k" && i + 1 < argc) {
      top_k = std::stoll(argv[++i]);
    } else if (arg == "--top-p" && i + 1 < argc) {
      top_p = std::stof(argv[++i]);
    } else if (arg == "--min-p" && i + 1 < argc) {
      min_p = std::stof(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
      have_seed = true;
    } else if (arg == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (arg == "--tokens") {
      have_literal = true;
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        literal_tokens.push_back(std::stoi(argv[++i]));
      }
    } else {
      return usage(argv[0]);
    }
  }

  try {
    pocketllm::GgufReader checkpoint(path);
    const pocketllm::Tokenizer tokenizer(checkpoint);
    /* The tool is the one place that picks a backend from the command line, so
     * that the same binary can run either path without a rebuild. */
    auto backend = pocketllm::kernel::make_backend(device);
    auto model = pocketllm::Qwen3Model::load(checkpoint, *backend);

    std::vector<int32_t> tokens =
        have_literal ? literal_tokens : tokenizer.encode(prompt, /*add_special=*/false,
                                                         /*parse_special=*/true);
    if (tokens.empty()) {
      std::fprintf(stderr, "the prompt tokenized to nothing\n");
      return 1;
    }

    std::fprintf(stderr, "prompt tokenized to %zu token(s):", tokens.size());
    for (const int32_t id : tokens) {
      std::fprintf(stderr, " %d", id);
    }
    std::fprintf(stderr, "\n");

    std::fputs(prompt.c_str(), stdout);
    std::fflush(stdout);

    /* The prompt goes through the graph once, as a batch. Every token in it is
     * attended to by the ones after it, which is what makes the prefill one
     * pass rather than n. */
    const pocketllm::kernel::DeviceBuffer logits_device = model->forward(
        tokens.data(), static_cast<int64_t>(tokens.size()), /*start_pos=*/0);
    std::vector<float> logits_host(static_cast<std::size_t>(model->n_vocab()));
    backend->copy_to_host(logits_host.data(), logits_device, model->n_vocab() * 4);
    const float *logits = logits_host.data();

    /* Seeded from a fixed default when `--seed` is absent, so two runs of the
     * same command produce the same text.  A tool whose output moved between
     * invocations would make every comparison in this file's docstring
     * impossible to repeat. */
    std::mt19937_64 rng(have_seed ? seed : 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<int64_t> order(static_cast<std::size_t>(model->n_vocab()));

    int32_t next = 0;
    for (int step = 0; step < steps; ++step) {
      if (temperature > 0.0F) {
        /* A copy, so the diagnostic below still prints the model's own logits:
         * `print_top` is answering "what did the model prefer", and a
         * temperature-scaled ranking changes the magnitudes it displays.  The
         * order is the same either way -- the transform is monotone -- so this
         * is about the numbers a reader is looking at. */
        std::vector<float> scaled(logits, logits + model->n_vocab());
        pocketllm::kernel::logits_temperature(scaled.data(), scaled.data(), model->n_vocab(),
                                              temperature);
        /* `topk_sample` writes an int64 and `next` is an int32: the cast that
         * would compile here is a four-byte-write-past-the-end of a local, so
         * the kernel writes into a correctly sized value and this narrows it.
         * The vocabulary is far below `INT32_MAX`, so the narrowing is exact. */
        int64_t sampled = 0;
        pocketllm::kernel::topk_sample(scaled.data(), model->n_vocab(),
                                       static_cast<float>(uniform(rng)), top_k, top_p, min_p,
                                       order.data(), &sampled);
        next = static_cast<int32_t>(sampled);
      } else {
        next = 0;
        /* Strictly greater, so a tie takes the lower id -- the same rule
         * `pocketllm_argmax` documents, and the same one llama.cpp's greedy
         * sampler applies.  Reached through `kernel::argmax` rather than a loop
         * here, so the greedy path and `pocketllm_argmax` cannot drift. */
        int64_t best = 0;
        pocketllm::kernel::argmax(logits, model->n_vocab(), &best);
        next = static_cast<int32_t>(best);
      }
      if (step == 0 && print_top_n > 0) {
        std::fprintf(stderr, "top %d at the first generated position:\n", print_top_n);
        print_top(logits, model->n_vocab(), print_top_n);
      }

      /* The ids as well as the text, on stderr, so that a test can diff the
       * sequence against llama.cpp's without re-deriving it by decoding. Two
       * implementations that agree on the text but not on the ids would still
       * be a real divergence -- a different spelling of the same string is the
       * same output and not the same model. */
      std::fprintf(stderr, step == 0 ? "[%d" : " %d", next);

      const std::string piece = tokenizer.decode({next});
      std::fputs(piece.c_str(), stdout);
      std::fflush(stdout);

      /* `cache_length()` is the position the *next* token occupies, and it is
       * read back from the model rather than counted here: the model advances
       * it inside `forward`, so a local counter would be a second copy of the
       * same fact that can disagree with it. */
      const int32_t one = next;
      const pocketllm::kernel::DeviceBuffer next_logits =
          model->forward(&one, 1, model->cache_length());
      backend->copy_to_host(logits_host.data(), next_logits, model->n_vocab() * 4);
      logits = logits_host.data();
    }

    std::fprintf(stderr, "]\n");
    std::fputs("\n", stdout);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}