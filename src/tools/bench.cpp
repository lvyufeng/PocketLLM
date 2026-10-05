/* Measure the engine's throughput, the way `llama-bench` measures llama.cpp's.
 *
 * This exists because there was no number to argue with.  `pocketllm-run` prints
 * generated text and nothing else, so the only way to time the engine was to
 * wrap it in `/usr/bin/time` and subtract the checkpoint load, which conflates
 * prefill, decode and the load into one figure.  A performance change that
 * cannot be measured is a change nobody can review, and the claim this tool
 * exists to test -- whether the engine is faster than llama.cpp -- needs both
 * halves timed the same way.
 *
 * Two tests, matching llama-bench's names so the two tables can be read side by
 * side:
 *
 *   pp<N>  prefill: one forward over N tokens, cache dropped first, timed.
 *   tg<N>  decode:  N forwards of one token each, `cache_length()` advancing
 *                  between them, timed as a whole.
 *
 * The prompt is synthetic token ids rather than text.  That is deliberate:
 * tokenization is not what is being measured, and a real prompt would make the
 * number depend on the tokenizer while llama-bench's comparable figure does not.
 *
 * The cache is reset between repetitions with `Qwen3Model::reset`, which drops
 * the KV entries but keeps the buffers, so a repetition is a fresh prefill and
 * not an ever-growing context.
 *
 * ## Reading the output
 *
 * A human table goes to stdout, and each row is also emitted as a line the test
 * harness can parse:
 *
 *     bench pp 32 123.456
 *     bench tg 32 12.3456
 *
 * The thread count is not a flag of its own; it is `$POCKETLLM_CPU_THREADS`, and
 * the tool prints what it resolved so a result is never ambiguous about how many
 * cores produced it.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "kernel/parallel.h"
#include "model/qwen3.h"
#include "runtime/status.h"
#include "tokenizer/bpe.h"

namespace {

using pocketllm::kernel::Backend;
using pocketllm::kernel::DeviceBuffer;

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <checkpoint.gguf> [--prompt-ids N] [--pp N] [--tg N] [--reps N]\n"
               "         [--warmup N] [--device cpu|cuda]\n"
               "  threads come from $POCKETLLM_CPU_THREADS (default: all cores)\n",
               argv0);
  return 2;
}

/* A deterministic prompt of `n` ids, all inside a small vocabulary range so the
 * same ids are valid for any checkpoint this is pointed at.  The values do not
 * matter for timing; only that they vary, so the embedding gather and the cache
 * are not accidentally hitting one row. */
std::vector<int32_t> synthetic_tokens(int64_t n) {
  std::vector<int32_t> tokens(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    tokens[static_cast<std::size_t>(i)] = static_cast<int32_t>(64 + (i * 7) % 64);
  }
  return tokens;
}

double median(std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  return values[mid];
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    return usage(argv[0]);
  }
  const std::string path = argv[1];

  int64_t pp = 32;
  int64_t tg = 32;
  int64_t reps = 5;
  int64_t warmup = 1;
  std::string device = "cpu";

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--pp" && has_value) {
      pp = std::stoll(argv[++i]);
    } else if (arg == "--tg" && has_value) {
      tg = std::stoll(argv[++i]);
    } else if (arg == "--reps" && has_value) {
      reps = std::stoll(argv[++i]);
    } else if (arg == "--warmup" && has_value) {
      warmup = std::stoll(argv[++i]);
    } else if (arg == "--device" && has_value) {
      device = argv[++i];
    } else {
      return usage(argv[0]);
    }
  }
  if (pp < 0 || tg < 0 || reps < 1) {
    return usage(argv[0]);
  }

  try {
    pocketllm::GgufReader checkpoint(path);
    const pocketllm::Tokenizer tokenizer(checkpoint);
    static_cast<void>(tokenizer); /* the checkpoint must be one this build can open */
    auto backend = pocketllm::kernel::make_backend(device);
    auto model = pocketllm::Qwen3Model::load(checkpoint, *backend);

    const std::vector<int32_t> prompt = synthetic_tokens(pp == 0 ? 1 : pp);
    const int64_t threads = pocketllm::kernel::cpu_thread_count();

    /* One throwaway of each shape, so the first `reps` run is not measuring a
     * cold pool, an unpopulated page table or a cold instruction cache. */
    for (int64_t i = 0; i < warmup; ++i) {
      model->reset();
      model->forward(prompt.data(), static_cast<int64_t>(prompt.size()), 0);
      const int32_t one = prompt[0];
      model->forward(&one, 1, model->cache_length());
    }
    backend->synchronize();

    std::printf("| model | backend | threads | test | t/s |\n");
    std::printf("|---|---|---:|---|---:|\n");

    if (pp > 0) {
      std::vector<double> rates;
      rates.reserve(static_cast<std::size_t>(reps));
      for (int64_t i = 0; i < reps; ++i) {
        model->reset();
        const auto start = std::chrono::steady_clock::now();
        model->forward(prompt.data(), static_cast<int64_t>(prompt.size()), 0);
        backend->synchronize();
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        rates.push_back(static_cast<double>(pp) / seconds);
      }
      const double rate = median(rates);
      std::printf("| %s | %s | %lld | pp%lld | %.2f |\n", path.c_str(), backend->name(),
                  static_cast<long long>(threads), static_cast<long long>(pp), rate);
      std::printf("bench pp %lld %.6f\n", static_cast<long long>(pp), rate);
    }

    if (tg > 0) {
      std::vector<double> rates;
      rates.reserve(static_cast<std::size_t>(reps));
      for (int64_t i = 0; i < reps; ++i) {
        /* A prefill first: decode is measured against a cache that already
         * holds the prompt, which is the state a real generation is in for every
         * token after the first.  The cache is dropped at the top of the next
         * repetition. */
        model->reset();
        model->forward(prompt.data(), static_cast<int64_t>(prompt.size()), 0);
        backend->synchronize();
        const auto start = std::chrono::steady_clock::now();
        for (int64_t step = 0; step < tg; ++step) {
          const int32_t one = prompt[static_cast<std::size_t>(step % prompt.size())];
          model->forward(&one, 1, model->cache_length());
        }
        backend->synchronize();
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        rates.push_back(static_cast<double>(tg) / seconds);
      }
      const double rate = median(rates);
      std::printf("| %s | %s | %lld | tg%lld | %.2f |\n", path.c_str(), backend->name(),
                  static_cast<long long>(threads), static_cast<long long>(tg), rate);
      std::printf("bench tg %lld %.6f\n", static_cast<long long>(tg), rate);
    }

    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}