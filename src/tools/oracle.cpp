/* llama.cpp's logits for a token sequence, as JSON.
 *
 * This is the model's oracle, and the reason it is a program rather than a
 * `ctypes` binding: `llama_context_params` has twenty-nine fields and gains them
 * without notice, so a transcription is a struct that is right on the day it is
 * written and silently wrong afterwards. The failure mode is not a compile
 * error but a garbage read that segfaults inside `llama_init_from_model`, which
 * is how this file came to exist. Compiling against the header makes the
 * compiler the one that knows the layout, and it knows it correctly on every
 * version.
 *
 * Built by hand rather than by CMake -- see `tests/native/test_forward.py`,
 * which compiles it if it is not already present. The engine's own build must
 * not depend on a llama.cpp checkout: `libpocketllm.so` is the deliverable and
 * this is a test instrument, so the dependency lives with the test.
 *
 * Two modes:
 *
 *   `oracle <model> <token>...`              the logits for the last token
 *   `oracle <model> <token>... --steps N`    N greedily generated tokens
 *
 * The second exists because `llama-cli` is a *chat client*: given a prompt it
 * wraps it in the checkpoint's chat template and answers a different question
 * than the engine was asked. Driving the low-level API here means the
 * comparison is between two implementations of the same computation and not
 * between two prompt-formatting conventions.
 *
 * ## Flash attention is pinned off, and that is not a default left unset
 *
 * `llama_context_default_params()` sets `flash_attn_type = AUTO`, and AUTO
 * resolves to *enabled* wherever the backend has a kernel -- including on the
 * CPU path this oracle pins itself to. That is not a cosmetic choice of kernel:
 * flash attention computes the softmax over the attention span with a running
 * maximum and a rescaled merge, which is a different reduction from the one
 * shift a plain softmax uses, and the two disagree far enough to flip a
 * near-tie. Measured on this checkpoint, llama.cpp with flash attention on and
 * off produces greedy sequences that differ at the second token --
 * `[..., 11, 323, ...]` against `[..., 13, 576, ...]` -- where the top-2 margin
 * is 0.0925 against 0.0196 and the logits move by up to 1.16. It is
 * near-invisible on f16 (0.0004 of the logit spread), which is why the f16
 * tests never saw it.
 *
 * The engine's CPU attention kernel applies one shift per score row
 * (`src/kernel/kernels.cpp`), so an oracle left on AUTO would be comparing a
 * full-softmax answer against a flash one and reporting the difference as the
 * engine's error. Pinning the convention here makes the oracle ask the question
 * the CPU kernel answers; `--flash-attn on` is the other side of it, which the
 * CUDA backend's block-reduced attention lands on and
 * `test_quantized_forward.py` names per backend.
 *
 * Output is `{"n_vocab": N, "tokens": [...], "logits": [f, ...]}` on stdout,
 * one line, so a Python caller can read it without a parser in C. The logits
 * are printed with `%.9g`, which round-trips a float exactly -- a shortened
 * form would put a bound on the comparison that has nothing to do with either
 * implementation.
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <llama.h>

namespace {

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <model.gguf> <token> [<token> ...] [--steps N] "
               "[--flash-attn on|off|auto]\n",
               argv0);
  return 2;
}

/* The larger of two floats, ties going to the lower index -- the same rule
 * `pocketllm_argmax` documents, applied over the whole vocabulary. Spelled out
 * rather than using `std::max_element` because the tie rule is part of the
 * answer: two logits exactly equal are a real case on a quantized checkpoint
 * and the two implementations have to break it the same way. */
int32_t argmax(const float *logits, int32_t n) {
  int32_t best = 0;
  for (int32_t i = 1; i < n; ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    return usage(argv[0]);
  }

  const char *model_path = argv[1];
  std::vector<llama_token> tokens;
  int steps = 0;
  /* The full-softmax convention by default -- see the file header for what the
   * two conventions disagree about and why the CPU path needs this one. */
  llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--steps" && i + 1 < argc) {
      steps = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::string(argv[i]) == "--flash-attn" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "on") {
        flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
      } else if (mode == "off") {
        flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      } else if (mode == "auto") {
        flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
      } else {
        std::fprintf(stderr, "--flash-attn takes on, off or auto, not '%s'\n", mode.c_str());
        return 2;
      }
    } else {
      tokens.push_back(static_cast<llama_token>(std::strtol(argv[i], nullptr, 10)));
    }
  }
  if (tokens.empty()) {
    return usage(argv[0]);
  }

  /* Everything llama.cpp would say, suppressed. It is loud on stderr, and the
   * caller is diffing stdout. */
  llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  /* CPU only, and pinned rather than discovered: the engine under test is the
   * CPU path, so an oracle that silently ran a GPU kernel would be measuring
   * the wrong thing. The device list is built from ggml's own CPU device, which
   * also sidesteps `ggml_backend_load_all`'s search relative to the working
   * directory. */
  ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (cpu == nullptr) {
    std::fprintf(stderr, "llama.cpp has no CPU backend registered\n");
    return 1;
  }
  ggml_backend_dev_t devices[] = {cpu, nullptr};
  mparams.devices = devices;
  mparams.n_gpu_layers = 0;

  llama_model *model = llama_model_load_from_file(model_path, mparams);
  if (model == nullptr) {
    std::fprintf(stderr, "llama.cpp could not load %s\n", model_path);
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 512;
  cparams.n_ubatch = 512;
  /* One thread, so that the reduction order inside llama.cpp's own kernels is
   * fixed. A thread-count-dependent sum would make the oracle a little
   * different every run, and the tolerance would have to absorb that too. */
  cparams.n_threads = 1;
  cparams.n_threads_batch = 1;
  cparams.offload_kqv = false;
  cparams.no_perf = true;
  cparams.flash_attn_type = flash_attn;

  llama_context *ctx = llama_init_from_model(model, cparams);
  if (ctx == nullptr) {
    std::fprintf(stderr, "llama.cpp could not create a context\n");
    return 1;
  }

  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

  /* The prompt in one batch, exactly as the logits mode does it. Only the last
   * position's logits are needed, and `llama_batch_get_one` marks the final
   * token as the one to produce them for. */
  llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
  if (llama_decode(ctx, batch) != 0) {
    std::fprintf(stderr, "llama_decode failed\n");
    return 1;
  }
  const float *logits = llama_get_logits_ith(ctx, -1);
  if (logits == nullptr) {
    std::fprintf(stderr, "llama.cpp produced no logits\n");
    return 1;
  }

  std::vector<llama_token> generated;
  for (int step = 0; step < steps; ++step) {
    const llama_token next = argmax(logits, n_vocab);
    generated.push_back(next);
    /* One token at a time from here, which is the decode loop the engine's own
     * incremental path is being checked against. A single-token decode is where
     * a KV cache that is not per layer passes the prefill and fails here. */
    llama_batch one = llama_batch_get_one(&generated.back(), 1);
    if (llama_decode(ctx, one) != 0) {
      std::fprintf(stderr, "llama_decode failed at step %d\n", step);
      return 1;
    }
    logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
      std::fprintf(stderr, "llama.cpp produced no logits at step %d\n", step);
      return 1;
    }
  }

  std::printf("{\"n_vocab\": %d, \"tokens\": [", n_vocab);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    std::printf("%s%d", i ? ", " : "", tokens[i]);
  }
  std::printf("], \"generated\": [");
  for (std::size_t i = 0; i < generated.size(); ++i) {
    std::printf("%s%d", i ? ", " : "", generated[i]);
  }
  std::printf("], \"logits\": [");
  for (int32_t i = 0; i < n_vocab; ++i) {
    std::printf("%s%.9g", i ? ", " : "", static_cast<double>(logits[i]));
  }
  std::printf("]}\n");

  /* Not freed: the process is about to exit, and freeing a context whose graph
   * still holds references is a way to turn a clean run into a crash at exit.
   * `stdout` is flushed first, so a caller that reads the JSON gets it. */
  std::fflush(stdout);
  return 0;
}