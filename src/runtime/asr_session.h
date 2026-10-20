/* A speech session: one text model, one audio tower, and the composition that joins them.
 *
 * The same three-part shape `VlmSession` has -- text before, the media span as
 * embeddings, text after -- over a graph that is simpler in every place the vision one
 * is fiddly:
 *
 *   - **One position per token.**  `qwen3a`'s text model is a plain 1-D rotary, so
 *     there is no M-RoPE: a span of `n` tokens advances the position counter by `n`, not
 *     by a grid side.  The vision path's "576 tokens cost 24 positions" has no
 *     counterpart here.
 *   - **Causally decoded**, like the vision span and for the same measured reason:
 *     `mtmd_decode_use_non_causal` answers true only for the Gemma projectors.
 *   - **One span per window.**  A signal longer than 800 mel frames becomes several
 *     windows, and each is its own embedding span with its own `<|audio_start|>` and
 *     `<|audio_end|>` -- they are *not* concatenated into one long span, and a caller
 *     that flattened them would build a prompt the reference never produces.
 *
 * The window count is a function of the sample count alone, so `windows_for` answers
 * it before anything is encoded -- which is what a caller sizing a budget wants.
 */

#ifndef POCKETLLM_RUNTIME_ASR_SESSION_H
#define POCKETLLM_RUNTIME_ASR_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/qwen3a_audio.h"
#include "runtime/session.h"

namespace pocketllm {

class AsrSession {
 public:
  /* Open an audio tower and a text model on one device.  Throws `Error` with a
   * caller-facing message, the same convention `Session::open` uses. */
  static std::unique_ptr<AsrSession> open(const std::string &mmproj_path,
                                          const std::string &ggml_path, const std::string &backend);

  ~AsrSession();
  AsrSession(const AsrSession &) = delete;
  AsrSession &operator=(const AsrSession &) = delete;

  const Qwen3Audio &tower() const { return *tower_; }
  Session &text() { return *text_; }

  /* The marker this composition splits the prompt on.  One marker stands for all the
   * audio in the request: a signal long enough to need two windows is still one place
   * in the prompt. */
  static const char *marker() { return "<__media__>"; }

  /* Which tokens the composition puts around a span, so a caller can build or check a
   * prompt without knowing this projector's spelling. */
  static const char *audio_start() { return "<|audio_start|>"; }
  static const char *audio_end() { return "<|audio_end|>"; }

  /* How many inference windows `n_samples` becomes. */
  int64_t windows_for(int64_t n_samples) const;

  /* What one `compose` call did, beyond the logits. */
  struct Composition {
    int64_t tokens_before = 0;
    int64_t audio_windows = 0;
    /* Image -- well, audio -- tokens per window: `conv_out_w_ * (frames / 100)`. */
    int64_t tokens_per_window = 0;
    int64_t tokens_after = 0;
    /* The position the composition ended at.  For this graph that is also the number of
     * cache rows, because one position is one token -- stated rather than left implicit
     * because the vision path's two numbers are not the same one. */
    int64_t positions = 0;
    std::vector<float> logits;
  };

  /* Run the whole path and return the logits of the last token.
   *
   * `prompt` holds exactly one `marker()`; `pcm` is raw float32 mono at the
   * checkpoint's own rate.  The cache is reset first: this is a prefill, so a caller
   * that wants a second turn starts one here rather than continuing. */
  Composition compose(const std::string &prompt, const float *pcm, int64_t n_samples);

  /* The composition and then greedy decoding, returned as the tokens produced.
   *
   * Decoding belongs here rather than in the caller because the *session* is what owns
   * the cache the composition filled: a caller that had only the logits would have to
   * reopen the text model to continue from them, which is a second session reading a
   * cache it did not build.  `max_tokens` bounds the loop; stop at EOS if the vocabulary
   * declares one.
   *
   * Greedy rather than sampled, and not as a placeholder: it is the decode the reference
   * CLI runs by default, so it is what a transcript can be compared against. */
  std::vector<int32_t> generate(const std::string &prompt, const float *pcm, int64_t n_samples,
                                int64_t max_tokens);

 private:
  AsrSession(std::unique_ptr<Session> text, std::unique_ptr<Qwen3Audio> tower);

  std::unique_ptr<Session> text_;
  std::unique_ptr<Qwen3Audio> tower_;
  AudioFrontendParams params_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_ASR_SESSION_H */