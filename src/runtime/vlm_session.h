/* A vision-language session: one text model, one tower, and the three-call
 * composition that joins them.
 *
 * The composition is the contract, and it is the same shape `libmtmd` uses and
 * the same shape `pocketllm-vlm` drives by hand:
 *
 *   1. the text before the image, as tokens, ending in `<|vision_start|>`;
 *   2. the image span, as an *embeddings* batch: the tower's planes, one position
 *      per rotary section per token;
 *   3. the text after it, starting with `<|vision_end|>`.
 *
 * Three calls rather than one because the span's attention is a property of a
 * batch and not of a range inside one -- the same shape llama.cpp exposes as a
 * context flag around the decode of one image chunk -- and because that is how
 * the reference drives it.
 *
 * This class exists so that the two callers do not each own a copy of that
 * order.  `pocketllm-vlm` is the instrument that writes the logits to a file and
 * `pocketllm_vlm_forward` is the ABI entry point, and both reach the same
 * `compose` below; a disagreement between an instrument and the thing it
 * measures is a class of bug this repository has already paid for once.
 *
 * The image arrives as raw RGB bytes at **any** size.  The tower resizes it, with
 * the checkpoint's own pixel limits and the reference's kernels -- see
 * `model/image_preprocess.h` -- and `image_tokens` reports what a given shape
 * becomes, because that number is what a caller has to know before it asks for
 * logits: it is not a function of the pixel count alone.
 */

#ifndef POCKETLLM_RUNTIME_VLM_SESSION_H
#define POCKETLLM_RUNTIME_VLM_SESSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model/qwen3vl_vision.h"
#include "runtime/session.h"

namespace pocketllm {

class VlmSession {
 public:
  /* Open a tower and a text model, on one device.  Both paths are required: a
   * language model without its projector cannot take an image, and a projector
   * without its language model produces features nothing reads.  Throws `Error`
   * with a caller-facing message, the same convention `Session::open` uses. */
  static std::unique_ptr<VlmSession> open(const std::string &mmproj_path,
                                          const std::string &ggml_path, const std::string &backend);

  ~VlmSession();
  VlmSession(const VlmSession &) = delete;
  VlmSession &operator=(const VlmSession &) = delete;

  const Qwen3VlVision &tower() const { return *tower_; }
  Session &text() { return *text_; }
  const Session &text() const { return *text_; }

  /* How many image tokens an input of this shape becomes, and what the tower
   * resizes it to.  Exposed because a caller sizing a prompt or checking a
   * budget needs the number *before* the call, and because it is the one part of
   * the composition that a caller can observe without running anything. */
  int64_t image_tokens(int64_t width, int64_t height) const {
    return tower_->tokens_for(width, height);
  }

  /* The marker this composition splits the prompt on.  One image per call, so a
   * prompt with two is refused rather than resolved to the first. */
  static const char *marker() { return "<__media__>"; }

  /* What one `compose` call did, beyond the logits: the pieces a caller or an
   * instrument needs to state the shape of the answer.  Returned rather than
   * recomputed because the token counts are not a function of the pixel count
   * alone -- `image_tokens` is the same number only for the same input shape, and
   * the caller already had to pass that. */
  struct Composition {
    int64_t tokens_before = 0; /* the prompt before the marker, plus `<|vision_start|>` */
    int64_t image_tokens = 0;  /* what the tower's grid came to */
    int64_t tokens_after = 0;  /* the prompt after it, plus `<|vision_end|>` */
    /* The position label the composition ended at, which is *not* the number of
     * rows in the cache: an image span writes one row per token but advances the
     * position counter by one per grid cell, so a 576-token span on a 24x24 grid
     * costs 24 positions.  Both numbers are reported because telling them apart is
     * the whole subtlety of this path -- `cache_rows` is the length the three calls
     * left behind. */
    int64_t positions = 0;
    int64_t cache_rows = 0;
    int64_t target_width = 0;  /* what the preprocessing resized the image to */
    int64_t target_height = 0;
    int64_t image_span = 0;    /* the merged grid's side, which is the span's positions */
    std::vector<float> logits;
  };

  /* Run the whole path and return the logits of the last token.
   *
   * `prompt` holds `marker()` where the image goes; `rgb` is `width * height * 3`
   * bytes, row-major, three interleaved channels, at any size.  The cache is
   * consumed: this is a prefill, so a caller that wants a second turn resets
   * first.
   *
   * Throws on a failure -- a prompt with no marker or two of them, an RGB buffer
   * whose length does not match the shape, a prompt past the context window. */
  Composition compose(const std::string &prompt, const uint8_t *rgb, int64_t width,
                      int64_t height);

 private:
  VlmSession(std::unique_ptr<Session> text, std::unique_ptr<Qwen3VlVision> tower);

  std::unique_ptr<Session> text_;
  std::unique_ptr<Qwen3VlVision> tower_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_RUNTIME_VLM_SESSION_H */