#include "runtime/vlm_session.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/status.h"

namespace pocketllm {
namespace {

/* Where the span's markers come from, and they are not decoration: they are
 * `mtmd`'s for this projector (`PROJECTOR_TYPE_QWEN3VL`), and they are what makes
 * the two engines' token sequences the same sequence rather than merely the same
 * length.  A prompt without them tokenizes to a different sequence and the
 * comparison stops being one. */
constexpr const char *kVisionStart = "<|vision_start|>";
constexpr const char *kVisionEnd = "<|vision_end|>";

}  // namespace

VlmSession::VlmSession(std::unique_ptr<Session> text, std::unique_ptr<Qwen3VlVision> tower)
    : text_(std::move(text)), tower_(std::move(tower)) {}

VlmSession::~VlmSession() = default;

std::unique_ptr<VlmSession> VlmSession::open(const std::string &mmproj_path,
                                             const std::string &ggml_path,
                                             const std::string &backend) {
  /* The text model first: it is the half that the caller's prompt goes through,
   * and a session that cannot run it has nothing to attach a tower to.  The two
   * must agree on a device, which is why one name is given for both rather than
   * one each -- one process owns one device, and a tower on another one is a
   * copy of the activations per image. */
  std::unique_ptr<Session> text = Session::open(ggml_path, backend);
  if (text->model() == nullptr) {
    throw Error("'" + ggml_path +
                "' has no graph this build can run, so it cannot take an image; a vision "
                "session needs a text model as well as a projector");
  }

  /* The projector is opened through its own reader rather than through a second
   * `Session`: an mmproj has no vocabulary and no KV cache, so a session would
   * be a text engine with everything unused.  It is mapped and held for this
   * object's lifetime, the same way a session holds its checkpoint. */
  GgufReader reader(mmproj_path);

  /* Named refusal before the tower's own: `vision_arch_refusal` answers the
   * "which file is this" question for a caller who handed over something that is
   * not a projector at all, and its message says so more usefully than a missing
   * tensor would. */
  const std::string refusal = vision_arch_refusal(reader);
  if (!refusal.empty()) {
    throw Error("'" + mmproj_path + "': " + refusal);
  }

  /* The tower is loaded onto the *text session's* device rather than a second one
   * looked up by the same name: one process owns one device, and two objects that
   * each resolved "cpu" happen to be the same card until the day they are not.
   * The tower's buffers are allocated from this backend and freed by it. */
  std::unique_ptr<Qwen3VlVision> tower = Qwen3VlVision::load(reader, text->backend());

  /* `reader` is local because the tower reads everything it needs at load and
   * keeps device buffers rather than a view of the mapping. */
  return std::unique_ptr<VlmSession>(new VlmSession(std::move(text), std::move(tower)));
}

VlmSession::Composition VlmSession::compose(const std::string &prompt, const uint8_t *rgb,
                                            int64_t width, int64_t height) {
  if (rgb == nullptr) {
    throw Error("compose needs pixels, and rgb is NULL");
  }
  if (width <= 0 || height <= 0) {
    throw Error("compose needs a positive image size, and " + std::to_string(width) + "x" +
                std::to_string(height) + " was given");
  }

  /* One marker, not two: the flow above is one image per call, and a prompt with
   * two of them is a request this composition cannot answer rather than one it
   * should resolve to the first. */
  const std::string media = VlmSession::marker();
  const std::size_t at = prompt.find(media);
  if (at == std::string::npos) {
    throw Error("the prompt has no '" + media +
                "', and a vision call is defined by where the image goes");
  }
  if (prompt.find(media, at + media.size()) != std::string::npos) {
    throw Error("the prompt has more than one '" + media +
                "', and this composition places one image per call");
  }

  /* Where the time goes.  A synchronise after each stage, because the card's launches are
   * asynchronous and a total wall time says nothing about which of the four to look at. */
  const bool profiling = std::getenv("POCKETLLM_VLM_PROFILE") != nullptr;
  const auto mark = [&]() {
    if (!profiling) {
      return 0.0;
    }
    text_->backend().synchronize();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count()) /
           1000.0;
  };
  double t_tower = 0.0;
  double t_before = 0.0;
  double t_span = 0.0;
  double t_after = 0.0;
  double t_prev = mark();

  const Tokenizer *tokenizer = text_->tokenizer();
  if (tokenizer == nullptr) {
    throw Error("this checkpoint has no vocabulary this build can read, so it cannot tokenize "
                "the prompt around an image");
  }

  /* The tower first: it is the expensive half and it is the half that can fail
   * on the image's shape, so a bad image is refused before the cache has been
   * advanced by a partial composition. */
  const int64_t tokens = tower_->tokens_for(width, height);
  std::vector<float> planes(
      static_cast<std::size_t>(tower_->n_planes() * tokens * tower_->projection_dim()), 0.0F);
  tower_->encode(rgb, height, width, planes.data());
  t_tower = mark() - t_prev;
  t_prev = mark();

  const std::string before = prompt.substr(0, at) + kVisionStart;
  const std::string after = std::string(kVisionEnd) + prompt.substr(at + media.size());
  const std::vector<int32_t> tokens_before =
      tokenizer->encode(before, /*add_special=*/false, /*parse_special=*/true);
  const std::vector<int32_t> tokens_after =
      tokenizer->encode(after, /*add_special=*/false, /*parse_special=*/true);
  if (tokens_before.empty() || tokens_after.empty()) {
    throw Error("the prompt tokenized to nothing on one side of the image");
  }

  /* A full prefill rather than a continuation: the three calls below are one
   * prompt, so the cache starts empty and a caller that wants a second turn
   * resets first.  Leaving a previous composition's rows in place would have the
   * span attend to them, which is a different prompt and not an error anything
   * downstream would catch. */
  text_->reset();

  /* 1. The text before the image, at its own positions. */
  text_->forward(tokens_before.data(), static_cast<int64_t>(tokens_before.size()),
                 /*argmax_out=*/nullptr);
  t_before = mark() - t_prev;
  t_prev = mark();

  /* 2. The span.  Its positions are llama.cpp's for this projector: one temporal
   * position for the whole span, and each patch's row and column offset by it --
   * `mtmd_image_tokens_get_decoder_pos`, as measured.  The column is the fast
   * axis, which is the same order the merge gives the tokens, and the fourth
   * component is zero. */
  const int64_t span_start = text_->cache_length();
  /* The merged grid is square: `resize_target` rounds both axes onto the same
   * step, so one side describes it. */
  const int64_t side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(tokens))));
  Composition report;
  report.tokens_before = static_cast<int64_t>(tokens_before.size());
  report.image_tokens = tokens;
  report.tokens_after = static_cast<int64_t>(tokens_after.size());
  report.image_span = side;
  tower_->prices_for(width, height, &report.target_width, &report.target_height, &report.image_tokens);
  std::vector<int64_t> span_positions(static_cast<std::size_t>(tokens) * 4);
  for (int64_t i = 0; i < tokens; ++i) {
    span_positions[static_cast<std::size_t>(i * 4 + 0)] = span_start;
    span_positions[static_cast<std::size_t>(i * 4 + 1)] = span_start + i / side;
    span_positions[static_cast<std::size_t>(i * 4 + 2)] = span_start + i % side;
    span_positions[static_cast<std::size_t>(i * 4 + 3)] = 0;
  }
  text_->forward_embeddings(planes.data(), span_positions.data(), tokens, span_start);
  t_span = mark() - t_prev;
  t_prev = mark();

  /* 3. The text after it.  Its *positions* continue from the span's, which are
   * `side` of them and not `tokens`, while its cache rows continue from the
   * cache's own length -- the two numbers an image span pulls apart, and the
   * reason this call takes explicit positions at all. */
  const int64_t after_base = span_start + side;
  std::vector<int64_t> after_positions(static_cast<std::size_t>(tokens_after.size()) * 4);
  for (std::size_t i = 0; i < tokens_after.size(); ++i) {
    /* A text token is `(p, p, p, 0)`.  The fourth component is not decorative:
     * under `GGML_ROPE_TYPE_IMROPE` the pairs past the last declared section fall
     * through to it, and this checkpoint's sections leave two of its 64 rotating
     * pairs there. */
    for (int section = 0; section < 3; ++section) {
      after_positions[i * 4 + static_cast<std::size_t>(section)] =
          after_base + static_cast<int64_t>(i);
    }
    after_positions[i * 4 + 3] = 0;
  }
  report.logits = text_->forward_positions(tokens_after.data(), after_positions.data(),
                                           static_cast<int64_t>(tokens_after.size()),
                                           text_->cache_length());
  /* The span advances the position counter by `side` for `side * side` tokens, so
   * the position the text after it ends at is `after_base + its own length`, and
   * the cache holds one row per token.  Computing the first from the second is the
   * mistake this field exists to make impossible. */
  t_after = mark() - t_prev;
  if (profiling) {
    std::fprintf(stderr,
                 "[vlm] tower %.1f ms, text-before %.1f ms, span %.1f ms, text-after %.1f ms\n",
                 t_tower, t_before, t_span, t_after);
  }
  report.positions = after_base + static_cast<int64_t>(tokens_after.size());
  report.cache_rows = text_->cache_length();
  return report;
}

}  // namespace pocketllm