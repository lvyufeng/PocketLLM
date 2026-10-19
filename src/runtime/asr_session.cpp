#include "runtime/asr_session.h"

#include <cmath>
#include <cstring>

#include "runtime/status.h"

namespace pocketllm {

AsrSession::AsrSession(std::unique_ptr<Session> text, std::unique_ptr<Qwen3Audio> tower)
    : text_(std::move(text)), tower_(std::move(tower)) {}

AsrSession::~AsrSession() = default;

std::unique_ptr<AsrSession> AsrSession::open(const std::string &mmproj_path,
                                             const std::string &ggml_path,
                                             const std::string &backend) {
  std::unique_ptr<Session> text = Session::open(ggml_path, backend);
  if (text->model() == nullptr) {
    throw Error("'" + ggml_path +
                "' has no graph this build can run, so it cannot take audio; a speech "
                "session needs a text model as well as a projector");
  }

  GgufReader reader(mmproj_path);
  const std::string refusal = audio_arch_refusal(reader);
  if (!refusal.empty()) {
    throw Error("'" + mmproj_path + "': " + refusal);
  }

  /* The tower lives on the text session's device, the same way the vision tower does:
   * one process owns one device, and a second lookup of the same name would work until
   * it did not. */
  std::unique_ptr<Qwen3Audio> tower = Qwen3Audio::load(reader, text->backend());

  auto session = std::unique_ptr<AsrSession>(new AsrSession(std::move(text), std::move(tower)));
  session->params_ = session->tower_->frontend_params();
  return session;
}

int64_t AsrSession::windows_for(int64_t n_samples) const {
  return audio_window_count(n_samples, params_);
}

AsrSession::Composition AsrSession::compose(const std::string &prompt, const float *pcm,
                                            int64_t n_samples) {
  if (pcm == nullptr || n_samples <= 0) {
    throw Error("compose needs samples");
  }

  const std::string media = marker();
  const std::size_t at = prompt.find(media);
  if (at == std::string::npos) {
    throw Error("the prompt has no '" + media +
                "', and an audio call is defined by where the sound goes");
  }
  if (prompt.find(media, at + media.size()) != std::string::npos) {
    throw Error("the prompt has more than one '" + media +
                "', and this composition places one recording per call");
  }

  const Tokenizer *tokenizer = text_->tokenizer();
  if (tokenizer == nullptr) {
    throw Error("this checkpoint has no vocabulary this build can read, so it cannot tokenize "
                "the prompt around audio");
  }

  /* The text either side.  One `<|audio_start|>` before the whole recording and one
   * `<|audio_end|>` after it, whatever the window count: the markers bracket the audio,
   * they do not bracket each window. */
  const std::string before = prompt.substr(0, at) + audio_start();
  const std::string after = std::string(audio_end()) + prompt.substr(at + media.size());
  const std::vector<int32_t> tokens_before =
      tokenizer->encode(before, /*add_special=*/false, /*parse_special=*/true);
  const std::vector<int32_t> tokens_after =
      tokenizer->encode(after, /*add_special=*/false, /*parse_special=*/true);
  if (tokens_before.empty() || tokens_after.empty()) {
    throw Error("the prompt tokenized to nothing on one side of the audio");
  }

  const int64_t windows = audio_window_count(n_samples, params_);
  if (windows <= 0) {
    throw Error("the signal is too short to make one window");
  }

  /* A full prefill: the calls below are one prompt, so a previous composition's rows
   * must not be attended to. */
  text_->reset();

  Composition report;
  report.tokens_before = static_cast<int64_t>(tokens_before.size());
  report.audio_windows = windows;
  report.tokens_after = static_cast<int64_t>(tokens_after.size());

  /* 1. The text before, at its own positions. */
  text_->forward(tokens_before.data(), report.tokens_before, /*argmax_out=*/nullptr);

  /* 2. One span per window.  The positions are a plain run of `n` starters -- one per
   * token, because this text model's rotary is one-dimensional -- and the span is
   * decoded causally. */
  for (int64_t window = 0; window < windows; ++window) {
    const AudioMel mel = audio_log_mel_window(pcm, n_samples, params_, window);
    const int64_t tokens = tower_->tokens_for_frames(mel.n_len);
    if (window == 0) {
      report.tokens_per_window = tokens;
    }
    std::vector<float> features(
        static_cast<std::size_t>(tokens * tower_->projection_dim()), 0.0F);
    tower_->encode(mel, features.data());

    const int64_t span_start = text_->cache_length();
    /* No explicit positions: this text model's rotary is one-dimensional, so a span's
     * positions are `span_start` counting up and `forward_embeddings` builds that table
     * itself.  A vision span has to say otherwise -- its rotary is multi-section and its
     * positions are a grid -- and that is the difference the null encodes. */
    text_->forward_embeddings(features.data(), /*positions=*/nullptr, tokens, span_start);
  }

  /* 3. The text after, continuing from the cache's own length -- which for this graph is
   * also the position, since one token is one position.  A plain `forward`, because
   * there is no gap between the two numbers to bridge: the vision path needs
   * `forward_positions` exactly because a span's *positions* and its *cache rows* part
   * company, and this one's do not. */
  report.logits = text_->forward(tokens_after.data(), report.tokens_after,
                                 /*argmax_out=*/nullptr);
  report.positions = text_->cache_length();
  return report;
}

}  // namespace pocketllm