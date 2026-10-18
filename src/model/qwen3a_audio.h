/* The audio tower of a `qwen3a` checkpoint.
 *
 * A Qwen3-ASR checkpoint is two GGUFs the way a Qwen3-VL one is: a text model this
 * engine already runs, and an `mmproj` holding the encoder.  The encoder turns a mel
 * spectrogram into a block of feature vectors the text graph is fed at its
 * placeholder positions.
 *
 * The graph is a plain ViT with a convolutional front end, and it is simpler than the
 * vision tower next door in every respect that usually costs work: no rotary at all
 * (the only positional information is a learned table, added once), no merger and no
 * deepstack (one output plane, and the text graph's `n_deepstack_layers` is zero for
 * this checkpoint), and no causal masking (every frame sees every frame).
 *
 * ## The front end of the graph, which is not a transformer
 *
 *     mel           (n_frames, n_mel)        n_frames a multiple of 100
 *     view as       (100, n_mel, 1, chunks)  the conv sub-chunk, chunk slowest
 *     3x            conv2d 3x3, stride 2, pad 1, then gelu_erf
 *                   (100, 128) -> (50, 64) -> (25, 32) -> (13, 16), 1 -> 480 -> 480 -> 480
 *     permute+reshape  (13*16*480 per chunk) as (7680, 13 * chunks)
 *     conv_out      7680 -> d_model
 *
 * **The 100-frame chunk is part of the shape, not a batching detail.**  The three
 * convolutions are applied to each chunk separately, so their padding at the chunk's
 * edges is part of the arithmetic and a caller cannot feed 200 frames in one piece and
 * get the same answer.  That is why this takes the front end's `AudioMel` rather than
 * a bare matrix: its `n_len` is a multiple of 100 by construction.
 *
 * ## The position index resets every chunk
 *
 * Each chunk's 13 output tokens take positions `0..12` of the learned table, not
 * `0..12`, `13..25`, ... across chunks.  The reference comments on it and it is worth
 * repeating: "sequential positions" is the obvious guess and it is wrong, and it is
 * also the kind of thing a one-chunk test would never see.
 *
 * ## The output token order
 *
 * Token `t` of the whole block is `chunk * conv_out_w + ow`, and within it the feature
 * index is `oh * conv_channels + oc`.  That is what the reference's
 * `permute(2, 0, 1, 3)` followed by a reshape to `(flat, tokens)` produces, and it is
 * *not* the order this engine's `conv2d` writes: that kernel is the vision tower's
 * patch embedding and lays its output patch-major with the row slowest, which is a
 * documented difference (ggml's own output has `out_w` fastest).  The rearrange is
 * explicit, in one place, rather than implied by index arithmetic at each use.
 *
 * ## The weights
 *
 * The attention and FFN projections are `q8_0` blocks while the convolutions are f16,
 * which is why the shared tower loader grew a `q8_0` branch (`model/tower_loader.h`)
 * rather than this file carrying a second loader.
 */

#ifndef POCKETLLM_MODEL_QWEN3A_AUDIO_H
#define POCKETLLM_MODEL_QWEN3A_AUDIO_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"
#include "model/audio_frontend.h"

namespace pocketllm {

/* Whether this checkpoint is an audio tower this build can run.
 *
 * The same shape as `vision_arch_refusal`: an empty string means yes, and anything
 * else is a sentence for the caller naming what was found instead.  A message that
 * says which projector it *is* makes "this build reads another one" actionable. */
std::string audio_arch_refusal(const GgufReader &checkpoint);

class Qwen3Audio {
 public:
  /* Read the tower's hparams and bind every tensor.  Throws `Error` naming the key or
   * the tensor that was missing or the wrong shape -- a tower missing one of its 398
   * tensors produces a plausible embedding rather than a failure, and the failure
   * otherwise arrives as wrong text several hundred milliseconds later.
   *
   * The reader must outlive the load; the backend must outlive the model. */
  static std::unique_ptr<Qwen3Audio> load(const GgufReader &checkpoint, kernel::Backend &backend);

  ~Qwen3Audio();
  Qwen3Audio(const Qwen3Audio &) = delete;
  Qwen3Audio &operator=(const Qwen3Audio &) = delete;

  /* The front end's parameters, from this checkpoint: the mel bin count is the file's
   * and the rest is fixed for this projector family.  A caller driving the front end
   * asks for these rather than repeating the constants. */
  AudioFrontendParams frontend_params() const;

  /* How many tokens a mel block of `frames` frames becomes, or a thrown `Error` if
   * the frame count is not a multiple of the conv sub-chunk. */
  int64_t tokens_for_frames(int64_t frames) const;

  /* Floats per token, from `projection_dim`. */
  int64_t projection_dim() const { return projection_dim_; }

  /* Run the tower over one mel block and write its features to `out`, which holds
   * `tokens_for_frames(mel.n_len) * projection_dim()` floats, token-major.
   *
   * `mel` must be the front end's output: mel-major, `n_len` a multiple of the conv
   * sub-chunk.  Throws by name otherwise. */
  void encode(const AudioMel &mel, float *out);

 private:
  explicit Qwen3Audio(kernel::Backend &backend);

  void allocate_buffers();
  void release_buffers();

  kernel::Backend *backend_ = nullptr;
  std::vector<kernel::DeviceBuffer> owned_;

  int64_t n_mel_ = 0;
  int64_t n_layer_ = 0;
  int64_t n_embd_ = 0;
  int64_t n_head_ = 0;
  int64_t n_ff_ = 0;
  int64_t head_dim_ = 0;
  int64_t projection_dim_ = 0;
  int64_t conv_ch_ = 0;
  int64_t conv_chunk_ = 100;
  /* What the three convolutions produce from one chunk: 100x128 -> 13x16 in the
   * checkpoint's own configuration, both derived from the tensors rather than typed. */
  int64_t conv_out_w_ = 0;
  int64_t conv_out_h_ = 0;
  int64_t conv_flat_ = 0;
  /* The learned position table's row count.  Only `conv_out_w_` rows are ever read, but
   * the gather is an `embedding` over the whole table, so its extent is the vocabulary. */
  int64_t position_rows_ = 0;
  float eps_ = 1e-5F;

  struct Block {
    kernel::DeviceBuffer ln1_w;
    kernel::DeviceBuffer ln1_b;
    kernel::DeviceBuffer ln2_w;
    kernel::DeviceBuffer ln2_b;
    kernel::DeviceBuffer q_w;
    kernel::DeviceBuffer q_b;
    kernel::DeviceBuffer k_w;
    kernel::DeviceBuffer k_b;
    kernel::DeviceBuffer v_w;
    kernel::DeviceBuffer v_b;
    kernel::DeviceBuffer o_w;
    kernel::DeviceBuffer o_b;
    kernel::DeviceBuffer up_w;
    kernel::DeviceBuffer up_b;
    kernel::DeviceBuffer down_w;
    kernel::DeviceBuffer down_b;
  };
  std::vector<Block> blocks_;

  struct Conv {
    kernel::DeviceBuffer weight;
    kernel::DeviceBuffer bias;
  };
  Conv conv1_;
  Conv conv2_;
  Conv conv3_;
  /* The projection out of the conv stack.  Named for the tensor, because
   * `conv_out_w_` above is already the *width* the conv stack produces and the two
   * meanings would be one letter apart. */
  kernel::DeviceBuffer conv_out_weight_;
  kernel::DeviceBuffer conv_out_bias_;
  kernel::DeviceBuffer position_table_;
  kernel::DeviceBuffer post_ln_w_;
  kernel::DeviceBuffer post_ln_b_;
  kernel::DeviceBuffer mlp1_w_;
  kernel::DeviceBuffer mlp1_b_;
  kernel::DeviceBuffer mlp2_w_;
  kernel::DeviceBuffer mlp2_b_;

  /* Sized once, for the largest block the front end hands over: its inference window
   * is 800 frames, which is eight conv sub-chunks. */
  int64_t max_frames_ = 0;
  int64_t max_chunks_ = 0;
  int64_t max_tokens_ = 0;

  /* The conv staging: one chunk of mel at a time in `mel_`, and the three
   * convolutions ping-pong between `conv_a_` and `conv_b_`. */
  kernel::DeviceBuffer mel_;
  kernel::DeviceBuffer conv_a_;
  kernel::DeviceBuffer conv_b_;
  kernel::DeviceBuffer patches_;
  /* The position rows the gather reads, one index per token. */
  kernel::DeviceBuffer positions_;
  kernel::DeviceBuffer x_;
  kernel::DeviceBuffer x_norm_;
  kernel::DeviceBuffer q_;
  kernel::DeviceBuffer k_;
  kernel::DeviceBuffer v_;
  kernel::DeviceBuffer attn_;
  kernel::DeviceBuffer up_;
  kernel::DeviceBuffer ffn_;
  kernel::DeviceBuffer scores_;
  kernel::DeviceBuffer out_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_QWEN3A_AUDIO_H */