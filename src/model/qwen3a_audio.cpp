#include "model/qwen3a_audio.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "model/tower_loader.h"
#include "runtime/status.h"

namespace pocketllm {
namespace {

bool metadata_flag(const GgufReader &checkpoint, const std::string &name) {
  const GgufValue *value = checkpoint.find(name);
  if (value == nullptr) {
    return false;
  }
  if (const bool *flag = std::get_if<bool>(value)) {
    return *flag;
  }
  if (const int64_t *integer = std::get_if<int64_t>(value)) {
    return *integer != 0;
  }
  return false;
}

std::string audio_key(const std::string &suffix) { return "clip.audio." + suffix; }

/* The extent one `3x3` stride-2 padding-1 convolution leaves, which is the same
 * arithmetic `ggml_conv_2d(..., 2, 2, 1, 1, 1, 1)` runs. */
int64_t conv_side(int64_t input) { return (input + 2 * 1 - 3) / 2 + 1; }

}  // namespace

std::string audio_arch_refusal(const GgufReader &checkpoint) {
  const std::string arch = checkpoint.get_string("general.architecture", "");
  if (arch != "clip") {
    return "this file's architecture is '" + arch + "', and a projector declares 'clip'";
  }
  if (!metadata_flag(checkpoint, "clip.has_audio_encoder")) {
    return "this file has no audio encoder; a projector that takes audio declares one with "
           "'clip.has_audio_encoder'";
  }
  const std::string projector = checkpoint.get_string(audio_key("projector_type"), "");
  if (projector != "qwen3a") {
    return "this file's audio projector is '" + projector +
           "', and this build reads 'qwen3a' (Qwen3-ASR)";
  }
  return "";
}

Qwen3Audio::Qwen3Audio(kernel::Backend &backend) : backend_(&backend) {}

Qwen3Audio::~Qwen3Audio() {
  release_buffers();
  for (kernel::DeviceBuffer buffer : owned_) {
    backend_->release(buffer);
  }
}

void Qwen3Audio::release_buffers() {
  for (kernel::DeviceBuffer *buffer :
       {&mel_, &conv_a_, &conv_b_, &patches_, &positions_, &x_, &x_norm_, &q_, &k_, &v_, &attn_,
        &up_, &ffn_, &scores_, &out_}) {
    backend_->release(*buffer);
    *buffer = kernel::DeviceBuffer{};
  }
}

AudioFrontendParams Qwen3Audio::frontend_params() const {
  AudioFrontendParams params;
  params.n_mel = n_mel_;
  params.conv_chunk = conv_chunk_;
  /* The window length, the hop and the rest are fixed for this projector family and
   * they are the front end's own defaults: there is no key in the file for them, and a
   * caller that guessed different ones would build a different prompt. */
  return params;
}

int64_t Qwen3Audio::tokens_for_frames(int64_t frames) const {
  if (frames <= 0 || frames % conv_chunk_ != 0) {
    throw Error("an audio block is " + std::to_string(frames) +
                " mel frames, and this tower takes a multiple of " + std::to_string(conv_chunk_) +
                ", the conv sub-chunk the graph views it as");
  }
  return (frames / conv_chunk_) * conv_out_w_;
}

std::unique_ptr<Qwen3Audio> Qwen3Audio::load(const GgufReader &checkpoint,
                                             kernel::Backend &backend) {
  auto model = std::unique_ptr<Qwen3Audio>(new Qwen3Audio(backend));

  model->n_layer_ = tower::require_int(checkpoint, audio_key("block_count"));
  model->n_embd_ = tower::require_int(checkpoint, audio_key("embedding_length"));
  model->n_head_ = tower::require_int(checkpoint, audio_key("attention.head_count"));
  model->n_ff_ = tower::require_int(checkpoint, audio_key("feed_forward_length"));
  model->n_mel_ = tower::require_int(checkpoint, audio_key("num_mel_bins"));
  model->projection_dim_ = tower::require_int(checkpoint, audio_key("projection_dim"));
  model->eps_ =
      static_cast<float>(checkpoint.get_float(audio_key("attention.layer_norm_epsilon"), 1e-5));

  if (model->n_embd_ % model->n_head_ != 0) {
    throw Error("the audio tower's embedding length " + std::to_string(model->n_embd_) +
                " is not a multiple of its head count " + std::to_string(model->n_head_));
  }
  model->head_dim_ = model->n_embd_ / model->n_head_;

  /* The channel count comes from the tensor's own declared shape rather than from a
   * second copy of the configuration: a file that disagreed with the arithmetic below
   * would otherwise run and answer wrongly. */
  const GgufTensorInfo *first = checkpoint.tensor("a.conv2d.1.weight");
  if (first == nullptr || first->dimensions.size() != 4) {
    throw Error("the audio tower has no 'a.conv2d.1.weight', or it is not a 4-D convolution");
  }
  model->conv_ch_ = static_cast<int64_t>(first->dimensions[3]);

  /* (100, 128) through three convolutions that halve and round down. */
  model->conv_out_w_ = conv_side(conv_side(conv_side(model->conv_chunk_)));
  model->conv_out_h_ = conv_side(conv_side(conv_side(model->n_mel_)));
  model->conv_flat_ = model->conv_out_h_ * model->conv_ch_;

  model->conv1_.weight =
      tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.1.weight");
  model->conv1_.bias = tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.1.bias");
  model->conv2_.weight =
      tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.2.weight");
  model->conv2_.bias = tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.2.bias");
  model->conv3_.weight =
      tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.3.weight");
  model->conv3_.bias = tower::bind_tensor(checkpoint, backend, model->owned_, "a.conv2d.3.bias");

  const tower::Linear conv_out = tower::bind_linear(checkpoint, backend, model->owned_,
                                                    "a.conv_out", model->n_embd_, model->conv_flat_);
  model->conv_out_weight_ = conv_out.weight;
  model->conv_out_bias_ = conv_out.bias;
  model->position_table_ =
      tower::bind_tensor(checkpoint, backend, model->owned_, "a.position_embd.weight");
  {
    /* The table is `(n_embd, rows)` and only `conv_out_w_` rows of it are ever read --
     * the reference's per-chunk index.  Checked rather than assumed: a file with fewer
     * rows than a chunk needs is a file this graph would read past. */
    const GgufTensorInfo *info = checkpoint.tensor("a.position_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2 ||
        static_cast<int64_t>(info->dimensions[0]) != model->n_embd_ ||
        static_cast<int64_t>(info->dimensions[1]) < model->conv_out_w_) {
      throw Error("the audio tower's position table is not (" + std::to_string(model->n_embd_) +
                  ", at least " + std::to_string(model->conv_out_w_) + ")");
    }
    model->position_rows_ = static_cast<int64_t>(info->dimensions[1]);
  }
  model->post_ln_w_ = tower::bind_tensor(checkpoint, backend, model->owned_, "a.post_ln.weight");
  model->post_ln_b_ = tower::bind_tensor(checkpoint, backend, model->owned_, "a.post_ln.bias");
  const tower::Linear mlp1 = tower::bind_linear(checkpoint, backend, model->owned_, "mm.a.mlp.1",
                                                model->n_embd_, model->n_embd_);
  model->mlp1_w_ = mlp1.weight;
  model->mlp1_b_ = mlp1.bias;
  const tower::Linear mlp2 = tower::bind_linear(checkpoint, backend, model->owned_, "mm.a.mlp.2",
                                                model->projection_dim_, model->n_embd_);
  model->mlp2_w_ = mlp2.weight;
  model->mlp2_b_ = mlp2.bias;

  model->blocks_.resize(static_cast<std::size_t>(model->n_layer_));
  for (int64_t layer = 0; layer < model->n_layer_; ++layer) {
    const std::string base = "a.blk." + std::to_string(layer);
    Block &block = model->blocks_[static_cast<std::size_t>(layer)];
    block.ln1_w = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln1.weight");
    block.ln1_b = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln1.bias");
    block.ln2_w = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln2.weight");
    block.ln2_b = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln2.bias");
    const tower::Linear q = tower::bind_linear(checkpoint, backend, model->owned_,
                                               base + ".attn_q", model->n_embd_, model->n_embd_);
    block.q_w = q.weight;
    block.q_b = q.bias;
    const tower::Linear k = tower::bind_linear(checkpoint, backend, model->owned_,
                                               base + ".attn_k", model->n_embd_, model->n_embd_);
    block.k_w = k.weight;
    block.k_b = k.bias;
    const tower::Linear v = tower::bind_linear(checkpoint, backend, model->owned_,
                                               base + ".attn_v", model->n_embd_, model->n_embd_);
    block.v_w = v.weight;
    block.v_b = v.bias;
    const tower::Linear o = tower::bind_linear(checkpoint, backend, model->owned_,
                                               base + ".attn_out", model->n_embd_, model->n_embd_);
    block.o_w = o.weight;
    block.o_b = o.bias;
    const tower::Linear up = tower::bind_linear(checkpoint, backend, model->owned_,
                                                base + ".ffn_up", model->n_ff_, model->n_embd_);
    block.up_w = up.weight;
    block.up_b = up.bias;
    const tower::Linear down = tower::bind_linear(checkpoint, backend, model->owned_,
                                                  base + ".ffn_down", model->n_embd_, model->n_ff_);
    block.down_w = down.weight;
    block.down_b = down.bias;
  }

  /* The front end's inference window is 800 frames, which is eight conv sub-chunks: the
   * largest block `encode` can be handed. */
  model->max_frames_ = 800;
  model->max_chunks_ = model->max_frames_ / model->conv_chunk_;
  model->max_tokens_ = model->max_chunks_ * model->conv_out_w_;
  model->allocate_buffers();
  return model;
}

void Qwen3Audio::allocate_buffers() {
  /* One chunk's mel at a time: the convolutions take a batch axis and could run all
   * eight chunks in one call, but the rearrange has to gather across chunks anyway and
   * doing it per chunk keeps a single staging buffer. */
  mel_ = backend_->allocate(conv_chunk_ * n_mel_ * 4);
  /* The staging buffers hold a *stage's* output, and the first stage's is by far the
   * largest: (480, 64, 50) is fifteen times the (7680, 13) the stack ends at.  Sizing
   * them by the final shape is the mistake that reads right and writes past the end. */
  int64_t widest = 0;
  {
    int64_t h = n_mel_;
    int64_t w = conv_chunk_;
    for (int stage = 0; stage < 3; ++stage) {
      h = conv_side(h);
      w = conv_side(w);
      widest = std::max<int64_t>(widest, conv_ch_ * h * w);
    }
  }
  conv_a_ = backend_->allocate(widest * 4);
  conv_b_ = backend_->allocate(widest * 4);
  patches_ = backend_->allocate(max_tokens_ * conv_flat_ * 4);
  positions_ = backend_->allocate(max_tokens_ * 4);
  x_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  x_norm_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  q_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  k_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  v_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  attn_ = backend_->allocate(max_tokens_ * n_embd_ * 4);
  up_ = backend_->allocate(max_tokens_ * n_ff_ * 4);
  ffn_ = backend_->allocate(max_tokens_ * n_ff_ * 4);
  out_ = backend_->allocate(max_tokens_ * projection_dim_ * 4);
  /* Non-causal over the whole block: every frame sees every frame, which is the same
   * shape the vision tower's attention asks for.  The scratch bound is
   * `q_offset + q_len - first_key` with `q_offset = q_len - 1` -- **two** spans, not
   * one.  Sizing it at `max_tokens_` is half of what the kernel reads, and it cost a
   * silent heap overrun to rediscover what the vision tower's comment already says. */
  const int64_t scratch =
      backend_->attention_scratch(max_tokens_, n_head_, 2 * max_tokens_ - 1);
  scores_ = backend_->allocate(scratch);
}

void Qwen3Audio::encode(const AudioMel &mel, float *out) {
  if (mel.n_mel != n_mel_) {
    throw Error("the mel has " + std::to_string(mel.n_mel) + " bins and this tower takes " +
                std::to_string(n_mel_));
  }
  if (mel.n_len % conv_chunk_ != 0) {
    throw Error("the mel is " + std::to_string(mel.n_len) +
                " frames, and this tower views it in chunks of " + std::to_string(conv_chunk_));
  }
  if (mel.n_len > max_frames_) {
    throw Error("the mel is " + std::to_string(mel.n_len) + " frames, past the " +
                std::to_string(max_frames_) + " this tower's buffers are sized for");
  }
  if (static_cast<int64_t>(mel.data.size()) != mel.n_mel * mel.n_len) {
    throw Error("the mel claims " + std::to_string(mel.n_mel) + "x" + std::to_string(mel.n_len) +
                " and carries " + std::to_string(mel.data.size()) + " values");
  }
  const int64_t chunks = mel.n_len / conv_chunk_;
  const int64_t tokens = chunks * conv_out_w_;

  std::vector<float> stage_in(static_cast<std::size_t>(n_mel_ * conv_chunk_));
  std::vector<float> conv_host(static_cast<std::size_t>(conv_flat_ * conv_out_w_));
  /* The matrix the projection reads: `(conv_flat, tokens)`, the feature index slowest
   * because that is the contraction axis of the GEMM. */
  std::vector<float> patches(static_cast<std::size_t>(conv_flat_ * tokens), 0.0F);

  for (int64_t c = 0; c < chunks; ++c) {
    /* This chunk's mel as the image `conv2d` reads: a single channel, the height the mel
     * bin, the width the frame -- so `(1, n_mel, conv_chunk)` with the frame fastest. */
    for (int64_t m = 0; m < n_mel_; ++m) {
      for (int64_t t = 0; t < conv_chunk_; ++t) {
        stage_in[static_cast<std::size_t>(m * conv_chunk_ + t)] =
            mel.data[static_cast<std::size_t>(m * mel.n_len + c * conv_chunk_ + t)];
      }
    }
    backend_->copy_to_device(mel_, stage_in.data(), conv_chunk_ * n_mel_ * 4);

    int64_t h = n_mel_;
    int64_t w = conv_chunk_;
    int64_t in_ch = 1;
    for (int stage = 0; stage < 3; ++stage) {
      const Conv &conv = stage == 0 ? conv1_ : (stage == 1 ? conv2_ : conv3_);
      kernel::DeviceBuffer source = stage == 0 ? mel_ : (stage == 1 ? conv_a_ : conv_b_);
      kernel::DeviceBuffer dest = stage == 1 ? conv_b_ : conv_a_;
      const int64_t out_h = conv_side(h);
      const int64_t out_w = conv_side(w);
      backend_->conv2d(source, conv.weight, conv.bias, dest, /*batch=*/1, in_ch, h, w, conv_ch_,
                       3, 2, 1);
      backend_->gelu_erf(dest, dest, conv_ch_ * out_h * out_w);
      h = out_h;
      w = out_w;
      in_ch = conv_ch_;
    }
    /* Stage 0 wrote `conv_a_`, stage 1 `conv_b_`, stage 2 `conv_a_` again. */
    backend_->copy_to_host(conv_host.data(), conv_a_, conv_flat_ * conv_out_w_ * 4);

    /* `conv2d` writes patch-major with the row slowest: `(oy * out_w + ox) * out_ch + oc`.
     * The graph wants, for token `c * conv_out_w + ox`, the feature `oy * out_ch + oc` --
     * which is what the reference's permute-then-reshape produces.  Written out here
     * rather than folded into index arithmetic at each use, so the difference between
     * the two orders is one place a reader can check. */
    for (int64_t oy = 0; oy < conv_out_h_; ++oy) {
      for (int64_t ox = 0; ox < conv_out_w_; ++ox) {
        for (int64_t oc = 0; oc < conv_ch_; ++oc) {
          const int64_t from = (oy * conv_out_w_ + ox) * conv_ch_ + oc;
          const int64_t token = c * conv_out_w_ + ox;
          const int64_t feature = oy * conv_ch_ + oc;
          patches[static_cast<std::size_t>(feature * tokens + token)] =
              conv_host[static_cast<std::size_t>(from)];
        }
      }
    }
  }
  backend_->copy_to_device(patches_, patches.data(), tokens * conv_flat_ * 4);

  /* 7680 -> d_model. */
  /* (tokens, conv_flat) times (n_embd, conv_flat)^T gives (tokens, n_embd): the
   * contraction is the conv stack's flattened grid, not d_model. */
  backend_->gemm(patches_, conv_out_weight_, conv_out_bias_, x_, tokens, n_embd_, conv_flat_,
                 /*accumulate=*/false);

  /* The learned positions, and the index **resets every chunk**: token `t` takes row
   * `t % conv_out_w_`, not row `t`.  A gather over the table's own rows is the same
   * numbers as the reference's repeat, and it is the operator the graph already has. */
  {
    std::vector<int32_t> host_positions(static_cast<std::size_t>(tokens));
    for (int64_t t = 0; t < tokens; ++t) {
      host_positions[static_cast<std::size_t>(t)] = static_cast<int32_t>(t % conv_out_w_);
    }
    backend_->copy_to_device(positions_, host_positions.data(), tokens * 4);
    /* The table's rows are `(n_embd, table_rows)`, so the gather's vocabulary is the
     * table's row count and its width is `n_embd`. */
    backend_->embedding(positions_, tokens, position_table_, position_rows_, n_embd_, x_norm_);
    /* `x = x + position`; the residual is `add` rather than an accumulating GEMM because
     * the position is a gather and not a product. */
    backend_->add(x_, x_norm_, x_, tokens * n_embd_);
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  for (int64_t layer = 0; layer < n_layer_; ++layer) {
    const Block &block = blocks_[static_cast<std::size_t>(layer)];
    backend_->layer_norm(x_, block.ln1_w, block.ln1_b, x_norm_, tokens, n_embd_, eps_);
    backend_->gemm(x_norm_, block.q_w, block.q_b, q_, tokens, n_embd_, n_embd_, false);
    backend_->gemm(x_norm_, block.k_w, block.k_b, k_, tokens, n_embd_, n_embd_, false);
    backend_->gemm(x_norm_, block.v_w, block.v_b, v_, tokens, n_embd_, n_embd_, false);
    backend_->attention(q_, tokens, n_head_, k_, v_, n_head_, head_dim_, /*first_key=*/0,
                        /*q_offset=*/tokens - 1, /*causal=*/false, scale, attn_, scores_,
                        kernel::KVDtype::kF32);
    backend_->gemm(attn_, block.o_w, block.o_b, x_, tokens, n_embd_, n_embd_, /*accumulate=*/true);

    backend_->layer_norm(x_, block.ln2_w, block.ln2_b, x_norm_, tokens, n_embd_, eps_);
    /* `n` is the *output* width and `k` the contraction, so the up projection is
     * `(tokens, n_embd) -> (tokens, n_ff)` and not the other way round.  Written the
     * wrong way the shapes still "work" as numbers and the read runs past the
     * activation, which is a heap overrun rather than a wrong answer. */
    backend_->gemm(x_norm_, block.up_w, block.up_b, ffn_, tokens, n_ff_, n_embd_, false);
    backend_->gelu_erf(ffn_, ffn_, tokens * n_ff_);
    backend_->gemm(ffn_, block.down_w, block.down_b, x_, tokens, n_embd_, n_ff_,
                   /*accumulate=*/true);
  }

  /* The head: a layer norm, then the two-layer MLP that is the projector. */
  backend_->layer_norm(x_, post_ln_w_, post_ln_b_, x_norm_, tokens, n_embd_, eps_);
  backend_->gemm(x_norm_, mlp1_w_, mlp1_b_, up_, tokens, n_embd_, n_embd_, false);
  backend_->gelu_erf(up_, up_, tokens * n_embd_);
  backend_->gemm(up_, mlp2_w_, mlp2_b_, out_, tokens, projection_dim_, n_embd_, false);
  backend_->copy_to_host(out, out_, tokens * projection_dim_ * 4);
}

}  // namespace pocketllm