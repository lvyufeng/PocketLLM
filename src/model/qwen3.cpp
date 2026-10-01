#include "model/qwen3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "kernel/cpu/kernels.h"
#include "abi/spec.h"
#include "quant/half.h"
#include "runtime/status.h"

namespace pocketllm {

namespace {

/* The GGML storage ids this build reads. They are the values in
 * `reader.py`'s GGML_TYPES table, which the reader test already pins against
 * the Python reader; they are repeated here rather than exposed from `spec.h`
 * because `spec.h` answers "how many bytes" and this answers "which kernel",
 * and the day the two disagree is the day a quantized kernel is added. */
constexpr int kTypeF32 = 0;
constexpr int kTypeF16 = 1;

/* The initial cache and rotary-table size, in positions.
 *
 * A prompt is usually tens to a few thousand tokens and a decode grows one at a
 * time, so starting at 256 and doubling keeps a 12-token smoke test from
 * allocating the 335 MB a full 40960-position cache would cost. Both are capped
 * at the model's own context length, so the growth cannot run away. */
constexpr int64_t kInitialPositions = 256;

[[noreturn]] void missing(const std::string &what, const std::string &name) {
  throw Error("the checkpoint is missing " + what + " '" + name + "'");
}

}  // namespace

Qwen3Model::~Qwen3Model() = default;

const float *Qwen3Model::bind_dense(const GgufReader &checkpoint, const std::string &name) {
  uint64_t nbytes = 0;
  const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);
  const GgufTensorInfo *info = checkpoint.tensor(name);
  int64_t elements = 1;
  for (uint64_t dim : info->dimensions) {
    elements *= static_cast<int64_t>(dim);
  }

  auto blob = std::make_unique<float[]>(static_cast<std::size_t>(elements));
  if (info->type_id == kTypeF32) {
    std::memcpy(blob.get(), bytes, static_cast<std::size_t>(nbytes));
  } else if (info->type_id == kTypeF16) {
    for (int64_t i = 0; i < elements; ++i) {
      blob[i] = half_to_float(load_u16(bytes + 2 * i));
    }
  } else {
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", which the dense path does not read; this build handles f32 and f16");
  }

  const float *data = blob.get();
  blobs_.push_back(std::move(blob));
  return data;
}

Weight Qwen3Model::bind_matrix(const GgufReader &checkpoint, const std::string &name) {
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    missing("tensor", name);
  }
  if (info->dimensions.size() != 2) {
    throw Error("tensor '" + name + "' is not a matrix (it has " +
                std::to_string(info->dimensions.size()) + " dimensions)");
  }

  /* GGUF writes the fastest-varying axis first, so `dimensions[0]` is the
   * contraction axis and `dimensions[1]` is the output axis -- the transpose of
   * how the multiplication is written. The bytes are already row-major in that
   * order, so the kernel reads them without a transpose and this is the only
   * place the two conventions meet. */
  Weight weight;
  weight.cols = static_cast<int64_t>(info->dimensions[0]);
  weight.rows = static_cast<int64_t>(info->dimensions[1]);
  weight.type_id = info->type_id;
  weight.nbytes = static_cast<int64_t>(info->nbytes);

  /* Expanded to f32 at load. This is step 4's shape and not the final one: it
   * costs twice the checkpoint's size in host memory to run weights that are
   * stored in half that. A quantized kernel reads `blocks` in place instead,
   * and that is what replaces this -- which is why `Weight` carries the type id
   * and the raw size even though nothing here needs them. */
  weight.data = bind_dense(checkpoint, name);
  return weight;
}

std::unique_ptr<Qwen3Model> Qwen3Model::load(const GgufReader &checkpoint) {
  const std::string arch = checkpoint.get_string("general.architecture", "");
  if (arch != "qwen3") {
    throw Error("this build runs 'qwen3' checkpoints; this one declares architecture '" + arch + "'");
  }

  std::unique_ptr<Qwen3Model> model(new Qwen3Model());

  /* The keys are namespaced by architecture, and the prefix comes from the file
   * rather than from a literal so that a checkpoint declaring `qwen3` cannot be
   * read with `qwen2`'s hyperparameters. */
  const std::string p = arch + ".";
  model->n_embd_ = checkpoint.get_int(p + "embedding_length", 0);
  model->n_layer_ = checkpoint.get_int(p + "block_count", 0);
  model->n_head_ = checkpoint.get_int(p + "attention.head_count", 0);
  model->n_head_kv_ = checkpoint.get_int(p + "attention.head_count_kv", model->n_head_);
  model->head_dim_ = checkpoint.get_int(p + "attention.key_length", 0);
  model->n_ff_ = checkpoint.get_int(p + "feed_forward_length", 0);
  model->n_vocab_ = checkpoint.get_int(p + "vocab_size", 0);
  model->capacity_ = checkpoint.get_int(p + "context_length", 0);
  model->rms_eps_ = static_cast<float>(checkpoint.get_float(p + "attention.layer_norm_rms_epsilon", 1e-6));
  model->rope_theta_ = static_cast<float>(checkpoint.get_float(p + "rope.freq_base", 10000.0));

  if (model->n_vocab_ <= 0) {
    /* Not every conversion writes `vocab_size`; the embedding's own shape is
     * the authority and is what the gather will be bounded by. */
    const GgufTensorInfo *embd = checkpoint.tensor("token_embd.weight");
    if (embd == nullptr || embd->dimensions.size() != 2) {
      missing("tensor", "token_embd.weight");
    }
    model->n_vocab_ = static_cast<int64_t>(embd->dimensions[1]);
  }
  if (model->n_embd_ <= 0 || model->n_layer_ <= 0 || model->n_head_ <= 0 || model->head_dim_ <= 0) {
    throw Error("the checkpoint does not carry a complete set of qwen3 hyperparameters");
  }
  if (model->capacity_ <= 0) {
    model->capacity_ = 4096;
  }

  model->tok_embd_ = model->bind_matrix(checkpoint, "token_embd.weight");
  if (model->tok_embd_.cols != model->n_embd_) {
    throw Error("token_embd.weight has " + std::to_string(model->tok_embd_.cols) +
                " columns but embedding_length says " + std::to_string(model->n_embd_));
  }

  /* Qwen3 ties the output projection to the embedding when the checkpoint does
   * not carry a separate head, and this one does carry one. Binding whichever
   * is present -- rather than requiring `output.weight` -- is what lets the same
   * loader read a smaller conversion later. */
  if (checkpoint.tensor("output.weight") != nullptr) {
    model->output_ = model->bind_matrix(checkpoint, "output.weight");
  } else {
    model->output_ = model->tok_embd_;
  }

  model->output_norm_ = model->bind_dense(checkpoint, "output_norm.weight");

  model->layers_.reserve(static_cast<std::size_t>(model->n_layer_));
  for (int64_t il = 0; il < model->n_layer_; ++il) {
    const std::string base = "blk." + std::to_string(il) + ".";
    Layer layer;
    layer.attn_norm = model->bind_dense(checkpoint, base + "attn_norm.weight");
    layer.ffn_norm = model->bind_dense(checkpoint, base + "ffn_norm.weight");
    layer.q_norm = model->bind_dense(checkpoint, base + "attn_q_norm.weight");
    layer.k_norm = model->bind_dense(checkpoint, base + "attn_k_norm.weight");
    layer.wq = model->bind_matrix(checkpoint, base + "attn_q.weight");
    layer.wk = model->bind_matrix(checkpoint, base + "attn_k.weight");
    layer.wv = model->bind_matrix(checkpoint, base + "attn_v.weight");
    layer.wo = model->bind_matrix(checkpoint, base + "attn_output.weight");
    layer.w_gate = model->bind_matrix(checkpoint, base + "ffn_gate.weight");
    layer.w_up = model->bind_matrix(checkpoint, base + "ffn_up.weight");
    layer.w_down = model->bind_matrix(checkpoint, base + "ffn_down.weight");
    model->layers_.push_back(layer);
  }

  model->build_rope_table();
  model->ensure_capacity(1);
  return model;
}

void Qwen3Model::build_rope_table() {
  /* Neox rotary: pair `i` with `i + half`, at angle
   * `position * theta^(-2i/d)` for `i` in `[0, d/2)`. The exponent is the one
   * the reference and llama.cpp both use, and it is why the base is read from
   * the checkpoint rather than assumed: 1e6 spreads the angles far more slowly
   * than the 1e4 a smaller model would use, and the two are not interchangeable
   * at long context. */
  const int64_t half = head_dim_ / 2;
  const int64_t rows = std::min<int64_t>(kInitialPositions, capacity_);
  rope_cos_.resize(static_cast<std::size_t>(rows * half));
  rope_sin_.resize(static_cast<std::size_t>(rows * half));
  for (int64_t pos = 0; pos < rows; ++pos) {
    for (int64_t i = 0; i < half; ++i) {
      const double angle =
          static_cast<double>(pos) * std::pow(static_cast<double>(rope_theta_),
                                              -2.0 * static_cast<double>(i) / static_cast<double>(head_dim_));
      rope_cos_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::cos(angle));
      rope_sin_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
}

void Qwen3Model::ensure_capacity(int64_t n) {
  const std::size_t tokens = static_cast<std::size_t>(n);
  const std::size_t embd = static_cast<std::size_t>(n_embd_);
  const std::size_t q_width = static_cast<std::size_t>(n_head_ * head_dim_);
  const std::size_t kv_width = static_cast<std::size_t>(n_head_kv_ * head_dim_);
  const std::size_t ff = static_cast<std::size_t>(n_ff_);

  x_.resize(tokens * embd);
  x_norm_.resize(tokens * embd);
  q_.resize(tokens * q_width);
  k_.resize(tokens * kv_width);
  v_.resize(tokens * kv_width);
  attn_.resize(tokens * q_width);
  gate_.resize(tokens * ff);
  up_.resize(tokens * ff);
  ffn_.resize(tokens * ff);
  last_.resize(embd);
  logits_.resize(static_cast<std::size_t>(n_vocab_));
}

void Qwen3Model::matmul(const Weight &w, const float *x, float *out, int64_t m, bool accumulate) const {
  /* Every weight this build binds is f32 by now -- `bind_matrix` widened it --
   * so the type id is not consulted. It stays in `Weight` for the quantized
   * kernel, which is the one place the raw bytes are read. */
  cpu::gemm(x, w.data, nullptr, out, m, w.rows, w.cols, accumulate);
}

void Qwen3Model::reset() { cache_length_ = 0; }

const float *Qwen3Model::forward(const int32_t *tokens, int64_t n, int64_t start_pos) {
  if (n <= 0) {
    throw Error("forward needs at least one token");
  }
  if (start_pos != cache_length_) {
    throw Error("forward at position " + std::to_string(start_pos) + " but the cache holds " +
                std::to_string(cache_length_) + " positions");
  }
  if (start_pos + n > capacity_) {
    throw Error("the sequence would reach position " + std::to_string(start_pos + n) +
                ", past the checkpoint's context length of " + std::to_string(capacity_));
  }

  ensure_capacity(n);

  /* The KV cache and the rotary table both grow with the sequence, so both are
   * extended before anything reads them. The cache is sized to the end of this
   * batch rather than to the context: a session that decodes ten tokens should
   * not have allocated room for 40960. */
  const int64_t needed_cache = start_pos + n;
  if (needed_cache > cache_capacity_) {
    int64_t grown = std::max<int64_t>(cache_capacity_, kInitialPositions);
    while (grown < needed_cache) {
      grown = std::min<int64_t>(grown * 2, capacity_);
    }
    const std::size_t per_layer =
        static_cast<std::size_t>(grown) * static_cast<std::size_t>(n_head_kv_ * head_dim_);
    k_cache_.resize(per_layer * static_cast<std::size_t>(n_layer_));
    v_cache_.resize(per_layer * static_cast<std::size_t>(n_layer_));
    cache_capacity_ = grown;
  }
  const int64_t rope_rows = static_cast<int64_t>(rope_cos_.size()) / (head_dim_ / 2);
  if (start_pos + n > rope_rows) {
    int64_t grown = std::max<int64_t>(rope_rows, kInitialPositions);
    while (grown < start_pos + n) {
      grown = std::min<int64_t>(grown * 2, capacity_);
    }
    const int64_t half = head_dim_ / 2;
    rope_cos_.resize(static_cast<std::size_t>(grown * half));
    rope_sin_.resize(static_cast<std::size_t>(grown * half));
    for (int64_t pos = rope_rows; pos < grown; ++pos) {
      for (int64_t i = 0; i < half; ++i) {
        const double angle = static_cast<double>(pos) *
                             std::pow(static_cast<double>(rope_theta_),
                                      -2.0 * static_cast<double>(i) / static_cast<double>(head_dim_));
        rope_cos_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::cos(angle));
        rope_sin_[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::sin(angle));
      }
    }
  }

  const int64_t kv_width = n_head_kv_ * head_dim_;

  cpu::embedding(tokens, n, tok_embd_.data, n_vocab_, n_embd_, x_.data());

  for (int64_t il = 0; il < n_layer_; ++il) {
    const Layer &layer = layers_[static_cast<std::size_t>(il)];

    cpu::rms_norm(x_.data(), layer.attn_norm, x_norm_.data(), n, n_embd_, rms_eps_);

    matmul(layer.wq, x_norm_.data(), q_.data(), n, false);
    matmul(layer.wk, x_norm_.data(), k_.data(), n, false);
    matmul(layer.wv, x_norm_.data(), v_.data(), n, false);

    /* Per head, over `head_dim`. The projection is `n_head * head_dim` wide, so
     * normalizing the row instead would reduce over sixteen times as many
     * values and produce a different -- still finite, still plausible -- model.
     * The weight is shared across heads and tokens, which is why one call can
     * do all of them. */
    cpu::rms_norm(q_.data(), layer.q_norm, q_.data(), n * n_head_, head_dim_, rms_eps_);
    cpu::rms_norm(k_.data(), layer.k_norm, k_.data(), n * n_head_kv_, head_dim_, rms_eps_);

    cpu::rope_neox(q_.data(), n, n_head_, head_dim_, start_pos, rope_cos_.data(), rope_sin_.data());
    cpu::rope_neox(k_.data(), n, n_head_kv_, head_dim_, start_pos, rope_cos_.data(), rope_sin_.data());

    /* The cache holds the *rotated* key: attention scores are not a function of
     * the unrotated one, so storing before the rotation would silently drop the
     * positional signal for every token after the first.
     *
     * The write goes to this layer's own slab of the cache. Writing to a single
     * shared slab would leave the layer's keys overwritten by the next layer's
     * by the time the next call read them. */
    const std::size_t layer_stride =
        static_cast<std::size_t>(cache_capacity_ * kv_width) * static_cast<std::size_t>(il);
    const std::size_t span = static_cast<std::size_t>(n * kv_width);
    float *layer_k = k_cache_.data() + layer_stride;
    float *layer_v = v_cache_.data() + layer_stride;
    std::memcpy(layer_k + static_cast<std::size_t>(start_pos * kv_width), k_.data(),
                span * sizeof(float));
    std::memcpy(layer_v + static_cast<std::size_t>(start_pos * kv_width), v_.data(),
                span * sizeof(float));

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    scores_.resize(static_cast<std::size_t>(start_pos + n));
    cpu::attention(q_.data(), n, n_head_, layer_k, layer_v, n_head_kv_, head_dim_,
                   /*first_key=*/0, start_pos, scale, attn_.data(), scores_.data());

    matmul(layer.wo, attn_.data(), x_.data(), n, /*accumulate=*/true);

    cpu::rms_norm(x_.data(), layer.ffn_norm, x_norm_.data(), n, n_embd_, rms_eps_);
    matmul(layer.w_gate, x_norm_.data(), gate_.data(), n, false);
    matmul(layer.w_up, x_norm_.data(), up_.data(), n, false);
    cpu::silu_mul(gate_.data(), up_.data(), ffn_.data(), n * n_ff_);
    matmul(layer.w_down, ffn_.data(), x_.data(), n, /*accumulate=*/true);
  }

  /* Only the last position's logits are produced. A caller wanting a
   * continuation asks for one token at a time; the prefill's earlier positions
   * are computed because the layers are sequential, but their logits are not
   * wanted and projecting them would be 151936 values per token of waste. */
  cpu::rms_norm(x_.data() + (n - 1) * n_embd_, output_norm_, last_.data(), 1, n_embd_, rms_eps_);
  matmul(output_, last_.data(), logits_.data(), 1, false);

  cache_length_ = start_pos + n;
  return logits_.data();
}

}  // namespace pocketllm