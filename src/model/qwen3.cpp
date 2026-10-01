#include "model/qwen3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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

/* How many positions the cache and the rotary table start with.
 *
 * A prompt is usually tens to a few thousand tokens and a decode grows one at a
 * time, so starting at 256 and doubling keeps a 12-token smoke test from
 * allocating the 335 MB a full 40960-position cache would cost. Both grow with
 * the sequence and are capped at the model's own context length. */
constexpr int64_t kInitialPositions = 256;

[[noreturn]] void missing(const std::string &what, const std::string &name) {
  throw Error("the checkpoint is missing " + what + " '" + name + "'");
}

}  // namespace

Qwen3Model::~Qwen3Model() = default;

kernel::DeviceBuffer Qwen3Model::bind_dense(const GgufReader &checkpoint, const std::string &name) {
  uint64_t nbytes = 0;
  const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);
  const GgufTensorInfo *info = checkpoint.tensor(name);
  int64_t elements = 1;
  for (uint64_t dim : info->dimensions) {
    elements *= static_cast<int64_t>(dim);
  }

  std::vector<float> widened(static_cast<std::size_t>(elements));
  if (info->type_id == kTypeF32) {
    /* A byte-for-byte copy, which is also what makes the f32 path exact: no
     * arithmetic happens to a tensor that is already in the right format. */
    std::memcpy(widened.data(), bytes, static_cast<std::size_t>(nbytes));
  } else if (info->type_id == kTypeF16) {
    for (int64_t i = 0; i < elements; ++i) {
      widened[static_cast<std::size_t>(i)] = half_to_float(load_u16(bytes + 2 * i));
    }
  } else {
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", which the dense path does not read; this build handles f32 and f16");
  }

  kernel::DeviceBuffer device = backend_->allocate(elements * 4);
  backend_->copy_to_device(device, widened.data(), elements * 4);
  return device;
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
  weight.data = bind_dense(checkpoint, name);
  return weight;
}

std::unique_ptr<Qwen3Model> Qwen3Model::load(const GgufReader &checkpoint,
                                             kernel::Backend &backend) {
  const std::string arch = checkpoint.get_string("general.architecture", "");
  if (arch != "qwen3") {
    throw Error("this build runs 'qwen3' checkpoints; this one declares architecture '" + arch + "'");
  }

  std::unique_ptr<Qwen3Model> model(new Qwen3Model(backend));

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
  model->rms_eps_ =
      static_cast<float>(checkpoint.get_float(p + "attention.layer_norm_rms_epsilon", 1e-6));
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

  model->tok_embd_ = model->bind_dense(checkpoint, "token_embd.weight");
  const GgufTensorInfo *embd_info = checkpoint.tensor("token_embd.weight");
  if (static_cast<int64_t>(embd_info->dimensions[0]) != model->n_embd_) {
    throw Error("token_embd.weight has " + std::to_string(embd_info->dimensions[0]) +
                " columns but embedding_length says " + std::to_string(model->n_embd_));
  }

  /* Qwen3 ties the output projection to the embedding when the checkpoint does
   * not carry a separate head, and this one does carry one. Binding whichever
   * is present -- rather than requiring `output.weight` -- is what lets the same
   * loader read a smaller conversion later. */
  if (checkpoint.tensor("output.weight") != nullptr) {
    model->output_ = model->bind_matrix(checkpoint, "output.weight");
  } else {
    model->output_.cols = model->n_embd_;
    model->output_.rows = model->n_vocab_;
    model->output_.data = model->tok_embd_;
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
  model->ensure_capacity(1, 1);
  return model;
}

void Qwen3Model::build_rope_table() {
  const int64_t rows = std::min<int64_t>(kInitialPositions, capacity_);
  position_capacity_ = rows;
  const int64_t half = head_dim_ / 2;
  std::vector<float> cos_table(static_cast<std::size_t>(rows * half));
  std::vector<float> sin_table(static_cast<std::size_t>(rows * half));
  for (int64_t pos = 0; pos < rows; ++pos) {
    for (int64_t i = 0; i < half; ++i) {
      /* Neox rotary: pair `i` with `i + half`, at angle
       * `position * theta^(-2i/d)`. The base is read from the checkpoint rather
       * than assumed: 1e6 spreads the angles far more slowly than the 1e4 a
       * smaller model would use, and the two are not interchangeable at long
       * context. */
      const double angle =
          static_cast<double>(pos) *
          std::pow(static_cast<double>(rope_theta_),
                   -2.0 * static_cast<double>(i) / static_cast<double>(head_dim_));
      cos_table[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::cos(angle));
      sin_table[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
  rope_cos_ = backend_->allocate(rows * half * 4);
  rope_sin_ = backend_->allocate(rows * half * 4);
  backend_->copy_to_device(rope_cos_, cos_table.data(), rows * half * 4);
  backend_->copy_to_device(rope_sin_, sin_table.data(), rows * half * 4);
}

void Qwen3Model::grow_rope_table(int64_t positions) {
  if (positions <= position_capacity_) {
    return;
  }
  int64_t grown = position_capacity_;
  while (grown < positions) {
    grown = std::min<int64_t>(grown * 2, capacity_);
  }
  const int64_t half = head_dim_ / 2;
  std::vector<float> cos_table(static_cast<std::size_t>(grown * half));
  std::vector<float> sin_table(static_cast<std::size_t>(grown * half));
  for (int64_t pos = 0; pos < grown; ++pos) {
    for (int64_t i = 0; i < half; ++i) {
      const double angle =
          static_cast<double>(pos) *
          std::pow(static_cast<double>(rope_theta_),
                   -2.0 * static_cast<double>(i) / static_cast<double>(head_dim_));
      cos_table[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::cos(angle));
      sin_table[static_cast<std::size_t>(pos * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
  backend_->release(rope_cos_);
  backend_->release(rope_sin_);
  rope_cos_ = backend_->allocate(grown * half * 4);
  rope_sin_ = backend_->allocate(grown * half * 4);
  backend_->copy_to_device(rope_cos_, cos_table.data(), grown * half * 4);
  backend_->copy_to_device(rope_sin_, sin_table.data(), grown * half * 4);
  position_capacity_ = grown;
}

void Qwen3Model::ensure_capacity(int64_t n, int64_t end_pos) {
  backend_->release(x_);
  backend_->release(x_norm_);
  backend_->release(q_);
  backend_->release(k_);
  backend_->release(v_);
  backend_->release(attn_);
  backend_->release(gate_);
  backend_->release(up_);
  backend_->release(ffn_);
  backend_->release(last_);
  backend_->release(logits_);
  backend_->release(scores_);
  backend_->release(tokens_);

  x_ = backend_->allocate(n * n_embd_ * 4);
  x_norm_ = backend_->allocate(n * n_embd_ * 4);
  q_ = backend_->allocate(n * n_head_ * head_dim_ * 4);
  k_ = backend_->allocate(n * n_head_kv_ * head_dim_ * 4);
  v_ = backend_->allocate(n * n_head_kv_ * head_dim_ * 4);
  attn_ = backend_->allocate(n * n_head_ * head_dim_ * 4);
  gate_ = backend_->allocate(n * n_ff_ * 4);
  up_ = backend_->allocate(n * n_ff_ * 4);
  ffn_ = backend_->allocate(n * n_ff_ * 4);
  last_ = backend_->allocate(n_embd_ * 4);
  logits_ = backend_->allocate(n_vocab_ * 4);
  /* The score row is per query token per head, over the whole visible cache, and
   * the kernel indexes it by *absolute* key position -- so its length is the end
   * position of this batch, not the batch size. Sizing it to the batch is the
   * easy mistake and a quiet one: a fresh prompt of n tokens would happen to fit
   * (`end_pos == n`), and a decode at position 300 would write 300 floats past a
   * two-float allocation. */
  scores_ = backend_->allocate(end_pos * 4);
  /* The token ids go to the device once per forward: the embedding is a gather
   * and a backend whose indices live on the host would have to fetch the whole
   * table back. */
  tokens_ = backend_->allocate(n * 4 + 64);
}

void Qwen3Model::matmul(const Weight &w, kernel::DeviceBuffer x, kernel::DeviceBuffer out, int64_t m,
                        bool accumulate) const {
  /* Every weight this build binds is f32 by now -- `bind_matrix` widened it --
   * so the type id is not consulted. It stays in `Weight` for the quantized
   * kernel, which is the one place the raw bytes are read. */
  kernel::DeviceBuffer no_bias;
  backend_->gemm(x, w.data, no_bias, out, m, w.rows, w.cols, accumulate);
}

void Qwen3Model::reset() { cache_length_ = 0; }

kernel::DeviceBuffer Qwen3Model::forward(const int32_t *tokens, int64_t n, int64_t start_pos) {
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

  ensure_capacity(n, start_pos + n);

  /* Both the cache and the rotary table grow with the sequence, and they grow
   * together: a token that fits the cache but not the table is a read past the
   * end of the table. Sized to the end of this batch rather than to the context,
   * so a session that decodes ten tokens has not allocated room for 40960. */
  const int64_t needed = start_pos + n;
  if (needed > cache_capacity_) {
    int64_t grown = std::max<int64_t>(cache_capacity_, kInitialPositions);
    while (grown < needed) {
      grown = std::min<int64_t>(grown * 2, capacity_);
    }
    const std::size_t per_layer =
        static_cast<std::size_t>(grown) * static_cast<std::size_t>(n_head_kv_ * head_dim_);
    if (k_cache_.handle != 0) {
      backend_->release(k_cache_);
      backend_->release(v_cache_);
    }
    k_cache_ = backend_->allocate(static_cast<int64_t>(per_layer) * n_layer_ * 4);
    v_cache_ = backend_->allocate(static_cast<int64_t>(per_layer) * n_layer_ * 4);
    cache_capacity_ = grown;
  }
  grow_rope_table(needed);

  const int64_t kv_width = n_head_kv_ * head_dim_;

  /* The prompt is uploaded once and gathered on the device; only the ids cross
   * the bus, not the embedding rows they select. */
  backend_->copy_to_device(tokens_, tokens, n * 4);
  backend_->embedding(tokens_, n, tok_embd_, n_vocab_, n_embd_, x_);

  for (int64_t il = 0; il < n_layer_; ++il) {
    const Layer &layer = layers_[static_cast<std::size_t>(il)];

    backend_->rms_norm(x_, layer.attn_norm, x_norm_, n, n_embd_, rms_eps_);

    matmul(layer.wq, x_norm_, q_, n, false);
    matmul(layer.wk, x_norm_, k_, n, false);
    matmul(layer.wv, x_norm_, v_, n, false);

    /* Per head, over `head_dim`. The projection is `n_head * head_dim` wide, so
     * normalizing the row instead would reduce over sixteen times as many values
     * and produce a different -- still finite, still plausible -- model. The
     * weight is shared across heads and tokens, which is why one call can do all
     * of them. */
    backend_->rms_norm(q_, layer.q_norm, q_, n * n_head_, head_dim_, rms_eps_);
    backend_->rms_norm(k_, layer.k_norm, k_, n * n_head_kv_, head_dim_, rms_eps_);

    backend_->rope_neox(q_, n, n_head_, head_dim_, start_pos, rope_cos_, rope_sin_);
    backend_->rope_neox(k_, n, n_head_kv_, head_dim_, start_pos, rope_cos_, rope_sin_);

    /* The cache holds the *rotated* key: attention scores are not a function of
     * the unrotated one, so storing before the rotation would silently drop the
     * positional signal for every token after the first.
     *
     * The write goes to this layer's own slab. Writing to a single shared slab
     * would leave the layer's keys overwritten by the next layer's by the time
     * the next call read them. */
    const int64_t layer_offset = cache_capacity_ * kv_width * il;
    kernel::DeviceBuffer layer_k{k_cache_.handle +
                                     static_cast<uintptr_t>(layer_offset * 4),
                                 kv_width * cache_capacity_ * 4};
    kernel::DeviceBuffer layer_v{v_cache_.handle +
                                     static_cast<uintptr_t>(layer_offset * 4),
                                 kv_width * cache_capacity_ * 4};
    /* Both directions are device memory, so this is `copy_device_to_device` and
     * not `copy_to_device`: the source is a device address, and a backend that
     * dereferenced it as a host pointer would read the CPU's memory on a card.
     * The CPU backend cannot tell the difference, which is exactly why the
     * interface carries the distinction. */
    backend_->copy_device_to_device(
        kernel::DeviceBuffer{layer_k.handle + static_cast<uintptr_t>(start_pos * kv_width * 4),
                             n * kv_width * 4},
        k_, n * kv_width * 4);
    backend_->copy_device_to_device(
        kernel::DeviceBuffer{layer_v.handle + static_cast<uintptr_t>(start_pos * kv_width * 4),
                             n * kv_width * 4},
        v_, n * kv_width * 4);

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    backend_->attention(q_, n, n_head_, layer_k, layer_v, n_head_kv_, head_dim_,
                        /*first_key=*/0, start_pos, scale, attn_, scores_);

    matmul(layer.wo, attn_, x_, n, /*accumulate=*/true);

    backend_->rms_norm(x_, layer.ffn_norm, x_norm_, n, n_embd_, rms_eps_);
    matmul(layer.w_gate, x_norm_, gate_, n, false);
    matmul(layer.w_up, x_norm_, up_, n, false);
    backend_->silu_mul(gate_, up_, ffn_, n * n_ff_);
    matmul(layer.w_down, ffn_, x_, n, /*accumulate=*/true);
  }

  /* Only the last position's logits are produced. A caller wanting a
   * continuation asks for one token at a time; the prefill's earlier positions
   * are computed because the layers are sequential, but their logits are not
   * wanted and projecting them would be 151936 values per token of waste.
   *
   * The final row is a handle into `x_` with an offset, rather than a copy --
   * a handle is an address and an address plus an offset is the next row. That
   * is a legitimate thing to do to a `DeviceBuffer` precisely because the graph
   * never dereferences one. */
  const kernel::DeviceBuffer x_last{x_.handle + static_cast<uintptr_t>((n - 1) * n_embd_ * 4),
                                    n_embd_ * 4};
  backend_->rms_norm(x_last, output_norm_, last_, 1, n_embd_, rms_eps_);
  matmul(output_, last_, logits_, 1, false);

  cache_length_ = start_pos + n;
  return logits_;
}

}  // namespace pocketllm