#include "model/qwen3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "abi/spec.h"
#include "kernel/parallel.h"
#include "kernel/repack_q4k.h"
#include "quant/q8k.h"
#include "quant/blocks.h"
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

/* The two k-quants the packed kernels decode. A `q4_k_m` checkpoint -- the
 * format this project's ladder starts at -- is made of exactly these two plus
 * the f32 norms, so these are the only ids that need to name a kernel. Anything
 * else is refused at load with the type's own name in the message, which is how
 * a `q5_k_m` or an `iq4_xs` file gets a diagnosis rather than a wrong token. */
constexpr int kTypeQ4K = 12;
constexpr int kTypeQ6K = 14;

bool has_packed_kernel(int type_id) { return type_id == kTypeQ4K || type_id == kTypeQ6K; }

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

Qwen3Model::~Qwen3Model() {
  /* Every buffer the model owns is a raw handle, so a defaulted destructor
   * frees none of them -- and the leak is invisible on the host, where a few
   * gigabytes fall back into a terabyte of RAM, and fatal on a card, where the
   * second session is the one that runs out.  This was found by opening two
   * CUDA sessions in a row and watching the second fail to allocate.
   *
   * The buffers are not stored in one list because they are not one kind of
   * thing: the scratch is a fixed set of members, the layers are a vector of
   * structs holding both handles and weights, and the two caches are managed
   * together.  Releasing them in three statements says what the shape is. */
  if (backend_ == nullptr) {
    return;
  }
  for (kernel::DeviceBuffer buffer : {x_, x_norm_, q_, k_, v_, attn_, gate_, up_, ffn_, last_,
                                      logits_, scores_, tokens_, rope_cos_, rope_sin_, act_packed_,
                                      panel_tile_, k_cache_, v_cache_, output_norm_}) {
    if (buffer.handle != 0) {
      backend_->release(buffer);
    }
  }
  for (const Layer &layer : layers_) {
    for (kernel::DeviceBuffer buffer :
         {layer.attn_norm, layer.ffn_norm, layer.q_norm, layer.k_norm}) {
      if (buffer.handle != 0) {
        backend_->release(buffer);
      }
    }
    for (const Weight &weight : {layer.wq, layer.wk, layer.wv, layer.wo, layer.w_gate, layer.w_up,
                                 layer.w_down}) {
      release_weight(weight);
    }
  }
  /* The output projection is tied to the embedding when the checkpoint has no
   * separate head, in which case the two members hold the *same* handle and
   * releasing both would be a double free.  Comparing the handles is how the
   * tie is detected here; a flag would be a second copy of a fact that is
   * already in the values. */
  if (output_.rows != 0 && (output_.rows != tok_embd_.rows || output_.cols != tok_embd_.cols)) {
    release_weight(output_);
  }
  release_weight(tok_embd_);
}

void Qwen3Model::release_weight(const Weight &weight) {
  /* Exactly one of the two handles is live in a `Weight`; releasing the other
   * would free a null and, on a device, fault. */
  if (weight.quantized) {
    if (weight.blocks.handle != 0) {
      backend_->release(weight.blocks);
    }
  } else if (weight.data.handle != 0) {
    backend_->release(weight.data);
  }
}

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
    /* A quantized tensor reaching this function is a caller that chose the
     * dense binding for it, not a format this build cannot read: the paths are
     * picked in `bind_matrix` and `bind_table`. Naming both the type and the
     * function is what makes that distinction legible. */
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", which bind_dense cannot widen; the packed path binds that one");
  }

  kernel::DeviceBuffer device = backend_->allocate(elements * 4);
  backend_->copy_to_device(device, widened.data(), elements * 4);
  return device;
}

Weight Qwen3Model::bind_packed(const GgufReader &checkpoint, const std::string &name) {
  uint64_t nbytes = 0;
  const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);

  Weight weight;
  weight.blocks = backend_->allocate(static_cast<int64_t>(nbytes));
  backend_->copy_to_device(weight.blocks, bytes, static_cast<int64_t>(nbytes));
  weight.quantized = true;
  weight.nbytes = static_cast<int64_t>(nbytes);
  return weight;
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

  if (info->type_id == kTypeF32 || info->type_id == kTypeF16) {
    weight.data = bind_dense(checkpoint, name);
    return weight;
  }
  if (!has_packed_kernel(info->type_id)) {
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", which this build has no kernel for; it reads f32, f16, q4_k and q6_k");
  }
  /* The contraction axis has to be a whole number of super-blocks. GGUF
   * guarantees it for a valid file, and the alternative to checking is a
   * row stride that is a fraction of a block -- which would decode a weight
   * from the next row's bytes and look like a small numerical error rather
   * than a bad file. */
  if (weight.cols % 256 != 0) {
    throw Error("tensor '" + name + "' has " + std::to_string(weight.cols) +
                " columns, which is not a whole number of 256-weight blocks");
  }
  /* The size has to be exactly what the block geometry says. A reader whose
   * table and whose kernel disagreed about a block's width would otherwise
   * stride the file one way and walk it another, which reads every weight from
   * a neighbouring position -- a small numerical error rather than a bad-file
   * message. */
  const int64_t expected =
      weight.rows * (weight.cols / 256) * quant::block_bytes_of(info->type_id);
  if (weight.nbytes != expected) {
    throw Error("tensor '" + name + "' is " + std::to_string(weight.nbytes) + " bytes but " +
                std::to_string(weight.rows) + "x" + std::to_string(weight.cols) + " of " +
                ggml_type_of(info->type_id).name + " is " + std::to_string(expected));
  }
  weight.quantized = true;
  weight.blocks = bind_packed(checkpoint, name).blocks;
  /* A GEMM weight is the one place the panel layout is wanted, and only when it
   * is a q4_K whose shape tiles exactly.  The token embedding is deliberately
   * *not* repacked: it feeds `embedding_quant`'s gather as well as the tied head
   * GEMM, and the gather reads the file layout -- qwen3-0.6B carries a separate
   * `output.weight`, so the tie costs nothing here and the table stays readable
   * by the one kernel that already understands it.
   *
   * **The shape guard skips rather than throws, and that is not a fallback.**
   * The panel path is on by default now, so a checkpoint with a q4_K weight that
   * does not tile would otherwise be *rejected at load* -- an engine that used to
   * read a file suddenly refusing it, for a reason that is about a tuning
   * decision and not about the file.  A weight that does not tile keeps the row
   * kernel: `Weight::panel` stays false, and `matmul` reads exactly that flag.
   * The two layouts never meet inside one weight, so this is a per-matrix
   * decision, not the half-a-repack state `repack_weight` refuses. */
  if (panel_gemm_ && weight.type_id == kTypeQ4K && weight.rows % 8 == 0 && weight.cols % 8 == 0) {
    repack_weight(weight);
  }
  /* q6_K gets the byte-expanded repack, but only on a backend that reads it --
   * the layout is CUDA-only, and the flag is what `matmul` and the backend both
   * key off. Any shape works (the repack has no tiling requirement), so unlike
   * the q4_K panels there is nothing to guard but the type and the backend. */
  if (q6k_repack_enabled_ && weight.type_id == kTypeQ6K) {
    repack_weight_q6k(checkpoint, name, weight);
  }
  return weight;
}

void Qwen3Model::repack_weight_q6k(const GgufReader &checkpoint, const std::string &name,
                                   Weight &weight) const {
  /* The transform runs on *host* bytes and its result is uploaded.  The q4_K
   * panel repack reads and writes a device buffer in place because on the CPU
   * backend a handle is an address; this one runs on the card, where
   * `blocks.handle` is a device pointer the host cannot dereference -- taking
   * the same shape here segfaults at load.  Re-reading the file's bytes for the
   * repack costs one extra pass over a q6_K tensor at load and nothing per
   * token. */
  uint64_t nbytes = 0;
  const uint8_t *src = checkpoint.tensor_data(name, &nbytes);
  const int64_t blocks = weight.rows * (weight.cols / 256);
  const int64_t expanded = blocks * quant::kQ6KRepackedStride;
  std::vector<uint8_t> host(static_cast<std::size_t>(expanded));
  kernel::repack_weights_q6k(src, weight.rows, weight.cols, host.data());
  kernel::DeviceBuffer repacked = backend_->allocate(expanded);
  backend_->copy_to_device(repacked, host.data(), expanded);
  backend_->release(weight.blocks);
  weight.blocks = repacked;
  weight.nbytes = expanded;
  weight.q6k_repacked = true;
}

/* Panels are eight columns and four activation rows, so both axes have to
 * divide.  `cols` is already a multiple of 256 by the caller's check; the 8 is
 * the one that can be violated, and a matrix that did would have to fall back --
 * which is the "half a repack" state the flag exists to make impossible.  The
 * rest of the shape (a 256-multiple contraction axis, the exact byte count) is
 * the caller's, already checked above. */
void Qwen3Model::repack_weight(Weight &weight) const {
  if (weight.rows % 8 != 0 || weight.cols % 8 != 0) {
    throw Error("tensor of shape " + std::to_string(weight.rows) + "x" +
                std::to_string(weight.cols) +
                " cannot be repacked into eight-column panels; POCKETLLM_CPU_REPACK needs both "
                "axes to be multiples of 8");
  }
  if (weight.nbytes != weight.rows * (weight.cols / 256) * quant::kQ4KBlockBytes) {
    throw Error("tensor of shape " + std::to_string(weight.rows) + "x" +
                std::to_string(weight.cols) + " holds " + std::to_string(weight.nbytes) +
                " bytes, which is not a whole number of q4_K blocks; it cannot be repacked");
  }
  /* **Out of line, and not for tidiness.**  The repack is a *permutation* of the
   * matrix's own bytes: block `b` of column-panel `p` reads rows `p*8 .. p*8+7`
   * at one stride and writes a contiguous 1152-byte run at another, so a panel's
   * output lands on top of some *later* panel's input.  Written in place it reads
   * bytes it has already overwritten -- and the result is not obviously wrong:
   * the numbers are still in the right range, so it surfaces as fluent nonsense
   * from layer nine rather than as a crash -- and a prototype that repacked
   * through two independent arrays, which is the obvious way to write the
   * packer and the way it was first measured, cannot show the bug at all. */
  kernel::DeviceBuffer scratch = backend_->allocate(weight.nbytes);
  kernel::repack_weights_q4k(reinterpret_cast<const uint8_t *>(weight.blocks.handle), weight.rows,
                             weight.cols, reinterpret_cast<kernel::Q4Kx8 *>(scratch.handle));
  backend_->copy_device_to_device(weight.blocks, scratch, weight.nbytes);
  backend_->release(scratch);
  weight.panel = true;
}

Weight Qwen3Model::bind_table(const GgufReader &checkpoint, const std::string &name) {
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    missing("tensor", name);
  }
  if (info->dimensions.size() != 2) {
    throw Error("tensor '" + name + "' is not a matrix (it has " +
                std::to_string(info->dimensions.size()) + " dimensions)");
  }
  Weight table;
  table.cols = static_cast<int64_t>(info->dimensions[0]);
  table.rows = static_cast<int64_t>(info->dimensions[1]);
  table.type_id = info->type_id;
  table.nbytes = static_cast<int64_t>(info->nbytes);
  if (info->type_id == kTypeF32 || info->type_id == kTypeF16) {
    table.data = bind_dense(checkpoint, name);
    return table;
  }
  if (!has_packed_kernel(info->type_id) || table.cols % 256 != 0) {
    throw Error(std::string("token_embd.weight is ") + ggml_type_of(info->type_id).name +
                " with " + std::to_string(table.cols) +
                " columns; this build gathers from f32, f16, q4_k and q6_k tables whose row is a "
                "whole number of 256-weight blocks");
  }
  table.quantized = true;
  table.blocks = bind_packed(checkpoint, name).blocks;
  return table;
}

std::unique_ptr<Qwen3Model> Qwen3Model::load(const GgufReader &checkpoint,
                                             kernel::Backend &backend) {
  const std::string arch = checkpoint.get_string("general.architecture", "");
  if (arch != "qwen3") {
    throw Error("this build runs 'qwen3' checkpoints; this one declares architecture '" + arch + "'");
  }

  std::unique_ptr<Qwen3Model> model(new Qwen3Model(backend));
  /* The repacked panel GEMM is opt-in, CPU-only and AVX2-only.  Deciding it here,
   * once, keeps `matmul` from asking three questions per weight per layer, and
   * keeps the answer the same for every weight in the session -- which is what
   * makes the path a *selected whole GEMM* rather than a per-tensor choice. */
  model->panel_gemm_ = kernel::repack_enabled() && kernel::repack_available() &&
                       std::string(backend.name()) == "cpu";
  /* The byte-expanded q6_K layout is selected by the backend, not a build flag:
   * only the CUDA decoder reads it, so it is on exactly when the session's
   * backend is `cuda`. Deciding it here, once, is the same shape as the panels
   * above -- one answer for every weight, so the layout never varies within a
   * tensor. */
  model->q6k_repack_enabled_ = std::string(backend.name()) == "cuda";

  /* The cache width is the backend's decision and it is taken here, before the
   * first allocation, because every slab offset in `forward` is derived from
   * it.  f16 is the interface's default and the CPU backend's; the card
   * overrides it.  A backend that cannot consume an f16 cache overrides the
   * same call, which is what keeps this line from deciding for it. */
  model->kv_dtype_ = backend.preferred_kv_dtype();

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

  /* The table is bound through its own path because it is both a gather source
   * and -- when the checkpoint ties the head to it -- a GEMM operand, and the
   * two need different things from the same bytes. */
  model->tok_embd_ = model->bind_table(checkpoint, "token_embd.weight");
  const GgufTensorInfo *embd_info = checkpoint.tensor("token_embd.weight");
  if (static_cast<int64_t>(embd_info->dimensions[0]) != model->n_embd_) {
    throw Error("token_embd.weight has " + std::to_string(embd_info->dimensions[0]) +
                " columns but embedding_length says " + std::to_string(model->n_embd_));
  }

  /* Qwen3 ties the output projection to the embedding when the checkpoint does
   * not carry a separate head, and this one does carry one. Binding whichever
   * is present -- rather than requiring `output.weight` -- is what lets the same
   * loader read a smaller conversion later. The tie copies the whole `Weight`
   * and not just the handle: a tied head is the same tensor, so it is the same
   * shape and the same format, and a copy of the handle alone would lose which
   * of the two it is. */
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
  backend_->release(act_packed_);
  backend_->release(panel_tile_);

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
   * it is indexed by *absolute* key position -- so the span is the end position
   * of this batch, not the batch size. Sizing it to the batch is the easy mistake
   * and a quiet one: a fresh prompt of n tokens would happen to fit
   * (`end_pos == n`), and a decode at position 300 would write 300 floats past a
   * two-float allocation.
   *
   * How much room that needs on top is the backend's to say, because it depends
   * on how many of the (query, head) pairs it runs at once. */
  scores_ = backend_->allocate(backend_->attention_scratch(n, n_head_, end_pos));
  /* The token ids go to the device once per forward: the embedding is a gather
   * and a backend whose indices live on the host would have to fetch the whole
   * table back. */
  tokens_ = backend_->allocate(n * 4 + 64);
  /* The packed activations a panel kernel reads.  **The widest contraction axis
   * is `n_ff`, not `n_embd`**: `w_down` contracts over the FFN's hidden width,
   * which is four times the model's.  Sizing this to `n_embd` overflowed on that
   * one weight -- and overflowed *silently*, because a heap buffer has no
   * bounds, so the entries it trampled were the next weight's or the KV cache's
   * and the model answered `0! 0! 0!` nine layers later.  The `n / 4 + 4` is the
   * panel GEMM's own slop (`nr / 4` groups plus four for the tail pass). */
  int64_t act_bytes = 0;
  if (panel_gemm_) {
    const int64_t widest = n_embd_ > n_ff_ ? n_embd_ : n_ff_;
    act_bytes = (n / 4 + 4) * (widest / 256) * static_cast<int64_t>(sizeof(kernel::Q8Kx4));
  }
  act_packed_ = backend_->allocate(act_bytes > 0 ? act_bytes : 4);
  /* The accumulating tile is `n` rows by an output axis, and only two weights
   * ever write through it: `wo` and `w_down`, whose outputs are both `n_embd`
   * wide.  It is sized to the model's *widest* axis anyway (`n_ff`, four times
   * `n_embd` here) rather than to the 1024 the two of them actually need, so
   * that a future accumulating weight with a wider output cannot overflow it by
   * accident -- the failure the comment above records.  Nothing reads it before
   * the call that writes it, so it needs no initialization. */
  panel_tile_ = backend_->allocate(panel_gemm_ ? n * n_ff_ * 4 : 4);
}

void Qwen3Model::matmul(const Weight &w, kernel::DeviceBuffer x, kernel::DeviceBuffer out, int64_t m,
                        bool accumulate) const {
  kernel::DeviceBuffer no_bias;
  if (w.quantized) {
    /* A repacked weight is readable at every `m`, which is why the panel path is
     * selected for the whole session rather than per shape: the decode GEMM has
     * to come from the same bytes the prefill GEMM does.  When `w.panel` is set
     * the row kernel is never called for it -- not even for the rows a batched
     * call would leave over -- because `dot_row_q8k` reads the file's per-row
     * blocks and these are panels.  It would not fail; it would answer fluently.
     *
     * Two kernels, one weight layout:
     *   `m % 8 == 0` -> `gemm_q4k_8x8`, the panel GEMM, whose whole win is that
     *                   a prefill no longer re-decodes the matrix `m / 8` times;
     *   otherwise    -> `gemv_q4k_8x8`, one activation row at a time, which is
     *                   the decode path.
     * Neither is bit-identical to `dot_row_q8k`; `kernel/repack_q4k.h` says why,
     * and `gemm_quant`'s own schedule -- the thing the mutual-consistency
     * invariant pins -- is untouched because neither of these is it. */
    if (w.panel && m % 8 == 0) {
      const float *src = reinterpret_cast<const float *>(x.handle);
      /* An accumulating call builds its tile out of line and adds it to `out` at
       * the end, because `out` and `x` are different buffers: `out` is the
       * residual being accumulated into and `x` is the operand the GEMM is
       * consuming, and for `wo`/`w_down` the kernel would otherwise overwrite
       * the row it is still reading. */
      float *dst = reinterpret_cast<float *>(out.handle);
      if (accumulate) {
        dst = reinterpret_cast<float *>(panel_tile_.handle);
      }
      kernel::Q8Kx4 *acts = reinterpret_cast<kernel::Q8Kx4 *>(act_packed_.handle);
      kernel::repack_activations(src, m, w.cols, acts);
      kernel::gemm_q4k_8x8(static_cast<int>(w.cols), dst, static_cast<size_t>(w.rows),
                           reinterpret_cast<const kernel::Q4Kx8 *>(w.blocks.handle), acts,
                           static_cast<int>(m), static_cast<int>(w.rows));
      if (accumulate) {
        /* The residual is `out`, not `x`.  Both accumulating calls in the graph
         * pass `x_` as the destination and the projection's own input as `x`, so
         * `out` already holds the residual the new term is added to.  Adding `x`
         * instead is a plausible-looking line that silently replaces the residual
         * with the projection -- and because the graph's last two matmuls are
         * both accumulating, it corrupts every block from layer 0 on. */
        float *out_f = reinterpret_cast<float *>(out.handle);
        const int64_t total = m * w.rows;
        for (int64_t i = 0; i < total; ++i) {
          out_f[i] = out_f[i] + dst[i];
        }
      }
      return;
    }
    if (w.panel) {
      /* The panel GEMV writes its whole output row, so an accumulating call
       * needs the same out-of-line tile the GEMM does. */
      const int64_t row_blocks = w.cols / quant::kBlockWeights;
      const float *src = reinterpret_cast<const float *>(x.handle);
      quant::Q8KBlock *acts = reinterpret_cast<quant::Q8KBlock *>(act_packed_.handle);
      kernel::parallel_for(m, 1, [&](int64_t lo, int64_t hi, int64_t) {
        for (int64_t r = lo; r < hi; ++r) {
          quant::quantize_row_q8_k(src + r * w.cols, acts + r * row_blocks, w.cols);
        }
      });
      float *dst = accumulate ? reinterpret_cast<float *>(panel_tile_.handle)
                              : reinterpret_cast<float *>(out.handle);
      kernel::gemv_q4k_8x8(static_cast<int>(w.cols), dst, static_cast<size_t>(w.rows),
                           reinterpret_cast<const kernel::Q4Kx8 *>(w.blocks.handle), acts,
                           static_cast<int>(m), static_cast<int>(w.rows));
      if (accumulate) {
        /* `out` is the residual; see the panel GEMM above. */
        float *out_f = reinterpret_cast<float *>(out.handle);
        const int64_t total = m * w.rows;
        for (int64_t i = 0; i < total; ++i) {
          out_f[i] = out_f[i] + dst[i];
        }
      }
      return;
    }
    backend_->gemm_quant(x, w.blocks, no_bias, out, m, w.rows, w.cols, w.type_id, accumulate,
                         w.q6k_repacked);
    return;
  }
  backend_->gemm(x, w.data, no_bias, out, m, w.rows, w.cols, accumulate);
}

void Qwen3Model::embed(kernel::DeviceBuffer tokens, int64_t n, kernel::DeviceBuffer out) const {
  if (tok_embd_.quantized) {
    backend_->embedding_quant(tokens, n, tok_embd_.blocks, n_vocab_, n_embd_, tok_embd_.type_id, out);
    return;
  }
  backend_->embedding(tokens, n, tok_embd_.data, n_vocab_, n_embd_, out);
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
    const int64_t elem = kernel::kv_dtype_size(kv_dtype_);
    k_cache_ = backend_->allocate(static_cast<int64_t>(per_layer) * n_layer_ * elem);
    v_cache_ = backend_->allocate(static_cast<int64_t>(per_layer) * n_layer_ * elem);
    cache_capacity_ = grown;
  }
  grow_rope_table(needed);

  const int64_t kv_width = n_head_kv_ * head_dim_;
  const int64_t kv_elem = kernel::kv_dtype_size(kv_dtype_);

  /* The prompt is uploaded once and gathered on the device; only the ids cross
   * the bus, not the embedding rows they select. */
  backend_->copy_to_device(tokens_, tokens, n * 4);
  embed(tokens_, n, x_);

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
                                     static_cast<uintptr_t>(layer_offset * kv_elem),
                                 kv_width * cache_capacity_ * kv_elem};
    kernel::DeviceBuffer layer_v{v_cache_.handle +
                                     static_cast<uintptr_t>(layer_offset * kv_elem),
                                 kv_width * cache_capacity_ * kv_elem};
    /* Both directions are device memory, so this is `copy_device_to_device` and
     * not `copy_to_device`: the source is a device address, and a backend that
     * dereferenced it as a host pointer would read the CPU's memory on a card.
     * The CPU backend cannot tell the difference, which is exactly why the
     * interface carries the distinction.
     *
     * When the cache is wider than the activations -- an f32 cache, which is
     * every backend but the CPU one -- this is a plain copy of the rows.  The
     * f16 cache is a *lossy* copy and has to be written by a kernel that
     * converts, one row at a time, because the source rows are `head_dim`
     * apart and the destination rows are `n_head_kv * head_dim` apart: a flat
     * convert over the batch would interleave the heads into each other's
     * slots. */
    if (kv_elem == 4) {
      backend_->copy_device_to_device(
          kernel::DeviceBuffer{layer_k.handle + static_cast<uintptr_t>(start_pos * kv_width * 4),
                               n * kv_width * 4},
          k_, n * kv_width * 4);
      backend_->copy_device_to_device(
          kernel::DeviceBuffer{layer_v.handle + static_cast<uintptr_t>(start_pos * kv_width * 4),
                               n * kv_width * 4},
          v_, n * kv_width * 4);
    } else {
      const uintptr_t at = static_cast<uintptr_t>(start_pos * kv_width * kv_elem);
      const int64_t bytes = n * kv_width * kv_elem;
      backend_->kv_append(kernel::DeviceBuffer{layer_k.handle + at, bytes}, k_, n, n_head_kv_,
                          head_dim_, kv_elem);
      backend_->kv_append(kernel::DeviceBuffer{layer_v.handle + at, bytes}, v_, n, n_head_kv_,
                          head_dim_, kv_elem);
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
    backend_->attention(q_, n, n_head_, layer_k, layer_v, n_head_kv_, head_dim_,
                        /*first_key=*/0, start_pos, scale, attn_, scores_, kv_dtype_);

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