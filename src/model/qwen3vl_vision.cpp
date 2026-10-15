#include "model/qwen3vl_vision.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "quant/half.h"
#include "runtime/status.h"
#include "abi/spec.h"

namespace pocketllm {
namespace {

constexpr int kTypeF32 = 0;
constexpr int kTypeF16 = 1;

/* The tower's rotary base.  A constant here rather than a read, because the
 * mmproj carries no rope key at all: llama.cpp builds a `qwen3vl` vision graph
 * with `freq_base = 10000` written into the call, so there is nothing in the file
 * for this to disagree with.  A checkpoint that needed another base would need a
 * key, and this is where the refusal would go. */
constexpr float kVisionRopeFreqBase = 10000.0F;

std::string vision_key(const std::string &suffix) { return "clip.vision." + suffix; }

/* A positive integer hyperparameter, or a failure naming it.
 *
 * The reader's accessors default rather than throw, which is right for a key a
 * model may or may not have.  These are not optional: every one is geometry the
 * graph allocates from, and a defaulted `n_embd` of zero is a division rather
 * than a message. */
int64_t require_int(const GgufReader &checkpoint, const std::string &name) {
  const int64_t value = checkpoint.get_int(name, 0);
  if (value <= 0) {
    throw Error("the mmproj has no positive metadata key '" + name +
                "'; this build reads a vision tower's geometry from the file rather than assuming "
                "it");
  }
  return value;
}

const GgufArray *require_array(const GgufReader &checkpoint, const std::string &name) {
  const GgufValue *value = checkpoint.find(name);
  const GgufArray *array = value == nullptr ? nullptr : std::get_if<GgufArray>(value);
  if (array == nullptr) {
    throw Error("the mmproj has no metadata array '" + name + "'");
  }
  return array;
}

/* A metadata flag, which GGUF writes as its own boolean type and which the
 * reader's integer accessor therefore does not see -- asking for it as an integer
 * answers the fallback, so a file that *has* a vision encoder reads as one that
 * does not.  Both spellings are accepted because a converter is free to write
 * either, and the answer is the same. */
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

/* The per-channel normalization, which the preprocessing applies after its
 * resize.  Three values each, and read rather than assumed: a checkpoint whose
 * tower was trained with different statistics would otherwise be normalized with
 * this one's and answer with a different model. */
void read_channel_stats(const GgufReader &checkpoint, const std::string &name, float (&out)[3]) {
  const GgufArray *array = require_array(checkpoint, name);
  if (array->floats.size() != 3) {
    throw Error("metadata key '" + name + "' has " + std::to_string(array->floats.size()) +
                " values, not 3");
  }
  for (int c = 0; c < 3; ++c) {
    out[c] = static_cast<float>(array->floats[static_cast<std::size_t>(c)]);
    if (out[c] == 0.0F) {
      throw Error("metadata key '" + name +
                  "' has a zero entry; the normalization would divide by it");
    }
  }
}

/* A tensor's bytes, widened to f32 and uploaded.
 *
 * The same rule as `qwen3.cpp`'s `bind_dense` -- f32 is copied, f16 is widened,
 * anything else is refused by name -- and deliberately a second copy of it rather
 * than a shared helper.  The two loaders read two different kinds of file (a
 * quantized text model and a dense tower), and the text path is the one every
 * existing checkpoint goes through, so moving the helper would put this change's
 * blast radius on the model that already works. */
kernel::DeviceBuffer bind_dense(const GgufReader &checkpoint, kernel::Backend &backend,
                                std::vector<kernel::DeviceBuffer> &owned, const std::string &name) {
  uint64_t nbytes = 0;
  const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);
  const GgufTensorInfo *info = checkpoint.tensor(name);
  int64_t elements = 1;
  for (uint64_t dim : info->dimensions) {
    elements *= static_cast<int64_t>(dim);
  }

  std::vector<float> widened(static_cast<std::size_t>(elements));
  if (info->type_id == kTypeF32) {
    std::memcpy(widened.data(), bytes, static_cast<std::size_t>(nbytes));
  } else if (info->type_id == kTypeF16) {
    for (int64_t i = 0; i < elements; ++i) {
      widened[static_cast<std::size_t>(i)] = half_to_float(load_u16(bytes + 2 * i));
    }
  } else {
    throw Error("tensor '" + name + "' is " + ggml_type_of(info->type_id).name +
                ", and the vision tower is dense: this loader reads f32 and f16 only");
  }

  kernel::DeviceBuffer device = backend.allocate(elements * 4);
  backend.copy_to_device(device, widened.data(), elements * 4);
  owned.push_back(device);
  return device;
}

/* A matrix projection with its bias.  `base` is the tensor name without the
 * `.weight`/`.bias` suffix, which is how the checkpoint names both halves of
 * every projection in the tower. */
VisionLinear bind_linear(const GgufReader &checkpoint, kernel::Backend &backend,
                         std::vector<kernel::DeviceBuffer> &owned, const std::string &base,
                         int64_t expected_rows, int64_t expected_cols) {
  const std::string name = base + ".weight";
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    throw Error("the vision tower is missing tensor '" + name +
                "'; a tower missing one weight answers with a plausible embedding rather than an "
                "error");
  }
  /* GGUF writes the fastest-varying axis first, so a matrix arrives as
   * (contraction, output) -- the transpose of how the product is written.  The
   * bytes are already row-major in that order, so nothing is transposed here and
   * these two lines are the only place the conventions meet. */
  if (info->dimensions.size() != 2 || static_cast<int64_t>(info->dimensions[0]) != expected_cols ||
      static_cast<int64_t>(info->dimensions[1]) != expected_rows) {
    throw Error("tensor '" + name + "' is not (" + std::to_string(expected_cols) + ", " +
                std::to_string(expected_rows) + ")");
  }
  VisionLinear linear;
  linear.rows = expected_rows;
  linear.cols = expected_cols;
  linear.weight = bind_dense(checkpoint, backend, owned, name);
  linear.bias = bind_dense(checkpoint, backend, owned, base + ".bias");
  return linear;
}

/* The convolution's weight, whose four dimensions are the kernel's two spatial
 * axes and then the channel pair.  `conv2d`'s header says why that is already the
 * matrix layout and not something that has to be rearranged. */
VisionLinear bind_conv(const GgufReader &checkpoint, kernel::Backend &backend,
                       std::vector<kernel::DeviceBuffer> &owned, const std::string &name,
                       int64_t out_ch, int64_t in_ch, int64_t kernel) {
  const GgufTensorInfo *info = checkpoint.tensor(name);
  if (info == nullptr) {
    throw Error("the vision tower is missing tensor '" + name + "'");
  }
  const std::vector<uint64_t> &dims = info->dimensions;
  if (dims.size() != 4 || static_cast<int64_t>(dims[0]) != kernel ||
      static_cast<int64_t>(dims[1]) != kernel || static_cast<int64_t>(dims[2]) != in_ch ||
      static_cast<int64_t>(dims[3]) != out_ch) {
    throw Error("tensor '" + name + "' is not (" + std::to_string(kernel) + ", " +
                std::to_string(kernel) + ", " + std::to_string(in_ch) + ", " +
                std::to_string(out_ch) + ")");
  }
  VisionLinear linear;
  linear.rows = out_ch;
  linear.cols = in_ch * kernel * kernel;
  linear.weight = bind_dense(checkpoint, backend, owned, name);
  return linear;
}

kernel::DeviceBuffer slice_of(kernel::DeviceBuffer parent, int64_t byte_offset, int64_t bytes) {
  return kernel::DeviceBuffer{parent.handle + static_cast<uintptr_t>(byte_offset), bytes};
}

/* Which raster patch a merge-ordered token is: the inverse of
 * `Qwen3VlVision::build_merge_index`.  Tokens come in groups of `merge^2`, one
 * per cell of a `merge x merge` block, and the groups run over the merged grid in
 * row-major order. */
void raster_patch(int64_t token, int64_t merge, int64_t merged_side, int64_t *y, int64_t *x) {
  const int64_t within = token % (merge * merge);
  const int64_t cell = token / (merge * merge);
  const int64_t row = cell / merged_side;
  const int64_t column = cell % merged_side;
  *y = merge * row + within / merge;
  *x = merge * column + within % merge;
}

}  // namespace

std::string vision_arch_refusal(const GgufReader &checkpoint) {
  const std::string arch = checkpoint.get_string("general.architecture", "");
  if (arch != "clip") {
    return "this file's architecture is '" + arch + "', and a vision tower's mmproj declares 'clip'";
  }
  if (!metadata_flag(checkpoint, "clip.has_vision_encoder")) {
    return "this file has no vision encoder (`clip.has_vision_encoder` is false)";
  }
  const std::string projector = checkpoint.get_string("clip.projector_type", "");
  if (projector != "qwen3vl_merger") {
    return "this build reads the `qwen3vl_merger` projector, and this file declares '" + projector +
           "'";
  }
  return std::string();
}

Qwen3VlVision::Qwen3VlVision(kernel::Backend &backend) : backend_(&backend) {}

Qwen3VlVision::~Qwen3VlVision() {
  for (kernel::DeviceBuffer buffer : owned_) {
    backend_->release(buffer);
  }
}

std::unique_ptr<Qwen3VlVision> Qwen3VlVision::load(const GgufReader &checkpoint,
                                                   kernel::Backend &backend) {
  const std::string refusal = vision_arch_refusal(checkpoint);
  if (!refusal.empty()) {
    throw Error("this file is not a vision tower this build can run: " + refusal);
  }
  std::unique_ptr<Qwen3VlVision> model(new Qwen3VlVision(backend));

  model->n_layer_ = require_int(checkpoint, vision_key("block_count"));
  model->n_embd_ = require_int(checkpoint, vision_key("embedding_length"));
  model->n_head_ = require_int(checkpoint, vision_key("attention.head_count"));
  model->n_ff_ = require_int(checkpoint, vision_key("feed_forward_length"));
  model->patch_size_ = require_int(checkpoint, vision_key("patch_size"));
  model->image_size_ = require_int(checkpoint, vision_key("image_size"));
  model->projection_dim_ = require_int(checkpoint, vision_key("projection_dim"));
  model->merge_ = require_int(checkpoint, vision_key("spatial_merge_size"));
  model->eps_ =
      static_cast<float>(checkpoint.get_float(vision_key("attention.layer_norm_epsilon"), 1e-6));

  if (model->n_embd_ % model->n_head_ != 0) {
    throw Error("the tower's embedding length " + std::to_string(model->n_embd_) +
                " is not a multiple of its head count " + std::to_string(model->n_head_));
  }
  model->head_dim_ = model->n_embd_ / model->n_head_;
  if (model->image_size_ % model->patch_size_ != 0) {
    throw Error("the tower's image size " + std::to_string(model->image_size_) +
                " is not a multiple of its patch size " + std::to_string(model->patch_size_));
  }
  model->grid_side_ = model->image_size_ / model->patch_size_;
  if (model->grid_side_ % model->merge_ != 0) {
    throw Error("the tower's patch grid is " + std::to_string(model->grid_side_) +
                " across, which its " + std::to_string(model->merge_) +
                "-wide merge does not divide");
  }
  model->merged_side_ = model->grid_side_ / model->merge_;
  model->n_patches_ = model->grid_side_ * model->grid_side_;
  model->n_tokens_ = model->merged_side_ * model->merged_side_;
  model->merged_width_ = model->merge_ * model->merge_ * model->n_embd_;
  /* The rotary layout llama.cpp builds a `qwen3vl` vision graph with: `head_dim /
   * 4` per section over a `head_dim / 2`-wide rotation.  Four sections are
   * declared and only the first two are ever reached, because the rotation is
   * half the head wide while the sections are quarter-head sized -- a 64-wide
   * head has 32 rotating pairs, and the section boundaries at 16 and 32 fall
   * before the third section's start at 48.  There is nothing to correct here; it
   * is what the reference graph asks for, and `build_rope_tables` follows its
   * loop. */
  if (model->head_dim_ % 4 != 0) {
    throw Error("the tower's head width " + std::to_string(model->head_dim_) +
                " does not divide into the four rotary sections this graph is built for");
  }
  read_channel_stats(checkpoint, vision_key("image_mean"), model->image_mean_);
  read_channel_stats(checkpoint, vision_key("image_std"), model->image_std_);

  /* Which blocks hand up a deepstack tap.  The file states it as a per-layer flag
   * and the tensors state it again by existing, so both are read: the flag
   * decides, and binding the tensors is the check that catches a flag which
   * disagrees with the file it came from. */
  const GgufArray *taps = require_array(checkpoint, vision_key("is_deepstack_layers"));
  if (static_cast<int64_t>(taps->bools.size()) != model->n_layer_) {
    throw Error("metadata key '" + vision_key("is_deepstack_layers") + "' has " +
                std::to_string(taps->bools.size()) + " entries, not " +
                std::to_string(model->n_layer_));
  }
  for (int64_t layer = 0; layer < model->n_layer_; ++layer) {
    if (taps->bools[static_cast<std::size_t>(layer)] != 0) {
      model->deepstack_layers_.push_back(layer);
    }
  }

  const int64_t kernel = model->patch_size_;
  model->patch_w0_ =
      bind_conv(checkpoint, backend, model->owned_, "v.patch_embd.weight", model->n_embd_, 3, kernel);
  model->patch_w1_ = bind_conv(checkpoint, backend, model->owned_, "v.patch_embd.weight.1",
                               model->n_embd_, 3, kernel);
  model->patch_bias_ = bind_dense(checkpoint, backend, model->owned_, "v.patch_embd.bias");
  model->position_table_ =
      bind_dense(checkpoint, backend, model->owned_, "v.position_embd.weight");
  model->post_ln_w_ = bind_dense(checkpoint, backend, model->owned_, "v.post_ln.weight");
  model->post_ln_b_ = bind_dense(checkpoint, backend, model->owned_, "v.post_ln.bias");
  model->mm0_ = bind_linear(checkpoint, backend, model->owned_, "mm.0", model->merged_width_,
                            model->merged_width_);
  model->mm2_ = bind_linear(checkpoint, backend, model->owned_, "mm.2", model->projection_dim_,
                            model->merged_width_);

  model->blocks_.resize(static_cast<std::size_t>(model->n_layer_));
  for (int64_t layer = 0; layer < model->n_layer_; ++layer) {
    VisionBlock &block = model->blocks_[static_cast<std::size_t>(layer)];
    const std::string base = "v.blk." + std::to_string(layer);
    block.ln1_w = bind_dense(checkpoint, backend, model->owned_, base + ".ln1.weight");
    block.ln1_b = bind_dense(checkpoint, backend, model->owned_, base + ".ln1.bias");
    block.ln2_w = bind_dense(checkpoint, backend, model->owned_, base + ".ln2.weight");
    block.ln2_b = bind_dense(checkpoint, backend, model->owned_, base + ".ln2.bias");
    block.qkv = bind_linear(checkpoint, backend, model->owned_, base + ".attn_qkv",
                            3 * model->n_embd_, model->n_embd_);
    block.attn_out = bind_linear(checkpoint, backend, model->owned_, base + ".attn_out",
                                 model->n_embd_, model->n_embd_);
    block.ffn_up =
        bind_linear(checkpoint, backend, model->owned_, base + ".ffn_up", model->n_ff_, model->n_embd_);
    block.ffn_down = bind_linear(checkpoint, backend, model->owned_, base + ".ffn_down",
                                 model->n_embd_, model->n_ff_);
  }
  for (int64_t layer : model->deepstack_layers_) {
    VisionBlock &block = model->blocks_[static_cast<std::size_t>(layer)];
    const std::string base = "v.deepstack." + std::to_string(layer);
    block.has_deepstack = true;
    block.ds_norm_w = bind_dense(checkpoint, backend, model->owned_, base + ".norm.weight");
    block.ds_norm_b = bind_dense(checkpoint, backend, model->owned_, base + ".norm.bias");
    /* The tap normalizes the *merged* row: llama.cpp puts the norm after the
     * reshape, which is what separates a deepstack merger from the final one --
     * whose norm is `v.post_ln`, on the 1024-wide patch. */
    block.ds_fc1 = bind_linear(checkpoint, backend, model->owned_, base + ".fc1",
                               model->merged_width_, model->merged_width_);
    block.ds_fc2 = bind_linear(checkpoint, backend, model->owned_, base + ".fc2",
                               model->projection_dim_, model->merged_width_);
  }

  model->allocate_activations();
  model->build_merge_index();
  model->build_rope_tables();
  return model;
}

void Qwen3VlVision::allocate_activations() {
  const int64_t patches = n_patches_;
  const int64_t tokens = n_tokens_;
  pixels_ = backend_->allocate(3 * image_size_ * image_size_ * 4);
  conv_ = backend_->allocate(patches * n_embd_ * 4);
  conv_tmp_ = backend_->allocate(patches * n_embd_ * 4);
  tokens_ = backend_->allocate(patches * n_embd_ * 4);
  position_rows_ = backend_->allocate(patches * n_embd_ * 4);
  x_ = backend_->allocate(patches * n_embd_ * 4);
  q_ = backend_->allocate(patches * n_embd_ * 4);
  k_ = backend_->allocate(patches * n_embd_ * 4);
  v_ = backend_->allocate(patches * n_embd_ * 4);
  attn_out_ = backend_->allocate(patches * n_embd_ * 4);
  norm_ = backend_->allocate(patches * n_embd_ * 4);
  up_ = backend_->allocate(patches * n_ff_ * 4);
  merged_norm_ = backend_->allocate(tokens * merged_width_ * 4);
  merged_hidden_ = backend_->allocate(tokens * merged_width_ * 4);
  base_out_ = backend_->allocate(tokens * projection_dim_ * 4);
  /* One allocation per tap, and not `assign(size, buffer)` -- that form *copies*
   * the handle, so three planes would be one buffer written three times and freed
   * three times. */
  deepstack_out_.resize(deepstack_layers_.size());
  for (kernel::DeviceBuffer &buffer : deepstack_out_) {
    buffer = backend_->allocate(tokens * projection_dim_ * 4);
  }
  /* The contract's bound for this graph's one attention call: `q_offset + q_len -
   * first_key` is two spans, because the query chunk's own offset is the end of
   * the sequence.  The non-causal arm never reads the scratch -- it recomputes
   * the row per pass rather than holding it -- but sizing it to the contract is
   * what keeps that a kernel detail instead of something the graph knows. */
  scores_ = backend_->allocate(backend_->attention_scratch(patches, n_head_, 2 * patches - 1));
  cos_ = backend_->allocate(patches * (head_dim_ / 2) * 4);
  sin_ = backend_->allocate(patches * (head_dim_ / 2) * 4);
  merge_index_ = backend_->allocate(patches * 4);

  for (kernel::DeviceBuffer buffer :
       {pixels_, conv_, conv_tmp_, tokens_, position_rows_, x_, q_, k_, v_, attn_out_, norm_, up_,
        merged_norm_, merged_hidden_, base_out_, scores_, cos_, sin_, merge_index_}) {
    owned_.push_back(buffer);
  }
  for (kernel::DeviceBuffer buffer : deepstack_out_) {
    owned_.push_back(buffer);
  }
}

void Qwen3VlVision::build_merge_index() {
  std::vector<int32_t> index(static_cast<std::size_t>(n_patches_));
  for (int64_t token = 0; token < n_patches_; ++token) {
    int64_t y = 0;
    int64_t x = 0;
    raster_patch(token, merge_, merged_side_, &y, &x);
    index[static_cast<std::size_t>(token)] = static_cast<int32_t>(y * grid_side_ + x);
  }
  backend_->copy_to_device(merge_index_, index.data(), n_patches_ * 4);
}

void Qwen3VlVision::build_rope_tables() {
  /* The tower's rotary, as llama.cpp's `GGML_ROPE_TYPE_VISION` defines it.  Three
   * things about it are worth stating, because none of them is the generic RoPE
   * this engine already had:
   *
   *   - **The pairing is split-half over the whole head.**  Pair `i` rotates
   *     `x[i]` against `x[i + head_dim/2]`, which is `rope_neox` with
   *     `d = head_dim` exactly.  So no new op was needed: what differs is not the
   *     rotation but where the angles come from.
   *   - **The angles are per token rather than per position.**  Each patch has a
   *     row and a column and the two halves of the rotation use one each, so a
   *     table built once at load is the whole of it -- `rope_neox` reads row `t`
   *     for token `t` when `start_pos` is 0.
   *   - **Each half restarts its frequency ramp.**  llama.cpp's `indep_sects`
   *     resets the running angle at every section boundary, so pair `p` in the
   *     first half is `row * scale^p` and pair `p` in the second is
   *     `column * scale^(p - section)`.  One ramp across the head would leave the
   *     second half's angles where the first half's stopped, and that is a
   *     different model.
   *
   * `scale` is `freq_base^(-2 / (head_dim / 2))`, again following llama.cpp: the
   * exponent is over the rotating width, which is half the head.  The angle is
   * accumulated by repeated multiplication rather than recomputed as a power, so
   * this table carries the same rounding the reference's does. */
  const int64_t half = head_dim_ / 2;
  const int64_t section = head_dim_ / 4;
  const float scale = std::pow(kVisionRopeFreqBase, -2.0F / static_cast<float>(half));
  std::vector<float> cos_table(static_cast<std::size_t>(n_patches_ * half));
  std::vector<float> sin_table(static_cast<std::size_t>(n_patches_ * half));
  for (int64_t token = 0; token < n_patches_; ++token) {
    int64_t y = 0;
    int64_t x = 0;
    raster_patch(token, merge_, merged_side_, &y, &x);
    float theta_row = static_cast<float>(y);
    float theta_column = static_cast<float>(x);
    for (int64_t pair = 0; pair < half; ++pair) {
      if (pair == section) {
        theta_column = static_cast<float>(x);
      }
      const float theta = pair < section ? theta_row : theta_column;
      cos_table[static_cast<std::size_t>(token * half + pair)] = std::cos(theta);
      sin_table[static_cast<std::size_t>(token * half + pair)] = std::sin(theta);
      theta_row *= scale;
      theta_column *= scale;
    }
  }
  backend_->copy_to_device(cos_, cos_table.data(), n_patches_ * half * 4);
  backend_->copy_to_device(sin_, sin_table.data(), n_patches_ * half * 4);
}

void Qwen3VlVision::encode(const uint8_t *rgb, int64_t height, int64_t width, float *out) {
  if (height != image_size_ || width != image_size_) {
    throw Error("this tower's position table is stored for " + std::to_string(image_size_) + "x" +
                std::to_string(image_size_) +
                " images and the interpolation that would let it take " + std::to_string(width) +
                "x" + std::to_string(height) + " is not implemented; the caller resizes");
  }
  const int64_t side = image_size_;

  /* The pixels, normalized the way llama.cpp's preprocessing does it after its
   * resize -- `(value / 255 - mean[c]) / std[c]` -- and laid out (channel, row,
   * column) with the column fastest, which is what `conv2d` reads. */
  std::vector<float> host_pixels(static_cast<std::size_t>(3 * side * side));
  for (int64_t channel = 0; channel < 3; ++channel) {
    for (int64_t y = 0; y < side; ++y) {
      for (int64_t x = 0; x < side; ++x) {
        const float value =
            static_cast<float>(rgb[static_cast<std::size_t>((y * side + x) * 3 + channel)]) / 255.0F;
        host_pixels[static_cast<std::size_t>((channel * side + y) * side + x)] =
            (value - image_mean_[channel]) / image_std_[channel];
      }
    }
  }
  backend_->copy_to_device(pixels_, host_pixels.data(), 3 * side * side * 4);

  /* The patch embedding: two temporal taps of a 3-D convolution, which for the
   * still image the tower takes are two convolutions of the same picture summed,
   * plus the one bias. */
  const int64_t kernel = patch_size_;
  backend_->conv2d(pixels_, patch_w0_.weight, patch_bias_, conv_, 3, side, side, n_embd_, kernel,
                   kernel, 0);
  backend_->conv2d(pixels_, patch_w1_.weight, kernel::DeviceBuffer{}, conv_tmp_, 3, side, side,
                   n_embd_, kernel, kernel, 0);
  backend_->add(conv_, conv_tmp_, conv_, n_patches_ * n_embd_);

  /* Into merge order -- one permutation applied to the patches and to the
   * position rows both, because the whole point of the order is that the four
   * patches of a block end up adjacent.  A gather rather than a permute: the
   * `embedding` op already is a row gather, and the index table *is* the
   * permutation written down. */
  backend_->embedding(merge_index_, n_patches_, conv_, n_patches_, n_embd_, tokens_);
  backend_->embedding(merge_index_, n_patches_, position_table_, n_patches_, n_embd_,
                      position_rows_);
  backend_->add(tokens_, position_rows_, x_, n_patches_ * n_embd_);

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  std::size_t next_tap = 0;
  for (int64_t layer = 0; layer < n_layer_; ++layer) {
    const VisionBlock &block = blocks_[static_cast<std::size_t>(layer)];
    backend_->layer_norm(x_, block.ln1_w, block.ln1_b, norm_, n_patches_, n_embd_, eps_);

    /* The fused qkv holds three panels of `n_embd` rows, and a panel is a row
     * range -- which a `DeviceBuffer` names by offset.  Three calls rather than
     * one because the three results feed different ops and `gemm` has no way to
     * say "every third panel"; the arithmetic is the same either way. */
    const int64_t panel_bytes = n_embd_ * block.qkv.cols * 4;
    for (int64_t panel = 0; panel < 3; ++panel) {
      const kernel::DeviceBuffer weight =
          slice_of(block.qkv.weight, panel * panel_bytes, panel_bytes);
      const kernel::DeviceBuffer bias =
          slice_of(block.qkv.bias, panel * n_embd_ * 4, n_embd_ * 4);
      kernel::DeviceBuffer destination = panel == 0 ? q_ : (panel == 1 ? k_ : v_);
      backend_->gemm(norm_, weight, bias, destination, n_patches_, n_embd_, block.qkv.cols, false);
    }

    backend_->rope_neox(q_, n_patches_, n_head_, head_dim_, /*start_pos=*/0, cos_, sin_);
    backend_->rope_neox(k_, n_patches_, n_head_, head_dim_, /*start_pos=*/0, cos_, sin_);
    /* Non-causal: the chunk's offset is the last patch, so every query sees the
     * whole span rather than a prefix of it. */
    backend_->attention(q_, n_patches_, n_head_, k_, v_, n_head_, head_dim_, /*first_key=*/0,
                        /*q_offset=*/n_patches_ - 1, /*causal=*/false, scale, attn_out_, scores_,
                        kernel::KVDtype::kF32);
    backend_->gemm(attn_out_, block.attn_out.weight, block.attn_out.bias, x_, n_patches_, n_embd_,
                   n_embd_, /*accumulate=*/true);

    backend_->layer_norm(x_, block.ln2_w, block.ln2_b, norm_, n_patches_, n_embd_, eps_);
    backend_->gemm(norm_, block.ffn_up.weight, block.ffn_up.bias, up_, n_patches_, n_ff_, n_embd_,
                   false);
    backend_->gelu(up_, up_, n_patches_ * n_ff_);
    backend_->gemm(up_, block.ffn_down.weight, block.ffn_down.bias, x_, n_patches_, n_embd_, n_ff_,
                   /*accumulate=*/true);

    if (block.has_deepstack) {
      /* The tap reads this block's own output.  Four adjacent patch rows are one
       * merged row, which is the same memory under a wider row, and the norm here
       * is over *that* width -- the post-shuffle form, unlike the merger's. */
      const std::size_t tap = next_tap++;
      const kernel::DeviceBuffer merged{x_.handle, n_tokens_ * merged_width_ * 4};
      backend_->layer_norm(merged, block.ds_norm_w, block.ds_norm_b, merged_norm_, n_tokens_,
                           merged_width_, eps_);
      backend_->gemm(merged_norm_, block.ds_fc1.weight, block.ds_fc1.bias, merged_hidden_,
                     n_tokens_, merged_width_, merged_width_, false);
      backend_->gelu(merged_hidden_, merged_hidden_, n_tokens_ * merged_width_);
      backend_->gemm(merged_hidden_, block.ds_fc2.weight, block.ds_fc2.bias, deepstack_out_[tap],
                     n_tokens_, projection_dim_, merged_width_, false);
    }
  }

  backend_->layer_norm(x_, post_ln_w_, post_ln_b_, norm_, n_patches_, n_embd_, eps_);
  {
    const kernel::DeviceBuffer merged{norm_.handle, n_tokens_ * merged_width_ * 4};
    backend_->gemm(merged, mm0_.weight, mm0_.bias, merged_norm_, n_tokens_, merged_width_,
                   merged_width_, false);
    backend_->gelu(merged_norm_, merged_norm_, n_tokens_ * merged_width_);
    backend_->gemm(merged_norm_, mm2_.weight, mm2_.bias, base_out_, n_tokens_, projection_dim_,
                   merged_width_, false);
  }

  const int64_t plane = n_tokens_ * projection_dim_;
  backend_->copy_to_host(out, base_out_, plane * 4);
  for (std::size_t tap = 0; tap < deepstack_out_.size(); ++tap) {
    backend_->copy_to_host(out + (static_cast<int64_t>(tap) + 1) * plane, deepstack_out_[tap],
                           plane * 4);
  }
}

}  // namespace pocketllm
