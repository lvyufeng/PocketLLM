#include "model/qwen3vl_vision.h"

#include "model/image_preprocess.h"
#include "model/tower_loader.h"
#include "quant/q8_0.h"

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
/* `q8_0`, which the *audio* tower's file stores its attention and FFN weights in.
 * The vision tower's own file is dense, so this branch is here for the second
 * tower; `src/quant/q8_0.h` is the format. */
constexpr int kTypeQ8_0 = 8;

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
const GgufArray *require_array(const GgufReader &checkpoint, const std::string &name) {
  const GgufValue *value = checkpoint.find(name);
  const GgufArray *array = value == nullptr ? nullptr : std::get_if<GgufArray>(value);
  if (array == nullptr) {
    throw Error("the mmproj has no metadata array '" + name + "'");
  }
  return array;
}
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

/* The loader is shared with the audio tower (`model/tower_loader.h`): both read the
 * same kind of file and the part where being nearly right is worst is the same part.
 * What this file keeps is what is specific to a *vision* tower -- `VisionLinear`'s two
 * shapes, and the convolution binder below, whose tensor is not (rows, cols). */

/* The shared binder returns `tower::Linear`; a vision block also needs to remember
 * the shape it was bound at, because the qkv projection is read back in three panels
 * and `qkv.cols` is how wide each one is.  The check inside `tower::bind_linear` is
 * the one that matters -- this only records what it already verified. */
VisionLinear bind_vision_linear(const GgufReader &checkpoint, kernel::Backend &backend,
                                std::vector<kernel::DeviceBuffer> &owned, const std::string &base,
                                int64_t rows, int64_t cols) {
  const tower::Linear shared = tower::bind_linear(checkpoint, backend, owned, base, rows, cols);
  VisionLinear linear;
  linear.weight = shared.weight;
  linear.bias = shared.bias;
  linear.rows = rows;
  linear.cols = cols;
  return linear;
}

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
  linear.weight = tower::bind_tensor(checkpoint, backend, owned, name);
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

  model->n_layer_ = tower::require_int(checkpoint, vision_key("block_count"));
  model->n_embd_ = tower::require_int(checkpoint, vision_key("embedding_length"));
  model->n_head_ = tower::require_int(checkpoint, vision_key("attention.head_count"));
  model->n_ff_ = tower::require_int(checkpoint, vision_key("feed_forward_length"));
  model->patch_size_ = tower::require_int(checkpoint, vision_key("patch_size"));
  model->image_size_ = tower::require_int(checkpoint, vision_key("image_size"));
  model->projection_dim_ = tower::require_int(checkpoint, vision_key("projection_dim"));
  model->merge_ = tower::require_int(checkpoint, vision_key("spatial_merge_size"));
  /* The pixel limits the preprocessing uses to choose a size, read here because
   * `align` needs `merge_`.  A file that does not state them gets the reference's own
   * defaults for this projector family -- llama.cpp's `set_limit_image_tokens(8,
   * 4096)` for Qwen2/3-VL, in tokens of `patch * merge` pixels each -- so a
   * checkpoint that omits the keys still preprocesses the way the reference does. */
  const int64_t align = model->patch_size_ * model->merge_;
  model->image_min_pixels_ = checkpoint.get_int(vision_key("image_min_pixels"), 8 * align * align);
  model->image_max_pixels_ =
      checkpoint.get_int(vision_key("image_max_pixels"), 4096 * align * align);
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
  /* The largest grid the pixel limit admits, which is what the activations are sized
   * for.  `resize_target` never exceeds `image_max_pixels`, so the patch count is
   * bounded by `max_pixels / patch^2`; it is rounded up to a whole number of merge
   * blocks because every allocation below is a multiple of one. */
  {
    const int64_t merge_block = model->merge_ * model->merge_;
    int64_t cap = model->image_max_pixels_ / (model->patch_size_ * model->patch_size_);
    cap = ((cap + merge_block - 1) / merge_block) * merge_block;
    model->max_patches_ = std::max<int64_t>(model->n_patches_, cap + merge_block);
    model->max_side_ = static_cast<int64_t>(std::sqrt(static_cast<double>(model->max_patches_))) *
                       model->patch_size_;
  }
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
  model->patch_bias_ =
      tower::bind_tensor(checkpoint, backend, model->owned_, "v.patch_embd.bias");
  model->position_table_ =
      tower::bind_tensor(checkpoint, backend, model->owned_, "v.position_embd.weight");
  /* The same table on the host, for the interpolation an image of another size
   * needs: `bind_dense` widens f32/f16 into a device buffer and drops the host copy,
   * and the interpolation happens in the coordinates of the target grid, so it has
   * to be done over the numbers rather than in the graph. */
  {
    const std::string name = "v.position_embd.weight";
    uint64_t nbytes = 0;
    const uint8_t *bytes = checkpoint.tensor_data(name, &nbytes);
    const GgufTensorInfo *info = checkpoint.tensor(name);
    int64_t elements = 1;
    for (uint64_t dim : info->dimensions) {
      elements *= static_cast<int64_t>(dim);
    }
    model->position_table_host_.resize(static_cast<std::size_t>(elements));
    if (info->type_id == kTypeF32) {
      std::memcpy(model->position_table_host_.data(), bytes, static_cast<std::size_t>(nbytes));
    } else if (info->type_id == kTypeF16) {
      for (int64_t i = 0; i < elements; ++i) {
        model->position_table_host_[static_cast<std::size_t>(i)] =
            half_to_float(load_u16(bytes + 2 * i));
      }
    } else {
      throw Error(std::string("the tower's position table is ") +
                  ggml_type_of(info->type_id).name +
                  ", and this loader reads f32 and f16 only");
    }
  }
  model->post_ln_w_ = tower::bind_tensor(checkpoint, backend, model->owned_, "v.post_ln.weight");
  model->post_ln_b_ = tower::bind_tensor(checkpoint, backend, model->owned_, "v.post_ln.bias");
  model->mm0_ = bind_vision_linear(checkpoint, backend, model->owned_, "mm.0", model->merged_width_,
                            model->merged_width_);
  model->mm2_ = bind_vision_linear(checkpoint, backend, model->owned_, "mm.2", model->projection_dim_,
                            model->merged_width_);

  model->blocks_.resize(static_cast<std::size_t>(model->n_layer_));
  for (int64_t layer = 0; layer < model->n_layer_; ++layer) {
    VisionBlock &block = model->blocks_[static_cast<std::size_t>(layer)];
    const std::string base = "v.blk." + std::to_string(layer);
    block.ln1_w = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln1.weight");
    block.ln1_b = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln1.bias");
    block.ln2_w = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln2.weight");
    block.ln2_b = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".ln2.bias");
    block.qkv = bind_vision_linear(checkpoint, backend, model->owned_, base + ".attn_qkv",
                            3 * model->n_embd_, model->n_embd_);
    block.attn_out = bind_vision_linear(checkpoint, backend, model->owned_, base + ".attn_out",
                                 model->n_embd_, model->n_embd_);
    block.ffn_up =
        bind_vision_linear(checkpoint, backend, model->owned_, base + ".ffn_up", model->n_ff_, model->n_embd_);
    block.ffn_down = bind_vision_linear(checkpoint, backend, model->owned_, base + ".ffn_down",
                                 model->n_embd_, model->n_ff_);
  }
  for (int64_t layer : model->deepstack_layers_) {
    VisionBlock &block = model->blocks_[static_cast<std::size_t>(layer)];
    const std::string base = "v.deepstack." + std::to_string(layer);
    block.has_deepstack = true;
    block.ds_norm_w = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".norm.weight");
    block.ds_norm_b = tower::bind_tensor(checkpoint, backend, model->owned_, base + ".norm.bias");
    /* The tap normalizes the *merged* row: llama.cpp puts the norm after the
     * reshape, which is what separates a deepstack merger from the final one --
     * whose norm is `v.post_ln`, on the 1024-wide patch. */
    block.ds_fc1 = bind_vision_linear(checkpoint, backend, model->owned_, base + ".fc1",
                               model->merged_width_, model->merged_width_);
    block.ds_fc2 = bind_vision_linear(checkpoint, backend, model->owned_, base + ".fc2",
                               model->projection_dim_, model->merged_width_);
  }

  model->allocate_activations();
  /* Built once here for the checkpoint's own image_size, which is the shape most
   * callers use; `encode` rebuilds them if a different shape arrives. */
  model->build_merge_index(model->grid_side_, model->grid_side_);
  model->build_rope_tables(model->grid_side_, model->grid_side_);
  /* The identity case, uploaded so the gather has a table before any image arrives. */
  model->backend_->copy_to_device(model->position_table_resampled_,
                                  model->position_table_host_.data(),
                                  model->n_patches_ * model->n_embd_ * 4);
  model->active_grid_w_ = model->grid_side_;
  model->active_grid_h_ = model->grid_side_;
  return model;
}

void Qwen3VlVision::allocate_activations() {
  /* Sized for the **largest** image the checkpoint's own limit allows, not for
   * `image_size()`: `encode` resizes whatever it is given onto the grid the position
   * table covers for that size, and the two differ.  The attention scratch is scaled
   * with it because `attention_scratch` is a function of the query length. */
  const int64_t max_patches = max_patches_;
  const int64_t max_tokens = max_patches / (merge_ * merge_);
  pixels_ = backend_->allocate(3 * max_patches * patch_size_ * patch_size_ * 4);
  conv_ = backend_->allocate(max_patches * n_embd_ * 4);
  conv_tmp_ = backend_->allocate(max_patches * n_embd_ * 4);
  tokens_ = backend_->allocate(max_patches * n_embd_ * 4);
  position_rows_ = backend_->allocate(max_patches * n_embd_ * 4);
  x_ = backend_->allocate(max_patches * n_embd_ * 4);
  q_ = backend_->allocate(max_patches * n_embd_ * 4);
  k_ = backend_->allocate(max_patches * n_embd_ * 4);
  v_ = backend_->allocate(max_patches * n_embd_ * 4);
  attn_out_ = backend_->allocate(max_patches * n_embd_ * 4);
  norm_ = backend_->allocate(max_patches * n_embd_ * 4);
  up_ = backend_->allocate(max_patches * n_ff_ * 4);
  merged_norm_ = backend_->allocate(max_tokens * merged_width_ * 4);
  merged_hidden_ = backend_->allocate(max_tokens * merged_width_ * 4);
  base_out_ = backend_->allocate(max_tokens * projection_dim_ * 4);
  /* One allocation per tap, and not `assign(size, buffer)` -- that form *copies*
   * the handle, so three planes would be one buffer written three times and freed
   * three times. */
  deepstack_out_.resize(deepstack_layers_.size());
  for (kernel::DeviceBuffer &buffer : deepstack_out_) {
    buffer = backend_->allocate(max_tokens * projection_dim_ * 4);
  }
  /* The contract's bound for this graph's one attention call: `q_offset + q_len -
   * first_key` is two spans, because the query chunk's own offset is the end of
   * the sequence.  The non-causal arm never reads the scratch -- it recomputes
   * the row per pass rather than holding it -- but sizing it to the contract is
   * what keeps that a kernel detail instead of something the graph knows. */
  /* Sized for the checkpoint's own grid, and grown on demand in `encode`.  It is the one
   * allocation here that is quadratic in the patch count: `attention_scratch` is
   * `tasks * rows * (2 * patches - 1)`, so 576 patches want 42 MB and the 16384 that
   * `image_max_pixels` admits want 34 GB.  Allocating the maximum up front "worked" on the
   * CPU only because `new float[]` is lazy; on CUDA it is an immediate `cudaMalloc` and it
   * failed.  A tower whose declared limit is larger than its scratch buffer is a limit that
   * has to be enforced at `encode`, not reserved at `load`. */
  scores_capacity_ = n_patches_;
  scores_ = backend_->allocate(backend_->attention_scratch(scores_capacity_, n_head_,
                                                          2 * scores_capacity_ - 1));
  cos_ = backend_->allocate(max_patches * (head_dim_ / 2) * 4);
  sin_ = backend_->allocate(max_patches * (head_dim_ / 2) * 4);
  merge_index_ = backend_->allocate(max_patches * 4);
  /* The interpolated position table, one row per patch of whatever grid is active. */
  position_table_resampled_ = backend_->allocate(max_patches * n_embd_ * 4);

  for (kernel::DeviceBuffer buffer :
       {pixels_, conv_, conv_tmp_, tokens_, position_rows_, x_, q_, k_, v_, attn_out_, norm_, up_,
        merged_norm_, merged_hidden_, base_out_, scores_, cos_, sin_, merge_index_}) {
    owned_.push_back(buffer);
  }
  for (kernel::DeviceBuffer buffer : deepstack_out_) {
    owned_.push_back(buffer);
  }
}

void Qwen3VlVision::build_merge_index(int64_t grid_w, int64_t grid_h) {
  /* One index per token, into the *raster* patch grid: `raster_patch` is the inverse
   * of this map, and the two are the only place the merge order is written down. */
  (void) grid_h;
  const int64_t merged = grid_w / merge_;
  const int64_t patches = grid_w * grid_h;
  std::vector<int32_t> index(static_cast<std::size_t>(patches));
  for (int64_t token = 0; token < patches; ++token) {
    int64_t y = 0;
    int64_t x = 0;
    raster_patch(token, merge_, merged, &y, &x);
    index[static_cast<std::size_t>(token)] = static_cast<int32_t>(y * grid_w + x);
  }
  backend_->copy_to_device(merge_index_, index.data(), patches * 4);
}

void Qwen3VlVision::build_rope_tables(int64_t grid_w, int64_t grid_h) {
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
  const int64_t merged = grid_w / merge_;
  const int64_t patches = grid_w * grid_h;
  std::vector<float> cos_table(static_cast<std::size_t>(patches * half));
  std::vector<float> sin_table(static_cast<std::size_t>(patches * half));
  for (int64_t token = 0; token < patches; ++token) {
    int64_t y = 0;
    int64_t x = 0;
    raster_patch(token, merge_, merged, &y, &x);
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
  backend_->copy_to_device(cos_, cos_table.data(), patches * half * 4);
  backend_->copy_to_device(sin_, sin_table.data(), patches * half * 4);
}

void Qwen3VlVision::prices_for(int64_t width, int64_t height, int64_t *target_width,
                               int64_t *target_height, int64_t *tokens) const {
  int64_t chosen_w = 0;
  int64_t chosen_h = 0;
  resize_target(width, height, patch_size_ * merge_, image_min_pixels_, image_max_pixels_,
                &chosen_w, &chosen_h);
  *target_width = chosen_w;
  *target_height = chosen_h;
  *tokens = (chosen_w / patch_size_ / merge_) * (chosen_h / patch_size_ / merge_);
}

void Qwen3VlVision::encode(const uint8_t *rgb, int64_t height, int64_t width, float *out) {
  /* The image, resized onto a grid this tower can consume and then normalized.  The
   * resize is the reference's own -- `preprocess_image` transcribes both the size
   * rule and the bilinear kernel -- because a different target size is a different
   * number of tokens and so a different prompt rather than an approximate one. */
  const PreprocessedImage image =
      preprocess_image(rgb, width, height, patch_size_ * merge_, image_min_pixels_,
                       image_max_pixels_);
  const int64_t side = image.width;
  const int64_t grid_w = side / patch_size_;
  const int64_t grid_h = image.height / patch_size_;
  const int64_t patches = grid_w * grid_h;
  const int64_t merged_w = grid_w / merge_;
  const int64_t merged_h = grid_h / merge_;
  const int64_t tokens = merged_w * merged_h;
  if (patches > max_patches_ || patches % (merge_ * merge_) != 0) {
    throw Error("the preprocessing chose a " + std::to_string(grid_w) + "x" +
                std::to_string(grid_h) +
                " patch grid, which this tower's activations are not sized for");
  }
  /* The permutation and the rotary angles are a function of the grid, so they are
   * rebuilt whenever the grid changes -- a session that encodes one size pays for it
   * once, and one that alternates pays per change.  The alternative is a table per
   * size, which is the same work with a cache to get wrong. */
  if (grid_w != active_grid_w_ || grid_h != active_grid_h_) {
    build_merge_index(grid_w, grid_h);
    build_rope_tables(grid_w, grid_h);
    /* `scores_` is the one buffer that has to grow with the grid rather than being
     * reserved for the largest one it could ever be. */
    if (patches > scores_capacity_) {
      backend_->release(scores_);
      scores_capacity_ = patches;
      scores_ = backend_->allocate(
          backend_->attention_scratch(scores_capacity_, n_head_, 2 * scores_capacity_ - 1));
    }
    /* The position table is a *weight*, one row per patch of the grid the checkpoint
     * was trained on, so an image of another size needs it resampled onto that size's
     * grid -- which is what the reference does rather than refusing the image.  The
     * grid equal to the checkpoint's own needs no work beyond the copy that is
     * already there. */
    const std::vector<float> resampled =
        interpolate_position_table(position_table_host_.data(), grid_side_, grid_w, grid_h,
                                   n_embd_);
    backend_->copy_to_device(position_table_resampled_, resampled.data(),
                             patches * n_embd_ * 4);
    active_grid_w_ = grid_w;
    active_grid_h_ = grid_h;
  }

  /* The pixels, normalized the way llama.cpp's preprocessing does it after its
   * resize -- `(value / 255 - mean[c]) / std[c]` -- and laid out (channel, row,
   * column) with the column fastest, which is what `conv2d` reads. */
  const int64_t image_h = image.height;
  std::vector<float> host_pixels(static_cast<std::size_t>(3 * side * image_h));
  for (int64_t channel = 0; channel < 3; ++channel) {
    for (int64_t y = 0; y < image_h; ++y) {
      for (int64_t x = 0; x < side; ++x) {
        const float value = static_cast<float>(
                                image.rgb[static_cast<std::size_t>((y * side + x) * 3 + channel)]) /
                            255.0F;
        host_pixels[static_cast<std::size_t>((channel * image_h + y) * side + x)] =
            (value - image_mean_[channel]) / image_std_[channel];
      }
    }
  }
  backend_->copy_to_device(pixels_, host_pixels.data(), 3 * side * image_h * 4);

  /* The patch embedding: two temporal taps of a 3-D convolution, which for the
   * still image the tower takes are two convolutions of the same picture summed,
   * plus the one bias. */
  const int64_t kernel = patch_size_;
  backend_->conv2d(pixels_, patch_w0_.weight, patch_bias_, conv_, /*batch=*/1, 3, image_h,
                   side, n_embd_,
                   kernel, kernel, 0);
  backend_->conv2d(pixels_, patch_w1_.weight, kernel::DeviceBuffer{}, conv_tmp_, /*batch=*/1, 3,
                   image_h, side, n_embd_, kernel, kernel, 0);
  backend_->add(conv_, conv_tmp_, conv_, patches * n_embd_);

  /* Into merge order -- one permutation applied to the patches and to the
   * position rows both, because the whole point of the order is that the four
   * patches of a block end up adjacent.  A gather rather than a permute: the
   * `embedding` op already is a row gather, and the index table *is* the
   * permutation written down. */
  backend_->embedding(merge_index_, patches, conv_, patches, n_embd_, tokens_);
  backend_->embedding(merge_index_, patches, position_table_resampled_, patches, n_embd_,
                      position_rows_);
  backend_->add(tokens_, position_rows_, x_, patches * n_embd_);

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));
  std::size_t next_tap = 0;
  for (int64_t layer = 0; layer < n_layer_; ++layer) {
    const VisionBlock &block = blocks_[static_cast<std::size_t>(layer)];
    backend_->layer_norm(x_, block.ln1_w, block.ln1_b, norm_, patches, n_embd_, eps_);

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
      backend_->gemm(norm_, weight, bias, destination, patches, n_embd_, block.qkv.cols, false);
    }

    backend_->rope_neox(q_, patches, n_head_, head_dim_, /*start_pos=*/0, cos_, sin_);
    backend_->rope_neox(k_, patches, n_head_, head_dim_, /*start_pos=*/0, cos_, sin_);
    /* Non-causal: the chunk's offset is the last patch, so every query sees the
     * whole span rather than a prefix of it. */
    backend_->attention(q_, patches, n_head_, k_, v_, n_head_, head_dim_, /*first_key=*/0,
                        /*q_offset=*/patches - 1, /*causal=*/false, scale, attn_out_, scores_,
                        kernel::KVDtype::kF32);
    backend_->gemm(attn_out_, block.attn_out.weight, block.attn_out.bias, x_, patches, n_embd_,
                   n_embd_, /*accumulate=*/true);

    backend_->layer_norm(x_, block.ln2_w, block.ln2_b, norm_, patches, n_embd_, eps_);
    backend_->gemm(norm_, block.ffn_up.weight, block.ffn_up.bias, up_, patches, n_ff_, n_embd_,
                   false);
    backend_->gelu(up_, up_, patches * n_ff_);
    backend_->gemm(up_, block.ffn_down.weight, block.ffn_down.bias, x_, patches, n_embd_, n_ff_,
                   /*accumulate=*/true);

    if (block.has_deepstack) {
      /* The tap reads this block's own output.  Four adjacent patch rows are one
       * merged row, which is the same memory under a wider row, and the norm here
       * is over *that* width -- the post-shuffle form, unlike the merger's. */
      const std::size_t tap = next_tap++;
      const kernel::DeviceBuffer merged{x_.handle, tokens * merged_width_ * 4};
      backend_->layer_norm(merged, block.ds_norm_w, block.ds_norm_b, merged_norm_, tokens,
                           merged_width_, eps_);
      backend_->gemm(merged_norm_, block.ds_fc1.weight, block.ds_fc1.bias, merged_hidden_,
                     tokens, merged_width_, merged_width_, false);
      backend_->gelu(merged_hidden_, merged_hidden_, tokens * merged_width_);
      backend_->gemm(merged_hidden_, block.ds_fc2.weight, block.ds_fc2.bias, deepstack_out_[tap],
                     tokens, projection_dim_, merged_width_, false);
    }
  }

  backend_->layer_norm(x_, post_ln_w_, post_ln_b_, norm_, patches, n_embd_, eps_);
  {
    const kernel::DeviceBuffer merged{norm_.handle, tokens * merged_width_ * 4};
    backend_->gemm(merged, mm0_.weight, mm0_.bias, merged_norm_, tokens, merged_width_,
                   merged_width_, false);
    backend_->gelu(merged_norm_, merged_norm_, tokens * merged_width_);
    backend_->gemm(merged_norm_, mm2_.weight, mm2_.bias, base_out_, tokens, projection_dim_,
                   merged_width_, false);
  }

  const int64_t plane = tokens * projection_dim_;
  backend_->copy_to_host(out, base_out_, plane * 4);
  for (std::size_t tap = 0; tap < deepstack_out_.size(); ++tap) {
    backend_->copy_to_host(out + (static_cast<int64_t>(tap) + 1) * plane, deepstack_out_[tap],
                           plane * 4);
  }
}

}  // namespace pocketllm
