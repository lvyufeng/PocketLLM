/* The vision tower of a `qwen3vl` checkpoint, which lives in its own file.
 *
 * A Qwen3-VL checkpoint is two GGUFs.  The text one is the dense graph
 * `qwen3.cpp` runs; this one is the `mmproj`, architecture `clip`, and it holds
 * a vision transformer whose job is to turn an image into a block of feature
 * vectors the text graph is fed at its placeholder positions.
 *
 * The tower is a pre-norm transformer over patches, with three departures from
 * the generic shape and one operator that is not a transformer at all:
 *
 *   - **The patch embedding is a convolution**, and Qwen3-VL's is 3-D: the file
 *     carries two `v.patch_embd.weight` tensors because the temporal kernel is
 *     two frames wide and, for the still image the tower accepts, both frames
 *     are the same picture.  Two 2-D convolutions over the same input, added.
 *   - **The rotary embedding is multi-section and partial.**  Its angles come
 *     from a per-token table rather than from a position, and its two halves use
 *     the patch's row and the patch's column.  `build_rope_tables` says why that
 *     needs no op of its own.
 *   - **Attention is non-causal**: every patch sees every patch, which is the
 *     `causal = false` arm `attention` grew for this graph.
 *   - **The merger is not attention.**  Four adjacent patch tokens are
 *     concatenated into one 4x-wide row and projected back down, and three
 *     layers along the way hand up a second copy of the same treatment on a
 *     different projection -- the checkpoint's `deepstack` taps, which are what
 *     the text graph's `n_deepstack_layers` counts.
 *
 * ## The token order is the merge order, and the file does not say so
 *
 * A plain ViT would emit one token per patch in raster order.  This tower emits
 * them so that the four patches of a 2x2 block are *adjacent*, because that is
 * what makes the merger a reshape instead of a gather.  llama.cpp gets there by
 * permuting the patch tensor; this engine gathers rows with `embedding`, which is
 * the same permutation written as an index table (`merge_index_`), and two other
 * things were measured to follow that same order rather than assumed to: the
 * learned position table, whose own position index turns out to *be* the raster
 * patch index the convolution emits, and the rotary table, which is built here
 * per token rather than read.
 *
 * ## One size, and the interpolation is deliberately absent
 *
 * llama.cpp will interpolate the learned position table onto any patch grid
 * whose sides are multiples of `patch * merge`.  This engine accepts only the
 * grid the table was stored for -- the checkpoint's own `image_size` square --
 * and refuses anything else *by name*.  That is a scope boundary, not a
 * limitation of the graph: the interpolation is separate arithmetic (a bicubic
 * resample) that nothing else in this tree needs yet, and a tower that silently
 * used the uninterpolated table at another size would answer with a different
 * model instead of an error.
 */

#ifndef POCKETLLM_MODEL_QWEN3VL_VISION_H
#define POCKETLLM_MODEL_QWEN3VL_VISION_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "kernel/backend.h"

namespace pocketllm {

/* One projection of the tower: ``out = x @ weight^T + bias``, with `weight`
 * row-major `(rows, cols)` holding the *output* axis first -- GGUF's layout read
 * straight off the mapping, the same convention `qwen3.cpp` uses.  `bias` may be
 * the null handle, and is for nothing here: every projection in this tower has
 * one, which is why the text model's optional-bias plumbing is not needed. */
struct VisionLinear {
  kernel::DeviceBuffer weight;
  kernel::DeviceBuffer bias;
  int64_t rows = 0;
  int64_t cols = 0;
};

/* One transformer block, plus its deepstack tap when it has one. */
struct VisionBlock {
  kernel::DeviceBuffer ln1_w, ln1_b, ln2_w, ln2_b;
  VisionLinear qkv; /* (3 * n_embd, n_embd), the three panels in q, k, v order */
  VisionLinear attn_out;
  VisionLinear ffn_up;
  VisionLinear ffn_down;
  /* The deepstack tap: a norm over the *merged* 4x-wide row, then two
   * projections.  See `encode`. */
  bool has_deepstack = false;
  kernel::DeviceBuffer ds_norm_w, ds_norm_b;
  VisionLinear ds_fc1, ds_fc2;
};

/* Why `checkpoint` is not a vision tower this build can run, or the empty string
 * when it is.  The counterpart of `arch_refusal` for the mmproj half, and separate
 * from it because the two files carry different architectures.
 *
 * A string rather than `arch_refusal`'s `const char *`, because two of these
 * messages name a value read out of the file -- the architecture and the
 * projector -- and a refusal that says *which* projector it found is what makes
 * "this build reads another one" actionable rather than merely true. */
std::string vision_arch_refusal(const GgufReader &checkpoint);

class Qwen3VlVision {
 public:
  /* Read the tower's hparams and bind every tensor.  Throws `Error` naming the
   * key or the tensor that was missing or had the wrong shape, because a tower
   * missing one of its 316 tensors produces a plausible image embedding rather
   * than a failure, and the failure arrives as a wrong answer from the text
   * model several hundred milliseconds later.
   *
   * The reader must outlive the load; the backend must outlive the model. */
  static std::unique_ptr<Qwen3VlVision> load(const GgufReader &checkpoint,
                                             kernel::Backend &backend);

  ~Qwen3VlVision();
  Qwen3VlVision(const Qwen3VlVision &) = delete;
  Qwen3VlVision &operator=(const Qwen3VlVision &) = delete;

  /* The square image side the tower's position table was stored for, from the
   * checkpoint.  An input may be any size: `encode` resizes to a grid this tower can
   * consume, and `tokens_for` says how many tokens that size produces. */
  int64_t image_size() const { return image_size_; }
  /* How many image tokens one image becomes: the merged patch count.  This is the
   * count for `image_size()` square input; an image of another shape produces a
   * different one, which `tokens_for` gives. */
  int64_t n_tokens() const { return n_tokens_; }
  /* How many merged cells an image is across, which is both the grid the rotary
   * positions are indexed by and the number of *positions* the span consumes --
   * llama.cpp's multi-section rotary gives a span `max(nx, ny)` of them for
   * `nx * ny` tokens, so an image is much cheaper in positions than in tokens and
   * the text after it continues from `start_of_span + this`. */
  int64_t merged_side() const { return merged_side_; }
  /* What an arbitrary input shape becomes after preprocessing, and how many tokens
   * and positions that is.  A caller sizing a buffer or a prompt has to know both,
   * and the token count is not a function of the pixel count alone -- see
   * `resize_target`. */
  void prices_for(int64_t width, int64_t height, int64_t *target_width, int64_t *target_height,
                  int64_t *tokens) const;
  /* `n_tokens` for an input of this shape, without resizing it. */
  int64_t tokens_for(int64_t width, int64_t height) const {
    int64_t target_width = 0;
    int64_t target_height = 0;
    int64_t tokens = 0;
    prices_for(width, height, &target_width, &target_height, &tokens);
    return tokens;
  }
  /* The pixel limits the preprocessing takes from the checkpoint. */
  int64_t image_min_pixels() const { return image_min_pixels_; }
  int64_t image_max_pixels() const { return image_max_pixels_; }
  /* Floats per token per plane, from `projection_dim`. */
  int64_t projection_dim() const { return projection_dim_; }
  /* How many planes one image produces: the base features and one per deepstack
   * tap, in ascending layer order.  The text graph's `n_deepstack_layers` is
   * this minus one. */
  int64_t n_planes() const { return static_cast<int64_t>(deepstack_layers_.size()) + 1; }
  /* The layers the taps come from, ascending. */
  const std::vector<int64_t> &deepstack_layers() const { return deepstack_layers_; }

  /* Run the tower over one image and write its features to `out`, which holds
   * `n_planes() * n_tokens() * projection_dim()` floats laid out **plane-major**:
   * plane `p` occupies `[p * n_tokens() * projection_dim(), (p + 1) * ...)`.
   *
   * Plane-major rather than llama.cpp's interleaved order because each plane is
   * one `gemm`'s output here, and because the text graph consumes them one plane
   * per layer.  The correspondence between the two layouts is a transpose of the
   * plane axis and nothing else.
   *
   * `rgb` is `height * width * 3` bytes, row-major, top-to-bottom, at **any** size:
   * the image is resized to a grid this tower can consume by `preprocess_image`,
   * with the checkpoint's own pixel limits and the reference's bilinear kernel, and
   * `out` then holds `n_planes() * tokens_for(width, height) * projection_dim()`
   * floats.  The two steps are worth separating in the caller's mind because a
   * different target size is a different number of tokens and so a different prompt
   * -- `tokens_for` is how a caller knows how much room to leave.
   *
   * The pixels are normalized here with the checkpoint's own `image_mean`/
   * `image_std`, exactly as llama.cpp's preprocessing does after its resize:
   * `(value / 255 - mean[c]) / std[c]`. */
  void encode(const uint8_t *rgb, int64_t height, int64_t width, float *out);

 private:
  explicit Qwen3VlVision(kernel::Backend &backend);

  void allocate_activations();
  /* The merge permutation and the rotary tables, both of which depend only on
   * the patch grid and are therefore built once, at load. */
  void build_merge_index(int64_t grid_w, int64_t grid_h);
  void build_rope_tables(int64_t grid_w, int64_t grid_h);

  kernel::Backend *backend_ = nullptr;
  int64_t n_embd_ = 0;
  int64_t n_head_ = 0;
  int64_t head_dim_ = 0;
  int64_t n_layer_ = 0;
  int64_t n_ff_ = 0;
  int64_t patch_size_ = 0;
  int64_t merge_ = 0;
  int64_t image_size_ = 0;
  /* The checkpoint's own pixel limits for an input image.  The preprocessing uses
   * them to pick a size the tower can take, and they are read rather than defaulted
   * because they decide the token count of anything that is not already 768 square. */
  int64_t image_min_pixels_ = 0;
  int64_t image_max_pixels_ = 0;
  int64_t projection_dim_ = 0;
  int64_t grid_side_ = 0;   /* image_size / patch_size */
  int64_t merged_side_ = 0; /* grid_side / merge */
  int64_t n_patches_ = 0;   /* grid_side^2 */
  int64_t n_tokens_ = 0;    /* merged_side^2, what the text graph sees */
  /* The largest grid the checkpoint's pixel limit allows, which is what the
   * activations are sized for: `encode` takes any input shape and resizes it onto a
   * grid, so the buffers have to hold the biggest of those rather than the one
   * `image_size` implies. */
  int64_t max_patches_ = 0;
  int64_t max_side_ = 0; /* the pixel width that `max_patches_` is the grid of */
  /* The grid the merge permutation and the rotary tables were last built for.  They
   * are a function of the grid, so a caller encoding images of different shapes pays
   * a rebuild per change and one encoding one shape pays once. */
  int64_t active_grid_w_ = -1;
  int64_t active_grid_h_ = -1;
  /* The learned position table on the host, kept because an image of another size
   * needs it *interpolated* onto that size's patch grid and interpolation is a
   * computation over the table rather than a property of it.  The device copy the
   * gather reads is rebuilt whenever the grid changes; this is the source. */
  std::vector<float> position_table_host_;
  /* The interpolated table for the active grid, on the device. */
  kernel::DeviceBuffer position_table_resampled_;
  int64_t merged_width_ = 0; /* merge^2 * n_embd, the width the merger reads */
  float eps_ = 0.0F;
  float image_mean_[3] = {0.5F, 0.5F, 0.5F};
  float image_std_[3] = {0.5F, 0.5F, 0.5F};
  std::vector<int64_t> deepstack_layers_;

  /* Weights.  Every one is f32 on the device: the file's f16 tensors are widened
   * at load, the same rule the text model's dense weights follow and for the same
   * reason -- the kernels contract in f32, so an f16 tensor in the file is a
   * storage choice rather than a precision one.  (The tower is 407M weights, so
   * this costs 1.6 GB of device memory against the file's 819 MB.  A quantized or
   * deferred binding is future work and a different trade than correctness.) */
  VisionLinear patch_w0_, patch_w1_;
  kernel::DeviceBuffer patch_bias_;
  kernel::DeviceBuffer position_table_;
  std::vector<VisionBlock> blocks_;
  kernel::DeviceBuffer post_ln_w_, post_ln_b_;
  VisionLinear mm0_, mm2_;

  /* Activations, allocated once: one image's worth, because the tower processes
   * one image per call. */
  kernel::DeviceBuffer pixels_;      /* (3, image, image), width fastest */
  kernel::DeviceBuffer conv_;        /* (n_patches, n_embd), raster patch order */
  kernel::DeviceBuffer conv_tmp_;    /* the second temporal tap */
  kernel::DeviceBuffer tokens_;      /* (n_patches, n_embd), merge order */
  kernel::DeviceBuffer position_rows_;
  kernel::DeviceBuffer x_;           /* (n_patches, n_embd), the residual stream */
  kernel::DeviceBuffer q_, k_, v_;   /* (n_patches, n_head, head_dim) */
  kernel::DeviceBuffer attn_out_;    /* (n_patches, n_head, head_dim) */
  kernel::DeviceBuffer norm_;        /* (n_patches, n_embd) */
  kernel::DeviceBuffer up_;          /* (n_patches, n_ff) */
  kernel::DeviceBuffer merged_norm_; /* (n_tokens, merged_width) */
  kernel::DeviceBuffer merged_hidden_;
  kernel::DeviceBuffer base_out_;    /* (n_tokens, projection_dim) */
  std::vector<kernel::DeviceBuffer> deepstack_out_;
  kernel::DeviceBuffer scores_;
  kernel::DeviceBuffer cos_, sin_;   /* (n_patches, head_dim / 2) */
  kernel::DeviceBuffer merge_index_; /* n_patches int32 row indices */

  /* Every buffer this model allocated, so the destructor is a loop rather than a
   * list that a weight added later can be missing from. */
  std::vector<kernel::DeviceBuffer> owned_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_QWEN3VL_VISION_H */
