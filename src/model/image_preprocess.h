/* The image preprocessing a `qwen3vl` tower's caller has to do, as the reference does it.
 *
 * The tower itself takes one square image of the size its position table was stored
 * for.  Everything before that -- choosing a size for an arbitrary image and
 * resampling to it -- is the caller's, and a caller that chooses differently has
 * built a different prompt, not merely an approximate one: the token count is
 * `(nx / patch / merge) * (ny / patch / merge)` and it changes with the size.
 *
 * So both halves are here, transcribed from the reference rather than reinvented,
 * because the *pixels* have to agree too and a resampler is a dozen decisions:
 *
 *   - **The target size is `mtmd-image.cpp`'s `calc_size_preserved_ratio`**: round
 *     both sides to a multiple of `patch * merge` (32 here), then shrink by a single
 *     aspect-preserving factor if the result is over `max_pixels` and grow by one if
 *     it is under `min_pixels`.  It is *not* a plain aspect fit -- `800x600` becomes
 *     `800x608`, because the rounding-up happens first and the growth branch is what
 *     the limits are for.
 *   - **The resampler is `resize_bilinear` with `align_corners` semantics** --
 *     `x_ratio = (src - 1) / (dst - 1)`, so the two corner pixels map exactly onto
 *     each other -- and it truncates to `uint8_t` rather than rounding.  Truncation
 *     is worth its own sentence: a rounded resampler differs from this one by one
 *     grey level on a good fraction of pixels, which is invisible in a picture and
 *     not in the tower's output.
 *
 * The reference's own numbers for this checkpoint are `image_min_pixels` 8192,
 * `image_max_pixels` 4194304 and `align` 32 (`patch_size` 16 times `n_merge` 2), read
 * from the mmproj's metadata rather than assumed.  `tests/native/test_vlm.py`'s
 * resampling case checks the rule against the reference's own token counts, which
 * invert to the size it chose.
 */

#ifndef POCKETLLM_MODEL_IMAGE_PREPROCESS_H
#define POCKETLLM_MODEL_IMAGE_PREPROCESS_H

#include <cstdint>
#include <vector>

namespace pocketllm {

/* An image as the preprocessor passes it on: `nx * ny * 3` bytes, row-major, three
 * interleaved channels. */
struct PreprocessedImage {
  int64_t width = 0;
  int64_t height = 0;
  std::vector<uint8_t> rgb;
};

/* The size `calc_size_preserved_ratio` picks for an `width x height` input.  `align`
 * is the patch grid step the tower can consume, and the two pixel limits are the
 * checkpoint's own. */
void resize_target(int64_t width, int64_t height, int64_t align, int64_t min_pixels,
                   int64_t max_pixels, int64_t *target_width, int64_t *target_height);

/* Resample `rgb` to `target_width x target_height` with the reference's bilinear
 * kernel.  `rgb` is `width * height * 3` bytes and the result is
 * `target_width * target_height * 3`. */
std::vector<uint8_t> resize_bilinear(const uint8_t *rgb, int64_t width, int64_t height,
                                     int64_t target_width, int64_t target_height);

/* The tower's *learned position table* resampled onto a different patch grid.
 *
 * The table is a weight of shape `(side * side, n_embd)` -- one row per patch of the
 * grid the checkpoint was trained on -- and an image of another size needs a row per
 * patch of *its* grid.  llama.cpp interpolates the table rather than refusing the
 * image, and the mode is worth stating exactly because it is the antialiased one:
 * `F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)`, a
 * triangle filter whose support grows with the downscale factor, computed in the
 * target's coordinates with a half-pixel offset.
 *
 * This is not the same kernel as `resize_bilinear` above and the difference is not
 * cosmetic: that one is align-corners, truncating and un-antialiased, because it is
 * what the reference uses for *pixels*.  Two resamplers in one file is honest here --
 * they resample different things and the reference picked differently for each.
 *
 * `table` is `(source_side * source_side) * n_embd` floats, row-major by patch, and
 * the result is `(target_width * target_height) * n_embd`.  A target equal to the
 * source is returned unchanged.
 */
std::vector<float> interpolate_position_table(const float *table, int64_t source_side,
                                              int64_t target_width, int64_t target_height,
                                              int64_t n_embd);

/* Both steps: pick the size the tower can take and resample to it.  `align`,
 * `min_pixels` and `max_pixels` come from the checkpoint; an image already at the
 * target size comes back unchanged, which is the common case for a caller that
 * resized upstream. */
PreprocessedImage preprocess_image(const uint8_t *rgb, int64_t width, int64_t height, int64_t align,
                                   int64_t min_pixels, int64_t max_pixels);

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_IMAGE_PREPROCESS_H */