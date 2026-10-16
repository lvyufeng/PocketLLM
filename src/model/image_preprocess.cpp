#include "model/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "runtime/status.h"

namespace pocketllm {
namespace {

int64_t round_by(int64_t value, int64_t factor) {
  return static_cast<int64_t>(std::llround(static_cast<double>(value) /
                                          static_cast<double>(factor))) *
         factor;
}

int64_t ceil_by(int64_t value, int64_t factor) {
  return static_cast<int64_t>(std::ceil(static_cast<double>(value) /
                                        static_cast<double>(factor))) *
         factor;
}

int64_t floor_by(int64_t value, int64_t factor) {
  return static_cast<int64_t>(std::floor(static_cast<double>(value) /
                                         static_cast<double>(factor))) *
         factor;
}

float lerp(float a, float b, float t) { return a + t * (b - a); }

}  // namespace

void resize_target(int64_t width, int64_t height, int64_t align, int64_t min_pixels,
                   int64_t max_pixels, int64_t *target_width, int64_t *target_height) {
  if (align <= 0) {
    throw Error("image preprocessing needs a positive alignment, and " + std::to_string(align) +
                " was given");
  }
  if (width <= 0 || height <= 0) {
    throw Error("image preprocessing needs a positive size, and " + std::to_string(width) + "x" +
                std::to_string(height) + " was given");
  }
  /* Both sides sit on the grid before the limits are consulted, which is what makes
   * this rule differ from an aspect fit: `800x600` becomes `800x608`, not
   * `768x576`. */
  int64_t h_bar = std::max<int64_t>(align, round_by(height, align));
  int64_t w_bar = std::max<int64_t>(align, round_by(width, align));

  if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
    /* One factor for both axes, so the aspect ratio survives the shrink. */
    const double beta =
        std::sqrt(static_cast<double>(width) * static_cast<double>(height) /
                  static_cast<double>(max_pixels));
    h_bar = std::max<int64_t>(align, floor_by(static_cast<int64_t>(height / beta), align));
    w_bar = std::max<int64_t>(align, floor_by(static_cast<int64_t>(width / beta), align));
  } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                  (static_cast<double>(width) * static_cast<double>(height)));
    h_bar = ceil_by(height * beta, align);
    w_bar = ceil_by(width * beta, align);
  }
  *target_width = w_bar;
  *target_height = h_bar;
}

std::vector<uint8_t> resize_bilinear(const uint8_t *rgb, int64_t width, int64_t height,
                                     int64_t target_width, int64_t target_height) {
  if (target_width <= 0 || target_height <= 0) {
    throw Error("a resize target must be positive");
  }
  std::vector<uint8_t> out(static_cast<std::size_t>(3 * target_width * target_height));
  /* `align_corners`: the source's last pixel maps onto the destination's last, so a
   * resize that does not change the size is the identity.  The reference divides by
   * `target - 1` and takes the ratio from `src - 1`; both are zero-safe only because
   * a one-pixel axis gets a ratio of zero. */
  const float x_ratio = target_width > 1
                            ? static_cast<float>(width - 1) / static_cast<float>(target_width - 1)
                            : 0.0F;
  const float y_ratio = target_height > 1
                            ? static_cast<float>(height - 1) / static_cast<float>(target_height - 1)
                            : 0.0F;

  for (int64_t y = 0; y < target_height; ++y) {
    const float py = static_cast<float>(y) * y_ratio;
    const int64_t y0 = std::min<int64_t>(static_cast<int64_t>(py), height - 1);
    const int64_t y1 = std::min<int64_t>(y0 + 1, height - 1);
    const float yf = py - static_cast<float>(y0);
    for (int64_t x = 0; x < target_width; ++x) {
      const float px = static_cast<float>(x) * x_ratio;
      const int64_t x0 = std::min<int64_t>(static_cast<int64_t>(px), width - 1);
      const int64_t x1 = std::min<int64_t>(x0 + 1, width - 1);
      const float xf = px - static_cast<float>(x0);
      for (int64_t c = 0; c < 3; ++c) {
        const float top = lerp(static_cast<float>(rgb[3 * (y0 * width + x0) + c]),
                               static_cast<float>(rgb[3 * (y0 * width + x1) + c]), xf);
        const float bottom = lerp(static_cast<float>(rgb[3 * (y1 * width + x0) + c]),
                                  static_cast<float>(rgb[3 * (y1 * width + x1) + c]), xf);
        /* Truncation toward zero, not rounding: the reference casts, and a rounded
         * resampler parts from it by a grey level on many pixels. */
        out[static_cast<std::size_t>(3 * (y * target_width + x) + c)] =
            static_cast<uint8_t>(lerp(top, bottom, yf));
      }
    }
  }
  return out;
}

PreprocessedImage preprocess_image(const uint8_t *rgb, int64_t width, int64_t height, int64_t align,
                                   int64_t min_pixels, int64_t max_pixels) {
  PreprocessedImage result;
  resize_target(width, height, align, min_pixels, max_pixels, &result.width, &result.height);
  if (result.width == width && result.height == height) {
    /* Already the size the tower takes: copying rather than returning a view keeps
     * the caller's buffer its own, which is what the header promises. */
    result.rgb.assign(rgb, rgb + static_cast<std::size_t>(3 * width * height));
    return result;
  }
  result.rgb = resize_bilinear(rgb, width, height, result.width, result.height);
  return result;
}

std::vector<float> interpolate_position_table(const float *table, int64_t source_side,
                                              int64_t target_width, int64_t target_height,
                                              int64_t n_embd) {
  if (source_side <= 0 || target_width <= 0 || target_height <= 0 || n_embd <= 0) {
    throw Error("a position table needs positive dimensions");
  }
  if (target_width == source_side && target_height == source_side) {
    return std::vector<float>(table,
                              table + static_cast<std::size_t>(source_side * source_side * n_embd));
  }
  /* The reference's mode is `align_corners = false` with antialiasing, so the scale
   * is the plain ratio and the sample point carries a half-pixel offset. */
  const float scale_x = static_cast<float>(target_width) / static_cast<float>(source_side);
  const float scale_y = static_cast<float>(target_height) / static_cast<float>(source_side);
  const float pixel_offset = 0.5F;
  /* One filter support per axis, at least one source pixel wide; below a scale of 1
   * (an upscale) the triangle's own support is already wider, which is what the
   * `max` is for. */
  const float support_x = std::max(1.0F, 1.0F / scale_x);
  const float support_y = std::max(1.0F, 1.0F / scale_y);
  const float invscale_x = 1.0F / support_x;
  const float invscale_y = 1.0F / support_y;
  const auto triangle = [](float x) { return std::max(1.0F - std::fabs(x), 0.0F); };

  std::vector<float> out(static_cast<std::size_t>(target_width * target_height * n_embd));
  for (int64_t ty = 0; ty < target_height; ++ty) {
    const float y = (static_cast<float>(ty) + pixel_offset) / scale_y;
    const int64_t y_min = std::max<int64_t>(static_cast<int64_t>(y - support_y + pixel_offset), 0);
    const int64_t y_max = std::min<int64_t>(static_cast<int64_t>(y + support_y + pixel_offset),
                                            source_side);
    for (int64_t tx = 0; tx < target_width; ++tx) {
      const float x = (static_cast<float>(tx) + pixel_offset) / scale_x;
      const int64_t x_min = std::max<int64_t>(static_cast<int64_t>(x - support_x + pixel_offset), 0);
      const int64_t x_max = std::min<int64_t>(static_cast<int64_t>(x + support_x + pixel_offset),
                                              source_side);
      for (int64_t c = 0; c < n_embd; ++c) {
        float value = 0.0F;
        float total = 0.0F;
        for (int64_t sy = y_min; sy < y_max; ++sy) {
          const float weight_y =
              triangle((static_cast<float>(sy) - y + pixel_offset) * invscale_y);
          for (int64_t sx = x_min; sx < x_max; ++sx) {
            const float weight_x =
                triangle((static_cast<float>(sx) - x + pixel_offset) * invscale_x);
            const float weight = weight_x * weight_y;
            if (weight <= 0.0F) {
              continue;
            }
            value += table[static_cast<std::size_t>((sy * source_side + sx) * n_embd + c)] * weight;
            total += weight;
          }
        }
        /* The reference divides by the weight actually accumulated rather than by a
         * constant, which is what keeps an edge pixel from being darkened by the
         * taps that fell outside the table. */
        out[static_cast<std::size_t>((ty * target_width + tx) * n_embd + c)] =
            total > 0.0F ? value / total : 0.0F;
      }
    }
  }
  return out;
}

}  // namespace pocketllm
