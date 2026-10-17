#include "model/audio_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "runtime/status.h"

namespace pocketllm {
namespace {

constexpr double kPi = 3.14159265358979323846;

/* The twiddle tables, built once for `n_fft`.
 *
 * `sinf`/`cosf` of a `double` angle and not `std::sin` on a float: the reference
 * evaluates `theta` in double precision and narrows once, and the tables are indexed
 * arithmetically below, so the two have to be the same numbers rather than
 * nearly the same ones. */
struct Twiddles {
  std::vector<float> sin_vals;
  std::vector<float> cos_vals;

  explicit Twiddles(int64_t n) {
    sin_vals.resize(static_cast<std::size_t>(n));
    cos_vals.resize(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const double theta = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(n);
      sin_vals[static_cast<std::size_t>(i)] = std::sin(static_cast<float>(theta));
      cos_vals[static_cast<std::size_t>(i)] = std::cos(static_cast<float>(theta));
    }
  }
};

/* The reference's naive transform, which its recursion falls back to for an odd
 * size.  `out` is interleaved `(re, im)` and `RealInput` skips the imaginary term of
 * the input, which is the only reason this is two loops rather than one. */
void dft_real(const Twiddles &table, const float *in, int64_t n, float *out) {
  const int64_t table_size = static_cast<int64_t>(table.sin_vals.size());
  const int64_t step = table_size / n;
  for (int64_t k = 0; k < n; ++k) {
    float re = 0.0F;
    float im = 0.0F;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t index = (k * i * step) % table_size;
      const float cos_val = table.cos_vals[static_cast<std::size_t>(index)];
      const float sin_val = table.sin_vals[static_cast<std::size_t>(index)];
      const float value = in[i];
      re += value * cos_val;
      im += -value * sin_val;
    }
    out[k * 2 + 0] = re;
    out[k * 2 + 1] = im;
  }
}

/* The reference also carries a complex-input transform for its inverse direction;
 * it is not reproduced here because nothing on this path takes one -- the front end
 * transforms real frames and only forwards.  A copy of it would be dead code that a
 * reader has to check against the original for no gain. */

/* The recursion: an odd size goes to the naive transform, an even one splits into
 * even and odd halves.  `n_fft` 400 becomes 400 -> 200 -> 100 -> 50 -> 25, and 25 is
 * where the O(n^2) sum runs.
 *
 * The twiddle for output `k` in the upper half is read at `k * step` with
 * `step = table_size / n`, and the combination is the `W_N^k` butterfly.  The order
 * of the additions is the reference's, which matters because float addition is not
 * associative and this front end is compared to float32. */
void fft_real(const Twiddles &table, const float *in, int64_t n, float *out) {
  if (n == 1) {
    out[0] = in[0];
    out[1] = 0.0F;
    return;
  }
  const int64_t half = n / 2;
  if (n - half * 2 == 1) {
    dft_real(table, in, n, out);
    return;
  }

  std::vector<float> even(static_cast<std::size_t>(half), 0.0F);
  std::vector<float> odd(static_cast<std::size_t>(half), 0.0F);
  for (int64_t i = 0; i < half; ++i) {
    even[static_cast<std::size_t>(i)] = in[2 * i];
    odd[static_cast<std::size_t>(i)] = in[2 * i + 1];
  }
  std::vector<float> even_out(static_cast<std::size_t>(half * 2), 0.0F);
  std::vector<float> odd_out(static_cast<std::size_t>(half * 2), 0.0F);
  fft_real(table, even.data(), half, even_out.data());
  fft_real(table, odd.data(), half, odd_out.data());

  const int64_t table_size = static_cast<int64_t>(table.sin_vals.size());
  const int64_t step = table_size / n;
  for (int64_t k = 0; k < half; ++k) {
    const int64_t index = (k * step) % table_size;
    const float cos_val = table.cos_vals[static_cast<std::size_t>(index)];
    const float sin_val = table.sin_vals[static_cast<std::size_t>(index)];
    const float odd_re = odd_out[k * 2 + 0];
    const float odd_im = odd_out[k * 2 + 1];
    /* W_N^k * odd = (cos - i sin)(odd_re + i odd_im) */
    const float tw_re = odd_re * cos_val + odd_im * sin_val;
    const float tw_im = -odd_re * sin_val + odd_im * cos_val;
    const float even_re = even_out[k * 2 + 0];
    const float even_im = even_out[k * 2 + 1];
    out[k * 2 + 0] = even_re + tw_re;
    out[k * 2 + 1] = even_im + tw_im;
    out[(k + half) * 2 + 0] = even_re - tw_re;
    out[(k + half) * 2 + 1] = even_im - tw_im;
  }
}

/* The periodic Hann window.  `offset` is 0 here, which is the whole of the
 * periodicity: the symmetric window divides by `length - 1` and an earlier
 * reconstruction of this divided by `length + 1`, which is neither. */
std::vector<float> hann_window(int64_t length) {
  std::vector<float> window(static_cast<std::size_t>(length));
  for (int64_t i = 0; i < length; ++i) {
    const double theta = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(length);
    window[static_cast<std::size_t>(i)] = 0.5 * (1.0 - std::cos(static_cast<float>(theta)));
  }
  return window;
}

double hz_to_mel(double hz) {
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / (200.0 / 3.0);
  const double log_step = std::log(6.4) / 27.0;
  return hz < min_log_hz ? hz * (3.0 / 200.0) : min_log_mel + std::log(hz / min_log_hz) / log_step;
}

double mel_to_hz(double mel) {
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / (200.0 / 3.0);
  const double log_step = std::log(6.4) / 27.0;
  return mel < min_log_mel ? mel / (3.0 / 200.0)
                           : min_log_hz * std::exp((mel - min_log_mel) * log_step);
}

/* `(n_mel, n_fft / 2 + 1)`, triangles on the Slaney scale.
 *
 * The area normalisation `2 / (f_right - f_left)` is the *whole* of the scale: an
 * earlier reconstruction also multiplied by the mel step, which librosa does and the
 * reference does not, and that is a constant factor of 62.02 -- every mel value
 * exactly 1.793 too high in log10. */
std::vector<double> mel_filterbank(int64_t n_mel, int64_t n_fft, int64_t sample_rate) {
  const int64_t n_bins = n_fft / 2 + 1;
  const double bin_hz_step = static_cast<double>(sample_rate) / static_cast<double>(n_fft);
  const double fmax = 0.5 * static_cast<double>(sample_rate);

  std::vector<double> hz(static_cast<std::size_t>(n_mel + 2));
  const double mel_min = hz_to_mel(0.0);
  const double mel_max = hz_to_mel(fmax);
  for (int64_t i = 0; i < n_mel + 2; ++i) {
    const double mel = mel_min + (mel_max - mel_min) * static_cast<double>(i) /
                                     static_cast<double>(n_mel + 1);
    hz[static_cast<std::size_t>(i)] = mel_to_hz(mel);
  }

  std::vector<double> bank(static_cast<std::size_t>(n_mel * n_bins), 0.0);
  for (int64_t m = 0; m < n_mel; ++m) {
    const double left = hz[static_cast<std::size_t>(m)];
    const double center = hz[static_cast<std::size_t>(m + 1)];
    const double right = hz[static_cast<std::size_t>(m + 2)];
    const double denom_l = std::max(1e-30, center - left);
    const double denom_r = std::max(1e-30, right - center);
    const double enorm = 2.0 / std::max(1e-30, right - left);
    for (int64_t k = 0; k < n_bins; ++k) {
      const double f = static_cast<double>(k) * bin_hz_step;
      double w = 0.0;
      if (f >= left && f <= center) {
        w = (f - left) / denom_l;
      } else if (f > center && f <= right) {
        w = (right - f) / denom_r;
      }
      bank[static_cast<std::size_t>(m * n_bins + k)] = w * enorm;
    }
  }
  return bank;
}

/* How many frames the padded signal produces.  Kept apart from the transform because
 * the count is a function of the lengths alone, and a caller sizing a loop should not
 * have to run an FFT to find out -- or keep a second copy of the formula. */
void log_mel_spectrogram_shape(int64_t n_samples, const AudioFrontendParams &params,
                               int64_t *n_frames) {
  const int64_t padded = n_samples + 2 * (params.n_fft / 2);
  *n_frames = (padded - params.n_fft) / params.hop_len + 1;
}

/* The log-mel of the whole signal, before the normalisation: `(n_mel, n_frames)`. */
std::vector<float> log_mel_spectrogram(const float *samples, int64_t n_samples,
                                       const AudioFrontendParams &params, int64_t *n_frames_out) {
  const int64_t pad = params.n_fft / 2;
  /* The reflection is the reference's, sample for sample: the first `pad` outputs
   * read backwards from `samples[pad]`, and the last `pad` read backwards from
   * `samples[n - 2]`.  Both have a guard for a signal shorter than the padding. */
  std::vector<float> padded(static_cast<std::size_t>(n_samples + 2 * pad), 0.0F);
  for (int64_t i = 0; i < pad; ++i) {
    const int64_t src = pad - i;
    padded[static_cast<std::size_t>(i)] = src < n_samples ? samples[src] : 0.0F;
  }
  std::copy(samples, samples + n_samples, padded.begin() + pad);
  for (int64_t i = 0; i < pad; ++i) {
    const int64_t src = n_samples - 2 - i;
    padded[static_cast<std::size_t>(n_samples + pad + i)] = src >= 0 ? samples[src] : 0.0F;
  }

  const std::vector<float> window = hann_window(params.window_len);
  const std::vector<double> bank = mel_filterbank(params.n_mel, params.n_fft, params.sample_rate);
  const Twiddles table(params.n_fft);
  const int64_t n_bins = params.n_fft / 2 + 1;
  const int64_t n_frames = (static_cast<int64_t>(padded.size()) - params.n_fft) / params.hop_len + 1;
  if (n_frames <= 0) {
    throw Error("an audio signal of " + std::to_string(n_samples) +
                " samples is shorter than one analysis frame");
  }

  std::vector<float> out(static_cast<std::size_t>(params.n_mel * n_frames), 0.0F);
  std::vector<float> frame(static_cast<std::size_t>(params.n_fft), 0.0F);
  std::vector<float> spectrum(static_cast<std::size_t>(params.n_fft * 2), 0.0F);
  std::vector<float> power(static_cast<std::size_t>(n_bins), 0.0F);

  /* Only frames inside `n_samples / hop + 1` are computed; the rest are the
   * reference's `log10(floor)` fill, which the normalisation then turns into the
   * clamp value.  Computing them would cost the transform for nothing. */
  const int64_t computed = std::min(n_frames, n_samples / params.hop_len + 1);
  for (int64_t f = 0; f < computed; ++f) {
    const int64_t offset = f * params.hop_len;
    /* The window applies to `window_len` samples and the frame is zero-filled beyond
     * that if it runs past the padded signal, which is what the reference's own fill
     * does. */
    for (int64_t i = 0; i < params.window_len; ++i) {
      const int64_t at = offset + i;
      frame[static_cast<std::size_t>(i)] =
          at < static_cast<int64_t>(padded.size()) ? window[static_cast<std::size_t>(i)] *
                                                         padded[static_cast<std::size_t>(at)]
                                                   : 0.0F;
    }
    if (params.window_len < params.n_fft) {
      std::fill(frame.begin() + params.window_len, frame.end(), 0.0F);
    }
    fft_real(table, frame.data(), params.n_fft, spectrum.data());
    for (int64_t k = 0; k < n_bins; ++k) {
      const float re = spectrum[static_cast<std::size_t>(k * 2 + 0)];
      const float im = spectrum[static_cast<std::size_t>(k * 2 + 1)];
      power[static_cast<std::size_t>(k)] = re * re + im * im;
    }
    for (int64_t m = 0; m < params.n_mel; ++m) {
      /* The accumulation is `double` in the reference and the narrowing happens once,
       * at the store -- so a float accumulator here would differ from it in the last
       * places of every band. */
      double sum = 0.0;
      for (int64_t k = 0; k < n_bins; ++k) {
        sum += static_cast<double>(power[static_cast<std::size_t>(k)]) *
               bank[static_cast<std::size_t>(m * n_bins + k)];
      }
      sum = std::max(sum, static_cast<double>(params.mel_floor));
      out[static_cast<std::size_t>(m * n_frames + f)] = static_cast<float>(std::log10(sum));
    }
  }
  /* The tail the reference fills without transforming. */
  for (int64_t f = computed; f < n_frames; ++f) {
    const float value = static_cast<float>(std::log10(static_cast<double>(params.mel_floor)));
    for (int64_t m = 0; m < params.n_mel; ++m) {
      out[static_cast<std::size_t>(m * n_frames + f)] = value;
    }
  }

  *n_frames_out = n_frames;
  return out;
}

}  // namespace

int64_t audio_window_count(int64_t n_samples, const AudioFrontendParams &params) {
  if (n_samples <= 0) {
    return 0;
  }
  int64_t frames = 0;
  /* The frame count does not depend on the values, only on the lengths, and
   * `log_mel_spectrogram` is where that arithmetic lives -- so this asks it with a
   * null signal rather than keeping a second copy of the formula that can drift. */
  log_mel_spectrogram_shape(n_samples, params, &frames);
  const int64_t effective = std::min<int64_t>(frames, n_samples / params.hop_len + 1);
  return (effective + params.window_frames - 1) / params.window_frames;
}

AudioMel audio_log_mel_window(const float *samples, int64_t n_samples,
                              const AudioFrontendParams &params, int64_t window_index) {
  if (samples == nullptr || n_samples <= 0) {
    throw Error("the audio front end needs samples");
  }
  int64_t frames = 0;
  const std::vector<float> full = log_mel_spectrogram(samples, n_samples, params, &frames);

  /* `n_eff` is the reference's: the transform produces `frames`, but only
   * `n_samples / hop + 1` of them are audio, and the rest are the fill. */
  const int64_t effective = std::min<int64_t>(frames, n_samples / params.hop_len + 1);
  const int64_t windows = (effective + params.window_frames - 1) / params.window_frames;
  if (window_index < 0 || window_index >= windows) {
    throw Error("window " + std::to_string(window_index) + " was asked for, and this signal has " +
                std::to_string(windows) + " of them");
  }

  const int64_t offset = window_index * params.window_frames;
  const int64_t within = std::min<int64_t>(params.window_frames, effective - offset);

  /* Whisper's normalisation over the whole spectrogram, not per window: the maximum
   * is the signal's, so a caller that normalised window by window would scale the
   * quiet ones up and feed the tower something the reference never produced. */
  double maximum = -1e20;
  for (float value : full) {
    maximum = std::max(maximum, static_cast<double>(value));
  }
  const double floor_value = maximum - 8.0;

  /* Padded up to a whole number of conv sub-chunks, which is what the graph's
   * `n_chunks` view needs. */
  const int64_t chunks = (within + params.conv_chunk - 1) / params.conv_chunk;
  AudioMel out;
  out.n_mel = params.n_mel;
  out.n_len = chunks * params.conv_chunk;
  out.n_len_org = within;
  out.data.assign(static_cast<std::size_t>(out.n_mel * out.n_len), 0.0F);
  for (int64_t m = 0; m < params.n_mel; ++m) {
    for (int64_t t = 0; t < within; ++t) {
      const double value = static_cast<double>(full[static_cast<std::size_t>(m * frames + offset + t)]);
      const double clamped = std::max(value, floor_value);
      out.data[static_cast<std::size_t>(m * out.n_len + t)] =
          static_cast<float>((clamped + 4.0) / 4.0);
    }
  }
  return out;
}

}  // namespace pocketllm
