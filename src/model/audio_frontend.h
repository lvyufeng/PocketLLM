/* The log-mel front end a `qwen3a` audio tower is fed, as the reference computes it.
 *
 * Audio reaches the engine as PCM and reaches the tower as a mel spectrogram, and
 * every step in between is a decision that a transcript would not distinguish and
 * the numbers would: the window's periodicity, the mel scale, the log's base, the
 * floor, the padding, and where the frames are cut.  All of them are transcribed
 * from `tools/mtmd/mtmd-audio.cpp` and all of them were checked against the
 * reference value for value -- the reconstruction in that check agrees to 6.4e-07
 * in the mean over 128 x 1301 values, which is float32 storage rather than
 * agreement of intent.
 *
 * The contract, which `tests/native/test_audio_frontend.py` holds to:
 *
 *   - 16 kHz, `n_fft` 400, window 400, hop 160, 128 mel bins.
 *   - `n_fft / 2` samples of **reflection** at each end, mirroring
 *     `WhisperFeatureExtractor(center=True)`.
 *   - a **periodic** Hann window: `0.5 * (1 - cos(2*pi*i / length))`.  Dividing by
 *     `length + 1` is a different window and cost the first reconstruction 0.004 in
 *     the mean over the whole spectrogram -- it is one character and it is the kind of
 *     thing that survives a correlation of 0.99997.
 *   - **power**, not magnitude, and **log10**.
 *   - the mel bank is triangles on the **Slaney** scale with area normalisation
 *     `2 / (f_right - f_left)` and nothing else.  librosa additionally scales by the
 *     mel step; the reference does not, and adding it put every value exactly 1.793
 *     too high.
 *   - the power is clamped to `2^-24` before the log, which is llama.cpp's
 *     `mel_floor` -- not the `1e-10` that another of its audio preprocessors uses.
 *   - Whisper's normalisation: clamp to `max - 8`, then `(v + 4) / 4`.
 *   - `n_eff = min(n_len, n_samples / hop + 1)` frames are kept and the result is
 *     padded up to a multiple of 100, which is the conv sub-chunk size.
 *
 * The transform is the reference's own decomposition rather than a library FFT,
 * because the two round differently and the comparison is to float32: a naive DFT
 * for odd sizes, and a recursive even/odd split above that, so 400 -> 200 -> 100 ->
 * 50 -> 25, where the odd branch runs the O(n^2) sum.  The twiddle tables are built
 * the same way and indexed the same way.
 */

#ifndef POCKETLLM_MODEL_AUDIO_FRONTEND_H
#define POCKETLLM_MODEL_AUDIO_FRONTEND_H

#include <cstdint>
#include <vector>

namespace pocketllm {

/* What one `mtmd_audio_mel` carries: a mel-major block of frames, with `n_len` the
 * padded length the tower takes and `n_len_org` how many of those frames are audio
 * rather than padding.  The distinction is the reference's and a caller that ignores
 * it feeds the tower a different number of frames than the reference did. */
struct AudioMel {
  int64_t n_mel = 0;
  int64_t n_len = 0;
  int64_t n_len_org = 0;
  std::vector<float> data; /* (n_mel, n_len), mel-major */
};

/* The parameters, all of which come from the checkpoint's own hparams except the
 * ones llama.cpp fixes for this projector family. */
struct AudioFrontendParams {
  int64_t sample_rate = 16000;
  int64_t n_fft = 400;
  int64_t window_len = 400;
  int64_t hop_len = 160;
  int64_t n_mel = 128;
  /* The conv sub-chunk size the output is padded to, `n_window * 2`. */
  int64_t conv_chunk = 100;
  /* Frames per inference window, `n_window_infer`. */
  int64_t window_frames = 800;
  /* The power a mel band is clamped to before the log: llama.cpp's `mel_floor`,
   * which is 2^-24 -- the smallest positive f16 subnormal -- and not a round number
   * in decimal. */
  float mel_floor = 5.960464477539063e-08F;
};

/* One inference window of log-mel, normalised and padded, from `samples`.
 *
 * The whole signal is turned into frames once and then cut into `window_frames`-sized
 * windows, so a caller looping over the result must pass the *same* samples with a
 * different window index rather than slicing them: the reflection padding is applied
 * to the signal, not to the window, and slicing first would move every frame. */
AudioMel audio_log_mel_window(const float *samples, int64_t n_samples,
                              const AudioFrontendParams &params, int64_t window_index);

/* How many inference windows `n_samples` produces, and the padded frame count of
 * each.  A caller sizes its loop from this; asking for a window index past the end
 * throws rather than returning an empty block. */
int64_t audio_window_count(int64_t n_samples, const AudioFrontendParams &params);

}  // namespace pocketllm

#endif /* POCKETLLM_MODEL_AUDIO_FRONTEND_H */