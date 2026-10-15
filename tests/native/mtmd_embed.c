/* llama.cpp's vision tower output for one image, as float32.
 *
 * This is the tower's oracle, and it is the same shape of instrument as
 * `oracle.cpp` and `ggml_dequant.c`: a program compiled by the test that needs
 * it, against the library that owns the format, so `libpocketllm.so` keeps no
 * dependency on a llama.cpp checkout.
 *
 * The image is authored by the caller and handed over as raw RGB bytes, not as
 * a PNG.  That direction matters for the same reason it does in the op-level
 * harness: whatever produced the pixels has to be able to say *exactly* what
 * they were, and a decoder's own choices -- colour space, orientation, chroma
 * rounding -- would be a second description of the input that the engine would
 * have to match.  Raw bytes are the description.
 *
 * The text model is required even though nothing here runs its graph:
 * `mtmd_tokenize` needs a vocabulary to place the media marker in, and mtmd
 * loads it from the text checkpoint rather than from the mmproj.
 *
 * Usage:
 *   mtmd-embed <mmproj.gguf> <text_model.gguf> <rgb.bin> <width> <height> \
 *              <n_embd_out> <out.bin>
 *
 * `n_embd_out` is the number of floats one image token carries.  It is passed
 * in rather than read from the library because mtmd's public API does not expose
 * it (`clip_n_mmproj_embd` is exported but its `clip_ctx` is not reachable
 * through mtmd), and its documented size -- the text model's own embedding width
 * times the token count -- is wrong for a deepstack model, whose tower emits the
 * base features and one block per deepstack layer side by side.  The caller
 * derives it from the mmproj's metadata, which is where the engine's loader
 * derives it too, so a disagreement about the width is a disagreement the test
 * reports rather than one it hides.  Getting it wrong reads past the buffer and
 * fails loudly; it cannot silently compare the wrong numbers, because the rows
 * would not line up.
 *
 * Writes the embeddings to `out.bin` and `n_tokens`/`n_embd_out` to stdout, in
 * the order `mtmd_get_output_embd` lays them out: token-major, so token `t`
 * occupies `[t * n_embd_out, (t + 1) * n_embd_out)`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"
#include "mtmd.h"

static void *read_file(const char *path, long *size) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    perror(path);
    exit(2);
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    exit(2);
  }
  const long length = ftell(file);
  if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
    exit(2);
  }
  void *bytes = malloc((size_t) length);
  if (bytes == NULL || fread(bytes, 1, (size_t) length, file) != (size_t) length) {
    fprintf(stderr, "failed to read %s\n", path);
    exit(2);
  }
  fclose(file);
  *size = length;
  return bytes;
}

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: %s <mmproj.gguf> <text_model.gguf> <rgb.bin> <width> <height> "
            "<n_embd_out> <out.bin>\n",
            argv[0]);
    return 2;
  }
  const unsigned width = (unsigned) atoi(argv[4]);
  const unsigned height = (unsigned) atoi(argv[5]);
  const size_t n_embd_out = (size_t) atoll(argv[6]);

  long rgb_bytes = 0;
  unsigned char *rgb = read_file(argv[3], &rgb_bytes);
  if (rgb_bytes != (long) (3u * width * height)) {
    fprintf(stderr, "the image file is not width * height * 3 bytes\n");
    return 2;
  }

  llama_backend_init();
  struct llama_model_params mparams = llama_model_default_params();
  /* The text model is a vocabulary here, and nothing about this instrument
   * should depend on where its weights could be computed: leaving it on the CPU
   * keeps the oracle a statement about the tower rather than about a device. */
  mparams.n_gpu_layers = 0;
  struct llama_model *text_model = llama_model_load_from_file(argv[2], mparams);
  if (text_model == NULL) {
    fprintf(stderr, "could not load the text model %s\n", argv[2]);
    return 3;
  }

  struct mtmd_context_params cparams = mtmd_context_params_default();
  cparams.use_gpu = false;
  cparams.n_threads = 4;
  /* Flash attention is pinned off, and that is not a default left unset.
   *
   * `mtmd_context_params_default()` leaves `flash_attn_type` at AUTO, which
   * resolves to *enabled* wherever the backend has a kernel -- the CPU included.
   * A flash attention is a different reduction over the span than a plain
   * softmax: a running maximum with a rescaled merge, against one shift.  The
   * engine's `attention` is the full-softmax form, so an oracle left on AUTO
   * would be comparing the two conventions and reporting the difference as the
   * engine's error.  The same pin was needed by `tests/native/llama_oracle.py`,
   * for the same reason and one graph earlier. */
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  mtmd_context *ctx = mtmd_init_from_file(argv[1], text_model, cparams);
  if (ctx == NULL) {
    fprintf(stderr, "could not load the mmproj %s\n", argv[1]);
    return 3;
  }
  if (!mtmd_support_vision(ctx)) {
    fprintf(stderr, "%s does not carry a vision tower\n", argv[1]);
    return 3;
  }

  mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, rgb);
  const mtmd_bitmap *bitmaps[1] = {bitmap};
  struct mtmd_input_text text = {mtmd_default_marker(), false, true};
  mtmd_input_chunks *chunks = mtmd_input_chunks_init();
  const int32_t tokenized = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
  if (tokenized != 0) {
    fprintf(stderr, "mtmd_tokenize returned %d\n", tokenized);
    return 3;
  }

  const mtmd_image_tokens *image = NULL;
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks) && image == NULL; ++i) {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      image = mtmd_input_chunk_get_tokens_image(chunk);
    }
  }
  if (image == NULL) {
    fprintf(stderr, "the tokenizer produced no image chunk\n");
    return 3;
  }
  const size_t n_tokens = mtmd_image_tokens_get_n_tokens(image);

  const int32_t encoded = mtmd_encode(ctx, image);
  if (encoded != 0) {
    fprintf(stderr, "mtmd_encode returned %d\n", encoded);
    return 3;
  }
  const float *embeddings = mtmd_get_output_embd(ctx);
  if (embeddings == NULL) {
    fprintf(stderr, "mtmd_get_output_embd returned null\n");
    return 3;
  }

  FILE *out = fopen(argv[7], "wb");
  if (out == NULL) {
    perror(argv[7]);
    return 2;
  }
  const size_t total = n_tokens * n_embd_out;
  if (fwrite(embeddings, sizeof(float), total, out) != total) {
    fprintf(stderr, "short write to %s\n", argv[7]);
    return 2;
  }
  fclose(out);

  printf("n_tokens %zu\n", n_tokens);
  printf("n_embd_out %zu\n", n_embd_out);

  mtmd_input_chunks_free(chunks);
  mtmd_bitmap_free(bitmap);
  mtmd_free(ctx);
  llama_model_free(text_model);
  llama_backend_free();
  free(rgb);
  return 0;
}
