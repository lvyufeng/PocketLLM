/* A whole vision-language eval through libmtmd: a prompt with one image, and the
 * logits for the next token.
 *
 * The tower's own oracle (`mtmd_embed.c`) stops at the image's features.  This is
 * the rest of the path -- the text graph fed at the span -- and it is the only
 * comparison that can say whether an image *reaches* the model: a tower whose
 * features are right and a text graph that injects them in the wrong place passes
 * both of the narrower tests and fails this one.
 *
 * **The chunks are decoded one at a time, and that is the reference's shape.**
 * `mtmd_tokenize` cuts a prompt into a text chunk, an image chunk and a text
 * chunk; `tools/mtmd/mtmd-helper.cpp` decodes them in order, carrying `n_past`
 * between them, and turns the image chunk's attention non-causal for the duration
 * of its own decode.  This instrument does the same through the public API rather
 * than by calling the helper, because `mtmd-helper.h` is a source-tree header and
 * because the engine's driver has to make the same three calls -- the point of the
 * comparison is that the two sides are two implementations of one contract.
 *
 * Usage:
 *   mtmd-vlm <mmproj.gguf> <text_model.gguf> <rgb.bin> <width> <height> <prompt>
 *
 * The prompt is text with the media marker in it (`mtmd_default_marker()`), passed
 * whole: what the marker means is the library's business, which is what makes this
 * an oracle.  Writes `{"n_vocab": N, "n_tokens": N, "logits": [...]}` on stdout --
 * the last position's logits, after every chunk -- and, with `--dump-embeddings
 * FILE`, the image chunk's embedding buffer as llama.cpp laid it out.  That dump
 * exists so a failing end-to-end comparison can be bisected: the engine fed
 * *these* numbers isolates whether it is the tower's features or the text graph's
 * composition.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

static const char *g_dump_embeddings = NULL;

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
  if (argc < 7) {
    fprintf(stderr, "usage: %s <mmproj> <text_model> <rgb.bin> <width> <height> <prompt>\n",
            argv[0]);
    return 2;
  }
  for (int i = 7; i + 1 < argc; ++i) {
    if (strcmp(argv[i], "--dump-embeddings") == 0) {
      g_dump_embeddings = argv[i + 1];
    }
  }
  const unsigned width = (unsigned) atoi(argv[4]);
  const unsigned height = (unsigned) atoi(argv[5]);
  long rgb_bytes = 0;
  unsigned char *rgb = read_file(argv[3], &rgb_bytes);
  if (rgb_bytes != (long) (3u * width * height)) {
    fprintf(stderr, "the image file is not width * height * 3 bytes\n");
    return 2;
  }

  llama_backend_init();
  struct llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  struct llama_model *model = llama_model_load_from_file(argv[2], mparams);
  if (model == NULL) {
    fprintf(stderr, "could not load the text model %s\n", argv[2]);
    return 3;
  }
  struct mtmd_context_params cparams = mtmd_context_params_default();
  cparams.use_gpu = false;
  cparams.n_threads = 4;
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  mtmd_context *ctx = mtmd_init_from_file(argv[1], model, cparams);
  if (ctx == NULL) {
    fprintf(stderr, "could not load the mmproj %s\n", argv[1]);
    return 3;
  }

  struct llama_context_params lparams = llama_context_default_params();
  lparams.n_ctx = 4096;
  lparams.n_batch = 4096;
  lparams.n_ubatch = 4096;
  lparams.offload_kqv = false;
  lparams.no_perf = true;
  lparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  /* **Every pin here is load-bearing, and the cost of skipping one is measured.**
   * The same prompt through `mtmd_helper_eval_chunks` answers differently under
   * each of them: with llama.cpp's default f16 KV cache rather than f32 the logits
   * move by 2.4 over a spread of 47 and the argmax changes, which is why an oracle
   * and a probe that differ only in these lines are not comparable -- an earlier
   * revision of this file measured 10.5% against the helper and the number was
   * mostly the settings, not the composition.  The engine contracts in f32 and
   * refuses to widen (its non-causal arm takes an f32 cache as it stands), so the
   * reference is pinned to the same width; `llama_oracle.py` gives the long form
   * of the argument for the text graph. */
  lparams.type_k = GGML_TYPE_F32;
  lparams.type_v = GGML_TYPE_F32;
  lparams.n_threads = 1;
  lparams.n_threads_batch = 1;
  struct llama_context *lctx = llama_init_from_model(model, lparams);
  if (lctx == NULL) {
    fprintf(stderr, "could not create a context\n");
    return 3;
  }

  mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, rgb);
  const mtmd_bitmap *bitmaps[1] = {bitmap};
  struct mtmd_input_text text = {argv[6], false, true};
  mtmd_input_chunks *chunks = mtmd_input_chunks_init();
  const int32_t tokenized = mtmd_tokenize(ctx, chunks, &text, bitmaps, 1);
  if (tokenized != 0) {
    fprintf(stderr, "mtmd_tokenize returned %d\n", tokenized);
    return 3;
  }

  /* **The reference is the library's own composition.**  This used to hand-roll
   * the three batches, and it was 10.5% of the logit spread away from
   * `mtmd_helper_eval_chunks` for the same prompt -- the text chunks went in
   * through `llama_batch_get_one`, which does not start at `n_past`, and the
   * image chunk's multi-section positions were laid out as
   * `pos[i + section * n_tokens]` (right for an embeddings batch, wrong for the
   * token batches either side of it).  An oracle that disagrees with the
   * reference by 10% is not an oracle; the helper *is* the reference's own flow,
   * so it is what this calls.  The per-chunk memory query stays, because the
   * layout facts below are the ones the composition has to reproduce. */
  llama_pos n_past = 0;
  int64_t n_tokens = 0;
  const size_t n_chunks = mtmd_input_chunks_size(chunks);
  const int32_t evaluated = mtmd_helper_eval_chunks(ctx, lctx, chunks, /*n_past=*/0, /*seq_id=*/0,
                                                   /*n_batch=*/4096, /*logits_last=*/true,
                                                   &n_past);
  if (evaluated != 0) {
    fprintf(stderr, "mtmd_helper_eval_chunks returned %d\n", evaluated);
    return 3;
  }
  n_tokens = (int64_t) mtmd_helper_get_n_tokens(chunks);

  if (g_dump_embeddings != NULL) {
    /* The tower's features, for a case that wants to feed the engine the
     * *reference's own* image embeddings and so take the tower out of the
     * comparison.  `mtmd_encode_chunk` is the public entry point the helper uses
     * internally, and calling it after the evaluation is harmless: it re-encodes
     * the tower and rewrites the same buffer. */
    for (size_t i = 0; i < n_chunks; ++i) {
      const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
      if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        continue;
      }
      if (mtmd_encode_chunk(ctx, chunk) != 0) {
        fprintf(stderr, "mtmd_encode_chunk returned non-zero\n");
        return 3;
      }
      const float *embd = mtmd_get_output_embd(ctx);
      const int32_t n_image = (int32_t) mtmd_image_tokens_get_n_tokens(
          mtmd_input_chunk_get_tokens_image(chunk));
      const int32_t width_in = (int32_t) llama_model_n_embd_inp(llama_get_model(lctx));
      FILE *out = fopen(g_dump_embeddings, "wb");
      if (out == NULL) {
        perror(g_dump_embeddings);
        return 2;
      }
      /* llama.cpp's own order: one token's row interleaves the deepstack levels,
       * which is the transpose of the engine's plane-major layout. */
      const size_t count = (size_t) n_image * (size_t) width_in;
      if (fwrite(embd, sizeof(float), count, out) != count) {
        fprintf(stderr, "short write to %s\n", g_dump_embeddings);
        return 2;
      }
      fclose(out);
    }
  }

  /* The cache state the composition has to reproduce, measured rather than
   * reasoned about.  It is the finding that a composition error hides in: after
   * the image chunk `n_past` has advanced by the span's 24 positions while the
   * cache's own `pos_max` has moved by ONE -- the 576 span tokens share the slot
   * at `n_past`, because they all carry the same `t` and llama.cpp's slot search
   * finds the cell already holding that position.  An engine that appends 576
   * rows instead is not off by an offset, it is a different computation. */
  {
    llama_memory_t mem = llama_get_memory(lctx);
    fprintf(stderr, "final: n_past=%d pos_min=%d pos_max=%d\n", (int) n_past,
            (int) llama_memory_seq_pos_min(mem, 0), (int) llama_memory_seq_pos_max(mem, 0));
  }

  const float *logits = llama_get_logits_ith(lctx, -1);
  if (logits == NULL) {
    fprintf(stderr, "llama.cpp produced no logits\n");
    return 3;
  }
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  /* `n_pos` is what the *span* consumed in positions rather than in tokens, and it
   * is the number the text after it has to continue from.  Reported beside the
   * token count because the two differ by a factor of twenty-four here, and a
   * composition that conflated them would still tokenize the prompt correctly. */
  llama_pos span_pos = 0;
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      span_pos = mtmd_input_chunk_get_n_pos(chunk);
    }
  }
  printf("{\"n_vocab\": %d, \"n_tokens\": %lld, \"n_pos\": %d, \"chunks\": %zu, \"logits\": [",
         n_vocab, (long long) n_tokens, (int) span_pos, n_chunks);
  for (int32_t i = 0; i < n_vocab; ++i) {
    printf("%s%.9g", i ? ", " : "", (double) logits[i]);
  }
  printf("]}\n");
  fflush(stdout);
  return 0;
}
