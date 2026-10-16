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

/* One image chunk, as an embeddings batch: llama.cpp's own layout, one position
 * per rotary section, and the non-causal flag the chunk asks for. */
static int decode_image_chunk(mtmd_context *ctx, struct llama_context *lctx,
                              const mtmd_input_chunk *chunk, llama_pos *n_past) {
  const mtmd_image_tokens *image = mtmd_input_chunk_get_tokens_image(chunk);
  if (image == NULL) {
    fprintf(stderr, "the tokenizer produced an image chunk with no tokens\n");
    return 1;
  }
  if (mtmd_encode(ctx, image) != 0) {
    fprintf(stderr, "mtmd_encode failed\n");
    return 1;
  }
  float *embd = mtmd_get_output_embd(ctx);
  if (embd == NULL) {
    fprintf(stderr, "mtmd_get_output_embd returned null\n");
    return 1;
  }
  const int32_t n_tokens = (int32_t) mtmd_image_tokens_get_n_tokens(image);
  fprintf(stderr, "image chunk: n_tokens=%d n_pos=%d mrope=%d non_causal=%d\n", n_tokens,
          (int) mtmd_input_chunk_get_n_pos(chunk), (int) mtmd_decode_use_mrope(ctx),
          (int) mtmd_decode_use_non_causal(ctx, chunk));
  for (int32_t i = 0; i < 3; ++i) {
    const struct mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(image, *n_past, (size_t) i);
    fprintf(stderr, "  span[%d] = (t %d, y %d, x %d, z %d)\n", i, (int) p.t, (int) p.y, (int) p.x,
            (int) p.z);
  }
  const int32_t n_embd = llama_model_n_embd_inp(llama_get_model(lctx));
  if (g_dump_embeddings != NULL) {
    FILE *out = fopen(g_dump_embeddings, "wb");
    if (out == NULL) {
      perror(g_dump_embeddings);
      return 1;
    }
    const size_t count = (size_t) n_tokens * (size_t) n_embd;
    if (fwrite(embd, sizeof(float), count, out) != count) {
      fprintf(stderr, "short write to %s\n", g_dump_embeddings);
      return 1;
    }
    fclose(out);
  }
  const int32_t n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

  const size_t sections = (size_t) n_pos_per_embd;
  llama_pos *pos = malloc(sizeof(llama_pos) * (size_t) n_tokens * sections);
  int32_t *n_seq_id = malloc(sizeof(int32_t) * (size_t) n_tokens);
  llama_seq_id *seq_id_0 = malloc(sizeof(llama_seq_id));
  llama_seq_id **seq_ids = malloc(sizeof(llama_seq_id *) * (size_t) n_tokens);
  int8_t *logit_flags = calloc((size_t) n_tokens, sizeof(int8_t));
  if (pos == NULL || n_seq_id == NULL || seq_id_0 == NULL || seq_ids == NULL ||
      logit_flags == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  seq_id_0[0] = 0;
  for (int32_t i = 0; i < n_tokens; ++i) {
    const struct mtmd_decoder_pos decoded =
        mtmd_image_tokens_get_decoder_pos(image, *n_past, (size_t) i);
    const llama_pos values[4] = {decoded.t, decoded.y, decoded.x, decoded.z};
    for (int32_t section = 0; section < n_pos_per_embd; ++section) {
      pos[(size_t) section * n_tokens + i] = values[section];
    }
    n_seq_id[i] = 1;
    seq_ids[i] = seq_id_0;
    logit_flags[(size_t) i] = (i == n_tokens - 1) ? 1 : 0;
  }
  struct llama_batch batch = {0};
  batch.n_tokens = n_tokens;
  batch.embd = embd;
  batch.pos = pos;
  batch.n_seq_id = n_seq_id;
  batch.seq_id = seq_ids;
  batch.logits = logit_flags;

  const bool non_causal = mtmd_decode_use_non_causal(ctx, chunk);
  if (non_causal) {
    llama_set_causal_attn(lctx, false);
  }
  const int32_t decoded_status = llama_decode(lctx, batch);
  if (non_causal) {
    llama_set_causal_attn(lctx, true);
  }
  if (decoded_status != 0) {
    fprintf(stderr, "llama_decode failed on the image chunk\n");
    free(pos);
    free(n_seq_id);
    free(seq_id_0);
    free(seq_ids);
    free(logit_flags);
    return 1;
  }
  free(pos);
  free(n_seq_id);
  free(seq_id_0);
  free(seq_ids);
  free(logit_flags);
  *n_past += mtmd_input_chunk_get_n_pos(chunk);
  return 0;
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
  lparams.n_threads = 1;
  lparams.n_threads_batch = 1;
  lparams.offload_kqv = false;
  lparams.no_perf = true;
  lparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  lparams.type_k = GGML_TYPE_F32;
  lparams.type_v = GGML_TYPE_F32;
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

  llama_pos n_past = 0;
  int64_t n_tokens = 0;
  const size_t n_chunks = mtmd_input_chunks_size(chunks);
  for (size_t i = 0; i < n_chunks; ++i) {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (decode_image_chunk(ctx, lctx, chunk, &n_past) != 0) {
        return 3;
      }
      n_tokens += (int64_t) mtmd_input_chunk_get_n_tokens(chunk);
    } else {
      size_t n_text = 0;
      const llama_token *tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text);
      fprintf(stderr, "text chunk %zu:", i);
      for (size_t t = 0; t < n_text; ++t) {
        fprintf(stderr, " %d", (int) tokens[t]);
      }
      fprintf(stderr, "\n");
      struct llama_batch text_batch =
          llama_batch_get_one((llama_token *) tokens, (int32_t) n_text);
      if (llama_decode(lctx, text_batch) != 0) {
        fprintf(stderr, "llama_decode failed on a text chunk\n");
        return 3;
      }
      n_past += (llama_pos) n_text;
      n_tokens += (int64_t) n_text;
    }
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
