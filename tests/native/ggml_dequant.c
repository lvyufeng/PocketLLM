/* Decode GGML blocks with llama.cpp's own decoder.
 *
 * The other half of the codebook story. `pocketllm.quant` reads the tables out
 * of ggml's header and decodes the blocks with numpy; this tool asks the library
 * that *wrote* the format to decode the same bytes, so the two answers can be
 * compared block for block. `tests/native/test_codebook_decoders.py` drives it
 * and `ggml_dequant.py` compiles it, the same shape as the logits oracle.
 *
 * Usage: ggml-dequant <file.ggml-type-id> <blocks.bin>
 *
 * Writes `block_elems` little-endian float32 per block to stdout, in order. The
 * file is read as consecutive blocks, which is how a GGUF tensor's bytes are laid
 * out, so a test can hand it a tensor's payload verbatim.
 */

#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <ggml-type-id> <blocks.bin>\n", argv[0]);
    return 2;
  }
  const int type_id = atoi(argv[1]);
  FILE *in = fopen(argv[2], "rb");
  if (in == NULL) {
    perror(argv[2]);
    return 2;
  }
  if (fseek(in, 0, SEEK_END) != 0) {
    return 2;
  }
  const long size = ftell(in);
  if (size < 0 || fseek(in, 0, SEEK_SET) != 0) {
    return 2;
  }
  unsigned char *blocks = malloc((size_t) size);
  if (blocks == NULL || fread(blocks, 1, (size_t) size, in) != (size_t) size) {
    fprintf(stderr, "failed to read %s\n", argv[2]);
    return 2;
  }
  fclose(in);

  const struct ggml_type_traits *traits = ggml_get_type_traits((enum ggml_type) type_id);
  if (traits == NULL || traits->to_float == NULL) {
    fprintf(stderr, "llama.cpp has no decoder for GGML type %d\n", type_id);
    return 3;
  }
  const long count = size / (long) traits->type_size;
  float *values = malloc(sizeof(float) * (size_t) traits->blck_size);
  if (values == NULL) {
    return 2;
  }
  for (long block = 0; block < count; ++block) {
    traits->to_float(blocks + (size_t) block * traits->type_size, values, traits->blck_size);
    if (fwrite(values, sizeof(float), (size_t) traits->blck_size, stdout) !=
        (size_t) traits->blck_size) {
      return 2;
    }
  }
  free(values);
  free(blocks);
  return 0;
}
