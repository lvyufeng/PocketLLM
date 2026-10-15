/* Run llama.cpp's own 2-D convolution, as the outside authority for `conv2d`.
 *
 * `test_op_conformance.py` compares the C kernel against the numpy reference,
 * and that comparison cannot see a layout error the two agree on: the same
 * gather index written the same wrong way in both implementations is one
 * opinion stated twice.  So the patch embedding's index order -- which is the
 * whole content of a non-overlapping convolution -- is checked against the
 * library the checkpoint's weights were converted for.  `ggml_conv_2d` *is* the
 * function llama.cpp builds the tower's patch embedding with, and this tool
 * calls it on the tensor's own bytes.
 *
 * Compiled by `tests/native/ggml_conv.py`, the way `ggml_dequant.c` is, so the
 * library is a test dependency and not one of the deliverable's.
 *
 * Usage:
 *   ggml-conv <image.bin> <in_ch> <height> <width> \
 *             <weight.bin> <f16|f32> <kernel> <stride> <pad>
 *
 * `image.bin` is float32 little-endian, `width` fastest; `weight.bin` is the
 * GGUF tensor's payload verbatim, `kernel x kernel x in_ch x out_ch` with the
 * output channel outermost, which is the order `ggml_new_tensor_4d` reads it
 * in.  The output goes to stdout as float32 in ggml's own order -- `out_w`
 * fastest, then `out_h`, then `out_ch` -- and the caller transposes the patch
 * axis, because the engine's `conv2d` is patch-major and this tool reports what
 * ggml produced rather than what the test wishes it had.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
  if (argc != 10) {
    fprintf(stderr,
            "usage: %s <image.bin> <in_ch> <height> <width> <weight.bin> <f16|f32> "
            "<kernel> <stride> <pad>\n",
            argv[0]);
    return 2;
  }
  long image_bytes = 0;
  long weight_bytes = 0;
  const float *image = read_file(argv[1], &image_bytes);
  const int in_ch = atoi(argv[2]);
  const int height = atoi(argv[3]);
  const int width = atoi(argv[4]);
  void *weight = read_file(argv[5], &weight_bytes);
  const int is_f16 = strcmp(argv[6], "f16") == 0;
  const int kernel = atoi(argv[7]);
  const int stride = atoi(argv[8]);
  const int pad = atoi(argv[9]);

  if ((long) (sizeof(float) * (size_t) in_ch * (size_t) height * (size_t) width) != image_bytes) {
    fprintf(stderr, "the image file is not in_ch * height * width floats\n");
    return 2;
  }
  const long per_channel = (long) kernel * (long) kernel * (long) in_ch;
  const long elem = is_f16 ? 2 : 4;
  if (per_channel == 0 || weight_bytes % (per_channel * elem) != 0) {
    fprintf(stderr, "the weight file is not a whole number of kernel^2 * in_ch x out_ch rows\n");
    return 2;
  }
  const int out_ch = (int) (weight_bytes / (per_channel * elem));

  /* One arena for the tensors, the graph and ggml's own im2col scratch. Sized
   * for the tower's real geometry with room to spare: 768x768x3 floats in, a
   * (48, 48, 1024) float result, and the f16 im2col between them. */
  struct ggml_init_params params = {256 * 1024 * 1024, NULL, false};
  struct ggml_context *ctx = ggml_init(params);
  if (ctx == NULL) {
    fprintf(stderr, "ggml_init failed\n");
    return 2;
  }

  struct ggml_tensor *w = ggml_new_tensor_4d(ctx, is_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, kernel,
                                             kernel, in_ch, out_ch);
  struct ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, in_ch, 1);
  struct ggml_tensor *out = ggml_conv_2d(ctx, w, b, stride, stride, pad, pad, 1, 1);
  memcpy(w->data, weight, (size_t) weight_bytes);
  memcpy(b->data, image, (size_t) image_bytes);

  struct ggml_cgraph *graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, out);

  ggml_backend_t backend = ggml_backend_cpu_init();
  if (backend == NULL) {
    fprintf(stderr, "ggml_backend_cpu_init failed\n");
    return 3;
  }
  const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
  ggml_backend_free(backend);
  if (status != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "ggml_backend_graph_compute returned %d\n", (int) status);
    return 3;
  }

  const int64_t count = ggml_nelements(out);
  if (fwrite(out->data, sizeof(float), (size_t) count, stdout) != (size_t) count) {
    return 2;
  }
  ggml_free(ctx);
  free(weight);
  free((void *) image);
  return 0;
}
