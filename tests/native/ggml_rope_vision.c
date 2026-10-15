/* `ggml_rope_multi` in the vision mode, as the oracle for the tower's rotary.
 *
 * The one piece of the tower this engine derives from reading llama.cpp rather
 * than from a tensor's shape: the rotary embedding takes four positions per
 * token, applies two of them (a patch's row and its column) to the two halves of
 * a half-width rotation, and restarts each half's frequency ramp at its own
 * section boundary.  Every part of that is a place to be wrong in a way that
 * still produces a fluent-looking embedding, and the tower-level comparison
 * cannot see it -- the reference's own f16 activations are a wider difference
 * than this would be.
 *
 * So it is checked the way the patch order was: against the library.  This calls
 * `ggml_rope_multi` with `GGML_ROPE_TYPE_VISION` on a caller-supplied tensor and
 * the positions clip.cpp builds, and the test drives this engine's own `rope` op
 * over a matching table.  The agreement is float32 rounding, not a tolerance.
 *
 * Compiled by `tests/native/ggml_rope_vision.py`, the way the other llama.cpp
 * instruments here are.
 *
 * Usage:
 *   ggml-rope-vision <x.bin> <grid_side> <merge> <n_heads> <head_dim>
 *
 * `x.bin` is `grid_side^2 * n_heads * head_dim` float32 in ggml's own order --
 * `head_dim` fastest, then head, then token -- which is the layout the engine's
 * `rope` op takes too.  The token grid is `grid_side` square, the patches are
 * merged `merge` at a time, and the positions are built the way clip.cpp builds
 * them: four per token, a row and a column repeated, in the order the merge loop
 * walks the merged grid.  The rotated tensor goes to stdout in the same order.
 */

#include <math.h>
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
  if (argc != 6) {
    fprintf(stderr, "usage: %s <x.bin> <grid_side> <merge> <n_heads> <head_dim>\n", argv[0]);
    return 2;
  }
  const int grid = atoi(argv[2]);
  const int merge = atoi(argv[3]);
  const int n_heads = atoi(argv[4]);
  const int head_dim = atoi(argv[5]);
  const int n_pos = grid * grid;
  if (grid <= 0 || merge <= 0 || grid % merge != 0 || head_dim % 4 != 0) {
    fprintf(stderr, "grid, merge, head_dim out of range\n");
    return 2;
  }

  long bytes = 0;
  const float *input = read_file(argv[1], &bytes);
  const long expected = (long) n_pos * n_heads * head_dim * 4;
  if (bytes != expected) {
    fprintf(stderr, "x.bin is %ld bytes, and %dx%dx%d float32 is %ld\n", bytes, n_pos, n_heads,
            head_dim, expected);
    return 2;
  }

  struct ggml_init_params params = {64 * 1024 * 1024, NULL, false};
  struct ggml_context *ctx = ggml_init(params);
  struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_pos);
  struct ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) n_pos * 4);
  memcpy(x->data, input, (size_t) expected);

  /* clip.cpp's loop, verbatim in shape: for each merged cell, the four patches
   * in (dy, dx) order, with the row and the column written twice so that the
   * four components llama.cpp's multi-section rotary reads are (t, h, w, e) =
   * (row, column, row, column).  Only the first two are ever reached at this head
   * width -- the rotation is half the head and the sections are quarter-head
   * sized -- and writing all four is what the reference does. */
  int32_t *pos = (int32_t *) positions->data;
  int at = 0;
  for (int y = 0; y < grid; y += merge) {
    for (int x = 0; x < grid; x += merge) {
      for (int dy = 0; dy < merge; ++dy) {
        for (int dx = 0; dx < merge; ++dx) {
          pos[at] = y + dy;
          pos[n_pos + at] = x + dx;
          pos[2 * n_pos + at] = y + dy;
          pos[3 * n_pos + at] = x + dx;
          at++;
        }
      }
    }
  }
  int sections[4] = {head_dim / 4, head_dim / 4, head_dim / 4, head_dim / 4};
  struct ggml_tensor *out =
      ggml_rope_multi(ctx, x, positions, NULL, head_dim / 2, sections, GGML_ROPE_TYPE_VISION, 32768,
                      10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

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
  free((void *) input);
  return 0;
}
