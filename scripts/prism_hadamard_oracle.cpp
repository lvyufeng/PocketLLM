// Dump the Prism fork's own Hadamard rotation, for tests/test_prism_hadamard_transform.py.
//
// Ternary-Bonsai-2-27B stores pre-rotated weights and rotates the *activation* at run
// time, through three ops composed in this order (llama-graph.cpp, build_lora_mm):
// an optional feature permute for the GDN output, a multiply by the sign vector, and
// a matmul against the rotation matrix with GGML_HINT_SRC0_IS_HADAMARD, which the CPU
// backend replaces with its FWHT kernel.  That composition is what the fixture has to
// pin, and this program is the fork's own code producing it -- ggml's permute, ggml's
// mul, ggml's fwht -- so a misreading of ggml's ne/nb conventions on our side shows up
// as a mismatch rather than as agreement with our own mistake.
//
// Build (the fork's own build tree supplies the libraries):
//
//   g++ -O2 -std=c++17 -I/mnt/data1/llama_cpp_prism/ggml/include \
//       scripts/prism_hadamard_oracle.cpp -o /tmp/prism_hadamard_oracle \
//       -L/mnt/data1/llama_cpp_prism/build-sm75/bin -lggml -lggml-base -lggml-cpu \
//       -Wl,-rpath,/mnt/data1/llama_cpp_prism/build-sm75/bin -pthread
//
// Usage: prism_hadamard_oracle <in.f32> <out.f32> <width> <rows> <mode> [signs.f32]
//   mode = fwht     -- the rotation alone
//        = signs    -- signs then the rotation
//        = inverse  -- rotation then signs (the token_embd path)
//        = permute  -- the GDN permute alone
//        = gdn      -- permute, signs, rotation (what a folded ssm_out.weight sees)
//   <signs.f32> supplies the width-long sign vector; without it the program uses a
//   deterministic square wave.
//   <in.f32> is <rows> x <width> fp32 little-endian, row-major, and <width> is the
//   activation's own width (6144 for the gated-DeltaNet output).  The rotation's
//   block size is this checkpoint's 1024, which is what the fork's `rot` tensor is
//   square in; the permute modes take their geometry from width: head_dim 128,
//   groups 16, rep width/128/16.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static std::vector<float> read_f32(const char * path, size_t count) {
    std::vector<float> data(count);
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    if (fread(data.data(), sizeof(float), count, f) != count) {
        fprintf(stderr, "%s is shorter than %zu floats\n", path, count);
        exit(2);
    }
    fclose(f);
    return data;
}

// The fork's rotation matrix, built exactly as llama-model.cpp builds it: an
// N x N fp32 tensor with H[i][j] = (-1)^popcount(i & j) / sqrt(N).
static struct ggml_tensor * build_rotation(struct ggml_context * ctx, int n) {
    struct ggml_tensor * rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_set_name(rot, "prism.hadamard.rot");
    return rot;
}

static void fill_rotation(struct ggml_tensor * rot, int n) {
    std::vector<float> data((size_t) n * n);
    const float scale = 1.0f / sqrtf((float) n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            uint32_t parity = (uint32_t) (row & col);
            parity ^= parity >> 16; parity ^= parity >> 8; parity ^= parity >> 4;
            parity ^= parity >> 2;  parity ^= parity >> 1;
            data[(size_t) row * n + col] = (parity & 1) ? -scale : scale;
        }
    }
    ggml_backend_tensor_set(rot, data.data(), 0, data.size() * sizeof(float));
}

// llama_mul_mat_hadamard(): flatten to [n, rows], matmul, mark the hint, restore shape.
static struct ggml_tensor * mul_mat_hadamard(struct ggml_context * ctx, struct ggml_tensor * cur, struct ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    struct ggml_tensor * res = ggml_is_contiguous(cur)
        ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n)
        : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

int main(int argc, char ** argv) {
    CHECK(argc == 6 || argc == 7,
          "usage: %s <in.f32> <out.f32> <width> <rows> <mode> [signs.f32]", argv[0]);
    const char * in_path = argv[1];
    const char * out_path = argv[2];
    const int width = atoi(argv[3]);
    const int rows = atoi(argv[4]);
    const std::string mode = argv[5];
    const char * signs_path = argc == 7 ? argv[6] : nullptr;
    const int block = 1024;  // this checkpoint's prism.hadamard.block_size
    CHECK(width % block == 0, "width %d is not a whole number of %d-wide blocks", width, block);

    std::vector<float> input = read_f32(in_path, (size_t) width * rows);

    const int64_t hd  = 128;
    const int64_t nk  = 16;
    const int64_t rep = width / hd / nk;
    const bool  gdn   = mode == "permute" || mode == "gdn";

    struct ggml_init_params params = {
        /*.mem_size   =*/ (size_t) 4096 * 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,  // tensors are allocated from the backend below
    };
    struct ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init failed");

    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend != nullptr, "no CPU backend");

    struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, 1, 1, rows);
    ggml_set_name(x, "x");

    struct ggml_tensor * cur = x;
    if (gdn) {
        // The fork's permute, verbatim, including the cont() around it: a permuted view
        // is not contiguous and the reshape back would otherwise be wrong.
        const int64_t ne1 = cur->ne[1], ne2 = cur->ne[2], ne3 = cur->ne[3];
        struct ggml_tensor * p = ggml_reshape_4d(ctx, cur, hd, nk, rep, ne1 * ne2 * ne3);
        p = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));
        cur = ggml_reshape_4d(ctx, p, hd * nk * rep, ne1, ne2, ne3);
    }
    struct ggml_tensor * rot = nullptr;
    struct ggml_tensor * signs = nullptr;
    if (mode != "permute") {
        signs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        ggml_set_name(signs, "prism.hadamard.signs");
        if (mode == "signs" || mode == "gdn") {
            // forward: the signs come first, then the rotation
            cur = ggml_mul(ctx, cur, signs);
        }
        rot = build_rotation(ctx, block);
        cur = mul_mat_hadamard(ctx, cur, rot);
        if (mode == "inverse") {
            // token_embd: the rotation is undone after the lookup, so H first and the
            // signs after -- the opposite order from every other folded weight.
            cur = ggml_mul(ctx, cur, signs);
        }
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cur);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    CHECK(buffer != nullptr, "no backend buffer");

    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (rot != nullptr) { fill_rotation(rot, block); }
    if (signs != nullptr) {
        if (signs_path != nullptr) {
            std::vector<float> given = read_f32(signs_path, (size_t) width);
            ggml_backend_tensor_set(signs, given.data(), 0, given.size() * sizeof(float));
        } else {
            // Deterministic pseudo-signs: a fixture can pin the fork's *composition*
            // with any vector, and a square wave over the index makes a transposed or
            // shifted consumption obvious.
            std::vector<float> sign_values((size_t) width);
            for (int i = 0; i < width; ++i) {
                sign_values[(size_t) i] = ((i / 37) % 2 == 0) ? 1.0f : -1.0f;
            }
            ggml_backend_tensor_set(signs, sign_values.data(), 0, sign_values.size() * sizeof(float));
        }
    }

    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");

    std::vector<float> out((size_t) width * rows);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));

    FILE * f = fopen(out_path, "wb");
    CHECK(f != nullptr, "cannot write %s", out_path);
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
