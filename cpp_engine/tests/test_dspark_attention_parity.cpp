// Runs one DSpark stage's attention on inputs dumped by PyTorch and writes the
// result back out, so tests/test_dspark_attention_parity.py can diff the two.
//
// This exists because "it compiles and the weights load" says nothing about
// whether the attention math is right: the draft reads its KV from the main
// model's hidden, uses a ring cache, and applies an inverse rope on the way
// out. Any one of those being wrong still produces plausible-looking numbers.
//
// Binary layout (little-endian, host float32) for both files:
//   in:   int32 magic=0x44534b41, int32 stage_id, int32 start_pos,
//         int32 block_size, int32 dim, int32 window_size, int32 head_dim,
//         float x[block_size*dim], float main_x[dim],
//         float kv_cache[window_size*head_dim]
//   out:  float out[block_size*dim]
#include "dspark.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMagic = 0x44534b41;  // "DSKA"

std::vector<char> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(static_cast<size_t>(n));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fprintf(stderr, "short read on %s\n", path);
        std::exit(2);
    }
    std::fclose(f);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <checkpoint_dir> <input.bin> <output.bin> [tp_rank] [tp_world]\n",
                    argv[0]);
        return 2;
    }
    const char* ckpt = argv[1];
    const char* in_path = argv[2];
    const char* out_path = argv[3];
    const int tp_rank = argc > 4 ? std::atoi(argv[4]) : 0;
    const int tp_world = argc > 5 ? std::atoi(argv[5]) : 1;

    std::vector<char> raw = read_file(in_path);
    const char* p = raw.data();
    auto take_i32 = [&p]() { int32_t v; std::memcpy(&v, p, 4); p += 4; return v; };

    if (raw.size() < 28 || take_i32() != kMagic) {
        std::fprintf(stderr, "bad magic in %s\n", in_path);
        return 2;
    }
    const int stage_id = take_i32();
    const int start_pos = take_i32();
    const int block_size = take_i32();
    const int dim = take_i32();
    const int window_size = take_i32();
    const int head_dim = take_i32();

    const size_t x_n = static_cast<size_t>(block_size) * dim;
    const size_t cache_n = static_cast<size_t>(window_size) * head_dim;
    const size_t need = 28 + (x_n + dim + cache_n) * sizeof(float);
    if (raw.size() != need) {
        std::fprintf(stderr, "size mismatch in %s: have %zu want %zu\n",
                     in_path, raw.size(), need);
        return 2;
    }

    const float* h_x = reinterpret_cast<const float*>(p);
    const float* h_main_x = h_x + x_n;
    const float* h_cache = h_main_x + dim;

    dspark::DSparkEngine engine(ckpt, tp_rank, tp_world);
    const dspark::Config& cfg = engine.config();
    if (cfg.block_size != block_size || cfg.dim != dim ||
        cfg.window_size != window_size || cfg.head_dim != head_dim) {
        std::fprintf(stderr,
                     "geometry mismatch: file(bs=%d dim=%d win=%d hd=%d) "
                     "engine(bs=%d dim=%d win=%d hd=%d)\n",
                     block_size, dim, window_size, head_dim,
                     cfg.block_size, cfg.dim, cfg.window_size, cfg.head_dim);
        return 2;
    }

    engine.debug_set_kv_cache(stage_id, h_cache);
    std::vector<float> out(x_n);
    engine.debug_attention(stage_id, h_x, h_main_x, start_pos, out.data());

    std::FILE* f = std::fopen(out_path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);

    std::printf("stage=%d start_pos=%d wrote %zu floats to %s\n",
                stage_id, start_pos, out.size(), out_path);
    return 0;
}
