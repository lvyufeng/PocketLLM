// Runs one DSpark stage's MoE FFN on inputs dumped by PyTorch and writes the
// result back out, so tests/test_dspark_moe_parity.py can diff the two.
//
// The MoE is where a draft goes wrong quietly. Routing is discrete: pick the
// wrong expert and the output is still a well-scaled vector of plausible
// numbers, just the wrong one. The parts that can each be wrong on their own
// are the gate's score function (sqrt-softplus, not softmax), the fact that
// topk selects on score+bias but renormalizes the *unbiased* scores, the
// route_scale factor, the expert-major grouping of routes, and the shared
// expert that is added on top of all of it.
//
// Binary layout (little-endian, host float32) for both files:
//   in:   int32 magic=0x44534b4d, int32 stage_id, int32 rows, int32 dim,
//         float x[rows*dim]
//   out:  float out[rows*dim]
#include "dspark.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMagic = 0x44534b4d;  // "DSKM"

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

    if (raw.size() < 16 || take_i32() != kMagic) {
        std::fprintf(stderr, "bad magic in %s\n", in_path);
        return 2;
    }
    const int stage_id = take_i32();
    const int rows = take_i32();
    const int dim = take_i32();

    const size_t x_n = static_cast<size_t>(rows) * dim;
    const size_t need = 16 + x_n * sizeof(float);
    if (raw.size() != need) {
        std::fprintf(stderr, "size mismatch in %s: have %zu want %zu\n",
                     in_path, raw.size(), need);
        return 2;
    }

    const float* h_x = reinterpret_cast<const float*>(p);

    dspark::DSparkEngine engine(ckpt, tp_rank, tp_world);
    const dspark::Config& cfg = engine.config();
    if (cfg.dim != dim) {
        std::fprintf(stderr, "geometry mismatch: file dim=%d engine dim=%d\n", dim, cfg.dim);
        return 2;
    }
    if (rows > cfg.block_size) {
        std::fprintf(stderr, "rows=%d exceeds block_size=%d\n", rows, cfg.block_size);
        return 2;
    }

    std::vector<float> out(x_n);
    engine.debug_moe(stage_id, h_x, rows, out.data());

    std::FILE* f = std::fopen(out_path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);

    std::printf("stage=%d rows=%d wrote %zu floats to %s\n", stage_id, rows, out.size(), out_path);
    return 0;
}
