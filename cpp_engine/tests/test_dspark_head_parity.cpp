// Runs the DSpark last stage's output heads on inputs dumped by PyTorch and
// writes the drafted tokens, biased logits, and confidences back out, so
// tests/test_dspark_head_parity.py can diff them.
//
// The head is where a wrong draft becomes a *cheap* wrong draft rather than an
// obviously broken one. Every part of it produces a plausible token id:
//   - hc_head collapses [block_size, hc, dim] with its own gate, which is a
//     different parameterization from hc_pre (4 mixes, one shared scale, no
//     post/comb), so reusing hc_pre here would still emit tokens
//   - the loop is sequential: position i's markov bias is a lookup on the token
//     argmaxed at position i-1, so dropping the bias, or computing all the
//     biases from the input token, both still emit tokens
//   - markov_w2 is stored [vocab, rank] and used untransposed; the transpose
//     would only be caught by the numbers, not by a shape check
// The Python side therefore compares the token ids exactly, not just the norms.
//
// Binary layout (little-endian, host float32) for both files:
//   in:   int32 magic=0x44534b48, int32 input_token, int32 rows,
//         int32 hc, int32 dim, float x[rows*hc*dim]
//   out:  int32 tokens[rows+1], float confidence[rows], float logits[rows*vocab]
#include "dspark.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMagic = 0x44534b48;  // "DSKH"

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

    if (raw.size() < 20 || take_i32() != kMagic) {
        std::fprintf(stderr, "bad magic in %s\n", in_path);
        return 2;
    }
    const int input_token = take_i32();
    const int rows = take_i32();
    const int hc = take_i32();
    const int dim = take_i32();

    const size_t x_n = static_cast<size_t>(rows) * hc * dim;
    const size_t need = 20 + x_n * sizeof(float);
    if (raw.size() != need) {
        std::fprintf(stderr, "size mismatch in %s: have %zu want %zu\n",
                     in_path, raw.size(), need);
        return 2;
    }
    const float* h_x = reinterpret_cast<const float*>(p);

    dspark::DSparkEngine engine(ckpt, tp_rank, tp_world);
    const dspark::Config& cfg = engine.config();
    if (cfg.dim != dim || cfg.hc_mult != hc) {
        std::fprintf(stderr, "geometry mismatch: file dim=%d hc=%d engine dim=%d hc=%d\n",
                     dim, hc, cfg.dim, cfg.hc_mult);
        return 2;
    }
    if (rows != cfg.block_size) {
        std::fprintf(stderr, "rows=%d must equal block_size=%d\n", rows, cfg.block_size);
        return 2;
    }

    std::vector<int> tokens(static_cast<size_t>(rows) + 1);
    std::vector<float> confidence(static_cast<size_t>(rows));
    std::vector<float> logits(static_cast<size_t>(rows) * cfg.vocab_size);
    engine.debug_head(h_x, input_token, tokens.data(), confidence.data(), logits.data());

    std::FILE* f = std::fopen(out_path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    std::fwrite(tokens.data(), sizeof(int32_t), tokens.size(), f);
    std::fwrite(confidence.data(), sizeof(float), confidence.size(), f);
    std::fwrite(logits.data(), sizeof(float), logits.size(), f);
    std::fclose(f);

    std::printf("input_token=%d drafted=", input_token);
    for (size_t i = 1; i < tokens.size(); ++i) std::printf("%d%s", tokens[i],
                                                           i + 1 == tokens.size() ? "" : ",");
    std::printf(" wrote %s\n", out_path);
    return 0;
}
