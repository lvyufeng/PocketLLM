// Does the DSpark draft produce the same tokens at TP=1 and TP=4?
//
// The end-to-end accept rate is 0 (docs/dspark.md), and the sub-path parity
// tests all run at TP=1 from injected inputs, so two suspects are still open:
// the composition (stage 0 -> blocks -> head) and the TP sharding of the draft
// itself. This test settles the second one, which is the cheaper half: the
// draft alone is ~12.4 GB and fits on one card without the main model, so it
// can be run at both world sizes on identical input.
//
// The seed is synthetic and deterministic rather than a real capture -- what is
// being compared is one sharding against another, and a fixed vector removes
// the main model, the capture, and the checkpoint's own hidden distribution
// from the comparison. Both world sizes must see bit-identical input for the
// diff to mean anything, so the hidden is generated from the position index
// alone.
//
// The ring is primed the same way the real loop primes it, because the TP
// all-reduces sit inside the block forward that write_main_kv also drives: an
// unprimed run would compare two drafts that both attend to zeros and would
// agree for the wrong reason.
//
//   test_dspark_tp_consistency <ckpt_dir> <out.bin> [tp_world=1] [tp_rank=0] [nccl_id_path]
//
// Writes int32 n_tokens, int32 tokens[n], float confidence[n-1]. Run once per
// world size and diff the files; every rank writes its own, so a rank-dependent
// draft (which the replicated head is supposed to rule out) also shows up.

#include "dspark.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Deterministic stand-in for a captured hidden. Magnitudes are in the range a
// real capture produces (~1e-1) so the draft is not pushed somewhere the
// weights never see; the exact values do not matter because both sides get the
// same ones.
float seed_value(int pos, int i) {
    return 0.1f * std::sin(0.017f * static_cast<float>(i) + 0.31f * static_cast<float>(pos));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <ckpt_dir> <out.bin> [tp_world=1] [tp_rank=0]"
                     " [nccl_id_path]\n", argv[0]);
        return 2;
    }
    const char* ckpt = argv[1];
    const char* out_path = argv[2];
    const int tp_world = argc > 3 ? std::atoi(argv[3]) : 1;
    const int tp_rank = argc > 4 ? std::atoi(argv[4]) : 0;
    const char* nccl_id_path = argc > 5 ? argv[5] : nullptr;

    if (tp_world > 1 && nccl_id_path == nullptr) {
        std::fprintf(stderr, "tp_world > 1 needs an nccl_id_path\n");
        return 2;
    }
    // One rank per card, as main.cpp defaults. Omitting this puts every rank on
    // GPU 0 and OOMs after two.
    if (cudaSetDevice(tp_rank) != cudaSuccess) {
        std::fprintf(stderr, "failed to set CUDA device %d\n", tp_rank);
        return 2;
    }

    dspark::DSparkEngine engine(ckpt, tp_rank, tp_world, tp_rank, nccl_id_path);
    const dspark::Config& cfg = engine.config();
    const int n_target = static_cast<int>(cfg.target_layer_ids.size());
    const size_t stride = static_cast<size_t>(n_target) * cfg.dim;

    // Prime a short run of committed positions 0..rows-1, then draft from the
    // last of them -- the same shape of call the real loop makes after a
    // prefill of `rows` tokens.
    const int rows = 6;
    std::vector<float> hidden(static_cast<size_t>(rows) * stride);
    for (int p = 0; p < rows; ++p) {
        for (size_t i = 0; i < stride; ++i) {
            hidden[static_cast<size_t>(p) * stride + i] = seed_value(p, static_cast<int>(i));
        }
    }
    engine.write_main_kv(hidden.data(), rows, 0);

    // start_pos is the seed hidden's position, i.e. rows - 1; the committed
    // token lands at rows and is roped there.
    std::vector<float*> per_layer;
    float* d_seed = nullptr;
    if (cudaMalloc(&d_seed, stride * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc seed failed\n");
        return 2;
    }
    cudaMemcpy(d_seed, hidden.data() + static_cast<size_t>(rows - 1) * stride,
               stride * sizeof(float), cudaMemcpyHostToDevice);
    for (int k = 0; k < n_target; ++k) {
        per_layer.push_back(d_seed + static_cast<size_t>(k) * cfg.dim);
    }

    const int committed_token = 5654;
    const dspark::DraftOutput out = engine.draft(committed_token, rows - 1, per_layer);
    cudaFree(d_seed);

    std::printf("rank %d/%d tokens:", tp_rank, tp_world);
    for (int t : out.tokens) std::printf(" %d", t);
    std::printf("\n  confidence:");
    for (float c : out.confidence) std::printf(" %.6f", c);
    std::printf("\n");

    std::FILE* f = std::fopen(out_path, "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    const int32_t n = static_cast<int32_t>(out.tokens.size());
    std::fwrite(&n, sizeof(int32_t), 1, f);
    std::vector<int32_t> toks(out.tokens.begin(), out.tokens.end());
    std::fwrite(toks.data(), sizeof(int32_t), toks.size(), f);
    std::fwrite(out.confidence.data(), sizeof(float), out.confidence.size(), f);
    std::fclose(f);
    return 0;
}
