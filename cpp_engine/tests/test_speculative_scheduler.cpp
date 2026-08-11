// End-to-end test of the speculative scheduler: does a multi-round speculative
// loop produce the same tokens as plain decode, with fewer total model calls?
//
// The scheduler's job is to call draft() + verify() in a loop and advance the
// position by (accepted prefix + bonus token) each round, relying on overwrite-
// in-place for rejected drafts. This test confirms:
//
//   1. speculative_step() produces correct tokens (exact match vs decode_step)
//   2. The speculative path makes fewer model forward calls
//   3. Multi-round works (the hidden capture and ring priming wire together)
//
// It does NOT measure end-to-end tok/s -- that needs a production serving loop
// with real request batching and is deferred to a separate benchmark.
//
//   test_speculative_scheduler <ckpt_dir> [rounds=8] [layers=43] [tp_world=1] [tp_rank=0] [nccl_id_path]

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace dsv4;

namespace {

int failures = 0;

void fail(const std::string& msg) {
    std::cout << "  [FAIL] " << msg << "\n";
    ++failures;
}

std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += " ";
        s += std::to_string(v[i]);
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [rounds=8] [layers=43] [tp_world=1] [tp_rank=0]"
                     " [nccl_id_path]\n";
        return 2;
    }
    const std::string ckpt_dir = argv[1];
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 8;
    const int layer_count = argc > 3 ? std::atoi(argv[3]) : 43;

    ForwardSmokeOptions opts;
    opts.tp_world = argc > 4 ? std::atoi(argv[4]) : 1;
    opts.tp_rank = argc > 5 ? std::atoi(argv[5]) : 0;
    opts.device = opts.tp_rank;
    if (argc > 6) opts.nccl_id_path = argv[6];
    if (opts.tp_world > 1 && opts.nccl_id_path.empty()) {
        std::cerr << "tp_world > 1 needs an nccl_id_path\n";
        return 2;
    }
    const bool verbose = opts.tp_rank == 0;

    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::cerr << "failed to set CUDA device " << opts.device << "\n";
        return 2;
    }

    PersistentEngine engine(ckpt_dir, opts, layer_count, 2048);
    engine.load_dspark(ckpt_dir);
    if (!engine.dspark_loaded()) {
        std::cout << "[FAIL] load_dspark did not take effect\n";
        return 1;
    }

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    // A cyclic prompt where the main model is near-certain, so the draft
    // accepts most of its block. This is the same reasoning as test_dspark_draft.
    const std::vector<int> prompt = {16, 18, 16, 18, 16, 18, 16, 18, 16, 18, 16, 18};

    // Reference: plain decode from the same prompt.
    engine.reset_session();
    int token = engine.prefill(prompt, sp);
    std::vector<int> reference;
    int pos = static_cast<int>(prompt.size());
    for (int r = 0; r < rounds; ++r) {
        token = engine.decode_step(token, pos++, sp);
        reference.push_back(token);
    }
    if (verbose) {
        std::cout << "reference (plain decode): " << join(reference) << "\n";
    }

    // Speculative path: same prompt, speculative_step() instead of decode_step().
    engine.reset_session();
    token = engine.prefill(prompt, sp);
    pos = static_cast<int>(prompt.size());
    std::vector<int> speculative;
    int total_accepted = 0;
    int spec_rounds = 0;
    while (static_cast<int>(speculative.size()) < rounds) {
        const std::vector<int> generated = engine.speculative_step(token, pos, sp);
        if (generated.empty()) {
            fail("speculative_step returned empty vector");
            break;
        }
        ++spec_rounds;
        const int n_generated = static_cast<int>(generated.size());
        total_accepted += n_generated - 1;  // -1 for the bonus token

        // Append the generated tokens to the speculative sequence.
        for (int t : generated) {
            if (static_cast<int>(speculative.size()) < rounds) {
                speculative.push_back(t);
            }
        }

        // Advance position and update the committed token for the next round.
        pos += n_generated;
        token = generated.back();

        if (verbose) {
            std::cout << "  round " << spec_rounds << " pos=" << pos
                      << " generated=" << n_generated << " accepted=" << n_generated - 1 << "\n";
        }
    }

    if (verbose) {
        std::cout << "speculative: " << join(speculative) << "\n";
        std::cout << "\nspec rounds=" << spec_rounds << " accepted=" << total_accepted
                  << " avg=" << (spec_rounds > 0 ? static_cast<double>(total_accepted) / spec_rounds : 0.0)
                  << "\n";
    }

    // Compare token-by-token.
    int first_diff = -1;
    for (int i = 0; i < rounds; ++i) {
        if (speculative[i] != reference[i]) {
            first_diff = i;
            break;
        }
    }
    if (first_diff >= 0) {
        fail("tokens diverged at position " + std::to_string(first_diff) +
             " (spec=" + std::to_string(speculative[first_diff]) +
             " ref=" + std::to_string(reference[first_diff]) + ")");
    }
    if (spec_rounds >= rounds) {
        fail("speculative rounds " + std::to_string(spec_rounds) +
             " >= decode rounds " + std::to_string(rounds) +
             " (no amortization)");
    }

    if (failures != 0) {
        std::cout << "[FAIL] speculative_scheduler: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] speculative_scheduler (position and amortization verified)\n";
    return 0;
}
