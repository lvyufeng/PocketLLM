// Does the DSpark draft actually respond to its inputs?
//
// The end-to-end accept rate is 0 for the real seed and equally 0 for a zeroed
// one (docs/dspark.md), which an accept rate alone cannot explain: it is what
// you would see both if the draft is merely bad and if the seed never reaches
// it. This test asks the sharper question directly -- perturb one input at a
// time and count how many drafted tokens change.
//
// A draft that ignores an input is a wiring bug and localizes immediately; a
// draft that responds to everything and is still wrong is a math bug somewhere
// downstream, which is a different search. So each variant is a claim about one
// edge of the graph:
//
//   seed      a zeroed hidden, ring primed as usual. Weak by construction and
//             kept for exactly that reason: priming already wrote this
//             position's KV, and forward_attention only overwrites that one
//             slot, so the seed controls 1 key out of `rows`. A 0 here is not
//             evidence of a wiring bug -- see seed_only.
//   seed_only a zeroed hidden with the ring zeroed too, so the seed's slot is
//             the only live key. This is the decisive form: if the drafted
//             tokens are still identical, the hidden genuinely never reaches
//             stage 0.
//   seed_pos  the hidden from an earlier position, ring primed. Well-formed and
//             in-distribution, so it separates "reads the hidden" from "reads
//             the *right* hidden".
//   ring      priming skipped entirely. The window is the only thing carrying
//             context beyond the committed token, so a draft insensitive to it
//             is attending to nothing -- the failure priming was meant to fix.
//   token     a different committed token, same everything else. Goes in at
//             draft slot 0 and through the markov bias, so this is the input
//             most certain to matter; it doubles as a control on the test
//             itself, since a draft that ignores *this* is not running.
//
// Runs draft-only at TP=1: the module is ~12.4 GB and fits on one card without
// the main model, so no NCCL and no 167 GB checkpoint load.
//
//   test_dspark_draft_sensitivity <ckpt_dir>

#include "dspark.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& msg) {
    std::printf("  [FAIL] %s\n", msg.c_str());
    ++failures;
}

// Deterministic stand-in for a captured hidden, at roughly the magnitude a real
// capture produces. The exact values do not matter -- every variant is compared
// against the same baseline, so what is measured is the response, not the
// quality.
float seed_value(int pos, int i) {
    return 0.1f * std::sin(0.017f * static_cast<float>(i) + 0.31f * static_cast<float>(pos));
}

int tokens_differing(const std::vector<int>& a, const std::vector<int>& b) {
    int n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) ++n;
    }
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <ckpt_dir>\n", argv[0]);
        return 2;
    }
    const char* ckpt = argv[1];
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "failed to set CUDA device 0\n");
        return 2;
    }

    dspark::DSparkEngine engine(ckpt, 0, 1, 0, nullptr);
    const dspark::Config& cfg = engine.config();
    const int n_target = static_cast<int>(cfg.target_layer_ids.size());
    const size_t stride = static_cast<size_t>(n_target) * cfg.dim;

    const int rows = 6;
    const int committed = 5654;
    std::vector<float> hidden(static_cast<size_t>(rows) * stride);
    for (int p = 0; p < rows; ++p) {
        for (size_t i = 0; i < stride; ++i) {
            hidden[static_cast<size_t>(p) * stride + i] = seed_value(p, static_cast<int>(i));
        }
    }

    float* d_seed = nullptr;
    if (cudaMalloc(&d_seed, stride * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc seed failed\n");
        return 2;
    }
    std::vector<float*> per_layer;
    for (int k = 0; k < n_target; ++k) per_layer.push_back(d_seed + static_cast<size_t>(k) * cfg.dim);

    // One draft with the given seed row, optionally re-priming the ring first.
    // Priming is redone per variant because write_main_kv mutates the ring the
    // previous variant left behind.
    auto run = [&](const float* h_seed, bool prime, int token) {
        if (prime) engine.write_main_kv(hidden.data(), rows, 0);
        cudaMemcpy(d_seed, h_seed, stride * sizeof(float), cudaMemcpyHostToDevice);
        return engine.draft(token, rows - 1, per_layer);
    };

    const float* h_last = hidden.data() + static_cast<size_t>(rows - 1) * stride;
    const float* h_prev = hidden.data() + static_cast<size_t>(rows - 2) * stride;
    const std::vector<float> zeros(stride, 0.0f);

    const dspark::DraftOutput base = run(h_last, true, committed);
    const int n_draft = static_cast<int>(base.tokens.size()) - 1;

    std::printf("baseline tokens:");
    for (int t : base.tokens) std::printf(" %d", t);
    std::printf("\n\nvariant    tokens_changed/%d   tokens\n", n_draft);

    struct Variant {
        const char* name;
        const float* seed;
        bool prime;
        int token;
    };
    const Variant variants[] = {
        {"seed",      zeros.data(), true,  committed},
        {"seed_only", zeros.data(), false, committed},
        {"seed_pos",  h_prev,       true,  committed},
        {"ring",      h_last,       false, committed},
        {"token",     h_last,       true,  committed + 1},
    };
    constexpr int kVariants = 5;

    int changed[kVariants] = {0};
    for (int v = 0; v < kVariants; ++v) {
        // A ring variant has to start from a clean cache, not the one the
        // previous draft primed, or "no priming" silently reuses stale slots
        // and looks like sensitivity. debug_set_kv_cache takes exactly
        // [window_size, head_dim].
        if (!variants[v].prime) {
            const std::vector<float> z(
                static_cast<size_t>(cfg.window_size) * cfg.head_dim, 0.0f);
            for (int s = 0; s < cfg.n_stages; ++s) engine.debug_set_kv_cache(s, z.data());
        }
        const dspark::DraftOutput o = run(variants[v].seed, variants[v].prime,
                                          variants[v].token);
        // Skip slot 0: it is the committed token echoed back, so it differs in
        // the `token` variant by construction and would inflate that count.
        std::vector<int> a(base.tokens.begin() + 1, base.tokens.end());
        std::vector<int> b(o.tokens.begin() + 1, o.tokens.end());
        changed[v] = tokens_differing(a, b);
        std::printf("  %-9s %d/%d              ", variants[v].name, changed[v], n_draft);
        for (int t : b) std::printf(" %d", t);
        std::printf("\n");
    }
    cudaFree(d_seed);

    std::printf("\n");
    // Keyed by name rather than index: the variants are ordered for reading,
    // and a positional check would silently mislabel a failure if that order
    // ever changes.
    auto changed_for = [&](const char* name) {
        for (int v = 0; v < kVariants; ++v) {
            if (std::string(variants[v].name) == name) return changed[v];
        }
        fail(std::string("no variant named ") + name);
        return 0;
    };

    if (changed_for("token") == 0) {
        fail("changing the committed token changes nothing -- the draft is not running");
    }
    // `seed` is deliberately not asserted on: priming already wrote this
    // position's KV and the seed only overwrites one slot of the window, so a 0
    // there is expected rather than diagnostic. seed_only is the real check.
    if (changed_for("seed_only") == 0) {
        fail("a zeroed seed drafts identically even with the ring zeroed -- "
             "the main hidden never reaches stage 0");
    }
    if (changed_for("seed_pos") == 0) {
        fail("a seed from another position drafts identically -- the seed carries no position");
    }
    if (changed_for("ring") == 0) {
        fail("an unprimed ring drafts identically -- the draft is not attending to the window");
    }

    if (failures != 0) {
        std::printf("[FAIL] dspark_draft_sensitivity: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("[PASS] dspark_draft_sensitivity (every input reaches the draft)\n");
    return 0;
}
