// Accepted-prefix transaction test for PersistentEngine::batch_verify_step.
//
// Batched continuation GEMMs are not numerically interchangeable with repeated
// single-token decode, so plain decode is not a valid transaction oracle. This
// test instead compares two full batches with the same accepted prefix and two
// different rejected suffixes. After finalize, both sessions must expose the
// same committed hidden rows and produce the same continuation tokens.
//
//   test_batch_verify_transaction <ckpt_dir> [layers=43] [steps=8] [rows=6]
//                                 [prompt_len=6] [tp_world=1] [tp_rank=0]
//                                 [nccl_id_path]

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

std::string join(const std::vector<int>& values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out += " ";
        out += std::to_string(values[i]);
    }
    return out;
}

int accepted_rows(const std::vector<int>& block,
                  const std::vector<int>& next) {
    int committed = 1;
    while (committed < static_cast<int>(block.size()) &&
           next[static_cast<size_t>(committed - 1)] ==
               block[static_cast<size_t>(committed)]) {
        ++committed;
    }
    return committed;
}

std::vector<int> make_prompt(int prompt_len) {
    const int even_len = prompt_len % 2 == 0 ? prompt_len : prompt_len + 1;
    std::vector<int> prompt;
    prompt.reserve(static_cast<size_t>(even_len));
    for (int i = 0; i < even_len; ++i) {
        prompt.push_back((i & 1) == 0 ? 16 : 18);
    }
    return prompt;
}

int reset_and_prefill(PersistentEngine& engine,
                      const std::vector<int>& prompt,
                      const SamplingParams& sp) {
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(prompt);
    return engine.prefill(prompt, sp);
}

struct RunResult {
    int committed_rows = 0;
    std::vector<int> continuation;
    std::vector<float> committed_hidden;
};

RunResult run_variant(PersistentEngine& engine,
                      const std::vector<int>& prompt,
                      const std::vector<int>& block,
                      int expected_committed_rows,
                      int steps,
                      const SamplingParams& sp) {
    RunResult result;
    const int seed = reset_and_prefill(engine, prompt, sp);
    if (seed != block.front()) {
        fail("prefill seed changed while replaying transaction fixture");
        return result;
    }

    const int start_position = static_cast<int>(prompt.size());
    const std::vector<int> next =
        engine.batch_verify_step(block, start_position, sp);
    result.committed_rows = accepted_rows(block, next);
    result.committed_hidden = engine.last_verify_dspark_hidden();

    if (result.committed_rows != expected_committed_rows) {
        fail("transaction committed " + std::to_string(result.committed_rows) +
             " rows, expected " + std::to_string(expected_committed_rows));
        return result;
    }

    int token = next[static_cast<size_t>(result.committed_rows - 1)];
    int position = start_position + result.committed_rows;
    result.continuation.reserve(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        engine.worker_command_decode(token, position);
        token = engine.decode_step(token, position, sp);
        result.continuation.push_back(token);
        ++position;
    }
    return result;
}

int different_token(int token, int salt) {
    constexpr int kVocabSize = 163840;
    int candidate = (token + 1009 + salt * 7919) % kVocabSize;
    if (candidate == token) candidate = (candidate + 1) % kVocabSize;
    return candidate;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [layers=43] [steps=8] [rows=6] [prompt_len=6]"
                     " [tp_world=1] [tp_rank=0] [nccl_id_path]\n";
        return 2;
    }

    const std::string ckpt_dir = argv[1];
    const int layer_count = argc > 2 ? std::atoi(argv[2]) : 43;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 8;
    const int rows = argc > 4 ? std::atoi(argv[4]) : 6;
    const int prompt_len = argc > 5 ? std::atoi(argv[5]) : 6;
    if (layer_count < 1 || steps < 1 || rows < 1 || rows > 8 || prompt_len < 1) {
        std::cerr << "layers, steps, and prompt_len must be positive; rows must be 1..8\n";
        return 2;
    }

    ForwardSmokeOptions opts;
    opts.tp_world = argc > 6 ? std::atoi(argv[6]) : 1;
    opts.tp_rank = argc > 7 ? std::atoi(argv[7]) : 0;
    opts.device = opts.tp_rank;
    if (argc > 8) opts.nccl_id_path = argv[8];
    if (opts.tp_world > 1 && opts.nccl_id_path.empty()) {
        std::cerr << "tp_world > 1 needs an nccl_id_path\n";
        return 2;
    }
    const bool verbose = opts.tp_rank == 0;

    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::cerr << "failed to set CUDA device " << opts.device << "\n";
        return 2;
    }

    PersistentEngine engine(ckpt_dir, opts, layer_count,
                            std::max(2048, prompt_len + rows + steps + 8));
    engine.set_dspark_capture_layers({layer_count - 1});
    engine.warmup_tp();
    if (opts.tp_rank != 0) {
        engine.run_worker_loop();
        return 0;
    }

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    const std::vector<int> prompt = make_prompt(prompt_len);
    const int start_position = static_cast<int>(prompt.size());

    // The alternating fixture is the same near-certain continuation used by the
    // scheduler test. It avoids calibrating the chain with several extra full
    // model forwards, which would make this already expensive TP test impractical.
    std::vector<int> accepted_block(static_cast<size_t>(rows), 0);
    accepted_block[0] = reset_and_prefill(engine, prompt, sp);
    for (int row = 1; row < rows; ++row) {
        const int previous = accepted_block[static_cast<size_t>(row - 1)];
        accepted_block[static_cast<size_t>(row)] = previous == 16 ? 18 : 16;
    }

    // Confirm the fixture reaches full acceptance before using prefixes of it as
    // the transaction oracle.
    (void)reset_and_prefill(engine, prompt, sp);
    std::vector<int> calibrated_next =
        engine.batch_verify_step(accepted_block, start_position, sp);
    int calibrated_rows = accepted_rows(accepted_block, calibrated_next);
    if (rows > 1 && calibrated_rows != rows) {
        const int seed = reset_and_prefill(engine, prompt, sp);
        accepted_block.assign(static_cast<size_t>(rows), 0);
        accepted_block[0] = seed;
        const std::vector<int> calibration_next =
            engine.batch_verify_step(accepted_block, start_position, sp);
        accepted_block[1] = calibration_next[0];
        for (int row = 2; row < rows; ++row) {
            const int previous = accepted_block[static_cast<size_t>(row - 1)];
            accepted_block[static_cast<size_t>(row)] = previous == 16 ? 18 : 16;
        }
        (void)reset_and_prefill(engine, prompt, sp);
        calibrated_next = engine.batch_verify_step(
            accepted_block, start_position, sp);
        calibrated_rows = accepted_rows(accepted_block, calibrated_next);
    }
    if (calibrated_rows != rows) {
        fail("calibrated block accepted " + std::to_string(calibrated_rows) +
             " rows, expected " + std::to_string(rows));
    }
    if (verbose) {
        std::cout << "prompt_len=" << prompt.size() << " calibrated="
                  << join(accepted_block) << "\n";
    }

    const int first_committed = rows == 1 ? 1 : 2;
    for (int committed = first_committed; committed <= rows; ++committed) {
        std::vector<int> block_a = accepted_block;
        std::vector<int> block_b = accepted_block;
        if (committed < rows) {
            const int expected = accepted_block[static_cast<size_t>(committed)];
            block_a[static_cast<size_t>(committed)] = different_token(expected, committed);
            block_b[static_cast<size_t>(committed)] = different_token(expected, committed + rows);
            for (int row = committed + 1; row < rows; ++row) {
                block_a[static_cast<size_t>(row)] =
                    different_token(accepted_block[static_cast<size_t>(row)], row + 17);
                block_b[static_cast<size_t>(row)] =
                    different_token(accepted_block[static_cast<size_t>(row)], row + 41);
            }
        }

        const RunResult a = run_variant(engine, prompt, block_a, committed,
                                        steps, sp);
        const RunResult b = run_variant(engine, prompt, block_b, committed,
                                        steps, sp);

        if (a.committed_hidden.size() != b.committed_hidden.size()) {
            fail("committed_rows=" + std::to_string(committed) +
                 " exposed inconsistent hidden prefix lengths");
        }
        if (!a.committed_hidden.empty() &&
            a.committed_hidden.size() % static_cast<size_t>(committed) != 0) {
            fail("committed_rows=" + std::to_string(committed) +
                 " exposed a non-integral hidden prefix");
        }
        if (a.committed_hidden != b.committed_hidden &&
            a.continuation != b.continuation) {
            const auto hidden_mismatch = std::mismatch(
                a.committed_hidden.begin(), a.committed_hidden.end(),
                b.committed_hidden.begin());
            const size_t index = static_cast<size_t>(
                hidden_mismatch.first - a.committed_hidden.begin());
            float max_abs = 0.0f;
            for (size_t i = 0; i < a.committed_hidden.size(); ++i) {
                max_abs = std::max(max_abs,
                                   std::abs(a.committed_hidden[i] - b.committed_hidden[i]));
            }
            fail("committed_rows=" + std::to_string(committed) +
                 " changed committed hidden rows at value " +
                 std::to_string(index) + " (max_abs=" +
                 std::to_string(max_abs) + ")");
        }
        if (verbose && a.committed_hidden != b.committed_hidden) {
            float max_abs = 0.0f;
            for (size_t i = 0; i < a.committed_hidden.size(); ++i) {
                max_abs = std::max(max_abs,
                                   std::abs(a.committed_hidden[i] - b.committed_hidden[i]));
            }
            std::cout << "  committed hidden self-spread max_abs=" << max_abs
                      << " (continuation token-equivalent)\n";
        }
        if (a.continuation != b.continuation) {
            const auto mismatch = std::mismatch(
                a.continuation.begin(), a.continuation.end(),
                b.continuation.begin());
            const int index = static_cast<int>(mismatch.first - a.continuation.begin());
            fail("committed_rows=" + std::to_string(committed) +
                 " continuation diverged at token " + std::to_string(index));
        }

        if (verbose) {
            std::cout << "committed_rows=" << committed
                      << " hidden_values=" << a.committed_hidden.size()
                      << " continuation=" << join(a.continuation)
                      << (a.continuation == b.continuation ? " [match]" : " [DIVERGES]")
                      << "\n";
        }
    }

    engine.worker_command_shutdown();
    if (failures != 0) {
        std::cout << "[FAIL] batch_verify_transaction: " << failures
                  << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] batch_verify_transaction (accepted-prefix rollback verified)\n";
    return 0;
}
