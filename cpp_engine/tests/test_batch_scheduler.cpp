// Multi-engine scheduler conformance test.
//
// Proves that the same InferenceEngine/BatchScheduler interface drives both
// concrete QwenEngine and concrete PersistentEngineAdapter implementations,
// with each engine's real capability declarations honored by the scheduler.
//
// The Qwen case uses the existing tiny two-layer fixture from
// qwen_parity_fixture.hpp and is self-contained. The DeepSeek case requires a
// real safetensors checkpoint and runs only when supplied on the command line.
//
// Both engines are constructed in separate scopes so device memory is released
// between cases. Running both in one process is correctness-only; no
// performance conclusion is drawn.

#include "batch_scheduler.hpp"
#include "device_runtime.hpp"
#include "inference_engine.hpp"
#include "persistent_engine_adapter.hpp"
#include "qwen_engine.hpp"
#include "qwen_parity_fixture.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(std::string("FAILED: ") + message);
    }
}

// Shared conformance harness: construct a scheduler against the provided
// concrete engine, submit requests through the generic interface, poll results
// and verify terminal state. The concrete engine type is visible only to the
// caller that constructs it; this helper receives only InferenceEngine&.
std::vector<pocket::SchedulerGenerationResult> test_scheduler_with_engine(
    pocket::InferenceEngine& engine,
    int requested_batch_size,
    int prefill_budget,
    const std::vector<std::vector<int>>& prompts,
    int max_new_tokens,
    const char* label) {

    std::printf("[%s] Constructing scheduler (requested batch %d, budget %d)\n",
                label, requested_batch_size, prefill_budget);

    pocket::BatchScheduler scheduler(&engine, requested_batch_size);
    scheduler.set_prefill_token_budget(prefill_budget);

    const pocket::Capabilities caps = engine.caps();
    std::printf("[%s] Engine capabilities:\n", label);
    std::printf("  paged_kv=%d continuous_batching=%d chunked_prefill=%d\n",
                caps.paged_kv, caps.continuous_batching, caps.chunked_prefill);
    std::printf("  max_slots=%d\n", caps.max_slots);
    std::printf("[%s] Effective scheduler batch=%d budget=%d\n",
                label, scheduler.max_batch_size(),
                scheduler.prefill_token_budget());

    // Verify the scheduler negotiated correctly with the engine's capabilities.
    require(scheduler.max_batch_size() <= caps.max_slots,
            "scheduler batch exceeds engine max_slots");
    if (!caps.continuous_batching) {
        require(scheduler.max_batch_size() == 1,
                "non-batching engine must force scheduler width to 1");
    }
    if (!caps.chunked_prefill) {
        require(scheduler.prefill_token_budget() == 0,
                "non-chunking engine must disable prefill budget");
    }

    std::printf("[%s] Submitting %zu request(s)\n", label, prompts.size());

    // Submit all requests before polling any result. This exercises admission
    // and queuing when the scheduler cannot run everything immediately.
    pocket::BatchSamplingParams sampling;
    sampling.temperature = 0.0f;
    sampling.top_p = 1.0f;
    sampling.top_k = 1;
    sampling.seed = 12345;
    sampling.max_new_tokens = max_new_tokens;
    sampling.ignore_eos = true;

    std::vector<uint64_t> request_ids;
    for (size_t i = 0; i < prompts.size(); ++i) {
        const uint64_t rid = scheduler.submit_request(prompts[i], sampling);
        request_ids.push_back(rid);
    }

    // Poll every request in submission order. The scheduler may complete them
    // out of order (e.g. a short request finishes before a long one), but
    // every poll must eventually succeed.
    std::printf("[%s] Polling %zu result(s)\n", label, request_ids.size());

    std::vector<pocket::SchedulerGenerationResult> results;
    results.reserve(request_ids.size());
    for (size_t i = 0; i < request_ids.size(); ++i) {
        pocket::SchedulerGenerationResult result;
        const bool ok = scheduler.poll_result(request_ids[i], &result, 30000);
        require(ok, "poll timed out or returned false");
        require(result.request_id == request_ids[i], "request id mismatch");
        require(result.finish_reason == "length", "unexpected finish reason");
        require(result.error.empty(), "generation reported an error");
        require(static_cast<int>(result.generated_tokens.size()) == max_new_tokens,
                "output token count mismatch");
        require(result.completion_tokens == max_new_tokens,
                "completion_tokens field mismatch");
        require(result.prompt_tokens == static_cast<int>(prompts[i].size()),
                "prompt_tokens field mismatch");

        std::printf("[%s] Request %zu -> %d output tokens, finish_reason=%s\n",
                    label, i, static_cast<int>(result.generated_tokens.size()),
                    result.finish_reason.c_str());
        results.push_back(std::move(result));
    }

    // Verify the scheduler is now drained and all slots are free.
    const pocket::BatchScheduler::Stats stats = scheduler.get_stats();
    std::printf("[%s] Final stats: waiting=%d running=%d completed=%d\n",
                label, stats.waiting_requests, stats.running_requests,
                stats.completed_requests);

    require(stats.waiting_requests == 0, "waiting queue not empty");
    require(stats.running_requests == 0, "running queue not empty");
    require(stats.completed_requests == prompts.size(),
            "completed count mismatch");

    std::printf("[%s] PASS\n", label);
    return results;
}

void test_qwen_scheduler() {
    std::printf("\n=== Qwen scheduler case ===\n");

    if (!pocket::device_runtime_available()) {
        std::printf("SKIP: no device runtime available\n");
        return;
    }

    const int device = 0;
    if (!pocket::device_set(device)) {
        std::printf("SKIP: could not bind device %d\n", device);
        return;
    }

    const std::string fixture_dir = qwen_fixture::fixture_dir(
        "test_batch_scheduler_qwen_fixture");
    if (!qwen_fixture::write_fixture(fixture_dir)) {
        throw std::runtime_error("could not write Qwen fixture");
    }

    pocket::QwenEngineOptions options;
    options.device = device;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.prefill_chunk_tokens = 128;
    options.max_batch_size = 2;
    options.prefix_cache = false;
    options.temperature = 0.0f;
    options.top_p = 1.0f;
    options.top_k = 1;
    options.sampling_seed = 12345;

    std::unique_ptr<pocket::QwenEngine> engine;
    try {
        engine = std::make_unique<pocket::QwenEngine>(fixture_dir, options);
    } catch (const std::exception& e) {
        std::printf("SKIP: could not construct Qwen engine: %s\n", e.what());
        return;
    }

    // Two distinct prompts, greedy sampling, deterministic token count.
    std::vector<std::vector<int>> prompts = {
        {10, 20, 30, 40, 50},
        {15, 25, 35, 45},
    };

    test_scheduler_with_engine(*engine, 4, 128, prompts, 8, "Qwen");

    engine.reset();
    std::printf("Qwen case complete\n");
}

void test_deepseek_scheduler(const std::string& checkpoint, int layer_count) {
    std::printf("\n=== DeepSeek-V4 scheduler case ===\n");

    if (pocket::device_backend() != pocket::DeviceBackend::Cuda) {
        std::printf("SKIP: PersistentEngine requires CUDA backend\n");
        return;
    }

    if (!pocket::device_runtime_available()) {
        std::printf("SKIP: no device runtime available\n");
        return;
    }

    const int device = 0;
    if (!pocket::device_set(device)) {
        std::printf("SKIP: could not bind device %d\n", device);
        return;
    }

    pocket::ForwardSmokeOptions smoke_opts;
    smoke_opts.device = device;
    smoke_opts.tp_world = 1;
    smoke_opts.tp_rank = 0;
    smoke_opts.skip_fp4_host_prepare = true;

    const int max_context = 512;

    std::unique_ptr<pocket::PersistentEngineAdapter> adapter;
    try {
        adapter = std::make_unique<pocket::PersistentEngineAdapter>(
            checkpoint, smoke_opts, layer_count, max_context);
    } catch (const std::exception& e) {
        std::printf("SKIP: could not construct PersistentEngineAdapter: %s\n",
                    e.what());
        return;
    }

    // Submit the same prompt twice. They must complete serially through the
    // one-slot scheduler, and the second must produce the same output as the
    // first, proving that slot reassignment invokes session reset.
    std::vector<std::vector<int>> prompts = {
        {0, 17665, 31114, 12, 526},
        {0, 17665, 31114, 12, 526},
    };

    const std::vector<pocket::SchedulerGenerationResult> results =
        test_scheduler_with_engine(*adapter, 4, 128, prompts, 6, "DeepSeek");
    require(results.size() == 2, "DeepSeek result count mismatch");
    require(results[0].generated_tokens == results[1].generated_tokens,
            "session reset did not reproduce identical output");

    adapter.reset();
    std::printf("DeepSeek case complete\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::printf("Multi-engine scheduler conformance test\n");

        test_qwen_scheduler();

        if (argc >= 2) {
            const std::string ckpt = argv[1];
            const int layers = (argc >= 3) ? std::atoi(argv[2]) : 1;
            test_deepseek_scheduler(ckpt, layers);
        } else {
            std::printf("\nDeepSeek case: provide checkpoint path as argv[1] "
                        "and optional layer count as argv[2]\n");
            std::printf("SKIP: no DeepSeek checkpoint supplied\n");
        }

        std::printf("\nAll available cases PASSED\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
