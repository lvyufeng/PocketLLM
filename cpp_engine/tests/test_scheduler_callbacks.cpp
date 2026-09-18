// Scheduler result delivery, streaming token callbacks, and generation bounds.
//
// This is a host-only conformance test: the fake engine reports device == -1,
// so it needs no checkpoint and no accelerator. That is deliberate -- the
// callback contract belongs to the scheduler, not to any one model runtime.

#include "batch_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << "  " << message << ": " << (condition ? "PASS" : "FAIL") << "\n";
    if (!condition) ++failures;
}

bool is_stop(const pocket::BatchSamplingParams& sampling, int token) {
    if (sampling.ignore_eos) return false;
    return std::find(sampling.stop_token_ids.begin(),
                     sampling.stop_token_ids.end(), token) !=
           sampling.stop_token_ids.end();
}

// Predicts token 10 from every prompt, then 11, 12, ... on decode. This makes
// the exact callback sequence readable in the assertions below.
class CountingEngine final : public pocket::InferenceEngine {
public:
    pocket::Capabilities caps() const override {
        pocket::Capabilities c;
        c.continuous_batching = true;
        c.chunked_prefill = true;
        c.max_slots = 4;
        c.per_request_sampling = true;
        c.per_request_top_k = true;
        return c;
    }

    int max_context() const override { return 4096; }
    int device() const override { return -1; }

    void allocate_batch_slots(int max_batch_size) override {
        slots_.assign(static_cast<size_t>(max_batch_size), 0);
    }

    int allocate_slot(uint64_t request_id) override {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i] == 0) {
                slots_[i] = request_id;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void free_slot(uint64_t request_id) override {
        for (uint64_t& owner : slots_) {
            if (owner == request_id) owner = 0;
        }
    }

    bool kv_paged() const override { return false; }
    int kv_free_blocks() const override { return 0; }
    int kv_total_blocks() const override { return 0; }
    int kv_blocks_for_tokens(int) const override { return 0; }

    pocket::BatchPrefillResult batch_prefill(
        const std::vector<pocket::BatchedRequest*>& requests,
        int /*token_budget*/) override {
        ++prefill_calls;
        if (fail_prefill) throw std::runtime_error("synthetic prefill failure");
        pocket::BatchPrefillResult out;
        for (pocket::BatchedRequest* req : requests) {
            pocket::ForwardResult result;
            result.top_token = 10;
            req->seq_len = static_cast<int>(req->prompt_tokens.size());
            req->finished = is_stop(req->sampling, result.top_token);
            out.results.push_back(result);
            out.incomplete.push_back(false);
            out.total_tokens += req->seq_len;
        }
        return out;
    }

    pocket::BatchDecodeResult batch_decode_step(
        const std::vector<pocket::BatchedRequest*>& requests) override {
        ++decode_calls;
        if (fail_decode) throw std::runtime_error("synthetic decode failure");
        pocket::BatchDecodeResult out;
        for (pocket::BatchedRequest* req : requests) {
            const int token = req->last_token + 1;
            if (emit_multi_token) {
                // Deliberately overproduce one token so the scheduler must apply
                // the request's remaining output budget to a multi-token row.
                out.next_tokens.push_back(token + 1);
                out.finished.push_back(false);
                out.hit_stop_token.push_back(false);
                out.emitted_tokens.push_back({token, token + 1});
                out.position_advances.push_back(2);
                out.proposed_drafts.push_back(2);
                out.accepted_drafts.push_back(1);
                out.used_speculative.push_back(true);
                out.rolled_back.push_back(true);
            } else {
                const bool stopped = is_stop(req->sampling, token);
                out.next_tokens.push_back(token);
                out.finished.push_back(stopped);
                out.hit_stop_token.push_back(stopped);
            }
        }
        return out;
    }

    std::atomic<int> prefill_calls{0};
    std::atomic<int> decode_calls{0};
    bool fail_prefill = false;
    bool fail_decode = false;
    bool emit_multi_token = false;

private:
    std::vector<uint64_t> slots_;
};

struct CallbackState {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::vector<int> tokens;
    std::vector<std::string> events;
    pocket::SchedulerGenerationResult result;
};

bool wait_done(CallbackState& state) {
    std::unique_lock<std::mutex> lock(state.mutex);
    return state.cv.wait_for(lock, std::chrono::seconds(2), [&] { return state.done; });
}

uint64_t submit_with_callbacks(pocket::BatchScheduler& scheduler,
                               const pocket::BatchSamplingParams& sampling,
                               CallbackState& state) {
    return scheduler.submit_request(
        {1, 2, 3}, sampling,
        [&](const pocket::SchedulerGenerationResult& result) {
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.result = result;
                state.events.push_back("complete");
                state.done = true;
            }
            state.cv.notify_one();
        },
        [&](uint64_t, int token) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.tokens.push_back(token);
            state.events.push_back("token:" + std::to_string(token));
        });
}

void test_prefill_token_counts_toward_limit() {
    std::cout << "prefill's sampled token counts as generation token one\n";
    CountingEngine engine;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 1;
    CallbackState state;
    const uint64_t id = submit_with_callbacks(scheduler, sampling, state);

    check(id != 0, "request is accepted");
    check(wait_done(state), "completion callback arrives");
    check(state.result.generated_tokens == std::vector<int>{10},
          "max_new_tokens=1 returns exactly the prefill token");
    check(state.tokens == std::vector<int>{10},
          "the streaming callback sees exactly one token");
    check(state.events == std::vector<std::string>{"token:10", "complete"},
          "the token callback precedes completion");
    check(engine.decode_calls.load() == 0,
          "no decode forward runs past the one-token limit");

    pocket::SchedulerGenerationResult duplicate;
    check(!scheduler.poll_result(id, &duplicate, 0),
          "a callback result is not duplicated into the poll map");
    scheduler.stop();
}

void test_stop_token_is_not_streamed() {
    std::cout << "stop tokens stay in engine state but not in the stream\n";
    CountingEngine engine;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 8;
    sampling.stop_token_ids = {12};
    CallbackState state;
    submit_with_callbacks(scheduler, sampling, state);

    check(wait_done(state), "stop-token request completes");
    check(state.tokens == std::vector<int>({10, 11}),
          "the per-token callback excludes the stop token");
    check(state.result.generated_tokens == std::vector<int>({10, 11, 12}),
          "the scheduler result retains the stop token for KV consistency");
    check(state.result.finish_reason == "stop",
          "the completion reports stop rather than length");
    check(state.events ==
              std::vector<std::string>({"token:10", "token:11", "complete"}),
          "completion follows the last visible token");
    scheduler.stop();
}

void test_poll_result_without_callback() {
    std::cout << "poll_result remains the completion channel without a callback\n";
    CountingEngine engine;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 2;
    const uint64_t id = scheduler.submit_request({4, 5}, sampling);
    pocket::SchedulerGenerationResult result;

    check(id != 0, "poll-mode request is accepted");
    check(scheduler.poll_result(id, &result, 2000),
          "poll_result receives completion");
    check(result.generated_tokens == std::vector<int>({10, 11}),
          "poll mode observes the exact generation bound");
    check(result.finish_reason == "length", "the length cap is reported");
    check(!scheduler.poll_result(id, &result, 0),
          "poll_result consumes its map entry");
    scheduler.stop();
}

void test_multi_token_row_respects_budget_and_reports_stats() {
    std::cout << "multi-token rows respect the remaining budget and report telemetry\n";
    CountingEngine engine;
    engine.emit_multi_token = true;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 2;
    CallbackState state;
    const uint64_t id = submit_with_callbacks(scheduler, sampling, state);

    check(id != 0, "multi-token request is accepted");
    check(wait_done(state), "multi-token request completes");
    check(state.result.generated_tokens == std::vector<int>({10, 11}),
          "scheduler truncates emitted row at max_new_tokens");
    check(state.tokens == std::vector<int>({10, 11}),
          "truncated row streams only visible tokens");
    check(state.result.proposed_drafts == 2,
          "proposed draft telemetry is retained");
    check(state.result.accepted_drafts == 1,
          "accepted draft telemetry is retained");
    check(state.result.speculative_steps == 1,
          "speculative step telemetry is retained");
    check(state.result.rollback_steps == 1,
          "rollback telemetry is retained");
    check(state.result.finish_reason == "length",
          "truncated multi-token row finishes by length");
    scheduler.stop();
}

void test_engine_failure_is_terminal() {
    std::cout << "engine failures complete the request instead of retrying forever\n";
    CountingEngine engine;
    engine.fail_decode = true;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 4;
    CallbackState state;
    const uint64_t id = submit_with_callbacks(scheduler, sampling, state);

    check(id != 0, "failing request is initially accepted");
    check(wait_done(state), "the failure reaches the completion callback");
    check(state.tokens == std::vector<int>{10},
          "tokens produced before the failure are still delivered");
    check(state.result.finish_reason == "error",
          "the terminal reason is error");
    check(state.result.error.find("synthetic decode failure") != std::string::npos,
          "the engine exception reaches the caller");
    check(engine.decode_calls.load() == 1,
          "the failed forward is not retried in a loop");
    scheduler.stop();
}

// The phase split is what makes a request's duration attributable: queueing is
// the scheduler's fault, prefill and decode are the engine's. It is measured
// here rather than in the server because the admission instant -- the one that
// ends the queue wait -- happens inside admit_requests() under the queue mutex,
// where no caller can see it.
void test_phase_split_is_reported() {
    std::cout << "a completed request reports a queue/prefill/decode split\n";
    CountingEngine engine;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 4;
    CallbackState state;
    submit_with_callbacks(scheduler, sampling, state);

    check(wait_done(state), "phase-split request completes");
    const pocket::SchedulerGenerationResult& result = state.result;
    check(result.queue_seconds >= 0.0, "the queue wait was observed");
    check(result.prefill_seconds > 0.0, "the prefill interval was observed");
    check(result.decode_seconds > 0.0, "the decode interval was observed");
    // The three intervals are consecutive and cover exactly the total, which is
    // what lets a reader trust them over the `total - ttft` subtraction the
    // completion timestamp cannot support.
    check(std::fabs(result.queue_seconds + result.prefill_seconds +
                    result.decode_seconds - result.total_seconds) < 1e-9,
          "the three phases sum to the reported duration");
    check(std::fabs(result.queue_seconds + result.prefill_seconds -
                    result.ttft_seconds) < 1e-9,
          "time to first token is the queue wait plus the prefill");
    scheduler.stop();
}

// A request the engine failed before it produced anything still waited in the
// queue, so that phase is known while the other two are not. Reporting them as
// 0.0 would put two fake instantaneous samples into the histograms and make a
// request that never started look like the fastest one served.
void test_request_without_a_first_token_reports_only_its_queue() {
    std::cout << "a request that produced no token reports only its queue wait\n";
    CountingEngine engine;
    engine.fail_prefill = true;
    pocket::BatchScheduler scheduler(&engine, 1);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 4;
    CallbackState state;
    submit_with_callbacks(scheduler, sampling, state);

    check(wait_done(state), "the failed request completes");
    const pocket::SchedulerGenerationResult& result = state.result;
    check(result.finish_reason == "error", "the request is reported as failed");
    check(result.queue_seconds >= 0.0, "the queue wait it did serve is reported");
    check(result.prefill_seconds < 0.0,
          "a prefill that never finished its first token is unknown, not zero");
    check(result.decode_seconds < 0.0,
          "a decode interval is unknown when no token was ever produced");
    check(result.ttft_seconds == 0.0, "time to first token stays at its sentinel");
    scheduler.stop();
}

}  // namespace

int main() {
    test_prefill_token_counts_toward_limit();
    test_stop_token_is_not_streamed();
    test_poll_result_without_callback();
    test_multi_token_row_respects_budget_and_reports_stats();
    test_engine_failure_is_terminal();
    test_phase_split_is_reported();
    test_request_without_a_first_token_reports_only_its_queue();

    if (failures != 0) {
        std::cout << "[FAIL] " << failures << " scheduler callback checks failed\n";
        return 1;
    }
    std::cout << "[PASS] scheduler callbacks\n";
    return 0;
}
