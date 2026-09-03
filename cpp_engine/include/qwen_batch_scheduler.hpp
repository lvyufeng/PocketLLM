#pragma once

#include "qwen_engine.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dsv4 {

// Result returned to the caller after generation completes
struct SchedulerGenerationResult {
    uint64_t request_id = 0;
    std::vector<int> generated_tokens;
    std::string finish_reason;  // "stop" or "length"
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double total_seconds = 0.0;
    double ttft_seconds = 0.0;  // Time to first token
};

// Internal request wrapper with scheduling metadata
struct SchedulerRequest {
    uint64_t request_id = 0;
    std::vector<int> prompt_tokens;
    QwenBatchSamplingParams sampling;
    int slot_id = -1;
    int seq_len = 0;  // Tokens processed (prompt + generated)
    bool finished = false;
    int last_token = 0;
    std::vector<int> generated_tokens;

    // Timing
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point completion_time;

    // Result notification
    std::function<void(const SchedulerGenerationResult&)> callback;
    bool callback_invoked = false;
};

// Continuous batching scheduler for QwenEngine
//
// This scheduler runs a background thread that:
// 1. Admits waiting requests when slots are available
// 2. Separates requests into prefill vs decode batches
// 3. Calls engine->batch_prefill() for new requests
// 4. Calls engine->batch_decode_step() for running requests
// 5. Handles completions and frees slots
//
// Thread-safe: submit_request/cancel_request can be called concurrently
class QwenBatchScheduler {
public:
    explicit QwenBatchScheduler(QwenEngine* engine, int max_batch_size);
    ~QwenBatchScheduler();

    // No copy/move
    QwenBatchScheduler(const QwenBatchScheduler&) = delete;
    QwenBatchScheduler& operator=(const QwenBatchScheduler&) = delete;

    // Submit a new generation request
    // Returns request_id (> 0) on success, 0 on failure
    // The callback will be invoked from the scheduler thread when generation completes
    uint64_t submit_request(
        const std::vector<int>& prompt_tokens,
        const QwenBatchSamplingParams& sampling,
        std::function<void(const SchedulerGenerationResult&)> callback = nullptr);

    // Cancel a pending or running request
    // Returns true if the request was found and marked for cancellation
    bool cancel_request(uint64_t request_id);

    // Blocking poll for a request result (for sync API)
    // Returns true if result was populated, false on timeout
    bool poll_result(uint64_t request_id, SchedulerGenerationResult* out,
                     int timeout_ms = 30000);

    // Get scheduler statistics
    struct Stats {
        int waiting_requests = 0;
        int running_requests = 0;
        int completed_requests = 0;
        int cancelled_requests = 0;
        int free_slots = 0;
    };
    Stats get_stats() const;

    // Check if scheduler is running
    bool is_running() const { return running_.load(); }

    // Stop the scheduler (blocks until thread exits)
    void stop();

private:
    // Main scheduling loop (runs in background thread)
    void schedule_loop();

    // Admit new requests from waiting queue
    void admit_requests();

    // Run prefill batch
    void run_prefill_batch();

    // Run decode batch
    void run_decode_batch();

    // Handle completed requests
    void handle_completions();

    // Notify result (invoke callback or store for poll)
    void notify_result(SchedulerRequest* req);

    // Internal state
    QwenEngine* engine_;
    int max_batch_size_;
    std::atomic<uint64_t> next_request_id_{1};

    // Request queues (protected by queue_mutex_)
    mutable std::mutex queue_mutex_;
    std::queue<std::unique_ptr<SchedulerRequest>> waiting_queue_;
    std::unordered_map<int, std::unique_ptr<SchedulerRequest>> slot_to_request_;
    std::unordered_map<uint64_t, int> request_id_to_slot_;

    // Cancelled requests (protected by queue_mutex_)
    std::unordered_set<uint64_t> cancelled_requests_;

    // Completed results for polling (protected by results_mutex_)
    mutable std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::unordered_map<uint64_t, SchedulerGenerationResult> completed_results_;

    // Scheduler thread
    std::atomic<bool> running_{true};
    std::thread schedule_thread_;

    // Statistics
    std::atomic<int> total_completed_{0};
    std::atomic<int> total_cancelled_{0};
};

}  // namespace dsv4
