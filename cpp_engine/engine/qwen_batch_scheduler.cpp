#include "qwen_batch_scheduler.hpp"
#include "qwen_engine.hpp"
#include <iostream>
#include <algorithm>

namespace dsv4 {

QwenBatchScheduler::QwenBatchScheduler(QwenEngine* engine, int max_batch_size)
    : engine_(engine), max_batch_size_(max_batch_size) {
    if (!engine_) {
        throw std::invalid_argument("QwenBatchScheduler: engine cannot be null");
    }
    if (max_batch_size_ <= 0) {
        throw std::invalid_argument("QwenBatchScheduler: max_batch_size must be > 0");
    }

    // Allocate batch slots in the engine
    engine_->allocate_batch_slots(max_batch_size_);

    // Start scheduler thread
    schedule_thread_ = std::thread(&QwenBatchScheduler::schedule_loop, this);
}

QwenBatchScheduler::~QwenBatchScheduler() {
    stop();
}

void QwenBatchScheduler::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;  // Already stopped
    }

    // Wait for scheduler thread to exit
    if (schedule_thread_.joinable()) {
        schedule_thread_.join();
    }

    // Clean up remaining requests
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!waiting_queue_.empty()) {
        waiting_queue_.pop();
    }
    for (auto& [slot_id, req] : slot_to_request_) {
        if (req && req->slot_id >= 0) {
            engine_->free_slot(req->request_id);
        }
    }
    slot_to_request_.clear();
    request_id_to_slot_.clear();
}

uint64_t QwenBatchScheduler::submit_request(
    const std::vector<int>& prompt_tokens,
    const QwenBatchSamplingParams& sampling,
    std::function<void(const SchedulerGenerationResult&)> callback) {

    if (prompt_tokens.empty()) {
        return 0;  // Invalid request
    }

    uint64_t request_id = next_request_id_.fetch_add(1);

    auto req = std::make_unique<SchedulerRequest>();
    req->request_id = request_id;
    req->prompt_tokens = prompt_tokens;
    req->sampling = sampling;
    req->callback = std::move(callback);
    req->submit_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        waiting_queue_.push(std::move(req));
    }

    return request_id;
}

bool QwenBatchScheduler::cancel_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    // Check if request is waiting or running
    auto slot_it = request_id_to_slot_.find(request_id);
    if (slot_it == request_id_to_slot_.end()) {
        // Not found in running requests, might be in waiting queue
        // Mark as cancelled so it gets rejected during admission
        cancelled_requests_.insert(request_id);
        return true;
    }

    // Mark running request as cancelled
    cancelled_requests_.insert(request_id);
    return true;
}

bool QwenBatchScheduler::poll_result(
    uint64_t request_id, SchedulerGenerationResult* out, int timeout_ms) {

    if (!out) {
        return false;
    }

    auto deadline = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(results_mutex_);

    while (true) {
        // Check if result is ready
        auto it = completed_results_.find(request_id);
        if (it != completed_results_.end()) {
            *out = it->second;
            completed_results_.erase(it);
            return true;
        }

        // Wait with timeout
        if (timeout_ms <= 0) {
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;  // Timeout
        }

        results_cv_.wait_until(lock, deadline);
    }
}

QwenBatchScheduler::Stats QwenBatchScheduler::get_stats() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    Stats stats;
    stats.waiting_requests = static_cast<int>(waiting_queue_.size());
    stats.running_requests = static_cast<int>(slot_to_request_.size());
    stats.completed_requests = total_completed_.load();
    stats.cancelled_requests = total_cancelled_.load();
    stats.free_slots = max_batch_size_ - stats.running_requests;

    return stats;
}

void QwenBatchScheduler::schedule_loop() {
    while (running_.load()) {
        try {
            admit_requests();
            run_prefill_batch();
            run_decode_batch();
            handle_completions();
        } catch (const std::exception& e) {
            std::cerr << "QwenBatchScheduler: exception in schedule loop: "
                     << e.what() << std::endl;
        }

        // Small sleep to avoid busy loop
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void QwenBatchScheduler::admit_requests() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    while (!waiting_queue_.empty() &&
           slot_to_request_.size() < static_cast<size_t>(max_batch_size_)) {

        auto req = std::move(waiting_queue_.front());
        waiting_queue_.pop();

        // Check if cancelled
        if (cancelled_requests_.count(req->request_id)) {
            cancelled_requests_.erase(req->request_id);
            total_cancelled_.fetch_add(1);
            continue;
        }

        // Allocate slot
        int slot_id = engine_->allocate_slot(req->request_id);
        if (slot_id < 0) {
            // No slots available (shouldn't happen due to size check)
            waiting_queue_.push(std::move(req));
            break;
        }

        req->slot_id = slot_id;
        request_id_to_slot_[req->request_id] = slot_id;
        slot_to_request_[slot_id] = std::move(req);
    }
}

void QwenBatchScheduler::run_prefill_batch() {
    std::vector<QwenBatchedRequest*> prefill_batch;
    std::vector<SchedulerRequest*> prefill_requests;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (auto& [slot_id, req] : slot_to_request_) {
            // Check if cancelled
            if (cancelled_requests_.count(req->request_id)) {
                continue;
            }

            // Check if needs prefill (seq_len == 0)
            if (req->seq_len == 0) {
                // Create QwenBatchedRequest wrapper
                auto* batch_req = new QwenBatchedRequest();
                batch_req->request_id = req->request_id;
                batch_req->prompt_tokens = req->prompt_tokens;
                batch_req->slot_id = req->slot_id;
                batch_req->sampling = req->sampling;

                prefill_batch.push_back(batch_req);
                prefill_requests.push_back(req.get());
            }
        }
    }

    if (prefill_batch.empty()) {
        return;
    }

    // Call engine batch_prefill
    try {
        auto result = engine_->batch_prefill(prefill_batch);

        // Update request states
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (size_t i = 0; i < prefill_batch.size(); ++i) {
            auto* req = prefill_requests[i];
            if (i < result.results.size()) {
                req->last_token = result.results[i].top_token;
                req->generated_tokens.push_back(req->last_token);
                req->seq_len = static_cast<int>(req->prompt_tokens.size());

                // Record TTFT
                if (req->first_token_time == std::chrono::steady_clock::time_point{}) {
                    req->first_token_time = std::chrono::steady_clock::now();
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "QwenBatchScheduler: prefill failed: " << e.what() << std::endl;
    }

    // Clean up temporary batch requests
    for (auto* batch_req : prefill_batch) {
        delete batch_req;
    }
}

void QwenBatchScheduler::run_decode_batch() {
    std::vector<QwenBatchedRequest*> decode_batch;
    std::vector<SchedulerRequest*> decode_requests;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (auto& [slot_id, req] : slot_to_request_) {
            // Check if cancelled
            if (cancelled_requests_.count(req->request_id)) {
                continue;
            }

            // Check if in decode phase (seq_len > 0, not finished)
            if (req->seq_len > 0 && !req->finished) {
                // Create QwenBatchedRequest wrapper
                auto* batch_req = new QwenBatchedRequest();
                batch_req->request_id = req->request_id;
                batch_req->slot_id = req->slot_id;
                batch_req->seq_len = req->seq_len;
                batch_req->last_token = req->last_token;
                batch_req->sampling = req->sampling;

                decode_batch.push_back(batch_req);
                decode_requests.push_back(req.get());
            }
        }
    }

    if (decode_batch.empty()) {
        return;
    }

    // Call engine batch_decode_step
    try {
        auto result = engine_->batch_decode_step(decode_batch);

        // Update request states
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (size_t i = 0; i < decode_batch.size(); ++i) {
            auto* req = decode_requests[i];
            if (i < result.next_tokens.size()) {
                req->last_token = result.next_tokens[i];
                req->generated_tokens.push_back(req->last_token);
                req->seq_len++;

                // Record TTFT for first decode token (if prefill didn't set it)
                if (req->first_token_time == std::chrono::steady_clock::time_point{} &&
                    req->generated_tokens.size() == 1) {
                    req->first_token_time = std::chrono::steady_clock::now();
                }

                // Check if finished
                if (i < result.finished.size() && result.finished[i]) {
                    req->finished = true;
                }

                // Check max_new_tokens
                if (req->generated_tokens.size() >=
                    static_cast<size_t>(req->sampling.max_new_tokens)) {
                    req->finished = true;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "QwenBatchScheduler: decode failed: " << e.what() << std::endl;
    }

    // Clean up temporary batch requests
    for (auto* batch_req : decode_batch) {
        delete batch_req;
    }
}

void QwenBatchScheduler::handle_completions() {
    std::vector<std::unique_ptr<SchedulerRequest>> completed;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // Find completed or cancelled requests
        std::vector<int> slots_to_remove;
        for (auto& [slot_id, req] : slot_to_request_) {
            bool is_cancelled = cancelled_requests_.count(req->request_id) > 0;

            if (req->finished || is_cancelled) {
                slots_to_remove.push_back(slot_id);

                // Record completion time
                req->completion_time = std::chrono::steady_clock::now();

                // Move to completed list
                completed.push_back(std::move(req));

                // Clean up cancelled set
                if (is_cancelled) {
                    cancelled_requests_.erase(req->request_id);
                }
            }
        }

        // Remove from active maps
        for (int slot_id : slots_to_remove) {
            auto it = slot_to_request_.find(slot_id);
            if (it != slot_to_request_.end()) {
                request_id_to_slot_.erase(it->second->request_id);
                slot_to_request_.erase(it);
            }
        }
    }

    // Process completed requests (outside lock to avoid deadlock with callbacks)
    for (auto& req : completed) {
        // Free slot
        if (req->slot_id >= 0) {
            engine_->free_slot(req->request_id);
        }

        // Notify result
        notify_result(req.get());

        // Update stats
        bool was_cancelled = cancelled_requests_.count(req->request_id) > 0;
        if (was_cancelled) {
            total_cancelled_.fetch_add(1);
        } else {
            total_completed_.fetch_add(1);
        }
    }
}

void QwenBatchScheduler::notify_result(SchedulerRequest* req) {
    if (!req || req->callback_invoked) {
        return;
    }

    SchedulerGenerationResult result;
    result.request_id = req->request_id;
    result.generated_tokens = req->generated_tokens;
    result.finish_reason = req->finished ? "length" : "stop";
    result.prompt_tokens = static_cast<int>(req->prompt_tokens.size());
    result.completion_tokens = static_cast<int>(req->generated_tokens.size());

    // Calculate timings
    auto submit = req->submit_time;
    auto first_token = req->first_token_time;
    auto completion = req->completion_time;

    if (completion > submit) {
        result.total_seconds = std::chrono::duration<double>(completion - submit).count();
    }
    if (first_token > submit) {
        result.ttft_seconds = std::chrono::duration<double>(first_token - submit).count();
    }

    req->callback_invoked = true;

    // Invoke callback if provided
    if (req->callback) {
        try {
            req->callback(result);
        } catch (const std::exception& e) {
            std::cerr << "QwenBatchScheduler: callback exception: " << e.what() << std::endl;
        }
    }

    // Store result for polling
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        completed_results_[req->request_id] = result;
    }
    results_cv_.notify_all();
}

}  // namespace dsv4
