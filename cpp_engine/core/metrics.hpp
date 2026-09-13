#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace pocket {

// Forward declarations to avoid circular dependencies
struct BatchSchedulerStats;
struct QwenPrefixCacheStats;

// Prometheus metrics collector for cpp_engine observability.
//
// Metrics are updated inline by OpenAIServer request handlers and remain
// memory-resident until process exit. The serialize_prometheus() method
// generates Prometheus text format output suitable for /metrics endpoints.
//
// Thread-safe: all counters and gauges use std::atomic for lock-free updates.
struct EngineMetrics {
    // Counters (monotonic, never decrease)
    std::atomic<uint64_t> requests_total_success{0};
    std::atomic<uint64_t> requests_total_error{0};
    std::atomic<uint64_t> prompt_tokens_total{0};
    std::atomic<uint64_t> generation_tokens_total{0};

    // Gauges (current value, may increase or decrease)
    std::atomic<int64_t> requests_active{0};

    // Histogram: request duration in seconds
    struct Histogram {
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> sum_microseconds{0};
        std::array<std::atomic<uint64_t>, 9> buckets{};

        // Bucket upper bounds in seconds. The +Inf bucket is implicit.
        std::array<double, 8> bounds;

        Histogram(std::array<double, 8> b) : bounds(b) {}
        Histogram() : bounds{} {}

        void observe(double seconds);
    };

    Histogram request_duration;
    Histogram ttft;

    EngineMetrics();

    void record_request_start();
    void record_request_end(bool success, double duration_seconds,
                           double ttft_seconds, int prompt_tokens,
                           int generation_tokens);

    // Generate Prometheus text format output for all metrics.
    // Includes request/token/histogram metrics from this struct plus
    // scheduler and cache metrics passed as arguments.
    std::string serialize_prometheus(
        int waiting_requests, int running_requests,
        int free_slots, int reserved_blocks, int total_blocks,
        int free_blocks, int cache_pinned_blocks,
        const QwenPrefixCacheStats* cache_stats) const;
};

}  // namespace pocket
