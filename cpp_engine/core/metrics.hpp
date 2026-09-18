#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pocket {

// Forward declarations to avoid circular dependencies
struct BatchSchedulerStats;
struct QwenPrefixCacheStats;

// Histogram bucket boundaries, transcribed verbatim from vLLM's
// vllm/v1/metrics/buckets.py so that a pocket_*_bucket line can be lined up
// against the vllm:*_bucket line it corresponds to. The point of copying rather
// than choosing them is comparability: bounds invented here would put the same
// sample in a different bucket than vLLM puts it in, which is exactly the
// resolution a head-to-head comparison reads.
//
// Each family is shared by the metrics that carry the same shape of value, the
// same grouping upstream uses.
inline constexpr std::array<double, 21> kRequestLatencyBounds = {
    0.3,  0.5,  0.8,  1.0,   1.5,   2.0,   2.5,  5.0,  10.0,  15.0, 20.0,
    30.0, 40.0, 50.0, 60.0,  120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
// Sub-second scheduling delays through multi-hour requests; shared by the
// end-to-end latency and the three request phase histograms.

inline constexpr std::array<double, 22> kTimeToFirstTokenBounds = {
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,   0.75,
    1.0,   2.5,   5.0,  7.5,  10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
// Millisecond-scale prefill for tiny prompts through ~40-minute worst cases,
// densest around interactive sub-second latencies.

inline constexpr std::array<double, 19> kInterTokenLatencyBounds = {
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,  0.3,  0.4,  0.5,
    0.75, 1.0,   2.5,  5.0,   7.5, 10.0, 20.0, 40.0, 80.0};
// Per-decode-step latencies: 10 ms fast decode through multi-second stalls.
// Shared by inter-token latency and the per-request time-per-output-token mean,
// which occupy the same range.

// A cumulative Prometheus histogram over a fixed set of upper bounds.
//
// `buckets` holds one slot per finite bound plus a trailing +Inf slot, so the
// total observation count is `buckets[N]` and observe() always touches it.
// Parameterizing the bound count is what lets each family carry the resolution
// its metric needs instead of one compromise set, and sizing the array by N + 1
// makes the +Inf slot structural rather than a convention indexed by hand.
//
// Deliberately not std::vector<std::atomic<uint64_t>>: resize and push_back
// require MoveInsertable, which std::atomic is not.
template <std::size_t N>
struct HistogramT {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum_microseconds{0};
    std::array<std::atomic<uint64_t>, N + 1> buckets{};

    // Bucket upper bounds in seconds. The +Inf bucket is implicit.
    std::array<double, N> bounds;

    explicit HistogramT(std::array<double, N> b) : bounds(b) {}
    HistogramT() : bounds{} {}

    void observe(double seconds);
};

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

    // Latency histograms. Each carries the bucket family vLLM pairs it with:
    // request_latency for the end-to-end and per-phase families,
    // time_to_first_token for TTFT, inter_token_latency for ITL and for the
    // per-request time-per-output-token mean.
    //
    // The two ITL families are per *choice*; the other five are per *request*.
    // For a single-choice request -- what every acceptance run and the
    // compatibility harness issues -- the two coincide.
    HistogramT<kRequestLatencyBounds.size()> request_duration;
    HistogramT<kTimeToFirstTokenBounds.size()> ttft;
    HistogramT<kInterTokenLatencyBounds.size()> inter_token_latency;
    HistogramT<kInterTokenLatencyBounds.size()> request_time_per_output_token;
    HistogramT<kRequestLatencyBounds.size()> request_queue_time;
    HistogramT<kRequestLatencyBounds.size()> request_prefill_time;
    HistogramT<kRequestLatencyBounds.size()> request_decode_time;

    EngineMetrics();

    void record_request_start();
    void record_request_end(bool success, double duration_seconds,
                           double ttft_seconds, int prompt_tokens,
                           int generation_tokens);

    // The per-token measurements of one choice: the intervals between
    // successive tokens *as they were produced*, in seconds, one entry per token
    // after the first.
    //
    // Records one inter-token-latency sample per gap, and one
    // time-per-output-token sample for the choice -- the per-request mean vLLM
    // reports, not a pooled per-token sample. Deriving both from the same gaps is
    // what relates the two families by
    // `inter_token_latency_sum == time_per_output_token_sum * (n - 1)`, which a
    // drain-time stamp cannot satisfy: a drain pops a whole decode burst at once
    // and would report microseconds between tokens the engine spent real time
    // on. The sums are stored in microseconds, so the relation holds to that
    // quantum rather than to the last bit.
    //
    // An empty vector is a choice that produced fewer than two tokens: it has no
    // interval to average, so it contributes nothing to either family, matching
    // vLLM's exclusion of `output_len <= 1` and its zero denominator guard.
    void record_token_gaps(const std::vector<double>& gaps);

    // The phase split of one completed request, in seconds. A negative value
    // means that phase's boundaries were never both observed -- a request
    // cancelled before its first token has no prefill or decode interval -- and
    // records nothing, rather than a zero that would read as instantaneous.
    // The three therefore have independent counts, unlike vLLM's, which
    // observes all three for every finished request.
    void record_request_phases(double queue_seconds, double prefill_seconds,
                               double decode_seconds);

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
