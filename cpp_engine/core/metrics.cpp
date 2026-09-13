#include "metrics.hpp"

#include <sstream>

#include "qwen_engine.hpp"

namespace pocket {

EngineMetrics::EngineMetrics()
    : request_duration(std::array<double, 8>{0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0}),
      ttft(std::array<double, 8>{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}) {}

void EngineMetrics::Histogram::observe(double seconds) {
    count.fetch_add(1, std::memory_order_relaxed);
    const auto micros = static_cast<uint64_t>(seconds * 1e6);
    sum_microseconds.fetch_add(micros, std::memory_order_relaxed);

    for (size_t i = 0; i < bounds.size(); ++i) {
        if (seconds <= bounds[i]) {
            buckets[i].fetch_add(1, std::memory_order_relaxed);
        }
    }
    buckets[bounds.size()].fetch_add(1, std::memory_order_relaxed);
}

void EngineMetrics::record_request_start() {
    requests_active.fetch_add(1, std::memory_order_relaxed);
}

void EngineMetrics::record_request_end(bool success, double duration_seconds,
                                       double ttft_seconds, int prompt_tokens,
                                       int generation_tokens) {
    requests_active.fetch_sub(1, std::memory_order_relaxed);

    if (success) {
        requests_total_success.fetch_add(1, std::memory_order_relaxed);
    } else {
        requests_total_error.fetch_add(1, std::memory_order_relaxed);
    }

    prompt_tokens_total.fetch_add(static_cast<uint64_t>(prompt_tokens),
                                 std::memory_order_relaxed);
    generation_tokens_total.fetch_add(static_cast<uint64_t>(generation_tokens),
                                     std::memory_order_relaxed);

    request_duration.observe(duration_seconds);
    if (ttft_seconds > 0.0) {
        ttft.observe(ttft_seconds);
    }
}

std::string EngineMetrics::serialize_prometheus(
    int waiting_requests, int running_requests,
    int free_slots, int reserved_blocks, int total_blocks,
    int free_blocks, int cache_pinned_blocks,
    const QwenPrefixCacheStats* cache_stats) const {

    std::ostringstream out;

    // Request counters
    out << "# HELP pocket_requests_total Total number of completed requests\n";
    out << "# TYPE pocket_requests_total counter\n";
    out << "pocket_requests_total{status=\"success\"} "
        << requests_total_success.load(std::memory_order_relaxed) << "\n";
    out << "pocket_requests_total{status=\"error\"} "
        << requests_total_error.load(std::memory_order_relaxed) << "\n";

    // Token counters
    out << "# HELP pocket_tokens_total Total tokens processed\n";
    out << "# TYPE pocket_tokens_total counter\n";
    out << "pocket_tokens_total{type=\"prompt\"} "
        << prompt_tokens_total.load(std::memory_order_relaxed) << "\n";
    out << "pocket_tokens_total{type=\"generation\"} "
        << generation_tokens_total.load(std::memory_order_relaxed) << "\n";

    // Active requests gauge
    out << "# HELP pocket_requests_active Currently active requests\n";
    out << "# TYPE pocket_requests_active gauge\n";
    out << "pocket_requests_active "
        << requests_active.load(std::memory_order_relaxed) << "\n";

    // Scheduler gauges
    out << "# HELP pocket_requests_waiting Requests waiting for admission\n";
    out << "# TYPE pocket_requests_waiting gauge\n";
    out << "pocket_requests_waiting " << waiting_requests << "\n";

    out << "# HELP pocket_requests_running Requests currently running\n";
    out << "# TYPE pocket_requests_running gauge\n";
    out << "pocket_requests_running " << running_requests << "\n";

    out << "# HELP pocket_slots_free Available request slots\n";
    out << "# TYPE pocket_slots_free gauge\n";
    out << "pocket_slots_free " << free_slots << "\n";

    // Paged KV metrics
    if (total_blocks > 0) {
        out << "# HELP pocket_kv_blocks KV cache blocks by state\n";
        out << "# TYPE pocket_kv_blocks gauge\n";
        out << "pocket_kv_blocks{state=\"total\"} " << total_blocks << "\n";
        out << "pocket_kv_blocks{state=\"free\"} " << free_blocks << "\n";
        out << "pocket_kv_blocks{state=\"reserved\"} " << reserved_blocks << "\n";
        out << "pocket_kv_blocks{state=\"cache_pinned\"} " << cache_pinned_blocks << "\n";
    }

    // Prefix cache metrics
    if (cache_stats) {
        out << "# HELP pocket_prefix_cache_hits Prefix cache hit count\n";
        out << "# TYPE pocket_prefix_cache_hits counter\n";
        out << "pocket_prefix_cache_hits{scope=\"slot\"} " << cache_stats->hits << "\n";
        out << "pocket_prefix_cache_hits{scope=\"global\"} " << cache_stats->global_hits << "\n";

        out << "# HELP pocket_prefix_cache_misses Prefix cache miss count\n";
        out << "# TYPE pocket_prefix_cache_misses counter\n";
        out << "pocket_prefix_cache_misses{scope=\"slot\"} " << cache_stats->misses << "\n";
        out << "pocket_prefix_cache_misses{scope=\"global\"} " << cache_stats->global_misses << "\n";

        out << "# HELP pocket_prefix_cache_blocks Globally cached blocks\n";
        out << "# TYPE pocket_prefix_cache_blocks gauge\n";
        out << "pocket_prefix_cache_blocks " << cache_stats->global_cached_blocks << "\n";

        out << "# HELP pocket_prefix_cache_bytes Current cache memory usage\n";
        out << "# TYPE pocket_prefix_cache_bytes gauge\n";
        out << "pocket_prefix_cache_bytes{type=\"used\"} "
            << cache_stats->global_cache_bytes << "\n";
        out << "pocket_prefix_cache_bytes{type=\"budget\"} "
            << cache_stats->global_cache_budget_bytes << "\n";

        out << "# HELP pocket_prefix_cache_evictions Total number of cache evictions\n";
        out << "# TYPE pocket_prefix_cache_evictions counter\n";
        out << "pocket_prefix_cache_evictions " << cache_stats->global_evictions << "\n";
    }

    // Request duration histogram
    const auto& rd = request_duration;
    out << "# HELP pocket_request_duration_seconds Request end-to-end duration\n";
    out << "# TYPE pocket_request_duration_seconds histogram\n";
    for (size_t i = 0; i < rd.bounds.size(); ++i) {
        out << "pocket_request_duration_seconds_bucket{le=\"" << rd.bounds[i] << "\"} "
            << rd.buckets[i].load(std::memory_order_relaxed) << "\n";
    }
    out << "pocket_request_duration_seconds_bucket{le=\"+Inf\"} "
        << rd.buckets[rd.bounds.size()].load(std::memory_order_relaxed) << "\n";
    out << "pocket_request_duration_seconds_sum "
        << rd.sum_microseconds.load(std::memory_order_relaxed) / 1e6 << "\n";
    out << "pocket_request_duration_seconds_count "
        << rd.count.load(std::memory_order_relaxed) << "\n";

    // TTFT histogram
    const auto& tt = ttft;
    out << "# HELP pocket_ttft_seconds Time to first token\n";
    out << "# TYPE pocket_ttft_seconds histogram\n";
    for (size_t i = 0; i < tt.bounds.size(); ++i) {
        out << "pocket_ttft_seconds_bucket{le=\"" << tt.bounds[i] << "\"} "
            << tt.buckets[i].load(std::memory_order_relaxed) << "\n";
    }
    out << "pocket_ttft_seconds_bucket{le=\"+Inf\"} "
        << tt.buckets[tt.bounds.size()].load(std::memory_order_relaxed) << "\n";
    out << "pocket_ttft_seconds_sum "
        << tt.sum_microseconds.load(std::memory_order_relaxed) / 1e6 << "\n";
    out << "pocket_ttft_seconds_count "
        << tt.count.load(std::memory_order_relaxed) << "\n";

    return out.str();
}

}  // namespace pocket
