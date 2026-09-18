#include "metrics.hpp"

#include <sstream>

#include "qwen_engine.hpp"

namespace pocket {

EngineMetrics::EngineMetrics()
    : request_duration(kRequestLatencyBounds),
      ttft(kTimeToFirstTokenBounds),
      inter_token_latency(kInterTokenLatencyBounds),
      request_time_per_output_token(kInterTokenLatencyBounds),
      request_queue_time(kRequestLatencyBounds),
      request_prefill_time(kRequestLatencyBounds),
      request_decode_time(kRequestLatencyBounds) {}

template <std::size_t N>
void HistogramT<N>::observe(double seconds) {
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

// Explicit instantiations for the families EngineMetrics holds. Defining the
// template here rather than in the header keeps it to the one translation unit
// that serializes it.
template struct HistogramT<kRequestLatencyBounds.size()>;
template struct HistogramT<kTimeToFirstTokenBounds.size()>;
template struct HistogramT<kInterTokenLatencyBounds.size()>;

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

void EngineMetrics::record_token_gaps(const std::vector<double>& gaps) {
    if (gaps.empty()) return;
    double sum = 0.0;
    for (double gap : gaps) {
        inter_token_latency.observe(gap);
        sum += gap;
    }
    request_time_per_output_token.observe(sum / static_cast<double>(gaps.size()));
}

void EngineMetrics::record_request_phases(double queue_seconds, double prefill_seconds,
                                          double decode_seconds) {
    if (queue_seconds >= 0.0) request_queue_time.observe(queue_seconds);
    if (prefill_seconds >= 0.0) request_prefill_time.observe(prefill_seconds);
    if (decode_seconds >= 0.0) request_decode_time.observe(decode_seconds);
}

namespace {

// One Prometheus histogram family. N is the number of finite bounds; the +Inf
// bucket is the (N + 1)th slot of `buckets` and always equals the observation
// count.
//
// `le` labels keep default ostream formatting, so 1.0 prints as 1 -- the same
// text the two families that predate this helper already emitted. Upstream
// prints 1.0; matching that would rewrite those lines too, so it is a separate
// change rather than part of this one.
template <std::size_t N>
void write_histogram(std::ostringstream& out, const char* name, const char* help,
                     const HistogramT<N>& histogram) {
    out << "# HELP " << name << " " << help << "\n";
    out << "# TYPE " << name << " histogram\n";
    for (std::size_t i = 0; i < N; ++i) {
        out << name << "_bucket{le=\"" << histogram.bounds[i] << "\"} "
            << histogram.buckets[i].load(std::memory_order_relaxed) << "\n";
    }
    out << name << "_bucket{le=\"+Inf\"} "
        << histogram.buckets[N].load(std::memory_order_relaxed) << "\n";
    out << name << "_sum "
        << histogram.sum_microseconds.load(std::memory_order_relaxed) / 1e6 << "\n";
    out << name << "_count "
        << histogram.count.load(std::memory_order_relaxed) << "\n";
}

}  // namespace

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

    write_histogram(out, "pocket_request_duration_seconds",
                    "Request end-to-end duration", request_duration);
    write_histogram(out, "pocket_ttft_seconds", "Time to first token", ttft);
    write_histogram(out, "pocket_inter_token_latency_seconds",
                    "Inter-token latency, one sample per token after the first",
                    inter_token_latency);
    write_histogram(out, "pocket_request_time_per_output_token_seconds",
                    "Time per output token, one sample per request",
                    request_time_per_output_token);
    write_histogram(out, "pocket_request_queue_time_seconds",
                    "Time a request waited before a slot was reserved for it",
                    request_queue_time);
    write_histogram(out, "pocket_request_prefill_time_seconds",
                    "Time from a request's admission to its first token",
                    request_prefill_time);
    write_histogram(out, "pocket_request_decode_time_seconds",
                    "Time from a request's first token to its completion",
                    request_decode_time);

    return out.str();
}

}  // namespace pocket
