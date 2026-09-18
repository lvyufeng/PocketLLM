// Unit tests for the Prometheus exposition the native server serves at
// /metrics.
//
// The collector is a plain struct of relaxed atomics, so this test links
// pocket_core, needs no device and no checkpoint, and runs in under a second.
// That is the point: the metric contract -- the bucket sets, the +Inf slot, the
// negative "unknown" sentinels, and the series names a running server is scraped
// for -- is checked on every build rather than only in the acceptance runs that
// need eight NPUs and a 27B checkpoint.
//
// Plain `assert` is deliberately not used. Release builds define NDEBUG, which
// would compile every assertion in this file out and report success without
// running anything.

#include "metrics.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace pocket;

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    std::cout << "FAIL line " << line << ": " << expression << "\n";
    ++g_failures;
}

#define CHECK(condition) check((condition), #condition, __LINE__)

const QwenPrefixCacheStats* kNoCache = nullptr;

std::string scrape(const EngineMetrics& metrics) {
    return metrics.serialize_prometheus(0, 0, 0, 0, 0, 0, 0, kNoCache);
}

// Every exposition line beginning with `prefix`, where prefix already carries
// its trailing space so that `pocket_x_count` cannot match
// `pocket_x_count_something`.
std::vector<std::string> lines_with(const std::string& text, const std::string& prefix) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, prefix.size(), prefix) == 0) out.push_back(line);
    }
    return out;
}

// The value of a `name value` exposition line.
double value_of(const std::string& line) {
    const std::size_t space = line.rfind(' ');
    return std::stod(line.substr(space + 1));
}

// The single sample of a one-series family, or -1 when the family is absent.
double sample(const std::string& text, const std::string& name) {
    const std::vector<std::string> lines = lines_with(text, name + " ");
    return lines.empty() ? -1.0 : value_of(lines[0]);
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// The families the running server is scraped for, and the number of finite
// bounds each carries. The counts are the vLLM bucket sets transcribed into
// metrics.hpp; a change to any of them breaks comparability with an upstream
// scrape, so it should have to break this test too.
struct Family {
    const char* name;
    std::size_t finite_bounds;
};

const Family kFamilies[] = {
    {"pocket_request_duration_seconds", kRequestLatencyBounds.size()},
    {"pocket_ttft_seconds", kTimeToFirstTokenBounds.size()},
    {"pocket_inter_token_latency_seconds", kInterTokenLatencyBounds.size()},
    {"pocket_request_time_per_output_token_seconds", kInterTokenLatencyBounds.size()},
    {"pocket_request_queue_time_seconds", kRequestLatencyBounds.size()},
    {"pocket_request_prefill_time_seconds", kRequestLatencyBounds.size()},
    {"pocket_request_decode_time_seconds", kRequestLatencyBounds.size()},
};

// The counter families bench_cpp_openai_phases.py pins by name. It reads these
// line prefixes from a live scrape and fails when one is absent, so renaming or
// dropping any of them silently breaks the phase-split acceptance run.
void test_names_the_harness_pins_are_present() {
    EngineMetrics metrics;
    const std::string text = scrape(metrics);

    const char* pinned[] = {
        "pocket_ttft_seconds_sum",
        "pocket_ttft_seconds_count",
        "pocket_request_duration_seconds_sum",
        "pocket_request_duration_seconds_count",
        "pocket_tokens_total{type=\"prompt\"}",
        "pocket_tokens_total{type=\"generation\"}",
        "pocket_requests_total{status=\"success\"}",
    };
    for (const char* name : pinned) {
        CHECK(contains(text, name));
    }
}

// A scrape must be a valid Prometheus exposition: every metric family carries
// its HELP and TYPE lines, and every histogram emits one cumulative bucket per
// finite bound plus the +Inf bucket, a sum and a count. Without the TYPE line a
// scraper reads the _bucket series as untyped gauges, which is not a histogram.
void test_exposition_shape() {
    EngineMetrics metrics;
    const std::string text = scrape(metrics);

    for (const Family& family : kFamilies) {
        CHECK(contains(text, std::string("# HELP ") + family.name + " "));
        CHECK(contains(text, std::string("# TYPE ") + family.name + " histogram\n"));

        const std::vector<std::string> buckets =
            lines_with(text, std::string(family.name) + "_bucket{");
        CHECK(buckets.size() == family.finite_bounds + 1);
        if (!buckets.empty()) {
            CHECK(contains(buckets.back(), "le=\"+Inf\""));
        }
        CHECK(lines_with(text, std::string(family.name) + "_sum ").size() == 1);
        CHECK(lines_with(text, std::string(family.name) + "_count ").size() == 1);
    }
}

// An observation lands in the first bucket whose bound it does not exceed, and
// therefore in every later one: the counts are cumulative, which is what makes
// a quantile readable from them. The +Inf slot is the total, always.
void test_observe_is_cumulative() {
    EngineMetrics metrics;
    // 5.0 is a finite bound of the request-latency family; the index is what
    // makes this a statement about placement rather than about counting.
    metrics.request_duration.observe(5.0);

    std::size_t index = 0;
    while (index < kRequestLatencyBounds.size() && kRequestLatencyBounds[index] < 5.0) {
        ++index;
    }
    CHECK(index < kRequestLatencyBounds.size());
    CHECK(metrics.request_duration.buckets[index].load() == 1);

    // Read positionally rather than by label: the exposition emits the bounds in
    // order, so a bucket line's index *is* its bound, and this doubles as a check
    // that the `le` values ascend the way Prometheus requires.
    const std::vector<std::string> buckets =
        lines_with(scrape(metrics), "pocket_request_duration_seconds_bucket{");
    CHECK(buckets.size() == kRequestLatencyBounds.size() + 1);
    for (std::size_t i = 0; i < buckets.size() && i <= kRequestLatencyBounds.size(); ++i) {
        if (buckets[i].find("+Inf") != std::string::npos) continue;
        CHECK(value_of(buckets[i]) == (i < index ? 0.0 : 1.0));
    }
    CHECK(sample(scrape(metrics), "pocket_request_duration_seconds_count") == 1.0);
}

// A sample larger than every finite bound must still be counted, and only in the
// +Inf bucket. This is the regression test for the ceiling this collector used
// to have: pocket_ttft_seconds stopped at 5.0 s, so a 6497-token prefill -- 4.5
// to 5.6 s on this workload -- landed in +Inf alone and the histogram could not
// be quantiled at all.
void test_sample_above_the_top_bound_lands_only_in_inf() {
    EngineMetrics metrics;
    const double huge = kTimeToFirstTokenBounds.back() * 4.0;
    metrics.ttft.observe(huge);
    metrics.request_duration.observe(huge);

    CHECK(metrics.ttft.count.load() == 1);
    for (std::size_t i = 0; i < kTimeToFirstTokenBounds.size(); ++i) {
        CHECK(metrics.ttft.buckets[i].load() == 0);
    }
    CHECK(metrics.ttft.buckets[kTimeToFirstTokenBounds.size()].load() == 1);

    // 4x the largest TTFT bound is past the end of the request-latency family
    // too, so both families must agree about it.
    for (std::size_t i = 0; i < kRequestLatencyBounds.size(); ++i) {
        CHECK(metrics.request_duration.buckets[i].load() == 0);
    }
    CHECK(metrics.request_duration.buckets[kRequestLatencyBounds.size()].load() == 1);
}

// A phase whose boundaries were never both observed reports a negative sentinel,
// not a zero: a request cancelled before its first token has no decode interval,
// and recording 0.0 for it would put a fake instantaneous sample in the
// histogram. `record_request_end` keeps the same convention for TTFT.
void test_negative_sentinels_record_nothing() {
    EngineMetrics metrics;
    metrics.record_request_phases(-1.0, -1.0, -1.0);

    CHECK(metrics.request_queue_time.count.load() == 0);
    CHECK(metrics.request_prefill_time.count.load() == 0);
    CHECK(metrics.request_decode_time.count.load() == 0);

    // Each phase is independent: a request cancelled before its first token
    // still waited in the queue, so one known phase is recorded even when the
    // other two are not.
    metrics.record_request_phases(0.25, -1.0, -1.0);
    CHECK(metrics.request_queue_time.count.load() == 1);
    CHECK(metrics.request_prefill_time.count.load() == 0);
    CHECK(metrics.request_decode_time.count.load() == 0);

    metrics.record_request_end(true, 1.5, 0.0, 10, 4);
    CHECK(metrics.ttft.count.load() == 0);
    CHECK(metrics.request_duration.count.load() == 1);
}

// Inter-token latency and time per output token are derived from the same gaps,
// so `itl_sum == tpot_sum * (n - 1)` -- the identity a drain-time stamp cannot
// satisfy, because a drain pops a whole decode burst at once and would report
// microseconds between tokens the engine spent real time on.
void test_token_gaps_drive_both_families() {
    EngineMetrics metrics;
    const std::vector<double> gaps = {0.05, 0.1, 0.15};

    metrics.record_token_gaps(gaps);
    CHECK(metrics.inter_token_latency.count.load() == gaps.size());
    CHECK(metrics.request_time_per_output_token.count.load() == 1);

    const std::string text = scrape(metrics);
    const double itl_sum = sample(text, "pocket_inter_token_latency_seconds_sum");
    const double tpot_sum = sample(text, "pocket_request_time_per_output_token_seconds_sum");
    CHECK(std::fabs(itl_sum - 0.3) < 1e-6);
    CHECK(std::fabs(tpot_sum - 0.1) < 1e-6);
    CHECK(std::fabs(itl_sum - tpot_sum * gaps.size()) < 1e-6);

    // A choice that produced fewer than two tokens has no interval to average,
    // so it contributes to neither family -- the same exclusion vLLM makes for
    // `output_len <= 1` rather than a zero-length sample.
    metrics.record_token_gaps({});
    CHECK(metrics.inter_token_latency.count.load() == gaps.size());
    CHECK(metrics.request_time_per_output_token.count.load() == 1);
}

}  // namespace

int main() {
    test_names_the_harness_pins_are_present();
    test_exposition_shape();
    test_observe_is_cumulative();
    test_sample_above_the_top_bound_lands_only_in_inf();
    test_negative_sentinels_record_nothing();
    test_token_gaps_drive_both_families();

    if (g_failures != 0) {
        std::cout << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "test_engine_metrics: all checks passed\n";
    return 0;
}
