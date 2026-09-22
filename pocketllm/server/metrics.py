"""Dependency-free Prometheus text metrics for the PocketLLM server.

Histograms here are real Prometheus histograms: cumulative `_bucket{le=...}`
counters plus `_sum` and `_count`.  Quantiles are *not* exported as their own
series -- a Prometheus client is expected to derive them from the buckets, and
a `{quantile="0.9"}` line computed from a retained sample list is not something
any scraper reads.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

# Histogram bucket upper bounds, transcribed verbatim from vLLM's
# vllm/v1/metrics/buckets.py so that a `pocketllm_*_bucket` line can be lined up
# against the `vllm:*_bucket` line it corresponds to.  The point of copying
# rather than choosing them is comparability: bounds invented here would put the
# same sample in a different bucket than vLLM puts it in, which is exactly the
# resolution a head-to-head comparison reads.
#
# These mirror the C++ server's families in cpp_engine/core/metrics.hpp, which
# carries the same three sets.  A family that is not listed there is recorded by
# neither server.
REQUEST_LATENCY_BOUNDS: tuple[float, ...] = (
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0,
    30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
)
# Sub-second scheduling delays through multi-hour requests; shared by the
# end-to-end latency and the request phase histograms.

TIME_TO_FIRST_TOKEN_BOUNDS: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
)
# Millisecond-scale prefill for tiny prompts through ~40-minute worst cases,
# densest around interactive sub-second latencies.

INTER_TOKEN_LATENCY_BOUNDS: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
)
# Per-decode-step latencies: 10 ms fast decode through multi-second stalls.
# Shared by the inter-token latency and the per-request time-per-output-token
# mean, which occupy the same range.

# Every histogram this server may record, mapped to its bounds and its `# HELP`
# text.  A metric must be declared here before it can be observed: a `_bucket`
# series needs upper bounds above it and there is no defensible default to
# invent.  `observe` on an undeclared name raises rather than guessing.
#
# The declared families are exported from process start with zero counts, the
# way the native server reports them, so that a rate over the first scrape
# window is defined.  The counts matching zero on a fresh server is a fact about
# the server, not a missing series.
HISTOGRAMS: dict[str, tuple[tuple[float, ...], str]] = {
    "request_duration_seconds": (
        REQUEST_LATENCY_BOUNDS,
        "End-to-end latency of one request, arrival to last token.",
    ),
    "ttft_seconds": (
        TIME_TO_FIRST_TOKEN_BOUNDS,
        "Time from request arrival to the first generated token.",
    ),
    "inter_token_latency_seconds": (
        INTER_TOKEN_LATENCY_BOUNDS,
        "Intervals between successive tokens of one choice.",
    ),
    "request_time_per_output_token_seconds": (
        INTER_TOKEN_LATENCY_BOUNDS,
        "Per-request mean of the inter-token intervals.",
    ),
}


class _Histogram:
    """Cumulative bucket counters plus a running sum, and nothing else.

    Raw samples are deliberately not retained.  A long-lived server would grow
    without bound, and `_bucket` / `_sum` / `_count` is the entire state a
    Prometheus histogram is defined to carry.

    `increments` holds one non-cumulative slot per finite bound plus a trailing
    `+Inf` slot, so the overflow counter is structural rather than an index
    computed by hand.  The running total is taken at render time, which keeps
    the observation on the request path O(log n) rather than O(n).
    """

    __slots__ = ("bounds", "increments", "sum_seconds", "count")

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self.bounds = bounds
        self.increments = [0] * (len(bounds) + 1)
        self.sum_seconds = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        # `le` is inclusive, so a sample exactly on a bound belongs to that
        # bound's bucket.  A sample above every bound lands in the +Inf slot.
        self.increments[bisect.bisect_left(self.bounds, value)] += 1
        self.sum_seconds += value
        self.count += 1


def _bound_label(bound: float) -> str:
    """Format a bucket bound the way Prometheus clients do.

    `str` of a Python float prints `1.0` where C++ `ostream` would print `1`,
    which is the spelling a real `vllm:*_bucket` line uses -- and the whole
    point of the copied bounds is that the two can be compared line to line.
    """
    return str(bound)


class Metrics:
    """Small thread-safe counter/gauge/histogram collector.

    The exporter intentionally emits stable names without requiring the
    prometheus-client package.  Applications may wrap this collector with a
    richer exporter later without changing backend code.
    """

    def __init__(self, prefix: str = "pocketllm") -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._gauges: defaultdict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = {
            name: _Histogram(bounds) for name, (bounds, _) in HISTOGRAMS.items()
        }

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += float(value)

    def set_counter(self, name: str, value: float) -> None:
        """Publish a counter's absolute value rather than incrementing it.

        For a counter a backend owns end to end: it already keeps the running total, and the
        exporter's job is to copy it out at scrape time.  Adding it instead would square the count
        over two scrapes of the same value.
        """
        with self._lock:
            self._counters[name] = float(value)

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def add(self, name: str, value: float) -> None:
        """Add to a gauge while preserving counter monotonicity."""
        with self._lock:
            self._gauges[name] += float(value)

    def observe(self, name: str, value: float) -> None:
        """Fold one sample into a declared histogram.

        Unlike a counter, a histogram cannot be created on first use: its
        `_bucket` series only means something against a fixed set of bounds.
        """
        histogram = self._histograms.get(name)
        if histogram is None:
            known = ", ".join(sorted(self._histograms))
            raise KeyError(f"undeclared histogram {name!r}; known histograms: {known}")
        with self._lock:
            histogram.observe(float(value))

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {
                name: (list(item.increments), item.sum_seconds, item.count)
                for name, item in self._histograms.items()
            }
        # `.17g` and not `:g`: these are counts and byte totals, and the default six significant
        # digits render a 4 GiB budget as `4.29497e+09` -- a number that is still a valid float but no
        # longer the value. Seventeen digits is exact for every float64 and still prints `7.0` as `7`,
        # so the integer series read the way they always did.
        for name, value in sorted(counters.items()):
            lines.append(f"# TYPE {self.prefix}_{name} counter")
            lines.append(f"{self.prefix}_{name} {value:.17g}")
        for name, value in sorted(gauges.items()):
            lines.append(f"# TYPE {self.prefix}_{name} gauge")
            lines.append(f"{self.prefix}_{name} {value:.17g}")
        for name in sorted(histograms):
            bounds, help_text = HISTOGRAMS[name]
            increments, total, count = histograms[name]
            lines.append(f"# HELP {self.prefix}_{name} {help_text}")
            lines.append(f"# TYPE {self.prefix}_{name} histogram")
            cumulative = 0
            for bound, increment in zip(bounds, increments):
                cumulative += increment
                lines.append(
                    f'{self.prefix}_{name}_bucket{{le="{_bound_label(bound)}"}} {cumulative}'
                )
            # Every observation increments exactly one increment slot and the
            # count, so the +Inf bucket is the count by construction.
            cumulative += increments[-1]
            lines.append(f'{self.prefix}_{name}_bucket{{le="+Inf"}} {cumulative}')
            lines.append(f"{self.prefix}_{name}_sum {total:.9g}")
            lines.append(f"{self.prefix}_{name}_count {count}")
        return "\n".join(lines) + ("\n" if lines else "")
