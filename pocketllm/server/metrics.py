"""Dependency-free Prometheus text metrics for the PocketLLM server."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


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
        self._histograms: defaultdict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += float(value)

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def add(self, name: str, value: float) -> None:
        """Add to a gauge while preserving counter monotonicity."""
        with self._lock:
            self._gauges[name] += float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(float(value))

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
            histograms = {key: list(value) for key, value in self._histograms.items()}
        for name, value in sorted(counters.items()):
            lines.append(f"{self.prefix}_{name} {value:g}")
        for name, value in sorted(gauges.items()):
            lines.append(f"{self.prefix}_{name} {value:g}")
        for name, values in sorted(histograms.items()):
            if not values:
                continue
            ordered = sorted(values)
            total = sum(ordered)
            lines.append(f"{self.prefix}_{name}_count {len(ordered)}")
            lines.append(f"{self.prefix}_{name}_sum {total:.9g}")
            for quantile in (0.5, 0.9, 0.99):
                index = min(len(ordered) - 1, int(quantile * len(ordered)))
                lines.append(f'{self.prefix}_{name}{{quantile="{quantile}"}} {ordered[index]:.9g}')
        return "\n".join(lines) + ("\n" if lines else "")
