"""Shared GPU memory sampling for the engine comparison benchmarks.

PocketLLM and vLLM must report memory through the same instrument or the
comparison column is meaningless. They do not by default: PocketLLM reports
what its own engine allocated, while vLLM pre-reserves a
`gpu_memory_utilization` fraction of the card up front, so a single post-run
`nvidia-smi` reading describes vLLM's reservation rather than what the model
actually needed. Reading the same external instrument on both sides, and
keeping the peak rather than one final sample, is what makes the numbers
comparable.

`nvidia-smi` reports physical device indices and ignores
CUDA_VISIBLE_DEVICES, so callers pass physical indices here regardless of how
they masked the child processes.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

MIB = 1024 * 1024


def read_used_bytes(devices: list[int]) -> dict[int, int]:
    """One `nvidia-smi` poll of used memory, in bytes, per requested device."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    used: dict[int, int] = {}
    for line in out.strip().splitlines():
        index, value = (item.strip() for item in line.split(","))
        if int(index) in devices:
            used[int(index)] = int(value) * MIB
    return used


class GpuMemorySampler:
    """Polls used memory on a background thread and keeps the per-device peak.

    A single reading after the run misses the peak entirely: prefill at 65K
    allocates and frees activation workspace long before the process exits. The
    sampler runs for the whole measured window instead.

    The poll interval is a tradeoff against what it perturbs -- each poll forks
    `nvidia-smi`, so 250 ms is frequent enough to catch a multi-second prefill
    peak without adding measurable load to the benchmark itself.
    """

    def __init__(self, devices: list[int], interval_seconds: float = 0.25) -> None:
        self.devices = list(devices)
        self.interval_seconds = interval_seconds
        self.before: dict[int, int] = {}
        self.peak: dict[int, int] = {}
        self.after: dict[int, int] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _poll_into_peak(self) -> None:
        try:
            used = read_used_bytes(self.devices)
        except Exception:
            # A transient nvidia-smi failure must not take down the benchmark;
            # the peak simply misses that sample.
            return
        with self._lock:
            self.samples += 1
            for device, value in used.items():
                if value > self.peak.get(device, 0):
                    self.peak[device] = value

    def start(self) -> "GpuMemorySampler":
        self.before = read_used_bytes(self.devices)
        with self._lock:
            self.peak = dict(self.before)

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                self._poll_into_peak()

        self._thread = threading.Thread(target=loop, name="gpu-mem-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "GpuMemorySampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Fold in a final reading so a run shorter than one interval still has
        # a peak that reflects the run rather than only its starting point.
        self._poll_into_peak()
        self.after = read_used_bytes(self.devices)
        return self

    def __enter__(self) -> "GpuMemorySampler":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def report(self) -> dict[str, Any]:
        """Serializable result. `delta` is peak minus the pre-run baseline.

        `delta` is the figure to compare across engines: it excludes whatever
        was already resident on the card before the run started. `peak_bytes`
        is kept alongside it because an absolute ceiling is what tells you
        whether a configuration fits in 22 GiB at all.
        """
        with self._lock:
            peak = dict(self.peak)
        delta = {
            device: max(peak.get(device, 0) - self.before.get(device, 0), 0)
            for device in self.devices
        }
        return {
            "instrument": "nvidia-smi",
            "interval_seconds": self.interval_seconds,
            "samples": self.samples,
            "devices": self.devices,
            "before_bytes": dict(self.before),
            "peak_bytes": peak,
            "after_bytes": dict(self.after),
            "delta_bytes": delta,
            "max_peak_bytes": max(peak.values(), default=None),
            "max_delta_bytes": max(delta.values(), default=None),
        }
