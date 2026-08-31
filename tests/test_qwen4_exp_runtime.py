"""Runtime topology helpers for Qwen4-Exp heterogeneous inference."""

from __future__ import annotations

import io
import os

import torch

from src.models.qwen4_exp import runtime


def test_parse_cpu_list() -> None:
    assert runtime._parse_cpu_list("0-2,5,8-9\n") == [0, 1, 2, 5, 8, 9]


def test_cpu_affinity_splits_ranks_on_same_numa_node(monkeypatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(runtime, "_cuda_numa_node", lambda device: int(device.index or 0) // 2)
    monkeypatch.setattr(
        runtime,
        "_physical_core_groups",
        lambda cpus: [[cpu] for cpu in sorted(cpus)],
    )

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path.endswith("node0/cpulist"):
            return io.StringIO("0-7")
        if path.endswith("node1/cpulist"):
            return io.StringIO("8-15")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert runtime._cpu_affinity_for_device(torch.device("cuda:0"), 0, 4) == [0, 1, 2, 3]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:1"), 1, 4) == [4, 5, 6, 7]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:2"), 2, 4) == [8, 9, 10, 11]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:3"), 3, 4) == [12, 13, 14, 15]
