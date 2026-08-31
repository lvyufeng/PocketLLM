"""Profiler aggregation tests for Qwen4-Exp."""

from __future__ import annotations

from src.models.qwen4_exp.profiler import Profiler


def test_aggregate_sums_layer_children() -> None:
    profiler = Profiler()
    for layer in range(2):
        with profiler.scope("prefill"):
            with profiler.scope(f"layer_{layer}"):
                with profiler.scope("attention"):
                    pass
                with profiler.scope("moe"):
                    pass

    aggregated = profiler.aggregate()
    assert aggregated["attention"].count == 2
    assert aggregated["moe"].count == 2
    assert "attention" in profiler.aggregate_report()
    assert "moe" in profiler.aggregate_report()


def test_aggregate_keeps_external_attention_phases() -> None:
    profiler = Profiler()
    with profiler.scope("prefill"):
        with profiler.scope("layer_3"):
            profiler.add_external("attention_core", 0.125)

    aggregated = profiler.aggregate()
    assert aggregated["attention_core"].count == 1
    assert aggregated["attention_core"].total_s == 0.125
