# Archive: Phase 2 and Phase 3

!!! warning "These are historical records"

    Everything in this section was written while Phase 2 and Phase 3 were being
    built, and it is kept for provenance rather than for reference. Several
    documents describe work as incomplete or in progress, and some contain
    throughput figures that later measurements corrected. **For the current state
    of a model use [Model guides](../../models/README.md); for current numbers use
    [Performance](../../performance/index.md).**

Eight of the fourteen records were written in Chinese and are marked below. The
rest are in English. Nothing here has been translated or rewritten — the pages are
as they were at the time, so that a claim in them can still be traced.

| Record | What it covers |
| --- | --- |
| [FlashMemory + KV_SWAP Phase 2 validation results](phase2_validation_results.md) | Runtime scoring and KV_SWAP behaviour at long context, measured 2026-06-14. *(Chinese)* |
| [CppBackend automatic tensor parallelism](cpp_backend_auto_tp.md) | The design for letting `CppBackend` launch TP the way vLLM/SGLang/PyTorch do, instead of requiring a launch script. |
| [cpp_engine batching (Phase 3.1)](cpp_engine_batching_phase3_1.md) | The design for adding continuous batching to `QwenEngine` for 2–8 concurrent requests without losing the single-request path. *(Chinese)* |
| [Phase 3.1 progress report](phase3_1_progress_report.md) | Mid-phase status while the batching framework was being compiled and tested. *(Chinese)* |
| [Phase 3.1 completion summary](phase3_1_completion_summary.md) | The batching framework complete and compiling. *(Chinese)* |
| [Phase 3.2 implementation plan](phase3_2_implementation_plan.md) | The multi-slot KV cache layout for `max_batch_size` 2/4/8. *(Chinese)* |
| [Phase 3.2 completion summary](phase3_2_completion_summary.md) | Multi-slot KV cache implemented and compiling. *(Chinese)* |
| [Phase 3.3 implementation plan](phase3_3_implementation_plan.md) | Moving `slot_id` out of global state into explicit parameters, and wiring the batch API. *(partly Chinese)* |
| [Phase 3.3 completion summary](phase3_3_completion_summary.md) | Slot ID threading and batch API integration complete. *(Chinese)* |
| [Phase 3.4 completion summary](phase3_4_completion_summary.md) | The `QwenBatchScheduler` implementation. |
| [Phase 3.5 auto-TP results](phase3_5_auto_tp_results.md) | Results for automatic tensor parallelism without manual process management. |
| [Phase 3.5 benchmark guide](phase3_5_benchmark_guide.md) | How to run the benchmark scripts that validate `QwenBatchScheduler` performance. |
| [Phase 3.5 performance report](phase3_5_performance_report.md) | The performance validation report, left in progress. |
| [Phase 3.5 validation results](phase3_5_validation_results.md) | Measured on `c8c3b15`: 4× RTX 2080 Ti, TP4, Qwen3.8-27B-FP8, 16-token prompts, 32 new tokens. |
