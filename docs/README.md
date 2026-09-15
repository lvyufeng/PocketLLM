# PocketLLM documentation

This directory contains model-specific support notes, reproducible benchmark definitions, and engineering analyses. Start with the model matrix when you need to know whether a checkpoint is inspectable, runnable, or benchmarked end to end.

## Where things live

| Directory | What it holds |
|---|---|
| [`guides/`](guides/) | Rules and procedures: benchmark reporting, the native engine API, the PyPI release flow, Ascend platform notes |
| [`architecture/`](architecture/) | Engine design, refactor plans, roadmaps, and PocketLLM-vs-vLLM/SGLang comparisons |
| [`performance/`](performance/) | Measured results and bottleneck analyses for capabilities that are live today |
| [`models/`](models/) | Per-checkpoint guides and the support matrix |
| [`migration/`](migration/) | Breaking-change migration notes |
| [`reports/`](reports/) | Rendered long-form reports |
| [`archive/phase2-phase3/`](archive/phase2-phase3/) | Completed Phase 2/3 records, kept for their measurement context |

Every document in this tree is listed below, so nothing is reachable only by guessing a filename.

## Getting started

- [Project home](../README.md) · [中文首页](../README_CN.md)
- [C++/CUDA engine notes](../cpp_engine/README.md)
- [Migration: `dsv4` → `pocket`](migration/dsv4-to-pocket-rename.md) — breaking rename of the
  namespace, build targets, executable, and every `DSV4_*` environment variable

## Guides

- [Benchmarking and reporting rules](guides/benchmarking.md) — required metadata and the prefill/decode separation every number in this tree is expected to follow
- [Native engine API and backend guide](guides/pocketllm_api.md) — the `pocketllm` control plane over the Torch and C++ execution planes
- [PyPI release guide](guides/pypi_release.md) — the single source of truth for cutting a release, and where the credentials live
- [Ascend SoC generations](guides/ascend_soc_generations.md) — why `910B` with no trailing digit is first generation, and how to resolve the generation you are actually on

## Model guides

- [Model support matrix](models/README.md)
- [DeepSeek-V4](models/deepseek-v4.md)
- [DeepSeek-V4 GGUF Q2 single-GPU history](models/deepseek-v4-gguf-q2-single-gpu.md)
- [MiniMax-M2.7](models/minimax-m2.7.md)
- [GLM-5.2](models/glm-5.2.md)
- [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md)

The same runtime also covers [Qwen3.8-27B BF16](models/qwen3.8-27b-bf16.md) (inspect only) and
[Qwen3.8-27B NVFP4](models/qwen3.8-27b-nvfp4.md); the support matrix carries their status.

## Architecture

- [Backend unification design](architecture/backend_unification_design.md) — the unified-architecture
  proposal written after the Phase 3 batch refactor
- [cpp_engine multi-backend refactor plan](architecture/cpp_engine_multi_backend_plan.md) — the
  decisions that the `core/` + `engine/` + `backends/` layout implements
- [PocketLLM refactor analysis (2026-09)](architecture/pocketllm_refactor_analysis_2026_09.md) —
  the same design analysed against vLLM/SGLang at master `e59d5d3`
- [Old-hardware roadmap](architecture/pocketllm_roadmap_old_hardware.md) — the remaining gaps for
  2080 Ti and Ascend 910A; this is the anchor document for issues #151-#156
- [vLLM/SGLang architecture analysis](architecture/vllm_sglang_architecture_analysis.md) — the
  current comparison of schedulers, KV cache, TP communication, quantization, and speculative decoding
- [cpp_engine vs vLLM/SGLang comparison](architecture/vllm_sglang_comparison.md) — earlier
  comparison; its pre-Phase-1 sections are labelled as a historical baseline

Sections here overlap by design: the three design documents above cover the same proposal at
different points in time, and the two comparison documents bracket Phase 1. Read the newest of each
pair first — `pocketllm_refactor_analysis_2026_09.md` and `vllm_sglang_architecture_analysis.md`.

## Performance

- [DSpark speculative decoding](performance/dspark.md) — adaptive draft-length gating, and why the
  accept rate is the whole story
- [FlashMemory 1M context](performance/flashmemory_1m_context.md) — host/GPU chunk swapping and the
  resulting memory budget
- [MiniMax decode bottleneck analysis](performance/minimax_decode_bottleneck_analysis.md)
- [Qwen4-Exp heterogeneous TP4 performance](performance/qwen4_exp_performance.md)
- [Qwen quantized KV cache at 65K (TG512)](performance/qwen_kv_cache_65k_tg512.md) — including a
  correction to two throughput claims that were not backed by a run record
- [Native C++ OpenAI concurrency validation](performance/cpp_openai_concurrency_validation.md) —
  the end-to-end HTTP acceptance result for issue #106 and the vLLM concurrency head-to-head

## Historical reports

- [DeepSeek-V4 on 4×RTX 2080 Ti](reports/dsv4_2080ti_report.pdf)

## Archive: Phase 2 and Phase 3

These record work that is finished. They retain the measurement context and conclusions from the
experiment that produced them; current support status and the latest comparable figures belong in
the model pages and the performance notes above.

| Document | Covers |
|---|---|
| [Phase 2 validation results](archive/phase2-phase3/phase2_validation_results.md) | FlashMemory + KV_SWAP validation |
| [cpp_engine batching (Phase 3.1)](archive/phase2-phase3/cpp_engine_batching_phase3_1.md) | The Phase 3.1 implementation plan for batch mode |
| [Phase 3.1 progress report](archive/phase2-phase3/phase3_1_progress_report.md) | Phase 3.1 status tracking |
| [Phase 3.1 completion summary](archive/phase2-phase3/phase3_1_completion_summary.md) | Phase 3.1 results |
| [Phase 3.2 implementation plan](archive/phase2-phase3/phase3_2_implementation_plan.md) | Multi-slot KV cache design |
| [Phase 3.2 completion summary](archive/phase2-phase3/phase3_2_completion_summary.md) | Phase 3.2 results |
| [Phase 3.3 implementation plan](archive/phase2-phase3/phase3_3_implementation_plan.md) | Slot ID threading and batch API integration |
| [Phase 3.3 completion summary](archive/phase2-phase3/phase3_3_completion_summary.md) | Phase 3.3 results |
| [Phase 3.4 completion summary](archive/phase2-phase3/phase3_4_completion_summary.md) | QwenBatchScheduler |
| [CppBackend automatic tensor parallelism](archive/phase2-phase3/cpp_backend_auto_tp.md) | The Phase 3.5 auto-TP design |
| [Phase 3.5 auto-TP results](archive/phase2-phase3/phase3_5_auto_tp_results.md) | Phase 3.5 auto-TP implementation results |
| [Phase 3.5 benchmark guide](archive/phase2-phase3/phase3_5_benchmark_guide.md) | How the Phase 3.5 validation was run |
| [Phase 3.5 performance report](archive/phase2-phase3/phase3_5_performance_report.md) | Phase 3.5 performance figures |
| [Phase 3.5 validation results](archive/phase2-phase3/phase3_5_validation_results.md) | Phase 3.5 acceptance results |
