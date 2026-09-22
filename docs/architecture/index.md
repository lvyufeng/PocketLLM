# Architecture

How PocketLLM is put together, and how it compares to the serving stacks it is
usually measured against. Two things are worth knowing before reading:

- The three design documents — [Backend unification design](backend_unification_design.md),
  [PocketLLM refactor analysis](pocketllm_refactor_analysis_2026_09.md) and the
  [cpp_engine multi-backend plan](cpp_engine_multi_backend_plan.md) — are three
  drafts of the same proposal at different points in time. Where they disagree,
  the newest one wins; none of them describes work that is fully complete.
- [cpp_engine vs vLLM/SGLang](vllm_sglang_comparison.md) is a pre-Phase-1 baseline.
  For the current comparison read
  [PocketLLM vs vLLM vs SGLang](vllm_sglang_architecture_analysis.md).

| Document | What it covers |
| --- | --- |
| [Backend unification design](backend_unification_design.md) | The original proposal for one engine over swappable device backends. |
| [Cross-request prefix caching on V4.1](v41_prefix_cache.md) | How a served V4.1 request resumes a stored prefix instead of forwarding the prompt again: the buffers a snapshot is, the two anchors, the budget, and the metrics. |
| [cpp_engine multi-backend refactor plan](cpp_engine_multi_backend_plan.md) | The refactor plan that followed it: `core/` / `engine/` / `backends/` layering without giving up per-hardware kernels. |
| [PocketLLM refactor analysis (2026-09)](pocketllm_refactor_analysis_2026_09.md) | A vLLM/SGLang comparison that motivates the dual-backend design, against master `e59d5d3`. *(Chinese)* |
| [Feature roadmap for old hardware](pocketllm_roadmap_old_hardware.md) | What is worth building for 2080 Ti (SM75) and Ascend 910A, and what is not. *(Chinese)* |
| [PocketLLM vs vLLM vs SGLang architecture analysis](vllm_sglang_architecture_analysis.md) | The current comparison: scheduling, paged KV, batching, and serving surface. *(Chinese)* |
| [cpp_engine vs vLLM/SGLang comparison](vllm_sglang_comparison.md) | The earlier comparison, retained as the pre-Phase-1 baseline. |
| [Ascend 910A performance roadmap](ascend_performance_roadmap.md) | Where the prefill and decode targets actually stand after measurement, and the ranked next steps — including why the decode target needs quantization rather than tuning. |

The invariants these designs exist to protect are stated in the repository's
`CLAUDE.md`: kernels stay behind the C ABI, and backend selection happens at
configure time rather than in shared code.
