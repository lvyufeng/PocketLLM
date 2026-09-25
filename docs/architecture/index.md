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
| [MiMo-V2.6-Flash: design and measurements](mimo_v2_6_flash_design.md) | The per-model engineering record behind [the MiMo-V2.6-Flash model guide](../models/mimo-v2.6-flash.md): the heterogeneous four-rank runtime, its 262,144-token context, the expert copy that bounds a decode step, and every measurement behind a design choice. |
| [DeepSeek-V4.1-Flash: design and measurements](deepseek_v4_1_flash_design.md) | The per-model engineering record behind [the DeepSeek-V4.1-Flash model guide](../models/deepseek-v4.1-flash.md): the 475.24 GiB checkpoint's tensor inventory and audit, the CSA2 attention layers, the loader, the config schema, and every measurement behind a design choice. |
| [Qwen3.8-27B-FP8: design and measurements](qwen3_8_27b_fp8_design.md) | The per-model engineering record behind [the Qwen3.8-27B-FP8 model guide](../models/qwen3.8-27b-fp8.md): the C++ kernels, the three speculative-decoding paths, the prefix-reuse protocol, the DCP feasibility study, and the commands the numbers come from. |
| [Qwen3.8-27B-NVFP4: design and measurements](qwen3_8_27b_nvfp4_design.md) | The per-model engineering record behind [the NVFP4 model guide](../models/qwen3.8-27b-nvfp4.md): the mixed quantization scheme, the INT8 DP4A/WMMA kernels that consume it on Turing, the wide-N64 prefill tile, and the correctness evidence. |
| [Qwen3.8-27B BF16: design and measurements](qwen3_8_27b_bf16_design.md) | The per-model engineering record behind [the BF16 model guide](../models/qwen3.8-27b-bf16.md): multimodal config parsing, the dense BF16 weight map and its coverage rules, the TP4 shard contract, and what the host-only audit does and does not establish. |
| [DeepSeek-V4-Flash: design and measurements](deepseek_v4_design.md) | The per-model engineering record behind [the DeepSeek-V4 model guide](../models/deepseek-v4.md): the validated geometry, the three execution paths, the precision rules a parity claim has to respect, and the run records behind each number. |
| [GLM-5.2: design and measurements](glm_5_2_design.md) | The per-model engineering record behind [the GLM-5.2 model guide](../models/glm-5.2.md): the checkpoint's physical layout, the raw-block kernels, the decode profile's two hard floors, and the readings behind each milestone. |
| [MiniMax-M2.7: design and measurements](minimax_m2_7_design.md) | The per-model engineering record behind [the MiniMax-M2.7 model guide](../models/minimax-m2.7.md): the quantization layout, the IQ2_XXS and Q4_K/Q5_K kernels, the decode profile, and the layer scope each reading was taken at. |
| [Backend unification design](backend_unification_design.md) | The original proposal for one engine over swappable device backends. |
| [Cross-request prefix caching on V4.1](v41_prefix_cache.md) | How a served V4.1 request resumes a stored prefix instead of forwarding the prompt again: the buffers a snapshot is, the two anchors, the budget, and the metrics. |
| [Cross-request prefix caching on MiMo-V2.6-Flash](mimo_v2_6_flash_prefix_cache.md) | The same store over this model's cache: a ring stored whole and a global layer cut to its written prefix, the write head as part of the state, and why a resumed prefill here is not bit-exact. |
| [cpp_engine multi-backend refactor plan](cpp_engine_multi_backend_plan.md) | The refactor plan that followed it: `core/` / `engine/` / `backends/` layering without giving up per-hardware kernels. |
| [PocketLLM refactor analysis (2026-09)](pocketllm_refactor_analysis_2026_09.md) | A vLLM/SGLang comparison that motivates the dual-backend design, against master `e59d5d3`. *(Chinese)* |
| [Feature roadmap for old hardware](pocketllm_roadmap_old_hardware.md) | What is worth building for 2080 Ti (SM75) and Ascend 910A, and what is not. *(Chinese)* |
| [PocketLLM vs vLLM vs SGLang architecture analysis](vllm_sglang_architecture_analysis.md) | The current comparison: scheduling, paged KV, batching, and serving surface. *(Chinese)* |
| [cpp_engine vs vLLM/SGLang comparison](vllm_sglang_comparison.md) | The earlier comparison, retained as the pre-Phase-1 baseline. |
| [Ascend 910A performance roadmap](ascend_performance_roadmap.md) | Where the prefill and decode targets actually stand after measurement, and the ranked next steps — including why the decode target needs quantization rather than tuning. |

The invariants these designs exist to protect are stated in the repository's
`CLAUDE.md`: kernels stay behind the C ABI, and backend selection happens at
configure time rather than in shared code.
