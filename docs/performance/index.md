# Performance

Measured behaviour, reported separately for prefill and decode. Every page here is
a run record: hardware, checkpoint, commit, and invocation are stated with the
numbers. Read [Benchmarking and reporting rules](../guides/benchmarking.md) first
if you intend to compare any of them, and do not carry a decode figure to a
different generation length — a `TG4` measurement says nothing about `TG512`.

| Document | What it records |
| --- | --- |
| [DSpark speculative decoding with adaptive draft-length gating](dspark.md) | The DeepSeek-V4-Flash draft module, why the fixed accept threshold was falsified, and the online-calibrated gating that replaced it. |
| [FlashMemory 1M context](flashmemory_1m_context.md) | The FlashMemory + KV_SWAP design for long-context memory reduction, and its current implementation status. *(Chinese)* |
| [MiniMax-M2 decode bottleneck analysis](minimax_decode_bottleneck_analysis.md) | Per-phase profiling of MiniMax-M2 decode and prefill on TP4, and which phases the optimizations then targeted. |
| [Qwen4-Exp heterogeneous TP4 performance](qwen4_exp_performance.md) | Moving Qwen4-Exp experts to host memory: measured prefill gain, and why the real floor was the disk rather than PCIe. |
| [CPU offload and prefetch: the measured ceiling](cpu_offload_profile.md) | Layer-granularity offload measured rather than planned: the H2D and overlap ceilings, the storage rate behind them, and why #156's decode and prefetch targets are unreachable on this host. |
| [Qwen quantized KV cache at 65K (TG512)](qwen_kv_cache_65k_tg512.md) | The authoritative quantized-KV table at 65K/TG512, including corrections to two earlier throughput claims. |
| [Native C++ OpenAI concurrency validation](cpp_openai_concurrency_validation.md) | The end-to-end HTTP acceptance test and the vLLM head-to-head that followed it. |
| [Native C++ OpenAI tool-calling acceptance](cpp_openai_tool_acceptance.md) | Tool calling driven through a second turn, the `openai` SDK and `langchain-openai`, and the sidecar templating defect the first real two-turn request exposed. |
| [Ascend 910A attention and its measured ceilings](ascend_attention_optimization.md) | The Cube (`Mmad`) GQA attention operator, the three-way decode dispatch, the prefill phase table, and the bandwidth, all-reduce and `aclrtMemcpy` limits every remaining target has to clear. |
| [DeepSeek-V4 PersistentEngine serial baseline](deepseek_v4_serial_baseline.md) | The serial prefill/decode baseline for the native engine on DeepSeek-V4-Flash-0731, the environment ablation behind it, and why `--max-batch-size 8` currently changes nothing. |
| [Auditing the DeepSeek-V4.1-Flash shards from arrival to complete](deepseek_v4_1_shard_audit.md) | A header audit that distinguishes a missing shard from a wrong one, run at four points of the download: 44.0% of the checkpoint readable at 20 shards, four Engram checks still undecided at 46, and 39 of 39 passed with nothing undecided on all 48 — plus the tensor shapes and per-layer counts the shards assert. |
| [DeepSeek-V4.1-Flash: what the released checkpoint costs to run on one host](deepseek_v4_1_flash_host_run.md) | The byte census over all 96,085 tensors, the load report, what the tree occupies at TP4, the Engram gather's cold/warm/resident costs, what a generated token costs and what it is made of — 15 to 42 s on the host, 99.7% of it the fp4-to-bf16 expansion — and the measured PCIe and card-side rates that make the four cards the cheaper half of this model. |
| [DeepSeek-V4.1-Flash: the routed experts on the four cards](deepseek_v4_1_flash_device_experts.md) | The expert-parallel split over four 2080 Ti: the static 2/2/1/1 deal, the fp4 kernel's parity against the host expert, a measured 1.06–1.14 s per decode step phase by phase — then the dense tree cut across the same cards, one process per card, at 722–747 ms a step; and the honest negatives and caveats behind both: the four cards are worth 0.1–0.2 s of the expert step, because staging is host work and the largest term; the launch was four kernels serialized rather than one plus copies, worth 3.4× once the drain moved out of the card loop; and every figure is a warm-page-cache one, 19× worse on the same row once the cache is emptied — except with the resident bank attached, where the same emptied row is 782.9 ms against 17.01 s (21.7×) and warm is a wash, because `_stage` is a memcpy into the pinned arena whatever it reads from. |

The currently authoritative per-model numbers live in the
[model guides](../models/README.md); pages under
[Archive](../archive/phase2-phase3/index.md) are historical records whose numbers
have been superseded.
