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
| [Auditing the DeepSeek-V4.1-Flash shards while the checkpoint is arriving](deepseek_v4_1_shard_audit.md) | A header audit that distinguishes a missing shard from a wrong one: 44.0% of a 48-shard checkpoint readable as it downloads, the 8 checks that are still undecided and the shards each waits on, and the tensor shapes and per-layer counts the landed shards assert. |

The currently authoritative per-model numbers live in the
[model guides](../models/README.md); pages under
[Archive](../archive/phase2-phase3/index.md) are historical records whose numbers
have been superseded.
