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
| [Qwen quantized KV cache at 65K (TG512)](qwen_kv_cache_65k_tg512.md) | The authoritative quantized-KV table at 65K/TG512, including corrections to two earlier throughput claims. |
| [Native C++ OpenAI concurrency validation](cpp_openai_concurrency_validation.md) | The end-to-end HTTP acceptance test and the vLLM head-to-head that followed it. |
| [Ascend 910A attention and its measured ceilings](ascend_attention_optimization.md) | The Cube (`Mmad`) GQA attention operator, the three-way decode dispatch, the prefill phase table, and the bandwidth, all-reduce and `aclrtMemcpy` limits every remaining target has to clear. |
| [Ascend 910A TP collective overlap](ascend_tp_collective_overlap.md) | Why the overlapped all-reduce must slice by row count rather than a constant, the crossover sweep, and the profiles showing prefill is 84-87% collectives at both 512 and 4096 tokens. |

The currently authoritative per-model numbers live in the
[model guides](../models/README.md); pages under
[Archive](../archive/phase2-phase3/index.md) are historical records whose numbers
have been superseded.
