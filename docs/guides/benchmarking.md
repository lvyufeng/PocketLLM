# Benchmarking and reporting rules

PocketLLM reports prefill and decode separately because they stress different parts of the runtime. Prefill is a multi-token operation dominated by projection/GEMM, attention over the prompt, and expert batching. Decode is a single-token latency path dominated by recurrent state updates, KV attention, expert staging, kernel launch overhead, and TP communication.

## Required metadata

A benchmark result should record:

- model name and exact checkpoint/quantization variant;
- PocketLLM commit or release;
- runtime (`cpp_engine`, PyTorch resident, heterogeneous GGUF, etc.);
- GPU model, per-card memory, GPU count, TP/EP world size, PCIe/NVLink topology;
- CPU model, NUMA layout, system RAM, CUDA/driver/runtime versions when relevant;
- prompt token count, generated token count, context length, and tokenizer source;
- cold/warm state, page-cache prewarm, expert-cache state, and relevant environment switches;
- prefill wall time and tokens/s;
- decode wall time and tokens/s for generated tokens after the first result;
- peak GPU memory per rank and host/pinned memory when it is material;
- token parity, numerical error, or an explicit statement that no reference comparison was run.

## Timing convention

PocketLLM's standard timed generation path measures the first model result as prefill and measures subsequent single-token forwards as decode:

```text
prefill_tps = prompt_tokens / prefill_seconds
decode_tps  = decode_tokens / decode_seconds
```

The first generated token is produced by the prompt forward and therefore belongs to the prefill phase. `decode_tokens` counts only later generated tokens. A benchmark that uses another convention must say so explicitly.

Do not report a combined tokens/s number as a replacement for these two fields. Combined throughput is useful only as an additional end-to-end figure.

## Comparison rules

Results are directly comparable only when the checkpoint, quantization, runtime, prompt, generation length, hardware, TP/EP layout, and warm/cold policy match. In particular:

- PyTorch heterogeneous expert staging, GGUF raw-block TP, GPU-resident FP4, and Qwen GPU-resident FP8 are different execution paths.
- A 256-token prefill microbenchmark is not a long-context prefill result.
- A synthetic kernel benchmark is evidence about a kernel, not about model-level TPS.
- A first-token match does not prove that a quantized KV-cache path remains sequence-identical for later tokens.
- Experimental opt-in switches must be reported with their values and must not silently replace the default baseline.
- If rank outputs differ only by floating-point reduction order, report the max error and token-level result rather than claiming exact tensor identity.

## Recommended table

Use this shape for model pages:

| Model/checkpoint | Runtime | Hardware / parallelism | Prompt | Decode tokens | Prefill tok/s | Decode tok/s | Peak GPU/rank | Correctness |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Then add a short paragraph describing warmup, cache state, and the source script or command.

## PocketLLM baseline machine

Many repository measurements use:

- 4× NVIDIA GeForce RTX 2080 Ti, 22 GiB each;
- Turing architecture, no native BF16/FP8/FP4 tensor cores;
- PCIe Gen3 and no NVLink;
- dual-socket Intel Xeon E5-2696 v4, 1 TiB system memory;
- one process/rank per GPU, usually TP4.

These constraints are part of the result. Moving to a newer GPU, NVLink system, or different NUMA placement can change the bottleneck and invalidate an apparent A/B comparison.

## Reproducibility and honesty rules

1. Prefer real checkpoints and real prompts for end-to-end claims.
2. Keep short and long prompt cases separate.
3. Run repeated measurements serially when comparing single-request latency.
4. Preserve the fastest known command before changing an optimization switch.
5. Report regressions and disabled experiments alongside wins.
6. Keep model architecture specifications separate from runtime support status.
7. Link the test, script, commit, or analysis note that produced each non-trivial number.
