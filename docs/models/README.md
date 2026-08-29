# PocketLLM model support

PocketLLM uses model-specific runtimes rather than treating every checkpoint as the same Transformer. The table below describes the current repository state.

## Status definitions

- **Validated generation:** the complete model has generated tokens from a real checkpoint on the stated hardware.
- **Validated text generation:** tokenizer/chat framing and full-model text-in/text-out generation are covered.
- **CLI only:** generation is available through a command-line or benchmark entrypoint but is not wired to the OpenAI-compatible server.
- **Experimental:** functionality exists, but performance, determinism, or output parity has an explicit caveat.
- **Inspect only:** metadata/tensor validation exists without a complete generation runtime.

## Support matrix

| Model | Architecture | Format | Runtime | Generation | OpenAI server | Detailed guide |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | MLA + sparse/indexed attention + MoE | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | PyTorch and C++/CUDA | Validated | Safetensors C++ and PyTorch paths | [DeepSeek-V4](deepseek-v4.md) |
| MiniMax-M2.7 | GQA + 256-expert MoE | GGUF `UD-IQ1_M` | PyTorch orchestration + raw-block CUDA | Validated TP4 | No dedicated adapter | [MiniMax-M2.7](minimax-m2.7.md) |
| GLM-5.2 | DSA/MLA-indexed attention + dense prefix + MoE | GGUF `UD-Q2_K_XL` | PyTorch orchestration + raw-block CUDA | Validated text generation | No dedicated adapter | [GLM-5.2](glm-5.2.md) |
| Qwen3.8-27B-FP8 | 48 Gated DeltaNet + 16 GQA layers, dense MLP | Safetensors FP8 E4M3 | Native C++/CUDA | Validated TP4 text CLI | Not implemented | [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) |
| Qwen3.8-27B-NVFP4 | Same text architecture as the FP8 checkpoint | Safetensors mixed NVFP4 group-16 + FP8 per-channel | Native C++/CUDA | Validated TP2 text CLI | Not implemented | [Qwen3.8-27B-NVFP4](qwen3.8-27b-nvfp4.md) |
| Qwen3.8-27B (official BF16) | Same text architecture as the FP8 checkpoint | Safetensors BF16, vision tower bundled | Native C++/CUDA | Inspect only: TP audit validated, generation unvalidated | Not implemented | [Qwen3.8-27B BF16](qwen3.8-27b-bf16.md) |

## Shared baseline

The headline results use 4×RTX 2080 Ti 22 GiB unless the model page says otherwise. Model TPS numbers are not directly comparable unless their checkpoint, prompt, runtime, warm state, and measurement convention match. See [Benchmarking](../benchmarking.md).

## Adding or updating a model page

Use the same sections as the existing pages:

1. Runtime status
2. Checkpoint/model specification
3. Implemented execution path
4. Validated performance
5. Correctness and precision
6. Reproduction
7. Known limitations
8. Evidence and related notes

Never infer runtime support from model metadata alone. A checkpoint may advertise a long context, vision tower, or MTP layer that PocketLLM does not execute.
