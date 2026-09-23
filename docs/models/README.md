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
| Qwen3.8-27B-FP8 | 48 Gated DeltaNet + 16 GQA layers, dense MLP | Safetensors FP8 E4M3 | Native C++/CUDA | Validated TP4 text runtime and server | Validated native C++ text server | [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) |
| Qwen3.8-27B-NVFP4 | Same text architecture as the FP8 checkpoint | Safetensors mixed NVFP4 group-16 + FP8 per-channel | Native C++/CUDA | Validated TP2 text CLI | Shared native text path; no dedicated serving benchmark | [Qwen3.8-27B-NVFP4](qwen3.8-27b-nvfp4.md) |
| Qwen3.8-27B (official BF16) | Same text architecture as the FP8 checkpoint | Safetensors BF16, vision tower bundled | Native C++/CUDA | Inspect only: TP audit validated, generation unvalidated | Not validated: generation is unvalidated | [Qwen3.8-27B BF16](qwen3.8-27b-bf16.md) |
| DeepSeek-V4.1-Flash | 20-layer causal encoder + 20-layer decoder, CSA2 shared-KV attention, Engram, MoE, ViT | Safetensors FP8 E4M3 dense, FP4 E2M1 experts | `pocketllm serve --backend v41`: host PyTorch over a mapped checkpoint, the dense tree and the packed fp4 experts on the cards, one process a rank | Validated TP4 text generation behind the server: 137–141 tok/s prefill and 4.45–4.53 tok/s decode at a 1364-token prompt, 64 greedy tokens, one request at a time | `--backend v41` (`v41` adapter): OpenAI-compatible chat, completions and streaming, serialized — no batching and no MTP | [DeepSeek-V4.1-Flash](deepseek-v4.1-flash.md) |
| MiMo-V2.6-Flash | 9 global-attention + 39 sliding-window layers with a sink, MoE | Safetensors FP8 E4M3 dense, MXFP4 experts, BF16 attention output | `device_model.py`, `device_experts.py` and `ep.py`: the whole 48-layer text backbone out of a host-resident expert bank, a token a step and a prompt a chunk through a KV cache, and the experts dealt out over four ranks | Validated on the release: 5.63 tok/s decode at a short context (177.6 ms a token on four ranks, 1.64 tok/s on one), 104.4 tok/s prefill at a 64k prompt and 48.32 at 256k (a 262144-token prompt, four ranks byte-identical, the argmax matching the float32 host reference at full depth); no batching, attention replicated | `--backend mimo`: OpenAI-compatible chat, completions, streaming, cancel and metrics on four ranks, one request at a time | [MiMo-V2.6-Flash](mimo-v2.6-flash.md) |

## Shared baseline

The headline results use 4×RTX 2080 Ti 22 GiB unless the model page says otherwise. Model TPS numbers are not directly comparable unless their checkpoint, prompt, runtime, warm state, and measurement convention match. See [Benchmarking](../guides/benchmarking.md).

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
