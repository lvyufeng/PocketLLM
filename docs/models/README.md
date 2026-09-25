# PocketLLM model support

PocketLLM uses model-specific runtimes rather than treating every checkpoint as the same Transformer. The table below describes the current repository state.

## Status definitions

- **Validated generation:** the complete model has generated tokens from a real checkpoint on the stated hardware.
- **Validated text generation:** tokenizer/chat framing and full-model text-in/text-out generation are covered.
- **CLI only:** generation is available through a command-line or benchmark entrypoint but is not wired to the OpenAI-compatible server.
- **Experimental:** functionality exists, but performance, determinism, or output parity has an explicit caveat.
- **Inspect only:** metadata/tensor validation exists without a complete generation runtime.

## Support matrix

| Model | Architecture | Format | Runtime | Generation | OpenAI server | Guide | Design doc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | MLA + sparse/indexed attention + MoE | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | PyTorch and C++/CUDA | Validated | Safetensors C++ and PyTorch paths | [Guide](deepseek-v4.md) | [Design](../architecture/deepseek_v4_design.md) |
| MiniMax-M2.7 | GQA + 256-expert MoE | GGUF `UD-IQ1_M` | PyTorch orchestration + raw-block CUDA | Validated TP4 | No dedicated adapter | [Guide](minimax-m2.7.md) | [Design](../architecture/minimax_m2_7_design.md) |
| GLM-5.2 | DSA/MLA-indexed attention + dense prefix + MoE | GGUF `UD-Q2_K_XL` | PyTorch orchestration + raw-block CUDA | Validated text generation | No dedicated adapter | [Guide](glm-5.2.md) | [Design](../architecture/glm_5_2_design.md) |
| Qwen3.8-27B-FP8 | 48 Gated DeltaNet + 16 GQA layers, dense MLP | Safetensors FP8 E4M3 | Native C++/CUDA | Validated TP4 text runtime and server | Validated native C++ text server | [Guide](qwen3.8-27b-fp8.md) | [Design](../architecture/qwen3_8_27b_fp8_design.md) |
| Qwen3.8-27B-NVFP4 | Same text architecture as the FP8 checkpoint | Safetensors mixed NVFP4 group-16 + FP8 per-channel | Native C++/CUDA | Validated TP2 text CLI | Shared native text path; no dedicated serving benchmark | [Guide](qwen3.8-27b-nvfp4.md) | [Design](../architecture/qwen3_8_27b_nvfp4_design.md) |
| Qwen3.8-27B (official BF16) | Same text architecture as the FP8 checkpoint | Safetensors BF16, vision tower bundled | Native C++/CUDA | Inspect only: TP audit validated, generation unvalidated | Not validated: generation is unvalidated | [Guide](qwen3.8-27b-bf16.md) | [Design](../architecture/qwen3_8_27b_bf16_design.md) |
| Ternary-Bonsai-2-27B | The Qwen3.8-27B text architecture: 48 Gated DeltaNet + 16 GQA layers, dense MLP | GGUF `PTQ1_0` (GGML type 143, **1.75 bits a weight**, 5.53 GiB), a Hadamard rotation declared in the file | Native C++/CUDA, **one** card | Validated on the release behind the server: a 4,096-token prompt prefills in 6.44 s (**636.0 tok/s**, level with the upstream runtime's 642.5) and decodes at 25.9 tok/s, 6,566 MiB of weights and runtime on the card, 245,760 tokens of context at an FP16 KV cache | `pocketllm serve` picks it from the file's own `general.architecture`, no flag: chat, completions, streaming, cancel and metrics; batching opt-in and measured | [Guide](ternary-bonsai-2-27b.md) | [Design](../architecture/bonsai_2_27b_design.md) |
| DeepSeek-V4.1-Flash | 20-layer causal encoder + 20-layer decoder, CSA2 shared-KV attention, Engram, MoE, ViT | Safetensors FP8 E4M3 dense, FP4 E2M1 experts | `pocketllm serve --backend v41`: host PyTorch over a mapped checkpoint, the dense tree and the packed fp4 experts on the cards, one process a rank | Validated TP4 text generation behind the server: 137–141 tok/s prefill and 4.45–4.53 tok/s decode at a 1364-token prompt, 64 greedy tokens, one request at a time | `--backend v41` (`v41` adapter): OpenAI-compatible chat, completions and streaming, serialized — no batching and no MTP | [Guide](deepseek-v4.1-flash.md) | [Design](../architecture/deepseek_v4_1_flash_design.md) |
| MiMo-V2.6-Flash | 9 global-attention + 39 sliding-window layers with a sink, MoE | Safetensors FP8 E4M3 dense, MXFP4 experts, BF16 attention output | `device_model.py`, `device_experts.py` and `ep.py`: the whole 48-layer text backbone out of a host-resident expert bank, a token a step and a prompt a chunk through a KV cache, the experts dealt out over four ranks, the attention divided along the checkpoint's own four-way `qkv_proj` partition and joined by an all-gather | Validated on the release: a 262144-token prompt through four ranks at **104.04 tok/s** (the same prompt with the attention replicated is 48.37), four ranks byte-identical and the argmax matching the float32 host reference at full depth, 10.21 GiB on the card; a decode step at that depth is **180.0 ms — 5.56 tok/s** — **156.3 ms, 6.40 tok/s** at a short context, and 117.3 ms, 8.53 tok/s, with each routed layer's hottest experts held on the card (1.64 tok/s on one card); no batching, and the attention and dense linears are torch above 16384 keys | `--backend mimo`: OpenAI-compatible chat, completions, streaming, cancel and metrics on four ranks, one request at a time, with cross-request prefix caching on by default | [Guide](mimo-v2.6-flash.md) | [Design](../architecture/mimo_v2_6_flash_design.md) |

The **Design doc** column points at the engineering record for that model: what its runtime does and why, the measurements behind each design choice, and the probes those numbers come from. It lives under `docs/architecture/`, not next to the guides, because it is written for changing the runtime rather than for running it.

## Shared baseline

The headline results use 4×RTX 2080 Ti 22 GiB unless the model page says otherwise. Model TPS numbers are not directly comparable unless their checkpoint, prompt, runtime, warm state, and measurement convention match. See [Benchmarking](../guides/benchmarking.md).

## Adding or updating a model page

Every model has two documents, and they have different readers.

**`docs/models/<model>.md` is the guide.** Someone who wants to run the checkpoint reads it, so it holds only what they need, in this order:

1. What the model is, and four spec bullets — backend, parallelism, context, validated hardware.
2. **Overview** — the architecture table, and the one or two ideas that explain how PocketLLM runs it.
3. **Run it** — the serve command and a `curl`, the serving options table, and the non-server entrypoints.
4. **What is supported** — one table, including the rows that say "not implemented".
5. **Performance** — one summary table, with the conditions stated above it.
6. **Hardware and memory**, then **Known limitations**.
7. **Where the detail is** — links onward, ending with this matrix.

**`docs/architecture/<model>_design.md` is the record.** It holds the kernels, the intermediate measurements, the rejected alternatives, the correctness evidence, the reproduction commands and the source-file inventory. It opens by saying which of its numbers are single readings and which are differences, and it points back at the guide.

Do not put the record on the guide page, and do not claim a capability the record does not establish: never infer runtime support from model metadata alone, because a checkpoint may advertise a long context, a vision tower or an MTP layer that PocketLLM does not execute.
