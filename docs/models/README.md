# PocketLLM model support

PocketLLM uses model-specific runtimes rather than treating every checkpoint as the same Transformer. The table below describes the current repository state.

## Status definitions

A status is a claim about evidence, so each one names the evidence it stands on rather than
asserting a level. The record behind a row is that model's page, and where a run needed its own
document it is under [performance](../performance/index.md).

| Status | What it means, and what proves it |
| --- | --- |
| **Text + server** | The complete model has generated tokens from a real checkpoint on the stated hardware, and `pocketllm serve` answers OpenAI chat and completions against it. Proven by the run record on the model page — for V4.1 and MiMo also [behind the server](../performance/deepseek_v4_1_flash_served_gate.md), for Qwen by [the concurrency validation](../performance/cpp_openai_concurrency_validation.md). |
| **Text, CLI only** | Full-model text-in/text-out generation is covered through a CLI or benchmark entrypoint. There is no OpenAI adapter for it. |
| **Experimental** | Functionality exists with an explicit caveat on performance, determinism or output parity. The caveat is the `## Known limitations` entry that carries it. |
| **Inspect only** | Metadata and tensor validation exist without a complete generation runtime: the audit is the evidence, not generated text. |

## Support matrix

The headline result for each model is on its page, together with the conditions it was taken under.
This table is the runtime status and nothing else, so a row stays scannable.

| Model | Architecture | Format | Runtime | Status | Design |
| --- | --- | --- | --- | --- | --- |
| [DeepSeek-V4-Flash](deepseek-v4.md) | MLA + sparse attention + MoE | Safetensors FP4/FP8, GGUF Q2 | PyTorch and C++/CUDA, TP4 | Text + server | [Design](../architecture/deepseek_v4_design.md) |
| [MiniMax-M2.7](minimax-m2.7.md) | GQA + 256-expert MoE | GGUF `UD-IQ1_M` | Raw-block CUDA, TP4 | Text, CLI only | [Design](../architecture/minimax_m2_7_design.md) |
| [GLM-5.2](glm-5.2.md) | DSA/MLA-indexed + dense prefix + MoE | GGUF `UD-Q2_K_XL` | Raw-block CUDA, TP4 | Text, CLI only | [Design](../architecture/glm_5_2_design.md) |
| [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) | 48 Gated DeltaNet + 16 GQA | Safetensors FP8 E4M3 | C++/CUDA, TP4 | Text + server | [Design](../architecture/qwen3_8_27b_fp8_design.md) |
| [Qwen3.8-27B-NVFP4](qwen3.8-27b-nvfp4.md) | Same text architecture | Safetensors NVFP4 + FP8 | C++/CUDA, TP2 | Text, CLI only | [Design](../architecture/qwen3_8_27b_nvfp4_design.md) |
| [Qwen3.8-27B (official BF16)](qwen3.8-27b-bf16.md) | Same text architecture | Safetensors BF16, vision tower | C++/CUDA | Inspect only | [Design](../architecture/qwen3_8_27b_bf16_design.md) |
| [Ternary-Bonsai-2-27B](ternary-bonsai-2-27b.md) | Same text architecture | GGUF `PTQ1_0`, 1.75 bits a weight | C++/CUDA, **one card**, no flag | Text + server | [Design](../architecture/bonsai_2_27b_design.md) |
| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash.md) | Encoder-decoder, CSA2 shared-KV, MoE | Safetensors FP8 + FP4 | `--backend v41`, host PyTorch, TP4 | Text + server | [Design](../architecture/deepseek_v4_1_flash_design.md) |
| [MiMo-V2.6-Flash](mimo-v2.6-flash.md) | 9 global + 39 sliding-window, MoE | Safetensors FP8 + MXFP4 | `--backend mimo`, host expert bank, TP4 | Text + server | [Design](../architecture/mimo_v2_6_flash_design.md) |
| [Xing4.0-29B-A4B](xing4.0-29b-a4b.md) | MLA + matrix hyper-connection, 64-expert MoE | GGUF `IQ4_NL` | `--backend xing4`, **one card**, experts resident | Text + server | [Design](../architecture/xing4_0_29b_a4b_design.md) |

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
