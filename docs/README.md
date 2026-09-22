<div class="pll-hero">
<div class="pll-hero__eyebrow">Multi-backend LLM inference engine</div>
<h1 class="pll-hero__title">PocketLLM</h1>
<p class="pll-hero__tagline">
Run large language models on consumer multi-GPU systems — including the ones
everyone else stopped optimizing for.
</p>
<p class="pll-hero__badges">
<a href="https://pypi.org/project/pocketllm/"><img src="https://img.shields.io/pypi/v/pocketllm.svg" alt="PyPI version"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>
<p class="pll-hero__actions">
<a class="pll-btn pll-btn--primary" href="https://lvyufeng.github.io/PocketLLM/getting-started/">Get started</a>
<a class="pll-btn" href="https://github.com/lvyufeng/PocketLLM">View on GitHub</a>
</p>
</div>

**PocketLLM** is an experimental C++/CUDA and PyTorch inference stack for running
large language models on consumer multi-GPU systems. It combines model-specific
kernels, low-bit formats, tensor and expert parallelism, CPU/GPU placement, and
reproducible single-request benchmarks.

The project started with DeepSeek-V4 on 4×RTX 2080 Ti and now covers DeepSeek-V4,
MiniMax-M2.7, GLM-5.2, Qwen3.8-27B and DeepSeek-V4.1-Flash. It is **not** a
single universal backend: each model has a runtime matched to its architecture
and checkpoint format, and it does not trade away per-hardware kernel
optimization for portability.

Two of those models are served end to end over the OpenAI-compatible API:
**Qwen3.8-27B-FP8** through the native C++ runtime, and
**DeepSeek-V4.1-Flash** through `pocketllm serve --backend v41`.

!!! warning "Status"

    Research and engineering software. Every number on this site is a measurement
    from a specific checkpoint and hardware configuration, not a performance
    guarantee. Read [Benchmarking and reporting rules](guides/benchmarking.md)
    before comparing any two results.

## What PocketLLM provides

<div class="grid cards" markdown>

- **Model-specific inference paths**

    ---

    Hybrid attention, MLA, GQA, Gated DeltaNet, dense MLPs and routed MoE layers,
    each with the kernels its architecture actually needs.

- **Low-bit execution without expansion**

    ---

    FP4, FP8 E4M3, GGUF Q4/Q5/Q8, IQ1/IQ2/IQ3 and Q2 paths consume quantized
    blocks directly in the hot path. Raw weights are not expanded to a full FP32
    copy where it matters.

- **Consumer-GPU parallelism**

    ---

    TP4/NCCL execution on PCIe-connected GPUs with no NVLink, plus CPU/NUMA
    expert placement for checkpoints that do not fit in device memory.

- **Separate prefill and decode dispatch**

    ---

    Large-row kernels are optimized independently from the single-token latency
    path, so improving one does not cost the other.

- **Native C++/CUDA runtime**

    ---

    `cpp_engine/` covers the DeepSeek-V4 GGUF/Safetensors flows, Qwen3.8 FP8
    Safetensors text generation, and the validated Qwen OpenAI-compatible text
    server.

- **A host-PyTorch adapter for a checkpoint the cards cannot hold**

    ---

    `pocketllm serve --backend v41` runs DeepSeek-V4.1-Flash as four processes,
    one a card, over the 475 GiB checkpoint: the dense tree and the packed FP4
    experts execute on the GPUs while the routed experts read from a pinned host
    bank.

- **Inspection and validation tools**

    ---

    GGUF architecture and spec reports, Safetensors audits, tensor-shape checks,
    numerical parity tests and real-checkpoint benchmarks.

</div>

## Measured on 4×RTX 2080 Ti

Real checkpoints, PCIe Gen3, no NVLink, single requests, TP4 where applicable.
These are architecture-specific results and must not be averaged into one
PocketLLM score.

| Model | Checkpoint / format | Validated path | Reference result |
| --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](models/deepseek-v4.1-flash.md) | Safetensors FP8 E4M3 dense + FP4 E2M1 experts | `pocketllm serve --backend v41`, host PyTorch, TP4 | Served: 150.3–152.0 tok/s prefill at a 260,244-token prompt, 3.48–3.54 tok/s decode, one request at a time |
| [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | C++/CUDA TP4, GPU-resident FP8 | 864.54 tok/s prefill, 43.22 tok/s decode on a 512-token prompt |
| [DeepSeek-V4-Flash](models/deepseek-v4.md) | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | PyTorch heterogeneous, C++/CUDA, GGUF TP4 | C++ FP4: ~401 tok/s prefill at 32K–64K; ~3.7 tok/s decode |
| [MiniMax-M2.7](models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | Raw-block CUDA, GGUF TP4 | 256-token prefill ~104.9–107 tok/s; 43-layer decode benchmark 10.32 tok/s |
| [GLM-5.2](models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | Raw-block CUDA, GGUF TP4 | ~0.79 tok/s prefill; ~0.66 tok/s decode |

The model pages separate architecture specifications from what PocketLLM actually
implements. `inspect`, `smoke` and a benchmark are not automatically equivalent to
a production serving guarantee.

## Documentation

| Directory | What it holds |
| --- | --- |
| [Guides](guides/index.md) | Benchmark reporting rules, the native engine API, the PyPI release flow, Ascend platform notes |
| [Model guides](models/README.md) | The support matrix and one page per checkpoint |
| [Performance](performance/index.md) | Run records and bottleneck analyses for capabilities that are live today |
| [Architecture](architecture/index.md) | Engine design, refactor plans, roadmaps, and the vLLM/SGLang comparisons |
| [Migration](migration/dsv4-to-pocket-rename.md) | Breaking-change notes — currently the `dsv4` → `pocket` rename |
| [Reports](reports/dsv4_2080ti_report.pdf) | Rendered long-form reports |
| [Archive: Phase 2 and Phase 3](archive/phase2-phase3/index.md) | Completed records, kept for their measurement context and superseded numbers |

Every document in this tree is listed in one of those sections, so nothing is
reachable only by guessing a filename.

## Elsewhere in the repository

- [Repository home](https://github.com/lvyufeng/PocketLLM) · [中文首页](https://github.com/lvyufeng/PocketLLM/blob/master/README_CN.md)
- [C++/CUDA engine notes](https://github.com/lvyufeng/PocketLLM/blob/master/cpp_engine/README.md)
- [Changelog](https://github.com/lvyufeng/PocketLLM/blob/master/CHANGELOG.md)
- [pocketllm on PyPI](https://pypi.org/project/pocketllm/)

## License

PocketLLM is released under the [MIT License](https://github.com/lvyufeng/PocketLLM/blob/master/LICENSE).
Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party
components are governed by their own licenses; PocketLLM's code license grants no
additional rights to third-party model assets.
