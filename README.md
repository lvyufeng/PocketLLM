# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[中文](README_CN.md) | English

PocketLLM is an experimental C++/CUDA and PyTorch inference stack for running large language models on consumer multi-GPU systems. It combines model-specific kernels, low-bit formats, tensor/expert parallelism, CPU/GPU placement, and reproducible single-request benchmarks.

The project started with DeepSeek-V4 on 4×RTX 2080 Ti and now includes validated runtimes for DeepSeek-V4, MiniMax-M2.7, GLM-5.2, Qwen3.8-27B, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Ternary-Bonsai-2-27B. PocketLLM is not a single universal backend: each model has a runtime matched to its architecture and checkpoint format.

Four of those models are served end to end over the OpenAI-compatible API: **Qwen3.8-27B-FP8** through the native C++ runtime, **Ternary-Bonsai-2-27B** through the same runtime on **one** card, **DeepSeek-V4.1-Flash** through `pocketllm serve --backend v41`, and **MiMo-V2.6-Flash** through `pocketllm serve --backend mimo`. All four paths are validated with real checkpoints.

> **Status:** research and engineering software. Every number in this repository is a measurement from a specific checkpoint and hardware configuration, not a performance guarantee.

## News

- [2026/09] [Ternary-Bonsai-2-27B served end to end on one card](docs/models/ternary-bonsai-2-27b.md)
- [2026/09] [MiMo-V2.6-Flash served end to end on four](docs/models/mimo-v2.6-flash.md)
- [2026/09] [DeepSeek-V4.1-Flash served end to end](docs/models/deepseek-v4.1-flash.md)
- [2026/09] [Qwen3.8-27B-FP8 gained a native OpenAI-compatible server](docs/models/qwen3.8-27b-fp8.md)
- [2026/08] [Two external speculative drafters for Qwen3.8-27B](docs/models/qwen3.8-27b-fp8.md#optional-speculative-decoding)

[Older entries and the numbers behind each →](https://lvyufeng.github.io/PocketLLM/#news)

## Installation

### Quick install (full capabilities)

```bash
# Install the build prerequisites first; the build imports them from the
# environment rather than fetching them. See the note below.
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11

pip install pocketllm --no-build-isolation
```

This installs PocketLLM with both PyTorch and C++ engine backends. The build process compiles CUDA extensions and the native C++ engine, which takes 5-15 minutes.

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0, < 2.7 (install first: `pip install "torch>=2.0,<2.7"`)
- CUDA toolkit 11.8+ (for GPU acceleration)
- CMake >= 3.18, pybind11 >= 2.10, Ninja >= 1.11
- `setuptools >= 68` and `wheel`
- NCCL (for tensor parallelism with TP > 1)
- 16GB+ system RAM (for compilation)

**Note:** `--no-build-isolation` is required so the build uses your environment's PyTorch, which must match your CUDA toolkit version. It also means pip will not fetch the build prerequisites listed above: they have to be in the environment before you run the install. A freshly created virtualenv has none of them — `python -m venv` bootstraps the interpreter's bundled `setuptools`, which on Python 3.10 is older than the version that provides the `bdist_wheel` command, and on Python 3.12+ installs no setuptools at all — so the install can fail at metadata generation with `invalid command 'bdist_wheel'`, and then at the native-engine build for a missing `pybind11` or `cmake`. The first command above installs all of them.

### PyTorch-only install (skip C++ engine)

If you only need the PyTorch backend or lack the C++ build dependencies:

```bash
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

This skips the C++ engine build but still compiles PyTorch CUDA extensions, so it still needs `torch` and a `setuptools` new enough to build a wheel.

### Development install

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11
pip install -e . --no-build-isolation
```

The same prerequisites apply: an editable install builds the extensions too.

## Quick Start

### Python API

```python
from pocketllm import LLM

# Initialize with automatic backend selection
llm = LLM(
    model="/path/to/checkpoint",
    backend="auto",  # or "torch", "cpp"
    tensor_parallel_size=1
)

# Generate text
result = llm.generate("What is artificial intelligence?")
print(result.text)

# Stream tokens
for token in llm.stream("Explain quantum computing"):
    print(token.text, end="", flush=True)
```

### OpenAI-Compatible Server

```bash
# Start server on default port 8000
pocketllm serve \
    --model /path/to/checkpoint \
    --backend auto \
    --tensor-parallel-size 4

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pocketllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Tensor Parallel Inference (Multi-GPU)

```bash
# 4-GPU setup (TP4)
pocketllm serve \
    --model /path/to/qwen-27b-fp8 \
    --backend cpp \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

## When to use PocketLLM

**PocketLLM excels at:**
- ✅ Single-request low-latency inference on consumer GPUs (RTX 2080 Ti, 3090, 4090)
- ✅ Running large models on older hardware with aggressive quantization (GGUF IQ1/IQ2, FP4, and a 1.75-bit ternary GGUF whose tensors are never upcast)
- ✅ Checkpoints far larger than the aggregate VRAM: DeepSeek-V4.1-Flash's 475 GiB across 4×22 GiB cards
- ✅ TP4 inference without NVLink (PCIe-only multi-GPU systems)
- ✅ Concurrent serving on the native C++ path: a scheduler with a paged KV block pool, a chunked-prefill admission budget and one batched decode step for the whole active set, measured at **4.68×** the serial wall time at eight concurrent requests ([validation record](docs/performance/cpp_openai_concurrency_validation.md))
- ✅ Research and experimentation with model-specific kernel optimization

**Consider alternatives like vLLM or SGLang if you need:**
- ❌ Batching on every backend. The native engine's own server runs it by default (`--max-batch-size`, 8) and the Python adapter exposes it behind `--backend-option enable_batching=true`, but `--backend v41` and `--backend mimo` serve one request at a time, and the batched decode step refuses MTP, DSpark and DFlash2 outright rather than interleave a single-sequence speculative context into a batch.
- ❌ Broad model support (PocketLLM focuses on a short list of checkpoints with deep optimization, not on covering every architecture)
- ❌ Production features (advanced scheduling, monitoring, multi-LoRA)
- ❌ Multimodal inputs (images/video are not yet supported)

## What PocketLLM provides

- **Model-specific inference paths** for hybrid attention, MLA, GQA, Gated DeltaNet, dense MLPs, and routed MoE layers.
- **Low-bit execution without unnecessary expansion:** FP4, FP8 E4M3, GGUF Q4/Q5/Q8, IQ1/IQ2/IQ3, and Q2 paths consume quantized blocks directly in the hot path where supported.
- **Consumer-GPU parallelism:** TP4/NCCL execution on PCIe-connected GPUs, with CPU/NUMA expert placement for checkpoints that do not fit in device memory.
- **Separate prefill and decode dispatch:** large-row kernels are optimized independently from single-token latency paths.
- **Native C++/CUDA runtime:** the `cpp_engine/` path supports DeepSeek-V4 GGUF/Safetensors flows, Qwen3.8 FP8 Safetensors text generation, a 1.75-bit ternary GGUF read as ternary end to end, and the validated OpenAI-compatible text server.
- **A host-PyTorch adapter for a checkpoint the cards cannot hold:** `--backend v41` runs DeepSeek-V4.1-Flash as four processes, one a card, over a memory-mapped checkpoint — the dense tree and the packed FP4 experts execute on the GPUs while the routed experts read from a pinned host bank.
- **A host-resident expert bank shared by four ranks:** `--backend mimo` keeps MiMo-V2.6-Flash's 149.81 GiB of MXFP4 experts in one `/dev/shm` segment that every rank attaches to, and deals the experts out per layer — by the drawing for a decode step, by the expert for a prefill chunk — so a rank stages two of a token's eight drawn experts rather than all eight.
- **Inspection and validation tools:** GGUF architecture/spec reports, Safetensors audits, tensor-shape checks, numerical parity tests, and real-checkpoint benchmarks.

## Supported models at a glance

Each model has a runtime matched to its architecture and checkpoint format, and each name links to
its model page. The numbers are the headline from that page, not a benchmark of their own; the record
is the page, together with its conditions and its `## Known limitations`.

| Model | Format | Runtime | Status | Headline |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md) | Safetensors FP8 + FP4 | `--backend v41`, host PyTorch, TP4 | Text + OpenAI server | 150–152 tok/s prefill at 260k |
| [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md) | Safetensors FP8 + MXFP4 | `--backend mimo`, host expert bank, TP4 | Text + OpenAI server | 104 tok/s prefill at 262k |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | C++/CUDA, TP4 | Text + OpenAI server | 865 tok/s prefill, 43 tok/s decode |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF ternary, 1.75 bits | C++/CUDA, **one card**, no flag | Text + OpenAI server | 636 tok/s prefill, 26 tok/s decode |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8, GGUF Q2 | PyTorch and C++/CUDA, TP4 | Text + server (C++/PyTorch) | ~401 tok/s prefill, C++ FP4 |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | Raw-block CUDA, TP4 | Text, CLI only | ~105 tok/s 256-token prefill |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | Raw-block CUDA, TP4 | Text, CLI only | ~0.79 tok/s prefill |

[Qwen3.8-27B](docs/models/qwen3.8-27b-fp8.md) also covers an [NVFP4](docs/models/qwen3.8-27b-nvfp4.md)
and an [official BF16](docs/models/qwen3.8-27b-bf16.md) checkpoint over the same text architecture;
the full eight-column version of this table, with the format and validation detail, is the
[support matrix](docs/models/README.md).

The model pages separate architecture specifications from what PocketLLM currently implements. `inspect`, `smoke`, and a benchmark are not automatically equivalent to a production serving guarantee.

## Performance

Every number this project publishes is a measurement from one checkpoint, on one hardware
configuration, taken one way — never a general performance guarantee. The results live next to the
runtime that produced them, so each model page carries its own Performance section alongside the
conditions it was taken under and the comparisons that are not valid. Longer records stand on their
own under [performance records](docs/performance/index.md), among them the
[DeepSeek-V4.1-Flash served run](docs/performance/deepseek_v4_1_flash_served_gate.md) and the
[Qwen concurrency validation](docs/performance/cpp_openai_concurrency_validation.md).

Read [Benchmarking and reporting rules](docs/guides/benchmarking.md) before comparing any two
results from anywhere in this repository.

## Architecture overview

PocketLLM has two complementary execution families:

1. **GPU-resident and low-bit execution** keeps local weights or expert blocks on device when the aggregate memory budget permits it.
2. **Heterogeneous execution** keeps routed experts in CPU/NUMA memory and stages only the active quantized blocks needed by the current token or prefill chunk.

The runtime is intentionally model-specific. DeepSeek-V4 uses MLA/indexing and routed-expert scheduling; DeepSeek-V4.1-Flash uses a causal encoder-decoder with CSA2 shared-KV attention over a checkpoint whose 268.95 GiB of routed experts and 189.13 GiB of Engram tables stay in host memory or on disk; MiMo-V2.6-Flash uses a hybrid of global and sliding-window attention with a per-head sink over 149.81 GiB of MXFP4 experts in a shared host bank, with the attention divided along the checkpoint's own four-way partition; MiniMax-M2.7 and GLM-5.2 use GGUF raw-block paths; Qwen3.8 uses Safetensors FP8 online unpacking plus hybrid linear/full attention; Ternary-Bonsai-2-27B is the same hybrid attention in a 1.75-bit GGUF whose tensors are consumed as ternary, with the incoherence rotation the file declares applied to the activations. Raw quantized weights are not expanded to a full FP32 copy in the intended hot paths.

## Documentation

The documentation is published at **<https://lvyufeng.github.io/PocketLLM/>**, built from the
`docs/` tree in this repository. It has full-text search, per-topic navigation, and the same
content as the files below. Each model page is linked from the model table above; these are the
entries that are not a model page.

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Model support matrix](docs/models/README.md)
- [Benchmarking and reporting rules](docs/guides/benchmarking.md)
- [Native engine API and backends](docs/guides/pocketllm_api.md)
- [Architecture overview](docs/architecture/index.md)
- [Performance records](docs/performance/index.md)
- [Serving V4.1 behind the OpenAI server](docs/performance/deepseek_v4_1_flash_served_gate.md)
- [OpenAI concurrency validation](docs/performance/cpp_openai_concurrency_validation.md)
- [DSpark speculative decoding](docs/performance/dspark.md)
- [FlashMemory 1M context](docs/performance/flashmemory_1m_context.md)
- [Historical 2080 Ti report](docs/reports/dsv4_2080ti_report.pdf)

## Roadmap

- [x] DeepSeek-V4 FP4/FP8 and GGUF Q2/IQ2/IQ1 generation paths.
- [x] MiniMax-M2.7 and GLM-5.2 GGUF raw-block generation paths.
- [x] Qwen3.8-27B-FP8 C++ TP4 text runtime.
- [x] Generalize the C++ model dispatch and binary naming without breaking existing scripts.
- [x] Qwen OpenAI-compatible text serving adapter.
- [x] DeepSeek-V4.1-Flash TP4 text generation behind the OpenAI server, with cross-request prefix caching.
- [x] MiMo-V2.6-Flash TP4 text generation behind the OpenAI server: a host-resident expert bank, the attention split along the checkpoint's own partition, 256k context.
- [x] Ternary-Bonsai-2-27B: a 1.75-bit ternary GGUF served on **one** card, 636 tok/s of prefill and 245,760 tokens of context out of 5.53 GiB of weights.
- [ ] CUDA Graph and persistent decode dispatch where measured beneficial.
- [ ] More model-specific benchmark fixtures and automated regression dashboards.

## Known limitations

Three that change what a number means, rather than merely qualify it. Every model page ends with
its own `## Known limitations` listing the rest.

- **A prefill rate does not imply a decode rate, and no figure transfers between configurations.**
  PCIe topology, NUMA placement, driver and toolkit versions, checkpoint variant and warm state all
  move these results. GGUF expert staging in particular can dominate decode while prefill looks
  healthy. See [Benchmarking and reporting rules](docs/guides/benchmarking.md).
- **Ternary-Bonsai-2-27B pays a one-off cost when a prompt is not a whole number of 64-token
  tiles**: 4,097 tokens prefill in 18.11 s where 4,096 take 6.44 s. That is a 3x error from one
  extra token, and it is measured and reproducible with the mechanism not yet identified —
  [model page](docs/models/ternary-bonsai-2-27b.md#known-limitations).
- **Not every backend batches.** The native C++ runtime has a request scheduler with a paged KV
  pool and a batched decode step; `--backend v41` and `--backend mimo` take a single request lock
  and serve one at a time. See [the concurrency record](docs/performance/cpp_openai_concurrency_validation.md).

Some experimental optimizations are opt-in or disabled after a real end-to-end regression; the
model pages name each one.

## License

PocketLLM is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, including for commercial purposes, provided the copyright notice and permission notice are retained.

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets, and other third-party components are governed by their respective licenses. PocketLLM's code license does not grant additional rights to third-party model assets.

## Acknowledgements

PocketLLM builds on CUDA, PyTorch, safetensors, GGUF, Transformers, NCCL, and llama.cpp quantization research. The model-specific runtimes and benchmarks are engineering work for reproducible local inference on consumer hardware.
