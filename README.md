# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[中文](README_CN.md) | English

PocketLLM runs a large language model on **one accelerator** — a single consumer GPU, or an edge and
mobile target. It keeps the checkpoint on the card it fits on, and where it does not fit it lowers
the weight format until it does.

> **This repository has been re-scoped.** It used to describe the whole multi-GPU stack. That stack
> — tensor and expert parallelism, host offload, the multi-card serving paths — now lives in
> **[RelicLLM](https://github.com/lvyufeng/RelicLLM)**, and the native kernels in
> **[relic-core](https://github.com/lvyufeng/relic-core)**. The retired C++ engine is archived in
> **[relic-engine](https://github.com/lvyufeng/relic-engine)**. The code here has not been cut yet;
> this README and the documentation state where the boundary is going.

> **Status:** research and engineering software. Every number in this repository is a measurement
> from a specific checkpoint and hardware configuration, not a performance guarantee.

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the model
still answers correctly in. Host offload and multi-card tensor parallelism are deliberately out of
scope: a hybrid GPU/CPU expert path measured **2.3× slower** than keeping the experts on the card,
and a checkpoint that needs four cards to answer a prompt is a different product.

Three checkpoints fit one card whole:

| Model | Format | Footprint |
| --- | --- | --- |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0` — 1.75 bits a weight | **5.53 GiB** |
| [Xing4.0-29B-A4B](docs/models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` — 4.5 bits a weight | **17.94 GiB** |
| [DeepSeek-V4 on GGUF Q2](docs/models/deepseek-v4-gguf-q2-single-gpu.md) | GGUF Q2 / IQ2 / IQ1 | one 22 GiB card |

## News

- [2026/09] [Ternary-Bonsai-2-27B served end to end on one card](docs/models/ternary-bonsai-2-27b.md)
- [2026/09] [Xing4.0-29B-A4B served end to end on one card](docs/models/xing4.0-29b-a4b.md)
- [2026/09] [DeepSeek-V4.1-Flash served end to end on four](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.1-flash.md)
- [2026/09] [Qwen3.8-27B-FP8 gained a native OpenAI-compatible server](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/qwen3.8-27b-fp8.md)
- [2026/05] [DeepSeek-V4 on GGUF Q2 in a single GPU](docs/models/deepseek-v4-gguf-q2-single-gpu.md)

The multi-GPU entries are listed for continuity; their records live in
[RelicLLM's documentation](https://lvyufeng.github.io/RelicLLM/).

## Installation

### Quick install

```bash
# Install the build prerequisites first; the build imports them from the
# environment rather than fetching them. See the note below.
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11

pip install pocketllm --no-build-isolation
```

The build compiles CUDA extensions and the native C++ engine, which takes 5–15 minutes.

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0, < 2.7 (install first: `pip install "torch>=2.0,<2.7"`)
- CUDA toolkit 11.8+ (for GPU acceleration)
- CMake >= 3.18, pybind11 >= 2.10, Ninja >= 1.11
- `setuptools >= 68` and `wheel`
- 16 GB+ system RAM (for compilation)

**Note:** `--no-build-isolation` is required so the build uses your environment's PyTorch, which must
match your CUDA toolkit version. It also means pip will not fetch the build prerequisites listed
above: they have to be in the environment before you run the install. A freshly created virtualenv
has none of them — `python -m venv` bootstraps the interpreter's bundled `setuptools`, which on
Python 3.10 is older than the version that provides the `bdist_wheel` command, and on Python 3.12+
installs no setuptools at all — so the install can fail at metadata generation with
`invalid command 'bdist_wheel'`, and then at the native-engine build for a missing `pybind11` or
`cmake`. The `pip install "torch…"` line above installs all of them.

### PyTorch-only install (skip C++ engine)

If you only need the PyTorch backend or lack the C++ build dependencies:

```bash
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

This skips the C++ engine build but still compiles the PyTorch CUDA extensions, so it still needs
`torch` and a `setuptools` new enough to build a wheel.

### Development install

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11
pip install -e . --no-build-isolation
```

## Quick Start

### Python API

```python
from pocketllm import LLM

llm = LLM(
    model="/path/to/checkpoint",
    backend="auto",  # or "torch", "cpp"
    tensor_parallel_size=1,
)

result = llm.generate("What is artificial intelligence?")
print(result.text)

for token in llm.stream("Explain quantum computing"):
    print(token.text, end="", flush=True)
```

### OpenAI-Compatible Server

```bash
# One card, the default
pocketllm serve --model /path/to/checkpoint --backend auto

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pocketllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

`--tensor-parallel-size` still exists in the CLI, and a value above 1 still works today — but it is
the multi-card path, which belongs to [RelicLLM](https://github.com/lvyufeng/RelicLLM) and is not
what this library is for.

## When to use PocketLLM

**PocketLLM is for:**
- ✅ One consumer GPU (RTX 2080 Ti, 3090, 4090) running a checkpoint **whole**, with no host bank
  and no second card
- ✅ Aggressive quantization as the way to make that fit — GGUF Q4/Q2/IQ2/IQ1, FP4, and a 1.75-bit
  ternary GGUF whose tensors are consumed as ternary without ever being upcast
- ✅ Low-latency single-request inference, with prefill and decode dispatched separately so
  improving one does not cost the other
- ✅ Edge and mobile targets, where "it fits" is a hard constraint rather than a tuning parameter

**Consider [RelicLLM](https://github.com/lvyufeng/RelicLLM) instead if you need:**
- ❌ **Multiple GPUs.** Tensor parallelism, expert parallelism, host expert banks and CPU/NUMA
  placement are RelicLLM's, by design — this repository does not go there.
- ❌ **A checkpoint larger than one card.** The answer here is a lower weight format, not a second
  card.

**Consider vLLM or SGLang if you need** broad model coverage, multi-LoRA, multimodal inputs, or
production scheduling features. PocketLLM is a short list of checkpoints with deep, model-specific
optimization, not a universal backend.

## What PocketLLM provides

- **Single-card execution.** The checkpoint is resident on one card. No host bank, no second
  process, no collective to synchronize with.
- **Low-bit execution without expansion.** GGUF Q4/Q2/IQ2/IQ1, FP4, FP8 E4M3 and the 1.75-bit
  ternary format are consumed as quantized blocks in the hot path; raw weights are not expanded to a
  full FP32 copy where it matters.
- **A quantization ladder that reaches 1.75 bits.** Ternary-Bonsai-2-27B is a 27B model in
  **5.53 GiB**, leaving room for a 245,760-token context on a 22 GiB card.
- **A 29B MoE with every expert resident.** Xing4.0-29B-A4B fits its **17.94 GiB** of `IQ4_NL`
  weights — all 64 experts of all 38 MoE layers — on one card, so there is no expert directory to
  consult at decode time.
- **Separate prefill and decode dispatch,** so the large-row kernels and the single-token latency
  path are optimized independently.
- **Inspection and validation tools:** GGUF architecture and spec reports, Safetensors audits,
  tensor-shape checks, numerical parity tests and real-checkpoint benchmarks.

## Supported models

Each name links to its model page, which carries the conditions the numbers were taken under and a
`## Known limitations` section. Four checkpoints are single-card, and are what this repository is
for:

| Model | Format | Runtime | Headline |
| --- | --- | --- | --- |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF ternary, 1.75 bits | native C++/CUDA, **one card**, no flag needed | 636 tok/s prefill, 26 tok/s decode, 5.53 GiB resident |
| [Xing4.0-29B-A4B](docs/models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` | **one card**, all 64 experts resident | 75.22 tok/s prefill, 6.72 tok/s decode, 17.94 GiB resident |
| [DeepSeek-V4 on GGUF Q2](docs/models/deepseek-v4-gguf-q2-single-gpu.md) | GGUF Q2 / IQ2 / IQ1 | PyTorch and C++/CUDA, one 22 GiB card | ~401 tok/s prefill at 32K–64K |
| [DeepSeek-V4-Flash](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.md) | Safetensors FP4/FP8 | the GGUF Q2/IQ2/IQ1 single-card path | see the [RelicLLM page](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.md) |

The multi-card runtimes — DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, Qwen3.8-27B and MiniMax-M2.7 —
have model pages in [RelicLLM](https://lvyufeng.github.io/RelicLLM/). The
[support matrix](docs/models/README.md) in this repository covers what is single-card only.

`inspect`, `smoke` and a benchmark are not automatically equivalent to a production serving
guarantee.

## Architecture

One card, and the checkpoint resident on it. What the design is organized around is **fitting**:
which format is the lowest one this checkpoint still answers correctly in, and what that leaves for
the KV cache.

- **Ternary-Bonsai-2-27B** is a 27B hybrid-attention model — 48 Gated DeltaNet layers and 16 GQA
  layers over a dense MLP — in a GGUF whose weights are 1.75 bits each, with the Hadamard rotation
  the file declares applied to the activations. 5.53 GiB of weights out of a 22 GiB card is most of
  a 262,144-token context.
- **Xing4.0-29B-A4B** is a 29B MoE — MLA attention, 64 routed experts activated top-4 plus one
  shared, and four residual streams per block mixed by a matrix hyper-connection — whose official
  `IQ4_NL` GGUF fits whole. The residual streams are carried wider than the sublayers, because this
  checkpoint's activations leave fp16's range.
- **DeepSeek-V4** in GGUF Q2/IQ2/IQ1 keeps the MLA/indexing and routed-expert scheduling of the
  full-precision path inside one card's memory budget.

Raw quantized weights are not expanded to a full FP32 copy in the hot paths.

## Documentation

Published at **<https://lvyufeng.github.io/PocketLLM/>**, built from the `docs/` tree in this
repository, with full-text search and per-topic navigation.

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Model support matrix](docs/models/README.md)
- [Architecture and the old-hardware roadmap](docs/architecture/pocketllm_roadmap_old_hardware.md)
- [New model support on the 2080 Ti](docs/architecture/pocketllm_new_model_roadmap.md)
- [PyPI release](docs/guides/pypi_release.md)
- [Historical 2080 Ti report](docs/reports/dsv4_2080ti_report.pdf)

Pages whose subject is the multi-GPU runtime, the operator layer or the retired engine are not here
— they sit next to the code they describe, and the links above and throughout the site point at
them.

## Roadmap

- [x] Ternary-Bonsai-2-27B: a 1.75-bit ternary GGUF served on **one** card, 636 tok/s prefill and
  245,760 tokens of context out of 5.53 GiB of weights.
- [x] Xing4.0-29B-A4B: all 64 experts of all 38 MoE layers resident on **one** card.
- [x] DeepSeek-V4 GGUF Q2/IQ2/IQ1 generation on a single GPU.
- [ ] Cut the multi-card code out of this repository and hand it to RelicLLM.
- [ ] Edge and mobile backends — the target that "single card" is a stand-in for.
- [ ] CUDA Graph and persistent decode dispatch where measured beneficial.
- [ ] More single-card benchmark fixtures and automated regression dashboards.

## Known limitations

Three that change what a number means, rather than merely qualify it. Every model page ends with its
own `## Known limitations` listing the rest.

- **A prefill rate does not imply a decode rate, and no figure transfers between configurations.**
  PCIe topology, NUMA placement, driver and toolkit versions, checkpoint variant and warm state all
  move these results. See [benchmarking and reporting rules](https://github.com/lvyufeng/RelicLLM/blob/master/docs/guides/benchmarking.md).
- **Ternary-Bonsai-2-27B pays a one-off cost when a prompt is not a whole number of 64-token
  tiles**: 4,097 tokens prefill in 18.11 s where 4,096 take 6.44 s. That is a 3× error from one
  extra token, measured and reproducible, with the mechanism not yet identified —
  [model page](docs/models/ternary-bonsai-2-27b.md#known-limitations).
- **The multi-card paths still build.** `--tensor-parallel-size 4`, `--backend v41` and
  `--backend mimo` are in this tree until the code is cut. They are RelicLLM's by intent, and the
  single-card paths above are what this repository is maintained for.

## License

PocketLLM is released under the [MIT License](LICENSE). You are free to use, modify, and distribute
the code, including for commercial purposes, provided the copyright notice and permission notice are
retained.

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets, and other third-party components are
governed by their respective licenses. PocketLLM's code license does not grant additional rights to
third-party model assets.

## Acknowledgements

PocketLLM builds on CUDA, PyTorch, safetensors, GGUF, Transformers, NCCL, and llama.cpp quantization
research. The model-specific runtimes and benchmarks are engineering work for reproducible local
inference on consumer hardware.