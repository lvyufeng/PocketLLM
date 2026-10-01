# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[中文](README_CN.md) | English

PocketLLM runs a large language model on **one accelerator** — a single consumer GPU, or an edge and
mobile target. It keeps the checkpoint on the card it fits on, and where it does not fit it lowers
the weight format until it does.

> **This repository has been re-scoped, and the cut has landed.** It used to carry the whole
> multi-GPU stack. That stack — tensor and expert parallelism, host offload, the multi-card serving
> paths — now lives in **[RelicLLM](https://github.com/lvyufeng/RelicLLM)**, the native kernels in
> **[relic-core](https://github.com/lvyufeng/relic-core)**, and the retired C++ engine in the
> **[relic-engine](https://github.com/lvyufeng/relic-engine)** archive. The Python that was
> multi-card-only was deleted rather than left half-live, so this build has **one runtime**:
> `--backend xing4`.

> **Status:** research and engineering software. Every number in this repository is a measurement
> from a specific checkpoint and hardware configuration, not a performance guarantee.

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the model
still answers correctly in. Host offload and multi-card tensor parallelism are deliberately out of
scope: a hybrid GPU/CPU expert path measured **2.3× slower** than keeping the experts on the card,
and a checkpoint that needs four cards to answer a prompt is a different product.

## News

- [2026/09] [Xing4.0-29B-A4B served end to end on one card](docs/models/xing4.0-29b-a4b.md)
- [2026/09] [DeepSeek-V4.1-Flash served end to end on four](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/deepseek-v4.1-flash.md)
- [2026/09] [Qwen3.8-27B-FP8 gained a native OpenAI-compatible server](https://github.com/lvyufeng/RelicLLM/blob/master/docs/models/qwen3.8-27b-fp8.md)

The multi-GPU entries are listed for continuity; their records live in
[RelicLLM's documentation](https://lvyufeng.github.io/RelicLLM/). The single-card records this
repository used to list beside them — Ternary-Bonsai-2-27B and the DeepSeek-V4 GGUF Q2/IQ2/IQ1 path
— are kept as measurement pages under `docs/models/`, each marked with what no longer runs here.

## Installation

### Install

```bash
pip install "torch>=2.0,<2.7"

# The shared operator library, which is where every native kernel now lives.
# Not on PyPI yet, so install it from its checkout first:
git clone https://github.com/lvyufeng/relic-core.git
pip install -e ./relic-core --no-build-isolation

pip install pocketllm
```

PocketLLM itself is **pure Python** and installs in seconds: it declares `relic-core` as a
dependency and compiles nothing. All of the CUDA work — the quantized kernel library and the CPU
host ops — is relic-core's, and so is the CUDA toolchain requirement; see
[its README](https://github.com/lvyufeng/relic-core). Building relic-core is the step that wants
`CUDA_HOME` pointed at a toolkit matching your PyTorch.

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.0, < 2.7 (install first: `pip install "torch>=2.0,<2.7"`)
- `relic-core`, installed from its checkout with a CUDA toolkit it can build against

### Development install

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7"
pip install -e ../relic-core --no-build-isolation
pip install -e . --no-build-isolation
```

## Quick Start

### Python API

```python
from pocketllm import LLM

llm = LLM(
    model="/path/to/checkpoint",
    backend="auto",  # or "xing4"
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

`pocketllm serve` is the only subcommand. `--backend` names the runtime, and `xing4` is the one this
build ships.

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
- **A 29B MoE with every expert resident.** Xing4.0-29B-A4B fits its **17.94 GiB** of `IQ4_NL`
  weights — all 64 experts of all 38 MoE layers — on one card, so there is no expert directory to
  consult at decode time.
- **Separate prefill and decode dispatch,** so the large-row kernels and the single-token latency
  path are optimized independently.
- **A server and a library surface.** `pocketllm serve` speaks OpenAI chat and completions, and the
  same engine is reachable as `from pocketllm import LLM`.

## Supported models

Each name links to its model page, which carries the conditions the numbers were taken under and a
`## Known limitations` section.

| Model | Format | Runtime | Headline |
| --- | --- | --- | --- |
| [Xing4.0-29B-A4B](docs/models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` | `--backend xing4`, **one card**, all 64 experts resident | 75.22 tok/s prefill, 6.72 tok/s decode, 17.94 GiB resident |

Two pages under `docs/models/` name runtimes that were deleted with the multi-card cut —
Ternary-Bonsai-2-27B and the DeepSeek-V4 GGUF Q2/IQ2/IQ1 path. They are kept as measurement
records, and each carries a notice at the top saying so, because a number with no live code behind
it is still evidence about the checkpoint:

| Model | Format | Why it is kept |
| --- | --- | --- |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0`, 1.75 bits a weight | the 5.53 GiB / 245,760-token measurement, and what a 1.75-bit format costs |
| [DeepSeek-V4 on GGUF Q2](docs/models/deepseek-v4-gguf-q2-single-gpu.md) | GGUF Q2 / IQ2 / IQ1 | the measurement behind "a single-card host-expert MoE is not a serving configuration" |

The multi-card runtimes — DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, Qwen3.8-27B and MiniMax-M2.7 —
have model pages in [RelicLLM](https://lvyufeng.github.io/RelicLLM/). The
[support matrix](docs/models/README.md) in this repository covers what is single-card only.

## Architecture

One card, and the checkpoint resident on it. What the design is organized around is **fitting**:
which format is the lowest one this checkpoint still answers correctly in, and what that leaves for
the KV cache.

- **Xing4.0-29B-A4B** is a 29B MoE — MLA attention, 64 routed experts activated top-4 plus one
  shared, and four residual streams per block mixed by a matrix hyper-connection — whose official
  `IQ4_NL` GGUF fits whole. The residual streams are carried wider than the sublayers, because this
  checkpoint's activations leave fp16's range.

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

- [x] Xing4.0-29B-A4B: all 64 experts of all 38 MoE layers resident on **one** card.
- [x] Cut the multi-card code out of this repository and hand it to RelicLLM.
- [ ] Edge and mobile backends — the target that "single card" is a stand-in for.
- [ ] CUDA Graph and persistent decode dispatch where measured beneficial.
- [ ] More single-card benchmark fixtures and automated regression dashboards.
- [ ] Port the deleted single-card runtimes back as relic-core consumers — the 1.75-bit ternary
  path is the one with no equivalent anywhere else.

## Known limitations

Three that change what a number means, rather than merely qualify it. Every model page ends with its
own `## Known limitations` listing the rest.

- **A prefill rate does not imply a decode rate, and no figure transfers between configurations.**
  PCIe topology, NUMA placement, driver and toolkit versions, checkpoint variant and warm state all
  move these results. See [benchmarking and reporting rules](https://github.com/lvyufeng/RelicLLM/blob/master/docs/guides/benchmarking.md).
- **This build serves one checkpoint.** The `cpp`, `v41`, `mimo` and `torch` backends, the native C++
  front end and the C++ build were deleted with the cut. What is left of the multi-card interface is
  the `tensor_parallel_size`/`tensor_parallel_rank` fields on `EngineArgs` (and their
  `TENSOR_PARALLEL_*` environment variables), which the one runtime reads to pick its card when a
  launcher names several — there is no `--tensor-parallel-size` flag and no sharding.
  Ternary-Bonsai-2-27B and DeepSeek-V4 GGUF Q2 have **no runtime in this tree**; their pages are
  records, not run instructions.
- **Ternary-Bonsai-2-27B pays a one-off cost when a prompt is not a whole number of 64-token
  tiles**: 4,097 tokens prefill in 18.11 s where 4,096 take 6.44 s. That is a 3× error from one
  extra token, measured and reproducible, with the mechanism not yet identified —
  [model page](docs/models/ternary-bonsai-2-27b.md#known-limitations).

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