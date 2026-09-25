# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[中文](README_CN.md) | English

PocketLLM is an experimental C++/CUDA and PyTorch inference stack for running large language models on consumer multi-GPU systems. It combines model-specific kernels, low-bit formats, tensor/expert parallelism, CPU/GPU placement, and reproducible single-request benchmarks.

The project started with DeepSeek-V4 on 4×RTX 2080 Ti and now includes validated runtimes for DeepSeek-V4, MiniMax-M2.7, GLM-5.2, Qwen3.8-27B, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Ternary-Bonsai-2-27B. PocketLLM is not a single universal backend: each model has a runtime matched to its architecture and checkpoint format.

Four of those models are served end to end over the OpenAI-compatible API: **Qwen3.8-27B-FP8** through the native C++ runtime, **Ternary-Bonsai-2-27B** through the same runtime on **one** card, **DeepSeek-V4.1-Flash** through `pocketllm serve --backend v41`, and **MiMo-V2.6-Flash** through `pocketllm serve --backend mimo`. All four paths are validated with real checkpoints.

> **Status:** research and engineering software. The numbers below are measurements from specific checkpoints and hardware configurations, not general performance guarantees.

## News

- **[2026/09] Ternary-Bonsai-2-27B is served end to end on one card.** A 27B hybrid-attention model
  — 48 Gated DeltaNet layers and 16 GQA layers over a dense MLP — released as a GGUF whose weights
  are **1.75 bits each** (GGML type 143, 5.53 GiB) with a Hadamard rotation declared in the file.
  `pocketllm serve` picks the native engine from the container's own `general.architecture`, so no
  flag is needed: prefill is **636.0 tok/s at a 4,096-token prompt** and decode **25.9 tok/s**, the
  same card's upstream reference measures 642.5 and 30.7, and 5.53 GiB of weights leaves room for a
  **245,760-token context** (262,144 with an fp8 KV cache). [Model page](docs/models/ternary-bonsai-2-27b.md)
- **[2026/09] MiMo-V2.6-Flash is served end to end.** `pocketllm serve --backend mimo` runs the
  release as four processes on four cards, with the 149.81 GiB of routed experts in a host bank and
  the 48-layer backbone on the GPUs. The nine global layers' attention is divided along the
  checkpoint's own four-way `qkv_proj` partition and joined by an all-gather, so a
  **262,144-token prompt reaches 104.04 tok/s of prefill**, a decode step at that depth is
  **180.0 ms — 5.56 tok/s**, and a short-context step — the softmax, the rotation and the
  norms each one kernel instead of eighteen, ten and two dispatches — is **156.3 ms and 6.40 tok/s**,
  **117.3 and 8.53** with each routed layer's hottest experts kept on the card. The four ranks are
  byte-identical throughout. [Model page](docs/models/mimo-v2.6-flash.md)
- **[2026/09] DeepSeek-V4.1-Flash is served end to end.** `pocketllm serve --backend v41` runs the
  released 475 GiB checkpoint as four processes on four 22 GiB cards, with the 457.8 GiB of routed
  experts pinned in host memory rather than resident on the device. The runtime accepts up to
  262,144 tokens of context; a 260,244-token prompt measures 150.3–152.0 tok/s of prefill and
  3.48–3.54 tok/s of decode. Cross-request prefix caching landed in the same batch, so a prompt
  whose prefix has already been served forwards only its tail.
  [Model page](docs/models/deepseek-v4.1-flash.md) ·
  [Run record](docs/performance/deepseek_v4_1_flash_served_gate.md)
- **[2026/09] Qwen3.8-27B-FP8 gained a native OpenAI-compatible server** — health and model
  discovery, streaming and non-streaming chat and completions, per-token log probabilities,
  stop-sequence truncation, request-field refusals and concurrent scheduler admission, all verified
  against a real checkpoint. [Model page](docs/models/qwen3.8-27b-fp8.md)
- **[2026/08] Two external speculative drafters for Qwen3.8-27B.**
  [DSpark](docs/architecture/qwen3_8_27b_fp8_design.md#external-dspark-speculative-decoding) came first and
  [DFlash2](docs/architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding) after it,
  measuring 2.78× full-request and 3.02× decode on a 512-token fixture with exact token parity in
  every case. Both are opt-in, because their gains are acceptance-dependent and upstream's
  published 2.67–3.43× band is a decode-latency ratio rather than a full-request one.
- **[2026/08] Qwen3.8-27B-FP8 on the C++/CUDA runtime** — FP8 E4M3 Safetensors text generation at
  TP4, 864.54 tok/s of prefill and 43.22 tok/s of decode on a 512-token prompt, with a 256K context
  path and a persistent TP4 worker that keeps prefix state alive across requests.
- **[2026/07] GLM-5.2 text generation** through the shared GGUF raw-block path, on the same
  `src.cli.generate_glm` entry point the other GGUF models use.
- **[2026/06] MiniMax-M2.7 on GGUF `UD-IQ1_M`** — ~104.9–107 tok/s full-model 256-token prefill, and
  a 43-layer decode benchmark at 10.32 tok/s after fused RMSNorm.
- **[2026/05] DeepSeek-V4-Flash**, the checkpoint this project started on — FP4/FP8 Safetensors and
  GGUF Q2/IQ2/IQ1 generation, ~401 tok/s of C++ FP4 prefill at 32K–64K.
  [Model page](docs/models/deepseek-v4.md)

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
- ✅ Research and experimentation with model-specific kernel optimization

**Consider alternatives like vLLM or SGLang if you need:**
- ❌ High-throughput serving with dynamic batching (PocketLLM batching is sequential)
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

| Model | Checkpoint / format | Runtime status | Validated path | Reference result on 4×RTX 2080 Ti |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md) | Safetensors FP8 E4M3 dense + FP4 E2M1 experts | **Validated TP4 text generation behind the OpenAI server** | `pocketllm serve --backend v41`: host PyTorch over a mapped checkpoint, dense tree and packed FP4 experts on the cards, one process a rank | Served TP4: **150.3–152.0 tok/s prefill** at a 260,244-token prompt (137–141 at 1,364) and **3.48–3.54 tok/s decode**, one request at a time |
| [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md) | Safetensors FP8 E4M3 dense + MXFP4 experts, BF16 attention output | **Validated TP4 text generation behind the OpenAI server** | `pocketllm serve --backend mimo`: the 48-layer backbone on the cards, routed experts out of a 149.81 GiB host bank, attention split along the checkpoint's own four-way `qkv_proj` partition | Served TP4: **104.04 tok/s prefill at a 262,144-token prompt** (48.37 with the attention replicated) and **5.56 tok/s decode** at that depth, **6.40 tok/s at a short context** and 8.53 with each routed layer's hottest experts kept on the card, one request at a time |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | **Validated C++ text runtime and OpenAI server** | C++/CUDA TP4, GPU-resident FP8 | Served TP4: 864.54 tok/s prefill and 43.22 tok/s decode on a 512-token prompt, 128 tokens generated |
| [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0` (GGML type 143), **1.75 bits a weight**, 5.53 GiB, Hadamard declared in the file | **Validated C++ text runtime and OpenAI server on one card** | `pocketllm serve` selects the native engine from the file's own `general.architecture`, no flag needed | Served on **1×**RTX 2080 Ti: **636.0 tok/s prefill** at a 4,096-token prompt and **25.9 tok/s decode** — the same card's upstream reference is 642.5 and 30.7 — with 245,760 tokens of context at a 22 GiB card's memory budget |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | **Validated generation** | PyTorch heterogeneous, C++/CUDA, GGUF TP4 | C++ FP4: ~401 tok/s prefill at 32K–64K; ~3.7 tok/s decode |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | **Validated TP4 generation** | Raw-block CUDA, GGUF TP4 | Full-model 256-token prefill: ~104.9–107 tok/s; 43-layer decode benchmark: 10.32 tok/s |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | **Validated text generation** | Raw-block CUDA, GGUF TP4 | ~0.79 tok/s prefill; ~0.66 tok/s decode |

The Qwen3.8-27B line covers three checkpoints over the same text architecture — the validated FP8 runtime above, an [NVFP4](docs/models/qwen3.8-27b-nvfp4.md) variant, and the [official BF16](docs/models/qwen3.8-27b-bf16.md) release, which is audited but not run. See the [support matrix](docs/models/README.md) for the exact status of each.

The model pages separate architecture specifications from what PocketLLM currently implements. `inspect`, `smoke`, and a benchmark are not automatically equivalent to a production serving guarantee.

## Performance highlights

All figures in this section use real checkpoints on the same baseline system unless noted otherwise: 4× NVIDIA RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink, single-request execution, TP4 where applicable. See [Benchmarking](docs/guides/benchmarking.md) before comparing results.

### DeepSeek-V4.1-Flash v41 runtime (served)

The released 475.24 GiB checkpoint runs over four ranks, one process a card, with the 457.8 GiB of routed experts pinned in host memory rather than resident on the cards; the dense tree and the packed FP4 experts both execute on the GPUs. The longest leg the runtime accepts — a 260,244-token prompt over `pocketllm serve --backend v41`, three back-to-back requests, 64 greedy tokens each:

- **150.3, 152.0 and 152.0 tok/s of prefill**, against 137–141 tok/s at a 1,364-token prompt. The long prompt is the *faster* of the two, because a short one is dominated by fixed per-process cost.
- **3.53, 3.48 and 3.54 tok/s of decode**, 253–262 ms a step.

Against the reference launcher's own 262,144-token row (103.54 tok/s at 3.88) that is 1.45–1.47× on prefill, 0.90–0.91× on decode, and 1,730.0 s of wall against 2,555.4 s. Two qualifications travel with it: the two prompts are not the same text — the served leg is repeated filler and the reference row is a chapter template — so part of the prefill margin may be the prompt rather than the runtime, and the decode half is the half that does not travel, since the same service reads 4.45–4.53 tok/s at 1,364 prompt tokens.

At a 1,364-token prompt on the short-context configuration (`--max-model-len 2048`, 288 expert pool rows), prefill is 137.5, 140.8 and 138.3 tok/s and decode 4.45–4.53 tok/s; the first request reads 108.0 because it pays the capture pass. These are cold-prompt numbers, taken before cross-request prefix caching landed — a repeat of a prompt already served now forwards only its tail.

What this runtime does not have is batching, continuous batching, or an MTP layer. The three DSpark draft layers are 7.39 GiB the loader deliberately leaves in the shards, so there is no speculative decoding here, and requests are serialized rather than batched. There is also no numeric oracle: the reference stack needs `torch>=2.10.0` and `tilelang==0.1.8` and neither is available on this host, so the acceptance evidence is generated text rather than a logit comparison.

### MiMo-V2.6-Flash heterogeneous runtime (served)

The release runs as four processes, one a card, with the 149.81 GiB of routed MXFP4 experts in a single `/dev/shm` segment that every rank attaches to; the 48-layer backbone — nine global-attention layers and thirty-nine sliding-window layers over a 128-slot ring — executes on the cards, and each layer's experts are dealt out over the ranks. The deal is the call's row count: a decode step draws `top_k / world` experts a rank, and a prefill chunk needs the experts themselves partitioned.

- **262,144-token prompt: 104.04 tok/s of prefill** through 2048-token chunks, 10.21 GiB on the card, the four ranks' last row byte-identical.
- **Decode at that depth: 180.0 ms a step, 5.56 tok/s**; at a short context the same step is **156.3 ms, 6.40 tok/s** — 177.6 and 5.63 before the decode step's softmax, its rotation and its norms each became one kernel instead of eighteen, ten and two dispatches. One card, for contrast, is 610 ms a token, 1.64 tok/s.
- **The larger lever is not a kernel.** A step's floor is the expert copy — 1198.5 MiB a rank, 47 layers of the two experts the deal leaves it, at a PCIe 3.0 link's own rate — and `--backend-option resident_rows=16` keeps each routed layer's hottest sixteen experts on the card instead, for **117.3 ms and 8.53 tok/s**, exact to `torch.equal` on 24 steps of a greedy chain. It is off by default because 9.36 GiB and a 262144-token cache do not fit on one 22 GiB card, and it is a short-context lever for a second reason: a copy removed is worth 0.58 ms of step at eight positions and 0.19 at 262144, so the eight rows that do fit at that depth buy 3%.
- The prefill rate is the attention split's: the same prompt with the attention replicated on every rank is **48.37 tok/s** and 18.71 GiB on the card. A 64k prompt runs at 104.4 tok/s with a 4096-token chunk.

What is left in the step is a copy and a schedule. **99.9 ms of it is the expert H2D the kernel waited for** — 1198.5 MiB at 12.0 GiB/s, a PCIe 3.0 x16 link at essentially its rate — against 38.8 ms of attention, 12.5 of expert kernel in 47 calls, and roughly 10 of collectives; an arm that *removes* copies rather than attributing a window to them prices one at 0.19 ms of step there against the link's 1.23, so the bytes at that depth are partly overlapped. Prefetching the copy from the previous token's draw was measured and closed: a rank's rows hold the expert they held a step earlier 9–13.5% of the time, and the whole set repeats in 1.9–2.4% of draws.

There is no batching and no speculative decoding here, and the attention and dense linears are torch rather than kernels.

### Qwen3.8-27B-FP8 C++ runtime

One serial sweep of the engine defaults on master `cfad866`, with 128 generated tokens per run:

- 64-token prompt: 115.91 tok/s prefill (0.55 s), 45.05 tok/s decode.
- 512-token prompt: 864.54 tok/s prefill (0.59 s), 43.22 tok/s decode.
- 8,192-token prompt: 1,818.65 tok/s prefill (4.50 s), 43.99 tok/s decode.
- 65,536-token prompt: 1,453.51 tok/s prefill (45.09 s), 39.11 tok/s decode.

Per rank the engine accounts for 6.86 GiB of resident FP8 weights and scales, plus 1.00 GiB of KV data and 1.01 GiB of chunk workspace at 65,536 tokens; `nvidia-smi` peaks a further 3.4–3.5 GiB in CUDA context, cuBLAS workspaces and NCCL buffers, which is constant across prompt lengths. Token sequences were identical across all four TP ranks. The native OpenAI-compatible server is validated for text requests; image and video inputs remain unsupported.

The prefill figures for the 64- and 512-token prompts measure short-prompt latency, not steady-state throughput: both complete in 0.55–0.59 s because a fixed per-process cost dominates at that size. Above 4,096 tokens prefill runs at a marginal 1,670 tok/s up to 32,768 and 1,285 tok/s beyond it.

The same runtime is what serves Qwen3.8 over the OpenAI API, and it is the one model here with two optional external speculative drafters: [DSpark](docs/architecture/qwen3_8_27b_fp8_design.md#external-dspark-speculative-decoding) and [DFlash2](docs/architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding), mutually exclusive with each other and with the native MTP path. DFlash2 with its opt-in flags measures 2.78× full-request and 3.02× decode on a 512-token fixture, and 1.33× aggregate over eight GSM8K prompts, with exact token parity in every case. Both drafters are default-off because their gains are acceptance-dependent, and upstream's published 2.67–3.43× band is a decode-latency ratio rather than a full-request one. A persistent TP4 worker also keeps prefix state alive across requests, so a client whose next prompt extends or compresses the previous one pays only for the difference.

### Ternary-Bonsai-2-27B ternary GGUF runtime (served, one card)

The same engine that serves Qwen3.8 also serves this checkpoint, because it is the same architecture
at the same shape — the difference is the container. One RTX 2080 Ti, the released `PTQ1_0` file,
one request at a time, greedy:

- 4,096-token prompt: **636.0 tok/s prefill** (6.44 s), 25.89 tok/s decode.
- 8,192-token prompt: 639.1 tok/s prefill (12.82 s), 25.44 tok/s decode.
- Upstream `llama-bench` on the same card and artifact: 642.5 tok/s prefill at 4,096 (615.2 at
  8,192), 30.7 tok/s decode — so prefill is **level with the reference** and decode is 84% of it.

**A prompt whose token count is not a whole number of 64-token tiles pays a one-off penalty of up to
12 seconds** in its final partial tile, which is what an earlier measurement of this checkpoint
recorded as a slow kernel: 4,097 tokens prefills in 18.11 s where 4,096 takes 6.44 s. The per-token
rate is otherwise flat at 1.55 ms from 2,048 tokens to 22,378. It is measured and reproducible and
the mechanism is not yet identified. Memory on the card is 5.53 GiB of weights and 6,566 MiB with the
runtime, plus **64 KiB of KV a token**.

### DeepSeek-V4 C++ FP4 runtime

- 32K prompt: approximately 402 tok/s prefill, approximately 11.2 GiB/rank.
- 64K prompt: approximately 401 tok/s prefill, approximately 14.5 GiB/rank.
- Decode: approximately 3.7 tok/s on the measured 4×RTX 2080 Ti configuration.

### MiniMax-M2.7 and GLM-5.2 GGUF runtimes

- MiniMax-M2.7 reaches approximately 104.9–107 tok/s full-model 256-token prefill after Q4/Q5 MMA and IQ2 DP4A paths; a separate 43-layer decode benchmark reached 10.32 tok/s after fused RMSNorm.
- GLM-5.2 generation is functional through the raw-block GGUF path. Its current decode floor is much lower because of the model size, expert staging, and per-layer synchronization; experimental resident-cache, routed-TP, and fused-RMSNorm switches are not enabled by default.

These are architecture-specific results. They should not be averaged into one PocketLLM score.

## Architecture overview

PocketLLM has two complementary execution families:

1. **GPU-resident and low-bit execution** keeps local weights or expert blocks on device when the aggregate memory budget permits it.
2. **Heterogeneous execution** keeps routed experts in CPU/NUMA memory and stages only the active quantized blocks needed by the current token or prefill chunk.

The runtime is intentionally model-specific. DeepSeek-V4 uses MLA/indexing and routed-expert scheduling; DeepSeek-V4.1-Flash uses a causal encoder-decoder with CSA2 shared-KV attention over a checkpoint whose 268.95 GiB of routed experts and 189.13 GiB of Engram tables stay in host memory or on disk; MiMo-V2.6-Flash uses a hybrid of global and sliding-window attention with a per-head sink over 149.81 GiB of MXFP4 experts in a shared host bank, with the attention divided along the checkpoint's own four-way partition; MiniMax-M2.7 and GLM-5.2 use GGUF raw-block paths; Qwen3.8 uses Safetensors FP8 online unpacking plus hybrid linear/full attention; Ternary-Bonsai-2-27B is the same hybrid attention in a 1.75-bit GGUF whose tensors are consumed as ternary, with the incoherence rotation the file declares applied to the activations. Raw quantized weights are not expanded to a full FP32 copy in the intended hot paths.

## Quick start

### Install the Python package

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation .
```

This installs the `pocketllm` package and the `pocketllm` CLI. `--no-build-isolation` keeps the
build using the active environment's Torch, which must match the CUDA toolkit the extensions
compile against.

To also build the optional native C++ engine module (`pocketllm_cpp`), which the
`backend="cpp"` path needs:

```bash
POCKETLLM_BUILD_CPP=1 python -m pip install --no-build-isolation .
```

Building the Torch extensions in place, without installing, still works:

```bash
python setup.py build_ext
```

The Python package metadata is named `pocketllm`; existing Python imports under `src.*` remain unchanged for compatibility.

### Build the C++/CUDA engine

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

The executable is `pocketllm_engine`:

```text
build/cpp_engine/pocketllm_engine
```

It was formerly `dsv4_cpp_engine`. That rename, along with the `pocket::` namespace and the
`POCKETLLM_*` environment variables, is a breaking change — see
[the migration note](docs/migration/dsv4-to-pocket-rename.md).

The backend is selected at configure time via `POCKET_BACKEND`, which defaults
to `cuda`, so the command above is unchanged from before:

```bash
cmake -S cpp_engine -B build/cpp_engine -DPOCKET_BACKEND=cuda
```

`POCKET_BACKEND=ascend` reserves the layout for Ascend NPUs. It configures but
does not yet link, because the ACL runtime, AscendC kernels and HCCL collectives
under `cpp_engine/backends/ascend/` are not implemented.

`POCKET_BUILD_DEV_TARGETS` controls the inspection tools and the per-kernel test
and benchmark binaries — everything `cpp_engine/tools/` and `cpp_engine/tests/`
hold. It defaults to whether those directories are present, so a working copy
configures them and an unpacked sdist does not: an install builds the libraries,
the `pocketllm_engine` executable and the optional Python module, and nothing an
install builds needs a test binary. Pass `-DPOCKET_BUILD_DEV_TARGETS=OFF` to
configure a checkout for a library-only build.

The source tree is layered so that a second backend can reuse everything that is
not vendor-specific:

```text
cpp_engine/
  core/              device-agnostic: loaders, tokenizer, HTTP server
  engine/            one engine implementation, shared by all backends
  backends/
    api/             vendor-neutral contracts (to be populated)
    cuda/            kernels/ runtime/ collective/
    ascend/          kernels/ runtime/ collective/
```

`core/` and the public headers under `include/` must not include a vendor SDK.
This is enforced, not merely documented:

```bash
cmake --build build/cpp_engine --target check_layering
```

### Run DeepSeek-V4 C++ TP4 serving

```bash
CKPT=/path/to/DeepSeek-V4-Flash \
PORT=8000 \
MAX_CONTEXT=8192 \
PYTHON=python \
bash scripts/run_cpp_serve_tp4.sh
```

This starts rank 0 as the OpenAI-compatible server and ranks 1–3 as NCCL workers.

### Run DeepSeek-V4.1-Flash OpenAI serving

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python -m pocketllm serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --backend v41 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000 \
  --backend-option expert_pool_rows=148 \
  --backend-option prefill_chunk=4096 \
  --backend-option decode_graphs=true \
  --backend-option threads=22
```

The CLI's own supervisor starts one process a rank — rank 0 binds the listener, ranks 1–3 are NCCL workers — and all four must report ready before the server answers. Startup is not quick: with `DEEPSEEK_V41_RESIDENT_EXPERTS=1` each rank pins its share of the 457.8 GiB expert bank, about 100 s a rank, and the 48 shards load in another 130 s. The same flags take `--max-model-len 262144`, which is the longest context the runtime accepts and the configuration the long-context numbers above were measured on.

### Run Ternary-Bonsai-2-27B OpenAI serving

```bash
python -m pocketllm serve \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --served-model-name bonsai \
  --max-model-len 245760 \
  --port 8000
```

No backend flag and no tensor-parallel flag: the checkpoint is one `.gguf` file, the adapter reads
`general.architecture=qwen35` from its header and selects the native engine that claims that name,
and the tokenizer, the special-token ids and the chat template come out of the same header. One card
holds it, and `--max-model-len` is a memory decision as much as a context one at **64 KiB a token**
of KV — 245,760 tokens is the largest FP16-KV context that fits in 22 GiB beside the 5.53 GiB of
weights, and `--kv-cache-dtype fp8` halves the KV so the checkpoint's own 262,144 fits.

### Run MiMo-V2.6-Flash OpenAI serving

```bash
python -m pocketllm serve \
  --backend mimo \
  --model /path/to/MiMo-V2.6-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --port 8000 \
  --backend-option prefill_chunk=2048 \
  --backend-option chunk_rows=16 \
  --backend-option resident_rows=16   # 9.36 GiB of experts held on the card; 0 at a long context
```

Same supervisor, same four processes. The first start fills the 149.81 GiB expert bank into `/dev/shm` from the release, which takes about 12 minutes at 213 MiB/s; a later run attaches to the existing segment in 0.07 s, and every rank attaches to the same one. The routed experts come out of that bank, so the cards hold only the dense weights, the attention and the two expert arenas — **10.21 GiB a card at a 262144-token context**. `rm -rf /dev/shm/pocketllm_mimo_experts_*` gives the memory back.

`resident_rows` is the one option that is a deployment's decision rather than a constant: it keeps each routed layer's hottest `N` experts resident, which is **117.3 ms a step against 156.3** at a short context, at 0.585 GiB a row over the 47 routed layers. Sixteen rows and a 262144-token cache do not fit on one 22 GiB card, eight do and are worth 3% there, and the shipped default is 0.

### Run a GGUF model through the shared raw-block CLI

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_gguf \
  --gguf-path /path/to/model.gguf \
  --seed-file /path/to/prompt_tokens.bin \
  --max-new-tokens 32 \
  --prewarm
```

For GLM-5.2 text prompts:

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_glm \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --prompt "Hello" \
  --chat \
  --max-new-tokens 32 \
  --prewarm
```

### Inspect a GGUF checkpoint

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/model.gguf \
  --architecture auto \
  --spec-summary \
  --validate-spec \
  --capability-report \
  --placement-report
```

### Run a Qwen3.8-27B-FP8 C++ smoke/benchmark

The Qwen path accepts a text prompt or token IDs and uses TP4 ranks with an NCCL ID file:

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --generate-token 123 --max-new-tokens 32 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

For a normal run, use the same command-line options as the Qwen smoke entrypoint and let rank 0 report `prefill_tokens_per_s`, `decode_tokens_per_s`, resident weight bytes, and GPU memory. The native Qwen text server can be verified against a real checkpoint with:

```bash
python scripts/verify_cpp_qwen_openai.py \\
  --ckpt /path/to/Qwen3.8-27B-FP8 \\
  --binary build/cpp_engine/pocketllm_engine \\
  --python /path/to/python-with-transformers \\
  --sidecar src/server/cpp_sidecar.py \\
  --devices 0,1,2,3
```

The harness checks health, model discovery, non-streaming and streaming chat and text completions, fixed-sampling validation, request-field refusals, stop-sequence truncation, multiple choices, per-token log probabilities, and concurrent scheduler admission.

External Qwen DSpark is available as an opt-in with `--qwen-dspark /path/to/Qwen3.8-27B-DSpark`; it cannot be combined with native MTP. The real five-layer drafter proposes seven tokens and verifies eight target rows at once. It remains default-off because measured gains are acceptance-dependent. See the [Qwen speculative-decoding design notes](docs/architecture/qwen3_8_27b_fp8_design.md#external-dspark-speculative-decoding) for real 512/8K/32K results and the prefix/cold-parity command.

External Qwen DFlash2 is a second opt-in drafter, `--qwen-dflash2 /path/to/Qwen3.8-27B-DFlash2`, mutually exclusive with both DSpark and native MTP. With its four opt-in flags enabled it measures 2.78x full-request and 3.02x decode on a 512-token fixture, and 1.33x aggregate on eight GSM8K prompts, with exact token parity in every case. Decode-phase speedup falls inside upstream's published 2.67–3.43x band. See the [Qwen speculative-decoding design notes](docs/architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding) for the full table, the FP32-residual numerical requirement, and the reproduction commands.

For a single-concurrency client whose next request extends or compresses the previous one, keep one TP4 process group alive with the persistent token-ID worker. Rank 0 reads `<max_new_tokens> token0 token1 ...` lines and reports exact prefix accounting; the worker reuses live state for appends and device snapshots for branches:

```bash
python scripts/bench_qwen_prefix_cache.py \\
  --ckpt /path/to/Qwen3.8-27B-FP8 \\
  --token-ids-file /path/to/prompt_ids.csv \\
  --max-context 32768 \\
  --max-new-tokens 4 \\
  --compression-prefix-tokens 4096
```

The benchmark starts ranks 1–3 as command workers and keeps rank 0 alive for all requests. Use `--disable-prefix-cache` for a cold parity A/B. One-shot Qwen commands disable prefix snapshots because their engine lifetime covers only one request; `--qwen-persistent-stdin` enables the cache, while `--qwen-no-prefix-cache` explicitly disables it.

## Documentation

The documentation is published at **<https://lvyufeng.github.io/PocketLLM/>**, built from the
`docs/` tree in this repository. It has full-text search, per-topic navigation, and the same
content as the files below.

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Model support matrix](docs/models/README.md)
- [Benchmarking and reporting rules](docs/guides/benchmarking.md)
- [DeepSeek-V4.1-Flash](docs/models/deepseek-v4.1-flash.md)
- [MiMo-V2.6-Flash](docs/models/mimo-v2.6-flash.md)
- [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md)
- [Ternary-Bonsai-2-27B](docs/models/ternary-bonsai-2-27b.md)
- [DeepSeek-V4](docs/models/deepseek-v4.md)
- [MiniMax-M2.7](docs/models/minimax-m2.7.md)
- [GLM-5.2](docs/models/glm-5.2.md)
- [Serving V4.1 behind the OpenAI server](docs/performance/deepseek_v4_1_flash_served_gate.md)
- [What one V4.1 request costs](docs/performance/deepseek_v4_1_flash_single_request_capability.md)
- [Cross-request prefix caching on V4.1](docs/architecture/v41_prefix_cache.md)
- [DSpark speculative decoding](docs/performance/dspark.md)
- [FlashMemory 1M context](docs/performance/flashmemory_1m_context.md)
- [MiniMax decode bottleneck analysis](docs/performance/minimax_decode_bottleneck_analysis.md)
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

- Performance is highly sensitive to GPU model, PCIe topology, NUMA placement, driver/runtime versions, and checkpoint variant.
- GGUF expert staging can dominate decode on PCIe-only systems; a high prefill number does not imply high decode TPS.
- DeepSeek-V4.1-Flash serves one request at a time: `--backend v41` takes a single request lock and reports `supports_batch=False`, so there is no continuous batching, no chunked prefill, and no paged KV pool. A later request that shares a prefix with one already served does reuse it, but that changes how much a request costs, not how many run at once.
- MiMo-V2.6-Flash serves one request at a time for a stronger reason: every routed layer closes with an all-reduce at the same point in every rank's program, so a rank that is not running the request its peers are running is not idle but at a different collective, and NCCL answers a mismatch by hanging. Rank 0 therefore broadcasts the whole request before it starts, and a cancel or a stop string has to be agreed between the ranks rather than acted on by one.
- MiMo-V2.6-Flash's attention and dense linears are torch above 16384 keys, not kernels, and its decode step is bounded below by the expert copy: 99.9 ms of a 180.0 ms step at 256k is the H2D the kernel waited for, at a PCIe 3.0 link's ceiling — though an arm that removes copies prices one at 0.19 ms of step there, so part of that traffic is overlapped. The one schedule that would hide the rest needs a prediction the router does not offer: prefetching from the previous token's draw was measured at a 9–13.5% row hit rate. The one that removes copies outright, `resident_rows`, costs 0.585 GiB a row a routed layer, which a 262144-token cache has room for eight of and buys 3%.
- DeepSeek-V4.1-Flash has no speculative decoding on this path. The checkpoint carries three MTP layers and a DSpark draft head, and the loader deliberately leaves all of it in the shards.
- DeepSeek-V4.1-Flash is validated by generated text, not by a logit comparison. The reference stack needs `torch>=2.10.0` and `tilelang==0.1.8` and neither is available here, so no numeric oracle exists for a V4.1 forward pass.
- DeepSeek-V4 DSpark's current C++ verify path is sequential and should not be presented as a speedup claim. Qwen DSpark is a separate external drafter with one eight-row target verification and model-specific parity/performance data.
- Qwen DFlash2 wall-clock speedup is acceptance-dependent and prefill-capped: the synthetic fixtures accept the full eight-row block while GSM8K accepts 2.9–4.4, and shared prefill limits the 8,192-token case to 1.95x even with zero decode time. Upstream's 2.67–3.43x is a decode-latency ratio, not a full-request wall ratio.
- The Qwen runtime currently supports the text checkpoint path only. Vision inputs and multimodal serving are not implemented.
- Ternary-Bonsai-2-27B has one measured and unexplained prefill cost: a prompt whose token count is not a whole number of 64-token tiles puts up to 12 seconds into its final partial tile, so a 4,097-token prompt takes 18.11 s where a 4,096-token one takes 6.44 s. Aligned, prefill is flat at 1.55 ms a token and level with the upstream reference. Its batch scheduler is off by default because it buys aggregate rate at the cost of per-request latency, its speculative decoding paths are unvalidated on this artifact, and its prefix reuse is a resume of the immediately preceding prompt rather than a store.
- Some experimental optimizations are intentionally opt-in or disabled after real end-to-end regressions. See the model pages and historical notes for details.

## License

PocketLLM is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, including for commercial purposes, provided the copyright notice and permission notice are retained.

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets, and other third-party components are governed by their respective licenses. PocketLLM's code license does not grant additional rights to third-party model assets.

## Acknowledgements

PocketLLM builds on CUDA, PyTorch, safetensors, GGUF, Transformers, NCCL, and llama.cpp quantization research. The model-specific runtimes and benchmarks are engineering work for reproducible local inference on consumer hardware.
