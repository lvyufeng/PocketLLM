# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

[中文](README_CN.md) | English

PocketLLM is an experimental C++/CUDA and PyTorch inference stack for running large language models on consumer multi-GPU systems. It combines model-specific kernels, low-bit formats, tensor/expert parallelism, CPU/GPU placement, and reproducible single-request benchmarks.

The project started with DeepSeek-V4 on 4×RTX 2080 Ti and now includes validated runtimes for DeepSeek-V4, MiniMax-M2.7, GLM-5.2, and Qwen3.8-27B-FP8. PocketLLM is not a single universal backend: each model has a runtime matched to its architecture and checkpoint format.

> **Status:** research and engineering software. The numbers below are measurements from specific checkpoints and hardware configurations, not general performance guarantees.

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
- ✅ Running large models on older hardware with aggressive quantization (GGUF IQ1/IQ2, FP4)
- ✅ TP4 inference without NVLink (PCIe-only multi-GPU systems)
- ✅ Research and experimentation with model-specific kernel optimization

**Consider alternatives like vLLM or SGLang if you need:**
- ❌ High-throughput serving with dynamic batching (PocketLLM batching is sequential)
- ❌ Broad model support (PocketLLM focuses on 4 models with deep optimization)
- ❌ Production features (advanced scheduling, monitoring, multi-LoRA)
- ❌ Multimodal inputs (images/video are not yet supported)

## What PocketLLM provides

- **Model-specific inference paths** for hybrid attention, MLA, GQA, Gated DeltaNet, dense MLPs, and routed MoE layers.
- **Low-bit execution without unnecessary expansion:** FP4, FP8 E4M3, GGUF Q4/Q5/Q8, IQ1/IQ2/IQ3, and Q2 paths consume quantized blocks directly in the hot path where supported.
- **Consumer-GPU parallelism:** TP4/NCCL execution on PCIe-connected GPUs, with CPU/NUMA expert placement for checkpoints that do not fit in device memory.
- **Separate prefill and decode dispatch:** large-row kernels are optimized independently from single-token latency paths.
- **Native C++/CUDA runtime:** the `cpp_engine/` path supports DeepSeek-V4 GGUF/Safetensors flows, Qwen3.8 FP8 Safetensors text generation, and the validated Qwen OpenAI-compatible text server.
- **Inspection and validation tools:** GGUF architecture/spec reports, Safetensors audits, tensor-shape checks, numerical parity tests, and real-checkpoint benchmarks.

## Supported models at a glance

| Model | Checkpoint / format | Runtime status | Validated path | Reference result on 4×RTX 2080 Ti |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | **Validated generation** | PyTorch heterogeneous, C++/CUDA, GGUF TP4 | C++ FP4: ~401 tok/s prefill at 32K–64K; ~3.7 tok/s decode |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | **Validated TP4 generation** | Raw-block CUDA, GGUF TP4 | Full-model 256-token prefill: ~104.9–107 tok/s; 43-layer decode benchmark: 10.32 tok/s |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | **Validated text generation** | Raw-block CUDA, GGUF TP4 | ~0.79 tok/s prefill; ~0.66 tok/s decode |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | **Validated C++ text runtime and server** | C++/CUDA TP4, GPU-resident FP8 | 864.54 tok/s prefill and 43.22 tok/s decode on a 512-token prompt, 128 tokens generated |

The model pages separate architecture specifications from what PocketLLM currently implements. `inspect`, `smoke`, and a benchmark are not automatically equivalent to a production serving guarantee.

## Performance highlights

All figures in this section use real checkpoints on the same baseline system unless noted otherwise: 4× NVIDIA RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink, single-request execution, TP4 where applicable. See [Benchmarking](docs/guides/benchmarking.md) before comparing results.

### Qwen3.8-27B-FP8 C++ runtime

One serial sweep of the engine defaults on master `cfad866`, with 128 generated tokens per run:

- 64-token prompt: 115.91 tok/s prefill (0.55 s), 45.05 tok/s decode.
- 512-token prompt: 864.54 tok/s prefill (0.59 s), 43.22 tok/s decode.
- 8,192-token prompt: 1,818.65 tok/s prefill (4.50 s), 43.99 tok/s decode.
- 65,536-token prompt: 1,453.51 tok/s prefill (45.09 s), 39.11 tok/s decode.

Per rank the engine accounts for 6.86 GiB of resident FP8 weights and scales, plus 1.00 GiB of KV data and 1.01 GiB of chunk workspace at 65,536 tokens; `nvidia-smi` peaks a further 3.4–3.5 GiB in CUDA context, cuBLAS workspaces and NCCL buffers, which is constant across prompt lengths. Token sequences were identical across all four TP ranks. The native OpenAI-compatible server is validated for text requests; image and video inputs remain unsupported.

The prefill figures for the 64- and 512-token prompts measure short-prompt latency, not steady-state throughput: both complete in 0.55–0.59 s because a fixed per-process cost dominates at that size. Above 4,096 tokens prefill runs at a marginal 1,670 tok/s up to 32,768 and 1,285 tok/s beyond it.

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

The runtime is intentionally model-specific. DeepSeek-V4 uses MLA/indexing and routed-expert scheduling; MiniMax-M2.7 and GLM-5.2 use GGUF raw-block paths; Qwen3.8 uses Safetensors FP8 online unpacking plus hybrid linear/full attention. Raw quantized weights are not expanded to a full FP32 copy in the intended hot paths.

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

The harness checks health, model discovery, non-streaming and streaming chat completions, fixed-sampling validation, and concurrent scheduler admission.

External Qwen DSpark is available as an opt-in with `--qwen-dspark /path/to/Qwen3.8-27B-DSpark`; it cannot be combined with native MTP. The real five-layer drafter proposes seven tokens and verifies eight target rows at once. It remains default-off because measured gains are acceptance-dependent. See the [Qwen model page](docs/models/qwen3.8-27b-fp8.md#external-dspark-speculative-decoding) for real 512/8K/32K results and the prefix/cold-parity command.

External Qwen DFlash2 is a second opt-in drafter, `--qwen-dflash2 /path/to/Qwen3.8-27B-DFlash2`, mutually exclusive with both DSpark and native MTP. With its four opt-in flags enabled it measures 2.78x full-request and 3.02x decode on a 512-token fixture, and 1.33x aggregate on eight GSM8K prompts, with exact token parity in every case. Decode-phase speedup falls inside upstream's published 2.67–3.43x band. See the [Qwen model page](docs/models/qwen3.8-27b-fp8.md#external-dflash2-speculative-decoding) for the full table, the FP32-residual numerical requirement, and the reproduction commands.

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
- [DeepSeek-V4](docs/models/deepseek-v4.md)
- [MiniMax-M2.7](docs/models/minimax-m2.7.md)
- [GLM-5.2](docs/models/glm-5.2.md)
- [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md)
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
- [ ] CUDA Graph and persistent decode dispatch where measured beneficial.
- [ ] More model-specific benchmark fixtures and automated regression dashboards.

## Known limitations

- Performance is highly sensitive to GPU model, PCIe topology, NUMA placement, driver/runtime versions, and checkpoint variant.
- GGUF expert staging can dominate decode on PCIe-only systems; a high prefill number does not imply high decode TPS.
- DeepSeek-V4 DSpark's current C++ verify path is sequential and should not be presented as a speedup claim. Qwen DSpark is a separate external drafter with one eight-row target verification and model-specific parity/performance data.
- Qwen DFlash2 wall-clock speedup is acceptance-dependent and prefill-capped: the synthetic fixtures accept the full eight-row block while GSM8K accepts 2.9–4.4, and shared prefill limits the 8,192-token case to 1.95x even with zero decode time. Upstream's 2.67–3.43x is a decode-latency ratio, not a full-request wall ratio.
- The Qwen runtime currently supports the text checkpoint path only. Vision inputs and multimodal serving are not implemented.
- Some experimental optimizations are intentionally opt-in or disabled after real end-to-end regressions. See the model pages and historical notes for details.

## License

PocketLLM is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, including for commercial purposes, provided the copyright notice and permission notice are retained.

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets, and other third-party components are governed by their respective licenses. PocketLLM's code license does not grant additional rights to third-party model assets.

## Acknowledgements

PocketLLM builds on CUDA, PyTorch, safetensors, GGUF, Transformers, NCCL, and llama.cpp quantization research. The model-specific runtimes and benchmarks are engineering work for reproducible local inference on consumer hardware.
