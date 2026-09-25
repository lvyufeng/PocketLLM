# Getting started

This page is the short route from a checkout to a running model. The repository
[`README.md`](https://github.com/lvyufeng/PocketLLM#installation) is the
authoritative install text — it carries the caveats about `--no-build-isolation`
and about which prerequisites a fresh virtualenv does not have. What follows is
the same route with the detours removed.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0, < 2.7 — installed *before* PocketLLM, and matching your CUDA toolkit
- CUDA toolkit 11.8+ for GPU execution
- CMake >= 3.18, pybind11 >= 2.10, Ninja >= 1.11, `setuptools >= 68`, `wheel`
- NCCL for tensor parallelism with `TP > 1`
- 16 GB+ system RAM to compile

## Install

```bash
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11
pip install pocketllm --no-build-isolation
```

The build compiles both CUDA extensions and the native C++ engine, which takes
5–15 minutes. `--no-build-isolation` is what makes the build use the Torch you
just installed, and it also means pip fetches none of the prerequisites above —
they must already be present.

If you only need the PyTorch plane, or lack the C++ toolchain:

```bash
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

For a working copy:

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel ninja cmake pybind11
pip install -e . --no-build-isolation
```

## Build the C++/CUDA engine

The Python install already builds the `pocketllm_cpp` module. To build the
standalone engine — the executable the C++ Qwen and DeepSeek-V4 server paths
launch. The `v41` backend is a Python runtime and does not use it:

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

The result is `build/cpp_engine/pocketllm_engine`. The backend is chosen at
configure time and defaults to CUDA:

```bash
cmake -S cpp_engine -B build/cpp_engine -DPOCKET_BACKEND=cuda
```

`POCKET_BACKEND=ascend` configures but does not yet link; the ACL runtime,
AscendC kernels and HCCL collectives under `cpp_engine/backends/ascend/` are not
implemented. See [Ascend SoC generations](guides/ascend_soc_generations.md) before
assuming two Ascend cards can share a kernel.

The layering that keeps a second backend possible is enforced, not just
documented:

```bash
cmake --build build/cpp_engine --target check_layering
```

## Verify the install

```bash
python -m pytest tests/ -q --continue-on-collection-errors
```

`tests/test_gguf_q2_precision.py` fails at collection because it still imports the
pre-move `src.gguf.reader` path, which is why `--continue-on-collection-errors` is
there. Modules that need a GPU, a real checkpoint or a built `pocketllm_cpp` skip
themselves — a skip is not a pass.

## Run something

Python API:

```python
from pocketllm import LLM

llm = LLM(model="/path/to/checkpoint", backend="auto", tensor_parallel_size=4)
print(llm.generate("What is artificial intelligence?").text)
```

OpenAI-compatible server:

```bash
pocketllm serve --model /path/to/checkpoint --backend cpp --tensor-parallel-size 4
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pocketllm", "messages": [{"role": "user", "content": "Hello!"}]}'
```

DeepSeek-V4.1-Flash runs on the host-PyTorch `v41` backend instead, over a
checkpoint far larger than the aggregate VRAM. Startup pins a 457.8 GiB expert
bank in host memory and takes roughly four minutes before the server answers:

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

Raise `--max-model-len` to `262144` for the longest context the runtime accepts.
The adapter takes one request lock, so requests are served one at a time.

MiMo-V2.6-Flash is the other four-rank heterogeneous path. Its routed experts live in a host bank
rather than in device memory, and the first start fills that bank — 149.81 GiB, about 12 minutes;
later starts attach to the existing segment:

```bash
python -m pocketllm serve \
  --backend mimo \
  --model /path/to/MiMo-V2.6-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --port 8000 \
  --backend-option prefill_chunk=2048 \
  --backend-option chunk_rows=16
```

It is one request at a time too, and for a firmer reason: every routed layer closes with an
all-reduce that every rank has to reach, so the ranks run the request as a symmetric group and rank
0 broadcasts the whole request before it starts generating. See the
[MiMo-V2.6-Flash model page](models/mimo-v2.6-flash.md) for the numbers and the memory that bank
takes.

**Ternary-Bonsai-2-27B** is the one that fits on a single card, and the only thing you have to pass
is the file. It is a GGUF whose weights are 1.75 bits each, so a 27B model is 5.53 GiB and the same
card still has room for a 245,760-token KV cache:

```bash
python -m pocketllm serve \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --served-model-name bonsai \
  --max-model-len 245760 \
  --port 8000
```

No `--backend` and no `--tensor-parallel-size`: the adapter reads
`general.architecture=qwen35` out of the container's own header and selects the native engine that
claims that name, and the tokenizer, the special-token ids and the chat template come from the same
header. `--max-model-len` is a memory decision here as much as a context one, at **64 KiB a token**;
`--kv-cache-dtype fp8` halves the KV cache and is what makes the checkpoint's own 262,144 fit.
[Model page](models/ternary-bonsai-2-27b.md) for the measured numbers and the one limitation worth
reading before you benchmark it.

## Pick your path

| If you are running | Start here |
| --- | --- |
| DeepSeek-V4 | [DeepSeek-V4](models/deepseek-v4.md), or [GGUF Q2 on one GPU](models/deepseek-v4-gguf-q2-single-gpu.md) |
| DeepSeek-V4.1-Flash | [DeepSeek-V4.1-Flash](models/deepseek-v4.1-flash.md), then [serving it behind the OpenAI server](performance/deepseek_v4_1_flash_served_gate.md) |
| MiMo-V2.6-Flash | [MiMo-V2.6-Flash](models/mimo-v2.6-flash.md) |
| Ternary-Bonsai-2-27B (**one card**) | [Ternary-Bonsai-2-27B](models/ternary-bonsai-2-27b.md) |
| MiniMax-M2.7 | [MiniMax-M2.7](models/minimax-m2.7.md) |
| GLM-5.2 | [GLM-5.2](models/glm-5.2.md) |
| Qwen3.8-27B (FP8 / NVFP4 / BF16) | [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md) |
| A model not listed above | [Model support matrix](models/README.md) first |

Before quoting any number you measure or read here, read
[Benchmarking and reporting rules](guides/benchmarking.md).
