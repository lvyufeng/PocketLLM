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
standalone engine — the executable the Qwen and DeepSeek-V4 server paths launch:

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

## Pick your path

| If you are running | Start here |
| --- | --- |
| DeepSeek-V4 | [DeepSeek-V4](models/deepseek-v4.md), or [GGUF Q2 on one GPU](models/deepseek-v4-gguf-q2-single-gpu.md) |
| MiniMax-M2.7 | [MiniMax-M2.7](models/minimax-m2.7.md) |
| GLM-5.2 | [GLM-5.2](models/glm-5.2.md) |
| Qwen3.8-27B (FP8 / NVFP4 / BF16) | [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md) |
| A model not listed above | [Model support matrix](models/README.md) first |

Before quoting any number you measure or read here, read
[Benchmarking and reporting rules](guides/benchmarking.md).
