# Getting started

This page is the short route from a checkout to a running model. PocketLLM now runs a checkpoint on
**one accelerator** — a single card, or an edge or mobile target — and the runtime it ships is the
`xing4` backend for [Xing4.0-29B-A4B](models/xing4.0-29b-a4b.md). The repository
[`README.md`](https://github.com/lvyufeng/PocketLLM#installation) is the authoritative install text.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0, < 2.7
- CUDA toolkit 11.8+ for GPU execution
- The `relic-core` package — the shared operator library that carries the native kernels — installed
  from its checkout before PocketLLM (see Install)
- 16 GB+ system RAM

## Install

The package is pure Python: nothing here compiles and there is no CMake configure step. The native
kernels live in the separate `relic-core` operator library, which is not on PyPI yet, so install it
from its checkout first.

```bash
# The operator library first; it does compile, against your CUDA toolkit.
pip install -e ../relic-core --no-build-isolation

# Then PocketLLM itself; no build isolation is needed.
pip install -e .
```

For a working copy:

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install -e ../relic-core --no-build-isolation
pip install -e .
```

The kernels that used to be built here as `src/csrc/` now live in
[relic-core](https://github.com/lvyufeng/relic-core), and the C++ engine that used to sit in
`cpp_engine/` is archived in [relic-engine](https://github.com/lvyufeng/relic-engine). Neither is
built from this repository any more.

## Verify the install

```bash
python -m pytest tests/ -q
```

Modules that need a GPU, a real checkpoint or an extension this build has not got skip themselves —
a skip is not a pass.

## Run something

Python API:

```python
from pocketllm import LLM

llm = LLM(model="/path/to/xing4_0-29b-IQ4_NL.gguf", backend="auto")
print(llm.generate("What is artificial intelligence?").text)
```

OpenAI-compatible server:

```bash
pocketllm serve \
  --model /path/to/xing4_0-29b-IQ4_NL.gguf \
  --tokenizer-path /path/to/Xing4.0-29B-A4B \
  --backend auto
```

The weights are a `.gguf` and the tokenizer, chat template and clamp bounds are a directory, so the
checkpoint is two things — see the [model page](models/xing4.0-29b-a4b.md).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pocketllm", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`--backend` defaults to `auto`, which reads the checkpoint's own `general.architecture` out of the
GGUF header and selects the runtime that claims that name. The only runtime in this build is
`xing4`; `--backend xing4` names it by hand. There is no `--tensor-parallel-size`, no `--backend
cpp`, and no `--backend torch`: this repository no longer carries those paths.

## Where the other models went

PocketLLM used to describe the whole multi-GPU stack. The code for it has moved out:

| What | Where it lives now |
| --- | --- |
| Tensor and expert parallelism, host expert banks, multi-card serving (DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, Qwen3.8-27B, MiniMax-M2.7) | [RelicLLM](https://github.com/lvyufeng/RelicLLM) |
| The native C++/CUDA engine | the [relic-engine](https://github.com/lvyufeng/relic-engine) archive |
| The operator library and kernel build documentation | [relic-core](https://github.com/lvyufeng/relic-core) |

Two single-card pages stay here even though their runtime left: [Ternary-Bonsai-2-27B](models/ternary-bonsai-2-27b.md)
and [DeepSeek-V4 on GGUF Q2](models/deepseek-v4-gguf-q2-single-gpu.md). Both ran on the C++ front
end, both are marked **Stale** at the top of the page and in the matrix, and neither can be run from
this tree. They are kept because the measurements are still evidence about the checkpoints.

The [model support matrix](models/README.md) in this repository covers what is single-card only.

Before quoting any number you measure or read here, read
[Benchmarking and reporting rules](https://github.com/lvyufeng/RelicLLM/blob/master/docs/guides/benchmarking.md).