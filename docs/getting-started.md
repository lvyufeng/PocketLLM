# Getting started

This page is the short route from a checkout to the point where the tooling can tell you what your
device can do. PocketLLM is **pre-release**, and it has two halves at different levels of done: the
C engine under `src/` reads a GGUF, tokenizes and runs Qwen3-0.6B on f16 or on packed `q4_k`/`q6_k`
weights, and `pocketllm run` drives it through the `ctypes` bridge — while the *Python* package still
has no backend that implements a kernel, so `pocketllm serve` has nothing to serve from. The
[repository README](https://github.com/lvyufeng/PocketLLM#current-status) carries the authoritative
status table, and [The C engine](architecture/c_engine.md) is the record of what `src/` does.

## Requirements

- Python >= 3.10
- **Nothing else** for the base install. No compiler, no CUDA toolkit, no `torch`, no `relic-core`.
  The kernel ABI is stdlib-only and the reference backend is numpy.

A device runtime is an optional extra and is only needed for the backend that uses it:

| Extra | Brings | For |
|---|---|---|
| *(base)* | `numpy` | the reference backend, the loader, the quantization decoders |
| `cuda` | `torch>=2.0,<2.7` | `--backend cuda` |
| `mps` | `torch>=2.2` | `--backend mps` on Apple Silicon |
| `cuda-kernels` | `relic-core` | an optional acceleration source a `cuda` backend may wrap |

## Install

```bash
pip install pocketllm
```

Or from a checkout, with the test suite:

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install -e ".[dev]"
```

The install compiles nothing. `pyproject.toml` declares no extension modules and the Python package
carries no compiled artifact. The engine is a **separate native library** built from the `src/` tree
in this repository and loaded at runtime, so `pip install` never needs a compiler. Every native
kernel *inside* a Python backend belongs to
[relic-core](https://github.com/lvyufeng/relic-core), which is an optional extra here rather than a
dependency.

## What your device can do

Four commands need no checkpoint, no accelerator and no backend runtime. They are the whole of what
this build can do today, and they are the ones to run first.

```bash
pocketllm devices
```

One row per backend, whether or not this host can load it. An unavailable backend names the missing
runtime, because "no backend for this device" is unhelpful and "qnn: needs libQnnHtp*.so" is
actionable:

```
reference  cpu       available                     eager only        17 ops
cpu        cpu       available                     eager only        16 ops
mps        mps       missing torch>=2.2 with an MPS device  eager only        16 ops
cuda       cuda      available                     stream_capture    17 ops
qnn        qnn       missing the QNN SDK (libQnnHtp*.so) and a Hexagon DSP device node  aot_compile       14 ops
horizon    horizon   missing the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device  aot_compile       14 ops
ascend     ascend    missing CANN (libascendcl.so) and an Ascend NPU (a /dev/davinci node)  stream_capture    14 ops
```

`available` means *the runtime is present*, not *the kernels are written*. That distinction is why
this command is a filesystem probe rather than an import: it has to work on the machine where
everything is broken.

```bash
pocketllm backends        # what each backend declares, available or not
pocketllm architectures   # the model structures this tree can build
pocketllm ops --op gemm_quant --device cuda
```

`ops` is the debugging surface: it resolves an op against the backends for a device and prints why
each candidate was accepted or rejected. Resolution reads no bytes and opens no device, so it
answers for a phone's backends from a development host.

```console
$ pocketllm ops --op gemm_quant --device cuda
gemm_quant on cuda -> cuda (100)
  rejected cpu: device kind 'cpu' != 'cuda'
  rejected mps: device kind 'mps' != 'cuda'
  rejected reference: a better candidate was found
```

`--allow-reference-fallback / --no-allow-reference-fallback` controls whether the numpy oracle stays
a candidate. It does for `run`, so a wrong device still gets an answer; it does not for `serve`,
because silently running a 29B model on numpy is a ten-minute first token, not graceful degradation.

## The commands that load a model

```bash
pocketllm run   --model /path/to/model.gguf --prompt "hello"
pocketllm serve --model /path/to/model.gguf --port 8000
```

Both parse and validate their arguments — that part is real, and the errors are the errors a working
build would give — and then exit naming what is absent: no architecture builder for the checkpoint,
no tokenizer, no backend. `pocketllm serve` names `pocketllm.server.openai.serve`, which is the
importable HTTP surface the CLI will call once a backend exists.

## Verify the source checkout

```bash
python -m pytest tests/ -q
```

Run it **from the repository root**: the import path comes from the `pythonpath = ["python"]` setting
in `pyproject.toml` and from `tests/conftest.py`, both of which pytest reads from the root and
nowhere else. Modules that need a device or a real checkpoint skip themselves — and a skip is not a
pass. `tests/baseline_failures.txt` records the known failures as a set; compare a run against it
with:

```bash
python scripts/check_test_baseline.py
```

## Using it as a library

```python
import pocketllm

print(pocketllm.__version__)
```

`import pocketllm` imports neither numpy nor torch: it declares the kernel vocabulary and nothing
else, and `LLM` / `AsyncLLM` resolve lazily through PEP 562. `tests/test_package_boundaries.py`
asserts this on a fresh interpreter, because it is the property that lets one wheel install on both
a workstation and a phone.

## Where to go next

- [The kernel ABI](architecture/kernel_abi_v1.md) — what a backend must implement, and what it may declare
- [Backends and dispatch](architecture/backend_model.md) — how an op finds a backend
- [Device targets](architecture/devices.md) — what each backend is waiting for
- [Model support](models/README.md) — what an architecture is here, and which ones exist
- [PyPI release](guides/pypi_release.md) — the release flow