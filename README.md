# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

**English** | [中文](README_CN.md)

Run a large language model on **one accelerator** — a single GPU, an edge board, the phone in your
pocket. One process owns one device. If the checkpoint does not fit, it is quantized further.

> **Status: pre-release skeleton.** This tree defines the interfaces and ports the parts that do not
> need a device. **No device backend ships a working implementation yet**, so there is nothing to
> run a model on today: `pocketllm run` and `pocketllm serve` parse their arguments and then say
> what they are missing. What is real is everything around them — the kernel ABI, the dispatch and
> backend registry, the GGUF loader, the quantization decoders, the execution layer, the model IR,
> and the OpenAI-compatible HTTP surface. Read [Current status](#current-status) before installing.

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the
model still answers correctly in. Host offload and multi-card parallelism are out of scope by
construction: the ABI has no rank, no collective, and no second device, and `EngineArgs` has no
`tensor_parallel_size` to set.

## Current status

What works, what is a stub, and what has not been written. This table is the honest one; the rest of
this page is the design those pieces are being built toward.

| Piece | State |
|---|---|
| `pocketllm.kernels` — the kernel ABI: descriptors, op schemas, dispatch, graph IR | **Done.** 19 ops declared; stdlib-only, no numpy |
| `pocketllm.backends.reference` — numpy oracle, every op, host memory | **Done.** The normative implementation |
| `pocketllm.backends.cpu` — host CPU | **Stub.** Selection and declaration only |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **Stubs.** Each names the runtime it waits for |
| `pocketllm.quant` — GGML block decoders, vendored tables, no relic-core | **Done.** IQ4_NL, IQ4_XS, IQ1_M, IQ2/IQ3, q2_k–q6_k, q8_0 |
| `pocketllm.loader.gguf` — GGUF reader, de-torched | **Done.** numpy in, descriptors out |
| `pocketllm.engine` — executor, planner, memory, session lifecycle | **Done**, on the reference backend |
| `pocketllm.architectures` — model IR and builders | **Scaffold.** `toy` only; `xing4_0` is not ported |
| `pocketllm.tokenizer` — GGUF-vocabulary BPE | **Skeleton.** Whitespace works; BPE raises |
| `pocketllm.protocol` / `pocketllm.server` — OpenAI-compatible HTTP | **Ported.** Importable and testable; needs a backend to serve |
| `pocketllm.cli` | **Done** for `devices` / `backends` / `architectures` / `ops` |

There is no `main`-branch history before the seed commit: this tree was rebuilt on an orphan branch
and the previous one is preserved as `legacy`. See [Where the code lives now](#where-the-code-lives-now).

## Installation

The base install is **one dependency**. The ABI is stdlib-only, the reference backend is numpy, and
a device runtime is an optional extra — which is what lets the same wheel install on a CUDA box and
on a phone.

```bash
pip install pocketllm
```

```bash
# With a device runtime
pip install "pocketllm[cuda]"      # NVIDIA, via torch
pip install "pocketllm[mps]"       # Apple Silicon, via torch
```

**Requirements:** Python >= 3.10. Nothing else is required for the base install — no compiler, no
CUDA toolkit, no `relic-core`. There is no `ext_modules` in this tree and no build step: every
native kernel belongs to another repository.

For a working copy:

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install -e ".[dev]"
```

## Try it

The read-only commands work on any host, including one with no accelerator and no torch:

```bash
pocketllm devices        # what this host can open, and what the rest are missing
pocketllm backends       # what each backend declares, available or not
pocketllm architectures  # the model structures this tree can build
pocketllm ops --op gemm_quant --device cuda
```

`devices` is the command to run when something is wrong, so it is the one command that must not
depend on anything being installed. Every check behind it is a filesystem probe, and it imports no
runtime:

```
reference  cpu       available                     eager only        17 ops
cpu        cpu       available                     eager only        16 ops
mps        mps       missing torch>=2.2 with an MPS device  eager only        16 ops
cuda       cuda      available                     stream_capture    17 ops
qnn        qnn       missing the QNN SDK (libQnnHtp*.so) and a Hexagon DSP device node  aot_compile       14 ops
horizon    horizon   missing the Horizon OpenExplorer runtime (libhbrt4.so) and a BPU device  aot_compile       14 ops
ascend     ascend    missing CANN (libascendcl.so) and an Ascend NPU (a /dev/davinci node)  stream_capture    14 ops
```

`ops` answers the other half of the same question — *why* an op did or did not resolve for a
device — and it does so from a development host, because resolution is a pure function over
declarations and reads no bytes:

```console
$ pocketllm ops --op gemm_quant --device cuda
gemm_quant on cuda -> cuda (100)
  rejected cpu: device kind 'cpu' != 'cuda'
  rejected mps: device kind 'mps' != 'cuda'
  rejected reference: a better candidate was found
  ...
```

Note that `available` and `implemented` are different questions. `cuda` reports `available` on this
host because torch is installed; it still has no kernels, which is why `pocketllm run` refuses.

The two commands that load a checkpoint do not have a backend to load into yet:

```bash
pocketllm run   --model /path/to/model.gguf --prompt "hello"
pocketllm serve --model /path/to/model.gguf
```

Both validate their arguments first and then exit with the specific piece that is missing rather
than pretending to serve.

### Library

```python
import pocketllm

print(pocketllm.__version__)   # 0.2.0.dev0
```

`import pocketllm` pulls in neither numpy nor torch — it declares the kernel vocabulary and nothing
else, and `LLM` / `AsyncLLM` resolve lazily. A test enforces that on a fresh interpreter
(`tests/test_package_boundaries.py`).

## Why another inference engine

Because the constraint is different. RelicLLM scales a checkpoint across cards; PocketLLM's job is
the opposite one — make one card, or one phone, sufficient. That single constraint changes the
design rather than tuning it:

- **A portable kernel ABI, not a framework binding.** `pocketllm.kernels` is descriptors and
  declarations with no dependency at all — no numpy, no torch. A backend implements it; the engine
  drives it. That is what makes a Qualcomm NPU and a CUDA card the same kind of object.
- **Torch is optional, and it is a backend's business.** The core never imports it. The loader,
  the decoders and the reference backend are numpy, so a phone install reads a `.gguf` without the
  training stack.
- **Quantization as the fit strategy.** The decoders are in-tree and dequantize to numpy; they do
  not expand weights to a full FP32 copy in the hot path.
- **Native backends are declared, discovered and explained.** A backend can arrive from a
  third-party package through the `pocketllm.backends` entry-point group, and `pocketllm ops` will
  tell you exactly why it was or was not chosen.

## Documentation

Published at **<https://lvyufeng.github.io/PocketLLM/>**.

| Section | What it holds |
|---|---|
| [Getting started](docs/getting-started.md) | Install, verify, and what to expect from a device with no accelerator |
| [Architecture](docs/architecture/kernel_abi_v1.md) | The kernel ABI, the backend model, execution, and the device targets |
| [Models](docs/models/README.md) | The support matrix and what an architecture is in this tree |
| [Guides](docs/guides/index.md) | Procedures, release flow |
| [Reports](docs/reports/index.md) | Long-form rendered reports |

## Where the code lives now

This tree used to carry the whole multi-GPU stack. The split is by hardware, and PocketLLM is one
end of it:

| Repository | What it is |
|---|---|
| **PocketLLM** (this tree) | single card, edge and mobile. One process owns one device |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | multi-GPU PyTorch runtime and serving shell |
| [relic-core](https://github.com/lvyufeng/relic-core) | the shared torch operator library (CUDA sm_75 + CPU host ops) |
| [relic-engine](https://github.com/lvyufeng/relic-engine) | the retired `cpp_engine` tree, kept as a frozen archive |

**`relic-core` is not a dependency of this package.** The one thing this tree used to read from it —
the GGML codebook header — is [vendored here](python/pocketllm/loader/gguf/vendor/README.md) with
provenance, because a loader that cannot read a checkpoint without a kernel library is a loader that
cannot run on a phone. A CUDA backend may wrap `relic_core` as an optional extra; nothing in the
core imports it.

## Roadmap

- [x] Cut the multi-card code out, hand it to RelicLLM, and rebuild on a torch-free ABI.
- [x] Port and de-torch the GGUF loader and the quantization decoders.
- [x] Port the serving and protocol layer.
- [ ] **First real backend.** CUDA is the natural one — the hardware is here and the kernels exist
      in relic-core.
- [ ] `architectures/xing4_0/`, rebuilt against the ABI, and a golden fixture.
- [ ] The GGUF-vocabulary BPE tokenizer.
- [ ] A QNN backend and an Android delivery path — the target the whole repository is named for.

## License

PocketLLM is released under the [Apache License 2.0](LICENSE).

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party components are
governed by their respective licenses. PocketLLM's code license grants no additional rights to
third-party model assets. The vendored GGML header is MIT, from llama.cpp — see
[its provenance note](python/pocketllm/loader/gguf/vendor/README.md).