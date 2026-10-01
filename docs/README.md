<div class="pll-hero">
<div class="pll-hero__eyebrow">One card · edge · mobile</div>
<h1 class="pll-hero__title">PocketLLM</h1>
<p class="pll-hero__tagline">
Run a large language model on one accelerator — one GPU, one edge board, one phone.
If the checkpoint does not fit, quantize it; do not reach for a second card.
</p>
<p class="pll-hero__badges">
<a href="https://pypi.org/project/pocketllm/"><img src="https://img.shields.io/pypi/v/pocketllm.svg" alt="PyPI version"></a>
<a href="https://github.com/lvyufeng/PocketLLM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache-2.0"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>
<p class="pll-hero__actions">
<a class="pll-btn pll-btn--primary" href="getting-started/">Get started</a>
<a class="pll-btn" href="https://github.com/lvyufeng/PocketLLM">View on GitHub</a>
</p>
</div>

!!! warning "This tree is a pre-release skeleton"

    The interfaces are defined and the parts that need no device are ported and tested. **No device
    backend ships a working implementation yet**, so nothing runs a model today. `pocketllm run` and
    `pocketllm serve` parse their arguments and then name the piece that is missing. See
    [Current status](#current-status) below before installing anything.

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the
model still answers correctly in. Host offload and multi-card parallelism are excluded by
construction rather than by policy: the kernel ABI has no notion of a rank, a collective or a second
device, and `EngineArgs` has no `tensor_parallel_size` to set.

## Current status

| Piece | State |
|---|---|
| [`pocketllm.kernels`](architecture/kernel_abi_v1.md) — descriptors, op schemas, dispatch, graph IR | **Done.** 19 ops declared; stdlib-only |
| `pocketllm.backends.reference` — numpy oracle | **Done.** Normative: implements every declared op |
| `pocketllm.backends.cpu` | **Stub.** Declaration and selection only |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **Stubs**, each naming the runtime it waits for |
| [`pocketllm.quant`](architecture/execution.md) — GGML block decoders | **Done.** No `relic-core`; tables vendored |
| `pocketllm.loader.gguf` — GGUF reader | **Done.** numpy in, descriptors out |
| [`pocketllm.engine`](architecture/execution.md) — executor, planner, memory, sessions | **Done**, on the reference backend |
| [`pocketllm.architectures`](models/architectures.md) — model IR and builders | **Scaffold.** `toy` only |
| `pocketllm.tokenizer` — GGUF-vocabulary BPE | **Skeleton.** Whitespace works; BPE raises |
| `pocketllm.protocol` / `pocketllm.server` — OpenAI-compatible HTTP | **Ported.** Testable without a device |

Read [Getting started](getting-started.md) for the install and the four read-only commands that work
on any host.

### Two trees

The repository holds a Python tree under `python/`, and `src/` beside it is where the C++ engine
lands. The Python side is the host: the kernel ABI is its **spec**, and the reference backend, the
loader and the quantization decoders are the **numeric oracle**. The `src/` side will be the C++17
engine and the part that actually runs a model — it owns the GGUF read, the tokenizer and the graph
walk, and the CLI reaches it through a C ABI. Its status row appears here when its first artifact
builds.

## Where the code lives now

This repository used to describe the whole multi-GPU stack. The split is by hardware, and this is
one end of it:

| Repository | What it is |
|---|---|
| **PocketLLM** (this one) | single card, edge and mobile — one process owns one device |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | multi-GPU PyTorch runtime and serving shell |
| [relic-core](https://github.com/lvyufeng/relic-core) | the shared torch operator library, CUDA sm_75 and CPU |
| [relic-engine](https://github.com/lvyufeng/relic-engine) | frozen archive of the retired C++ engine |

`relic-core` is **not** a dependency here. The one thing this tree used to read from it — the GGML
codebook header — is vendored in-tree with a provenance note, because a loader that needs a kernel
library to read a checkpoint cannot run on a phone. A CUDA backend may wrap `relic_core` as an
optional extra; nothing in the core imports it, and a test enforces that.

## Documentation

| Section | What it holds |
|---|---|
| [Getting started](getting-started.md) | Install, verify, and what a device with no accelerator can do |
| [Architecture](architecture/index.md) | The kernel ABI, the backend model, execution, and the device targets |
| [Models](models/README.md) | The support matrix, and what an architecture is in this tree |
| [Guides](guides/index.md) | Procedures — the release flow today |
| [Reports](reports/index.md) | Rendered long-form reports |

Every published page is listed in the nav and indexed in [`llms.txt`](llms.txt), which is generated
from that nav by `scripts/gen_llms_txt.py` — a page outside the nav fails the strict build rather
than shipping unreachable.

## Elsewhere in the repository

- [Repository home](https://github.com/lvyufeng/PocketLLM) — install steps and the status table
- [中文首页](https://github.com/lvyufeng/PocketLLM/blob/main/README_CN.md)
- [Contributor and architecture conventions](https://github.com/lvyufeng/PocketLLM/blob/main/CLAUDE.md)

## License

PocketLLM is released under the [Apache License 2.0](https://github.com/lvyufeng/PocketLLM/blob/main/LICENSE).
Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party components are
governed by their own licenses; PocketLLM's code license grants no additional rights to third-party
model assets. The vendored GGML header is MIT, from llama.cpp.