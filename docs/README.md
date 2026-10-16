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

## What it runs

Qwen3 runs on four accelerator classes through one engine. The C core (`src/`, `libpocketllm.so`)
is where a model actually runs — it reads a GGUF, tokenizes with the checkpoint's own BPE, walks
the graph and decodes, on `f16` weights and on packed quantized ones, the width ladder's K-quant
rung (`q2_k` through `q6_k`) and its IQ2 rung (`iq2_xxs`, `iq2_xs`), decoded inside the kernel.

| Device | Path | Qwen3 sizes | Format | Accuracy vs the reference |
|---|---|---|---|---|
| **CPU** (AVX2) | C core | 0.6B, 1.7B | `f16`, `q4_k_m`, `q2_k`/`q3_k`/`q5_k`, `iq2_xxs`/`iq2_xs` | **token-identical to llama.cpp** |
| **1× NVIDIA** (sm_75+) | C core, eager kernels | 0.6B, 1.7B | `f16`, `q4_k_m`, `q2_k`/`q3_k`/`q5_k`, `iq2_xxs`/`iq2_xs` | **token-identical to llama.cpp** |
| **RDK S600** (Horizon Nash BPU) | prebuilt `.hbm` via the `xlm` delegate | 0.6B → 8B | `leap_llm` int4 (`w4`/`w8`) | token agreement (prefix); at the quantizer's floor |
| **Orange Pi AIpro** (Ascend 310B1) | C core, CANN backend | 0.6B → 8B | `q4_k_m` | token agreement; **at the quantizer's floor at every size** |

Two limits carry across the table. The C-core rows are checked **token-for-token against llama.cpp**;
the two boards expose **no logits surface**, so "aligned to 1e-3" is expressible there only as
**token agreement** — and their weight schemes differ from the reference's, so a cross-board number
is a *quantizer-provenance* comparison, not a hardware ranking. The S600 is not a C-core backend: it
runs a **prebuilt graph** the host compiles ahead of time, so it is listed by the artifact it runs,
not by an op it implements.

The C-core rows' format list is where the ladder is real: `q2_k`, `q3_k`, `q5_k`, `iq2_xxs` and
`iq2_xs` join `q4_k` and `q6_k` as formats the engine decodes, each checked **block for block
against llama.cpp's own decode of the same bytes** and by a greedy walk against a **fresh llama.cpp
prefill at every step**. The block-for-block comparison is what caught this tree's Python `q3_k`
decoder reading ggml's six-bit scales as four, and it is what pins the two codebook formats, where a
table read with the wrong stride produces weights that look like weights.

The five new formats are correctness-complete rather than fast: `q4_k`'s packed AVX2 and
int8-activation kernels are written for its own group arithmetic, so the others are decoded one
weight at a time for now. Packing those kernels is the follow-up.

Beyond Qwen3, the same C-core rows run a dense **`llama`-architecture** checkpoint: MiniCPM5-1B
(1B, `Q4_K_M`, a 2026 model) is checked **token for token against llama.cpp** on `cpu` and on a
card, twelve steps each. It differs from `qwen3` in the metadata prefix, an *absent* per-head QK
norm, an untied head, and its rotary embedding — the interleaved convention, which is why there are
two rope kernels.

They also run **`qwen3vl`**: Qwen3-VL-2B's **text half** is checked the same way (8/8 on `cpu` and on
a card), and the **vision tower** that ships beside it in the `mmproj` file runs as well — checked
against llama.cpp's own tower through `libmtmd`, on a 768×768 image, with the token order, the
position table and the rotary pinned exactly and the numbers compared across the two engines' widths.
The text graph's half of the path is in too — its input widens by the tower's deepstack levels and
`forward_embeddings` feeds them, and an image span's per-section rotary positions and its non-causal
attention ride with them — all checked against llama.cpp's own embeddings path. What is missing is the
preprocessing's resize and an ABI entry point that takes an image.

**Reference throughput — Qwen3-1.7B, `q4_k_m`, against llama.cpp:** the CPU path is **1.04× on
prefill and 1.37× on decode** (ahead on both); the CUDA path is 0.11× / 0.40× and deliberately
naive. Every number, its method and its provenance is in
[the fleet support matrix](architecture/fleet_support_matrix.md).

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the
model still answers correctly in. Host offload and multi-card parallelism are excluded by
construction rather than by policy: the kernel ABI has no notion of a rank, a collective or a second
device, and `EngineArgs` has no `tensor_parallel_size` to set.

## Current status

| Piece | State |
|---|---|
| [`src/`](architecture/c_engine.md) — the C++ engine (`libpocketllm.so`) | **Runs Qwen3-0.6B and 1.7B** in `f16` and the seven packed formats `q2_k` through `q6_k` plus `iq2_xxs`/`iq2_xs`, on `cpu` and one `cuda` card. Greedy is checked token-for-token against llama.cpp; each packed format against llama.cpp's own decode block for block; the CPU path is ahead of llama.cpp on both halves |
| [`src/kernel/ascend/`](architecture/ascend_310b.md) — Ascend 310B backend | **Runs Qwen3-0.6B … 8B** on an Orange Pi AIpro, at the quantizer's own floor |
| [`pocketllm.kernels`](architecture/kernel_abi_v1.md) — descriptors, op schemas, dispatch, graph IR | **Done.** 19 ops declared; stdlib-only |
| `pocketllm.backends.reference` — numpy oracle | **Done.** Normative: implements every declared op |
| [`pocketllm.quant`](architecture/execution.md) — GGML block decoders | **Done.** Codebook tables vendored in-tree |
| `pocketllm.loader.gguf` — GGUF reader | **Done.** numpy in, descriptors out |
| [`pocketllm.engine`](architecture/execution.md) — executor, planner, memory, sessions | **Done**, on the reference backend |
| `pocketllm.protocol` / `pocketllm.server` — OpenAI-compatible HTTP | **Done.** Driven over the C core by `server/native_backend.py` |
| `pocketllm.backends.cpu` | **Stub.** Declaration and selection only |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **Stubs**, each naming the runtime it waits for |
| [`pocketllm.architectures`](models/architectures.md) — model IR and builders | **Scaffold.** `toy` only |
| `pocketllm.tokenizer` — GGUF-vocabulary BPE | **Skeleton.** Whitespace works; BPE raises |

**The Python package does not run a model.** It is the *host side* — the kernel ABI as spec, the
numpy oracle, the loader, the decoders and the execution layer — and **no Python backend implements
a kernel**: every backend except `reference` is a declaration. The two halves meet at one place,
`python/pocketllm/native.py`, the `ctypes` bridge that both `pocketllm run` and `pocketllm serve`
drive into the C core. Neither is a *Python* backend, which is why the `pocketllm.backends.*` rows
stay stubs. Read [Getting started](getting-started.md) for the install and the read-only commands
that work on any host.

## Why another inference engine

Most runtimes scale a checkpoint across cards; PocketLLM's job is the opposite one — make **one**
card, or one phone, sufficient. That single constraint changes the design rather than tuning it:

- **A portable kernel ABI, not a framework binding.** `pocketllm.kernels` is descriptors and
  declarations with no dependency at all — no numpy, no torch. A backend implements it; the engine
  drives it. That is what lets one C core sit behind a CUDA card, an Ascend NPU and a CPU unchanged.
- **The ladder is the fit strategy.** A checkpoint that does not fit is quantized until it does.
  There is no offload path to take and no second device to reach for.
- **Torch is optional, and it is a backend's business.** The loader, the decoders and the reference
  backend are numpy, so a phone install reads a `.gguf` without the training stack.

## Documentation

| Section | What it holds |
|---|---|
| [Getting started](getting-started.md) | Install, verify, and what a device with no accelerator can do |
| [The fleet support matrix](architecture/fleet_support_matrix.md) | What each accelerator class runs, its accuracy, and prefill/decode throughput |
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
