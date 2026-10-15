# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

**English** | [中文](README_CN.md)

Run a large language model on **one accelerator** — a single GPU, an edge board, the phone in your
pocket. One process owns one device. If the checkpoint does not fit, it is quantized further — never
offloaded, never split across cards.

## What it runs

Qwen3, on four accelerator classes, through one engine. The C core (`src/`, `libpocketllm.so`) is
where a model actually runs: it reads a GGUF, tokenizes with the checkpoint's own BPE, walks the
graph and decodes — on `f16` weights and on packed quantized ones, the width ladder's K-quant rung
(`q2_k` through `q6_k`) and its IQ2 rung (`iq2_xxs`, `iq2_xs`), decoded inside the kernel and never
widened.

| Device | Path | Qwen3 sizes | Format | Accuracy vs the reference |
|---|---|---|---|---|
| **CPU** (AVX2) | C core | 0.6B, 1.7B | `f16`, `q4_k_m`, `q2_k`/`q3_k`/`q5_k`, `iq2_xxs`/`iq2_xs` | **token-identical to llama.cpp** |
| **1× NVIDIA** (sm_75+) | C core, eager kernels | 0.6B, 1.7B | `f16`, `q4_k_m`, `q2_k`/`q3_k`/`q5_k`, `iq2_xxs`/`iq2_xs` | **token-identical to llama.cpp** |
| **RDK S600** (Horizon Nash BPU) | prebuilt `.hbm` via the `xlm` delegate | 0.6B → 8B | `leap_llm` int4 (`w4`/`w8`) | token agreement (prefix); at the quantizer's floor |
| **Orange Pi AIpro** (Ascend 310B1) | C core, CANN backend | 0.6B → 8B | `q4_k_m` | token agreement; **at the quantizer's floor at every size** |

Two honest limits run through the table. The C-core rows are checked **token-for-token against
llama.cpp** on both widths; the two boards expose **no logits surface**, so "aligned to 1e-3" is
expressible on them only as **token agreement** — and their weight schemes differ from the
reference's, so a cross-board agreement number is a *quantizer-provenance* comparison, not a
hardware ranking. The S600 is not a C-core backend: it runs a **prebuilt graph** the host compiles
ahead of time, so it is listed by the artifact it executes, not by an op it implements.

The C-core rows' format list is where the width ladder is real: `q2_k`, `q3_k`, `q5_k`, `iq2_xxs`
and `iq2_xs` join `q4_k` and `q6_k` as formats the engine decodes, and each new one is checked twice
— the decoder **block for block against llama.cpp's own decode of the same bytes**, and the engine's
greedy walk against a **fresh llama.cpp prefill at every step** on a file of that format. The
block-for-block comparison is what caught this tree's Python `q3_k` decoder reading ggml's six-bit
scales as four, and it is what pins the two codebook formats, whose decode is a table lookup: a
wrong stride there produces weights that look like weights. The five are correctness-complete rather
than fast: `q4_k`'s packed AVX2 and int8 activation kernels are written for its own group
arithmetic, so the others are decoded one weight at a time for now — 5.3 s against 0.7 s for 26
tokens at 0.6B on this machine. Packing those kernels is the follow-up, and the ladder's IQ3, IQ4
and IQ1 rungs behind them.

Beyond Qwen3, the same C-core rows run a dense **`llama`-architecture** checkpoint: MiniCPM5-1B
(1B, `Q4_K_M`, a 2026 model) is checked **token for token against llama.cpp** on `cpu` and on a
card, twelve steps each. It differs from `qwen3` in the metadata prefix, an *absent* per-head QK
norm, an untied head, and its rotary embedding — the interleaved convention, which is why there are
two rope kernels.

They also run **`qwen3vl`**: Qwen3-VL-2B's **text half** is checked the same way (8/8 on `cpu` and on
a card), which is the prerequisite for the vision path rather than the vision path itself — images
need the tower that ships beside the model in an `mmproj` file, and that half is not wired.

### Reference performance — Qwen3-1.7B, `q4_k_m`, against llama.cpp

| | prefill (`pp512`) | decode (`tg128`/`tg64`) |
|---|---|---|
| **CPU**, 22 threads | **1.04×** (ahead) | **1.37×** (ahead) |
| **1× CUDA** | 0.11× | 0.40× |

The CPU path is **ahead of llama.cpp on both halves** and is the one to quote for the reference. The
CUDA kernels are deliberately naive (prefill is slow next to a tuned BLAS); its decode GEMM's
k-split took it from 0.259× to **0.404×**, and the `dp4a` activation path that would close more has
been measured and **closed** — see [the C engine page](https://lvyufeng.github.io/PocketLLM/architecture/c_engine/).
Full per-device numbers, methods and provenance:
[the fleet support matrix](https://lvyufeng.github.io/PocketLLM/architecture/fleet_support_matrix/).

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The width ladder is Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, stopping at the lowest format the
model still answers correctly in. Host offload and multi-card parallelism are out of scope by
construction: the ABI has no rank, no collective, and no second device, and `EngineArgs` has no
`tensor_parallel_size` to set.

## Current status

What works, what is a stub, and what has not been written. This table is the honest one.

| Piece | State |
|---|---|
| `src/` — the C++ engine (`libpocketllm.so`) | **Runs Qwen3-0.6B and 1.7B in `f16` and the seven packed formats `q2_k` through `q6_k` plus `iq2_xxs`/`iq2_xs`** on `cpu` and one `cuda` card — GGUF read, BPE tokenize, forward, greedy decode, and `temperature`/`top-k`/`top-p`/`min-p`/`seed` sampling; every one of them decodes inside the kernel. Greedy is checked token-for-token against llama.cpp on both widths, and each packed format against llama.cpp's own decode of the same file block for block. The sampler is checked against the numpy reference. The CPU path is **ahead of llama.cpp on prefill and decode**; the CUDA path is correctness-complete and slow, with the open lever (int8 activations) measured and closed. Details: [the C engine page](https://lvyufeng.github.io/PocketLLM/architecture/c_engine/) |
| `src/kernel/ascend/` — Ascend 310B backend | **Runs Qwen3-0.6B … 8B** on an Orange Pi AIpro, at the quantizer's own floor. |
| `pocketllm.cli` | **Done** for all six commands — `devices`, `backends`, `architectures`, `ops`, `run`, `serve` |
| `pocketllm.protocol` / `pocketllm.server` — OpenAI-compatible HTTP | **Done.** Driven over the C core by `server/native_backend.py`; one request at a time, no batch, no cancellation |
| `pocketllm.kernels` — the kernel ABI: descriptors, op schemas, dispatch, graph IR | **Done.** 18 ops declared; stdlib-only, no numpy |
| `pocketllm.backends.reference` — numpy oracle, every op, host memory | **Done.** The normative implementation |
| `pocketllm.quant` — GGML block decoders, vendored tables | **Done.** IQ4_NL, IQ4_XS, IQ1_M, IQ2/IQ3, q2_k–q6_k, q8_0 |
| `pocketllm.loader.gguf` — GGUF reader | **Done.** numpy in, descriptors out |
| `pocketllm.engine` — executor, planner, memory, session lifecycle | **Done**, on the reference backend |
| `pocketllm.backends.cpu` — host CPU | **Stub.** Selection and declaration only |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **Stubs.** Each names the runtime it waits for |
| `pocketllm.architectures` — model IR and builders | **Scaffold.** `toy` only |
| `pocketllm.tokenizer` — GGUF-vocabulary BPE | **Skeleton.** Whitespace works; BPE raises |

**The Python package does not run a model.** It is the *host side* — the kernel ABI as spec, the
numpy oracle, the GGUF loader, the quantization decoders, the execution layer and the OpenAI-compatible
HTTP surface — and **no Python backend implements a kernel**: every backend except `reference` is a
declaration whose session raises `BackendNotImplementedError`. The two halves meet at exactly one
place: `python/pocketllm/native.py`, the `ctypes` bridge, which both `pocketllm run` and
`pocketllm serve` drive — each reaches the C core. Neither is a *Python* backend, which is why the
`pocketllm.backends.*` rows above stay stubs.

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
CUDA toolkit. There is no `ext_modules` in this tree and no build step: the C core is a **separate
native library** (`src/`, CMake) loaded at runtime through `ctypes`, not a Python extension.

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
runtime. Note that `available` and `implemented` are different questions — and that this command
answers both about the **Python** backends. An `available` row still has no *Python* kernels, and
nothing here describes `src/`: `pocketllm devices` does not know the C core exists.

`run` reaches the C engine, so it generates — greedily by default, and only for an architecture the
C core implements (Qwen3). It needs `libpocketllm.so` built; without it the command says so rather
than pretending:

```bash
cmake -B build -S src && cmake --build build -j8
pocketllm run --model /path/to/qwen3-0.6b-f16.gguf --prompt "The capital of France is" --max-tokens 8
```

```
The capital of France is Paris, and the capital of Italy is Rome
```

`run` is greedy by default and samples when `--temperature` is above zero, with `--top-k`, `--top-p`,
`--min-p` and `--seed` beside it. The draw comes from the host — `random.Random(seed)` — because the
engine takes a uniform variate and holds no RNG, so the same seed reproduces the same text.
`src/tools/run.cpp` is the same thing a level lower: it links the engine's objects directly and
also takes `--tokens`, `--device`, and `--print-top`.

`serve` puts the same engine behind OpenAI-compatible HTTP, over the same checkpoint:

```bash
pocketllm serve --model ckpt.gguf --host 0.0.0.0 --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

The prompt goes through the checkpoint's **own chat template**, read out of the GGUF's
`tokenizer.chat_template` and rendered with Jinja — which needs `jinja2`, deliberately not a
dependency of this package, so a chat request without it falls back to a plain rendering the model
was not trained on. Sampling is per request (`"temperature"`, `"top_p"`, `"top_k"`, `"min_p"`,
`"seed"`), as is `"stop"` and `"n"`.

**It serves one request at a time.** A C `Session` holds one position and one KV cache with no
locking, while the server is a thread-per-request `ThreadingHTTPServer`, so the adapter serializes
behind a lock and declares `supports_batch = False` rather than promising overlap the engine cannot
provide. Measured with six concurrent requests: 6/6 correct with the lock, 0/6 without it — and no
error either way, which is why the lock is not an optimisation to revisit. Cancellation is refused
for the same kind of reason. Fields with no op behind them — `logprobs`, the penalties,
`logit_bias`, `response_format`, `echo`, `suffix`, `best_of` — are refused with a `400` naming the
parameter rather than silently ignored.

### Library

```python
import pocketllm

print(pocketllm.__version__)   # 0.2.0.dev0
```

`import pocketllm` pulls in neither numpy nor torch — it declares the kernel vocabulary and nothing
else, and `LLM` / `AsyncLLM` resolve lazily. A test enforces that on a fresh interpreter
(`tests/test_package_boundaries.py`).

## Why another inference engine

Because the constraint is different. Most runtimes scale a checkpoint across cards; PocketLLM's job
is the opposite one — make **one** card, or one phone, sufficient. That single constraint changes
the design rather than tuning it:

- **A portable kernel ABI, not a framework binding.** `pocketllm.kernels` is descriptors and
  declarations with no dependency at all — no numpy, no torch. A backend implements it; the engine
  drives it. That is what makes a Qualcomm NPU and a CUDA card the same kind of object, and what
  lets the same C core sit behind a CUDA card, an Ascend NPU and a CPU unchanged.
- **The design is a ladder, not a flag.** The width ladder is the fit strategy: a checkpoint that
  does not fit is quantized until it does. There is no offload path to take and no second device to
  reach for.
- **Torch is optional, and it is a backend's business.** The loader, the decoders and the reference
  backend are numpy, so a phone install reads a `.gguf` without the training stack.
- **Native backends are declared, discovered and explained.** A backend can arrive from a
  third-party package through the `pocketllm.backends` entry-point group, and `pocketllm ops` will
  tell you exactly why it was or was not chosen.

## Documentation

Published at **<https://lvyufeng.github.io/PocketLLM/>**.

| Section | What it holds |
|---|---|
| [Getting started](docs/getting-started.md) | Install, verify, and what to expect from a device with no accelerator |
| [The fleet support matrix](docs/architecture/fleet_support_matrix.md) | What each accelerator class runs, its accuracy, and prefill/decode throughput |
| [Architecture](docs/architecture/kernel_abi_v1.md) | The kernel ABI, the backend model, execution, and the device targets |
| [Models](docs/models/README.md) | The support matrix and what an architecture is in this tree |
| [Guides](docs/guides/index.md) | Procedures, release flow |
| [Reports](docs/reports/index.md) | Long-form rendered reports |

## Roadmap

- [x] **The C core runs Qwen3-0.6B and 1.7B.** GGUF read, BPE tokenize, dense forward and greedy
      decode, on `cpu` and on one `cuda` card, checked against llama.cpp.
- [x] **The K-quant and IQ2 rungs of the ladder in the C core.** `q2_k`, `q3_k`, `q4_k`, `q5_k`,
      `q6_k`, `iq2_xxs` and `iq2_xs` decode inside the kernel, packed, never widened.
- [ ] **The ladder's remaining rungs in the C core.** `iq3_xxs`, `iq4_xs`, `iq4_nl` and `iq1_m`
      decode in Python and are refused by name at load; the ternary packs are addressable and not
      decoded.
- [x] **`pocketllm run` and `pocketllm serve`.** Both reach the C core through `native.py`.
- [x] **The CPU path is ahead of llama.cpp** on prefill and decode at a matched KV depth.
- [ ] **The Python backends.** `cuda` first; each is a declaration today.
- [ ] **`python/pocketllm/tokenizer/`** — the BPE is a stub. The C core tokenizes correctly; the
      Python skeleton is behind it.
- [ ] `architectures/xing4_0/`, rebuilt against the ABI, and a golden fixture.
- [ ] A QNN backend and an Android delivery path — the target the whole repository is named for.

## License

PocketLLM is released under the [Apache License 2.0](LICENSE).

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party components are
governed by their respective licenses. PocketLLM's code license grants no additional rights to
third-party model assets. The vendored GGML header is MIT, from llama.cpp — see
[its provenance note](python/pocketllm/loader/gguf/vendor/README.md).
