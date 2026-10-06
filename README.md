# PocketLLM

[![PyPI version](https://badge.fury.io/py/pocketllm.svg)](https://pypi.org/project/pocketllm/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-lvyufeng.github.io%2FPocketLLM-blue.svg)](https://lvyufeng.github.io/PocketLLM/)

**English** | [中文](README_CN.md)

Run a large language model on **one accelerator** — a single GPU, an edge board, the phone in your
pocket. One process owns one device. If the checkpoint does not fit, it is quantized further.

> **Status: the C engine runs Qwen3-0.6B, f16 or `q4_k_m`; the Python package does not run a model.**
> `src/` (`libpocketllm.so`) reads a GGUF, tokenizes with the checkpoint's own BPE, walks the Qwen3
> graph and decodes greedily — on `f32`/`f16` weights, and on packed `q4_k`/`q6_k` that are decoded
> inside the kernel and never widened. Checked token-for-token against llama.cpp on `cpu` and on a
> `cuda` card, for both a 1.4 GB f16 checkpoint and the 456 MB `q4_k_m` one quantized from it — each
> backend against the attention convention it implements, since llama.cpp's default flash-attention
> mode and its full-softmax mode are different arithmetic and pick different tokens at a near-tie.
> The Python package is the *host side* — the kernel ABI as spec, the numpy oracle, the
> GGUF loader, the quantization decoders, the execution layer and the OpenAI-compatible HTTP surface
> — and **no Python backend implements a kernel**: every backend except `reference` is a declaration
> whose session raises `BackendNotImplementedError`.
>
> These are two halves that meet at exactly one place: `python/pocketllm/native.py`, the `ctypes`
> bridge, which both `pocketllm run` and `pocketllm serve` now drive — `pocketllm run --model
> ckpt.gguf --prompt "…"` generates through the C core, and `pocketllm serve --model ckpt.gguf` puts
> the same engine behind an OpenAI-compatible HTTP surface. Neither is a *Python* backend: they are
> the C core reached from Python, which is why the backend table below is still all stubs.
> Read [Current status](#current-status) before installing.

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
| `pocketllm.kernels` — the kernel ABI: descriptors, op schemas, dispatch, graph IR | **Done.** 17 ops declared; stdlib-only, no numpy |
| `pocketllm.backends.reference` — numpy oracle, every op, host memory | **Done.** The normative implementation |
| `pocketllm.backends.cpu` — host CPU | **Stub.** Selection and declaration only |
| `pocketllm.backends.{cuda,mps,qnn,horizon,ascend}` | **Stubs.** Each names the runtime it waits for |
| `pocketllm.quant` — GGML block decoders, vendored tables, no relic-core | **Done.** IQ4_NL, IQ4_XS, IQ1_M, IQ2/IQ3, q2_k–q6_k, q8_0 |
| `pocketllm.loader.gguf` — GGUF reader, de-torched | **Done.** numpy in, descriptors out |
| `pocketllm.engine` — executor, planner, memory, session lifecycle | **Done**, on the reference backend |
| `pocketllm.architectures` — model IR and builders | **Scaffold.** `toy` only; `xing4_0` is not ported |
| `pocketllm.tokenizer` — GGUF-vocabulary BPE | **Skeleton.** Whitespace works; BPE raises |
| `pocketllm.protocol` / `pocketllm.server` — OpenAI-compatible HTTP | **Done.** Driven over the C core by `server/native_backend.py`; one request at a time, no batch, no cancellation |
| `pocketllm.cli` | **Done** for all six commands — `devices`, `backends`, `architectures`, `ops`, `run`, `serve` |
| `src/` — the C++ engine (`libpocketllm.so`) | **Runs Qwen3-0.6B in f16 and in `q4_k_m`.** GGUF read, BPE tokenize, forward, greedy decode, and temperature/top-k/top-p/min-p sampling; `q4_k`/`q6_k` decoded in the kernel; greedy checked token-for-token against llama.cpp on `cpu` and `cuda` (each backend against the attention convention it implements — see the [the C engine page](https://lvyufeng.github.io/PocketLLM/architecture/c_engine/#the-packed-weights)); the sampler checked token-for-token against the numpy reference. The CPU `gemm_quant` is threaded, AVX2-vectorized, and quantizes its activations to int8 the way llama.cpp does — on the thread pool, one block per task, which took the quantizer itself from 1087 us to 55.6 us at the 512×1024 prefill shape; its packed dot walks a weight row once per **eight** activation rows rather than once per row (`$POCKETLLM_CPU_GEMM_RPW=4` selects the old count), bit-identical to the one-row kernel, and the row tile is worth 8% of `pp512` over the four-row form; `attention`'s score dot is four lanes wide and bit-identical to the scalar `dot` it replaced, which is what put short-context decode at parity with llama.cpp, and its score pass now walks **four query rows per key row** — one tiled code path with the causal tail left on the one-row kernel, bit-exact against the pre-change engine on 90 shapes × thread counts, 1.42× on the attention call and 1.15× end to end at `pp512`, its weighted sum walks **four output rows per V row** — the same tiling at the other end of the call, bit-exact to the row-at-a-time loop on every chunk from 1 to 512, 1.16× on the call and 1.05× end to end, and its score pass now scores **two query heads per key walk** — the two heads that share a KV head, so a grouped-attention key row is loaded once instead of twice, with the pairing expressed as one `__m256`'s two halves so the results are two `dot4` calls bit for bit, 1.49× on the isolated score pass and 2.1× on the attention call at a decode-shaped context, and its K/V cache is now **f16** — the width llama.cpp's `-ctk`/`-ctv` default to, so the two engines stream the same bytes and the cache is half its f32 size. It is read by widening each row to f32 and running the same score kernels, which is why the f16 path's whole correctness claim is that the expansion is exact: it reproduces the rounded f32 cache **bit for bit** on decode and prefill shapes alike, tested as an equality and not a tolerance. Halving the cache is worth **1.19× on decode and 1.08× on prefill at a 1024-row context** and slightly less than 1.0 at 256 rows, where there is not yet enough cache for the conversion to be paid back; which together took that row to **976 t/s against llama.cpp's 823** at the configuration both engines default to (880 t/s / `-fa 0`, both engines measured on the same host and at matched core counts; llama.cpp with `-fa 1` measures 1222, which this engine does not implement) (see [the C engine page](https://lvyufeng.github.io/PocketLLM/architecture/c_engine/) for the numbers, the comparison, and why a fused multiply-add there would have been a different model); and `build/pocketllm-bench` measures it |

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

Note that `available` and `implemented` are different questions — and that this command answers both
about the **Python** backends. `cuda` reports `available` on this host because torch is installed; it
still has no *Python* kernels. Nothing here describes `src/`, which is a separate implementation over
a separate interface: `pocketllm devices` does not know the C core exists, and the C core does not
use `pocketllm.backends` at all.

`run` reaches the C engine, so it generates — greedily, and only for an architecture the C core
implements (Qwen3 today). It needs `libpocketllm.so` built; without it the command says so rather
than pretending:

```bash
cmake -B build -S src && cmake --build build -j8
pocketllm run --model /path/to/qwen3-0.6b-f16.gguf --prompt "The capital of France is" --max-tokens 8
```

```
The capital of France is Paris, and the capital of Italy is Rome
```

`src/tools/run.cpp` is the same thing a level lower — it links the engine's objects directly instead
of going through `ctypes`, which is what lets it print inside the loop. It also takes `--tokens` to
bypass the tokenizer, `--device` to pick a backend, the same sampling flags as `run`, and
`--print-top` for the diagnostic list of candidates at the first generated position.

`run` is greedy by default and samples when `--temperature` is above zero, with `--top-k`, `--top-p`,
`--min-p` and `--seed` beside it. The draw comes from the host — `random.Random(seed)` — because the
engine takes a uniform variate and holds no RNG, so the same seed reproduces the same text.

`serve` puts the same engine behind OpenAI-compatible HTTP, over the same checkpoint:

```bash
pocketllm serve --model ckpt.gguf --host 0.0.0.0 --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

The prompt goes through the checkpoint's **own chat template**, read out of the GGUF's
`tokenizer.chat_template` and rendered with Jinja — that needs `jinja2`, which is deliberately not a
dependency of this package, so a chat request without it falls back to a plain rendering the model
was not trained on. Sampling is per request (`"temperature"`, `"top_p"`, `"top_k"`, `"min_p"`,
`"seed"`), as is `"stop"` and `"n"`.

**It serves one request at a time.** A C `Session` holds one position and one KV cache with no
locking, while the server is a thread-per-request `ThreadingHTTPServer`, so the adapter serializes
behind a lock and declares `supports_batch = False` rather than promising overlap the engine cannot
provide. Measured on this host with six concurrent requests: 6/6 correct answers with the lock, 0/6
without it — and no error either way, which is why the lock is not an optimisation to revisit.
Cancellation is refused for the same kind of reason: `Session::forward` runs to completion, so a
generation already started cannot be abandoned.

Fields with no op behind them — `logprobs`, the penalties, `logit_bias`, `response_format`, `echo`,
`suffix`, `best_of` — are refused with a `400` naming the parameter rather than silently ignored.

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
- [x] **The C core runs Qwen3-0.6B.** GGUF read, BPE tokenize, dense forward and greedy decode, on
      `cpu` and on one `cuda` card, checked against llama.cpp.
- [x] **Wire the host shell.** `pocketllm run` reaches the C core through `native.py` and generates.
- [x] **Quantized weights in the C core.** `q4_k` and `q6_k` decode inside the kernel, packed,
      never widened — which is what a `q4_k_m` file is made of, so a 456 MB checkpoint now reads
      where the 1.4 GB f16 one did. Checked against llama.cpp token for token on `cpu` and `cuda`.
- [x] **Sampling.** `softmax`, `logits_temperature` and `topk_sample` are implemented in the C core
      and checked against the numpy reference; `run` and `run.cpp` take
      `--temperature/--top-k/--top-p/--min-p/--seed`, with greedy unchanged as the default. The
      draw is the host's, so the engine stays a pure function of `(logits, uniform)`.
- [x] **`pocketllm serve`.** The HTTP surface is driven over the C core by a `NativeBackend`
      adapter: one request at a time behind a lock, `supports_batch = False`, the checkpoint's own
      chat template, and the unapplied fields refused by name. Concurrent batching needs per-request
      KV slots in the C session and is not this step.
- [ ] **The Python backends.** `cuda` first; each is a declaration today.
- [ ] **`python/pocketllm/tokenizer/`** — the BPE is a stub. The C core tokenizes the checkpoint's
      vocabulary correctly; the Python skeleton is behind it and is the one `pocketllm run` would need.
- [ ] `architectures/xing4_0/`, rebuilt against the ABI, and a golden fixture.
- [ ] A QNN backend and an Android delivery path — the target the whole repository is named for.

## License

PocketLLM is released under the [Apache License 2.0](LICENSE).

Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party components are
governed by their respective licenses. PocketLLM's code license grants no additional rights to
third-party model assets. The vendored GGML header is MIT, from llama.cpp — see
[its provenance note](python/pocketllm/loader/gguf/vendor/README.md).