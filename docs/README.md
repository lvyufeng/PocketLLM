<div class="pll-hero">
<div class="pll-hero__eyebrow">Single card · edge · mobile</div>
<h1 class="pll-hero__title">PocketLLM</h1>
<p class="pll-hero__tagline">
Run a large language model on one consumer GPU — or on the device in your hand.
If the checkpoint does not fit, quantize it; do not reach for a second card.
</p>
<p class="pll-hero__badges">
<a href="https://pypi.org/project/pocketllm/"><img src="https://img.shields.io/pypi/v/pocketllm.svg" alt="PyPI version"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>
<p class="pll-hero__actions">
<a class="pll-btn pll-btn--primary" href="getting-started/">Get started</a>
<a class="pll-btn" href="https://github.com/lvyufeng/PocketLLM">View on GitHub</a>
</p>
</div>

**PocketLLM** is an inference library for a **single** accelerator: one consumer GPU, or an edge and
mobile target. It runs the checkpoint on the card it fits on, and where it does not fit it lowers
the weight format until it does.

!!! warning "This is a re-scoping"

    PocketLLM used to describe the whole multi-GPU stack. That stack — tensor and expert
    parallelism, host offload, the multi-card serving paths — now lives in
    [RelicLLM](https://lvyufeng.github.io/RelicLLM/), and the native kernels in
    [relic-core](https://lvyufeng.github.io/relic-core/). What is left here is the single-accelerator
    story, and the pages that were never about one card have moved with it. **The cut has landed**:
    the multi-card Python, the native C++ front end and the C++ build are gone, and this build has
    one runtime, `--backend xing4`.

## The rule

**If it does not fit, it is quantized — not offloaded, and not split across cards.**

The ladder runs Q4 → Q2 → IQ2 → IQ1 → ternary, in that order, and it stops at the lowest format the
model still answers correctly in. Host offload and multi-card tensor parallelism are deliberately
out of scope: a hybrid GPU/CPU expert path was measured at **2.3× slower** than keeping the experts
on the card, and a checkpoint that needs four cards to answer a prompt is a different product.

**One checkpoint has a runtime here**, and it fits one card whole:

| Model | Format | Fits in |
|---|---|---|
| [Xing4.0-29B-A4B](models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` — 4.5 bits a weight | **17.94 GiB** |

Two more single-card pages are kept as records of what was measured before the cut, each marked
**Stale** at the top and in the [support matrix](models/README.md): Ternary-Bonsai-2-27B (GGUF
`PTQ1_0`, 1.75 bits a weight, **5.53 GiB**) and DeepSeek-V4 on GGUF Q2/IQ2/IQ1. Neither has a
runtime in this tree — they ran on the C++ front end, which is now in the relic-engine archive.
Ternary-Bonsai is the one worth re-porting: nothing else here reaches 1.75 bits.

Nothing on this site needs a second card. A page that documents a four-card run is not here; it is
in [RelicLLM](https://lvyufeng.github.io/RelicLLM/).

## Where the code lives now

The split is by hardware, and this repository is one end of it:

| Repository | What it is |
|---|---|
| **PocketLLM** (this one) | single card, edge and mobile |
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | multi-GPU PyTorch runtime and serving shell |
| [relic-core](https://github.com/lvyufeng/relic-core) | the shared torch operator library, CUDA sm_75 and CPU |
| [relic-engine](https://github.com/lvyufeng/relic-engine) | frozen archive of the retired C++ engine |

The operator layer is shared: `relic_core.kernels` is the same code whatever runs it, so a kernel
improves both this library and the multi-GPU one.

## Does the checkpoint fit?

This is the question the whole library is organised around, so it is worth showing the answer rather
than the headline. Both served models report their own footprint and their ceiling:

- **Xing4.0-29B-A4B** — 17.94 GiB resident with **all 64 experts of all 38 MoE layers on the card**,
  flat in context to 32,768 tokens, at **75.22 tok/s of prefill** and **6.72 tok/s of decode**.

Read the model page before quoting that number: it carries the checkpoint, the prompt, the warm
state and the measurement convention, and it has a `## Known limitations` section. The same applies
to the two **Stale** pages — 5.53 GiB of Ternary-Bonsai-2-27B left **245,760 tokens of context**
(262,144 with an fp8 KV cache) on a 22 GiB card at **636.0 tok/s of prefill** and **25.9 tok/s of
decode** on a 4,096-token prompt — but those were measured on code this repository no longer has,
so nothing users can run reproduces them.

## Documentation

| Section | What it holds |
| --- | --- |
| [Model guides](models/README.md) | The support matrix and one page per single-card checkpoint |
| [Architecture](architecture/pocketllm_roadmap_old_hardware.md) | What is worth building for this class of card, the per-model design records, and the checkpoint audits |
| [Guides](guides/index.md) | The PyPI release flow |
| [Reports](reports/dsv4_2080ti_report.pdf) | Rendered long-form reports |

Every page in this tree is listed in the nav, so nothing is reachable only by guessing a filename.
Pages whose subject is the multi-GPU runtime, the operator layer, or the retired engine are not
here — they sit next to the code they describe, and the links throughout this site point at them.

## Elsewhere in the repository

- [Repository home](https://github.com/lvyufeng/PocketLLM) — install steps and the news index
- [中文首页](https://github.com/lvyufeng/PocketLLM/blob/master/README_CN.md)
- [Changelog](https://github.com/lvyufeng/PocketLLM/blob/master/CHANGELOG.md)

## License

PocketLLM is released under the [MIT License](https://github.com/lvyufeng/PocketLLM/blob/master/LICENSE).
Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party
components are governed by their own licenses; PocketLLM's code license grants no
additional rights to third-party model assets.