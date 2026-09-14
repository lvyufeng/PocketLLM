# Changelog

All notable changes to PocketLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-14

### Fixed

- **The PyPI project page contradicted the package metadata.** `pyproject.toml` declared the MIT
  license, while the rendered long description — which is `README.md`, published verbatim as the
  project page — still presented PocketLLM as PolyForm Noncommercial 1.0.0 and stated that
  commercial use required separate written permission. The 0.1.0 page therefore read as
  non-commercial-only while its classifier and license field said MIT. `twine check` does not
  compare the description against the declared license, so it passed. `README.md` and
  `README_CN.md` now both state MIT.
- **Installation failed only after compiling for several minutes when the native build toolchain was
  missing.** `setup.py` checked for `cmake` and `pybind11` inside `build_native_module()`, which runs
  after the Torch CUDA extensions have already compiled. The prerequisites are now checked before
  any compilation starts, and the failure names each missing tool, how to install it, and the
  `POCKETLLM_BUILD_CPP=0` route for a PyTorch-only install. Regression tests are in
  `tests/test_native_build_preflight.py`.
- **The publish workflow's production upload could never run.** `publish-pypi.yml` gated the step on
  `github.event.inputs.publish_production` without declaring an `inputs` block, so the expression was
  always null and the only way to reach production PyPI from Actions was to edit the file. The
  workflow now declares a `publish_production` choice input, defaulting to `no`.
- **The publish workflow's Test PyPI install check could never pass either.** It installed the sdist
  from Test PyPI on a runner without a CUDA toolkit, which cannot compile it, and the failure was
  suppressed by a trailing `|| echo`. It is replaced by a verification of the metadata Test PyPI is
  actually serving, which fails on the 0.1.0 artifact for exactly the license defect above.
- **A default install from the sdist failed at CMake configuration.** `cpp_engine/CMakeLists.txt`
  declared an executable for every file under `tools/` and `tests/`, which `MANIFEST.in` does not
  ship, so an install reported 110 "Cannot find source file" errors and the native engine never
  built. This is the default install path: `POCKETLLM_BUILD_CPP` is on by default, and it failed with
  a complete toolchain rather than a missing one. The development targets are now behind
  `POCKET_BUILD_DEV_TARGETS`, defaulting to whether `tools/` and `tests/` are present, so a working
  copy keeps building them and an unpacked sdist does not declare them. Regression tests are in
  `tests/test_sdist_native_sources.py`.

  This one was found by the release's own install verification, not by a user report — 0.1.0 shipped
  with it as well.
- **The native C++ engine did not compile in any CUDA configuration.** `cpp_engine/include/qwen_ops.hpp`
  declared a `qwen_gqa_decode_attention_flashdec_f16` that forwarded to
  `qwen_gqa_decode_attention_flashdec_f16_cuda`, a kernel that was never declared or defined; the
  operator exists only on Ascend. The header's operators are `inline`, so every translation unit that
  includes it compiles the whole body, including the arm it never takes — four of them failed on the
  name. A second, unrelated break sat in `cpp_engine/engine/deepseek_v4_engine.cpp`, which called the
  four-parameter `run_safetensors_continuation_batch_impl` with five arguments, a signature its two
  sibling entry points gained in the multi-slot batching change and it did not. The CUDA arm of the
  flashdec operator now refuses with an explanation instead of naming a kernel that does not exist,
  and the call matches the declaration again. Neither defect was reachable by any check the project
  ran: no workflow compiles the native engine, and `python -m build --sdist` does not run
  `build_ext`. `POCKETLLM_BUILD_CPP=1` therefore failed for every user of 0.1.0, by one route or the
  other — at CMake configuration from the sdist, or here from a checkout.
- **The sdist omitted two files that its own sources `#include`.** `MANIFEST.in` listed the
  extensions to ship per subtree and neither `*.inl` nor `*.inc` was among them, so
  `cpp_engine/engine/qwen_layer_components.inl` (included by `qwen_engine.cpp`) and
  `cpp_engine/backends/cuda/kernels/iq1_grid.inc` (included by `iq1_ops.cu`) were absent from the
  archive. Unlike the missing `tools/` and `tests/` sources above, nothing named these files as a
  CMake target source, so configuration succeeded and the compile failed. Both extensions are now
  shipped from every `cpp_engine/` subtree, and `tests/test_sdist_native_sources.py` reads the quoted
  `#include` directives out of the files the archive ships and fails if any of them names a file it
  does not carry.

### Changed

- Release documentation consolidated into `docs/PYPI_RELEASE.md`, now linked from the documentation
  index. `docs/RELEASE_CHECKLIST.md` and the untracked `PYPI_UPLOAD_GUIDE.md` were duplicates that
  had drifted from it, including a stale statement of the pre-MIT license, and are removed.
- The publish workflow installs Torch from the CPU index within the version range `pyproject.toml`
  declares, rather than an unconstrained latest.
- The `[0.1.0]` entry's date is corrected from 2024-09-14 to 2026-09-14.
- The `[0.1.0]` entry's FlashDecoding bullet now says what it is: an Ascend 910A measurement, with no
  CUDA implementation of a separate entry point behind it. As written it read as a shipped CUDA
  feature, and the defect above is the other half of the same confusion.

## [0.1.0] - 2026-09-14

### Added

#### Models
- **Qwen3.8-27B-FP8**: Validated C++ text runtime and OpenAI-compatible server
  - 416.48 tok/s prefill (512-token), 35.87 tok/s decode on 4×RTX 2080 Ti
  - GPU-resident FP8 E4M3 weights
  - TP4 support over PCIe
- **DeepSeek-V4-Flash**: C++/CUDA and PyTorch heterogeneous runtimes
  - FP4/FP8 Safetensors and GGUF Q2/IQ2/IQ1 support
  - ~401 tok/s prefill at 32K-64K context
- **MiniMax-M2.7**: GGUF TP4 generation with raw-block CUDA kernels
  - ~105 tok/s prefill, 10.32 tok/s decode
- **GLM-5.2**: GGUF TP4 text generation

#### Inference Features
- Multi-slot batching infrastructure (KV cache, position, prefix cache, snapshots)
- `batch_decode_step` API for multi-slot decoding
- OpenAI-compatible API server:
  - `/v1/chat/completions` with streaming
  - `/v1/completions` (raw text)
  - JSON mode and schema constraints
  - Health check endpoints (`/ready`, `/alive`, `/health`)
- Token streaming callbacks for async generation
- Speculative decoding (DSpark, DFlash2, MTP)
- FlashDecoding for long-context decode on Ascend (2.68× decode speedup, Ascend 910A, TP4, 4096-token
  context). There is no separate CUDA FlashDecoding entry point; the CUDA path reaches split-partial
  decode through the fused `qwen_gqa_decode_attention_f16_fused_cuda` kernel instead.
- Prefix caching
- Chunked prefill

#### Quantization Support
- GGUF formats: Q2/Q4/Q5/Q8, IQ1/IQ2/IQ3
- FP4/FP8 E4M3 Safetensors
- Direct quantized block consumption without FP32 expansion
- FP8 KV cache and TurboQuant K8V4

#### Kernels and Optimizations
- DP4A/MMA/tensor core specialized paths for Turing (sm_75)
- GQA tensor core prefill kernel (m16n8k8)
- Fused RMSNorm
- Batched attention with multiple chunk sizes
- Expert staging and CPU/GPU hybrid MoE placement

#### Developer Tools
- Unified Python API (`LLM`, `AsyncLLM`)
- CLI: `pocketllm serve`
- Capability introspection (`engine_caps()`)
- GGUF architecture inspection and Safetensors audit tools
- Numerical parity validation

#### Backends
- C++/CUDA native runtime (`cpp_engine/`)
- PyTorch runtime with Triton kernels
- Ascend NPU support (experimental)

### Known Limitations
- Continuous batching infrastructure exists but dynamic scheduler not implemented
- `batch_decode_step` is sequential loop (concurrent throughput 1.00×)
- Multimodal inputs (image/video) not supported
- PagedAttention not implemented (contiguous KV cache only)
- Limited model coverage (4 models vs 50+ in vLLM/SGLang)

### Performance Highlights

All measurements on 4×NVIDIA RTX 2080 Ti (22 GiB, PCIe Gen3, no NVLink), TP4 where applicable.

**Qwen3.8-27B-FP8:**
- 64-token prompt: 138.6 tok/s prefill, 36.82 tok/s decode
- 512-token prompt: 416.48 tok/s prefill, 35.87 tok/s decode
- Single-request decode latency leads vLLM by 1.09× (8K context)

**DeepSeek-V4-Flash FP4:**
- 32K prompt: ~402 tok/s prefill, ~11.2 GiB/rank
- 64K prompt: ~401 tok/s prefill, ~14.5 GiB/rank
- Decode: ~3.7 tok/s

**MiniMax-M2.7 GGUF:**
- 256-token prefill: ~105 tok/s
- 43-layer decode: 10.32 tok/s

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0, < 2.7
- Transformers >= 4.40
- Safetensors >= 0.4
- CMake >= 3.18
- pybind11 >= 2.10
- CUDA toolkit 11.8+ (for CUDA extensions)
- NCCL (for TP > 1)

### Installation
```bash
pip install pocketllm --no-build-isolation
```

By default, PocketLLM builds both PyTorch CUDA extensions and the native C++ engine, providing full capabilities out of the box. The build takes 5-15 minutes and requires CUDA toolkit, CMake, and pybind11.

For PyTorch-only installation (skip C++ engine):
```bash
POCKETLLM_BUILD_CPP=0 pip install pocketllm --no-build-isolation
```

### License
- Changed from PolyForm Noncommercial 1.0.0 to MIT License

[0.1.1]: https://github.com/lvyufeng/PocketLLM/releases/tag/v0.1.1
[0.1.0]: https://github.com/lvyufeng/PocketLLM/releases/tag/v0.1.0
