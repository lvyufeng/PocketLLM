# Changelog

All notable changes to PocketLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-09-14

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
- FlashDecoding for long-context decode (2.68× speedup)
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

[0.1.0]: https://github.com/lvyufeng/PocketLLM/releases/tag/v0.1.0
