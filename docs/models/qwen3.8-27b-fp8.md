# Qwen3.8-27B-FP8

## Runtime status

**Validated native C++/CUDA TP4 text runtime.** PocketLLM detects the nested Qwen3.5 text configuration, maps rank-local Safetensors weights, converts BF16 scales/non-FP8 tensors to FP16 for Turing where required, and keeps local FP8 weights resident on each GPU.

The current integration supports text prompt/token-ID smoke and timed greedy generation through `dsv4_cpp_engine`. It does not execute the checkpoint's vision tower and is not connected to the OpenAI-compatible server.

## Model specification

The validated checkpoint reports:

| Field | Value |
| --- | ---: |
| HF architecture | `Qwen3_5ForConditionalGeneration` |
| Text model type | `qwen3_5_text` |
| Text layers | 64 |
| Gated DeltaNet layers | 48 |
| Full GQA layers | 16 |
| Hidden size | 5120 |
| Dense MLP intermediate | 17,408 |
| Vocabulary | 248,320 |
| Maximum positions | 262,144 |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Partial RoPE | 64 dimensions (factor 0.25) |
| Linear-attention key heads | 16 × 128 |
| Linear-attention value heads | 48 × 128 |
| Convolution kernel | 4 |
| Quantization | FP8 E4M3, dynamic activation scheme |
| Weight scale block | 128×128 |

The root config also contains a vision tower, but PocketLLM deliberately dispatches only the text model tensors.

## Implemented execution path

- Nested Qwen config detection and strict tensor/scale shape validation.
- TP4 rank-local embedding, head, attention, and dense MLP sharding.
- FP8 E4M3 weights stored as bytes with FP16 block scales on RTX 2080 Ti.
- Online FP8 unpacking in CUDA tiles/registers; no full FP16/FP32 weight expansion.
- Separate multi-row prefill and single-token decode projection kernels.
- 48-layer Gated DeltaNet sequence/recurrent kernels with persistent state and convolution tails.
- 16-layer GQA prefill and KV-cache decode with local K/V heads.
- Decode-only fused FP8 gate/up projection plus SwiGLU.
- TP4 NCCL reductions and global greedy top-1 selection.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4, single request, real Qwen3.8-27B-FP8 checkpoint and prompts.

| Prompt | Generated tokens | Prefill | Decode | GPU used/rank |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 24 | 138.61–138.69 tok/s | 36.82 tok/s | ~8.04–8.46 GiB |
| 512 tokens | 24 | 416.48 tok/s | 35.87 tok/s | ~8.18–8.60 GiB |

Additional repeat runs on the 512-token fixture measured approximately 411.8–416.4 tok/s prefill and 35.66–35.85 tok/s decode.

Per rank, the engine reported:

```text
resident_weight_bytes=7367270656
resident_scale_bytes=742400
gpu_memory_total_bytes=23068868608
```

The stable pre-optimization decode baseline was approximately 22.4 tok/s. Rank-local full-attention K/V projection, grouped GQA value aggregation, and fused decode SwiGLU raised the measured result into the 35–37 tok/s range while preserving the separate prefill path.

## Correctness and precision

- All four TP ranks generated identical token sequences on both the 64-token and 512-token fixtures.
- GQA decode matched the CPU reference with worst absolute error `1.192e-7`.
- Fused FP8 SwiGLU matched the separate projection path with `max_abs=0` and `max_rel=0` in its test fixture.
- Qwen RMSNorm, gated RMSNorm, L2 normalization, online FP8 matvec/matmul, DeltaNet, convolution tail, GQA, and TP weight-sharding tests pass.
- Parallel GQA softmax changes reduction association. CPU-reference error remains near `1e-7`, and real generated token sequences were unchanged in the validated runs.
- The existing DeepSeek FP8 matvec/matmul and minimum-layer smoke tests were also run to protect the older path.

## Reproduction

Build the C++ engine, then start four ranks with one shared NCCL ID file:

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/dsv4_cpp_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --generate-token 123 --max-new-tokens 24 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

The numeric value passed to `--generate-token` is ignored once `--prompt` supplies the prompt IDs; it currently activates the generation mode in the compatibility CLI parser. Rank 0 prints the timed result and all ranks print their local runtime/accounting lines.

Audit only the rank-local weight mapping:

```bash
build/cpp_engine/dsv4_cpp_engine \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --tp-rank 0 \
  --qwen-audit
```

## Known limitations

- Text-only: no image/video preprocessing or vision-tower execution.
- CLI/smoke integration only: Qwen is explicitly rejected by the current DSV4 OpenAI server path.
- Greedy generation only in the current Qwen engine API.
- The 262K model limit is checkpoint metadata, not a validated 262K runtime benchmark or memory claim.
- The executable and internal C++ namespace retain DSV4 compatibility names.
- CUDA Graph, a decode megakernel, and DSpark integration remain future work; none is included in the reported TPS.

## Evidence and related notes

- `cpp_engine/include/qwen_config.hpp`
- `cpp_engine/src/qwen_config.cpp`
- `cpp_engine/src/qwen_weights.cpp`
- `cpp_engine/src/qwen_engine.cpp`
- `cpp_engine/cuda/qwen_fp8_ops.cu`
- `cpp_engine/cuda/qwen_attention_ops.cu`
- `cpp_engine/tests/test_qwen_config.cpp`
- `cpp_engine/tests/test_qwen_fp8_online.cpp`
- `cpp_engine/tests/test_qwen_gqa_attention.cpp`
- `cpp_engine/tests/test_qwen_weights.cpp`
- [Benchmark reporting rules](../benchmarking.md)
