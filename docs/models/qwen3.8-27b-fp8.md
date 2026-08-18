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
- FP16 activation storage with FP32 local accumulation/state where required; no prompt-length FP32 activation expansion.
- Chunked prefill (default 512 tokens) that retains only recurrent state, convolution tails, and full-attention KV cache between chunks.
- FP16 KV cache by default, plus explicit opt-in FP8 E4M3 cache with per-token/KV-head FP16 scales over 64-channel blocks.
- Decode-only fused FP8 gate/up projection plus SwiGLU.
- TP4 NCCL reductions and global greedy top-1 selection.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4, single request, real Qwen3.8-27B-FP8 checkpoint and prompts.

| Prompt | Generated tokens | Prefill | Decode | GPU used/rank |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 24 | 138.61–138.69 tok/s | 36.82 tok/s | ~8.04–8.46 GiB |
| 512 tokens | 24 | 416.48 tok/s | 35.87 tok/s | ~8.18–8.60 GiB |

Additional repeat runs on the 512-token fixture measured approximately 411.8–416.4 tok/s prefill and 35.66–35.85 tok/s decode.

### Long-context TP4 baseline

The following recent serial runs use the real checkpoint, deterministic natural-language tokenizer IDs, four generated tokens, complete 64-layer execution, 512-token prefill chunks, and greedy-token parity across all four ranks. Decode TPS excludes the first generated token, which is produced by prefill. The activation workspace is the peak capacity of the reusable chunk workspace, not a prompt-length buffer.

| Cache | Prompt | Prefill | Decode | Activation workspace | KV data / scales | Highest rank memory | Rank parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 32,768 | 99.32 tok/s | 10.39 tok/s | 63.97 MB | 512.0 / 0 MB | 8.47 GiB | PASS |
| FP8 | 32,768 | 81.58 tok/s | 4.44 tok/s | 63.97 MB | 256.0 / 8.00 MB | 8.22 GiB | PASS |
| FP16 | 65,536 | 60.42 tok/s | 6.48 tok/s | 63.97 MB | 1,024.0 / 0 MB | 8.97 GiB | PASS |
| FP8 | 65,536 | 47.91 tok/s | 2.41 tok/s | 63.97 MB | 512.0 / 16.00 MB | 8.48 GiB | PASS |
| FP16 | 131,072 | 33.80 tok/s | 3.58 tok/s | 62.50 MB | 2,048.0 / 0 MB | 9.97 GiB | PASS |
| FP8 | 131,072 | 26.03 tok/s | 1.25 tok/s | 62.50 MB | 1,024.0 / 32.00 MB | 9.01 GiB | PASS |
| FP16 | 262,140 | 17.88 tok/s | 1.92 tok/s | 65.50 MB | 4,096.0 / 0 MB | 11.91 GiB | PASS |

The 32K, 64K, and 128K FP16/FP8 runs produced identical four-token sequences for each cache-dtype pair. The FP16 262,140-token boundary run generated `[321, 5979, 13914, 13]` with `max_context=262144`, completed without OOM, and preserved TP-rank token parity. FP8 cache is retained as an explicit memory-saving option, not the default: on this RTX 2080 Ti setup, online cache dequantization materially reduces prefill and decode throughput. FP16 KV cache remains the precision/performance baseline.

These measurements establish that chunked prefill removes the previous prompt-length FP32 activation allocation and that a 262,140-token prompt plus four generated positions completes within the 22 GiB/rank budget using the FP16 cache. The FP8 262,140-token boundary run is still being evaluated separately.

### Decode Context Parallelism feasibility on four GPUs

**Rejected for the current four-GPU topology.** DCP can directly shard only the 16 full-GQA layers. The 48 Gated DeltaNet layers retain complete recurrent state on every replica and do not benefit from ordinary KV position sharding.

Keeping four GPUs constrains the proposed topology to TP2xDCP2. For a context length `C`, the per-GPU full-attention decode work is unchanged: TP4 performs `6` local Q heads over `C` positions, while TP2xDCP2 performs `12` local Q heads over `C/2` positions. Both equal `6C` head-position evaluations per device; the DCP topology then adds two DCP all-reduces per full-attention layer and doubles the local TP2 weights.

This was tested with a deliberately favorable upper bound: plain TP2 at half the TP4 context length, without the two DCP collectives or cache compaction. It was already slower and used substantially more memory:

| TP4 context / result | TP2 half-context, no DCP communication | Result |
| --- | --- | --- |
| 512 / 35.84 decode tok/s | 256 / 22.56 decode tok/s | 37% slower upper bound |
| 4,096 / 29.43 decode tok/s | 2,048 / 20.87 decode tok/s | 29% slower upper bound |
| 8,192 / 24.24 decode tok/s | 4,096 / 19.07 decode tok/s | 21% slower upper bound |
| 32,768 / 11.57 decode tok/s | 16,384 / OOM before prefill | infeasible |

TP2 local resident weights measured 13.72 GiB per rank, versus 6.86 GiB under TP4. Therefore an actual TP2xDCP2 implementation would be slower than these already-negative upper bounds and would introduce numerical/communicator complexity without reducing per-device attention work. The default TP4 path remains unchanged; no DCP code is enabled.

A useful context-parallel experiment requires at least eight ranks/GPUs for TP4xDCP2, which preserves the TP4 weight shard while halving per-device full-attention context. Even there, it would apply only to the 16 full-GQA layers and must beat the added two DCP collectives per such layer.

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
    --generate-token 123 --max-new-tokens 24 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

The numeric value passed to `--generate-token` is ignored once `--prompt` supplies the prompt IDs; it currently activates the generation mode in the compatibility CLI parser. Rank 0 prints the timed result and all ranks print their local runtime/accounting lines. The CLI defaults to one smoke layer; use `--smoke-layers 0` for a complete 64-layer performance claim.

For reproducible serial long-context TP4 measurements:

```bash
python scripts/bench_qwen_long_context.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,4096,8192,32768,65536 \
  --max-new-tokens 4 \
  --prefill-chunk-tokens 512 \
  --kv-cache-dtype fp16 \
  --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

The harness persists one log per rank, records rank-local timing and memory fields, checks greedy-token parity across TP ranks, and writes `results.json` after every successful context length. FP16-versus-FP8 cache parity is a separate comparison of the generated sequences from two serial runs.

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
- The model limit is 262,144 positions; with four generated tokens, the longest valid benchmark prompt is 262,140 tokens. This boundary is validated with the default FP16 KV cache; the separate FP8 boundary validation is reported only when complete.
- The executable and internal C++ namespace retain DSV4 compatibility names.
- CUDA Graph, a decode megakernel, and DSpark integration remain future work; none is included in the reported TPS.

## Evidence and related notes

- `cpp_engine/include/qwen_config.hpp`
- `cpp_engine/src/qwen_config.cpp`
- `cpp_engine/src/qwen_weights.cpp`
- `cpp_engine/src/qwen_engine.cpp`
- `cpp_engine/cuda/qwen_fp8_ops.cu`
- `cpp_engine/cuda/qwen_half_ops.cu`
- `cpp_engine/cuda/qwen_attention_ops.cu`
- `cpp_engine/tests/test_qwen_config.cpp`
- `cpp_engine/tests/test_qwen_fp8_online.cpp`
- `cpp_engine/tests/test_qwen_gqa_attention.cpp`
- `cpp_engine/tests/test_qwen_half_ops.cpp`
- `cpp_engine/tests/test_qwen_weights.cpp`
- [Benchmark reporting rules](../benchmarking.md)
