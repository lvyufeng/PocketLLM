# Qwen3.8-27B-NVFP4

The same 27B hybrid-attention text model as [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md), shipped with its
dense MLPs in NVFP4. PocketLLM runs it on the native C++/CUDA engine at TP2, unpacking the 4-bit
blocks in registers and consuming them with INT8 DP4A and WMMA kernels, and generates tokens that are
bit-identical to the FP8 checkpoint's on the validated fixtures.

- **Backend**: `--backend cpp` (native C++/CUDA; the shared text path serves it, but this checkpoint
  has no separate serving benchmark)
- **Parallelism**: TP2, on the NVLink-connected pair
- **Context**: up to 262,144 positions (validated at 512 and 8,192)
- **Validated on**: 2×RTX 2080 Ti 22 GiB, GPUs 2 and 3

## Overview

The text architecture is the FP8 checkpoint's: 64 layers (48 Gated DeltaNet + 16 full GQA), hidden
5,120, dense MLP intermediate 17,408, vocabulary 248,320, 24 query heads over 4 KV heads at head
dimension 256.

What differs is the quantization, and the checkpoint is **not** uniformly NVFP4. Its MLP projections
are 4-bit float with one E4M3 scale per 16 input channels; the attention and linear-attention
projections, the `lm_head` and the MLPs of layers 56–63 stay FP8 per-channel. Rank-local telemetry at
TP2 reports 168 NVFP4 group-16 linears, 233 FP8 per-channel linears and 96 dense FP16 linears, so
"NVFP4" is a per-tensor property here, not a model-wide one. The bundled vision stack is ignored.

**Read this before choosing the format:** RTX 2080 Ti has no FP4 tensor-core instruction, so nothing
here executes Blackwell FP4 MMA. The weights are unpacked in registers to INT8 and consumed by DP4A and
INT8 WMMA. On this hardware **NVFP4 buys memory, not speed** — it is about 0.73× the per-rank device
usage of FP8 and roughly half the throughput.

## Run it

Two ranks pinned to the NVLink pair, one shared NCCL id file:

```bash
rm -f /tmp/pocketllm_qwen_nvfp4_nccl.id
for rank in 0 1; do
  physical=$((rank + 2))   # GPUs 2 and 3 are the NVLink pair
  CUDA_VISIBLE_DEVICES=$physical POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
  taskset -c 22-43,66-87 \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-NVFP4 \
    --tp-world 2 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nvfp4_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --max-new-tokens 16 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_nvfp4_rank${rank}.log 2>&1 &
done
wait
```

The server route is the shared native one — `pocketllm serve --backend cpp` — with the same checkpoint
and `--tensor-parallel-size 2`.

### Kernel gates

All of these default to the fast path; they exist for A/B work.

| Variable | Default | Effect |
| --- | --- | --- |
| `POCKETLLM_QWEN_NVFP4` | `auto` | `dp4a`, `wmma` or `reference` forces a single kernel family |
| `POCKETLLM_QWEN_NVFP4_WIDE_N64` | on | `0` falls back to the narrow WMMA prefill tile |
| `POCKETLLM_QWEN_NVFP4_WIDE_N64_MIN_ROWS` | `128` | Row count at which the wide tile engages |
| `POCKETLLM_QWEN_NVFP4_SHARED_Q8_SWIGLU` | on | Share one activation quantization between gate and up |
| `QWEN_PHASE_PROFILE` | off | Per-phase timing breakdown |

## What is supported

| Capability | State |
| --- | --- |
| Packed NVFP4 weight residency (no FP16/FP32 weight expansion) | Supported |
| Per-token dynamic INT8 activation quantization | Supported |
| DP4A single-row decode kernel | Supported |
| Wide-N64 prefill tile (`rows >= 128`) | Supported, on by default |
| Narrow-batch INT8 WMMA fallback | Supported |
| Everything from the FP8 runtime: DeltaNet, GQA, chunked prefill, prefix reuse, TP sharding | Shared |
| MTP / DSpark / DFlash2 speculation over NVFP4 weights | Loads and passes parity; acceleration not re-measured on this format |
| Native OpenAI-compatible serving | Shared text path only; no NVFP4-specific serving benchmark |
| Vision tower, image and video inputs | **Not implemented** |

## Performance

2×RTX 2080 Ti on the NVLink pair, single request, 16 generated tokens, real checkpoint and tokenizer
fixture, three fresh-process repetitions, medians. All three configurations ran on the **same** engine
binary with `POCKETLLM_QWEN_GQA_OPTIMIZED=1`, the same fixture, prefill chunk 512 and FP16 KV cache.

| Prompt | NVFP4 TP2 prefill / decode | FP8 TP2 prefill / decode | NVFP4 memory/rank | FP8 memory/rank |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 292.84 / 14.01 tok/s | 646.01 / 23.15 tok/s | 10.80 GiB | 14.77 GiB |
| 8,192 | 240.40 / 11.11 tok/s | 483.70 / 17.03 tok/s | 11.15 GiB | 15.11 GiB |

**NVFP4 is slower than FP8 on this hardware**: 0.45× prefill and 0.61× decode at 512, 0.50× and 0.65×
at 8,192. What it buys is memory — 0.73× per-rank device usage. That trade is the whole story on
SM75: there is no FP4 tensor-core path to recover the arithmetic, so every NVFP4 GEMM pays nibble
unpacking and a group-16 rescale that FP8 does not.

The single change that moved NVFP4 prefill materially is the wide-N64 tile: **2.84× at 512 and 2.66×
at 8,192** with identical generated tokens. Decode is unchanged, because it has one row and stays on
the DP4A matvec. The tile only engages at 128 rows and above, so decode and very narrow prefill chunks
see none of that gain.

## Known limitations

- **Slower than FP8 on SM75.** Choose NVFP4 only when about 4 GiB per rank matters more than roughly
  half the throughput.
- **No native FP4 tensor-core execution.** All NVFP4 math is emulated through INT8 DP4A/WMMA after
  register-level nibble unpacking.
- **Text-only and greedy generation.** The vision tower is skipped, and there is no separate serving
  benchmark for this checkpoint.
- **The wide tile needs 128 rows.** Below that the 2.7–2.8× prefill gain does not apply.
- **Validated at 512 and 8,192 on TP2 only.** Longer contexts, other TP widths and FP8 KV cache over
  NVFP4 weights are not measured.
- **Layers 56–63 are FP8 in this checkpoint**, so any coverage claim is per-tensor, not whole-model.
- **The checkpoint is `unsloth/Qwen3.8-27B-NVFP4`** at revision
  `9e3d73c76eddb75f795cc24ccfbc5affe41c66bd`, not an official Qwen release.

## Where the detail is

- [Design and measurements](../architecture/qwen3_8_27b_nvfp4_design.md) — the NVFP4 block format and
  the kernels that consume it, the rejected experiments, and the correctness evidence behind the
  parity claim.
- [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) — the validated runtime this checkpoint reuses.
- [Benchmarking rules](../guides/benchmarking.md) — comparisons are valid only when the binary SHA,
  fixture SHA, prefill chunk, KV dtype and generated length all match.
- The support matrix in [models/README.md](README.md).
