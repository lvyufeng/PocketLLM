# Qwen3.8-27B (official BF16)

The official `Qwen/Qwen3.8-27B` release, with the checkpoint's multimodal root config and its bundled
vision tower. PocketLLM maps all 866 text tensors, classifies the 333 vision tensors as deliberately
ignored, and produces rank-local shard descriptors for any TP world size — **host-side only: on-device
generation from this checkpoint is not validated yet.**

- **Backend**: `--backend cpp` (the shared text runtime; the CUDA path is not validated for this
  checkpoint)
- **Parallelism**: TP4 audited
- **Context**: up to 262,144 positions (from the config; not measured on this checkpoint)
- **Validated on**: host-only audit of the real 51.7 GiB release, no accelerator required

## Overview

The text architecture is the same one the FP8 and NVFP4 checkpoints use: 64 layers (48 Gated DeltaNet
+ 16 full GQA), hidden 5,120, dense MLP intermediate 17,408, vocabulary 248,320, 24 query heads over 4
KV heads at head dimension 256. What is new here is the weight source.

| | |
| --- | ---: |
| Checkpoint dtype | BF16, no quantization metadata |
| Index entries / total size | 1,199 / 51.747 GiB |
| Text tensors | 866 (50.889 GiB) |
| Vision tensors (`model.visual.*`) | 333 (0.858 GiB) |
| Shards | 18 — the vision tower ships inside them, so every shard is required even though PocketLLM executes text only |
| Native MTP | 1 layer, shared embeddings |

Two things differ from the FP8 checkpoint's page:

- **Dense BF16 weights, no scales.** All 505 mapped linears per rank classify as dense FP16, and the
  FP8-block, FP8-channel and NVFP4 counts are zero.
- **BF16 storage becomes FP16 residency.** RTX 2080 Ti has no native BF16 arithmetic, so every BF16
  tensor is converted at materialization. That is a precision-narrowing conversion at load time, not a
  lossless path, and it is specific to Turing. It costs memory: 12.8 GiB per rank at TP4, well above
  the FP8 and NVFP4 checkpoints.

Coverage is accounted for explicitly: every index entry must be either mapped by the text map,
recognized as a vision tensor, or reported as unexpected, and strict mode throws on anything
unexpected rather than loading a partial model.

## Run it

The audit needs no accelerator:

```bash
cmake -S cpp_engine -B build/cpp_engine
cmake --build build/cpp_engine --target qwen_audit -j
build/cpp_engine/tools/qwen_audit /path/to/Qwen3.8-27B --tp-world 4 --strict
```

The same audit through the engine CLI, which also stays on the host in audit mode:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B \
  --tp-world 4 --tp-rank 0 --qwen-audit-strict
```

Generation, if you want to try it, follows the FP8 page's four-rank NCCL procedure with this
checkpoint path substituted — see [the FP8 model guide](qwen3.8-27b-fp8.md). Treat it as unvalidated
for this checkpoint.

## What is supported

| Capability | State |
| --- | --- |
| Root/nested multimodal config parsing | Supported |
| Dense BF16 weight mapping, TP4 shard descriptors | Supported |
| Strict coverage accounting (fail on an unexpected tensor) | Supported |
| BF16 → FP16 device materialization for Turing | Supported |
| TP4 shard contract audit | Validated on the real checkpoint |
| Full-model CUDA generation | **Not validated** — no TPS, no cross-rank determinism, no MTP behaviour measured |
| Native OpenAI-compatible serving | **Not validated** for this checkpoint |
| Vision tower, image and video inputs | **Not implemented** — never mapped or uploaded |

## Hardware and memory

| | |
| --- | --- |
| Resident weights at TP4 | **12.796 GiB a rank** — 12.697 GiB sharded and 0.099 GiB replicated |
| Per-rank totals | Deliberately do not sum to the checkpoint size: norms, `mtp.fc` and other replicated tensors exist on every rank |
| Headroom | 12.8 GiB a rank is the largest resident set of the three Qwen3.8 checkpoints, so long-context headroom on a 22 GiB card is the smallest |

## Known limitations

- **No on-device validation.** Generation, TPS, determinism across ranks and MTP on/off behaviour have
  not been measured for this checkpoint. Treat the support matrix's "inspect only" status as exact.
- **BF16 is materialized as FP16.** Precision-narrowing at load time, specific to Turing; an
  accelerator with native BF16 must supply its own dtype policy rather than reusing
  `qwen_device_dtype`.
- **12.8 GiB of resident weights a rank at TP4**, well above the other two Qwen checkpoints.
- **Text only.** The vision tower is never mapped or uploaded, though its tensors occupy 0.858 GiB of
  the 18 shards the loader reads.
- **Ascend is not an alternative here.** That backend configures but does not link; no kernels exist
  yet. A BF16-versus-Hugging-Face parity run at TP4 does exist for the Ascend backend, but that is a
  different record and does not transfer to this one.

## Where the detail is

- [Design and measurements](../architecture/qwen3_8_27b_bf16_design.md) — the config parsing, the
  weight map and coverage rules, the TP4 shard contract, and what the host-only audit establishes.
- [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) — the validated CUDA runtime this checkpoint reuses.
- The support matrix in [models/README.md](README.md).
