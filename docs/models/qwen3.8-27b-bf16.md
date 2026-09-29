# Qwen3.8-27B (official BF16)

The official `Qwen/Qwen3.8-27B` release, with the checkpoint's multimodal root config and its bundled
vision tower. PocketLLM maps all 866 text tensors, classifies the 333 vision tensors as deliberately
ignored, and produces rank-local shard descriptors for any TP world size. On CUDA that is host-side
only — generation from this checkpoint has not been validated there. **On Ascend it is the checkpoint
the backend was brought up on:** every Ascend performance record in this repository measures this
weight source.

- **Backend**: `--backend cpp` (the shared text runtime); CUDA is not validated for this checkpoint
- **Parallelism**: TP4 audited, and TP4 is how it runs on Ascend
- **Context**: up to 262,144 positions (from the config; not measured on this checkpoint)
- **Validated on**: 4 × Ascend 910B (first generation, `Short_SoC_version=Ascend910`), TP4, CANN 9.0.0,
  through the engine CLI — not through `pocketllm serve`; on CUDA, a host-only audit of the real
  51.7 GiB release with no accelerator required

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
- **BF16 storage becomes FP16 residency.** Every BF16 tensor is converted at materialization, on both
  of the backends that run this checkpoint, for two unrelated reasons: RTX 2080 Ti has no native BF16
  arithmetic or storage path, and first-generation 910 has no BF16 at all. That is a precision-narrowing
  conversion at load time, not a lossless path. It costs memory: 12.8 GiB per rank at TP4, well above
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

For CUDA generation, the FP8 page's four-rank NCCL procedure applies with this checkpoint path
substituted — see [the FP8 model guide](qwen3.8-27b-fp8.md) — but treat it as unvalidated for this
checkpoint: nothing here has run it on a GPU. The procedure that *has* run it is the Ascend one, in
[On the Ascend 910B](#on-the-ascend-910b) below:

```bash
source scripts/ascend_env.sh
scripts/run_qwen_ascend_tp4.sh "The capital of France is" 8
```

The same checkpoint behind the OpenAI-compatible server, on the same cards:

```bash
source scripts/ascend_env.sh
python -m pocketllm serve \
    --model /path/to/Qwen3.8-27B \
    --backend cpp --device ascend --device-ids 4,5,6,7 \
    --tensor-parallel-size 4 --served-model-name qwen3.8-27b-bf16 \
    --host 127.0.0.1 --port 8124
```

[Qwen3.8-27B behind the server](../performance/ascend_qwen_bf16_served.md) is what that produced.

## What is supported

| Capability | State |
| --- | --- |
| Root/nested multimodal config parsing | Supported |
| Dense BF16 weight mapping, TP4 shard descriptors | Supported |
| Strict coverage accounting (fail on an unexpected tensor) | Supported |
| BF16 → FP16 device materialization | Supported, and shared by both backends — neither has native BF16 |
| TP4 shard contract audit | Validated on the real checkpoint |
| Full-model CUDA generation | **Not validated** — no TPS, no cross-rank determinism, no MTP behaviour measured |
| Full-model Ascend generation, TP4 | **Validated and measured** — the six records under [Performance](../performance/index.md) and the [roadmap](../architecture/ascend_performance_roadmap.md) all run this checkpoint |
| Native OpenAI-compatible serving | **Served and measured on Ascend TP4** — one ladder, one workload, one shape: see [Qwen3.8-27B behind the server](../performance/ascend_qwen_bf16_served.md). **Not validated** on CUDA |
| Vision tower, image and video inputs | **Not implemented** — never mapped or uploaded |

## Hardware and memory

| | |
| --- | --- |
| Resident weights at TP4 | **12.796 GiB a rank** — 12.697 GiB sharded and 0.099 GiB replicated |
| Per-rank totals | Deliberately do not sum to the checkpoint size: norms, `mtp.fc` and other replicated tensors exist on every rank |
| Headroom | 12.8 GiB a rank is the largest resident set of the three Qwen3.8 checkpoints, so long-context headroom on a 22 GiB card is the smallest |

## On the Ascend 910B

This checkpoint is the one the Ascend backend runs. Four first-generation 910B cards (32 GiB HBM
each, CANN 9.0.0), one process a rank, TP4 — the same four-card layout the CUDA path describes —
driven by `scripts/run_qwen_ascend_tp4.sh` against the engine binary. The same checkpoint has since
been put behind `pocketllm serve` on the same four cards — [Qwen3.8-27B behind the
server](../performance/ascend_qwen_bf16_served.md) — where the HTTP path emitted the identical greedy
token sequence at 22.18 output tok/s at concurrency one and 69.88 at concurrency eight, and where the
ladder stops at `DEFAULT_BATCH_SLOTS = 8` rather than at anything about this checkpoint.

Residency is the number the table above reports, and it is not a coincidence. `qwen_device_dtype`
narrows BF16 to FP16 for both backends and lives in `core/`, where it names both: RTX 2080 Ti has no
native BF16, and the first-generation 910 has no BF16 at all — no `support_bf16` in its
`platform_config` and no BF16 conversion intrinsic. **12.796 GiB a rank**, against 32 GiB a card.

| | Result |
| --- | ---: |
| Single request, the shipped row count | 18.6 TPS |
| Batched decode, 16 rows, TP4 | 114.5 TPS |
| Resident weights, one rank | **12.796 GiB** of 32 |
| Prefill, 512 / 4096 tokens | 878.8 / 1261.6 TPS |

Those four are points, not a ladder: each is the settled figure of its own record, taken on the tree
that record describes, and the records below are where the conditions live. Read them before
comparing one to another — several of this backend's published numbers have been withdrawn and
re-measured, and each page says which of its own figures survived.

## Known limitations

- **No on-device validation on CUDA.** Generation, TPS, determinism across ranks and MTP on/off
  behaviour have not been measured there, so for CUDA the support matrix's "inspect only" status is
  exact. The Ascend column is a different measurement on a different backend and does not transfer
  back.
- **BF16 is materialized as FP16.** Precision-narrowing at load time, for both backends; an
  accelerator with native BF16 must supply its own dtype policy rather than reusing
  `qwen_device_dtype`.
- **12.8 GiB of resident weights a rank at TP4**, well above the other two Qwen checkpoints.
- **Text only.** The vision tower is never mapped or uploaded, though its tensors occupy 0.858 GiB of
  the 18 shards the loader reads.
- **The Ascend figures are the engine CLI, not the server.** The [serving
  record](../performance/ascend_qwen_bf16_served.md) is the one that went through the HTTP path, and
  it is the only one that should be read for serving. Several of this backend's published figures
  have been withdrawn and re-measured as artefacts rather than results, and the backend's defaults
  have moved under them — each record says which of its own numbers survived, so read the record a
  figure comes from before quoting it.

## Where the detail is

- [Design and measurements](../architecture/qwen3_8_27b_bf16_design.md) — the config parsing, the
  weight map and coverage rules, the TP4 shard contract, and what the host-only audit establishes.
- [Ascend attention and its measured ceilings](../performance/ascend_attention_optimization.md),
  [TP collective overlap](../performance/ascend_tp_collective_overlap.md),
  [decode collectives and batch scaling](../performance/ascend_decode_collective_ab.md),
  [single-request decode](../performance/ascend_single_request_tps.md),
  [the gated-delta value-axis slice](../performance/ascend_gated_delta_slice.md) and
  [the RoPE table's workspace slot](../performance/ascend_rope_table_workspace_aliasing.md) — the six
  Ascend records on this checkpoint.
- [Ascend performance roadmap](../architecture/ascend_performance_roadmap.md) — what the prefill and
  decode targets stand at, and the ranked next steps.
- [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) — the validated CUDA runtime this checkpoint reuses.
- The support matrix in [models/README.md](README.md).
