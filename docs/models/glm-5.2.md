# GLM-5.2

A 78-layer MoE text model with `glm-dsa` indexed attention and a dense prefix, shipped as a
`UD-Q2_K_XL` GGUF bundle. PocketLLM runs text-in/text-out greedy generation on four cards, with the
main residual stream and rank-local expert execution, including GLM's own chat framing.

- **Backend**: the shared GGUF CLI (`src.cli.generate_gguf` / `src.cli.generate_glm`); no
  OpenAI-compatible adapter
- **Parallelism**: TP4, expert parallelism for the routed experts
- **Context**: the checkpoint advertises 1,048,576 tokens; full-context performance and memory are
  **not established** for this runtime
- **Validated on**: 4×RTX 2080 Ti 22 GiB, real bundle, warm page cache

## Overview

| Field | Value |
| --- | ---: |
| Architecture | `glm-dsa` |
| Physical blocks / runnable trunk | 79 / 78 layers |
| Leading dense layers | 3 |
| Hidden size | 6,144 |
| Query heads / KV heads / head dim | 64 / 1 / 576 |
| Routed experts / active | 256 / top-8 |
| Vocabulary | 154,880 |
| GGUF tensors | 1,809 |

The runnable trunk excludes the final NextN/MTP block: PocketLLM follows the main residual stream
rather than treating all 79 physical blocks as ordinary Transformer layers.

The default routed-expert layout is **expert parallelism**. The alternatives — a resident CPU expert
cache, routed tensor parallelism, and a fused RMSNorm — are available as opt-in experiments and are
disabled, because each one either regressed or was neutral in end-to-end measurement.

## Run it

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_glm \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --prompt "Please introduce yourself in one sentence." \
  --chat \
  --max-new-tokens 32 \
  --prewarm
```

The command reads the real block count from the bundle and runs the 78-layer main trunk. `--prewarm`
pulls the file through the page cache first, which matters on HDD/SMR-backed checkpoints.

Inspect the bundle without running it:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --architecture auto \
  --spec-summary --validate-spec \
  --capability-report --placement-report
```

## What is supported

| Capability | State |
| --- | --- |
| GLM-specific tokenizer and `[gMASK]<sop>` chat framing | Supported |
| Raw-block GGUF loading of the dense prefix and MoE trunk | Supported |
| IQ2_XS w1/w3 and IQ3_XXS w2 DP4A grouped MoE kernels | Supported, on by default |
| Q8_0 attention and shared-expert dispatch, rank-sharded vocabulary head | Supported |
| Expert-parallel expert partition across TP4 ranks | Supported, default |
| Optional full-file page-cache prewarm for HDD/SMR-backed checkpoints | Supported |
| Per-stage decode profiler (`GLM_PROFILE=1`) | Supported, diagnostic only |
| OpenAI-compatible serving | **Not wired** |
| 1M-token context | **Not established** — the config states it; nothing here measures it |

## Performance

4×RTX 2080 Ti 22 GiB, real `UD-Q2_K_XL`, warm page cache, full model unless noted.

| Runtime state | Prefill | Decode | Status |
| --- | ---: | ---: | --- |
| Float/general MoE baseline | ~0.66 tok/s | ~0.54 tok/s | Historical baseline |
| IQ2_XS/IQ3_XXS DP4A default | **~0.79 tok/s** | **~0.66 tok/s** | Current validated fast path |
| Resident CPU expert cache | ~0.17 tok/s | ~0.44 tok/s | Regression; opt-in, off |
| Routed tensor parallelism | not a win | ~0.54 vs ~0.60 tok/s EP | Regression; opt-in, off |
| Fused RMSNorm | ~0.78 tok/s | ~0.64 tok/s | Neutral/noisy; opt-in, off |

These come from different targeted A/B sessions and are runtime milestones, not one controlled
leaderboard. The current default is the DP4A expert-parallel configuration.

The decode profile found two hard floors on this system: active-expert staging and per-layer NCCL
synchronization. A routed-TP experiment removed rank skew but had to stage all eight active experts on
every rank, and the extra per-expert staging calls outweighed the communication win — which is why the
default did not move.

## Known limitations

- **Decode is slow on this hardware and is highly sensitive** to page-cache, storage, NUMA, expert
  staging and NCCL behaviour.
- **The 1M-token context is a config value here, not a measured capability.** No full-context
  performance or memory claim exists for this runtime.
- **The three opt-in experiments are documented regressions or neutral results:**
  `GLM_ROUTED_TP=1`, `GLM_ENABLE_RESIDENT_EXPERTS=1` and `GLM_FUSED_RMSNORM=1`.
- **OpenAI-compatible serving is not wired**, so this model is CLI-only.
- **Profiler runs are diagnostic.** `GLM_PROFILE=1` adds overhead; those numbers are not headline
  benchmarks.

## Where the detail is

- [Design and measurements](../architecture/glm_5_2_design.md) — the checkpoint layout, the kernels,
  the decode profile's two floors, and the per-experiment readings.
- [Benchmarking rules](../guides/benchmarking.md) — required before any GLM number is quoted.
- The support matrix in [models/README.md](README.md).
